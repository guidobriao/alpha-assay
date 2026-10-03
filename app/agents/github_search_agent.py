from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.progress import emit_progress
from app.core.state import TaskState, RepoCandidate
from app.tools.github_tool import (
    canonical_github_repo_url,
    get_github_repo_urls_from_page,
    get_repo_info,
    get_repo_readme,
    normalize_github_repo_url,
    parse_github_url,
    search_github_repos,
)
from app.tools.llm import call_llm_json

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are helping identify the official GitHub repository for a research paper.

You will receive paper metadata and a finite list of candidate repositories.
Choose at most one candidate from the list. Do not invent URLs.

Return a JSON object with exactly these keys:
- "selected_url": string or null
- "confidence": float from 0.0 to 1.0
- "reason": short string explaining the evidence

Prefer repositories that are official implementations: linked from the paper or
project page, mention the paper title or arXiv ID, use the method name directly,
have strong stars/README evidence, and are not forks or unrelated demos."""


class GitHubSearchAgent:
    def run(self, state: TaskState) -> TaskState:
        candidates: list[RepoCandidate] = []
        seen_urls: set[str] = set()

        title = state.paper_metadata.title if state.paper_metadata else None
        arxiv_id = state.paper_metadata.arxiv_id if state.paper_metadata else None
        name_signals = [
            signal
            for signal in [title, _input_name_signal(state.input_value)]
            if signal
        ]

        # 1. GitHub links found in paper text
        if state.reproduction_brief:
            link_count = len(state.reproduction_brief.github_links_in_paper)
            emit_progress(
                "Search GitHub",
                "checking links from paper",
                detail=f"{link_count} link(s)",
            )
            for url in state.reproduction_brief.github_links_in_paper[:5]:
                emit_progress("Search GitHub", "resolving paper link", detail=url)
                repo_urls = get_github_repo_urls_from_page(url, max_results=5)
                reason = (
                    "Found GitHub link in paper"
                    if parse_github_url(url)
                    else f"Resolved project page to GitHub repo: {url}"
                )
                base_score = 70.0 if parse_github_url(url) else 80.0
                for repo_url in repo_urls:
                    self._add_repo_url(
                        repo_url,
                        candidates,
                        seen_urls,
                        reasons=[reason],
                        base_score=base_score,
                        source="paper",
                        title=title,
                        arxiv_id=arxiv_id,
                    )

        # 2. Search by arXiv ID (repos often reference their arXiv ID)
        if arxiv_id:
            self._search_and_add(
                arxiv_id,
                candidates,
                seen_urls,
                "GitHub arXiv ID search",
                base_score=50.0,
            )

        # 3. Search by title — high confidence signal, score above keyword searches
        if title:
            self._search_and_add(
                title,
                candidates,
                seen_urls,
                "GitHub title search",
                base_score=55.0,
            )
            self._search_and_add(
                f"{title} github",
                candidates,
                seen_urls,
                "GitHub title+github search",
                base_score=50.0,
            )

        # 4. Search by method keywords, prioritizing method names over generic acronyms.
        keywords = (
            state.reproduction_brief.method_keywords if state.reproduction_brief else []
        )
        if keywords:
            for kw in self._rank_keywords_for_search(keywords)[:5]:
                self._search_and_add(
                    kw,
                    candidates,
                    seen_urls,
                    f"GitHub keyword search: {kw}",
                    base_score=35.0,
                )

        # Boost score for repos whose name matches a keyword
        if keywords:
            emit_progress(
                "Search GitHub",
                "scoring keyword/name matches",
                detail=f"{len(candidates)} candidate(s)",
            )
            keyword_lower = {kw.lower() for kw in keywords}
            for c in candidates:
                if c.name and c.name.lower() in keyword_lower:
                    c.score += 15
                    c.reasons.append("Repo name matches method keyword")
                if c.name and _name_contains_any_signal(c.name, keywords):
                    c.score += 8
                    c.reasons.append("Repo name contains method keyword")

        # Boost score for repos whose name appears in the paper title or input filename.
        # PDF text extraction can mistake a body line for the title; local filenames
        # such as lightglue.pdf are often a cleaner method-name signal.
        if name_signals:
            emit_progress(
                "Search GitHub",
                "scoring title/name matches",
                detail="; ".join(name_signals),
            )
            title_words = (
                {
                    w.lower()
                    for w in title.replace(":", " ").replace("-", " ").split()
                    if len(w) > 2
                }
                if title
                else set()
            )
            for c in candidates:
                if c.name:
                    name_words = c.name.lower().replace("-", " ").split()
                    if title and any(nw in title_words for nw in name_words):
                        c.score += 10
                        c.reasons.append("Repo name appears in paper title")
                    if _name_contains_any_signal(c.name, name_signals):
                        c.score += 20
                        c.reasons.append("Repo name contains paper/input method token")

        emit_progress(
            "Search GitHub",
            "reranking repository candidates",
            detail=f"{len(candidates)} candidate(s)",
        )
        self._llm_rerank_candidates(state, candidates)

        candidates.sort(key=lambda c: c.score, reverse=True)
        state.repo_candidates = candidates

        if candidates:
            state.selected_repo = candidates[0]
            state.status = "repo_found"
            emit_progress(
                "Search GitHub",
                "selected repository",
                detail=f"{candidates[0].url} (score={candidates[0].score:.1f})",
                selected_repo=candidates[0].url,
                candidate_count=len(candidates),
            )
        else:
            state.errors.append(
                {"agent": "GitHubSearchAgent", "error": "No repo candidates found"}
            )
            state.status = "failed"
            emit_progress(
                "Search GitHub", "no repository candidates found", level="error"
            )

        return state

    def _rank_keywords_for_search(self, keywords: list[str]) -> list[str]:
        return sorted(
            dict.fromkeys(kw.strip() for kw in keywords if kw and kw.strip()),
            key=self._keyword_search_sort_key,
        )

    def _keyword_search_sort_key(self, keyword: str) -> tuple[int, int, str]:
        lowered = keyword.lower()
        generic_short_terms = {
            "vit-b",
            "vit-l",
            "vit-h",
            "cnn",
            "rnn",
            "gan",
            "bert",
            "clip",
        }
        if lowered in generic_short_terms:
            return (4, len(keyword), keyword)

        compact = " " not in keyword and "-" not in keyword
        has_mixed_case = any(c.islower() for c in keyword) and any(
            c.isupper() for c in keyword
        )
        if compact and len(keyword) >= 6 and has_mixed_case:
            return (0, len(keyword), keyword)

        if compact and len(keyword) >= 6:
            return (1, len(keyword), keyword)

        if 6 <= len(keyword) <= 24:
            return (2, len(keyword), keyword)

        return (3, len(keyword), keyword)

    def _search_and_add(
        self,
        query: str,
        candidates: list[RepoCandidate],
        seen_urls: set[str],
        reason_prefix: str,
        base_score: float = 40.0,
    ) -> None:
        emit_progress("Search GitHub", "searching GitHub", detail=query, query=query)
        repos = search_github_repos(query, max_results=5)
        added = 0
        for repo in repos:
            url = repo["html_url"]
            normalized = normalize_github_repo_url(url)
            if normalized in seen_urls:
                continue
            seen_urls.add(normalized)

            owner = repo["owner"]["login"]
            name = repo["name"]
            score = base_score
            reasons = [reason_prefix]

            if repo.get("archived"):
                score -= 20
                reasons.append("Repo is archived")

            stars = repo.get("stargazers_count", 0)
            if stars > 1000:
                score += 12
                reasons.append("Repo has >1000 stars")
            elif stars > 50:
                score += 5
                reasons.append("Repo has >50 stars")

            if repo.get("fork"):
                score -= 10
                reasons.append("Repo is a fork")

            candidates.append(
                RepoCandidate(
                    url=canonical_github_repo_url(owner, name),
                    owner=owner,
                    name=name,
                    stars=stars,
                    source="github_search",
                    score=score,
                    confidence="medium" if score >= 50 else "low",
                    reasons=reasons,
                )
            )
            added += 1
        emit_progress(
            "Search GitHub",
            "GitHub search complete",
            detail=f"{added} new candidate(s) from {query}",
            query=query,
            added_count=added,
            result_count=len(repos),
        )

    def _add_repo_url(
        self,
        repo_url: str,
        candidates: list[RepoCandidate],
        seen_urls: set[str],
        reasons: list[str],
        base_score: float,
        source: str,
        title: str | None,
        arxiv_id: str | None,
    ) -> None:
        parsed = parse_github_url(repo_url)
        if not parsed:
            return

        normalized = normalize_github_repo_url(repo_url)
        if normalized in seen_urls:
            return

        import time as _time

        owner, name = parsed
        info = get_repo_info(owner, name)
        # Retry once if API call failed — direct paper links should not be
        # penalized by transient network / rate-limit issues.
        if info is None and source == "paper":
            _time.sleep(0.5)
            info = get_repo_info(owner, name)
        score = base_score
        stars = None

        if info:
            stars = info.get("stargazers_count", 0)
            if stars > 1000:
                score += 12
                reasons.append("Repo has >1000 stars")
            elif stars > 50:
                score += 5
                reasons.append("Repo has >50 stars")

            if info.get("archived"):
                score -= 20
                reasons.append("Repo is archived")
            if info.get("fork"):
                score -= 10
                reasons.append("Repo is a fork")
        else:
            # Even without star data, being a direct paper link is a strong
            # signal — award partial star bonus to avoid being outranked
            # by unrelated repos whose API calls happened to succeed.
            if source == "paper":
                score += 6
                reasons.append("Direct paper link (star data unavailable)")

        readme = get_repo_readme(owner, name)
        if readme and title and title.lower() in readme.lower():
            score += 15
            reasons.append("README contains paper title")

        if readme and arxiv_id and arxiv_id in readme:
            score += 15
            reasons.append("README contains arXiv ID")

        seen_urls.add(normalized)
        candidates.append(
            RepoCandidate(
                url=canonical_github_repo_url(owner, name),
                owner=owner,
                name=name,
                stars=stars,
                source=source,  # type: ignore[arg-type]
                score=score,
                confidence="high" if source == "paper" else "medium",
                reasons=reasons,
            )
        )
        emit_progress(
            "Search GitHub",
            "added repository candidate",
            detail=f"{owner}/{name} score={score:.1f}",
            repo_url=canonical_github_repo_url(owner, name),
            score=score,
        )

    def _llm_rerank_candidates(
        self, state: TaskState, candidates: list[RepoCandidate]
    ) -> None:
        if len(candidates) < 2:
            return

        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)[:12]
        payload = {
            "paper": {
                "title": state.paper_metadata.title if state.paper_metadata else None,
                "arxiv_id": (
                    state.paper_metadata.arxiv_id if state.paper_metadata else None
                ),
                "task": (
                    state.reproduction_brief.task if state.reproduction_brief else None
                ),
                "method_keywords": (
                    state.reproduction_brief.method_keywords
                    if state.reproduction_brief
                    else []
                ),
                "links_in_paper": (
                    state.reproduction_brief.github_links_in_paper
                    if state.reproduction_brief
                    else []
                ),
            },
            "candidates": [self._candidate_llm_context(c) for c in ranked],
        }

        result = call_llm_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            purpose="repo_candidate_rerank",
        )
        if not result:
            emit_progress("Search GitHub", "LLM rerank unavailable", level="warning")
            return

        selected_url = result.get("selected_url")
        if not selected_url:
            return

        selected = self._find_candidate_by_url(candidates, str(selected_url))
        if not selected:
            logger.info("LLM selected URL outside candidate set: %s", selected_url)
            return

        try:
            confidence = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if confidence < 0.5:
            emit_progress(
                "Search GitHub",
                "LLM rerank confidence too low",
                level="warning",
                detail=f"{confidence:.2f}",
            )
            return

        max_score = max(c.score for c in candidates)
        selected.score = max(selected.score, max_score + 20.0)
        reason = str(
            result.get("reason") or "LLM selected as likely official repository"
        )
        selected.reasons.append(f"LLM rerank selected this repo: {reason[:200]}")
        if confidence >= 0.75:
            selected.confidence = "high"
        emit_progress(
            "Search GitHub",
            "LLM rerank selected candidate",
            detail=f"{selected.url} confidence={confidence:.2f}",
            selected_repo=selected.url,
            confidence=confidence,
        )

    def _candidate_llm_context(self, candidate: RepoCandidate) -> dict:
        owner = candidate.owner
        name = candidate.name
        info = get_repo_info(owner, name) if owner and name else None
        readme = get_repo_readme(owner, name) if owner and name else None

        return {
            "url": candidate.url,
            "owner": owner,
            "name": name,
            "stars": candidate.stars,
            "current_score": candidate.score,
            "source": candidate.source,
            "reasons": candidate.reasons,
            "description": info.get("description") if info else None,
            "homepage": info.get("homepage") if info else None,
            "fork": info.get("fork") if info else None,
            "archived": info.get("archived") if info else None,
            "readme_excerpt": readme[:1600] if readme else None,
        }

    def _find_candidate_by_url(
        self,
        candidates: list[RepoCandidate],
        selected_url: str,
    ) -> RepoCandidate | None:
        selected_normalized = normalize_github_repo_url(selected_url)
        for candidate in candidates:
            if normalize_github_repo_url(candidate.url) == selected_normalized:
                return candidate
        return None


def _name_contains_any_signal(repo_name: str, signals: list[str]) -> bool:
    repo_compact = _compact_signal(repo_name)
    if not repo_compact:
        return False
    for signal in signals:
        for token in _signal_tokens(signal):
            if len(token) >= 4 and token in repo_compact:
                return True
    return False


def _input_name_signal(input_value: str | None) -> str | None:
    if not input_value:
        return None
    raw = input_value.strip()
    if raw.startswith("@"):
        raw = raw[1:]
    path = Path(raw)
    stem = path.stem
    if stem and stem.lower() not in {"paper", "main", "download"}:
        return stem
    return None


def _signal_tokens(text: str) -> list[str]:
    import re

    cleaned = re.sub(r"-\s+", "", text)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", cleaned)
    return [_compact_signal(token) for token in tokens]


def _compact_signal(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "", text.lower())
