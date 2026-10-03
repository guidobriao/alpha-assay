from __future__ import annotations

import re
import base64
import time
from html import unescape

import httpx

from app.core.config import settings


_GITHUB_REPO_RE = re.compile(r"github\.com/([\w\-\.]+)/([\w\-\.]+)")


def parse_github_url(url: str) -> tuple[str, str] | None:
    match = _GITHUB_REPO_RE.search(url)
    if not match:
        return None

    owner, repo = match.groups()
    repo = repo.rstrip(".,;:)#?]}")
    repo = re.sub(r"\.\d+(?:\.[A-Za-z][\w-]*)*$", "", repo)
    if repo.endswith(".git"):
        repo = repo[:-4]
    if owner.lower() in {"features", "topics", "about", "marketplace", "collections"}:
        return None
    if repo.lower() in {"issues", "pulls", "projects", "actions", "wiki", "releases"}:
        return None
    return owner, repo


def canonical_github_repo_url(owner: str, repo: str) -> str:
    return f"https://github.com/{owner}/{repo}"


def normalize_github_repo_url(url: str) -> str:
    parsed = parse_github_url(url)
    if not parsed:
        normalized = url.strip().rstrip("/")
        if normalized.endswith(".git"):
            normalized = normalized[:-4]
        return normalized.lower()
    owner, repo = parsed
    return canonical_github_repo_url(owner, repo).lower()


def extract_github_repo_urls_from_html(html: str, max_results: int = 5) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    for match in _GITHUB_REPO_RE.finditer(unescape(html)):
        parsed = parse_github_url(match.group(0))
        if not parsed:
            continue
        owner, repo = parsed
        url = canonical_github_repo_url(owner, repo)
        normalized = normalize_github_repo_url(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        urls.append(url)
        if len(urls) >= max_results:
            break

    return urls


def get_github_repo_urls_from_page(url: str, max_results: int = 5) -> list[str]:
    parsed = parse_github_url(url)
    if parsed:
        return [canonical_github_repo_url(*parsed)]

    page_url = normalize_project_page_url(url)
    try:
        response = httpx.get(page_url, follow_redirects=True, timeout=30)
        response.raise_for_status()
    except Exception:
        return []

    return extract_github_repo_urls_from_html(response.text, max_results=max_results)


def normalize_project_page_url(url: str) -> str:
    url = url.strip().strip("<>()[]{}.,;")
    url = re.sub(r"(?<=\w)\.\s+(?=\w)", ".", url)
    url = re.sub(r"(?<=/)\s+(?=\w)", "", url)
    url = re.sub(r"(?<=\w)\s+(?=/)", "", url)
    if re.match(r"^[a-z][a-z0-9+\-.]*://", url, flags=re.IGNORECASE):
        return url
    if url.startswith("//"):
        return "https:" + url
    return "https://" + url


def github_headers() -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
    }

    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"

    return headers


def get_repo_info(owner: str, repo: str) -> dict | None:
    url = f"https://api.github.com/repos/{owner}/{repo}"

    try:
        response = httpx.get(url, headers=github_headers(), timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def get_repo_readme(owner: str, repo: str) -> str | None:
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"

    try:
        response = httpx.get(url, headers=github_headers(), timeout=30)
        response.raise_for_status()
        data = response.json()

        content = data.get("content")
        if not content:
            return None

        return base64.b64decode(content).decode("utf-8", errors="ignore")
    except Exception:
        return None


def search_github_repos(query: str, max_results: int = 5, max_retries: int = 3) -> list[dict]:
    url = "https://api.github.com/search/repositories"
    params = {"q": query, "sort": "stars", "per_page": max_results}

    for attempt in range(1, max_retries + 1):
        try:
            response = httpx.get(url, headers=github_headers(), params=params, timeout=30)
            response.raise_for_status()
            return response.json().get("items", [])
        except Exception:
            if attempt == max_retries:
                return []
            time.sleep(0.5 * attempt)

    return []
