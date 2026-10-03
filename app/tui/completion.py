from __future__ import annotations

"""Slash command completion engine."""

from dataclasses import dataclass, field
import re

from .commands import COMMANDS


@dataclass
class CompletionItem:
    command: str
    args: str           # internal args
    description: str    # Chinese description
    category: str       # Chinese category
    score: int = 0
    display_args: str = ""

    @property
    def display_text(self) -> str:
        """One-line display for the completion popup."""
        arg_part = f" {self.display_args or self.args}" if (self.display_args or self.args) else ""
        return f"/{self.command}{arg_part}  {self.description}"

    @property
    def insert_text(self) -> str:
        return f"/{self.command}"

    @property
    def has_required_args(self) -> bool:
        """True if the command requires arguments (Enter should NOT auto-execute)."""
        return bool(self.args) and not self.args.startswith("[")

    @property
    def has_any_args(self) -> bool:
        """True if the command takes any arguments."""
        return bool(self.args)


def normalize_query(text: str) -> str:
    """Remove /, -, _, spaces; lowercase.  Preserve CJK characters."""
    lowered = text.lower()
    # Keep alphanumeric and CJK unified ideographs
    return "".join(
        ch for ch in lowered
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"
    )


def _candidate_strings(meta) -> list[str]:
    """All searchable strings for a command."""
    return [
        meta.name,
        f"/{meta.name}",
        meta.args or "",
        meta.display_args or "",
        meta.description,
        meta.category,
    ]


def fuzzy_subsequence_score(query: str, candidate: str) -> int | None:
    """Return gap-penalty score for subsequence match, or None if no match.

    Lower score = better match.  Each character match costs 0, each gap costs 1.
    """
    q = normalize_query(query)
    c = normalize_query(candidate)
    if not q:
        return 0
    qi = 0
    score = 0
    for ch in c:
        if qi >= len(q):
            break
        if ch == q[qi]:
            qi += 1
        else:
            score += 1
    if qi < len(q):
        return None
    return score


def complete_command(prefix: str, limit: int = 8) -> list[CompletionItem]:
    """Return sorted completion candidates for a slash-command prefix."""
    if not prefix.startswith("/"):
        return []

    query = prefix[1:]
    q_norm = normalize_query(query)

    effective_limit = max(limit, len(COMMANDS)) if not query else limit

    items: list[CompletionItem] = []

    for name, meta in COMMANDS.items():
        score = _score_command(query, q_norm, meta)
        if score is not None:
            items.append(CompletionItem(
                command=meta.name,
                args=meta.args,
                description=meta.description,
                category=meta.category,
                score=score,
                display_args=meta.display_args,
            ))

    # Deduplicate by command name, keeping best (lowest) score
    seen: dict[str, CompletionItem] = {}
    for item in items:
        if item.command not in seen or item.score < seen[item.command].score:
            seen[item.command] = item

    result = sorted(seen.values(), key=lambda x: x.score)
    return result[:effective_limit]


def _score_command(query: str, q_norm: str, meta) -> int | None:
    """Score how well a command matches a query.  Lower is better."""
    name = meta.name
    name_norm = normalize_query(name)

    # 0 — exact match on command name
    if query.lower() == name.lower():
        return 0

    # 10+len — command prefix match
    if name.lower().startswith(query.lower()):
        return 10 + len(name)

    # 20+len — slash-command prefix
    slash_name = f"/{name}"
    if slash_name.lower().startswith(query.lower()):
        return 20 + len(name)

    # 40+len — word-prefix on any candidate string
    for s in _candidate_strings(meta):
        for word in s.lower().split():
            if word.startswith(query.lower()) and len(query) >= 2:
                return 40 + len(name)

    # 60 + gap — fuzzy on command name
    fs = fuzzy_subsequence_score(query, name)
    if fs is not None:
        return 60 + fs

    # 80 + gap — fuzzy on slash command form
    fs = fuzzy_subsequence_score(query, f"/{name}")
    if fs is not None:
        return 80 + fs

    # 100 + gap — fuzzy on full candidate strings
    for s in _candidate_strings(meta):
        fs = fuzzy_subsequence_score(query, s)
        if fs is not None:
            return 100 + fs

    # 150 + gap — normalized fuzzy
    for s in _candidate_strings(meta):
        if q_norm:
            fs = fuzzy_subsequence_score(q_norm, normalize_query(s))
            if fs is not None:
                return 150 + fs

    return None
