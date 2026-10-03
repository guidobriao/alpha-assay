from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def stable_paper_slug(state: Any) -> str:
    metadata = getattr(state, "paper_metadata", None)
    if metadata is not None:
        arxiv_id = getattr(metadata, "arxiv_id", None)
        if arxiv_id:
            return _slugify(f"arxiv-{arxiv_id}")

    paper_input = getattr(state, "paper_input", None)
    if paper_input is not None:
        arxiv_id = getattr(paper_input, "arxiv_id", None)
        if arxiv_id:
            return _slugify(f"arxiv-{arxiv_id}")

    value = getattr(state, "input_value", None)
    if value:
        stem = Path(str(value).strip().lstrip("@")).stem
        if stem and _is_specific_input_stem(stem):
            return _slugify(stem)

    selected_repo = getattr(state, "selected_repo", None)
    if selected_repo is not None:
        name = getattr(selected_repo, "name", None)
        if name:
            return _slugify(name)

    if metadata is not None:
        title = getattr(metadata, "title", None)
        if title:
            return _slugify(title)

    task_id = getattr(state, "task_id", None)
    return _slugify(task_id or "paper")


def _slugify(value: str, max_len: int = 80) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if not text:
        text = "paper"
    return text[:max_len].strip("-") or "paper"


def _is_specific_input_stem(stem: str) -> bool:
    lowered = stem.strip().lower()
    return bool(lowered) and lowered not in {"paper", "input", "download", "downloaded", "main", "article"}
