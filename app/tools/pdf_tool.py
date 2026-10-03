from __future__ import annotations

import re
from pathlib import Path

import fitz


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def extract_title_heuristic(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = []

    for line in lines[:20]:
        if 10 <= len(line) <= 180:
            lower = line.lower()
            if lower.startswith(("arxiv", "proceedings", "conference")):
                continue
            # Skip lines that look like email/author affiliations
            if "@" in line or lower.startswith(("abstract", "keywords", "1 ")):
                continue
            candidates.append(line)

    if not candidates:
        return None

    # Merge consecutive short lines that likely form a multi-line title
    merged = []
    i = 0
    while i < len(candidates[:8]):
        line = candidates[i]
        # If this line is all uppercase and the next line is also short, merge them
        if i + 1 < len(candidates[:8]) and line.isupper() and len(line) < 80:
            next_line = candidates[i + 1]
            combined = f"{line} {next_line}"
            if len(combined) <= 200:
                merged.append(combined)
                i += 2
                continue
        merged.append(line)
        i += 1

    if not merged:
        return None

    # Pick the longest candidate, which is usually the full title
    return max(merged, key=len)


def extract_abstract_heuristic(text: str) -> str | None:
    match = re.search(
        r"abstract\s*(.*?)(?:\n\s*1\s+introduction|\n\s*1\.\s+introduction|\n\s*introduction)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return None

    abstract = re.sub(r"\s+", " ", match.group(1)).strip()
    return abstract[:2000] if abstract else None


def extract_github_links(text: str) -> list[str]:
    normalized = normalize_wrapped_urls(text)
    links = re.findall(r"https?://github\.com/[^\s\)\]\}\>,;]+", normalized)
    cleaned = [link.rstrip(".,;:") for link in links]
    return sorted(set(cleaned))


def extract_reproduction_links(text: str) -> list[str]:
    normalized = normalize_wrapped_urls(text)
    candidates: list[str] = []
    url_re = re.compile(r"(?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}/[^\s\)\]\}\>,;]+")
    for match in url_re.finditer(normalized):
        url = match.group(0).rstrip(".,;:")
        context = normalized[max(0, match.start() - 100): match.end() + 100].lower()
        if "github.com" in url.lower() or any(
            term in context
            for term in ("code", "github", "implementation", "project", "weights", "available at")
        ):
            candidates.append(url)
    return sorted(set(candidates))


def normalize_wrapped_urls(text: str) -> str:
    text = re.sub(r"(?<=\w)\.\s+(?=\w)", ".", text)
    text = re.sub(r"(?<=/)\s+(?=\w)", "", text)
    text = re.sub(r"(?<=\w)\s+(?=/)", "", text)
    return text


def extract_basic_metadata(text: str) -> dict:
    return {
        "title": extract_title_heuristic(text),
        "abstract": extract_abstract_heuristic(text),
    }
