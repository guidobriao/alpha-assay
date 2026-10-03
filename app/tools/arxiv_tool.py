from __future__ import annotations

import re
from pathlib import Path
import xml.etree.ElementTree as ET

import httpx

from app.tools.network import sanitize_proxy_env


def extract_arxiv_id(value: str) -> str | None:
    patterns = [
        r"arxiv\.org/abs/(\d{4}\.\d{4,5})(v\d+)?",
        r"arxiv\.org/pdf/(\d{4}\.\d{4,5})(v\d+)?",
        r"^(\d{4}\.\d{4,5})(v\d+)?$",
    ]

    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            version = match.group(2) or ""
            return match.group(1) + version

    return None


def get_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def download_arxiv_pdf(arxiv_id: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sanitize_proxy_env()
    response = httpx.get(get_pdf_url(arxiv_id), follow_redirects=True, timeout=60)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    return output_path


def get_arxiv_metadata(arxiv_id: str) -> dict | None:
    papers = _query_arxiv_api({"id_list": arxiv_id}, max_results=1)
    return papers[0] if papers else None


def search_arxiv_papers(query: str, max_results: int = 5) -> list[dict]:
    cleaned = query.strip()
    if not cleaned:
        return []
    return _query_arxiv_api(
        {
            "search_query": f"all:{cleaned}",
            "sortBy": "relevance",
            "sortOrder": "descending",
        },
        max_results=max_results,
    )


def _query_arxiv_api(params: dict[str, str], max_results: int = 5) -> list[dict]:
    request_params = {
        "start": "0",
        "max_results": str(max_results),
        **params,
    }
    sanitize_proxy_env()
    response = httpx.get(
        "https://export.arxiv.org/api/query",
        params=request_params,
        follow_redirects=True,
        timeout=30,
    )
    response.raise_for_status()
    return parse_arxiv_atom_feed(response.text)


def parse_arxiv_atom_feed(text: str) -> list[dict]:
    root = ET.fromstring(text)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    papers = []
    for entry in root.findall("atom:entry", ns):
        entry_id = _text(entry, "atom:id", ns)
        arxiv_id = _arxiv_id_from_entry_id(entry_id)
        links = entry.findall("atom:link", ns)
        pdf_url = None
        for link in links:
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break
        papers.append({
            "arxiv_id": arxiv_id,
            "title": _normalize_space(_text(entry, "atom:title", ns)),
            "summary": _normalize_space(_text(entry, "atom:summary", ns)),
            "authors": [
                _normalize_space(_text(author, "atom:name", ns))
                for author in entry.findall("atom:author", ns)
            ],
            "published": _text(entry, "atom:published", ns),
            "updated": _text(entry, "atom:updated", ns),
            "abs_url": entry_id,
            "pdf_url": pdf_url or (get_pdf_url(arxiv_id) if arxiv_id else None),
            "primary_category": (
                entry.find("arxiv:primary_category", ns).attrib.get("term")
                if entry.find("arxiv:primary_category", ns) is not None else None
            ),
        })
    return papers


def _text(node: ET.Element, path: str, ns: dict[str, str]) -> str:
    found = node.find(path, ns)
    return found.text.strip() if found is not None and found.text else ""


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _arxiv_id_from_entry_id(entry_id: str) -> str | None:
    if not entry_id:
        return None
    match = re.search(r"arxiv\.org/abs/([^/?#]+)", entry_id)
    return match.group(1) if match else None
