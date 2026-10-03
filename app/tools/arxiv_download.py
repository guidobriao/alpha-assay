from __future__ import annotations

"""Download arXiv papers as PDF for the TUI paper reproduction pipeline."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

_ARXIV_ID_RE = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?")


@dataclass
class ArxivDownloadResult:
    success: bool
    arxiv_id: str
    abs_url: str
    pdf_url: str
    pdf_path: Path | None = None
    reused_existing: bool = False
    error: str | None = None


def normalize_arxiv_id(value: str) -> str:
    """Extract a normalized arXiv ID from various input formats."""
    raw = value.strip()
    if not raw:
        raise ValueError("arXiv 输入为空")

    # arxiv:id prefix
    if raw.lower().startswith("arxiv:"):
        raw = raw.split(":", 1)[1].strip()

    # URL
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        if "arxiv.org" not in parsed.netloc.lower():
            raise ValueError("不是 arXiv 链接")
        path = parsed.path.strip("/")
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] in {"abs", "pdf"}:
            raw = parts[1]
        else:
            raise ValueError(f"无法从 arXiv 链接中解析 ID：{value}")

    # Strip .pdf suffix
    raw = raw.removesuffix(".pdf")

    if not _ARXIV_ID_RE.fullmatch(raw):
        raise ValueError(f"不支持的 arXiv ID 格式：{value}。支持格式：1911.11763、arXiv:1911.11763、https://arxiv.org/abs/1911.11763 等。")

    return raw


def arxiv_abs_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def safe_arxiv_pdf_filename(arxiv_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", arxiv_id)
    return f"arXiv-{safe}.pdf"


def make_progress_bar(percent: int | None, width: int = 18) -> str:
    if percent is None or percent < 0:
        return ""
    pct = min(int(percent), 100)
    fill = int(width * pct / 100)
    return "[" + "█" * fill + "░" * (width - fill) + f"] {pct:3d}%"


def download_arxiv_pdf(
    value: str,
    *,
    output_dir: Path,
    overwrite: bool = False,
    timeout_seconds: int = 120,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> ArxivDownloadResult:
    """Download an arXiv paper as PDF with progress callback."""
    try:
        arxiv_id = normalize_arxiv_id(value)
    except Exception as e:
        return ArxivDownloadResult(success=False, arxiv_id="", abs_url="", pdf_url="", error=str(e))

    abs_url = arxiv_abs_url(arxiv_id)
    pdf_url = arxiv_pdf_url(arxiv_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / safe_arxiv_pdf_filename(arxiv_id)
    part_path = pdf_path.with_suffix(pdf_path.suffix + ".part")

    if pdf_path.exists() and not overwrite:
        return ArxivDownloadResult(
            success=True, arxiv_id=arxiv_id, abs_url=abs_url, pdf_url=pdf_url,
            pdf_path=pdf_path, reused_existing=True,
        )

    try:
        if progress_cb:
            progress_cb({"phase": "start", "arxiv_id": arxiv_id, "pdf_url": pdf_url, "pdf_path": str(pdf_path)})

        downloaded = 0
        with httpx.stream("GET", pdf_url, follow_redirects=True, timeout=timeout_seconds) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", "0") or 0)
            with open(part_path, "wb") as f:
                for chunk in resp.iter_bytes():
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        pct = int(downloaded * 100 / total) if total else None
                        progress_cb({
                            "phase": "progress",
                            "downloaded": downloaded, "total": total,
                            "percent": pct, "pdf_path": str(pdf_path),
                        })

        if part_path.stat().st_size < 1024:
            raise RuntimeError("下载结果过小，可能不是有效 PDF")

        with open(part_path, "rb") as f:
            if f.read(5) != b"%PDF-":
                raise RuntimeError("下载结果不是有效 PDF 文件")

        part_path.replace(pdf_path)

        if progress_cb:
            progress_cb({"phase": "finish", "downloaded": pdf_path.stat().st_size, "pdf_path": str(pdf_path)})

        return ArxivDownloadResult(
            success=True, arxiv_id=arxiv_id, abs_url=abs_url, pdf_url=pdf_url, pdf_path=pdf_path,
        )

    except Exception as e:
        try:
            if part_path.exists():
                part_path.unlink()
        except Exception:
            pass
        return ArxivDownloadResult(
            success=False, arxiv_id=arxiv_id, abs_url=abs_url, pdf_url=pdf_url, error=str(e),
        )
