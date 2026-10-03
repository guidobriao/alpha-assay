from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PaperCandidate:
    arxiv_id: str | None
    title: str
    abs_url: str | None = None
    summary: str | None = None
    published: str | None = None


@dataclass
class PaperInputResolution:
    success: bool
    input_value: str | None = None
    input_kind: str = "unknown"
    exists: bool = False
    searched: bool = False
    title: str | None = None
    arxiv_id: str | None = None
    reason: str = ""
    failure_reason: str | None = None
    candidates: list[PaperCandidate] = field(default_factory=list)


class InputResolverAgent:
    """Resolve TUI input into a validated local PDF path."""

    def resolve(self, raw_input: str) -> PaperInputResolution:
        cleaned = _strip_at(raw_input)
        if not cleaned:
            return PaperInputResolution(
                success=False,
                input_kind="local_pdf",
                failure_reason="输入为空。请提供本地论文 PDF 路径。",
            )

        candidate_path = Path(cleaned).expanduser()
        if candidate_path.exists() and candidate_path.is_file() and candidate_path.suffix.lower() == ".pdf":
            return PaperInputResolution(
                success=True,
                input_value=str(candidate_path.resolve()),
                input_kind="local_pdf",
                exists=True,
                reason="输入已识别为本地 PDF，文件存在。",
            )

        if candidate_path.exists() and candidate_path.is_dir():
            return PaperInputResolution(
                success=False,
                input_kind="local_pdf",
                failure_reason=f"输入是目录，不是 PDF 文件：{candidate_path}",
            )

        if candidate_path.exists():
            return PaperInputResolution(
                success=False,
                input_kind="local_pdf",
                failure_reason=f"本地文件不是 PDF：{candidate_path}",
            )

        return PaperInputResolution(
            success=False,
            input_kind="local_pdf",
            failure_reason=(
                f"本地 PDF 不存在或不可读取：{candidate_path}。"
                "当前版本不会自行搜索 arXiv，请先下载论文 PDF 后传入本地路径。"
            ),
        )


def _strip_at(value: str) -> str:
    stripped = value.strip()
    return stripped[1:] if stripped.startswith("@") else stripped
