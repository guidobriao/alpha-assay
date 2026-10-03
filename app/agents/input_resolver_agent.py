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
                failure_reason="Input is empty. Provide a local paper PDF path.",
            )

        candidate_path = Path(cleaned).expanduser()
        if candidate_path.exists() and candidate_path.is_file() and candidate_path.suffix.lower() == ".pdf":
            return PaperInputResolution(
                success=True,
                input_value=str(candidate_path.resolve()),
                input_kind="local_pdf",
                exists=True,
                reason="Input recognized as a local PDF; the file exists.",
            )

        if candidate_path.exists() and candidate_path.is_dir():
            return PaperInputResolution(
                success=False,
                input_kind="local_pdf",
                failure_reason=f"Input is a directory, not a PDF file: {candidate_path}",
            )

        if candidate_path.exists():
            return PaperInputResolution(
                success=False,
                input_kind="local_pdf",
                failure_reason=f"Local file is not a PDF: {candidate_path}",
            )

        return PaperInputResolution(
            success=False,
            input_kind="local_pdf",
            failure_reason=(
                f"Local PDF does not exist or is unreadable: {candidate_path}. "
                "This version does not search arXiv automatically; download the paper PDF first and pass the local path."
            ),
        )


def _strip_at(value: str) -> str:
    stripped = value.strip()
    return stripped[1:] if stripped.startswith("@") else stripped
