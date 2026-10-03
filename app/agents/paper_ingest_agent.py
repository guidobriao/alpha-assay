from __future__ import annotations

import shutil
from pathlib import Path

from app.core.file_utils import save_json, save_text
from app.core.progress import emit_progress
from app.core.state import TaskState, PaperInput, PaperMetadata
from app.tools.pdf_tool import extract_text_from_pdf, extract_basic_metadata


class PaperIngestAgent:
    def run(self, state: TaskState) -> TaskState:
        task_dir = Path(state.task_dir)
        pdf_path = task_dir / "input" / "paper.pdf"

        raw_input = _strip_at(state.input_value)
        source_pdf = Path(raw_input).expanduser()

        try:
            emit_progress("Ingest paper", "validating PDF input", detail=str(source_pdf))
            if not source_pdf.exists() or not source_pdf.is_file() or source_pdf.suffix.lower() != ".pdf":
                state.paper_input = PaperInput(raw_input=raw_input, input_type="unknown")
                state.errors.append({
                    "agent": "PaperIngestAgent",
                    "error": "Input must be an existing local PDF path; arXiv lookup/download is disabled",
                })
                state.status = "failed"
                emit_progress("Ingest paper", "invalid PDF input", level="error", detail=str(source_pdf))
                return state

            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            if source_pdf.resolve() != pdf_path.resolve():
                emit_progress("Ingest paper", "copying PDF into task workspace", detail=str(pdf_path))
                shutil.copy(source_pdf, pdf_path)
            else:
                emit_progress("Ingest paper", "using task workspace PDF", detail=str(pdf_path))
            input_type = "pdf"

            emit_progress("Ingest paper", "extracting PDF text", detail=pdf_path.name)
            text = extract_text_from_pdf(pdf_path)
            parsed_text_path = task_dir / "paper" / "parsed_text.txt"
            save_text(parsed_text_path, text)
            emit_progress(
                "Ingest paper",
                "saved parsed text",
                detail=f"{len(text):,} characters",
                parsed_text_path=str(parsed_text_path),
                text_chars=len(text),
            )

            emit_progress("Ingest paper", "extracting basic metadata")
            metadata = extract_basic_metadata(text)
            state.paper_input = PaperInput(
                raw_input=raw_input,
                input_type=input_type,
                local_pdf_path=str(pdf_path),
            )
            state.paper_metadata = PaperMetadata(
                title=metadata.get("title"),
                abstract=metadata.get("abstract"),
                pdf_path=str(pdf_path),
                parsed_text_path=str(parsed_text_path),
                parse_confidence=0.5 if metadata.get("title") else 0.2,
            )

            save_json(task_dir / "paper" / "paper_metadata.json", state.paper_metadata)
            state.status = "paper_ingested"
            emit_progress(
                "Ingest paper",
                "metadata extracted",
                detail=state.paper_metadata.title or "title not detected",
                title=state.paper_metadata.title,
                parse_confidence=state.paper_metadata.parse_confidence,
            )

        except Exception as e:
            state.errors.append({"agent": "PaperIngestAgent", "error": str(e)})
            state.status = "failed"
            emit_progress("Ingest paper", "ingest error", level="error", detail=str(e))

        return state


def _strip_at(value: str) -> str:
    stripped = value.strip()
    return stripped[1:] if stripped.startswith("@") else stripped
