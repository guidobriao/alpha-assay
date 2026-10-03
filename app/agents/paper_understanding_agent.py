from __future__ import annotations

import logging
from pathlib import Path

from app.core.file_utils import save_json
from app.core.progress import emit_progress
from app.core.state import TaskState, ReproductionBrief
from app.tools.llm import call_llm_json
from app.tools.paper_parser import (
    extract_datasets,
    extract_metrics,
    extract_tasks,
    extract_method_keywords,
)
from app.tools.pdf_tool import extract_github_links, extract_reproduction_links

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research paper analysis assistant. Given the text of an academic paper, \
extract structured information for reproduction purposes.

Respond with a JSON object containing exactly these keys:
- "task": string or null — the main research task (e.g. "image classification", \
"object detection", "text generation")
- "datasets": list of strings — datasets mentioned (e.g. ["ImageNet", "COCO"])
- "metrics": list of strings — evaluation metrics (e.g. ["accuracy", "F1", "mAP"])
- "method_keywords": list of strings — key method/technique names mentioned \
frequently (max 10)
- "github_links": list of strings — any GitHub URLs found in the paper
- "confidence": float between 0.0 and 1.0 — your confidence in this extraction

Only include items you are confident about. Return empty lists if unsure."""


_PROTOCOL_PROMPT = """\
You are a benchmark protocol extraction assistant. Given paper text, extract the
paper-table reproduction protocol, not just the task summary.

Return a JSON object with exactly these keys:
- "main_tables": list of table/figure identifiers or names likely needed for reproduction
- "tasks": list of task names
- "datasets": list of dataset names
- "splits": list of dataset split names or protocols
- "metrics": list of metric names
- "model_variants": list of model/checkpoint/config variants
- "reference_values": list of objects with keys metric, value, dataset, model, table, notes
- "hardware": list of hardware or speed-measurement conditions
- "preprocessing": list of preprocessing/protocol details
- "confidence": float between 0 and 1

If a field is unclear, return an empty list for that field. Do not invent values."""


class PaperUnderstandingAgent:
    def run(self, state: TaskState) -> TaskState:
        task_dir = Path(state.task_dir)
        parsed_text_path = task_dir / "paper" / "parsed_text.txt"

        if not parsed_text_path.exists():
            state.errors.append(
                {
                    "agent": "PaperUnderstandingAgent",
                    "error": "parsed_text.txt not found",
                }
            )
            state.status = "failed"
            emit_progress(
                "Understand paper",
                "parsed text missing",
                level="error",
                detail=str(parsed_text_path),
            )
            return state

        emit_progress(
            "Understand paper",
            "loading parsed paper text",
            detail=str(parsed_text_path),
        )
        text = parsed_text_path.read_text(encoding="utf-8", errors="ignore")
        emit_progress(
            "Understand paper",
            "loaded paper text",
            detail=f"{len(text):,} characters",
            text_chars=len(text),
        )

        emit_progress(
            "Understand paper",
            "asking LLM for reproduction brief",
            detail="task, datasets, metrics, GitHub links",
        )
        brief = self._llm_understand(text)
        if brief is None:
            logger.info("LLM unavailable or failed, falling back to heuristic parsing")
            emit_progress(
                "Understand paper",
                "LLM brief unavailable, using heuristics",
                level="warning",
            )
            brief = self._heuristic_understand(text)
        else:
            emit_progress(
                "Understand paper",
                "LLM brief extracted",
                detail=brief.task or "task not detected",
                dataset_count=len(brief.datasets),
                metric_count=len(brief.metrics),
                github_link_count=len(brief.github_links_in_paper),
            )
            brief.github_links_in_paper = _merge_links(
                brief.github_links_in_paper,
                extract_reproduction_links(text),
            )
        emit_progress(
            "Understand paper",
            "extracting benchmark protocol",
            detail="tables, splits, metrics, reference values",
        )
        protocol = self._llm_extract_protocol(text)
        if protocol:
            brief.benchmark_protocol = protocol
            emit_progress(
                "Understand paper",
                "benchmark protocol extracted",
                detail=f"confidence={protocol.get('confidence', 'unknown')}",
                protocol_confidence=protocol.get("confidence"),
            )
        else:
            emit_progress(
                "Understand paper", "benchmark protocol unavailable", level="warning"
            )

        state.reproduction_brief = brief
        save_json(task_dir / "paper" / "reproduction_brief.json", brief)
        if brief.benchmark_protocol:
            save_json(
                task_dir / "paper" / "benchmark_protocol_brief.json",
                brief.benchmark_protocol,
            )
        emit_progress(
            "Understand paper",
            "saved reproduction brief",
            detail=f"datasets={len(brief.datasets)}, metrics={len(brief.metrics)}, links={len(brief.github_links_in_paper)}",
            task=brief.task,
            datasets=brief.datasets,
            metrics=brief.metrics,
            github_links=brief.github_links_in_paper,
        )

        state.status = "paper_understood"
        return state

    def _llm_understand(self, text: str) -> ReproductionBrief | None:
        truncated = text[:6000]
        result = call_llm_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Analyze the following paper text and extract structured information:\n\n{truncated}",
            purpose="paper_understanding",
        )
        if result is None:
            return None

        try:
            return ReproductionBrief(
                task=result.get("task"),
                datasets=result.get("datasets", []),
                metrics=result.get("metrics", []),
                method_keywords=result.get("method_keywords", []),
                github_links_in_paper=result.get("github_links", []),
                confidence=result.get("confidence", 0.5),
            )
        except Exception as e:
            logger.warning("Failed to parse LLM result into ReproductionBrief: %s", e)
            return None

    def _heuristic_understand(self, text: str) -> ReproductionBrief:
        tasks = extract_tasks(text)
        return ReproductionBrief(
            task=tasks[0] if tasks else None,
            datasets=extract_datasets(text),
            metrics=extract_metrics(text),
            method_keywords=extract_method_keywords(text),
            github_links_in_paper=_merge_links(
                extract_github_links(text), extract_reproduction_links(text)
            ),
            confidence=0.5,
        )

    def _llm_extract_protocol(self, text: str) -> dict | None:
        truncated = text[:12000]
        result = call_llm_json(
            system_prompt=_PROTOCOL_PROMPT,
            user_prompt=f"Extract benchmark protocol details from this paper text:\n\n{truncated}",
            purpose="paper_protocol_extraction",
            max_tokens=3072,
        )
        if not isinstance(result, dict):
            return None
        return result


def _merge_links(primary: list[str], secondary: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for link in [*primary, *secondary]:
        cleaned = _clean_github_url(link)
        if not cleaned:
            continue
        key = cleaned.lower().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        merged.append(cleaned)
    return merged


def _clean_github_url(url: str) -> str | None:
    """Remove trailing sentence fragments from extracted GitHub URLs.

    PDF text extraction often concatenates sentence-ending periods
    with section headings, producing artifacts like:
      https://github.com/OpenAI/CLIP.1.Introduction
    """
    import re as _re

    url = url.strip().rstrip(".,;:)#?]}")
    if not url.startswith(("http://", "https://", "git@")):
        return None

    parsed = _re.match(r"(https?://github\.com/[\w\-\.]+/[\w\-\.]+)", url)
    if parsed:
        return parsed.group(1).rstrip(".,;:)#?]}")
    return url
