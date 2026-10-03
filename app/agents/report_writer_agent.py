from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime

import jinja2

from app.core.file_utils import save_json
from app.core.progress import emit_progress
from app.core.state import TaskState, ReportResult
from app.tools.llm import call_llm_json, telemetry_records

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

_SYSTEM_PROMPT = """\
You are a paper replication assistant. Based on the structured execution results below, \
produce a concise conclusion and actionable next steps.

Respond with a JSON object containing exactly these keys:
- "conclusion": 1-3 sentences summarizing the overall result, specifically noting \
which steps succeeded and which failed.
- "next_steps": 3-5 concrete suggestions telling the user what to do next. Reference \
specific file paths or commands where applicable.

Language requirements:
- All natural language must be in English.
- File paths, commands, environment variables, status codes, metric names and proper \
nouns may stay verbatim.
- Do not guess beyond the provided data. Be concise and practical."""


_SMOKE_ENV_FAILURE_TYPES = {
    "missing_dependency",
    "cuda_error",
    "runtime_linker_error",
}


class ReportWriterAgent:
    def run(self, state: TaskState) -> TaskState:
        task_dir = Path(state.task_dir)
        report_dir = task_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)

        final_status = self._determine_final_status(state)
        emit_progress("Write report", "determined final status", detail=final_status, final_status=final_status)

        finance_context = self._load_finance_context(state)
        emit_progress(
            "Write report",
            "loaded finance context",
            detail=(
                f"finance_brief={'yes' if finance_context['finance_brief'] else 'no'}, "
                f"leakage={'yes' if finance_context['leakage'] else 'no'}, "
                f"oos={'yes' if finance_context['oos'] else 'no'}, "
                f"robustness={'yes' if finance_context['robustness'] else 'no'}"
            ),
        )

        emit_progress("Write report", "generating report insights", detail="LLM summary and next steps")
        llm_result = self._llm_generate_insights(state, final_status, finance_context)
        if llm_result and _is_valid_insight(llm_result):
            short_conclusion = llm_result.get("conclusion", self._short_conclusion(final_status))
            next_steps = llm_result.get("next_steps", self._next_steps(final_status))
            emit_progress("Write report", "LLM insights accepted", detail=short_conclusion)
        else:
            logger.info("LLM unavailable or failed, falling back to template conclusion")
            short_conclusion = self._short_conclusion(final_status)
            next_steps = self._next_steps(final_status)
            emit_progress("Write report", "using template report insights", level="warning", detail=short_conclusion)

        next_steps = self._augment_next_steps_with_finance(state, list(next_steps))

        emit_progress("Write report", "rendering report template")
        template = jinja2.Template(
            (_TEMPLATES_DIR / "smoke_report.md.j2").read_text(encoding="utf-8")
        )

        report_content = template.render(
            timestamp=datetime.now().isoformat(),
            task_id=state.task_id,
            final_status=final_status,
            backend=state.backend,
            short_conclusion=short_conclusion,
            next_steps=next_steps,
            input=state.paper_input,
            paper=state.paper_metadata,
            brief=state.reproduction_brief,
            finance_brief=finance_context["finance_brief"],
            data_access=state.data_access,
            leakage=finance_context["leakage"],
            oos=finance_context["oos"],
            robustness=finance_context["robustness"],
            selected_repo=state.selected_repo,
            repo_candidates=state.repo_candidates[:5],
            repo_eval=state.repo_evaluation,
            env_build=state.env_build,
            smoke_run=state.smoke_run,
            reproduction_run=state.reproduction_run,
            benchmark_plan=state.benchmark_plan,
            benchmark_run=state.benchmark_run,
            errors=state.errors,
            step_timings=state.step_timings,
            api_calls=telemetry_records() or state.api_calls,
        )

        md_path = report_dir / "reproduction_smoke_report.md"
        json_path = report_dir / "reproduction_smoke_report.json"

        md_path.write_text(report_content, encoding="utf-8")
        emit_progress("Write report", "wrote markdown report", detail=str(md_path))

        report = ReportResult(
            final_status=final_status,
            report_markdown_path=str(md_path),
            report_json_path=str(json_path),
            short_conclusion=short_conclusion,
        )

        state.report = report
        save_json(json_path, report)
        emit_progress("Write report", "wrote report metadata", detail=str(json_path))

        state.status = "report_written"
        return state

    # -- finance context ---------------------------------------------------

    def _load_finance_context(self, state: TaskState) -> dict:
        task_dir = Path(state.task_dir)
        finance_brief = _load_json(task_dir / "paper" / "finance_brief.json")
        leakage = _load_json(task_dir / "leakage" / "leakage_report.json")
        oos = _load_json(task_dir / "oos" / "oos_report.json") or (
            state.oos if isinstance(state.oos, dict) else None
        )
        robustness = _load_json(task_dir / "robustness" / "robustness_matrix.json") or (
            state.robustness if isinstance(state.robustness, dict) else None
        )
        return {
            "finance_brief": finance_brief,
            "leakage": leakage,
            "oos": oos,
            "robustness": robustness,
        }

    # -- LLM insights ------------------------------------------------------

    def _llm_generate_insights(self, state: TaskState, final_status: str, finance_context: dict) -> dict | None:
        lines = [
            "Generate a report summary and next steps based on the structured execution results below.",
            "Output English natural language only; file paths, commands, environment variables, status codes, metric names and proper nouns may stay verbatim.",
            f"Final status: {final_status}",
        ]

        if state.paper_metadata and state.paper_metadata.title:
            lines.append(f"Paper title: {state.paper_metadata.title}")

        if state.selected_repo:
            lines.append(f"Repository: {state.selected_repo.url}")

        if state.repo_evaluation:
            lines.append(f"Runnable score: {state.repo_evaluation.runnable_score}")
            if state.repo_evaluation.risk_flags:
                lines.append(f"Risk flags: {', '.join(state.repo_evaluation.risk_flags)}")
            if state.repo_evaluation.benchmark_surface:
                lines.append(f"Benchmark surface analysis: {state.repo_evaluation.benchmark_surface}")

        if state.env_build:
            lines.append(f"Environment build success: {state.env_build.build_success}")
            if state.env_build.failure_summary:
                lines.append(f"Environment build failure summary: {state.env_build.failure_summary}")

        if state.smoke_run:
            if state.smoke_run.command:
                lines.append(f"Smoke command: {state.smoke_run.command.display}")
            lines.append(f"Smoke success: {state.smoke_run.success}")
            if state.smoke_run.exit_code is not None:
                lines.append(f"Exit code: {state.smoke_run.exit_code}")
            if state.smoke_run.summary:
                lines.append(f"Smoke summary: {state.smoke_run.summary}")
            if state.smoke_run.failure_type:
                lines.append(f"Smoke failure type: {state.smoke_run.failure_type}")
            if state.smoke_run.failure_evidence:
                lines.append(f"Smoke failure evidence: {state.smoke_run.failure_evidence}")

        if state.reproduction_run:
            if state.reproduction_run.command:
                lines.append(f"Lightweight reproduction command: {state.reproduction_run.command.display}")
            lines.append(f"Lightweight reproduction eligible: {state.reproduction_run.eligible}")
            lines.append(f"Lightweight reproduction skipped: {state.reproduction_run.skipped}")
            lines.append(f"Lightweight reproduction success: {state.reproduction_run.success}")
            if state.reproduction_run.skip_reason:
                lines.append(f"Lightweight reproduction skip reason: {state.reproduction_run.skip_reason}")
            if state.reproduction_run.summary:
                lines.append(f"Lightweight reproduction summary: {state.reproduction_run.summary}")
            if state.reproduction_run.failure_type:
                lines.append(f"Lightweight reproduction failure type: {state.reproduction_run.failure_type}")
            if state.reproduction_run.failure_evidence:
                lines.append(f"Lightweight reproduction failure evidence: {state.reproduction_run.failure_evidence}")
            if state.reproduction_run.output_artifacts:
                lines.append(f"Lightweight reproduction output artifacts: {state.reproduction_run.output_artifacts[:10]}")
            if state.reproduction_run.metrics:
                lines.append(f"Lightweight reproduction metrics: {state.reproduction_run.metrics}")
            if state.reproduction_run.reference_results:
                lines.append(f"Reference results: {state.reproduction_run.reference_results}")
            if state.reproduction_run.comparisons:
                lines.append(f"Metric comparisons: {state.reproduction_run.comparisons}")

        if state.benchmark_run:
            if state.benchmark_run.selected_spec:
                lines.append(
                    f"Selected benchmark: {state.benchmark_run.selected_spec.level} "
                    f"{state.benchmark_run.selected_spec.title}"
                )
                lines.append(f"Benchmark task family: {state.benchmark_run.selected_spec.task_family}")
            lines.append(f"Benchmark eligible: {state.benchmark_run.eligible}")
            lines.append(f"Benchmark skipped: {state.benchmark_run.skipped}")
            lines.append(f"Benchmark success: {state.benchmark_run.success}")
            if state.benchmark_run.summary:
                lines.append(f"Benchmark summary: {state.benchmark_run.summary}")
            if state.benchmark_run.downgrade_reasons:
                lines.append(f"Benchmark downgrade reasons: {state.benchmark_run.downgrade_reasons}")
            if state.benchmark_run.metrics:
                lines.append(f"Benchmark metrics: {state.benchmark_run.metrics}")
            if state.benchmark_run.comparisons:
                lines.append(f"Benchmark metric comparisons: {state.benchmark_run.comparisons}")
            if state.benchmark_run.parser_hints:
                lines.append(f"Benchmark parser hints: {state.benchmark_run.parser_hints}")
            if state.benchmark_run.failure_diagnosis:
                lines.append(f"Benchmark failure diagnosis: {state.benchmark_run.failure_diagnosis}")

        brief = finance_context.get("finance_brief")
        if brief:
            lines.append(f"Finance hypothesis: {brief.get('hypothesis') or 'N/A'}")
            period = brief.get("sample_period") or {}
            lines.append(
                f"Finance sample period: {period.get('start') or '?'} to "
                f"{period.get('end') or '?'} ({period.get('frequency') or 'unknown'})"
            )

        da = state.data_access
        if da is not None and da.is_finance_study:
            lines.append(f"Data access: ready={da.ready}; {da.summary or ''}")

        leakage = finance_context.get("leakage")
        if leakage:
            counts = leakage.get("counts") or {}
            lines.append(
                f"Leakage audit verdict: {leakage.get('verdict')} "
                f"(critical={counts.get('critical', 0)}, warning={counts.get('warning', 0)})"
            )
            for finding in (leakage.get("findings") or [])[:5]:
                lines.append(
                    f"  leakage finding [{finding.get('severity')}] "
                    f"{finding.get('rule_id')}: {finding.get('message')}"
                )

        oos = finance_context.get("oos")
        if oos and oos.get("ran"):
            lines.append(f"Out-of-sample verdict: {oos.get('verdict')}")
            for comparison in (oos.get("comparisons") or [])[:6]:
                lines.append(
                    f"  OOS comparison {comparison.get('metric')}: "
                    f"in-sample={comparison.get('in_sample')}, "
                    f"out-of-sample={comparison.get('out_of_sample')}, "
                    f"status={comparison.get('status')}"
                )
        elif oos and oos.get("skip_reason"):
            lines.append(f"Out-of-sample skipped: {oos.get('skip_reason')}")

        robustness = finance_context.get("robustness")
        if robustness and robustness.get("ran"):
            lines.append(
                f"Robustness verdict: {robustness.get('verdict')} "
                f"(sign consistency {robustness.get('sign_consistency')})"
            )
        elif robustness and robustness.get("skip_reason"):
            lines.append(f"Robustness skipped: {robustness.get('skip_reason')}")

        if state.errors:
            lines.append("Error records:")
            for err in state.errors:
                lines.append(f"  - {err}")

        return call_llm_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt="\n".join(lines),
            purpose="report_generation",
        )

    # -- status / conclusions ----------------------------------------------

    def _determine_final_status(self, state: TaskState):
        if not state.paper_metadata:
            return "paper_parse_failed"

        if not state.selected_repo:
            return "repo_not_found"

        # backend=none: only static analysis, no execution
        if state.backend == "none" and state.repo_evaluation:
            return "repo_found_smoke_not_run"

        if state.env_build and not state.env_build.build_success and not state.env_build.skipped:
            return "repo_found_but_env_failed"

        if state.benchmark_run:
            if state.benchmark_run.success:
                if state.benchmark_run.metrics or state.benchmark_run.comparisons:
                    return "benchmark_success"
                if state.reproduction_run and state.reproduction_run.success:
                    return "reproduction_success"
                if state.smoke_run and state.smoke_run.success:
                    if state.smoke_run.command and state.smoke_run.command.kind == "help":
                        return "partial_success_help_only"
                    return "success"
                return "success"
            if state.benchmark_run.skipped:
                if not state.reproduction_run:
                    return "repo_found_benchmark_not_run"
            else:
                if state.reproduction_run and state.reproduction_run.success:
                    return "reproduction_success_benchmark_failed"
                return "repo_found_but_benchmark_failed"

        if state.reproduction_run:
            if state.reproduction_run.success:
                return "reproduction_success"
            if state.reproduction_run.skipped:
                if state.smoke_run and state.smoke_run.success:
                    if state.smoke_run.command and state.smoke_run.command.kind == "help":
                        return "partial_success_help_only"
                    return "success"
                return "repo_found_reproduction_not_run"
            return "repo_found_but_reproduction_failed"

        # Check smoke_run results first (applies to both local and docker backends)
        if state.smoke_run and state.smoke_run.success:
            if state.smoke_run.command and state.smoke_run.command.kind == "help":
                return "partial_success_help_only"
            return "success"

        if state.smoke_run and not state.smoke_run.success:
            if state.smoke_run.failure_type in _SMOKE_ENV_FAILURE_TYPES:
                return "repo_found_but_env_failed"
            return "repo_found_but_smoke_failed"

        if state.env_build and state.env_build.skipped:
            return "skipped_docker"

        return "failed"

    def _short_conclusion(self, status: str) -> str:
        mapping = {
            "reproduction_success": "Lightweight end-to-end reproduction completed: repository, environment and a non-help reproduction command all ran successfully.",
            "reproduction_success_benchmark_failed": "Lightweight end-to-end reproduction succeeded, but the protocolized benchmark command failed; treat the reproduction success and the benchmark failure separately.",
            "benchmark_success": "Protocolized benchmark reproduction completed with structured metrics, reference results and downgrade notes.",
            "success": "Repository was found and the smoke-test command executed successfully.",
            "partial_success_help_only": "The --help command ran fine. This is a partial smoke-test success, not a full reproduction.",
            "repo_found_but_env_failed": "Repository was found, but the conda/venv/Docker dependency or execution environment was not ready.",
            "repo_found_but_smoke_failed": "Repository was found, but the smoke-test command failed.",
            "repo_found_but_reproduction_failed": "Repository and environment were ready, but the lightweight full-reproduction command failed.",
            "repo_found_but_benchmark_failed": "Repository and environment were ready, but the protocolized benchmark command failed.",
            "repo_found_reproduction_not_run": "Repository was found, but no lightweight full-reproduction command was executed.",
            "repo_found_benchmark_not_run": "Repository was found, but no runnable protocolized benchmark plan was produced.",
            "repo_found_smoke_not_run": "Repository was found and statically evaluated; no code was executed (backend=none).",
            "repo_not_found": "No suitable repository was found or provided.",
            "paper_parse_failed": "Paper parsing failed.",
            "skipped_docker": "The user skipped the Docker build and smoke test.",
            "failed": "The pipeline failed before reaching a conclusive result.",
        }
        return mapping.get(status, "Unknown status.")

    def _next_steps(self, status: str) -> list[str]:
        if status == "benchmark_success":
            return [
                "Inspect runs/benchmark_001/stdout.log, stderr.log and benchmark_summary.json to confirm protocol, metrics and downgrade reasons.",
                "If the achieved level is below L3, first supply the datasets, splits, weights or reference tables listed in the report.",
                "Re-run with the same BenchmarkSpec on the full dataset, keeping the metric parser and comparator unchanged.",
            ]

        if status == "reproduction_success":
            return [
                "Inspect runs/reproduction_001/stdout.log, stderr.log and the output artifact list to confirm reproduction outputs.",
                "Manually compare the lightweight reproduction outputs with the results shown in the paper or README.",
                "If stronger evidence is needed, extend to small-scale real data or official weights.",
            ]

        if status == "reproduction_success_benchmark_failed":
            return [
                "Inspect runs/reproduction_001/stdout.log to confirm the lightweight reproduction output — this is the part that already works.",
                "Inspect runs/benchmark_001/benchmark_candidates.json and stderr.log to check whether the benchmark planner selected the right task family.",
                "If the benchmark failure comes from missing dependencies or data, supply the official small dataset, weights or optional dependencies and re-run the benchmark.",
            ]

        if status == "success":
            return [
                "Manually run the evaluation commands described in the repository documentation.",
                "Download the required datasets or model weights as needed.",
                "Compare the outputs with the metrics reported in the paper.",
            ]

        if status == "partial_success_help_only":
            return [
                "Read the repository README to find the actual demo or evaluation command.",
                "If the repository ships sample inputs, try running them.",
                "Confirm dataset and weight requirements before attempting larger experiments.",
            ]

        if status == "repo_found_but_env_failed":
            return [
                "Open env/conda_build.log, env/venv_build.log, env/build.log or runs/smoke_001/stderr.log to find the first dependency error.",
                "Check whether the repository requires a specific Python, CUDA or PyTorch version.",
                "Retry with --backend conda in a local isolated environment; use --backend docker if Docker is available.",
            ]

        if status == "repo_found_but_smoke_failed":
            return [
                "Open runs/smoke_001/stderr.log for the error details.",
                "Check whether the command needs dataset paths, weight files or configuration files.",
                "Try running the command manually with --help first.",
            ]

        if status == "repo_found_but_reproduction_failed":
            return [
                "Open runs/reproduction_001/stderr.log for the lightweight reproduction failure reason.",
                "Check whether the command implicitly requires weights, sample inputs or configuration files.",
                "If the failure is due to small missing files, supply the official sample resources and re-run.",
            ]

        if status == "repo_found_but_benchmark_failed":
            return [
                "Open runs/benchmark_001/stderr.log for the benchmark command failure reason.",
                "Inspect runs/benchmark_001/benchmark_candidates.json to confirm the planner selected the right task family and level.",
                "Fix the official eval/benchmark script's sample data, weights or CUDA arguments first.",
            ]

        if status == "repo_found_benchmark_not_run":
            return [
                "Inspect runs/benchmark_001/benchmark_candidates.json for the L3/L2/L1 candidate protocols.",
                "Supply the official small dataset, sample files or README benchmark entry and re-run.",
                "If this is a new task family, add a task-family adapter rather than a paper-specific branch.",
            ]

        if status == "repo_found_reproduction_not_run":
            return [
                "Inspect runs/reproduction_001/command_candidates.json to see why no safe command was selected.",
                "Prefer demo/inference commands from the README that use bundled sample inputs.",
                "Avoid training commands, large-dataset evaluations and commands that require manually downloaded weights.",
            ]

        if status == "repo_not_found":
            return [
                "Specify the repository manually with --repo or --repo-dir.",
                "Search the paper PDF, project page, Papers with Code or the authors' homepages for the code repository.",
                "Re-run once the repository is found.",
            ]

        if status == "paper_parse_failed":
            return [
                "Confirm the input is a valid, readable local PDF file.",
                "If the paper is on arXiv, download the PDF first and pass the local path.",
                "Check whether the PDF is scanned or image-only.",
            ]

        if status == "skipped_docker":
            return [
                "Review the repository's runnable-score evaluation.",
                "Re-run with --backend conda to test environment building.",
                "Use --repo-dir for faster local iteration.",
            ]

        if status == "repo_found_smoke_not_run":
            return [
                "Run the smoke command locally with --backend local.",
                "Build a local isolated environment and test with --backend conda.",
                "Review the runnable score and candidate scripts to confirm the static evaluation is reasonable.",
            ]

        return [
            "Check state.json for the failing step.",
            "Review the error messages recorded in the report.",
            "Re-run with --repo-dir to rule out GitHub search and clone issues.",
        ]

    def _augment_next_steps_with_finance(self, state: TaskState, steps: list[str]) -> list[str]:
        da = state.data_access
        if da is None or not da.is_finance_study:
            return steps
        subscription = [r for r in da.records if r.status == "requires_subscription"]
        if subscription and not da.ready:
            names = ", ".join(r.name for r in subscription[:3])
            steps.append(
                f"Provision subscription-licensed extracts ({names}) and set their "
                "ALPHA_ASSAY_*_DIR environment variables, then re-run to unlock the "
                "hypothesis-level (L3) replication and out-of-sample tests."
            )
        return steps


def _is_valid_insight(result: dict) -> bool:
    conclusion = result.get("conclusion")
    next_steps = result.get("next_steps")
    if not isinstance(conclusion, str) or not conclusion.strip():
        return False
    if not isinstance(next_steps, list) or not next_steps:
        return False
    return all(isinstance(step, str) and step.strip() for step in next_steps)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None
