from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from app.agents.smoke_run_agent import SmokeRunAgent, classify_smoke_failure
from app.benchmark.device import is_cuda_available_for_state
from app.benchmark.comparator import compare_metrics, protocol_match
from app.benchmark.llm_planner import apply_llm_review, llm_review_benchmark_plan
from app.benchmark.parsers import parse_metrics
from app.benchmark.planner import LEVEL_RANK, downgrade_reasons, plan_benchmarks, select_best_benchmark
from app.benchmark.schema import BenchmarkRunResult, BenchmarkSpec, ExecutionBudget, KNOWN_TASK_FAMILIES
from app.core.file_utils import save_json
from app.core.progress import emit_progress
from app.core.state import TaskState
from app.tools.llm import call_llm_json


class BenchmarkReproductionAgent:
    """Plan and run the highest feasible task-family benchmark."""

    def __init__(self, timeout_minutes: int = 30):
        self.timeout_minutes = timeout_minutes
        self._runner = SmokeRunAgent(timeout_minutes=timeout_minutes)

    def run(self, state: TaskState) -> TaskState:
        task_dir = Path(state.task_dir).resolve()
        run_dir = task_dir / "runs" / "benchmark_001"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        result = BenchmarkRunResult(target_level="L3")
        skip_reason = self._skip_reason(state)
        if skip_reason:
            result.skipped = True
            result.skip_reason = skip_reason
            result.summary = f"Skipped benchmark reproduction: {skip_reason}"
            state.benchmark_run = result
            state.status = "benchmark_ran"
            save_json(run_dir / "benchmark_summary.json", result)
            emit_progress("Run benchmark reproduction", "skipped", level="warning", detail=skip_reason)
            return state

        budget = ExecutionBudget(target_level="L3", max_runtime_minutes=self.timeout_minutes)
        emit_progress("Run benchmark reproduction", "planning benchmark candidates", detail=budget.target_level)
        specs = plan_benchmarks(state, budget)
        emit_progress("Run benchmark reproduction", "reviewing benchmark plan", detail=f"{len(specs)} candidate(s)")
        review = llm_review_benchmark_plan(state, specs)
        specs = apply_llm_review(specs, review)
        state.benchmark_plan = specs
        save_json(run_dir / "benchmark_candidates.json", [spec.model_dump() for spec in specs])
        emit_progress(
            "Run benchmark reproduction",
            "saved benchmark candidates",
            detail=f"{len(specs)} candidate(s)",
            candidate_count=len(specs),
        )
        result.candidate_specs = specs
        result.eligible = bool(specs)

        # If CUDA unavailable, prefer CPU specs
        repo_dir = Path(state.repo_evaluation.repo_dir).resolve()
        if not is_cuda_available_for_state(state, repo_dir):
            emit_progress("Run benchmark reproduction", "CUDA 不可用，优先选择 CPU benchmark",
                          level="warning", detail="当前 PyTorch 不支持 CUDA，降级为 CPU")
        selected = select_best_benchmark(specs, budget)
        result.selected_spec = selected
        result.downgrade_reasons = downgrade_reasons(specs, selected, budget)
        if selected is not None:
            emit_progress(
                "Run benchmark reproduction",
                "selected benchmark plan",
                detail=f"{selected.level} {selected.title}",
                benchmark_id=selected.id,
                benchmark_level=selected.level,
            )

        if selected is None:
            result.skipped = True
            result.skip_reason = "No runnable benchmark plan found"
            result.summary = result.skip_reason
            state.benchmark_run = result
            state.status = "benchmark_ran"
            save_json(run_dir / "benchmark_summary.json", result)
            emit_progress("Run benchmark reproduction", "no runnable benchmark", level="warning")
            return state

        deadline = time.monotonic() + self.timeout_minutes * 60
        tried_ids: set[str] = set()
        while selected is not None:
            tried_ids.add(selected.id)
            self._execute_selected_benchmark(result, selected, state, task_dir, run_dir, deadline)
            if result.success or result.skipped:
                break

            # CUDA failure → try CPU fallback from same spec
            if result.failure_type == "cuda_unavailable":
                cpu_spec = _cpu_fallback_spec(selected)
                if cpu_spec and cpu_spec.id not in tried_ids:
                    emit_progress("Run benchmark reproduction", "CUDA 失败，尝试 CPU fallback",
                                  level="warning", detail=f"{cpu_spec.level} {cpu_spec.title}")
                    selected = cpu_spec
                    continue

            fallback = self._fallback_benchmark(specs, selected, tried_ids, budget)
            if fallback is None:
                break
            reason = (
                f"{selected.level} {selected.title} failed with "
                f"{result.failure_type or 'unknown'}; retrying {fallback.level} {fallback.title}"
            )
            result.downgrade_reasons.append(reason)
            emit_progress("Run benchmark reproduction", "retrying fallback", level="warning", detail=f"{fallback.level} {fallback.title}")
            selected = fallback

        state.benchmark_run = result
        state.status = "benchmark_ran"
        save_json(run_dir / "benchmark_summary.json", result)
        return state

    def _skip_reason(self, state: TaskState) -> str | None:
        if state.backend == "none":
            return "backend=none"
        if state.backend in {"venv", "conda", "docker"} and (
            not state.env_build or not state.env_build.build_success
        ):
            return f"{state.backend} environment did not build"
        if not state.repo_evaluation or not state.repo_evaluation.repo_dir:
            return "repo was not evaluated"
        return None

    def _argv_for_backend(self, argv: list[str], state: TaskState, task_dir: Path) -> list[str]:
        if state.backend in {"venv", "conda"}:
            if len(argv) >= 3 and argv[0] == "python" and argv[1] == "-m":
                python_bin = self._runner._venv_python(state)
                if python_bin:
                    return [python_bin, *argv[1:]]
            return self._runner._venv_argv(argv, state, task_dir)
        return argv

    def _execute_selected_benchmark(
        self,
        result: BenchmarkRunResult,
        selected: BenchmarkSpec,
        state: TaskState,
        task_dir: Path,
        run_dir: Path,
        deadline: float,
    ) -> None:
        result.selected_spec = selected
        repo_dir = Path(state.repo_evaluation.repo_dir).resolve()
        if selected.generated_script_name and selected.generated_script_body:
            (repo_dir / selected.generated_script_name).write_text(selected.generated_script_body, encoding="utf-8")

        ok, reason = _is_safe_benchmark_argv(selected.command, state.repo_evaluation.candidate_scripts, selected)
        if not ok:
            result.skipped = True
            result.skip_reason = f"Command blocked: {reason}"
            result.summary = result.skip_reason
            emit_progress("Run benchmark reproduction", "command blocked", level="warning", detail=reason)
            return

        before = _snapshot_files(repo_dir)
        emit_progress("Run benchmark reproduction", "selected plan", detail=f"{selected.level} {selected.title}")
        emit_progress("Run benchmark reproduction", "running benchmark command", detail=" ".join(selected.command), command=selected.command)
        log_stem = _safe_artifact_stem(selected.id)
        stdout_path = run_dir / f"{log_stem}_stdout.log"
        stderr_path = run_dir / f"{log_stem}_stderr.log"
        latest_stdout_path = run_dir / "stdout.log"
        latest_stderr_path = run_dir / "stderr.log"

        try:
            proc = subprocess.run(
                self._argv_for_backend(selected.command, state, task_dir),
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=max(1, int(deadline - time.monotonic())),
                env=self._runner._venv_env(state) if state.backend in {"venv", "conda"} else os.environ.copy(),
            )
            stdout_path.write_text(proc.stdout, encoding="utf-8")
            stderr_path.write_text(proc.stderr, encoding="utf-8")
            latest_stdout_path.write_text(proc.stdout, encoding="utf-8")
            latest_stderr_path.write_text(proc.stderr, encoding="utf-8")

            result.exit_code = proc.returncode
            result.success = proc.returncode == 0
            result.skipped = False
            result.skip_reason = None
            result.timed_out = False
            result.stdout_path = str(stdout_path)
            result.stderr_path = str(stderr_path)
            result.achieved_level = selected.level
            result.output_artifacts = _collect_artifacts(repo_dir, before)
            emit_progress(
                "Run benchmark reproduction",
                "parsing benchmark outputs",
                detail=f"exit={proc.returncode}",
                exit_code=proc.returncode,
            )
            result.metrics = parse_metrics(selected, proc.stdout, proc.stderr, repo_dir, run_dir)
            if not result.metrics and selected.task_family not in KNOWN_TASK_FAMILIES:
                from app.benchmark.generic_metric_parser import parse_with_llm_fallback
                result.metrics = parse_with_llm_fallback(selected, proc.stdout, proc.stderr, repo_dir, run_dir)
            result.reference_results = selected.reference
            result.comparisons = compare_metrics(selected, result.metrics)
            result.protocol_match = protocol_match(selected)

            if result.success:
                result.failure_type = None
                result.failure_evidence = None
                result.failure_diagnosis = {}
                result.summary = f"Benchmark reproduction completed at {selected.level}: {selected.title}"
                if not result.metrics:
                    result.parser_hints = _llm_synthesize_metric_parser(selected, proc.stdout, proc.stderr, result.output_artifacts)
                emit_progress("Run benchmark reproduction", "completed", level="success", detail=selected.level)
            else:
                failure_type, evidence = classify_smoke_failure(proc.stderr, proc.stdout)
                result.failure_type = failure_type
                result.failure_evidence = evidence
                result.failure_diagnosis = _llm_diagnose_benchmark_failure(selected, proc.stdout, proc.stderr, failure_type)
                result.summary = f"Benchmark reproduction failed with exit code {proc.returncode}"
                emit_progress("Run benchmark reproduction", "failed", level="warning", detail=failure_type)

        except subprocess.TimeoutExpired as e:
            stdout_text = _process_output_text(e.stdout)
            stderr_text = _process_output_text(e.stderr)
            stdout_path.write_text(stdout_text, encoding="utf-8")
            stderr_path.write_text(stderr_text, encoding="utf-8")
            latest_stdout_path.write_text(stdout_text, encoding="utf-8")
            latest_stderr_path.write_text(stderr_text, encoding="utf-8")
            result.success = False
            result.timed_out = True
            result.failure_type = "timeout"
            result.failure_evidence = "Benchmark timed out"
            result.stdout_path = str(stdout_path)
            result.stderr_path = str(stderr_path)
            result.failure_diagnosis = _llm_diagnose_benchmark_failure(selected, stdout_text, stderr_text or "Benchmark timed out", "timeout")
            result.summary = "Benchmark reproduction timed out"
            emit_progress("Run benchmark reproduction", "timed out", level="error")
        except Exception as e:
            result.success = False
            result.failure_type = "execution_error"
            result.failure_evidence = str(e)
            result.failure_diagnosis = _llm_diagnose_benchmark_failure(selected, "", str(e), "execution_error")
            result.summary = f"Benchmark reproduction execution error: {e}"
            emit_progress("Run benchmark reproduction", "execution error", level="error", detail=str(e))

    def _fallback_benchmark(
        self,
        specs: list[BenchmarkSpec],
        failed: BenchmarkSpec,
        tried_ids: set[str],
        budget: ExecutionBudget,
    ) -> BenchmarkSpec | None:
        lower_specs = [
            spec for spec in specs
            if spec.id not in tried_ids and LEVEL_RANK[spec.level] < LEVEL_RANK[failed.level]
        ]
        return select_best_benchmark(lower_specs, budget)


def _is_safe_benchmark_argv(argv: list[str], candidate_scripts: list[str], spec: BenchmarkSpec) -> tuple[bool, str | None]:
    if not argv:
        return False, "empty command"
    if any(any(char in part for char in [";", "&", "|", "$", "`", ">", "<"]) for part in argv):
        return False, "shell metacharacter blocked"
    if argv[0] != "python":
        return False, "executable not allowed"
    if len(argv) < 2:
        return False, "missing script"
    if len(argv) >= 3 and argv[1] == "-m":
        module = argv[2]
        if not _is_safe_benchmark_module(module):
            return False, "unsafe module"
        for arg in argv[3:]:
            if not _is_safe_benchmark_arg(arg):
                return False, f"unsafe argument: {arg}"
        return True, None
    script = argv[1]
    if Path(script).is_absolute() or ".." in Path(script).parts or not script.endswith(".py"):
        return False, "unsafe script path"
    allowed_scripts = set(candidate_scripts)
    if spec.generated_script_name:
        allowed_scripts.add(spec.generated_script_name)
    if script not in allowed_scripts:
        return False, "script is not in candidate scripts or generated benchmark runner"
    lowered = [part.lower() for part in argv]
    if any("train" in part for part in lowered):
        return False, "training command blocked"
    if any(part in {"--help", "-h", "--version"} for part in lowered):
        return False, "help/version command is not a benchmark"
    for arg in argv[2:]:
        if not _is_safe_benchmark_arg(arg):
            return False, f"unsafe argument: {arg}"
    return True, None


def _is_safe_benchmark_module(module: str) -> bool:
    import re

    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+", module)) and not any(
        part in {"os", "sys", "subprocess", "shutil"}
        for part in module.split(".")
    )


def _is_safe_benchmark_arg(arg: str) -> bool:
    import re

    if not arg or len(arg) > 260:
        return False
    if any(char in arg for char in [";", "&", "|", "$", "`", ">", "<"]):
        return False
    if arg.startswith("-"):
        return bool(re.fullmatch(r"-{1,2}[A-Za-z0-9][A-Za-z0-9_\-]*", arg))
    return bool(re.fullmatch(r"[A-Za-z0-9_./:=,+@%\\-]+", arg))


def _snapshot_files(repo_dir: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_dir)
        if _ignored_artifact_path(rel):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[rel.as_posix()] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _collect_artifacts(repo_dir: Path, before: dict[str, tuple[int, int]], limit: int = 50) -> list[str]:
    artifacts: list[str] = []
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_dir)
        if _ignored_artifact_path(rel):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        key = rel.as_posix()
        current = (stat.st_mtime_ns, stat.st_size)
        if before.get(key) != current and stat.st_size <= 50 * 1024 * 1024:
            artifacts.append(key)
        if len(artifacts) >= limit:
            break
    return sorted(artifacts)


def _ignored_artifact_path(path: Path) -> bool:
    ignored = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}
    return any(part in ignored or part.startswith(".") for part in path.parts)


def _safe_artifact_stem(value: str) -> str:
    import re

    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem[:120] or "benchmark"


def _process_output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


_FAILURE_DIAGNOSIS_PROMPT = """\
You are a benchmark failure diagnosis assistant.

Given a benchmark spec and stdout/stderr, explain the failure in structured form.
Do not suggest training unless the failure is explicitly about training.

Return a JSON object with exactly these keys:
- "failure_category": one of "missing_dataset", "missing_weight", "dependency", "cuda", "argument", "timeout", "parser", "unknown"
- "root_cause": short string
- "evidence": list of short strings copied or paraphrased from logs
- "recommended_next_actions": list of concrete actions
- "fallback_level_recommendation": "L3", "L2", "L1", "L0", or null
- "confidence": float between 0 and 1
"""


_PARSER_SYNTHESIS_PROMPT = """\
You are a metric parser synthesis assistant.

The deterministic parser extracted no metrics. Inspect stdout/stderr and suggest
safe parsing hints. Do not invent metric values.

Return a JSON object with exactly these keys:
- "metric_patterns": object mapping metric names to regex patterns
- "output_files_to_check": list of relative output file paths or globs
- "observed_metric_like_lines": list of short lines from stdout/stderr that look metric-like
- "notes": list of short strings
- "confidence": float between 0 and 1
"""


def _llm_diagnose_benchmark_failure(spec: BenchmarkSpec, stdout: str, stderr: str, failure_type: str | None) -> dict:
    payload = {
        "failure_type": failure_type,
        "spec": spec.model_dump(),
        "stdout_tail": stdout[-5000:],
        "stderr_tail": stderr[-5000:],
    }
    result = call_llm_json(
        system_prompt=_FAILURE_DIAGNOSIS_PROMPT,
        user_prompt=_json_dumps(payload),
        purpose="benchmark_failure_diagnosis",
        max_tokens=2048,
    )
    return result if isinstance(result, dict) else {}


def _llm_synthesize_metric_parser(spec: BenchmarkSpec, stdout: str, stderr: str, artifacts: list[str]) -> dict:
    payload = {
        "spec": spec.model_dump(),
        "stdout_tail": stdout[-7000:],
        "stderr_tail": stderr[-3000:],
        "output_artifacts": artifacts[:30],
    }
    result = call_llm_json(
        system_prompt=_PARSER_SYNTHESIS_PROMPT,
        user_prompt=_json_dumps(payload),
        purpose="metric_parser_synthesis",
        max_tokens=2048,
    )
    return result if isinstance(result, dict) else {}


def _json_dumps(value: dict) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def _cpu_fallback_spec(spec) -> object | None:
    """Create a CPU-fallback version of a CUDA benchmark spec."""
    from app.benchmark.schema import BenchmarkSpec
    cmd = list(spec.command)
    lowered = [str(x).lower() for x in cmd]
    if "--device" not in lowered:
        return None
    idx = lowered.index("--device")
    if idx + 1 >= len(cmd) or cmd[idx + 1].lower() != "cuda":
        return None
    cmd[idx + 1] = "cpu"

    if "--repeat" in lowered:
        ridx = lowered.index("--repeat")
        if ridx + 1 < len(cmd):
            cmd[ridx + 1] = "1"

    if "--num_keypoints" in lowered:
        kidx = lowered.index("--num_keypoints")
        new_cmd = []
        i = 0
        while i < len(cmd):
            if str(cmd[i]).lower() == "--num_keypoints":
                new_cmd.extend(["--num_keypoints", "512"])
                i += 1
                while i < len(cmd) and not str(cmd[i]).startswith("--"):
                    i += 1
                continue
            new_cmd.append(cmd[i])
            i += 1
        cmd = new_cmd

    return BenchmarkSpec(
        id=spec.id + "_cpu",
        title=spec.title + " CPU",
        level="L1",
        command=cmd,
        task_family=spec.task_family,
        runnable=True,
        command_kind=spec.command_kind,
    )
