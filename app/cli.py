import json
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from app.core.config import settings
from app.core.file_utils import save_json, save_state, load_state
from app.core.paths import TaskPaths, generate_task_id
from app.core.progress import ProgressEvent, emit_progress, progress_events
from app.core.state import TaskState, RepoCandidate, EnvironmentBuildResult, StepTiming, ApiCallRecord
from app.tools.network import sanitize_proxy_env

app = typer.Typer()
console = Console()


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat()


def _format_progress_event(event: ProgressEvent) -> str:
    color = {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red",
    }.get(event.level, "white")
    text = f"[{color}]{escape(event.stage)}[/{color}] {escape(event.message)}"
    if event.detail:
        text += f" [dim]{escape(event.detail)}[/dim]"
    return text


def _run_pipeline(
    input_value: str,
    workspace: str = settings.default_workspace,
    backend: str = "conda",
    repo: Optional[str] = None,
    repo_dir: Optional[str] = None,
    timeout_minutes: int = 30,
    max_repair_attempts: int = settings.default_max_repair_attempts,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> TaskState:
    """Run the full pipeline and return the final TaskState."""
    sanitize_proxy_env()
    task_id, paths = _create_fresh_task_paths(workspace)

    state = TaskState(
        task_id=task_id,
        input_value=input_value,
        workspace_dir=str(Path(workspace)),
        task_dir=str(paths.task_dir),
        backend=backend,
    )

    if repo_dir:
        state.selected_repo = RepoCandidate(
            url=f"local:{repo_dir}",
            source="local",
            score=100.0,
            confidence="high",
            reasons=["Provided by --repo-dir"],
            local_path=str(Path(repo_dir).resolve()),
        )

    if repo:
        state.selected_repo = RepoCandidate(
            url=repo,
            source="manual",
            score=100.0,
            confidence="high",
            reasons=["Provided by --repo"],
        )

    save_state(state)
    emit_progress("Pipeline", f"created task {task_id}", detail=str(paths.task_dir))

    from app.agents.paper_ingest_agent import PaperIngestAgent
    from app.agents.paper_understanding_agent import PaperUnderstandingAgent
    from app.agents.github_search_agent import GitHubSearchAgent
    from app.agents.repo_evaluation_agent import RepoEvaluationAgent
    from app.agents.docker_build_agent import DockerBuildAgent
    from app.agents.conda_build_agent import CondaBuildAgent
    from app.agents.venv_build_agent import VenvBuildAgent
    from app.agents.runtime_decision_agent import RuntimeDecisionAgent
    from app.agents.smoke_run_agent import SmokeRunAgent
    from app.agents.benchmark_reproduction_agent import BenchmarkReproductionAgent
    from app.agents.simple_reproduction_agent import SimpleReproductionAgent
    from app.agents.report_writer_agent import ReportWriterAgent

    agents = [
        ("Ingest paper", PaperIngestAgent()),
        ("Understand paper", PaperUnderstandingAgent()),
    ]

    if not state.selected_repo:
        agents.append(("Search GitHub", GitHubSearchAgent()))

    agents.append(("Evaluate repo", RepoEvaluationAgent()))
    agents.append(("Decide runtime", RuntimeDecisionAgent()))

    if backend == "docker":
        agents.extend([
            ("Build Docker image", DockerBuildAgent(timeout_minutes=timeout_minutes)),
            ("Run smoke command", SmokeRunAgent(timeout_minutes=timeout_minutes, max_repair_attempts=max_repair_attempts)),
            ("Run benchmark reproduction", BenchmarkReproductionAgent(timeout_minutes=timeout_minutes)),
            ("Run simple reproduction", SimpleReproductionAgent(timeout_minutes=timeout_minutes)),
        ])
    elif backend == "venv":
        agents.extend([
            ("Build virtualenv", VenvBuildAgent(timeout_minutes=timeout_minutes)),
            ("Run smoke command", SmokeRunAgent(timeout_minutes=timeout_minutes, max_repair_attempts=max_repair_attempts)),
            ("Run benchmark reproduction", BenchmarkReproductionAgent(timeout_minutes=timeout_minutes)),
            ("Run simple reproduction", SimpleReproductionAgent(timeout_minutes=timeout_minutes)),
        ])
    elif backend == "conda":
        agents.extend([
            ("Build conda env", CondaBuildAgent(timeout_minutes=timeout_minutes)),
            ("Run smoke command", SmokeRunAgent(timeout_minutes=timeout_minutes, max_repair_attempts=max_repair_attempts)),
            ("Run benchmark reproduction", BenchmarkReproductionAgent(timeout_minutes=timeout_minutes)),
            ("Run simple reproduction", SimpleReproductionAgent(timeout_minutes=timeout_minutes)),
        ])
    elif backend == "local":
        state.env_build = EnvironmentBuildResult(
            skipped=True,
            failure_summary="Isolated environment build skipped (backend=local)",
        )
        save_state(state)
        agents.extend([
            ("Run smoke command", SmokeRunAgent(timeout_minutes=timeout_minutes, max_repair_attempts=max_repair_attempts)),
            ("Run benchmark reproduction", BenchmarkReproductionAgent(timeout_minutes=timeout_minutes)),
            ("Run simple reproduction", SimpleReproductionAgent(timeout_minutes=timeout_minutes)),
        ])
    else:  # none
        state.env_build = EnvironmentBuildResult(
            skipped=True,
            failure_summary="Docker build skipped (backend=none)",
        )
        save_state(state)

    agents.append(("Write report", ReportWriterAgent()))

    # Enable LLM telemetry collection
    from app.tools.llm import enable_telemetry, disable_telemetry
    api_sink: list[dict] = []
    enable_telemetry(api_sink)

    try:
        for desc, agent in agents:
            if should_cancel and should_cancel():
                state.status = "cancelled"
                emit_progress("Pipeline", "cancelled", level="warning", phase="fail")
                save_state(state)
                break
            t0 = time.time()
            t0_iso = _now_iso()
            step_ok = True
            emit_progress(desc, "started", phase="start")
            try:
                state = agent.run(state)
                step_ok = _step_succeeded(desc, state)
            except Exception as e:
                state.errors.append({"step": desc, "error": str(e)})
                state.status = "failed"
                step_ok = False
                emit_progress(desc, "failed", level="error", phase="fail", detail=str(e))

            elapsed_ms = int((time.time() - t0) * 1000)
            state.step_timings.append(StepTiming(
                step=desc,
                started_at=t0_iso,
                ended_at=_now_iso(),
                duration_ms=elapsed_ms,
                success=step_ok,
            ))

            # Sync API calls collected so far
            state.api_calls = [ApiCallRecord(**r) for r in api_sink]

            save_state(state)
            if step_ok:
                emit_progress(
                    desc,
                    "completed",
                    level="success",
                    phase="finish",
                    detail=f"{elapsed_ms / 1000:.1f}s",
                    duration_ms=elapsed_ms,
                    status=state.status,
                )
            else:
                if not state.errors or all(e.get("step") != desc for e in state.errors):
                    state.errors.append({"step": desc, "error": "Step completed with errors"})
                    state.status = "failed"
                    save_state(state)
                emit_progress(
                    desc,
                    "failed",
                    level="error",
                    phase="fail",
                    detail=f"business check failed ({elapsed_ms / 1000:.1f}s)",
                    duration_ms=elapsed_ms,
                    status=state.status,
                )
    finally:
        disable_telemetry()

    # Collect API call telemetry into state
    state.api_calls = [ApiCallRecord(**r) for r in api_sink]
    save_state(state)

    return state


def _step_succeeded(desc: str, state: TaskState) -> bool:
    """Check business-level success for a pipeline step based on TaskState."""
    if desc in {"Build virtualenv", "Build conda env", "Build Docker image"}:
        return bool(state.env_build and state.env_build.build_success)
    if desc == "Run smoke command":
        return bool(state.smoke_run and state.smoke_run.success)
    if desc == "Run simple reproduction":
        return bool(
            state.reproduction_run
            and (state.reproduction_run.success or state.reproduction_run.skipped)
        )
    if desc == "Run benchmark reproduction":
        return bool(
            state.benchmark_run
            and (state.benchmark_run.success or state.benchmark_run.skipped)
        )
    if desc == "Write report":
        return state.report is not None
    if desc == "Ingest paper":
        return state.paper_metadata is not None
    if desc == "Understand paper":
        return state.reproduction_brief is not None
    if desc == "Search GitHub":
        return bool(state.repo_candidates)
    if desc == "Evaluate repo":
        return state.repo_evaluation is not None
    if desc == "Decide runtime":
        return state.runtime_decision is not None
    return state.status not in {"failed", "cancelled"}


def _create_fresh_task_paths(workspace: str) -> Tuple[str, TaskPaths]:
    for _ in range(100):
        task_id = generate_task_id()
        paths = TaskPaths(workspace, task_id)
        try:
            paths.create_all_dirs()
        except FileExistsError:
            continue
        return task_id, paths
    raise RuntimeError("Could not create a unique task directory")


def _normalize_repo_url(url: str) -> str:
    """Normalize GitHub URL for comparison (strip trailing slash, .git, http→https, lowercase)."""
    url = url.strip().rstrip("/")
    if url.startswith("http://"):
        url = "https://" + url[7:]
    if url.endswith(".git"):
        url = url[:-4]
    return url.lower()


def _resolve_gold_item_input(item: dict, gold_path: Path) -> str:
    """Prefer checked-in local PDFs when a gold item uses an external paper URL."""
    local_pdf = gold_path.parent / "gold_papers" / f"{item['name']}.pdf"
    if local_pdf.exists():
        return str(local_pdf)
    return item["input"]


@app.command()
def run(
    input: str = typer.Option(..., "--input", "--paper-path", help="Local paper PDF path"),
    repo: Optional[str] = typer.Option(None, "--repo", help="Manual GitHub repo URL"),
    repo_dir: Optional[str] = typer.Option(None, "--repo-dir", help="Local repository directory"),
    workspace: str = typer.Option(settings.default_workspace, "--workspace"),
    backend: str = typer.Option(settings.default_backend, "--backend", help="Execution backend: none, local, venv, conda, or docker"),
    timeout_minutes: int = typer.Option(settings.default_timeout_minutes, "--timeout-minutes"),
    max_repair_attempts: int = typer.Option(settings.default_max_repair_attempts, "--max-repair-attempts"),
    skip_docker_build: bool = typer.Option(False, "--skip-docker-build", help="Deprecated: use --backend none"),
):
    if skip_docker_build:
        backend = "none"

    if backend not in ("none", "local", "venv", "conda", "docker"):
        console.print(f"[red]Invalid backend: {backend}. Must be none, local, venv, conda, or docker.[/red]")
        raise typer.Exit(1)

    console.print("[bold cyan]alpha-assay v0.1.1[/bold cyan]")
    console.print(f"Input: {input}")
    console.print(f"Backend: {backend}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running pipeline", total=None)
        def on_progress(event: ProgressEvent) -> None:
            progress.update(task, description=event.message)
            progress.console.print(_format_progress_event(event))

        with progress_events(on_progress):
            state = _run_pipeline(
                input_value=input,
                workspace=workspace,
                backend=backend,
                repo=repo,
                repo_dir=repo_dir,
                timeout_minutes=timeout_minutes,
                max_repair_attempts=max_repair_attempts,
            )
        progress.update(task, completed=True)

    report_path = Path(state.task_dir) / "report" / "reproduction_smoke_report.md"
    console.print(f"\n[green]Done[/green] Task completed: {state.task_id}")
    console.print(f"[green]Done[/green] Report: {report_path}")


@app.command("eval-goldset")
def eval_goldset(
    gold_set: str = typer.Option("examples/gold_set.json", "--gold-set", help="Path to gold set JSON"),
    workspace: str = typer.Option("./goldset_results", "--workspace"),
    backend: str = typer.Option("none", "--backend", help="Execution backend: none, local, venv, conda, or docker"),
    max_items: int = typer.Option(0, "--max-items", help="Max items to run (0=all)"),
    timeout_minutes: int = typer.Option(settings.default_timeout_minutes, "--timeout-minutes"),
    max_repair_attempts: int = typer.Option(settings.default_max_repair_attempts, "--max-repair-attempts"),
):
    """Run the pipeline on a gold set of papers and evaluate repo discovery accuracy."""
    gold_path = Path(gold_set)
    if not gold_path.exists():
        console.print(f"[red]Gold set file not found: {gold_set}[/red]")
        raise typer.Exit(1)

    gold_items = json.loads(gold_path.read_text(encoding="utf-8"))
    if max_items > 0:
        gold_items = gold_items[:max_items]

    console.print(f"[bold cyan]Gold Set Evaluation[/bold cyan]")
    console.print(f"Items: {len(gold_items)}, Backend: {backend}")
    console.print()

    results = []

    for i, item in enumerate(gold_items):
        name = item["name"]
        run_input = _resolve_gold_item_input(item, gold_path)
        console.rule(f"[bold]{i + 1}/{len(gold_items)}: {name}[/bold]")
        console.print(f"Input: {item['input']}")
        if run_input != item["input"]:
            console.print(f"Local PDF: {run_input}")
        console.print(f"Expected repo: {item['expected_repo']}")
        console.print(f"Hint type: {item['repo_hint_type']}")
        console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running {name}", total=None)
            try:
                def on_progress(event: ProgressEvent) -> None:
                    progress.update(task, description=event.message)
                    progress.console.print(_format_progress_event(event))

                with progress_events(on_progress):
                    state = _run_pipeline(
                        input_value=run_input,
                        workspace=str(Path(workspace) / "runs"),
                        backend=backend,
                        timeout_minutes=timeout_minutes,
                        max_repair_attempts=max_repair_attempts,
                    )
            except Exception as e:
                console.print(f"[red]Pipeline crashed: {e}[/red]")
                state = None
            progress.update(task, completed=True)

        # Evaluate result
        result = {
            "name": name,
            "input": item["input"],
            "effective_input": run_input,
            "expected_repo": item["expected_repo"],
            "difficulty": item["difficulty"],
            "repo_hint_type": item["repo_hint_type"],
        }

        if state is None:
            result.update({
                "pipeline_crashed": True,
                "paper_parse_success": False,
                "repo_discovered": False,
                "repo_correct": False,
                "selected_repo": None,
                "candidate_count": 0,
                "top_candidates": [],
                "final_status": "crashed",
                "task_dir": None,
            })
        else:
            selected_url = state.selected_repo.url if state.selected_repo else None
            expected = _normalize_repo_url(item["expected_repo"])
            actual = _normalize_repo_url(selected_url) if selected_url else None

            # Check if expected repo appears in top 5 candidates
            top_candidates = []
            for c in state.repo_candidates[:5]:
                top_candidates.append({
                    "url": c.url,
                    "score": c.score,
                    "source": c.source,
                    "reasons": c.reasons,
                })

            expected_in_top5 = any(
                _normalize_repo_url(c.url) == expected for c in state.repo_candidates[:5]
            )

            result.update({
                "pipeline_crashed": False,
                "paper_parse_success": state.paper_metadata is not None and state.paper_metadata.title is not None,
                "paper_title": state.paper_metadata.title if state.paper_metadata else None,
                "repo_discovered": selected_url is not None,
                "repo_correct": actual == expected if actual else False,
                "expected_in_top5": expected_in_top5,
                "selected_repo": selected_url,
                "selected_repo_score": state.selected_repo.score if state.selected_repo else None,
                "selected_repo_source": state.selected_repo.source if state.selected_repo else None,
                "selected_repo_reasons": state.selected_repo.reasons if state.selected_repo else [],
                "candidate_count": len(state.repo_candidates),
                "top_candidates": top_candidates,
                "github_links_in_paper": (
                    state.reproduction_brief.github_links_in_paper
                    if state.reproduction_brief else []
                ),
                "final_status": state.report.final_status if state.report else state.status,
                "errors": state.errors,
                "task_dir": state.task_dir,
            })

        results.append(result)

        # Print per-item summary
        if result["pipeline_crashed"]:
            console.print(f"  [red]CRASHED[/red]")
        elif result["repo_correct"]:
            console.print(f"  [green]CORRECT[/green] repo found: {result['selected_repo']}")
        elif result["repo_discovered"]:
            console.print(f"  [yellow]WRONG[/yellow] found: {result['selected_repo']}")
            if result["expected_in_top5"]:
                console.print(f"  [yellow]But expected repo was in top 5[/yellow]")
        else:
            console.print(f"  [red]NOT FOUND[/red]")
        console.print()

    # Compute summary
    total = len(results)
    crashed = sum(1 for r in results if r["pipeline_crashed"])
    paper_ok = sum(1 for r in results if r["paper_parse_success"])
    repo_discovered = sum(1 for r in results if r["repo_discovered"])
    repo_correct = sum(1 for r in results if r["repo_correct"])
    expected_in_top5 = sum(1 for r in results if r.get("expected_in_top5", False))

    summary = {
        "total": total,
        "paper_parse_success": paper_ok,
        "repo_discovered": repo_discovered,
        "repo_correct": repo_correct,
        "expected_in_top5": expected_in_top5,
        "pipeline_crashed": crashed,
        "backend": backend,
    }

    # Print summary table
    console.rule("[bold]Summary[/bold]")

    table = Table(title="Gold Set Evaluation Results")
    table.add_column("Item", style="cyan")
    table.add_column("Difficulty", style="dim")
    table.add_column("Hint", style="dim")
    table.add_column("Paper Parsed", justify="center")
    table.add_column("Repo Correct", justify="center")
    table.add_column("In Top 5", justify="center")
    table.add_column("Selected Repo", max_width=50)

    for r in results:
        parsed = "[green]OK[/green]" if r["paper_parse_success"] else "[red]FAIL[/red]"
        correct = "[green]OK[/green]" if r["repo_correct"] else ("[yellow]WRONG[/yellow]" if r["repo_discovered"] else "[red]MISS[/red]")
        top5 = "[green]Y[/green]" if r.get("expected_in_top5") else "-"
        repo_display = (r["selected_repo"] or "-")
        if len(repo_display) > 48:
            repo_display = "..." + repo_display[-45:]
        table.add_row(r["name"], r["difficulty"], r["repo_hint_type"], parsed, correct, top5, repo_display)

    console.print(table)

    console.print()
    console.print(f"[bold]Total:[/bold] {total}")
    console.print(f"  Paper parse success: {paper_ok}/{total}")
    console.print(f"  Correct repo (top 1): {repo_correct}/{total}")
    console.print(f"  Expected in top 5: {expected_in_top5}/{total}")
    console.print(f"  Pipeline crashed: {crashed}/{total}")

    # Save results
    output_dir = Path(workspace)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "summary.json", summary)

    # Per-item results
    for r in results:
        save_json(output_dir / "runs" / r["name"] / "result.json", r)

    console.print(f"\nResults saved to: {output_dir}")


@app.command("inspect-task")
def inspect_task(
    task_dir: str = typer.Option(..., "--task-dir"),
):
    state = load_state(task_dir)
    console.print_json(state.model_dump_json(indent=2))


@app.command("print-report")
def print_report(
    task_dir: str = typer.Option(..., "--task-dir"),
):
    report_path = Path(task_dir) / "report" / "reproduction_smoke_report.md"
    if not report_path.exists():
        console.print("[red]Report not found[/red]")
        raise typer.Exit(1)

    console.print(report_path.read_text(encoding="utf-8"))


@app.command("tui")
def tui(
    workspace: str = typer.Option(settings.default_workspace, "--workspace"),
    backend: str = typer.Option(settings.default_backend, "--backend", help="Execution backend: none, local, venv, conda, or docker"),
    timeout_minutes: int = typer.Option(settings.default_timeout_minutes, "--timeout-minutes"),
    max_repair_attempts: int = typer.Option(settings.default_max_repair_attempts, "--max-repair-attempts"),
    no_splash: bool = typer.Option(False, "--no-splash", help="Skip the splash / preflight screen"),
):
    """Launch the OpenCode-inspired terminal UI."""
    sanitize_proxy_env()
    if backend not in ("none", "local", "venv", "conda", "docker"):
        console.print(f"[red]Invalid backend: {backend}. Must be none, local, venv, conda, or docker.[/red]")
        raise typer.Exit(1)

    import os
    skip = no_splash or bool(os.environ.get("PRA_SKIP_SPLASH"))

    from app.tui import run_tui
    from app.core.paths import resolve_workspace_path

    ws = str(resolve_workspace_path(workspace, explicit=False))
    run_tui(
        _run_pipeline,
        workspace=ws,
        backend=backend,
        timeout_minutes=timeout_minutes,
        max_repair_attempts=max_repair_attempts,
        skip_splash=skip,
    )


if __name__ == "__main__":
    app()
