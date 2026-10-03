from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.benchmark.schema import BenchmarkSpec, ExecutionBudget


@dataclass(frozen=True)
class AdapterContext:
    workspace_dir: Path
    repo_dir: Path
    readme_text: str
    task: str | None
    datasets: list[str]
    metrics: list[str]
    method_keywords: list[str]
    scripts: list[str]
    budget: ExecutionBudget
    paper_slug: str


class BenchmarkAdapter(Protocol):
    task_family: str

    def propose_benchmarks(self, context: AdapterContext) -> list[BenchmarkSpec]:
        ...


def has_script(context: AdapterContext, name: str) -> bool:
    return name in context.scripts and (context.repo_dir / name).exists()


def first_existing_script(context: AdapterContext, names: list[str]) -> str | None:
    for name in names:
        if has_script(context, name):
            return name
    return None


def find_sample_files(context: AdapterContext, suffixes: set[str], limit: int = 20) -> list[str]:
    roots = [
        context.repo_dir / name
        for name in ("assets", "asset", "examples", "example", "demo", "demos", "data", "samples", "test", "tests")
    ]
    roots.append(context.repo_dir)
    found: list[str] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if len(found) >= limit:
                return found
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if any(part.startswith(".") or part in {"__pycache__", ".git"} for part in path.relative_to(context.repo_dir).parts):
                continue
            try:
                if path.stat().st_size > 30 * 1024 * 1024:
                    continue
            except OSError:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path.relative_to(context.repo_dir).as_posix())
    return found
