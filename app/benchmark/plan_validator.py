"""Validate LLM-generated benchmark plans against real filesystem state and safety rules.

Every BenchmarkSpec produced by the generic planner must pass through this
validator before it reaches the execution layer.  Checks include file
existence, command safety, budget constraints, structural validity, and
generated-script inspection.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.benchmark.schema import BenchmarkSpec, ExecutionBudget

_LEVELS = {"L0", "L1", "L2", "L3"}


class BenchmarkPlanValidator:
    """Validate a list of LLM-generated BenchmarkSpec objects."""

    def __init__(
        self,
        repo_dir: Path,
        candidate_scripts: list[str],
        budget: ExecutionBudget | None = None,
    ) -> None:
        self._repo_dir = repo_dir.resolve()
        self._allowed_scripts = set(candidate_scripts)
        self._budget = budget or ExecutionBudget()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, specs: list[BenchmarkSpec]) -> list[BenchmarkSpec]:
        """Filter and fix specs.  Returns only valid specs with updated feasibility."""
        validated: list[BenchmarkSpec] = []
        seen_ids: set[str] = set()
        for spec in specs:
            updated = self.validate_single(spec)
            if updated.id in seen_ids:
                updated.feasibility["runnable"] = False
                updated.feasibility["reason"] = f"duplicate id: {updated.id}"
            seen_ids.add(updated.id)
            validated.append(updated)
        return validated

    def validate_single(self, spec: BenchmarkSpec) -> BenchmarkSpec:
        """Validate one spec.  Returns the spec with updated feasibility notes."""
        issues: list[str] = []

        # Structural checks
        if not spec.id:
            issues.append("missing id")
        if spec.level not in _LEVELS:
            issues.append(f"invalid level: {spec.level!r}")

        # Command checks (only for specs that intend to be runnable)
        if spec.command:
            cmd_issues = self._check_command(spec)
            issues.extend(cmd_issues)
        elif spec.feasibility.get("runnable", True):
            issues.append("runnable spec has empty command")

        # Dataset budget
        budget_issue = self._check_budget(spec)
        if budget_issue:
            issues.append(budget_issue)

        # Generated script check
        if spec.generated_script_body:
            script_issues = self._check_generated_script(spec.generated_script_body)
            issues.extend(script_issues)

        # Apply results
        if issues:
            spec.feasibility["runnable"] = False
            spec.feasibility["validation_issues"] = issues
            if not spec.fallback_reason:
                spec.fallback_reason = "; ".join(issues[:3])

        return spec

    # ------------------------------------------------------------------
    # Command validation
    # ------------------------------------------------------------------

    def _check_command(self, spec: BenchmarkSpec) -> list[str]:
        issues: list[str] = []
        argv = spec.command
        if not argv:
            return issues

        # Shell metacharacters
        dangerous_chars = {";", "&", "|", "$", "`", ">", "<"}
        for part in argv:
            if any(ch in part for ch in dangerous_chars):
                issues.append(f"shell metacharacter in command: {part!r}")
                return issues

        # Executable check
        if argv[0] != "python":
            issues.append(f"executable not allowed: {argv[0]!r}")
            return issues

        if len(argv) < 2:
            issues.append("missing script argument")
            return issues

        # Module mode
        if len(argv) >= 3 and argv[1] == "-m":
            module = argv[2]
            if not self._is_safe_module(module):
                issues.append(f"unsafe module: {module!r}")
            for arg in argv[3:]:
                if not self._is_safe_arg(arg):
                    issues.append(f"unsafe argument: {arg!r}")
            return issues

        # Script mode
        script = argv[1]
        script_path = self._repo_dir / script
        if Path(script).is_absolute() or ".." in Path(script).parts:
            issues.append(f"unsafe script path: {script!r}")
        elif not script.endswith(".py"):
            issues.append(f"script must end with .py: {script!r}")
        elif not script_path.exists():
            allowed = self._allowed_scripts | {spec.generated_script_name} - {None}
            if script not in allowed:
                issues.append(f"script does not exist: {script!r}")

        # Training check
        lowered = [part.lower() for part in argv]
        if any("train" in part for part in lowered):
            issues.append("training command blocked")

        for arg in argv[2:]:
            if not self._is_safe_arg(arg):
                issues.append(f"unsafe argument: {arg!r}")

        return issues

    @staticmethod
    def _is_safe_module(module: str) -> bool:
        return bool(re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+", module,
        )) and not any(
            part in {"os", "sys", "subprocess", "shutil"}
            for part in module.split(".")
        )

    @staticmethod
    def _is_safe_arg(arg: str) -> bool:
        if not arg or len(arg) > 260:
            return False
        if any(ch in arg for ch in {";", "&", "|", "$", "`", ">", "<"}):
            return False
        if arg.startswith("-"):
            return bool(re.fullmatch(r"-{1,2}[A-Za-z0-9][A-Za-z0-9_\-]*", arg))
        return bool(re.fullmatch(r"[A-Za-z0-9_./:=,+@%\\-]+", arg))

    # ------------------------------------------------------------------
    # Budget validation
    # ------------------------------------------------------------------

    def _check_budget(self, spec: BenchmarkSpec) -> str | None:
        if self._budget.allow_large_downloads:
            return None
        size_gb = spec.dataset.size_gb
        if size_gb is not None and size_gb > self._budget.max_dataset_size_gb:
            return (
                f"dataset size {size_gb:.2f}GB exceeds budget "
                f"{self._budget.max_dataset_size_gb:.2f}GB"
            )
        if (
            spec.level == "L3"
            and spec.dataset.source not in {"bundled", "readme", "synthetic"}
            and size_gb is None
        ):
            return (
                f"L3 dataset size unknown, held under "
                f"{self._budget.max_dataset_size_gb:.2f}GB budget"
            )
        return None

    # ------------------------------------------------------------------
    # Generated script validation
    # ------------------------------------------------------------------

    _DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"os\.system\s*\("), "os.system call"),
        (re.compile(r"subprocess\.\w+call\s*\([^)]*shell\s*=\s*True"), "subprocess with shell=True"),
        (re.compile(r"subprocess\.\w+run\s*\([^)]*shell\s*=\s*True"), "subprocess with shell=True"),
        (re.compile(r"__import__\s*\("), "__import__ usage"),
        (re.compile(r"exec\s*\(\s*['\"]"), "exec with string literal"),
        (re.compile(r"eval\s*\(\s*['\"]"), "eval with string literal"),
        (re.compile(r"open\s*\([^)]*['\"]w"), "file write in generated script"),
    ]

    def _check_generated_script(self, body: str) -> list[str]:
        issues: list[str] = []
        for pattern, label in self._DANGEROUS_PATTERNS:
            if pattern.search(body):
                issues.append(f"generated script contains {label}")
        return issues
