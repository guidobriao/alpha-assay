from __future__ import annotations

"""Artifact / output path panel — fully in Chinese."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from .. import theme as T


class ArtifactPanel(Widget):
    """Panel showing current task outputs and file paths."""

    DEFAULT_CSS = """
    ArtifactPanel {
        width: 34;
        height: 1fr;
        background: #000000;
        border-left: solid $primary-darken-2;
        padding: 0 1;
    }
    ArtifactPanel #artifact-title {
        height: 1;
        color: $text;
        padding: 1 0 0 0;
    }
    ArtifactPanel #artifact-body {
        height: 1fr;
        padding: 1 0 0 0;
    }
    """

    _ARTIFACT_KEYS = [
        ("task_dir", "任务目录"),
        ("report_md", "报告 Markdown"),
        ("report_json", "报告 JSON"),
        ("state_json", "状态 JSON"),
        ("env_log", "环境构建日志"),
        ("smoke_log", "冒烟测试输出"),
        ("benchmark_log", "Benchmark 输出"),
        ("reproduction_log", "轻量复现输出"),
    ]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._paths: dict[str, str] = {}

    def update_artifacts(
        self,
        task_dir: str | None = None,
        report_md: str | None = None,
        report_json: str | None = None,
        state_json: str | None = None,
        env_log: str | None = None,
        smoke_log: str | None = None,
        benchmark_log: str | None = None,
        reproduction_log: str | None = None,
    ) -> None:
        updates: dict[str, str | None] = {
            "task_dir": task_dir,
            "report_md": report_md,
            "report_json": report_json,
            "state_json": state_json,
            "env_log": env_log,
            "smoke_log": smoke_log,
            "benchmark_log": benchmark_log,
            "reproduction_log": reproduction_log,
        }
        for k, v in updates.items():
            if v is not None:
                self._paths[k] = v
        self._refresh()

    def _refresh(self) -> None:
        if not self._paths:
            content = f"[{T.FG_DIM}]暂无任务产物。请先运行流水线。[/]"
        else:
            lines: list[str] = []
            for key, label in self._ARTIFACT_KEYS:
                path = self._paths.get(key, "")
                if path:
                    exists = "✓" if Path(path).exists() else "✗"
                    ec = T.GREEN if exists == "✓" else T.RED
                    lines.append(
                        f"[{ec}]{exists}[/] [{T.FG_DIM}]{label}:[/] [{T.FG}]{_trunc(path, 42)}[/]"
                    )
                else:
                    lines.append(f"[{T.FG_DIM}]– {label}[/]")
            content = "\n".join(lines)
        try:
            self.query_one("#artifact-body", Static).update(content)
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("[bold]任务产物[/]", id="artifact-title")
            yield Static("", id="artifact-body")

    def on_mount(self) -> None:
        self._refresh()


def _trunc(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len // 2 - 1] + "…" + text[-(max_len // 2 - 1) :]
