from __future__ import annotations

"""Pipeline stage status panel — fully in Chinese."""

from dataclasses import dataclass, field
from typing import Literal

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from .. import theme as T


PIPELINE_STAGES: list[str] = [
    "Ingest paper",
    "Understand paper",
    "Search GitHub",
    "Evaluate repo",
    "Build conda env",
    "Build virtualenv",
    "Build Docker image",
    "Run smoke command",
    "Run benchmark reproduction",
    "Run simple reproduction",
    "Write report",
]

STAGE_LABELS_CN: dict[str, str] = {
    "Ingest paper": "解析论文",
    "Understand paper": "理解论文",
    "Search GitHub": "搜索 GitHub",
    "Evaluate repo": "评估仓库",
    "Build conda env": "构建 conda 环境",
    "Build virtualenv": "构建虚拟环境",
    "Build Docker image": "构建 Docker 镜像",
    "Run smoke command": "运行冒烟测试",
    "Run benchmark reproduction": "运行 benchmark 复现",
    "Run simple reproduction": "运行轻量复现",
    "Write report": "生成报告",
}

STATUS_LABELS_CN: dict[str, str] = {
    "queued": "等待中",
    "running": "运行中",
    "success": "成功",
    "failed": "失败",
    "skipped": "已跳过",
    "disabled": "未启用",
    "cancelled": "已取消",
}

_BUILD_STAGES = {"Build conda env", "Build virtualenv", "Build Docker image"}


@dataclass
class StageView:
    """Lightweight view-model for a pipeline stage."""

    name: str
    status: Literal["queued", "running", "success", "failed", "skipped", "disabled"] = (
        "queued"
    )
    message: str = ""
    detail: str = ""
    duration: float | None = None
    attempts: int = 0

    @property
    def label_cn(self) -> str:
        return STAGE_LABELS_CN.get(self.name, self.name)

    @property
    def status_cn(self) -> str:
        return STATUS_LABELS_CN.get(self.status, self.status)

    @property
    def icon(self) -> str:
        return T.stage_icon(self.status)

    @property
    def color(self) -> str:
        return T.stage_color(self.status)


class PipelinePanel(Widget):
    """Right sidebar: live pipeline stage list."""

    DEFAULT_CSS = """
    PipelinePanel {
        width: 34;
        height: 1fr;
        background: #000000;
        border-left: solid $primary-darken-2;
        padding: 0 1;
    }
    PipelinePanel #pipeline-title {
        height: 1;
        color: $text;
        padding: 1 0 0 0;
    }
    PipelinePanel #pipeline-body {
        height: 1fr;
        padding: 1 0 0 0;
    }
    """

    def __init__(self, backend: str = "conda", **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._backend = backend
        self._stages: dict[str, StageView] = {}
        self.reset(backend)

    def reset(self, backend: str = "") -> None:
        if backend:
            self._backend = backend
        self._stages.clear()
        for stage in PIPELINE_STAGES:
            if self._is_stage_active(stage):
                self._stages[stage] = StageView(name=stage, status="queued")
            else:
                self._stages[stage] = StageView(name=stage, status="disabled")
        self._refresh()

    def _is_stage_active(self, stage: str) -> bool:
        if stage in _BUILD_STAGES:
            backend_map = {
                "Build conda env": "conda",
                "Build virtualenv": "venv",
                "Build Docker image": "docker",
            }
            return backend_map.get(stage) == self._backend
        if self._backend == "none" and stage in (
            "Run smoke command",
            "Run benchmark reproduction",
            "Run simple reproduction",
        ):
            return False
        return True

    def update_stage(self, stage: StageView) -> None:
        if stage.name in self._stages:
            existing = self._stages[stage.name]
            if (
                existing.status in ("failed", "success", "skipped")
                and stage.status == "running"
            ):
                stage.attempts = existing.attempts + 1
        self._stages[stage.name] = stage
        self._refresh()

    def update_from_name(
        self,
        name: str,
        status: str = "running",
        message: str = "",
        detail: str = "",
        duration: float | None = None,
    ) -> None:
        self.update_stage(
            StageView(
                name=name,
                status=status,  # type: ignore[arg-type]
                message=message,
                detail=detail,
                duration=duration,
            )
        )

    def _refresh(self) -> None:
        lines: list[str] = []
        for stage_name in PIPELINE_STAGES:
            sv = self._stages.get(stage_name)
            if sv is None:
                continue
            icon = sv.icon
            color = sv.color
            label = sv.label_cn
            dur = f" {sv.duration:.1f}s" if sv.duration is not None else ""
            msg = f" — {sv.message}" if sv.message else ""
            if sv.attempts > 1:
                msg += f" [{sv.attempts}]"
            line = f"[{color}]{icon} {label}[/][{T.FG_DIM}]{dur}{msg}[/]"
            lines.append(line)
        content = "\n".join(lines) if lines else "[dim]暂无流水线数据[/]"
        try:
            self.query_one("#pipeline-body", Static).update(content)
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("[bold]流水线[/]", id="pipeline-title")
            yield Static("", id="pipeline-body")

    def on_mount(self) -> None:
        self._refresh()
