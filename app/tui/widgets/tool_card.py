from __future__ import annotations

"""Collapsible stage card — click to expand/collapse, log buffer, running=open by default."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.events import MouseDown
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from ..display_utils import clean_display_text

# ── Labels ────────────────────────────────────────────────

_STAGE_CN: dict[str, str] = {
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

_STATUS_CN: dict[str, str] = {
    "queued": "等待",
    "running": "运行中",
    "success": "完成",
    "failed": "失败",
    "skipped": "跳过",
    "cancelled": "取消",
    "disabled": "禁用",
}

_ICON_MAP: dict[str, str] = {
    "queued": "○",
    "running": "●",
    "success": "✓",
    "failed": "✗",
    "skipped": "–",
    "cancelled": "–",
    "disabled": "–",
}


class ToolCard(Widget):
    """A single pipeline stage card — status-colored left border, dark background, click to expand."""

    can_focus = True

    DEFAULT_CSS = """
    ToolCard {
        margin: 0 0 1 0;
        padding: 0 1;
        height: auto;
        background: #000000;
        border-left: thick #585b70;
        border-bottom: solid #000000;
    }
    ToolCard.running {
        border-left: thick #8be9fd;
    }
    ToolCard.success {
        border-left: thick #50fa7b;
    }
    ToolCard.failed {
        border-left: thick #ff5555;
    }
    ToolCard.skipped,
    ToolCard.cancelled,
    ToolCard.disabled {
        border-left: thick #ffb86c;
    }
    ToolCard .tool-header {
        color: $text;
        height: 1;
        padding: 0 0;
    }
    ToolCard .tool-body {
        color: $text-muted;
        margin-left: 2;
        height: auto;
        padding: 0 0 1 0;
    }
    """

    BINDINGS = [
        Binding("enter", "toggle_collapsed", "展开/折叠", show=False),
        Binding("space", "toggle_collapsed", "展开/折叠", show=False),
    ]

    class StatusChanged(Message):
        def __init__(self, status: str) -> None:
            self.status = status
            super().__init__()

    def __init__(
        self,
        name: str,
        status: str = "running",
        message: str = "",
        detail: str | None = None,
        duration: float | None = None,
        attempts: int = 1,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = name
        self._status = status
        self._message = clean_display_text(message)
        self._detail = clean_display_text(detail or "")
        self._duration = duration
        self._attempts = attempts

        self._collapsed = status != "running"
        self._user_toggled = False
        self._log_lines: list[str] = []
        self._keyed_log_indices: dict[str, int] = {}
        self._max_expanded = 5

    # ── Properties ──────────────────────────────────────────

    @property
    def status(self) -> str:
        return self._status

    @property
    def stage_label(self) -> str:
        return _STAGE_CN.get(self.tool_name, self.tool_name)

    @property
    def status_label(self) -> str:
        return _STATUS_CN.get(self._status, self._status)

    @property
    def icon(self) -> str:
        return _ICON_MAP.get(self._status, "●")

    def latest_log(self) -> str:
        return self._log_lines[-1] if self._log_lines else self._message or ""

    def get_visible_logs(self) -> list[str]:
        return self._log_lines[-self._max_expanded :]

    # ── Log buffer ──────────────────────────────────────────

    def append_log(self, line: str) -> None:
        clean = clean_display_text(line)
        if not clean:
            return

        # Heartbeat: replace instead of stacking
        if clean.startswith("仍在运行："):
            self.upsert_log("heartbeat", clean)
            return

        if self._log_lines and clean == self._log_lines[-1]:
            return
        self._log_lines.append(clean)
        if len(self._log_lines) > 200:
            self._log_lines = self._log_lines[-200:]
            self._keyed_log_indices.clear()

    def upsert_log(self, key: str, line: str) -> None:
        """Replace or insert a log line by key. Same key → same slot."""
        clean = clean_display_text(line)
        if not clean:
            return

        if key in self._keyed_log_indices:
            idx = self._keyed_log_indices[key]
            if 0 <= idx < len(self._log_lines):
                self._log_lines[idx] = clean
            else:
                self._keyed_log_indices.pop(key, None)
                self._log_lines.append(clean)
                self._keyed_log_indices[key] = len(self._log_lines) - 1
        else:
            self._log_lines.append(clean)
            self._keyed_log_indices[key] = len(self._log_lines) - 1

        if len(self._log_lines) > 200:
            drop = len(self._log_lines) - 200
            self._log_lines = self._log_lines[-200:]
            new_map: dict[str, int] = {}
            for k, old_idx in self._keyed_log_indices.items():
                new_idx = old_idx - drop
                if 0 <= new_idx < len(self._log_lines):
                    new_map[k] = new_idx
            self._keyed_log_indices = new_map

    # ── Collapse / expand ──────────────────────────────────

    def toggle_collapsed(self) -> None:
        self._collapsed = not self._collapsed
        self._user_toggled = True
        self._refresh_display()

    def action_toggle_collapsed(self) -> None:
        self.toggle_collapsed()

    def on_mouse_down(self, event: MouseDown) -> None:
        event.stop()
        self.toggle_collapsed()

    # ── Update ──────────────────────────────────────────────

    def update(
        self,
        status: str | None = None,
        message: str | None = None,
        detail: str | None = None,
        duration: float | None = None,
        attempts: int | None = None,
        append_log: str | None = None,
        replace_log_key: str | None = None,
    ) -> None:
        old_status = self._status

        if status is not None:
            self._status = status
        if message is not None:
            self._message = clean_display_text(message)
        if detail is not None:
            self._detail = clean_display_text(detail)
            if detail.strip():
                for line in detail.splitlines():
                    self.append_log(line)
        if duration is not None:
            self._duration = duration
        if attempts is not None:
            self._attempts = attempts
        if append_log is not None:
            if replace_log_key:
                self.upsert_log(replace_log_key, append_log)
            else:
                self.append_log(append_log)

        if not self._user_toggled and status is not None and status != old_status:
            if status == "running":
                self._collapsed = False
            elif status in ("success", "failed", "skipped", "cancelled", "disabled"):
                self._collapsed = True

        self._sync_status_class()
        self._refresh_display()

    def _sync_status_class(self) -> None:
        for cls in (
            "queued",
            "running",
            "success",
            "failed",
            "skipped",
            "cancelled",
            "disabled",
        ):
            self.remove_class(cls)
        self.add_class(
            self._status
            if self._status
            in {
                "queued",
                "running",
                "success",
                "failed",
                "skipped",
                "cancelled",
                "disabled",
            }
            else "running"
        )

    # ── Display ─────────────────────────────────────────────

    def _format_duration(self) -> str:
        if self._duration is None:
            return ""
        d = self._duration
        if d >= 60:
            return f"{int(d // 60)}m{d % 60:.0f}s"
        return f"{d:.1f}s"

    _MAX_BODY_LINE = 180

    def _refresh_display(self) -> None:
        dur = self._format_duration()
        dur_str = f" {dur}" if dur else ""
        latest = self.latest_log()
        if len(latest) > 80:
            latest = latest[:77] + "..."
        latest_str = f" · {latest}" if latest else ""
        collapse_icon = "▾" if not self._collapsed else "▸"

        header_text = (
            f"[bold]{collapse_icon} {self.icon} {self.stage_label}[/] "
            f"[dim]{self.status_label}{dur_str}{latest_str}[/]"
        )
        try:
            self.query_one(".tool-header", Static).update(header_text)
        except Exception:
            pass

        try:
            body = self.query_one(".tool-body", Static)
            if self._collapsed:
                body.display = False
            else:
                visible = self.get_visible_logs()
                if visible:
                    body.update(
                        "\n".join(
                            (
                                f"  {l}"
                                if len(l) <= self._MAX_BODY_LINE
                                else f"  {l[:self._MAX_BODY_LINE - 3]}..."
                            )
                            for l in visible
                        )
                    )
                    body.display = True
                else:
                    body.display = False
        except Exception:
            pass

    # ── Compose ─────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Static("", classes="tool-header")
        body = Static("", classes="tool-body")
        body.display = False
        yield body

    def on_mount(self) -> None:
        self._sync_status_class()
        self._refresh_display()
