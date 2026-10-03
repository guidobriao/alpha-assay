from __future__ import annotations

"""Panel showing session configuration info — fully in Chinese."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from .. import theme as T


def _mode_cn(mode: str) -> str:
    value = (mode or "").lower()
    if value == "act":
        return "ACT"
    if value == "plan":
        return "PLAN"
    return mode.upper() if mode else "-"


def _status_cn(status: str) -> str:
    mapping = {
        "draft": "draft",
        "running": "running",
        "success": "success",
        "failed": "failed",
        "cancelled": "cancelled",
        "input_failed": "input failed",
        "completed": "completed",
    }
    return mapping.get((status or "").lower(), status or "-")


def _truncate_path(p: str, max_len: int = 36) -> str:
    if len(p) <= max_len:
        return p
    return p[: max_len // 2 - 2] + "..." + p[-(max_len // 2 - 2) :]


class SessionPanel(Widget):
    """Left sidebar: key-value display of session configuration."""

    DEFAULT_CSS = """
    SessionPanel {
        width: 28;
        height: 1fr;
        background: #000000;
        border-right: solid $primary-darken-2;
        padding: 0 1;
    }
    SessionPanel #session-title {
        height: 1;
        color: $text;
        padding: 1 0 0 0;
    }
    SessionPanel #session-body {
        height: 1fr;
        padding: 1 0 0 0;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._data: dict[str, str] = {}

    def update_session(
        self,
        session_id: str = "",
        mode: str = "",
        backend: str = "",
        workspace: str = "",
        paper: str = "",
        repo: str = "",
        repo_dir: str = "",
        timeout: str = "",
        repairs: str = "",
        task_dir: str = "",
        report_path: str = "",
        status: str = "",
        cancel_requested: bool = False,
    ) -> None:
        self._data = {
            "Session ID": session_id[:8] if session_id else "-",
            "Mode": _mode_cn(mode),
            "Backend": backend or "-",
            "Workspace": _truncate_path(workspace) if workspace else "-",
            "Paper": _truncate_path(paper) if paper else "-",
            "Repo URL": _truncate_path(repo) if repo else "-",
            "Local repo": _truncate_path(repo_dir) if repo_dir else "-",
            "Timeout": f"{timeout} minutes" if timeout else "-",
            "Repairs": repairs or "-",
            "Task directory": _truncate_path(task_dir) if task_dir else "-",
            "Report path": _truncate_path(report_path) if report_path else "-",
            "Status": _status_cn(status),
            "Cancel requested": "yes" if cancel_requested else "no",
        }
        self._refresh()

    def _refresh(self) -> None:
        lines: list[str] = []
        for key, val in self._data.items():
            v_color = T.status_color(val) if key in ("Status",) else T.FG
            lines.append(f"[{T.FG_DIM}]{key:<10}[/] [{v_color}]{val}[/]")
        content = "\n".join(lines)
        try:
            self.query_one("#session-body", Static).update(content)
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("[bold]Session info[/]", id="session-title")
            yield Static("", id="session-body")

    def on_mount(self) -> None:
        self._refresh()
