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
        return "执行"
    if value == "plan":
        return "计划"
    return mode.upper() if mode else "-"


def _status_cn(status: str) -> str:
    mapping = {
        "draft": "草稿",
        "running": "运行中",
        "success": "成功",
        "failed": "失败",
        "cancelled": "已取消",
        "input_failed": "输入失败",
        "completed": "已完成",
    }
    return mapping.get((status or "").lower(), status or "-")


def _truncate_path(p: str, max_len: int = 36) -> str:
    if len(p) <= max_len:
        return p
    return p[: max_len // 2 - 2] + "…" + p[-(max_len // 2 - 2) :]


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
            "会话 ID": session_id[:8] if session_id else "-",
            "模式": _mode_cn(mode),
            "执行后端": backend or "-",
            "工作目录": _truncate_path(workspace) if workspace else "-",
            "论文": _truncate_path(paper) if paper else "-",
            "仓库 URL": _truncate_path(repo) if repo else "-",
            "本地仓库": _truncate_path(repo_dir) if repo_dir else "-",
            "超时": f"{timeout} 分钟" if timeout else "-",
            "修复次数": repairs or "-",
            "任务目录": _truncate_path(task_dir) if task_dir else "-",
            "报告路径": _truncate_path(report_path) if report_path else "-",
            "状态": _status_cn(status),
            "取消请求": "是" if cancel_requested else "否",
        }
        self._refresh()

    def _refresh(self) -> None:
        lines: list[str] = []
        for key, val in self._data.items():
            v_color = T.status_color(val) if key in ("状态",) else T.FG
            lines.append(f"[{T.FG_DIM}]{key:<10}[/] [{v_color}]{val}[/]")
        content = "\n".join(lines)
        try:
            self.query_one("#session-body", Static).update(content)
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("[bold]会话信息[/]", id="session-title")
            yield Static("", id="session-body")

    def on_mount(self) -> None:
        self._refresh()
