from __future__ import annotations

"""Help panel — fully in Chinese, no Rich markup."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

_COMMAND_HELP: dict[str, list[tuple[str, str]]] = {
    "输入": [
        ("/input <PDF路径>", "设置本地论文 PDF 路径"),
        ("/repo <仓库URL>", "直接指定 GitHub 仓库地址"),
        ("/repo-dir <本地仓库目录>", "指定本地代码仓库目录"),
    ],
    "运行": [
        ("/run", "执行复现流水线"),
        ("/cancel", "取消当前任务"),
        ("/backend [后端]", "设置后端：none | local | venv | conda | docker"),
        ("/workspace <目录>", "设置输出工作目录"),
        ("/timeout <分钟数>", "设置步骤超时时间"),
        ("/repairs <次数>", "设置最大依赖修复次数"),
    ],
    "查看": [
        ("/status", "查看当前任务状态"),
        ("/report", "查看复现报告路径和摘要"),
        ("/logs <日志类型>", "查看日志：env | smoke | benchmark | reproduction"),
        ("/artifact", "显示当前任务产物路径"),
        ("/open-report", "显示报告文件路径"),
    ],
    "模式": [
        ("/plan", "切换到计划模式（不执行）"),
        ("/act", "切换到执行模式"),
        ("/panel <面板>", "切换右侧面板：session | pipeline | help | artifacts"),
        ("/mode", "显示当前 PLAN / ACT 模式"),
    ],
    "会话": [
        ("/sessions", "列出历史会话"),
        ("/resume <会话ID>", "恢复历史会话"),
        ("/reset", "清空当前会话输入，不删除磁盘文件"),
        ("/clear", "清空消息时间线"),
    ],
    "系统": [
        ("/help", "显示帮助"),
        ("!shell <命令>", "执行 shell 命令（需确认）"),
        ("/quit 或 /exit", "退出 TUI"),
        ("Ctrl+P", "切换 PLAN / ACT 模式"),
        ("Ctrl+L", "清空消息时间线"),
        ("Ctrl+C", "强制退出"),
    ],
}


class HelpPanel(Widget):
    """Scrollable slash-command reference."""

    DEFAULT_CSS = """
    HelpPanel {
        width: 34;
        height: 1fr;
        background: #000000;
        border-left: solid $primary-darken-2;
        padding: 0 1;
    }
    HelpPanel #help-title {
        height: 1;
        color: $text;
        padding: 1 0 0 0;
    }
    HelpPanel #help-body {
        height: 1fr;
        padding: 1 0 0 0;
    }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("[bold]命令帮助[/]", id="help-title")
            yield Static("", id="help-body")

    def on_mount(self) -> None:
        self._build()

    def _build(self) -> None:
        lines: list[str] = []
        for category, cmds in _COMMAND_HELP.items():
            lines.append(f"\n{category}")
            lines.append("-" * len(category))
            for cmd, desc in cmds:
                lines.append(f"  {cmd}")
                lines.append(f"    {desc}")
        content = "\n".join(lines)
        try:
            self.query_one("#help-body", Static).update(content)
        except Exception:
            pass
