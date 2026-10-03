from __future__ import annotations

"""Help panel — fully in Chinese, no Rich markup."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

_COMMAND_HELP: dict[str, list[tuple[str, str]]] = {
    "Input": [
        ("/input <PDF path>", "Set local paper PDF path"),
        ("/repo <repo URL>", "Set the GitHub repository URL directly"),
        ("/repo-dir <local repo dir>", "Use a local repository directory"),
    ],
    "running": [
        ("/run", "Run the replication pipeline"),
        ("/cancel", "Cancel the current task"),
        ("/backend [backend]", "Set backend: none | local | venv | conda | docker"),
        ("/workspace <dir>", "Set the output workspace directory"),
        ("/timeout <minutes>", "Set per-step timeout"),
        ("/repairs <n>", "Set max dependency repair attempts"),
    ],
    "View": [
        ("/status", "Show current task status"),
        ("/report", "Show replication report path and summary"),
        ("/logs <log type>", "Viewlog: env | smoke | benchmark | reproduction"),
        ("/artifact", "Show current task artifact paths"),
        ("/open-report", "Show report file path"),
    ],
    "Mode": [
        ("/plan", "Switch to PLAN mode (no execution)"),
        ("/act", "Switch to act mode"),
        ("/panel <panel>", "Switch the side panel: session | pipeline | help | artifacts"),
        ("/mode", "Show current PLAN / ACT mode"),
    ],
    "Sessions": [
        ("/sessions", "List past sessions"),
        ("/resume <session-id>", "Resume a past session"),
        ("/reset", "Clear current session input (keeps files)"),
        ("/clear", "Clear message timeline"),
    ],
    "System": [
        ("/help", "Show help"),
        ("!shell <cmd>", "Run shell command (requires confirmation)"),
        ("/quit or /exit", "Quit the TUI"),
        ("Ctrl+P", "Toggle PLAN / ACT mode"),
        ("Ctrl+L", "Clear message timeline"),
        ("Ctrl+C", "Force quit"),
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
            yield Static("[bold]Command help[/]", id="help-title")
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
