from __future__ import annotations

"""Slash command metadata for the TUI — fully in Chinese."""

from typing import NamedTuple


class CommandMeta(NamedTuple):
    name: str
    args: str                  # internal args string (for parsing reference)
    description: str           # Chinese description
    category: str              # Chinese category
    safe_during_run: bool = False
    display_args: str = ""     # Chinese display string for completion popup

    @property
    def has_required_args(self) -> bool:
        """True if the command takes non-optional arguments."""
        return bool(self.args) and not self.args.startswith("[")


COMMANDS: dict[str, CommandMeta] = {
    "help": CommandMeta("help", "", "Show help", "System", True),
    "clear": CommandMeta("clear", "", "Clear message timeline", "System", True),
    "status": CommandMeta("status", "", "Show current task status", "View", True),
    "plan": CommandMeta("plan", "", "Switch to plan mode (no execution)", "Mode", False),
    "act": CommandMeta("act", "", "Switch to act mode", "Mode", False),
    "input": CommandMeta("input", "<path>", "Set local paper PDF path", "Input", False, display_args="<PDF path>"),
    "arxiv": CommandMeta("arxiv", "<arXiv ID or URL>", "Download PDF from arXiv and set as current paper", "Input", False, display_args="<arXiv ID or URL>"),
    "download-arxiv": CommandMeta("download-arxiv", "<arXiv ID or URL>", "Download PDF from arXiv and set as current paper", "Input", False, display_args="<arXiv ID or URL>"),
    "repo": CommandMeta("repo", "<url>", "Set the GitHub repository URL directly", "Input", False, display_args="<repo URL>"),
    "repo-dir": CommandMeta("repo-dir", "<path>", "Use a local repository directory", "Input", False, display_args="<local repo dir>"),
    "backend": CommandMeta("backend", "[none|local|venv|conda|docker]", "Set the execution backend", "running", False, display_args="<backend>"),
    "workspace": CommandMeta("workspace", "<path>", "Set the output workspace directory", "running", False, display_args="<dir>"),
    "timeout": CommandMeta("timeout", "<minutes>", "Set the per-step timeout (minutes)", "running", False, display_args="<minutes>"),
    "repairs": CommandMeta("repairs", "<n>", "Set max dependency repair attempts", "running", False, display_args="<n>"),
    "run": CommandMeta("run", "", "Run the replication pipeline", "running", False),
    "report": CommandMeta("report", "", "Show the replication report", "View", True),
    "logs": CommandMeta("logs", "<type>", "Show logs (env/smoke/benchmark etc.)", "View", True, display_args="<log type>"),
    "cancel": CommandMeta("cancel", "", "Cancel the current task", "running", True),
    "kill": CommandMeta("kill", "", "Force-kill the current task and child processes", "running", True),
    "force-cancel": CommandMeta("force-cancel", "", "Force-kill the current task and child processes", "running", True),
    "abort": CommandMeta("abort", "", "Force-kill the current task and child processes", "running", True),
    "sessions": CommandMeta("sessions", "", "List past sessions", "Sessions", True),
    "session": CommandMeta("session", "", "List past sessions", "Sessions", True),
    "resume": CommandMeta("resume", "<session-id>", "Resume a past session", "Sessions", False, display_args="<session-id>"),
    "quit": CommandMeta("quit", "", "Quit the TUI", "System", True),
    "exit": CommandMeta("exit", "", "Quit the TUI", "System", True),
    "panel": CommandMeta("panel", "[session|pipeline|help|artifacts|none]", "Switch the side panel", "Interface", True, display_args="<panel>"),
    "artifact": CommandMeta("artifact", "", "Show current task artifact paths", "View", True),
    "open-report": CommandMeta("open-report", "", "Show report file path", "View", True),
    "mode": CommandMeta("mode", "", "Show current PLAN/ACT mode", "Mode", True),
    "reset": CommandMeta("reset", "", "Clear current session input (keeps files)", "Sessions", False),
    "conda-envs": CommandMeta("conda-envs", "[list|delete|prune]", "Inspect or delete project conda environments", "Environment", True, "[list|delete|prune]"),
    "envs": CommandMeta("envs", "[list|delete|prune]", "Inspect or delete project conda environments", "Environment", True, "[list|delete|prune]"),
}

RUNNING_SAFE_COMMANDS: set[str] = {n for n, m in COMMANDS.items() if m.safe_during_run}
