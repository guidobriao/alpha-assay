"""PaperAgentApp – Textual-based Claude Code-style TUI."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import Static

from app.agents.input_resolver_agent import InputResolverAgent
from app.core.file_utils import load_state
from app.core.progress import ProgressEvent, progress_events
from app.core.state import TaskState
from app.runtime.events import AgentEvent
from app.runtime.session import Session, SessionStore

from .commands import COMMANDS, RUNNING_SAFE_COMMANDS
from .display_utils import clean_display_text
from .panels import ArtifactPanel, HelpPanel, PipelinePanel, SessionPanel, StageView
from .widgets import Composer, HeaderLogo, MessageTimeline, StatusBar, ToolCard

PipelineRunner = Callable[..., TaskState]

SLASH_COMMANDS_SET: set[str] = set(COMMANDS.keys())

_ARXIV_DETECT_RE = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?")


def _looks_like_arxiv(value: str) -> bool:
    v = value.strip()
    return (
        "arxiv.org/abs/" in v.lower()
        or "arxiv.org/pdf/" in v.lower()
        or v.lower().startswith("arxiv:")
        or bool(_ARXIV_DETECT_RE.fullmatch(v))
    )


def _cleanup_killed_task_files(task_dir: str | None) -> None:
    if not task_dir:
        return
    import shutil

    td = Path(task_dir)
    for pattern in (
        "repos/cloned_repo",
        "repos/*.zip",
        "repos/*.part",
        "downloads/*.part",
    ):
        for path in td.glob(pattern):
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
            except Exception:
                pass


# Help categories: grouped command names in display order
_HELP_CATEGORIES: dict[str, list[str]] = {
    "Input": ["input", "arxiv", "download-arxiv", "repo", "repo-dir"],
    "running": ["backend", "workspace", "timeout", "repairs", "run", "cancel"],
    "View": ["status", "logs", "report", "artifact", "open-report"],
    "Mode": ["plan", "act", "mode"],
    "Interface": ["panel"],
    "Sessions": ["sessions", "resume", "reset"],
    "System": ["help", "clear", "quit", "exit"],
    "Environment": ["conda-envs", "envs"],
}


def parse_command(line: str) -> tuple[str, str]:
    stripped = line.strip()
    if not stripped:
        return "", ""
    if stripped.startswith("!"):
        return "!", stripped[1:].strip()
    if not stripped.startswith("/"):
        return "message", stripped
    parts = stripped[1:].split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    if command not in SLASH_COMMANDS_SET:
        # If it looks like a file path (contains /), treat as message
        if "/" in stripped[1:] or "." in command:
            return "message", stripped
        return "unknown_command", stripped
    return command, args


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


class PaperAgentApp(App):
    """Agent TUI for the replication pipeline."""

    CSS = """
    Screen {
        layout: vertical;
        background: #000000;
    }
    #main-content {
        height: 1fr;
    }
    #timeline-area {
        height: 1fr;
        width: 1fr;
    }
    #right-panel {
        width: 34;
        height: 1fr;
    }
    SessionPanel {
        width: 28;
        height: 1fr;
    }
    PipelinePanel {
        width: 34;
        height: 1fr;
    }
    HelpPanel {
        width: 34;
        height: 1fr;
    }
    ArtifactPanel {
        width: 34;
        height: 1fr;
    }
    #bottom-spacer-1,
    #bottom-spacer-2 {
        height: 1;
        min-height: 1;
        background: #000000;
    }
    """

    TITLE = "Alpha-Assay"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=True),
        Binding("ctrl+p", "toggle_plan_mode", "PLAN/ACT", show=True),
        Binding("tab", "accept_completion", "Completion", show=False),
        Binding("down", "completion_down", "Next completion", show=False),
        Binding("up", "completion_up", "Previous completion", show=False),
        Binding("escape", "hide_completion", "Hide completion", show=False),
    ]

    def __init__(
        self,
        runner: PipelineRunner,
        *,
        workspace: str,
        backend: str,
        timeout_minutes: int,
        max_repair_attempts: int,
        resolver: InputResolverAgent | None = None,
        skip_splash: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._skip_splash = skip_splash
        self.runner = runner
        self.resolver = resolver or InputResolverAgent()
        self.session = Session(
            backend=backend,
            workspace=workspace,
            timeout_minutes=timeout_minutes,
            max_repair_attempts=max_repair_attempts,
        )
        self.store = SessionStore(self.session.id)
        self.agent_running = False
        self._pending_shell: str | None = None
        self._pending_env_delete: dict | None = None
        self._force_kill_requested: bool = False
        self._pipeline_worker: object | None = None
        self._active_panel = "pipeline"

        # Widget refs
        self._header: HeaderLogo | None = None
        self._timeline: MessageTimeline | None = None
        self._status_bar: StatusBar | None = None
        self._composer: Composer | None = None
        self._session_panel: SessionPanel | None = None
        self._pipeline_panel: PipelinePanel | None = None
        self._help_panel: HelpPanel | None = None
        self._artifact_panel: ArtifactPanel | None = None
        self._right_session_panel: SessionPanel | None = None

        # Tool cards tracking
        self._tool_cards: dict[str, ToolCard] = {}
        self._active_tool_by_stage: dict[str, str] = {}

        # Stage view state (survives panel switches)
        self._stage_views: dict[str, StageView] = {}
        self._reset_stage_views()

        # Run tracking for live status
        self._run_started_at: float | None = None
        self._current_stage: str | None = None
        self._current_stage_message: str | None = None
        self._last_progress_at: float | None = None
        self._last_live_log_sig: tuple | None = None
        self._last_chrome_update_at: float = 0.0
        self._heartbeat_timer: object | None = None

    # ── Compose ──────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield HeaderLogo(
            session_id=self.session.id,
            backend=self.session.backend,
            mode="ACT",
            status="draft",
        )

        with Horizontal(id="main-content"):
            yield SessionPanel(classes="left-panel")
            yield MessageTimeline(id="timeline-area")
            with Container(id="right-panel"):
                yield PipelinePanel(backend=self.session.backend)

        yield StatusBar()
        yield Composer(mode="act")
        yield Static("", id="bottom-spacer-1")
        yield Static("", id="bottom-spacer-2")

    def on_mount(self) -> None:
        if not self._skip_splash:
            from .widgets.boot_splash import BootSplash

            self.push_screen(BootSplash(), callback=self._on_splash_done)
            return
        self._on_mount_ready()

    def _on_splash_done(self, results=None) -> None:
        self._preflight_results = results or []
        self._on_mount_ready()
        self.refresh(layout=True)

    def _on_mount_ready(self) -> None:
        self._header = self.query_one(HeaderLogo)
        self._timeline = self.query_one(MessageTimeline)
        self._status_bar = self.query_one(StatusBar)
        self._composer = self.query_one(Composer)
        self._session_panel = self.query_one("#main-content SessionPanel")
        self._pipeline_panel = self.query_one("#right-panel PipelinePanel")

        self._composer.focus_input()
        self._add_assistant(
            "Welcome to **Alpha-Assay**.\n\n"
            "Enter a local paper PDF path to start, e.g.:\n\n"
            "    `@/path/to/paper.pdf`\n\n"
            "Common commands:\n"
            "  `/backend conda`\n"
            "  `/repo https://github.com/user/repo`\n"
            "  `/run`\n"
            "  `/logs smoke`\n"
            "  `/report`\n\n"
            "Enter `/help` to see all commands."
        )
        self._update_status()
        self._sync_session_panel()
        self._show_preflight_warnings()

    def _show_preflight_warnings(self) -> None:
        results = getattr(self, "_preflight_results", None) or []
        failed = [r for r in results if getattr(r, "status", "") == "fail"]
        if not failed:
            return
        lines = ["Startup checks reported warnings:", ""]
        for item in failed:
            level = "blocking" if getattr(item, "blocking", False) else "non-blocking"
            lines.append(f"- {item.name}: {item.message} ({level})")
        self._add_assistant("\n".join(lines))

    # ── Message helpers ──────────────────────────────────────

    def _add_assistant(self, text: str) -> None:
        if self._timeline:
            self._timeline.add_assistant(text)
        self.store.append(AgentEvent(type="assistant_message", payload={"text": text}))

    def _add_user(self, text: str) -> None:
        if self._timeline:
            self._timeline.add_user(text)
        self.store.append(AgentEvent(type="user_message", payload={"text": text}))

    def _add_error(self, text: str) -> None:
        if self._timeline:
            self._timeline.add_error(text)
        self.store.append(AgentEvent(type="error", payload={"text": text}))

    def _add_system(self, text: str) -> None:
        if self._timeline:
            self._timeline.add_system(text)

    def _add_report_msg(self, text: str) -> None:
        if self._timeline:
            self._timeline.add_report(text)

    def _update_status(self) -> None:
        if self._status_bar:
            paper_name = "-"
            if self.session.paper_path:
                paper_name = Path(self.session.paper_path).name
            self._status_bar.update_info(
                mode=self.session.mode.upper(),
                session=self.session.id,
                backend=self.session.backend,
                status=self.session.status,
                paper=paper_name,
                panel=self._active_panel,
            )
        if self._header:
            self._header.update_summary(
                session_id=self.session.id,
                backend=self.session.backend,
                mode=self.session.mode.upper(),
                status=self.session.status,
            )

    def _sync_session_panel(self) -> None:
        if self._session_panel:
            s = self.session
            self._session_panel.update_session(
                session_id=s.id,
                mode=s.mode,
                backend=s.backend,
                workspace=s.workspace,
                paper=s.paper_path or "",
                repo=s.repo or "",
                repo_dir=s.repo_dir or "",
                timeout=str(s.timeout_minutes),
                repairs=str(s.max_repair_attempts),
                task_dir=s.task_dir or "",
                report_path=s.report_path or "",
                status=s.status,
                cancel_requested=s.cancel_requested,
            )

    def _sync_artifact_panel(self) -> None:
        if not self._artifact_panel or not self.session.task_dir:
            return
        td = Path(self.session.task_dir)
        self._artifact_panel.update_artifacts(
            task_dir=str(td),
            report_md=str(td / "report" / "reproduction_smoke_report.md"),
            report_json=str(td / "report" / "reproduction_smoke_report.json"),
            state_json=str(td / "state.json"),
            env_log=str(td / "env" / "conda_build.log"),
            smoke_log=str(td / "runs" / "smoke_001" / "stdout.log"),
            benchmark_log=str(td / "runs" / "benchmark_001" / "stderr.log"),
            reproduction_log=str(td / "runs" / "reproduction_001" / "stderr.log"),
        )

    # ── Stage state management ───────────────────────────────

    def _reset_stage_views(self) -> None:
        """Initialize stage views based on current backend."""
        from .panels.pipeline_panel import PIPELINE_STAGES

        self._stage_views.clear()
        for stage_name in PIPELINE_STAGES:
            active = self._is_stage_active(stage_name)
            self._stage_views[stage_name] = StageView(
                name=stage_name,
                status="queued" if active else "disabled",
            )

    def _is_stage_active(self, stage: str) -> bool:
        build_map = {
            "Build conda env": "conda",
            "Build virtualenv": "venv",
            "Build Docker image": "docker",
        }
        if stage in build_map:
            return build_map[stage] == self.session.backend
        if self.session.backend == "none" and stage in (
            "Run smoke command",
            "Run benchmark reproduction",
            "Run simple reproduction",
        ):
            return False
        return True

    def _sync_pipeline_panel(self) -> None:
        """Sync stage views to the active pipeline panel."""
        if not self._pipeline_panel:
            return
        for sv in self._stage_views.values():
            self._pipeline_panel._stages[sv.name] = sv
        self._pipeline_panel._refresh()

    # ── Panel switching ──────────────────────────────────────

    def _switch_panel(self, name: str) -> None:
        self._active_panel = name
        right = self.query_one("#right-panel", Container)
        right.remove_children()

        if name == "session":
            panel = SessionPanel()
            right.mount(panel)
            self._right_session_panel = panel
            self._sync_right_session_panel()

        elif name == "pipeline":
            panel = PipelinePanel(backend=self.session.backend)
            right.mount(panel)
            self._pipeline_panel = panel
            # Restore stage state
            for sv in self._stage_views.values():
                panel._stages[sv.name] = sv
            panel._refresh()

        elif name == "help":
            panel = HelpPanel()
            right.mount(panel)
            self._help_panel = panel

        elif name == "artifacts":
            panel = ArtifactPanel()
            right.mount(panel)
            self._artifact_panel = panel
            self._sync_artifact_panel()

        elif name == "none":
            pass

        self._update_status()

    def _sync_right_session_panel(self) -> None:
        if self._right_session_panel:
            s = self.session
            self._right_session_panel.update_session(
                session_id=s.id,
                mode=s.mode,
                backend=s.backend,
                workspace=s.workspace,
                paper=s.paper_path or "",
                repo=s.repo or "",
                repo_dir=s.repo_dir or "",
                timeout=str(s.timeout_minutes),
                repairs=str(s.max_repair_attempts),
                task_dir=s.task_dir or "",
                report_path=s.report_path or "",
                status=s.status,
                cancel_requested=s.cancel_requested,
            )

    # ── Tool card helpers ────────────────────────────────────

    def _create_tool_card(self, stage: str, message: str) -> ToolCard:
        key = f"{stage}:{len(self._tool_cards)}"
        card = ToolCard(name=stage, status="running", message=message)
        if self._timeline:
            self._timeline.mount(card)
            self._timeline.scroll_end(animate=False)
        self._tool_cards[key] = card
        self._active_tool_by_stage[stage] = key
        return card

    def _get_active_tool_card(self, stage: str) -> ToolCard | None:
        key = self._active_tool_by_stage.get(stage)
        if key is None:
            return None
        return self._tool_cards.get(key)

    def _get_or_create_tool_card(self, stage: str, message: str) -> ToolCard:
        existing = self._get_active_tool_card(stage)
        if existing is not None:
            existing.update(status="running", message=message)
            return existing
        return self._create_tool_card(stage, message)

    def _update_tool_card(
        self,
        stage: str,
        status: str,
        detail: str | None = None,
        duration: float | None = None,
    ) -> None:
        """Update active tool card and release mapping on terminal states."""
        card = self._get_active_tool_card(stage)
        if card is not None:
            card.update(status=status, detail=detail, duration=duration)
        if status in ("success", "failed"):
            self._active_tool_by_stage.pop(stage, None)

    # ── Progress handling ────────────────────────────────────

    _STAGE_LABELS_CN: dict[str, str] = {
        "Ingest paper": "Ingest paper",
        "Understand paper": "Understand paper",
        "Search GitHub": "Search GitHub",
        "Evaluate repo": "Evaluate repo",
        "Build conda env": "Build conda env",
        "Build virtualenv": "Build virtualenv",
        "Build Docker image": "Build Docker image",
        "Decide runtime": "Decide runtime",
        "Run smoke command": "Run smoke command",
        "Run benchmark reproduction": "Run benchmark reproduction",
        "Run simple reproduction": "Run simple reproduction",
        "Write report": "Write report",
        "Download arXiv PDF": "Download arXiv PDF",
    }

    _PHASE_LABELS_CN: dict[str, str] = {
        "start": "start",
        "progress": "running",
        "finish": "success",
        "fail": "failed",
        "skip": "skipped",
    }

    _DATA_KEY_LABELS_CN: dict[str, str] = {
        "repo_url": "Repo URL",
        "repo_dir": "Repo directory",
        "command": "Command",
        "log_path": "logpath",
        "selected_repo": "Selected repo",
        "runnable_score": "Runnable score",
    }

    def _handle_progress_event(
        self, ev: ProgressEvent, stage_times: dict[str, float]
    ) -> None:
        """Handle a progress event from the pipeline thread (called via call_from_thread)."""
        name = ev.stage
        now = time.monotonic()
        is_cli_stream = bool(ev.data.get("cli_stream"))

        self._last_progress_at = now
        self._current_stage = name
        self._current_stage_message = ev.message

        if ev.phase == "start":
            stage_times[name] = now
        duration = None
        if name in stage_times:
            duration = now - stage_times[name]

        phase_status = {
            "start": "running",
            "finish": "success",
            "fail": "failed",
            "progress": "running",
            "skip": "skipped",
        }
        status = phase_status.get(ev.phase, "running")

        # Build log lines and update ToolCard
        log_lines = self._build_progress_log_lines(ev)
        progress_kind = ev.data.get("progress_kind")
        card = self._get_or_create_tool_card(name, ev.message)
        for line in log_lines:
            if progress_kind:
                card.upsert_log(str(progress_kind), line)
            else:
                card.append_log(line)
        card.update(status=status, message=ev.message, detail=None, duration=duration)

        # cli_stream is high-frequency — update card only, skip chrome
        if is_cli_stream:
            self._throttled_chrome_update(name, status, ev, duration, now)
            return

        # Full update for normal events
        if name in self._stage_views:
            sv = self._stage_views[name]
            if sv.status in ("failed", "success", "skipped") and status == "running":
                sv.attempts += 1
            sv.status = status  # type: ignore[assignment]
            sv.message = ev.message
            sv.detail = ev.detail or ""
            sv.duration = duration

        if self._pipeline_panel:
            self._pipeline_panel.update_from_name(
                name=name,
                status=status,
                message=ev.message,
                detail=ev.detail or "",
                duration=duration,
            )

        if ev.phase == "fail":
            self._add_error(
                f"{self._STAGE_LABELS_CN.get(name, name)} failed: {ev.message}"
            )

        if self._status_bar:
            self._status_bar.update_info(status=status)
        if self._header:
            self._header.update_summary(status=status)

        if ev.phase in ("finish", "fail"):
            self._sync_artifact_panel()

    def _throttled_chrome_update(
        self,
        name: str,
        status: str,
        ev: ProgressEvent,
        duration: float | None,
        now: float,
    ) -> None:
        # Full chrome update every 5 seconds
        last = getattr(self, "_last_cli_chrome_update_at", 0.0)
        if now - last < 5.0:
            self._throttled_screen_refresh(now)
            return
        self._last_cli_chrome_update_at = now
        if self._pipeline_panel:
            self._pipeline_panel.update_from_name(
                name=name,
                status=status,
                message=ev.message,
                detail="",
                duration=duration,
            )
        if self._status_bar:
            self._status_bar.update_info(status=status)
        if self._header:
            self._header.update_summary(status=status)
        self.screen.refresh()

    def _throttled_screen_refresh(self, now: float) -> None:
        """Force full screen repaint at 1 Hz to prevent Windows Terminal
        from showing black/unrendered areas during high-frequency progress."""
        last = getattr(self, "_last_screen_refresh_at", 0.0)
        if now - last < 1.0:
            return
        self._last_screen_refresh_at = now
        self.screen.refresh()

    def _build_progress_log_lines(self, ev: ProgressEvent) -> list[str]:
        """Extract log lines from a progress event for the ToolCard buffer. Deduplicates progress bar text."""
        lines: list[str] = []

        # Progress-specific: single bar+text line, skip redundant fields
        bar = ev.data.get("progress_bar")
        ptext = ev.data.get("progress_text")
        progress_kind = ev.data.get("progress_kind")

        if progress_kind and (bar or ptext):
            if bar and ptext:
                return [f"{bar} {clean_display_text(str(ptext))}"]
            elif ptext:
                return [clean_display_text(str(ptext))]
            elif bar:
                return [str(bar)]

        seen: set[str] = set()

        def _add(clean: str) -> None:
            if clean and clean not in seen:
                seen.add(clean)
                lines.append(clean)

        if ev.detail:
            for line in ev.detail.splitlines():
                _add(clean_display_text(line))

        for key in (
            "repo_url",
            "repo_dir",
            "command",
            "log_path",
            "selected_repo",
            "runnable_score",
            "proxy_status",
        ):
            val = ev.data.get(key)
            if val:
                _add(
                    f"{self._DATA_KEY_LABELS_CN.get(key, key)}: {clean_display_text(str(val))}"
                )

        raw_lines = ev.data.get("log_lines")
        if isinstance(raw_lines, list):
            for line in raw_lines:
                _add(clean_display_text(str(line)))

        if not lines and ev.message:
            _add(clean_display_text(ev.message))
        return lines

    # ── Composer input ───────────────────────────────────────

    def on_composer_submitted(self, event: Composer.Submitted) -> None:
        line = event.value
        if not line:
            return
        self.handle_line(line)

    def handle_line(self, line: str) -> None:
        if self._pending_shell is not None:
            self._confirm_shell(line)
            return

        if self._pending_env_delete is not None:
            self._confirm_env_delete(line)
            return

        command, args = parse_command(line)
        if command == "":
            return

        # Running guard — blocks !shell and other unsafe commands
        if self.agent_running and command not in RUNNING_SAFE_COMMANDS:
            if command == "!":
                self._add_assistant(
                    "Agent is running; shell commands are not allowed right now."
                    "Use /status /logs /cancel to inspect or control the task."
                )
            else:
                safe = " ".join(f"/{c}" for c in sorted(RUNNING_SAFE_COMMANDS))
                self._add_assistant(f"Agent is running. Available commands: {safe}")
            return

        if command == "!":
            self._run_shell(args)
            return
        if command == "message":
            if _looks_like_arxiv(args):
                self._cmd_arxiv(args)
                return
            self._submit_paper(args)
            return
        if command == "unknown_command":
            self._handle_unknown_command(args)
            return

        handlers = {
            "help": self._cmd_help,
            "clear": self._cmd_clear,
            "status": self._cmd_status,
            "plan": self._cmd_plan,
            "act": self._cmd_act,
            "input": self._cmd_input,
            "arxiv": self._cmd_arxiv,
            "download-arxiv": self._cmd_arxiv,
            "repo": self._cmd_repo,
            "repo-dir": self._cmd_repo_dir,
            "backend": self._cmd_backend,
            "workspace": self._cmd_workspace,
            "timeout": self._cmd_timeout,
            "repairs": self._cmd_repairs,
            "run": self._cmd_run,
            "report": self._cmd_report,
            "logs": self._cmd_logs,
            "cancel": self._cmd_cancel,
            "kill": self._cmd_kill,
            "force-cancel": self._cmd_kill,
            "abort": self._cmd_kill,
            "sessions": self._cmd_sessions,
            "session": self._cmd_sessions,
            "resume": self._cmd_resume,
            "quit": self._cmd_quit,
            "exit": self._cmd_quit,
            "panel": self._cmd_panel,
            "artifact": self._cmd_artifact,
            "open-report": self._cmd_open_report,
            "mode": self._cmd_mode,
            "reset": self._cmd_reset,
            "conda-envs": self._cmd_conda_envs,
            "envs": self._cmd_conda_envs,
        }
        handler = handlers.get(command)
        if handler is None:
            self._add_assistant(
                f"Unknown command /{command}. Enter /help for available commands.\n"
                f"Usage: /{command} <args>"
            )
            return
        handler(args)

    # ── Commands ──────────────────────────────────────────────

    def _cmd_help(self, args: str) -> None:
        cat = args.strip() if args else ""
        lines: list[str] = ["## Available commands", ""]
        shown = False
        for cat_name, cmds in _HELP_CATEGORIES.items():
            if cat and cat_name != cat:
                continue
            shown = True
            lines.append(f"### {cat_name}")
            for name in cmds:
                meta = COMMANDS[name]
                arg_str = (
                    f" {meta.display_args or meta.args}"
                    if (meta.display_args or meta.args)
                    else ""
                )
                lines.append(f"- `/{name}{arg_str}` — {meta.description}")
            lines.append("")
        if not shown and cat:
            cats = ", ".join(_HELP_CATEGORIES.keys())
            lines.append(f"Category '{cat}' not found. Available: {cats}")
        if not lines:
            lines = ["Enter `/help` to see all commands."]
        self._add_assistant("\n".join(lines))

    def _handle_unknown_command(self, text: str) -> None:
        """Handle an unrecognized slash command."""
        cmd_name = text.lstrip("/").split()[0].lower() if text.startswith("/") else text

        # Try completion to suggest alternatives
        from .completion import complete_command

        suggestions = complete_command(f"/{cmd_name}", limit=4)
        if suggestions:
            sug_lines = [f"Unknown command: `/{cmd_name}`", "", "Did you mean:"]
            for s in suggestions:
                arg_str = (
                    f" {s.display_args or s.args}" if (s.display_args or s.args) else ""
                )
                sug_lines.append(f"  `/{s.command}{arg_str}` — {s.description}")
            self._add_assistant("\n".join(sug_lines))
        else:
            self._add_assistant(
                f"Unknown command: `/{cmd_name}`. \n"
                "Enter `/help` for available commands, or a local PDF path to start replication. "
            )

    def _cmd_clear(self, _: str) -> None:
        if self._timeline:
            self._timeline.clear_messages()

    def _cmd_status(self, _: str) -> None:
        if self.agent_running:
            elapsed = (
                time.monotonic() - self._run_started_at if self._run_started_at else 0
            )
            stage_label = self._STAGE_LABELS_CN.get(
                self._current_stage or "", self._current_stage or "-"
            )
            self._add_assistant(
                "## Task running\n\n"
                f"- Current stage: **{stage_label}**\n"
                f"- Stage message: {self._current_stage_message or '-'}\n"
                f"- Backend: `{self.session.backend}`\n"
                f"- Paper: `{self.session.paper_path or '-'}`\n"
                f"- Elapsed: {elapsed:.1f}s\n"
                "- Available commands: `/status` `/logs` `/cancel`\n"
            )
            return

        s = self.session
        if not s.task_dir:
            self._add_assistant("No task has run in this session yet.")
            return
        try:
            state = load_state(s.task_dir)
            self._add_assistant(
                f"**Task status**\n\n"
                f"- Task: `{state.task_id}`\n"
                f"- Status: `{state.status}`\n"
                f"- Task dir: `{state.task_dir}`"
            )
            if state.report:
                self._add_assistant(f"- Final status: `{state.report.final_status}`")
        except Exception as e:
            self._add_error(f"Cannot load state: {e}")

    def _cmd_plan(self, _: str) -> None:
        self.session.mode = "plan"
        self._update_status()
        if self._composer:
            self._composer.set_mode("plan")
        self._add_assistant("Switched to **[PLAN]** mode — analysis only, no commands executed.")

    def _cmd_act(self, _: str) -> None:
        self.session.mode = "act"
        self._update_status()
        if self._composer:
            self._composer.set_mode("act")
        self._add_assistant("Switched to **[ACT]** mode — execution allowed.")

    def _cmd_mode(self, _: str) -> None:
        self._add_assistant(f"Current mode: **{self.session.mode.upper()}**")

    def _cmd_input(self, args: str) -> None:
        self.session.paper_path = args.strip().lstrip("@")
        self.session.input_resolved = False
        self._sync_session_panel()
        self._add_assistant(f"Input set: `{self.session.paper_path}`")

    def _cmd_arxiv(self, args: str) -> None:
        value = args.strip()
        if not value:
            self._add_error(
                "Usage: `/arxiv <arXiv ID or URL>`, e.g. `/arxiv https://arxiv.org/abs/1911.11763`"
            )
            return
        if self.agent_running:
            self._add_error("Agent is running; wait for completion or `/cancel` first.")
            return
        self._add_assistant(f"Starting arXiv PDF download: `{value}`")
        self.run_worker(self._download_arxiv_worker(value), exclusive=False)

    async def _download_arxiv_worker(self, value: str) -> None:
        from app.core.paths import project_pdf_dir
        from app.tools.arxiv_download import (
            download_arxiv_pdf,
            make_progress_bar as arxiv_bar,
        )

        def progress_cb(event: dict) -> None:
            phase = event.get("phase")
            if phase == "start":
                self.call_from_thread(
                    lambda: self._add_system(
                        f"arXiv download started\n- URL: {event.get('pdf_url')}\n- Saved to: {event.get('pdf_path')}"
                    )
                )
            elif phase == "progress":
                pct = event.get("percent")
                dl = int(event.get("downloaded") or 0)
                tot = int(event.get("total") or 0)
                bar = arxiv_bar(pct)
                msg = f"{bar} Downloaded {dl / 1024 / 1024:.1f} MiB"
                if tot:
                    msg += f" / {tot / 1024 / 1024:.1f} MiB"
                final_msg = msg
                self.call_from_thread(
                    lambda m=final_msg: self._upsert_arxiv_progress(m)
                )
            elif phase == "finish":
                self.call_from_thread(
                    lambda: self._upsert_arxiv_progress("Download complete, importing")
                )

        result = await asyncio.to_thread(
            download_arxiv_pdf,
            value,
            output_dir=project_pdf_dir(),
            overwrite=False,
            progress_cb=progress_cb,
        )

        if result.success and result.pdf_path:
            self.session.paper_path = str(result.pdf_path.resolve())
            self.session.input_resolved = True
            self.session.status = "draft"
            self._sync_session_panel()
            self._update_status()
            reused = " (reusing existing file)" if result.reused_existing else ""
            self._add_assistant(
                f"arXiv PDF download complete{reused}. \n\n"
                f"- arXiv ID: `{result.arxiv_id}`\n"
                f"- PDF: `{result.pdf_path}`\n\n"
                "Set as the current paper. You can now run `/run`."
            )
            card = self._get_active_tool_card("Download arXiv PDF")
            if card:
                card.update(status="success", message="Download arXiv PDF")
        else:
            self._add_error(f"arXiv PDF download failed: {result.error or 'unknown error'}")

    def _upsert_arxiv_progress(self, text: str) -> None:
        stage = "Download arXiv PDF"
        card = self._get_or_create_tool_card(stage, "Download arXiv PDF")
        card.upsert_log("arxiv_download", text)
        card.update(status="running", message="Download arXiv PDF")

    def _cmd_repo(self, args: str) -> None:
        self.session.repo = args.strip() or None
        self._sync_session_panel()
        self._add_assistant(f"Repository: {self.session.repo or '-'}")

    def _cmd_repo_dir(self, args: str) -> None:
        self.session.repo_dir = args.strip().lstrip("@")
        self._sync_session_panel()
        self._add_assistant(f"Local repository: {self.session.repo_dir}")

    def _cmd_backend(self, args: str) -> None:
        backend = args.strip().lower()
        if backend:
            if backend not in {"none", "local", "venv", "conda", "docker"}:
                self._add_error(
                    "Backend must be: none, local, venv, conda or docker\nUsage: `/backend conda`"
                )
                return
            self.session.backend = backend
            self._sync_session_panel()
            if backend == "none":
                self._add_assistant(
                    "Switched to **none** mode: static analysis only, no code execution.\n\n"
                    "This mode will run: Ingest paper → Understand paper → Search GitHub → Evaluate repo → Write report\n"
                    "It will NOT run: environment builds, smoke test, benchmark reproduction, lightweight reproduction\n\n"
                    "To run reproduction steps, use `/backend conda`."
                )
            elif backend == "conda":
                self._add_assistant(
                    "Switched to the **conda**  mode: will build a conda environment and run smoke/benchmark/lightweight reproduction."
                )
            elif backend == "venv":
                self._add_assistant(
                    "Switched to the **venv**  mode: will build a virtualenv and run smoke/benchmark/lightweight reproduction."
                )
            elif backend == "local":
                self._add_assistant(
                    "Switched to the **local**  mode: skipping environment build, running smoke/benchmark/lightweight reproduction directly on the host."
                )
            elif backend == "docker":
                self._add_assistant(
                    "Switched to the **docker**  mode: will build a Docker image and run smoke/benchmark/lightweight reproduction."
                )
            if self._pipeline_panel:
                self._pipeline_panel.reset(backend)
        else:
            opts = " | ".join(
                f"**{b}**" if b == self.session.backend else b
                for b in ["none", "local", "venv", "conda", "docker"]
            )
            self._add_assistant(f"Available backends: {opts}\ncurrent: **{self.session.backend}**")

    def _cmd_workspace(self, args: str) -> None:
        self.session.workspace = args.strip() or self.session.workspace
        self._sync_session_panel()
        self._add_assistant(f"Workspace: `{self.session.workspace}`")

    def _cmd_timeout(self, args: str) -> None:
        try:
            self.session.timeout_minutes = int(args.strip())
            self._sync_session_panel()
            self._add_assistant(f"Timeout: {self.session.timeout_minutes} minutes")
        except (ValueError, TypeError):
            self._add_error("Usage: `/timeout <minutes>` (integer)")

    def _cmd_repairs(self, args: str) -> None:
        try:
            self.session.max_repair_attempts = int(args.strip())
            self._sync_session_panel()
            self._add_assistant(f"Max repair attempts: {self.session.max_repair_attempts}")
        except (ValueError, TypeError):
            self._add_error("Usage: `/repairs <n>` (integer)")

    def _cmd_panel(self, args: str) -> None:
        name = args.strip().lower() or "pipeline"
        valid = {"session", "pipeline", "help", "artifacts", "none"}
        if name not in valid:
            self._add_error(f"Panel must be one of: {', '.join(valid)}\nUsage: `/panel help`")
            return
        self._switch_panel(name)
        self._add_system(f"Switched to the **{name}** panel")

    def _cmd_artifact(self, _: str) -> None:
        self._switch_panel("artifacts")
        self._sync_artifact_panel()
        if not self.session.task_dir:
            self._add_assistant("No task has run yet — no artifacts. Enter `/run` to start.")
        else:
            self._add_assistant(f"Current task dir: `{self.session.task_dir}`")

    def _cmd_open_report(self, _: str) -> None:
        if not self.session.report_path:
            self._add_assistant("No report generated yet. Enter `/run` to start.")
            return
        rp = self.session.report_path
        self._add_report_msg(f"Report path: `{rp}`")

    def _cmd_reset(self, _: str) -> None:
        self.session.paper_path = None
        self.session.repo = None
        self.session.repo_dir = None
        self.session.input_resolved = False
        self.session.status = "draft"
        self.session.task_dir = None
        self.session.report_path = None
        self.session.cancel_requested = False
        self._sync_session_panel()
        self._tool_cards.clear()
        self._active_tool_by_stage.clear()
        if self._pipeline_panel:
            self._pipeline_panel.reset(self.session.backend)
        self._add_assistant("Session reset (disk files kept).")

    @staticmethod
    def _status_label(status: str) -> str:
        return {"success": "✓", "failed": "✗", "running": "●", "cancelled": "○"}.get(
            status, "–"
        )

    def _cmd_run(self, _: str) -> None:
        paper = self.session.paper_path
        if not paper:
            self._add_error("Set the paper file first with /input <pdf path>.")
            return

        if not Path(paper).is_file():
            self._add_error(f"File not found: `{paper}`")
            return

        self._add_user(f"Replicating paper: {paper}")
        self._add_assistant("Got it, checking that the local PDF is readable.")

        # Warn if backend=none
        if self.session.backend == "none":
            self._add_assistant(
                "The current backend is **none**.\n\n"
                "This mode will run:\n"
                "- Ingest paper\n"
                "- Understand paper\n"
                "- Search GitHub\n"
                "- Evaluate repo\n"
                "- Write report\n\n"
                "It will NOT run:\n"
                "- conda / venv / docker environment builds\n"
                "- smoke test\n"
                "- benchmark reproduction\n"
                "- lightweight reproduction\n\n"
                "To run later reproduction steps, enter `/backend conda` first. "
            )

        if self.session.mode == "plan":
            self._add_assistant(
                "Currently in **PLAN** mode: showing the execution plan only.\n"
                "The pipeline will run these stages:\n"
                "1. Ingest paper PDF\n"
                "2. LLM task analysis\n"
                "3. GitHub search\n"
                "4. Repository evaluation\n"
                "5. Environment build\n"
                "6. Smoke test\n"
                "7. Benchmark reproduction\n"
                "8. Lightweight reproduction\n"
                "9. Report writing\n\n"
                "Enter `/act` to switch mode, then retry `/run`."
            )
            return

        # Run pipeline
        self.agent_running = True
        self.session.status = "running"
        self.session.cancel_requested = False

        # Initialize run tracking
        self._run_started_at = time.monotonic()
        self._current_stage = None
        self._current_stage_message = None
        self._last_progress_at = None
        self._last_live_log_sig = None

        self._update_status()
        self._sync_session_panel()
        if self._composer:
            self._composer.set_running(True)

        self._reset_stage_views()
        if self._pipeline_panel:
            self._pipeline_panel.reset(self.session.backend)

        # Start heartbeat
        self._heartbeat_timer = self.set_interval(10, self._heartbeat_running_task)

        def _cancel_check() -> bool:
            return self.session.cancel_requested

        self._pipeline_worker = self.run_worker(
            self._do_run(paper, _cancel_check),
            exclusive=False,
        )

    async def _do_run(self, paper: str, cancel_check) -> None:
        try:
            stage_times: dict[str, float] = {}

            def run_with_progress():
                def on_event(ev: ProgressEvent) -> None:
                    self.call_from_thread(self._handle_progress_event, ev, stage_times)

                with progress_events(on_event):
                    return self.runner(
                        input_value=paper,
                        workspace=self.session.workspace,
                        backend=self.session.backend,
                        repo=self.session.repo,
                        repo_dir=self.session.repo_dir,
                        timeout_minutes=self.session.timeout_minutes,
                        max_repair_attempts=self.session.max_repair_attempts,
                        should_cancel=cancel_check,
                    )

            state = await asyncio.to_thread(run_with_progress)

            self.session.task_dir = state.task_dir
            self.session.status = getattr(state, "status", "completed")
            report_path = (
                Path(state.task_dir) / "report" / "reproduction_smoke_report.md"
            )
            if report_path.exists():
                self.session.report_path = str(report_path)

            # Build final message
            final = (
                getattr(state.report, "final_status", "unknown")
                if state.report
                else "unknown"
            )
            if final == "repo_found_smoke_not_run" and self.session.backend == "none":
                self._add_assistant(
                    f"**Pipeline success**\n\n"
                    f"Static evaluation completed. backend=none, so no code reproduction steps were executed.\n"
                    f"- Final status: `{final}`\n"
                    f"- Report: `{report_path}`"
                )
            else:
                self._add_assistant(
                    f"**Pipeline success**\n\n"
                    f"- Final status: `{final}`\n"
                    f"- Report: `{report_path}`"
                )

            if report_path.exists():
                preview = report_path.read_text(encoding="utf-8", errors="ignore")[
                    :2000
                ]
                self._add_report_msg(preview)

            self._sync_session_panel()
            self._sync_artifact_panel()

        except Exception as e:
            self.session.status = "failed"
            self._add_error(f"Pipeline exception: {e}")
        finally:
            # Don't overwrite killed state
            if self._force_kill_requested:
                return
            self.agent_running = False
            # Stop heartbeat
            if self._heartbeat_timer is not None:
                try:
                    self._heartbeat_timer.stop()  # type: ignore[union-attr]
                except Exception:
                    pass
                self._heartbeat_timer = None
            if self._composer:
                self._composer.set_running(False)
                self._composer.focus_input()
        self._force_kill_requested = False
        self._update_status()
        self._sync_session_panel()
        self._sync_artifact_panel()

    def _cmd_cancel(self, _: str) -> None:
        if not self.agent_running:
            self._add_assistant("No task is currently running.")
            return
        self.session.cancel_requested = True
        self._add_system("Cancellation requested. Waiting for the current step...")

    def _cmd_kill(self, _: str) -> None:
        if not self.agent_running:
            self._add_assistant("No task is currently running. ")
            return

        self.session.cancel_requested = True
        self._force_kill_requested = True
        self.session.status = "killed"
        self._sync_session_panel()
        self._update_status()

        self._add_error("Force-kill requested: killing task child processes...")

        from app.core.process_control import get_process_registry

        killed = get_process_registry().kill_all()
        if killed:
            self._add_system("Killed child processes:\n" + "\n".join(f"- {x}" for x in killed))
        else:
            self._add_system("No registered running child processes found.")

        # Cancel Textual worker
        try:
            if self._pipeline_worker is not None:
                self._pipeline_worker.cancel()  # type: ignore[union-attr]
        except Exception:
            pass

        self._mark_run_killed()

    def _mark_run_killed(self) -> None:
        self.agent_running = False
        self.session.status = "killed"
        self._run_started_at = None
        self._current_stage_message = "Force-killed"

        if self._current_stage:
            card = self._get_active_tool_card(self._current_stage)
            if card is not None:
                card.append_log("The task was force-killed.")
                card.update(status="failed", message="Force-killed")

        if self._current_stage and self._pipeline_panel:
            self._pipeline_panel.update_from_name(
                name=self._current_stage,
                status="failed",
                message="Force-killed",
                detail="force killed",
            )

        # Cleanup incomplete files
        _cleanup_killed_task_files(self.session.task_dir)

        self._sync_session_panel()
        self._update_status()
        if self._composer:
            self._composer.set_running(False)
            self._composer.focus_input()
        if self._heartbeat_timer is not None:
            try:
                self._heartbeat_timer.stop()  # type: ignore[union-attr]
            except Exception:
                pass
            self._heartbeat_timer = None

    def _heartbeat_running_task(self) -> None:
        """Periodic check: show 'still running' in the current stage card."""
        if not self.agent_running:
            return
        now = time.monotonic()
        if self._last_progress_at and now - self._last_progress_at < 12:
            return
        elapsed = now - self._run_started_at if self._run_started_at else 0
        stage_label = self._STAGE_LABELS_CN.get(
            self._current_stage or "", self._current_stage or "Pipeline"
        )

        text = f"Still running: {stage_label} elapsed {elapsed:.0f}s — may be waiting on network, repository clone, dependency resolution or model response."

        card = (
            self._get_active_tool_card(self._current_stage or "")
            if self._current_stage
            else None
        )
        if card is not None:
            # Replace previous heartbeat in this card instead of stacking
            clean = clean_display_text(text)
            if clean.startswith("Still running: "):
                for i in range(len(card._log_lines) - 1, -1, -1):
                    if card._log_lines[i].startswith("Still running: "):
                        card._log_lines[i] = clean
                        card._refresh_display()
                        break
                else:
                    card.append_log(text)
                    card.update(status="running")
            else:
                card.append_log(text)
                card.update(status="running")
        else:
            self._add_system(text)

        self._last_progress_at = now

    # ── Paper submission ─────────────────────────────────────

    def _submit_paper(self, text: str) -> None:
        self.session.paper_path = text.strip().lstrip("@")
        self.session.input_resolved = True
        self._add_user(text)
        self._sync_session_panel()

        # Resolve input
        try:
            resolution = self.resolver.resolve(text)
        except Exception as e:
            self._add_error(f"Cannot parse input: {e}")
            return

        if not resolution.success:
            self._add_error(resolution.failure_reason or "invalid input")
            return

        self.session.paper_path = resolution.input_value or self.session.paper_path
        self._sync_session_panel()

        info_lines = [f"Resolved paper path: `{resolution.input_value}`"]
        if resolution.title:
            info_lines.append(f"Title: {resolution.title}")
        self._add_assistant("\n".join(info_lines))

        # Auto-run in ACT mode
        if self.session.mode == "act":
            self._cmd_run("")

    # ── Shell ─────────────────────────────────────────────────

    def _run_shell(self, cmd: str) -> None:
        cmd = cmd.strip()
        if not cmd:
            self._add_error("!shell needs a command argument, e.g. `!ls -la`")
            return
        self._pending_shell = cmd
        self._add_assistant(
            f"About to run the shell command:\n```\n{cmd}\n```\nEnter `yes` to confirm, or anything else to cancel. "
        )

    def _confirm_shell(self, line: str) -> None:
        if line.strip().lower() not in ("yes", "y"):
            self._add_assistant("Cancelled.")
            self._pending_shell = None
            return
        cmd = self._pending_shell
        self._pending_shell = None
        import subprocess

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            output = result.stdout[:3000] or "(no stdout)"
            if result.stderr:
                output += f"\n\nstderr:\n{result.stderr[:1000]}"
            self._add_tool_message(f"$ {cmd}\n\n{output}", "Shell")
        except subprocess.TimeoutExpired:
            self._add_error("Command timed out (30s).")
        except Exception as e:
            self._add_error(f"Command execution failed: {e}")

    def _add_tool_message(self, text: str, label: str = "tool") -> None:
        if self._timeline:
            self._timeline.add_tool(text, label=label)

    # ── Logs ──────────────────────────────────────────────────

    def _cmd_logs(self, args: str) -> None:
        log_type = args.strip().lower()
        if not self.session.task_dir:
            self._add_assistant("No task has run yet. Start with `/run`.")
            return

        td = Path(self.session.task_dir)

        # Backend-aware log paths
        backend = self.session.backend
        build_log = td / "env" / "conda_build.log"
        if backend == "venv":
            build_log = td / "env" / "venv_build.log"
        elif backend in ("docker", "local", "none"):
            build_log = td / "env" / "build.log"

        log_map: dict[str, Path] = {
            "env": build_log,
            "conda": td / "env" / "conda_build.log",
            "venv": td / "env" / "venv_build.log",
            "build": build_log,
            "smoke": td / "runs" / "smoke_001" / "stdout.log",
            "smoke-stderr": td / "runs" / "smoke_001" / "stderr.log",
            "benchmark": td / "runs" / "benchmark_001" / "stderr.log",
            "benchmark-stdout": td / "runs" / "benchmark_001" / "stdout.log",
            "reproduction": td / "runs" / "reproduction_001" / "stderr.log",
            "reproduction-stdout": td / "runs" / "reproduction_001" / "stdout.log",
            "stderr": td / "runs" / "smoke_001" / "stderr.log",
            "stdout": td / "runs" / "smoke_001" / "stdout.log",
        }

        if log_type not in log_map:
            opts = ", ".join(sorted(log_map.keys()))
            self._add_assistant(
                f"Unknown log type `{log_type}`. Available: {opts}\n" f"Usage: `/logs smoke`"
            )
            return

        path = log_map[log_type]
        if not path.exists():
            self._add_assistant(f"Log file not found: `{path}`")
            return

        content = path.read_text(encoding="utf-8", errors="ignore")
        if len(content) > 5000:
            content = content[-5000:] + "\n\n... [dim](last 5000 chars)[/]"
        self._add_tool_message(
            f"**{log_type} log** (`{path}`)\n\n```\n{content}\n```", "log"
        )

    # ── Report ───────────────────────────────────────────────

    def _cmd_report(self, _: str) -> None:
        if not self.session.task_dir:
            self._add_assistant("No task has run yet. Start with `/run`.")
            return
        rp = Path(self.session.task_dir) / "report" / "reproduction_smoke_report.md"
        if not rp.exists():
            self._add_assistant(f"Report not generated yet. Expected path: `{rp}`")
            return
        content = rp.read_text(encoding="utf-8", errors="ignore")
        self._add_report_msg(f"report: `{rp}`\n\n{content[:3000]}")

    # ── Sessions ─────────────────────────────────────────────

    def _cmd_sessions(self, _: str) -> None:
        sessions = self.store.list_sessions()
        if not sessions:
            self._add_assistant("No saved sessions.")
            return

        lines = [
            "| Session ID | Paper | Backend | Status | Created |",
            "|---|---|---|---|---|",
        ]
        for s in sessions[:20]:
            sid = s.get("id", "")[:8]
            paper = Path(s.get("paper_path", "")).name if s.get("paper_path") else "-"
            be = s.get("backend", "-")
            st = s.get("status", "-")
            created = s.get("created_at", "")
            if isinstance(created, (int, float)):
                from datetime import datetime

                created = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M")
            lines.append(f"| {sid} | {paper} | {be} | {st} | {created} |")
        self._add_assistant("\n".join(lines))

    def _cmd_resume(self, args: str) -> None:
        sid = args.strip()
        if not sid:
            self._add_error("Usage: `/resume <session-id>`")
            return
        try:
            other_store = SessionStore(sid)
            data = other_store.load_snapshot()
            if data is None:
                self._add_error(f"Sessions `{sid}` not found.")
                return
            self.session = Session(
                id=data.get("id", sid),
                paper_path=data.get("paper_path"),
                repo=data.get("repo"),
                repo_dir=data.get("repo_dir"),
                backend=data.get("backend", "conda"),
                workspace=data.get("workspace", "./workspace"),
                timeout_minutes=data.get("timeout_minutes", 30),
                max_repair_attempts=data.get("max_repair_attempts", 5),
                status=data.get("status", "draft"),
                mode=data.get("mode", "act"),
                task_dir=data.get("task_dir"),
                report_path=data.get("report_path"),
            )
            self.store = other_store
            self._sync_session_panel()
            self._update_status()
            self._add_assistant(f"Resumed session `{sid}` (status: {self.session.status})")
        except Exception as e:
            self._add_error(f"Resume failed: {e}")

    # ── Conda env management ─────────────────────────────────

    def _cmd_conda_envs(self, args: str) -> None:
        parts = args.strip().split()
        action = parts[0].lower() if parts else "list"

        if action in ("list", "ls"):
            self._cmd_conda_envs_list()
            return
        if action in ("delete", "del", "remove", "rm"):
            self._cmd_conda_envs_delete(" ".join(parts[1:]).strip())
            return
        if action == "prune":
            self._add_assistant(
                "`/conda-envs prune` is not implemented yet.\n\n"
                "Available commands:\n"
                "- `/conda-envs` ViewEnvironment\n"
                "- `/conda-envs delete <n>` delete an environment\n"
                "- `/conda-envs delete --all` delete all project environments"
            )
            return

        self._add_error(
            "Usage: \n"
            "`/conda-envs` or `/envs` — list project conda environments\n"
            "`/conda-envs delete <n|slug>` — delete an environment\n"
            "`/conda-envs delete --all` — delete all project environments"
        )

    def _cmd_conda_envs_list(self) -> None:
        from app.tools.conda_env_manager import (
            discover_project_conda_envs,
            format_env_table,
        )

        envs = discover_project_conda_envs(self.session.workspace)
        if not envs:
            self._add_assistant(
                "No project-created conda environments found under the workspace.\n\n"
                f"Checking directory: `{Path(self.session.workspace).resolve() / 'envs'}`"
            )
            return
        self._add_assistant(
            "## Project conda environments\n\n"
            f"Workspace: `{self.session.workspace}`\n\n"
            + format_env_table(envs)
            + "\n\nDeletion examples:\n"
            "- `/conda-envs delete 1`\n"
            "- `/conda-envs delete <slug>`\n"
            "- `/conda-envs delete --all`"
        )

    def _cmd_conda_envs_delete(self, selector: str) -> None:
        if self.agent_running:
            self._add_error(
                "Agent is running; conda environment deletion is not allowed right now. `/cancel` or wait for completion first."
            )
            return
        if not selector:
            self._add_error(
                "Usage: `/conda-envs delete <n|slug|path>` or `/conda-envs delete --all`"
            )
            return

        from app.tools.conda_env_manager import (
            discover_project_conda_envs,
            find_env_by_selector,
        )

        envs = discover_project_conda_envs(self.session.workspace)

        if selector == "--all":
            if not envs:
                self._add_assistant("No project conda environments to delete.")
                return
            self._pending_env_delete = {"mode": "all", "envs": envs}
            self._add_error(
                f"Confirm deletion of **{len(envs)}** project conda environment(s)?\n"
                "This deletes environments created by this project under `<workspace>/envs/*`.\n"
                "Enter `yes` to confirm, or anything else to cancel."
            )
            return

        env = find_env_by_selector(selector, envs)
        if env is None:
            self._add_error(
                f"Environment not found: `{selector}`. Run `/conda-envs` first to list deletable environments."
            )
            return

        self._pending_env_delete = {"mode": "one", "env": env}
        self._add_error(
            "Confirm deletion of this project conda environment?\n\n"
            f"- Slug: `{env.slug}`\n"
            f"- Path: `{env.path}`\n\n"
            "Enter `yes` to confirm, or anything else to cancel."
        )

    def _confirm_env_delete(self, line: str) -> None:
        pending = self._pending_env_delete
        self._pending_env_delete = None

        if line.strip().lower() not in ("yes", "y"):
            self._add_assistant("Deletion cancelled.")
            return

        from app.tools.conda_env_manager import remove_conda_env

        if pending["mode"] == "one":
            env = pending["env"]
            ok, msg = remove_conda_env(env, workspace=self.session.workspace)
            if ok:
                self._add_assistant(f"Deleted: `{env.slug}` — {msg}")
            else:
                self._add_error(msg)
            return

        if pending["mode"] == "all":
            results = []
            for env in pending["envs"]:
                ok, msg = remove_conda_env(env, workspace=self.session.workspace)
                results.append((env.slug, ok, msg))
            lines = ["## Conda environment deletion results", ""]
            for slug, ok, msg in results:
                mark = "✅" if ok else "❌"
                lines.append(f"- {mark} `{slug}` — {msg}")
            self._add_assistant("\n".join(lines))

    def _cmd_quit(self, _: str) -> None:
        self._cleanup_timers()
        self.exit()

    def action_quit(self) -> None:
        """Override built-in quit to clean up timers first."""
        self._cleanup_timers()
        super().action_quit()

    def _cleanup_timers(self) -> None:
        """Stop heartbeat timer before exit to avoid executor join timeout."""
        if self._heartbeat_timer is not None:
            try:
                self._heartbeat_timer.stop()  # type: ignore[union-attr]
            except Exception:
                pass
            self._heartbeat_timer = None

    # ── Bindings ──────────────────────────────────────────────

    # ── Bindings ──────────────────────────────────────────────

    def action_accept_completion(self) -> None:
        if self._composer and self._composer.accept_completion():
            return

    def action_completion_down(self) -> None:
        if self._composer and self._composer.move_completion(1):
            return

    def action_completion_up(self) -> None:
        if self._composer and self._composer.move_completion(-1):
            return

    def action_hide_completion(self) -> None:
        if self._composer:
            self._composer.hide_completion()

    def action_toggle_plan_mode(self) -> None:
        if self.session.mode == "plan":
            self._cmd_act("")
        else:
            self._cmd_plan("")

    def action_clear_screen(self) -> None:
        self._cmd_clear("")
