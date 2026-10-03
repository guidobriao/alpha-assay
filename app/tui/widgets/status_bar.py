from __future__ import annotations

"""Enhanced status bar with session info, error summary, and panel hint."""

from textual.widget import Widget
from textual.widgets import Static

from .. import theme as T


def _status_short_cn(status: str) -> str:
    mapping = {
        "draft": "draft",
        "running": "running",
        "success": "success",
        "failed": "failed",
        "cancelled": "cancelled",
        "completed": "completed",
    }
    return mapping.get(status.lower(), status)


def _panel_cn(panel: str) -> str:
    mapping = {
        "pipeline": "Pipeline",
        "help": "Help",
        "artifacts": "artifact",
        "session": "Sessions",
        "none": "None",
    }
    return mapping.get(panel, panel)


class StatusBar(Widget):
    """Bottom status bar with rich session & error info."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        min-height: 1;
        width: 100%;
        background: #000000;
        color: $text-muted;
        padding: 0 1;
        border-top: solid $primary-darken-2;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._mode = "ACT"
        self._session = ""
        self._backend = "conda"
        self._status = "draft"
        self._paper = "-"
        self._last_error = ""
        self._panel = ""

    def update_info(
        self,
        mode: str | None = None,
        session: str | None = None,
        backend: str | None = None,
        status: str | None = None,
        paper: str | None = None,
        last_error: str | None = None,
        panel: str | None = None,
    ) -> None:
        if mode is not None:
            self._mode = mode.upper()
        if session is not None:
            self._session = session[:8] if session else ""
        if backend is not None:
            self._backend = backend
        if status is not None:
            self._status = status
        if paper is not None:
            self._paper = paper
        if last_error is not None:
            self._last_error = last_error[:60] if last_error else ""
        if panel is not None:
            self._panel = panel
        self._refresh()

    def _refresh(self) -> None:
        mode_label = "PLAN mode" if self._mode == "PLAN" else "ACT mode"
        mode_c = T.PURPLE if self._mode == "PLAN" else T.GREEN
        status_c = T.status_color(self._status)
        status_label = _status_short_cn(self._status)
        panel_label = _panel_cn(self._panel)

        parts: list[str] = []
        parts.append(f" [{mode_c}]{mode_label}[/]")
        if self._session:
            parts.append(f"[{T.FG_DIM}]│ Session {self._session}[/]")
        parts.append(f"[{T.FG_DIM}]│[/] Backend [{T.INFO_BORDER}]{self._backend}[/]")
        parts.append(f"[{T.FG_DIM}]│[/] [{status_c}]{status_label}[/]")
        if self._panel and self._panel != "pipeline":
            parts.append(f"[{T.FG_DIM}]│ Panel {panel_label}[/]")
        if self._paper and self._paper != "-":
            parts.append(f"[{T.FG_DIM}]│ {self._paper[:12]}[/]")
        if self._last_error:
            parts.append(
                f"[{T.FG_DIM}]│[/] [{T.ERROR_BORDER}]{self._last_error[:20]}[/]"
            )

        parts.append(f"[{T.FG_DIM}] │ Ctrl+P toggle mode │ Ctrl+L clear │ Ctrl+C quit[/]")
        text = "".join(parts)
        try:
            self.query_one(Static).update(text)
        except Exception:
            pass

    def compose(self):
        yield Static("")
        self._refresh()
