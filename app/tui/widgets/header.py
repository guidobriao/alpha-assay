from __future__ import annotations

"""Gradient logo header with session summary."""

from rich.text import Text as RichText
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from ..logo import render_logo, render_compact_logo, SUBTITLE
from .. import theme as T
import time


def _status_cn_short(status: str) -> str:
    mapping = {
        "draft": "草稿",
        "running": "运行中",
        "success": "成功",
        "failed": "失败",
        "cancelled": "已取消",
        "completed": "已完成",
    }
    return mapping.get(status.lower(), status)


def join_rich_lines(lines: list[RichText]) -> RichText:
    """Join a list of Rich Text lines with newlines, preserving style spans."""
    result = RichText()
    for i, line in enumerate(lines):
        result.append_text(line)
        if i < len(lines) - 1:
            result.append("\n")
    return result


class HeaderLogo(Widget):
    """Top header: gradient ASCII logo, subtitle, and session summary."""

    DEFAULT_CSS = """
    HeaderLogo {
        height: auto;
        max-height: 8;
        padding: 0 1;
        background: #000000;
        border-bottom: solid $primary-darken-2;
    }
    HeaderLogo #logo-area {
        height: auto;
    }
    HeaderLogo #subtitle-area {
        height: auto;
        padding: 0 0 0 0;
    }
    HeaderLogo #summary-area {
        height: 1;
        color: $text-muted;
        padding: 0 0 0 0;
    }
    """

    def __init__(
        self,
        session_id: str = "",
        backend: str = "conda",
        mode: str = "ACT",
        status: str = "draft",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.session_id = session_id
        self.backend = backend
        self.mode = mode
        self.status = status
        self._last_render_width: int | None = None
        self._last_render_at: float = 0.0

    def update_summary(
        self,
        session_id: str = "",
        backend: str = "",
        mode: str = "",
        status: str = "",
    ) -> None:
        if session_id:
            self.session_id = session_id
        if backend:
            self.backend = backend
        if mode:
            self.mode = mode
        if status:
            self.status = status
        self._refresh_summary()

    def compose(self) -> ComposeResult:
        yield Static("", id="logo-area")
        yield Static("", id="subtitle-area")
        yield Static("", id="summary-area")

    def on_mount(self) -> None:
        self._render_and_display()

    def _render_and_display(self) -> None:
        """Render logo based on available width.  Pass RichText directly to preserve colour spans."""
        width = self.size.width if self.size and self.size.width > 20 else None
        if width == self._last_render_width:
            # Only refresh summary; logo already rendered for this width
            self._refresh_summary()
            return
        self._last_render_width = width
        try:
            logo_texts = render_logo(max_width=width)
        except Exception:
            logo_texts = [render_compact_logo(max_width=width)]

        # Use Rich renderable directly — do NOT convert to str()
        logo_renderable = join_rich_lines(logo_texts)
        try:
            self.query_one("#logo-area", Static).update(logo_renderable)
        except Exception:
            pass

        subtitle = f"[dim italic]{SUBTITLE}[/]"
        try:
            self.query_one("#subtitle-area", Static).update(subtitle)
        except Exception:
            pass

        self._refresh_summary()

    def _refresh_summary(self) -> None:
        mode_c = T.PURPLE if self.mode.upper() == "PLAN" else T.GREEN
        status_c = T.status_color(self.status)
        mode_label = "计划" if self.mode.upper() == "PLAN" else "执行"
        status_label = _status_cn_short(self.status)
        text = (
            f"[{T.FG_DIM}]会话：[/] [{T.INFO_BORDER}]{self.session_id[:8]}[/]  "
            f"[{T.FG_DIM}]模式：[/] [bold {mode_c}]{mode_label}[/]  "
            f"[{T.FG_DIM}]后端：[/] [{T.INFO_BORDER}]{self.backend}[/]  "
            f"[{T.FG_DIM}]状态：[/] [{status_c}]{status_label}[/]"
        )
        try:
            self.query_one("#summary-area", Static).update(text)
        except Exception:
            pass

    def on_resize(self) -> None:
        width = self.size.width if self.size and self.size.width > 20 else None
        if width == self._last_render_width:
            return
        now = time.monotonic()
        if now - self._last_render_at < 0.25:
            return
        self._last_render_at = now
        self._render_and_display()
