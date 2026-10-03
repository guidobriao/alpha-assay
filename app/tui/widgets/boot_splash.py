"""Boot splash screen with spinning logo and preflight checks."""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Static

from .. import theme as T
from ..logo_loader import load_logo_frames, FPS_INTERVAL, CAS_BLUE
from ..preflight import CheckItem, run_preflight


SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠴", "⠦", "⠧", "⠇")


class BootSplash(Screen):
    """Full-screen preflight check display with spinning logo."""

    DEFAULT_CSS = """
    BootSplash {
        align: center middle;
        background: #000000;
    }
    BootSplash > Vertical {
        background: #000000;
    }
    BootSplash #splash-logo {
        height: auto;
        content-align: center middle;
        text-wrap: nowrap;
        overflow: hidden;
        margin-bottom: 1;
        background: #000000;
    }
    BootSplash #splash-checks {
        height: auto;
        content-align: center middle;
        color: #888888;
        background: #000000;
    }
    BootSplash #splash-status {
        height: 1;
        content-align: center middle;
        color: #888888;
        margin-top: 1;
        background: #000000;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit_app", "退出", show=False),
        Binding("q", "quit_app", "退出", show=False),
        Binding("escape", "skip_splash", "跳过", show=False),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._checks: list[CheckItem] = []
        self._done = False
        self._cancelled = False
        self._frames: list = []
        self._frame_idx = 0
        self._spinner_idx = 0
        self._playing = False
        self._logo_timer = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("", id="splash-logo")
            yield Static("", id="splash-checks")
            yield Static("", id="splash-status")

    def on_mount(self) -> None:
        try:
            available_width = max(60, self.size.width - 4)
        except Exception:
            available_width = 100
        try:
            self._frames = load_logo_frames(max_width=available_width)
        except Exception:
            self._frames = []
        if self._frames:
            self.query_one("#splash-logo", Static).update(self._frames[0])
            self._playing = True
            self._logo_timer = self.set_interval(FPS_INTERVAL, self._next_frame)
        self.run_worker(self._run_checks(), exclusive=True)

    def on_unmount(self) -> None:
        self._playing = False
        if self._logo_timer is not None:
            self._logo_timer.stop()
            self._logo_timer = None

    def _next_frame(self) -> None:
        if not self._playing or not self._frames:
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(SPINNER_FRAMES)
        has_running = any(c.status == "running" for c in self._checks)
        try:
            self._frame_idx = (self._frame_idx + 1) % len(self._frames)
            self.query_one("#splash-logo", Static).update(self._frames[self._frame_idx])
            if has_running:
                self._refresh_checks(self.query_one("#splash-checks", Static))
        except Exception:
            self._playing = False

    def action_quit_app(self) -> None:
        self._playing = False
        self.app.exit()

    def action_skip_splash(self) -> None:
        self._cancelled = True
        self._playing = False
        self.dismiss([])

    async def _run_checks(self) -> None:
        status = self.query_one("#splash-status", Static)
        checks = self.query_one("#splash-checks", Static)

        status.update("正在检查系统环境……")

        results = await asyncio.to_thread(run_preflight)

        if self._cancelled:
            return

        animated: list[CheckItem] = [
            CheckItem(
                name=item.name, status="pending", message="", blocking=item.blocking
            )
            for item in results
        ]
        self._checks = animated
        self._refresh_checks(checks)

        for i, final_item in enumerate(results):
            if self._cancelled:
                return

            animated[i].status = "running"
            animated[i].message = "正在检查……"
            self._refresh_checks(checks)
            await asyncio.sleep(0.2)

            animated[i].status = final_item.status
            animated[i].message = final_item.message
            animated[i].blocking = final_item.blocking
            self._refresh_checks(checks)

        if self._cancelled:
            return

        passes = sum(1 for c in results if c.status == "pass")
        fails = sum(1 for c in results if c.status == "fail")
        blocking = sum(1 for c in results if c.status == "fail" and c.blocking)

        if blocking > 0:
            status.update(
                f"[{T.ERROR_BORDER}]检查完成：{passes} 通过，{fails} 失败"
                f"（{blocking} 项阻塞，将进入 TUI 显示警告）[/]"
            )
        elif fails > 0:
            status.update(
                f"[{T.WARNING_BORDER}]检查完成：{passes} 通过，{fails} 项非阻塞警告[/]"
            )
        else:
            status.update(f"[{T.GREEN}]全部检查通过 ✓[/]")

        await asyncio.sleep(1.0)

        if self._cancelled:
            return

        self._playing = False
        status.update(f"[{T.FG_DIM}]正在进入 TUI…[/]")
        await asyncio.sleep(0.2)

        self._done = True
        self.dismiss(results)

    def _refresh_checks(self, widget: Static) -> None:
        lines: list[str] = []
        left_w = 24
        spinner = SPINNER_FRAMES[self._spinner_idx]
        CAS_BLUE_HEX = f"#{CAS_BLUE[0]:02x}{CAS_BLUE[1]:02x}{CAS_BLUE[2]:02x}"
        for item in self._checks:
            if item.status == "running":
                icon = spinner
            else:
                icon = {"pending": "○", "pass": "✓", "fail": "✗"}.get(item.status, "?")
            color = {
                "pending": "#ffffff",
                "running": "#8be9fd",
                "pass": CAS_BLUE_HEX,
                "fail": T.ERROR_BORDER,
            }.get(item.status, T.FG)
            left = f"{icon} {item.name}"
            msg = item.message[:52] if item.message else ""
            lines.append(f"[{color}]{left:<{left_w}}[/][{T.FG_DIM}]{msg}[/]")
        widget.update("\n".join(lines))
