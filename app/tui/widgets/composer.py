from __future__ import annotations

"""Composer with slash-command completion popup — Enter accepts and may execute."""

from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, Static

from ..completion import complete_command, CompletionItem
from .. import theme as T


class Composer(Widget):
    """Bottom input area with mode-aware placeholder and slash-command completion."""

    DEFAULT_CSS = """
    Composer {
        height: 7;
        min-height: 7;
        padding: 1 1 2 1;
        background: #000000;
        border-top: solid $primary-darken-2;
    }
    Composer.has-completion {
        height: 11;
        min-height: 11;
    }
    Composer Input {
        height: 3;
        min-height: 3;
        width: 100%;
        content-align: left middle;
        margin-bottom: 1;
    }
    #completion-popup {
        height: 4;
        max-height: 4;
        display: none;
        color: $text-muted;
        padding: 0 0 0 2;
    }
    #completion-popup.visible {
        display: block;
    }
    """

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def __init__(
        self,
        placeholder: str | None = None,
        mode: str = "act",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._mode = mode
        self._running = False
        self._default = placeholder or "输入 PDF 路径，或 /help 查看命令"
        self._placeholder = self._default

        self._completion_items: list[CompletionItem] = []
        self._completion_index: int = 0
        self._completion_visible: bool = False
        self._completion_scroll: int = 0
        self._visible_count: int = 4

    def compose(self):
        yield Static("", id="completion-popup")
        yield Input(
            placeholder=self._placeholder,
            id="composer-input",
        )

    # ── Input handling ──────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # If completion is visible, accept the selected item
        if self.has_completion():
            accepted, submit_text = self._accept_completion(submit_if_complete=True)
            if accepted:
                if submit_text:
                    event.input.value = ""
                    self.post_message(self.Submitted(submit_text))
                return

        value = event.value.strip()
        if not value:
            return
        event.input.value = ""
        self._hide_completion()
        self.post_message(self.Submitted(value))

    def on_input_changed(self, event: Input.Changed) -> None:
        value = event.value
        if value.startswith("/") and " " not in value:
            items = complete_command(value)
            if items:
                self._show_completion(items)
            else:
                self._hide_completion()
        else:
            self._hide_completion()

    # ── Completion display ──────────────────────────────────

    def _show_completion(self, items: list[CompletionItem]) -> None:
        self._completion_items = items
        self._completion_index = 0
        self._completion_scroll = 0
        self._completion_visible = True
        self._refresh_completion_display()
        self.add_class("has-completion")

    def _hide_completion(self) -> None:
        self._completion_items.clear()
        self._completion_index = 0
        self._completion_scroll = 0
        self._completion_visible = False
        self.remove_class("has-completion")
        try:
            popup = self.query_one("#completion-popup", Static)
            popup.remove_class("visible")
            popup.update("")
        except Exception:
            pass

    def hide_completion(self) -> None:
        self._hide_completion()

    def _current_item(self) -> CompletionItem | None:
        if not self._completion_items:
            return None
        if 0 <= self._completion_index < len(self._completion_items):
            return self._completion_items[self._completion_index]
        return None

    def _refresh_completion_display(self) -> None:
        try:
            popup = self.query_one("#completion-popup", Static)
        except Exception:
            return

        if not self._completion_items:
            popup.remove_class("visible")
            popup.update("")
            return

        popup.add_class("visible")
        n = len(self._completion_items)
        self._completion_scroll = _ensure_visible(
            self._completion_index, self._completion_scroll, n, self._visible_count
        )

        start = self._completion_scroll
        end = min(n, start + self._visible_count)
        lines: list[str] = []
        for i in range(start, end):
            item = self._completion_items[i]
            prefix = ">" if i == self._completion_index else " "
            color = T.GREEN if i == self._completion_index else T.FG_DIM
            arg_part = (
                f" {item.display_args or item.args}"
                if (item.display_args or item.args)
                else ""
            )
            desc = item.description[:50]
            lines.append(
                f"[{color}]{prefix} /{item.command}{arg_part:<20}[/] [{T.FG_DIM}]{desc}[/]"
            )
        if n > self._visible_count:
            lines.append(f"[dim]{start + 1}-{end} / {n}[/]")
        popup.update("\n".join(lines))

    def move_completion(self, delta: int) -> bool:
        if not self._completion_items:
            return False
        n = len(self._completion_items)
        self._completion_index = max(0, min(n - 1, self._completion_index + delta))
        self._refresh_completion_display()
        return True

    # ── Completion navigation ───────────────────────────────

    def has_completion(self) -> bool:
        return self._completion_visible and len(self._completion_items) > 0

    def _accept_completion(
        self, submit_if_complete: bool = False
    ) -> tuple[bool, str | None]:
        """Accept current completion.

        Returns (accepted, submit_text).
        - accepted: whether a completion was applied
        - submit_text: if non-None, post this as a Submitted message
        """
        item = self._current_item()
        if item is None:
            return False, None

        try:
            inp = self.query_one("#composer-input", Input)
            inp.value = item.insert_text
            inp.action_end()
            inp.focus()
        except Exception:
            pass

        self._hide_completion()

        if submit_if_complete and not item.has_any_args:
            return True, item.insert_text.strip()

        return True, None

    def accept_completion(self) -> bool:
        """Tab: accept completion without executing."""
        accepted, _ = self._accept_completion(submit_if_complete=False)
        return accepted

    def move_completion(self, delta: int) -> bool:
        if not self.has_completion():
            return False
        self._completion_index += delta
        self._completion_index %= len(self._completion_items)
        self._refresh_completion_display()
        return True

    # ── Focus / placeholder ─────────────────────────────────

    def focus_input(self) -> None:
        try:
            input_widget = self.query_one("#composer-input", Input)
            input_widget.focus()
        except Exception:
            pass

    def set_placeholder_text(self, text: str) -> None:
        self._default = text
        if not self._running:
            self._placeholder = text
            self._update_placeholder()

    def set_running(self, running: bool) -> None:
        self._running = running
        if running:
            self._placeholder = "Agent 运行中 · 可用 /status /logs /cancel"
        else:
            self._placeholder = self._default
        self._update_placeholder()

    def set_mode(self, mode: str) -> None:
        self._mode = mode.lower()
        if not self._running:
            if self._mode == "plan":
                self._placeholder = "PLAN 模式：输入 /act 切换执行，或 /run 查看计划"
            else:
                self._placeholder = self._default
        self._update_placeholder()

    def _update_placeholder(self) -> None:
        try:
            input_widget = self.query_one("#composer-input", Input)
            input_widget.placeholder = self._placeholder
        except Exception:
            pass

    def set_disabled(self, disabled: bool) -> None:
        self.set_running(disabled)


def _ensure_visible(selected: int, scroll: int, total: int, visible: int) -> int:
    """Adjust scroll so that *selected* index is within the visible window."""
    if total <= visible:
        return 0
    if selected < scroll:
        return selected
    if selected >= scroll + visible:
        return selected - visible + 1
    return scroll
