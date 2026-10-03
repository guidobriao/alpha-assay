from __future__ import annotations

"""Enhanced scrollable message timeline with labels and Markdown."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Markdown

from .. import theme as T


class MessageBubble(Widget):
    """A single message in the timeline with colour-coded label."""

    DEFAULT_CSS = """
    MessageBubble {
        margin: 0 0 0 0;
        padding: 0 1;
        height: auto;
    }
    MessageBubble.user {
        border-left: thick $success;
    }
    MessageBubble.assistant {
        border-left: thick $primary;
    }
    MessageBubble.tool {
        border-left: thick #585b70;
    }
    MessageBubble.error {
        border-left: thick $error;
        border: solid $error;
    }
    MessageBubble.system {
        border-left: thick $warning;
    }
    MessageBubble.report {
        border-left: thick #f1fa8c;
    }
    """

    def __init__(
        self,
        text: str,
        kind: str = "assistant",
        label: str = "",
        **kwargs: object,
    ) -> None:
        super().__init__(classes=kind, **kwargs)
        self._text = text
        self._kind = kind
        self._label = label

    def compose(self) -> ComposeResult:
        color_map = {
            "user": T.USER_BORDER,
            "assistant": T.AGENT_BORDER,
            "tool": T.TOOL_BORDER,
            "error": T.ERROR_BORDER,
            "system": T.WARNING_BORDER,
            "report": T.REPORT_COLOR,
        }
        color = color_map.get(self._kind, T.FG)
        kind_names = {
            "user": "用户",
            "assistant": "Agent",
            "tool": "工具",
            "error": "错误",
            "system": "系统",
            "report": "报告",
        }
        label_display = self._label or kind_names.get(self._kind, self._kind)
        label_padded = f"[bold {color}]{label_display:<8}[/]"

        # Truncate very long text for non-report messages
        display_text = self._text
        if self._kind not in ("report",) and len(display_text) > 3000:
            display_text = display_text[:3000] + "\n\n... [dim](truncated — use /report or /logs for full content)[/]"

        if self._kind in ("assistant", "report") and _has_markdown(display_text):
            yield Static(label_padded)
            yield Markdown(display_text)
        else:
            content = f"{label_padded} {display_text}"
            yield Static(content)


def _has_markdown(text: str) -> bool:
    if any(line.startswith("#") for line in text.splitlines()):
        return True
    if any(marker in text for marker in ("**", "`", "```")):
        return True
    if any(line.strip().startswith(("- ", "* ")) for line in text.splitlines()):
        return True
    if "|" in text and "---" in text:
        return True
    return False


class MessageTimeline(VerticalScroll):
    """Scrollable list of message bubbles."""

    DEFAULT_CSS = """
    MessageTimeline {
        height: 1fr;
        scrollbar-size: 1 1;
        padding: 0 1;
    }
    """

    class ComposerFocused(Message):
        pass

    def add_message(
        self,
        text: str,
        kind: str = "assistant",
        label: str = "",
    ) -> MessageBubble:
        bubble = MessageBubble(text=text, kind=kind, label=label)
        self.mount(bubble)
        self.scroll_end(animate=False)
        return bubble

    def add_user(self, text: str) -> MessageBubble:
        return self.add_message(text, kind="user")

    def add_assistant(self, text: str) -> MessageBubble:
        return self.add_message(text, kind="assistant")

    def add_tool(self, text: str, label: str = "工具") -> MessageBubble:
        return self.add_message(text, kind="tool", label=label)

    def add_error(self, text: str) -> MessageBubble:
        return self.add_message(text, kind="error")

    def add_system(self, text: str) -> MessageBubble:
        return self.add_message(text, kind="system")

    def add_report(self, text: str) -> MessageBubble:
        return self.add_message(text, kind="report")

    def clear_messages(self) -> None:
        for child in list(self.children):
            child.remove()
