from __future__ import annotations

import re

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

_RICH_TAG_RE = re.compile(
    r"\[/?(?:bold|dim|italic|underline|reverse|blink|#[0-9a-fA-F]{6}|[a-zA-Z_ -]+)(?: [^\]]+)?\]"
)


def clean_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def clean_display_text(text: str) -> str:
    """Clean Markdown/Rich artifacts for terminal plain display."""
    if not text or not isinstance(text, str):
        return ""
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = _RICH_TAG_RE.sub("", text)
    return text.strip()


def clean_cli_line(line: str) -> str:
    """Clean a CLI output line: strip carriage-return overwrites, ANSI, and trim."""
    line = clean_ansi(line)
    if "\r" in line:
        line = line.split("\r")[-1]
    return line.strip()
