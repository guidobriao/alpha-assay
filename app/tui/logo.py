from __future__ import annotations

"""ASCII logo with rich gradient rendering — no pyfiglet dependency."""

from rich.text import Text

paper = r"""
██████╗   █████╗  ██████╗  ███████╗ ██████╗
██╔══██╗  ╚══██╗ ██╔══██╗ ██╔════╝ ██╔══██╗
██████╔╝  █████║ ██████╔╝ █████╗   ██║  ╚═╝
██╔═══╝  ██╔═██║ ██╔═══╝  ██╔══╝   ██║
██║      ╚█████║ ██║      ╚██████╗ ██║
╚═╝       ╚════╝ ╚═╝       ╚═════╝ ╚═╝
""".strip("\n").splitlines()

reproduct = r"""
██████╗  ███████╗ ██████╗  ██████╗   ██████╗  ██████╗  ██╗   ██╗  ██████╗ ████████╗
██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗ ██╔═══██╗ ██╔══██╗ ██║   ██║ ██╔════╝ ╚══██╔══╝
██████╔╝ █████╗   ██████╔╝ ██║  ╚═╝ ██║   ██║ ██║  ██║ ██║   ██║ ██║         ██║
██╔══██╗ ██╔══╝   ██╔═══╝  ██║      ██║   ██║ ██║  ██║ ██║   ██║ ██║         ██║
██║  ██║ ╚██████╗ ██║      ██║      ╚██████╔╝ ██████╔╝ ╚██████╔╝ ╚██████╗    ██║
╚═╝  ╚═╝  ╚═════╝ ╚═╝      ╚═╝       ╚═════╝  ╚═════╝   ╚═════╝   ╚═════╝    ╚═╝
""".strip("\n").splitlines()

agent = r"""
 █████╗   ██████╗  ███████╗ ███╗   ██╗ ████████╗
██╔══██╗ ██╔════╝  ██╔════╝ ████╗  ██║ ╚══██╔══╝
███████║ ██║  ███╗ █████╗   ██╔██╗ ██║    ██║
██╔══██║ ██║   ██║ ██╔══╝   ██║╚██╗██║    ██║
██║  ██║ ╚██████╔╝ ███████╗ ██║ ╚████║    ██║
╚═╝  ╚═╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═══╝    ╚═╝
""".strip("\n").splitlines()

GAP = " " * 10

GRADIENT = [
    (255, 238, 80),
    (255, 218, 55),
    (255, 185, 55),
    (245, 135, 95),
    (190, 110, 190),
    (100, 155, 255),
]

SUBTITLE = "Alpha-Assay — Replication & Out-of-Sample Validation"


def _rgb_to_rich(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def _gradient_color(position: float) -> str:
    """Return a hex colour from the gradient for position 0.0–1.0."""
    if len(GRADIENT) == 1:
        return _rgb_to_rich(*GRADIENT[0])
    segments = len(GRADIENT) - 1
    idx = min(int(position * segments), segments - 1)
    local = (position * segments) - idx
    c = _lerp(GRADIENT[idx], GRADIENT[idx + 1], local)
    return _rgb_to_rich(*c)


def build_logo_lines() -> list[str]:
    """Join the three ASCII art panels side-by-side with a gap."""
    height = max(len(paper), len(reproduct), len(agent))
    left_w = max(len(l) for l in paper)
    mid_w = max(len(l) for l in reproduct)
    right_w = max(len(l) for l in agent)

    result: list[str] = []
    for i in range(height):
        lp = paper[i] if i < len(paper) else " " * left_w
        rp = reproduct[i] if i < len(reproduct) else " " * mid_w
        ap = agent[i] if i < len(agent) else " " * right_w
        result.append(f"{lp:<{left_w}}{GAP}{rp:<{mid_w}}{GAP}{ap:<{right_w}}")
    return result


def gradient_text(line: str, width: int) -> Text:
    """Render a single line of the logo with horizontal gradient."""
    if not line.strip():
        return Text(" " * width)
    text = Text()
    for i, ch in enumerate(line):
        pos = i / max(width - 1, 1)
        color = _gradient_color(pos)
        text.append(ch, style=color)
    return text


def render_logo(max_width: int | None = None) -> list[Text]:
    """Return full Rich Text logo, or compact fallback if narrow.

    If *max_width* is provided and the trilingual logo exceeds it, a
    compact single-line title is returned instead.
    If *max_width* is wider than the logo, each line is padded with
    leading spaces so the logo appears centered.
    """
    lines = build_logo_lines()
    if not lines:
        return [Text("Paper Reproduct Agent", style="bold #bd93f9")]

    full_width = max(len(l) for l in lines)
    if max_width is not None and full_width > max_width:
        return [render_compact_logo(max_width=max_width)]

    target_width = max_width if max_width is not None and max_width > full_width else full_width

    result: list[Text] = []
    for line in lines:
        pad = max((target_width - len(line)) // 2, 0)
        padded = " " * pad + line
        result.append(gradient_text(padded, target_width))
    return result


def render_compact_logo(max_width: int | None = None) -> Text:
    """Single-line logo for narrow terminals.  Centered when max_width is given."""
    words = ["Paper", "Reproduct", "Agent"]

    # Build raw text first to measure width
    raw = "  ".join(words)

    if max_width is not None and max_width > len(raw):
        pad = max((max_width - len(raw)) // 2, 0)
    else:
        pad = 0

    text = Text()
    if pad:
        text.append(" " * pad)

    for i, w in enumerate(words):
        pos = i / max(len(words) - 1, 1)
        color = _gradient_color(pos)
        text.append(w, style=f"bold {color}")
        if i < len(words) - 1:
            text.append("  ", style="")
    return text
