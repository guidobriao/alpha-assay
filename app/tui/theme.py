from __future__ import annotations

"""Dracula-inspired colour palette for the TUI."""

# ── Accent colours ──────────────────────────────────────────

GREEN = "#50fa7b"
PURPLE = "#bd93f9"
BLUE = "#8be9fd"
RED = "#ff5555"
ORANGE = "#ffb86c"
YELLOW = "#f1fa8c"
PINK = "#ff79c6"

# ── Foreground ──────────────────────────────────────────────

FG = "#cdd6f4"
FG_DIM = "#6c7086"
FG_MUTED = "#585b70"

# ── Background ──────────────────────────────────────────────

BG = "#000000"
BG_SURFACE = "#000000"
BG_DARKER = "#000000"
BG_HOVER = "#000000"

# ── Semantic aliases ────────────────────────────────────────

USER_BORDER = GREEN
AGENT_BORDER = PURPLE
TOOL_BORDER = FG_MUTED
ERROR_BORDER = RED
SUCCESS_BORDER = GREEN
WARNING_BORDER = ORANGE
INFO_BORDER = BLUE
REPORT_COLOR = YELLOW

# ── Stage / status colours ──────────────────────────────────

STAGE_COLORS: dict[str, str] = {
    "queued": FG_DIM,
    "running": BLUE,
    "success": GREEN,
    "failed": RED,
    "skipped": FG_MUTED,
    "cancelled": ORANGE,
    "disabled": FG_MUTED,
}

STAGE_ICONS: dict[str, str] = {
    "queued": "○",
    "running": "●",
    "success": "✓",
    "failed": "✗",
    "skipped": "–",
    "cancelled": "–",
    "disabled": "–",
}

STATUS_COLORS: dict[str, str] = {
    "running": BLUE,
    "success": GREEN,
    "benchmark_success": GREEN,
    "reproduction_success": GREEN,
    "failed": RED,
    "error": RED,
    "cancelled": ORANGE,
    "skipped": ORANGE,
    "plan": PURPLE,
    "act": GREEN,
    "draft": FG_DIM,
    "warning": ORANGE,
    "partial_success_help_only": YELLOW,
}


def status_color(status: str) -> str:
    """Return a colour for a pipeline / session status string."""
    return STATUS_COLORS.get(status, FG_DIM)


def stage_icon(status: str) -> str:
    """Return a single-character icon for a stage status."""
    return STAGE_ICONS.get(status, "○")


def stage_color(status: str) -> str:
    """Return a colour for a stage status."""
    return STAGE_COLORS.get(status, FG_DIM)
