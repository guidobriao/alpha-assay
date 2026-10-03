from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    message: str
    level: str = "info"
    phase: str = "progress"  # "start", "progress", "finish", "fail"
    detail: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


ProgressHandler = Callable[[ProgressEvent], None]

_current_handler: ContextVar[ProgressHandler | None] = ContextVar(
    "paper_smoke_progress_handler",
    default=None,
)


@contextmanager
def progress_events(handler: ProgressHandler | None) -> Iterator[None]:
    token = _current_handler.set(handler)
    try:
        yield
    finally:
        _current_handler.reset(token)


def emit_progress(
    stage: str,
    message: str,
    *,
    level: str = "info",
    phase: str = "progress",
    detail: str | None = None,
    **data: Any,
) -> None:
    handler = _current_handler.get()
    if handler is None:
        return
    handler(ProgressEvent(stage=stage, message=message, level=level, phase=phase, detail=detail, data=data))
