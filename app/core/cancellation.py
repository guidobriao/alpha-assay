from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

CancelCheck = Callable[[], bool]

_current_cancel_check: ContextVar[CancelCheck | None] = ContextVar(
    "paper_smoke_cancel_check", default=None
)

_current_force_kill_check: ContextVar[CancelCheck | None] = ContextVar(
    "paper_smoke_force_kill_check", default=None
)


class PipelineCancelled(RuntimeError):
    """Raised when the pipeline is cancelled mid-execution."""
    pass


@contextmanager
def cancellation(check: CancelCheck | None) -> Iterator[None]:
    token = _current_cancel_check.set(check)
    try:
        yield
    finally:
        _current_cancel_check.reset(token)


@contextmanager
def force_cancellation(kill_check: CancelCheck | None) -> Iterator[None]:
    token = _current_force_kill_check.set(kill_check)
    try:
        yield
    finally:
        _current_force_kill_check.reset(token)


@contextmanager
def full_cancellation(check: CancelCheck | None, kill_check: CancelCheck | None = None) -> Iterator[None]:
    t1 = _current_cancel_check.set(check)
    t2 = _current_force_kill_check.set(kill_check)
    try:
        yield
    finally:
        _current_cancel_check.reset(t1)
        _current_force_kill_check.reset(t2)


def is_cancelled() -> bool:
    check = _current_cancel_check.get()
    if check is None:
        return False
    try:
        return bool(check())
    except Exception:
        return False


def raise_if_cancelled() -> None:
    if is_cancelled():
        raise PipelineCancelled("cancelled")


def is_force_killed() -> bool:
    check = _current_force_kill_check.get()
    if check is None:
        return False
    try:
        return bool(check())
    except Exception:
        return False


def raise_if_force_killed() -> None:
    if is_force_killed():
        from app.core.process_control import PipelineKilled
        raise PipelineKilled("force killed")
