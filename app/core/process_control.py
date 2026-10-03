from __future__ import annotations

import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable


class PipelineKilled(RuntimeError):
    """Raised when a running pipeline is force-killed."""
    pass


class ProcessRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._procs: dict[int, subprocess.Popen] = {}
        self._labels: dict[int, str] = {}

    def register(self, proc: subprocess.Popen, *, label: str) -> subprocess.Popen:
        with self._lock:
            self._procs[proc.pid] = proc
            self._labels[proc.pid] = label
        return proc

    def unregister(self, proc: subprocess.Popen | int) -> None:
        pid = proc if isinstance(proc, int) else proc.pid
        with self._lock:
            self._procs.pop(pid, None)
            self._labels.pop(pid, None)

    def list_labels(self) -> list[str]:
        with self._lock:
            return [f"{label} pid={pid}" for pid, label in self._labels.items()]

    def kill_all(self) -> list[str]:
        killed: list[str] = []
        with self._lock:
            procs = list(self._procs.items())

        for pid, proc in procs:
            if proc.poll() is not None:
                self.unregister(proc)
                continue
            label = self._labels.get(pid, "unknown")
            try:
                kill_process_tree(pid)
                killed.append(f"{label} pid={pid}")
            except Exception as e:
                try:
                    proc.kill()
                    killed.append(f"{label} pid={pid}")
                except Exception as e2:
                    killed.append(f"failed {label} pid={pid}: {e2}")
            self.unregister(proc)
        return killed


_registry = ProcessRegistry()


def get_process_registry() -> ProcessRegistry:
    return _registry


def managed_popen(argv: list[str], *, label: str, cwd: str | Path | None = None, **kwargs) -> subprocess.Popen:
    popen_kwargs = dict(kwargs)
    if sys.platform == "win32":
        popen_kwargs.setdefault("creationflags", subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        popen_kwargs.setdefault("start_new_session", True)
    proc = subprocess.Popen(argv, cwd=cwd, **popen_kwargs)
    get_process_registry().register(proc, label=label)
    return proc


def kill_process_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        import time
        time.sleep(1)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

import os  # noqa: E402 — used in kill_process_tree POSIX branch
