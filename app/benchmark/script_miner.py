from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScriptSignal:
    path: str
    score: int
    reasons: tuple[str, ...]


def mine_script_signals(repo_dir: Path, candidate_scripts: list[str]) -> list[ScriptSignal]:
    scripts = set(candidate_scripts)
    for pattern in ("benchmark*.py", "eval*.py", "evaluate*.py", "test*.py", "demo*.py", "example*.py", "minimal_example.py", "match*.py"):
        for path in repo_dir.rglob(pattern):
            if _ignored(path, repo_dir):
                continue
            scripts.add(path.relative_to(repo_dir).as_posix())

    signals: list[ScriptSignal] = []
    for script in scripts:
        path = repo_dir / script
        if not path.exists() or not path.is_file():
            continue
        score, reasons = _score_script(path)
        signals.append(ScriptSignal(path=script, score=score, reasons=tuple(reasons)))
    return sorted(signals, key=lambda item: (-item.score, item.path))


def _score_script(path: Path) -> tuple[int, list[str]]:
    name = path.name.lower()
    stem = path.stem.lower()
    score = 0
    reasons: list[str] = []

    weights = [
        ("benchmark", 5),
        ("eval", 5),
        ("evaluate", 5),
        ("test", 2),
        ("demo", 1),
        ("example", 1),
        ("match", 4),
    ]
    for token, weight in weights:
        if token in stem:
            score += weight
            reasons.append(f"name_contains:{token}")

    text = path.read_text(encoding="utf-8", errors="ignore")[:20000].lower()
    for token, weight in [
        ("auc", 4),
        ("wer", 4),
        ("f1", 4),
        ("accuracy", 3),
        ("fps", 3),
        ("dataset", 3),
        ("download", 2),
        ("argparse", 2),
    ]:
        if token in text:
            score += weight
            reasons.append(f"content_contains:{token}")

    if "train" in name:
        score -= 10
        reasons.append("training_name_penalty")
    return score, reasons


def _ignored(path: Path, repo_dir: Path) -> bool:
    try:
        rel = path.relative_to(repo_dir)
    except ValueError:
        return True
    return any(part in {".git", "__pycache__", ".venv", "venv", "node_modules"} or part.startswith(".") for part in rel.parts)
