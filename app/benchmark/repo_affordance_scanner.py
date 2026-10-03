"""Scan a repository for affordances that constrain LLM benchmark planning.

Discovers entrypoints, configs, dataset mentions, sample files, model
checkpoints, and framework signals so that the generic LLM planner only
proposes commands that could plausibly work.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from app.benchmark.script_miner import mine_script_signals


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EntrypointInfo(BaseModel):
    path: str
    kind: str = "script"          # "script" | "module" | "notebook"
    cli_args: list[str] = Field(default_factory=list)
    description: str = ""


class ConfigInfo(BaseModel):
    path: str
    format: str = ""              # "yaml" | "json" | "ini" | "toml"
    keys: list[str] = Field(default_factory=list)


class DatasetMention(BaseModel):
    name: str
    source: str = ""              # "readme" | "code" | "config"
    context: str = ""


class SampleFile(BaseModel):
    path: str
    suffix: str
    size_bytes: int = 0


class RepoAffordance(BaseModel):
    entrypoints: list[EntrypointInfo] = Field(default_factory=list)
    configs: list[ConfigInfo] = Field(default_factory=list)
    dataset_mentions: list[DatasetMention] = Field(default_factory=list)
    sample_files: list[SampleFile] = Field(default_factory=list)
    model_checkpoints: list[str] = Field(default_factory=list)
    framework_signals: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_SUFFIXES: set[str] = {
    # images
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    # audio
    ".wav", ".mp3", ".flac", ".ogg", ".opus",
    # text / data
    ".txt", ".csv", ".tsv", ".json", ".jsonl", ".xml",
    # video
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    # point clouds
    ".ply", ".pcd", ".xyz",
}

_CONFIG_SUFFIXES: dict[str, str] = {
    ".yaml": "yaml", ".yml": "yaml",
    ".json": "json",
    ".ini": "ini", ".cfg": "ini",
    ".toml": "toml",
}

_CHECKPOINT_PATTERNS = (
    "*.pt", "*.pth", "*.ckpt", "*.bin", "*.safetensors", "*.onnx",
)

_FRAMEWORK_PATTERNS: list[tuple[str, str]] = [
    ("import torch", "pytorch"),
    ("from torch", "pytorch"),
    ("pytorch", "pytorch"),
    ("import tensorflow", "tensorflow"),
    ("from tensorflow", "tensorflow"),
    ("tensorflow", "tensorflow"),
    ("import jax", "jax"),
    ("from jax", "jax"),
    ("import flax", "flax"),
    ("from transformers", "huggingface"),
    ("transformers", "huggingface"),
    ("huggingface", "huggingface"),
    ("import openmim", "openmmlab"),
    ("from mmdet", "mmdetection"),
    ("from mmseg", "mmsegmentation"),
    ("import detectron2", "detectron2"),
    ("from ultralytics", "ultralytics"),
    ("ultralytics", "ultralytics"),
    ("import keras", "keras"),
    ("keras", "keras"),
]

_DATASET_NAME_RE = re.compile(
    r"\b("
    r"ImageNet|COCO|CIFAR-10|CIFAR-100|MNIST|Fashion-MNIST|"
    r"LibriSpeech|Common\s*Voice|LJSpeech|"
    r"CoNLL-2003|OntoNotes|WNUT|"
    r"MegaDepth|ScanNet|HPatches|PhotoTourism|YFCC|"
    r"Pascal VOC|ADE20K|Cityscapes|KITTI|nuScenes|"
    r"SQuAD|GLUE|SuperGLUE|RACE|MMLU|HumanEval|"
    r"nuScenes|Waymo|Argoverse|Lyft|"
    r"WMT|IWSLT|Multi30k|"
    r"LAION|Conceptual Captions|"
    r"CelebA|FFHQ|LSUN|"
    r"SBD|PASCAL(Context|Part)|"
    r"OpenImages|Objects365|"
    r"DAVIS|YouTube-VOS|"
    r"LFW|VGGFace2|"
    r"ICDAR|Synth90k|"
    r"ESC-50|UrbanSound|GTZAN|AudioSet"
    r")\b",
    re.IGNORECASE,
)

_ARGPARSE_RE = re.compile(
    r"""add_argument\s*\(\s*['"](-{1,2}[\w-]+)['"]""",
)

_MAX_FILE_READ = 20_000       # bytes per file
_MAX_SCAN_FILES = 50


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_repo_affordances(
    repo_dir: Path,
    readme_text: str,
    candidate_scripts: list[str],
) -> RepoAffordance:
    """Discover what the repository supports for benchmark planning."""
    repo_dir = repo_dir.resolve()

    entrypoints = _scan_entrypoints(repo_dir, candidate_scripts)
    configs = _scan_configs(repo_dir)
    dataset_mentions = _scan_dataset_mentions(repo_dir, readme_text)
    sample_files = _scan_sample_files(repo_dir)
    checkpoints = _scan_checkpoints(repo_dir)
    frameworks = _scan_frameworks(repo_dir, readme_text)

    return RepoAffordance(
        entrypoints=entrypoints,
        configs=configs,
        dataset_mentions=dataset_mentions,
        sample_files=sample_files,
        model_checkpoints=checkpoints,
        framework_signals=frameworks,
    )


# ---------------------------------------------------------------------------
# Internal scanners
# ---------------------------------------------------------------------------

def _scan_entrypoints(repo_dir: Path, candidate_scripts: list[str]) -> list[EntrypointInfo]:
    signals = mine_script_signals(repo_dir, candidate_scripts)
    results: list[EntrypointInfo] = []
    for sig in signals[:_MAX_SCAN_FILES]:
        path = repo_dir / sig.path
        if not path.exists() or not path.is_file():
            continue
        text = _read_head(path)
        cli_args = _extract_argparse_args(text)
        description = _first_docstring_or_comment(text)
        kind = "notebook" if path.suffix == ".ipynb" else "script"
        results.append(EntrypointInfo(
            path=sig.path,
            kind=kind,
            cli_args=cli_args,
            description=description,
        ))
    return results


def _scan_configs(repo_dir: Path) -> list[ConfigInfo]:
    results: list[ConfigInfo] = []
    for pattern in ("configs/**/*", "config/**/*", "cfg/**/*", "conf/**/*"):
        for path in repo_dir.glob(pattern):
            if len(results) >= _MAX_SCAN_FILES:
                break
            if not path.is_file():
                continue
            fmt = _CONFIG_SUFFIXES.get(path.suffix.lower())
            if fmt is None:
                continue
            text = _read_head(path)
            keys = _extract_config_keys(text, fmt)
            results.append(ConfigInfo(
                path=path.relative_to(repo_dir).as_posix(),
                format=fmt,
                keys=keys,
            ))
    return results


def _scan_dataset_mentions(repo_dir: Path, readme_text: str) -> list[DatasetMention]:
    seen: set[str] = set()
    results: list[DatasetMention] = []

    def _add(name: str, source: str, context: str) -> None:
        key = name.lower()
        if key in seen:
            return
        seen.add(key)
        results.append(DatasetMention(name=name, source=source, context=context[:200]))

    for match in _DATASET_NAME_RE.finditer(readme_text):
        _add(match.group(1), "readme", readme_text[max(0, match.start() - 40):match.end() + 40])

    # Scan a subset of Python files for dataset references
    count = 0
    for py_path in repo_dir.rglob("*.py"):
        if count >= 20:
            break
        if _ignored(py_path, repo_dir):
            continue
        text = _read_head(py_path)
        rel = py_path.relative_to(repo_dir).as_posix()
        for match in _DATASET_NAME_RE.finditer(text):
            _add(match.group(1), "code", f"{rel}: {match.group(0)}")
        count += 1

    return results


def _scan_sample_files(repo_dir: Path) -> list[SampleFile]:
    roots = [
        repo_dir / name
        for name in (
            "assets", "asset", "examples", "example", "demo", "demos",
            "data", "samples", "sample", "test", "tests", "inputs",
        )
    ]
    roots.append(repo_dir)
    results: list[SampleFile] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if len(results) >= 30:
                return results
            if not path.is_file() or path.suffix.lower() not in _SAMPLE_SUFFIXES:
                continue
            if _ignored(path, repo_dir):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > 30 * 1024 * 1024:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            results.append(SampleFile(
                path=path.relative_to(repo_dir).as_posix(),
                suffix=path.suffix.lower(),
                size_bytes=size,
            ))
    return results


def _scan_checkpoints(repo_dir: Path) -> list[str]:
    results: list[str] = []
    for pattern in _CHECKPOINT_PATTERNS:
        for path in repo_dir.rglob(pattern):
            if len(results) >= 10:
                return results
            if _ignored(path, repo_dir):
                continue
            try:
                if path.stat().st_size > 5 * 1024 * 1024 * 1024:
                    continue
            except OSError:
                continue
            results.append(path.relative_to(repo_dir).as_posix())
    return results


def _scan_frameworks(repo_dir: Path, readme_text: str) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []

    def _add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            results.append(name)

    text_lower = readme_text.lower()
    for pattern, name in _FRAMEWORK_PATTERNS:
        if pattern.lower() in text_lower:
            _add(name)

    # Check requirements files
    for req_name in ("requirements.txt", "setup.py", "pyproject.toml"):
        req_path = repo_dir / req_name
        if req_path.exists():
            text = _read_head(req_path).lower()
            for pattern, name in _FRAMEWORK_PATTERNS:
                if pattern.lower() in text:
                    _add(name)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_head(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:_MAX_FILE_READ]
    except OSError:
        return ""


def _extract_argparse_args(text: str) -> list[str]:
    return _ARGPARSE_RE.findall(text)


def _first_docstring_or_comment(text: str) -> str:
    lines = text.splitlines()
    for line in lines[:30]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#!") or stripped.startswith("# -*-"):
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            end = stripped[3:]
            if end.endswith('"""') or end.endswith("'''"):
                return end[:-3].strip()
            return stripped[3:].strip()
        if stripped.startswith("#"):
            return stripped[1:].strip()
        return stripped
    return ""


def _extract_config_keys(text: str, fmt: str) -> list[str]:
    if fmt == "json":
        try:
            import json
            data = json.loads(text)
            if isinstance(data, dict):
                return list(data.keys())[:30]
        except Exception:
            pass
        return []
    if fmt in ("yaml", "yml"):
        keys: list[str] = []
        for line in text.splitlines():
            m = re.match(r"^(\w[\w-]*)\s*:", line)
            if m:
                keys.append(m.group(1))
            if len(keys) >= 30:
                break
        return keys
    return []


def _ignored(path: Path, repo_dir: Path) -> bool:
    try:
        rel = path.relative_to(repo_dir)
    except ValueError:
        return True
    return any(
        part in {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache"}
        or part.startswith(".")
        for part in rel.parts
    )
