from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkDatasetEntry:
    dataset_id: str
    name: str
    task_family: str
    env_var: str
    aliases: tuple[str, ...]
    metrics: tuple[str, ...]
    size_estimate: str = "external"
    estimated_size_gb: float | None = None
    auto_download: bool = False
    public: bool | None = True


@dataclass(frozen=True)
class DatasetSizeEstimate:
    dataset_ids: tuple[str, ...]
    names: tuple[str, ...]
    estimated_size_gb: float | None
    size_estimate: str
    evidence: tuple[str, ...]


DATASETS: dict[str, BenchmarkDatasetEntry] = {
    "megadepth1500": BenchmarkDatasetEntry(
        dataset_id="megadepth1500",
        name="MegaDepth-1500",
        task_family="local_feature_matching",
        env_var="PAPER_BENCH_MEGDEPTH1500_DIR",
        aliases=("megadepth", "megadepth-1500", "mega1500"),
        metrics=("AUC@5", "AUC@10", "AUC@20", "mAcc@5", "mAcc@10", "mAcc@20"),
        size_estimate="external eval subset, often >1GB",
        estimated_size_gb=10.0,
    ),
    "scannet1500": BenchmarkDatasetEntry(
        dataset_id="scannet1500",
        name="ScanNet-1500",
        task_family="local_feature_matching",
        env_var="PAPER_BENCH_SCANNET1500_DIR",
        aliases=("scannet", "scannet-1500"),
        metrics=("AUC@5", "AUC@10", "AUC@20", "mAcc@5", "mAcc@10", "mAcc@20"),
        size_estimate="external eval subset, often >1GB",
        estimated_size_gb=10.0,
    ),
    "hpatches": BenchmarkDatasetEntry(
        dataset_id="hpatches",
        name="HPatches",
        task_family="local_feature_matching",
        env_var="PAPER_BENCH_HPATCHES_DIR",
        aliases=("hpatches", "hpatches-sequences", "hpatches benchmark"),
        metrics=("MMA", "matching accuracy", "homography accuracy"),
        size_estimate="HPatches image sequences are commonly around 1-2GB depending on packaging",
        estimated_size_gb=1.5,
    ),
    "phototourism": BenchmarkDatasetEntry(
        dataset_id="phototourism",
        name="PhotoTourism",
        task_family="local_feature_matching",
        env_var="PAPER_BENCH_PHOTOTOURISM_DIR",
        aliases=("phototourism", "photo tourism", "image matching challenge", "imc"),
        metrics=("AUC@5", "AUC@10", "AUC@20", "mAA"),
        size_estimate="PhotoTourism/Image Matching Challenge data is an external multi-GB benchmark",
        estimated_size_gb=20.0,
    ),
    "yfcc100m": BenchmarkDatasetEntry(
        dataset_id="yfcc100m",
        name="YFCC100M",
        task_family="local_feature_matching",
        env_var="PAPER_BENCH_YFCC100M_DIR",
        aliases=("yfcc", "yfcc100m", "yfcc 100m"),
        metrics=("AUC@5", "AUC@10", "AUC@20", "mAA"),
        size_estimate="YFCC100M is a very large external collection; full data is far beyond smoke benchmark scope",
        estimated_size_gb=1000.0,
    ),
    "librispeech": BenchmarkDatasetEntry(
        dataset_id="librispeech",
        name="LibriSpeech",
        task_family="asr",
        env_var="PAPER_BENCH_LIBRISPEECH_DIR",
        aliases=("librispeech", "test-clean", "test-other"),
        metrics=("WER", "CER", "RTF"),
        size_estimate="LibriSpeech test-clean is about 0.34GB; full corpus is much larger",
        estimated_size_gb=0.34,
        auto_download=True,
    ),
    "conll03": BenchmarkDatasetEntry(
        dataset_id="conll03",
        name="CoNLL-03",
        task_family="sequence_labeling",
        env_var="PAPER_BENCH_CONLL03_DIR",
        aliases=("conll", "conll03", "conll-03", "conll2003"),
        metrics=("Precision", "Recall", "F1"),
        size_estimate="small, usually <0.02GB",
        estimated_size_gb=0.02,
        auto_download=True,
    ),
    "cifar100": BenchmarkDatasetEntry(
        dataset_id="cifar100",
        name="CIFAR-100",
        task_family="zero_shot_classification",
        env_var="PAPER_BENCH_CIFAR100_DIR",
        aliases=("cifar100", "cifar-100"),
        metrics=("Top-1 Accuracy", "Top-5 Accuracy"),
        size_estimate="about 0.16GB",
        estimated_size_gb=0.16,
        auto_download=True,
    ),
    "imagenet": BenchmarkDatasetEntry(
        dataset_id="imagenet",
        name="ImageNet",
        task_family="zero_shot_classification",
        env_var="PAPER_BENCH_IMAGENET_DIR",
        aliases=("imagenet", "ilsvrc"),
        metrics=("Top-1 Accuracy", "Top-5 Accuracy"),
        size_estimate="about 144GB for ILSVRC classification data",
        estimated_size_gb=144.0,
        public=None,
    ),
}


def dataset_entry(dataset_id: str) -> BenchmarkDatasetEntry:
    return DATASETS[dataset_id]


def estimate_dataset_size_from_context(
    *,
    task_family: str | None = None,
    datasets: list[str] | tuple[str, ...] = (),
    text: str = "",
) -> DatasetSizeEstimate | None:
    haystack_parts = [text, *datasets]
    haystack = "\n".join(part for part in haystack_parts if part).lower()
    if not haystack.strip():
        return None

    matched: list[BenchmarkDatasetEntry] = []
    evidence: list[str] = []
    for entry in DATASETS.values():
        if task_family and entry.task_family != task_family:
            continue
        aliases = (entry.name, entry.dataset_id, *entry.aliases)
        for alias in aliases:
            if alias.lower() in haystack:
                matched.append(entry)
                evidence.append(f"{entry.name} matched alias '{alias}'")
                break

    if not matched:
        return None

    unique = _dedupe_entries(matched)
    known_sizes = [entry.estimated_size_gb for entry in unique if entry.estimated_size_gb is not None]
    estimated_size_gb = sum(known_sizes) if known_sizes and len(known_sizes) == len(unique) else None
    names = tuple(entry.name for entry in unique)
    if estimated_size_gb is None:
        size_estimate = "matched known dataset name(s), but at least one size estimate is unavailable"
    elif len(unique) == 1:
        size_estimate = unique[0].size_estimate
    else:
        size_estimate = f"combined conservative estimate for {', '.join(names)}"
    return DatasetSizeEstimate(
        dataset_ids=tuple(entry.dataset_id for entry in unique),
        names=names,
        estimated_size_gb=estimated_size_gb,
        size_estimate=size_estimate,
        evidence=tuple(evidence),
    )


def data_root(dataset_id: str, paper_slug: str | None = None, workspace_dir: str | Path | None = None) -> str | None:
    entry = dataset_entry(dataset_id)
    direct = os.environ.get(entry.env_var)
    if direct:
        return direct

    if workspace_dir is not None:
        workspace_root = Path(workspace_dir) / "datasets"
        found = _find_dataset_under_root(workspace_root, entry, paper_slug)
        if found:
            return str(found)

    generic_root = os.environ.get("PAPER_BENCH_DATA_ROOT")
    if not generic_root:
        return None
    found = _find_dataset_under_root(Path(generic_root), entry, paper_slug)
    return str(found) if found else None


def expected_data_root(dataset_id: str, paper_slug: str | None = None, workspace_dir: str | Path | None = None) -> Path | None:
    if workspace_dir is not None and paper_slug:
        return Path(workspace_dir) / "datasets" / paper_slug / dataset_id
    generic_root = os.environ.get("PAPER_BENCH_DATA_ROOT")
    if generic_root and paper_slug:
        return Path(generic_root) / paper_slug / dataset_id
    if generic_root:
        return Path(generic_root) / dataset_id
    return None


def _find_dataset_under_root(root: Path, entry: BenchmarkDatasetEntry, paper_slug: str | None = None) -> Path | None:
    if not root.exists():
        return None
    candidates = [entry.dataset_id, entry.name, *entry.aliases]
    if paper_slug:
        for candidate in candidates:
            path = root / paper_slug / candidate
            if path.exists():
                return path
        for candidate in candidates:
            path = root / f"{paper_slug}-{candidate}"
            if path.exists():
                return path
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    return None


def dataset_size_gb(dataset_id: str, root: str | None = None) -> float | None:
    if root:
        return directory_size_gb(Path(root))
    return dataset_entry(dataset_id).estimated_size_gb


def directory_size_gb(path: Path) -> float | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size / (1024 ** 3)
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            total += item.stat().st_size
        except OSError:
            continue
    return total / (1024 ** 3)


def missing_data_reason(dataset_id: str, paper_slug: str | None = None, workspace_dir: str | Path | None = None) -> str:
    entry = dataset_entry(dataset_id)
    expected = expected_data_root(dataset_id, paper_slug, workspace_dir)
    reason = (
        f"{entry.name} dataset path is not configured. Set {entry.env_var} "
        f"or place the dataset under PAPER_BENCH_DATA_ROOT/{entry.dataset_id}."
    )
    if expected:
        reason += f" Preferred paper-named cache path: {expected}."
    return reason


def _dedupe_entries(entries: list[BenchmarkDatasetEntry]) -> list[BenchmarkDatasetEntry]:
    seen: set[str] = set()
    out: list[BenchmarkDatasetEntry] = []
    for entry in entries:
        if entry.dataset_id in seen:
            continue
        seen.add(entry.dataset_id)
        out.append(entry)
    return out
