from __future__ import annotations

import re

from app.benchmark.schema import MetricSpec, TaskOntology


METRIC_ALIASES: dict[str, list[str]] = {
    "auc@5": ["AUC@5", "pose_auc_5", "auc_5", "AUC 5"],
    "auc@10": ["AUC@10", "pose_auc_10", "auc_10", "AUC 10"],
    "auc@20": ["AUC@20", "pose_auc_20", "auc_20", "AUC 20"],
    "precision": ["Prec", "precision", "P"],
    "matching_score": ["MScore", "matching score", "matching_score"],
    "fps": ["FPS", "frames/s", "throughput"],
    "latency_ms": ["latency", "ms/img", "time"],
    "wer": ["WER", "word error rate"],
    "cer": ["CER", "character error rate"],
    "bleu": ["BLEU"],
    "accuracy": ["accuracy", "acc", "top-1", "top1"],
    "top5_accuracy": ["top-5", "top5", "top-5 accuracy"],
    "f1": ["F1", "micro-F1", "span-F1"],
    "precision_nlp": ["precision"],
    "recall": ["recall"],
}


def metric_specs_for_family(task_family: str, metrics: list[str] | None = None) -> list[MetricSpec]:
    if task_family == "local_feature_matching":
        return [
            MetricSpec(name="AUC@5", canonical_name="auc@5", direction="higher_is_better", unit="%"),
            MetricSpec(name="AUC@10", canonical_name="auc@10", direction="higher_is_better", unit="%"),
            MetricSpec(name="AUC@20", canonical_name="auc@20", direction="higher_is_better", unit="%"),
            MetricSpec(name="Precision", canonical_name="precision", direction="higher_is_better", unit="%"),
            MetricSpec(name="MatchingScore", canonical_name="matching_score", direction="higher_is_better", unit="%"),
            MetricSpec(name="FPS", canonical_name="fps", direction="higher_is_better"),
            MetricSpec(name="Latency", canonical_name="latency_ms", direction="lower_is_better", unit="ms"),
        ]
    if task_family == "zero_shot_classification":
        return [
            MetricSpec(name="Top-1 Accuracy", canonical_name="accuracy", direction="higher_is_better", unit="%"),
            MetricSpec(name="Top-5 Accuracy", canonical_name="top5_accuracy", direction="higher_is_better", unit="%"),
        ]
    if task_family == "asr":
        return [
            MetricSpec(name="WER", canonical_name="wer", direction="lower_is_better", unit="%"),
            MetricSpec(name="CER", canonical_name="cer", direction="lower_is_better", unit="%"),
            MetricSpec(name="BLEU", canonical_name="bleu", direction="higher_is_better"),
        ]
    if task_family == "sequence_labeling":
        return [
            MetricSpec(name="F1", canonical_name="f1", direction="higher_is_better", unit="%"),
            MetricSpec(name="Precision", canonical_name="precision_nlp", direction="higher_is_better", unit="%"),
            MetricSpec(name="Recall", canonical_name="recall", direction="higher_is_better", unit="%"),
        ]
    # For unknown families, infer MetricSpec from paper-provided metric names
    if metrics:
        return _infer_metric_specs_from_paper(metrics)
    return []


def _infer_metric_specs_from_paper(metrics: list[str]) -> list[MetricSpec]:
    specs: list[MetricSpec] = []
    for name in metrics[:10]:
        direction = _infer_metric_direction(name)
        unit = _infer_metric_unit(name)
        specs.append(MetricSpec(
            name=name,
            canonical_name=_canonicalize_metric_name(name),
            direction=direction,
            unit=unit,
        ))
    return specs


_LOWER_IS_BETTER_PATTERNS = re.compile(
    r"(wer|cer|loss|error.?rate|perplexity|latency|time|rmse|mae|mse|l1|l2|distance)",
    re.IGNORECASE,
)

_SPEED_PATTERNS = re.compile(r"(fps|speed|throughput|qps)", re.IGNORECASE)


def _infer_metric_direction(name: str) -> str:
    if _LOWER_IS_BETTER_PATTERNS.search(name):
        return "lower_is_better"
    if _SPEED_PATTERNS.search(name):
        return "higher_is_better"
    # Default to higher-is-better for most ML metrics
    return "higher_is_better"


def _infer_metric_unit(name: str) -> str | None:
    lower = name.lower()
    if any(kw in lower for kw in ("fps", "speed", "throughput", "qps")):
        return None
    if any(kw in lower for kw in ("time", "latency", "ms")):
        return "ms"
    if any(kw in lower for kw in ("loss", "rmse", "mae", "mse", "perplexity")):
        return None
    return "%"


def _canonicalize_metric_name(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip()).lower()


# ---------------------------------------------------------------------------
# Task classification
# ---------------------------------------------------------------------------

_FAMILY_TERMS: dict[str, list[str]] = {
    "local_feature_matching": [
        "feature matching", "local feature", "keypoint", "homography", "pose estimation",
        "megadepth", "scannet", "scan net", "hpatches", "phototourism", "photo tourism",
        "auc", "auc 5", "auc 10", "auc 20", "matching score", "mscore", "match_pairs",
        "superglue", "super glue", "superpoint", "super point", "loftr", "lightglue",
        "light glue", "xfeat",
    ],
    "zero_shot_classification": [
        "clip", "zero-shot", "zero shot", "image-text", "imagenet", "cifar", "top-1", "top-5",
    ],
    "asr": [
        "speech recognition", "automatic speech", "transcribe", "asr", "wer", "cer", "librispeech", "whisper",
    ],
    "sequence_labeling": [
        "sequence labeling", "named entity", "ner", "conll", "span-f1", "flair", "tagger",
    ],
}


def classify_task_family(*, task: str | None, datasets: list[str], metrics: list[str], keywords: list[str], repo_text: str, scripts: list[str]) -> str:
    """Classify task into a family string.  Returns 'unknown' for non-specialist tasks."""
    ontology = classify_task_ontology(
        task=task, datasets=datasets, metrics=metrics,
        keywords=keywords, repo_text=repo_text, scripts=scripts,
    )
    return ontology.family


def classify_task_ontology(
    *,
    task: str | None,
    datasets: list[str],
    metrics: list[str],
    keywords: list[str],
    repo_text: str,
    scripts: list[str],
) -> TaskOntology:
    """Classify task with rich taxonomy information."""
    haystack = _normalize_for_matching(" ".join([
        task or "",
        " ".join(datasets),
        " ".join(metrics),
        " ".join(keywords),
        repo_text[:20000],
        " ".join(scripts),
    ]))

    scores = {family: _score_terms(haystack, terms) for family, terms in _FAMILY_TERMS.items()}
    best, score = max(scores.items(), key=lambda item: item[1])

    if score > 0:
        family = best
        is_known = True
        confidence = score / max(sum(scores.values()), 1)
    else:
        family = _infer_family_from_text(haystack)
        is_known = False
        confidence = 0.0

    return TaskOntology(
        family=family,
        domain=_infer_domain(family, datasets, keywords, haystack),
        input_modalities=_infer_input_modalities(family, datasets, keywords, haystack),
        output_modalities=_infer_output_modalities(family, metrics, haystack),
        metric_types=_infer_metric_types(metrics),
        confidence=min(confidence, 1.0),
        is_known_family=is_known,
    )


def _infer_family_from_text(haystack: str) -> str:
    """Best-effort family name inference for non-specialist tasks."""
    cv_signals = ["detection", "segmentation", "classification", "recognition", "tracking", "depth", "reconstruction"]
    nlp_signals = ["translation", "summarization", "generation", "sentiment", "question answering", "qa"]
    audio_signals = ["speech", "audio", "voice", "speaker", "music"]
    multimodal_signals = ["image-text", "vision-language", "multimodal", "vqa", "image captioning"]

    if any(s in haystack for s in multimodal_signals):
        return "multimodal"
    if any(s in haystack for s in cv_signals):
        task_hint = next((s for s in cv_signals if s in haystack), "vision")
        return f"computer_vision_{task_hint}"
    if any(s in haystack for s in audio_signals):
        return "audio"
    if any(s in haystack for s in nlp_signals):
        return "nlp"
    return "unknown"


def _infer_domain(family: str, datasets: list[str], keywords: list[str], haystack: str) -> str:
    cv_datasets = {"coco", "imagenet", "cifar", "pascal", "ade20k", "cityscapes", "kitti"}
    nlp_datasets = {"squad", "glue", "conll", "wnut", "mnli", "qqp"}
    audio_datasets = {"librispeech", "ljspeech", "common voice", "gtzan"}

    ds_lower = {d.lower() for d in datasets}
    if ds_lower & cv_datasets:
        return "cv"
    if ds_lower & nlp_datasets:
        return "nlp"
    if ds_lower & audio_datasets:
        return "audio"

    if any(kw in haystack for kw in ("image", "visual", "bounding box", "segmentation", "detection")):
        return "cv"
    if any(kw in haystack for kw in ("text", "language", "token", "sentence", "word", "translation", "generation", "summarization")):
        return "nlp"
    if any(kw in haystack for kw in ("speech", "audio", "voice", "sound")):
        return "audio"
    if any(kw in haystack for kw in ("multimodal", "image-text", "vision-language")):
        return "multimodal"
    return "other"


def _infer_input_modalities(family: str, datasets: list[str], keywords: list[str], haystack: str) -> list[str]:
    mods: list[str] = []
    if any(kw in haystack for kw in ("image", "visual", "rgb", "photo", "video", "frame")):
        mods.append("image")
    if any(kw in haystack for kw in ("text", "language", "sentence", "token", "word", "document", "translation", "generation", "summarization", "question", "dialog")):
        mods.append("text")
    if any(kw in haystack for kw in ("speech", "audio", "voice", "sound", "waveform")):
        mods.append("audio")
    if any(kw in haystack for kw in ("point cloud", "3d", "lidar", "depth")):
        mods.append("pointcloud")
    return mods or ["unknown"]


def _infer_output_modalities(family: str, metrics: list[str], haystack: str) -> list[str]:
    if any(kw in haystack for kw in ("bounding box", "bbox", "detection", "object detection")):
        return ["bounding_boxes", "class_labels"]
    if any(kw in haystack for kw in ("segmentation", "mask", "pixel")):
        return ["pixel_mask"]
    if any(kw in haystack for kw in ("keypoint", "landmark", "pose")):
        return ["keypoints"]
    if any(kw in haystack for kw in ("transcription", "speech recognition", "asr")):
        return ["text"]
    if any(kw in haystack for kw in ("translation", "generation", "summarization", "caption")):
        return ["text"]
    if any(kw in haystack for kw in ("classification", "recogni", "zero-shot")):
        return ["class_label"]
    if any(kw in haystack for kw in ("ner", "named entity", "tag", "sequence labeling")):
        return ["spans", "tags"]
    return ["unknown"]


def _infer_metric_types(metrics: list[str]) -> list[str]:
    types: list[str] = []
    m_str = " ".join(metrics).lower()
    if any(kw in m_str for kw in ("accuracy", "acc", "top-1", "top-5")):
        types.append("accuracy")
    if any(kw in m_str for kw in ("f1", "precision", "recall")):
        types.append("f1")
    if any(kw in m_str for kw in ("wer", "cer", "error rate")):
        types.append("error_rate")
    if any(kw in m_str for kw in ("bleu", "rouge", "meteor")):
        types.append("generation_quality")
    if any(kw in m_str for kw in ("map", "iou", "dice")):
        types.append("overlap")
    if any(kw in m_str for kw in ("fps", "latency", "speed", "throughput")):
        types.append("speed")
    if any(kw in m_str for kw in ("auc", "auroc")):
        types.append("ranking")
    if any(kw in m_str for kw in ("loss", "perplexity")):
        types.append("training_loss")
    return types or ["unknown"]


def _score_terms(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if _normalize_for_matching(term) in text)


def _normalize_for_matching(text: str) -> str:
    """Normalize noisy PDF/repo text for task-family keyword matching."""
    text = text.lower()
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
