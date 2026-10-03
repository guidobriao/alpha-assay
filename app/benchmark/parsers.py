from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.benchmark.schema import BenchmarkSpec


def parse_metrics(spec: BenchmarkSpec, stdout: str, stderr: str, repo_dir: Path, run_dir: Path) -> dict[str, Any]:
    parser_type = spec.parser.get("type")
    if parser_type == "local_feature_table":
        return parse_local_feature_table(stdout)
    if parser_type == "local_feature_speed_table":
        return parse_local_feature_speed_table(stdout)
    if parser_type == "xfeat_pose_eval":
        return parse_xfeat_pose_eval(stdout)
    if parser_type == "json_file":
        rel_path = spec.parser.get("path")
        if isinstance(rel_path, str):
            return _augment_json_metrics(spec, _load_json(repo_dir / rel_path))
    if parser_type == "generic_metrics":
        return parse_generic_metrics(stdout + "\n" + stderr)
    if parser_type == "llm_fallback":
        from app.benchmark.generic_metric_parser import parse_with_llm_fallback
        return parse_with_llm_fallback(spec, stdout, stderr, repo_dir, run_dir)
    # Fallback for unknown task families: try generic regex first
    from app.benchmark.schema import KNOWN_TASK_FAMILIES
    if spec.task_family not in KNOWN_TASK_FAMILIES:
        result = parse_generic_metrics(stdout + "\n" + stderr)
        if result:
            return result
    return {}


def parse_local_feature_table(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if line.startswith("AUC@5") and idx + 1 < len(lines):
            values = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", lines[idx + 1])]
            keys = ["AUC@5", "AUC@10", "AUC@20", "Prec", "MScore"]
            if len(values) >= len(keys):
                return dict(zip(keys, values[: len(keys)]))
    return parse_generic_metrics(stdout)


def parse_local_feature_speed_table(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    section: str | None = None
    headers: list[str] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("easy") or line.startswith("difficult"):
            parts = line.split()
            section = parts[0]
            headers = parts[1:]
            continue
        if section and line.startswith(("LightGlue-", "matcher-", "Matcher-")):
            parts = line.split()
            name = parts[0]
            values = parts[1:]
            for key, value in zip(headers, values):
                try:
                    latency_ms = float(value)
                except ValueError:
                    continue
                metric_key = f"{name}_{section}_{key}"
                metrics[f"{metric_key}_latency_ms"] = latency_ms
                if latency_ms > 0:
                    metrics[f"{metric_key}_FPS"] = 1000.0 / latency_ms
    if metrics:
        return metrics
    return parse_generic_metrics(stdout)


def parse_generic_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    patterns = {
        "WER": r"\bWER\b\s*[:=]\s*([0-9.]+)",
        "CER": r"\bCER\b\s*[:=]\s*([0-9.]+)",
        "F1": r"\bF1\b\s*[:=]\s*([0-9.]+)",
        "Accuracy": r"\b(?:accuracy|acc)\b\s*[:=]\s*([0-9.]+)",
        "mAP": r"\bmAP\b\s*[:=]\s*([0-9.]+)",
        "AP50": r"\bAP50\b\s*[:=]\s*([0-9.]+)",
        "mIoU": r"\bmIoU\b\s*[:=]\s*([0-9.]+)",
        "FPS": r"\bFPS\b\s*[:=]\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                pass
    keypoints = re.search(r"keypoints:\s*torch\.Size\(\[(\d+),\s*(\d+)\]\)", text)
    if keypoints:
        metrics["num_keypoints"] = int(keypoints.group(1))
        metrics["keypoint_dim"] = int(keypoints.group(2))
    descriptors = re.search(r"descriptors:\s*torch\.Size\(\[(\d+),\s*(\d+)\]\)", text)
    if descriptors:
        metrics["num_descriptors"] = int(descriptors.group(1))
        metrics["descriptor_dim"] = int(descriptors.group(2))
    batch_features = re.search(r"# detected features on each batch item:\s*\[([0-9,\s]+)\]", text)
    if batch_features:
        values = [int(item.strip()) for item in batch_features.group(1).split(",") if item.strip()]
        metrics["batch_detected_features"] = values
        if values:
            metrics["batch_detected_features_mean"] = sum(values) / len(values)
    match_shape = re.search(r"torch\.Size\(\[(\d+),\s*4\]\)", text)
    if match_shape:
        metrics["num_matches"] = int(match_shape.group(1))
    return metrics


def parse_xfeat_pose_eval(stdout: str) -> dict[str, Any]:
    metrics = parse_generic_metrics(stdout)
    for match in re.finditer(r"\bauc@(\d+)\s*:\s*([0-9.]+)", stdout, flags=re.IGNORECASE):
        metrics[f"AUC@{match.group(1)}"] = float(match.group(2))
    for match in re.finditer(r"\bmAcc@(\d+):\s*([0-9.]+)", stdout, flags=re.IGNORECASE):
        metrics[f"mAcc@{match.group(1)}"] = float(match.group(2))
    pair_match = re.search(r"auc\s*/\s*mAcc\s+on\s+(\d+)\s+pairs", stdout, flags=re.IGNORECASE)
    if pair_match:
        metrics["num_pairs"] = int(pair_match.group(1))
    return metrics


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {"value": data}


def _augment_json_metrics(spec: BenchmarkSpec, data: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(data)
    if spec.task_family == "zero_shot_classification":
        label_probs = data.get("label_probs")
        if isinstance(label_probs, dict):
            for label, value in label_probs.items():
                metrics[f"label_probs.{label}"] = value
    elif spec.task_family == "asr":
        text = data.get("text")
        if isinstance(text, str):
            metrics["transcription_nonempty"] = bool(text.strip())
            metrics["transcription_chars"] = len(text.strip())
            metrics["transcription_words"] = len(text.split())
    elif spec.task_family == "sequence_labeling":
        entities = data.get("entities")
        if isinstance(entities, list):
            metrics["num_entities"] = len(entities)
            metrics["entity_tags"] = sorted({
                str(entity.get("tag"))
                for entity in entities
                if isinstance(entity, dict) and entity.get("tag") is not None
            })
    return metrics
