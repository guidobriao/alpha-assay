from __future__ import annotations

import re

from app.tools.pdf_tool import extract_github_links


COMMON_DATASETS = [
    "CIFAR-10", "CIFAR-100", "ImageNet", "COCO", "MNIST",
    "SST-2", "GLUE", "SQuAD", "Cityscapes", "KITTI",
    "WMT", "LibriSpeech", "MIMIC",
]

COMMON_METRICS = [
    "accuracy", "acc", "F1", "BLEU", "ROUGE", "mAP",
    "IoU", "AUC", "WER", "perplexity", "PSNR", "SSIM", "FID",
]

TASK_KEYWORDS = [
    "classification",
    "object detection",
    "segmentation",
    "machine translation",
    "speech recognition",
    "question answering",
    "image generation",
    "text generation",
]


def extract_datasets(text: str) -> list[str]:
    return sorted({d for d in COMMON_DATASETS if d.lower() in text.lower()})


def extract_metrics(text: str) -> list[str]:
    return sorted({m for m in COMMON_METRICS if m.lower() in text.lower()})


def extract_tasks(text: str) -> list[str]:
    return sorted({t for t in TASK_KEYWORDS if t.lower() in text.lower()})


def extract_method_keywords(text: str) -> list[str]:
    phrases = re.findall(r"\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){1,4}\b", text)
    counts = {}

    for phrase in phrases:
        if len(phrase) < 8:
            continue
        counts[phrase] = counts.get(phrase, 0) + 1

    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [phrase for phrase, count in ranked[:10]]
