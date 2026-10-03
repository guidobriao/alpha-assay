from __future__ import annotations

from app.benchmark.adapters.base import AdapterContext, find_sample_files
from app.benchmark.dataset_registry import data_root, dataset_entry, dataset_size_gb, expected_data_root, missing_data_reason
from app.benchmark.ontology import metric_specs_for_family
from app.benchmark.schema import BenchmarkSpec, DatasetSpec, ModelSpec


class ZeroShotClassificationAdapter:
    task_family = "zero_shot_classification"

    def propose_benchmarks(self, context: AdapterContext) -> list[BenchmarkSpec]:
        specs = [self._paper_table_protocol(context)]
        specs.extend(self._classification_dataset_protocols(context))
        image = _clip_readme_image(context)
        if image and (context.repo_dir / "clip").exists():
            script_name = "paper_benchmark_zero_shot_readme.py"
            specs.append(BenchmarkSpec(
                id="zero_shot_classification_readme_image_text",
                task_family="zero_shot_classification",
                level="L1",
                title="README-style image-text zero-shot probability example",
                dataset=DatasetSpec(name=image, source="bundled", size_estimate="tiny", public=True),
                model=ModelSpec(name="ViT-B/32", checkpoint_source="official"),
                command=["python", script_name],
                command_kind="generated_runner",
                expected_metrics=metric_specs_for_family("zero_shot_classification"),
                parser={"type": "json_file", "path": "paper_benchmark_zero_shot_results.json"},
                reference={
                    "source": "CLIP README usage example on bundled CLIP.png",
                    "scope": "single_image_readme_example_not_full_paper_dataset",
                    "metrics": {
                        "top_label": "a diagram",
                        "label_probs.a diagram": 0.9927937,
                        "label_probs.a dog": 0.00421068,
                        "label_probs.a cat": 0.00299572,
                    },
                },
                generated_script_name=script_name,
                generated_script_body=_clip_readme_script(image),
                evidence=["package:clip", f"sample_image:{image}"],
            ))
        return specs

    def _classification_dataset_protocols(self, context: AdapterContext) -> list[BenchmarkSpec]:
        specs: list[BenchmarkSpec] = []
        if not (context.repo_dir / "clip").exists():
            return specs
        for dataset_id in ("cifar100", "imagenet"):
            entry = dataset_entry(dataset_id)
            existing_root = data_root(dataset_id, context.paper_slug, context.workspace_dir)
            size_gb = dataset_size_gb(dataset_id, existing_root)
            will_download = False
            root = existing_root
            if (
                root is None
                and entry.auto_download
                and entry.estimated_size_gb is not None
                and entry.estimated_size_gb <= context.budget.max_dataset_size_gb
            ):
                expected_root = expected_data_root(dataset_id, context.paper_slug, context.workspace_dir)
                if expected_root is not None:
                    root = str(expected_root)
                    will_download = True
            script_name = f"paper_benchmark_zero_shot_{dataset_id}.py"
            specs.append(BenchmarkSpec(
                id=f"zero_shot_{dataset_id}_eval",
                task_family="zero_shot_classification",
                level="L3",
                title=f"{entry.name} zero-shot classification benchmark",
                dataset=DatasetSpec(
                    name=entry.name,
                    source="official_repo",
                    size_estimate=entry.size_estimate,
                    size_gb=size_gb,
                    public=entry.public,
                    notes=["Generic CLIP-style zero-shot classification runner."],
                ),
                model=ModelSpec(name="env PAPER_BENCH_CLIP_MODEL or ViT-B/32", checkpoint_source="official"),
                command=["python", script_name] if root else [],
                command_kind="generated_runner" if root else "manual_protocol",
                expected_metrics=metric_specs_for_family("zero_shot_classification"),
                parser={"type": "json_file", "path": f"paper_benchmark_zero_shot_{dataset_id}_results.json"},
                generated_script_name=script_name if root else None,
                generated_script_body=_zero_shot_dataset_script(dataset_id, root, will_download) if root else None,
                feasibility=(
                    {"runnable": True, "data_source": root, "dataset_id": entry.dataset_id, "will_download": will_download}
                    if root
                    else {"runnable": False, "reason": missing_data_reason(entry.dataset_id, context.paper_slug, context.workspace_dir), "required_env": entry.env_var, "dataset_id": entry.dataset_id}
                ),
                fallback_reason="Use README image-text example when full zero-shot classification data is unavailable.",
                evidence=[f"dataset:{entry.name}", f"required_env:{entry.env_var}"],
            ))
        return specs

    def _paper_table_protocol(self, context: AdapterContext) -> BenchmarkSpec:
        dataset = next((d for d in context.datasets if d.lower() in {"imagenet", "cifar-100", "cifar100"}), None)
        return BenchmarkSpec(
            id="zero_shot_classification_paper_table_protocol",
            task_family="zero_shot_classification",
            level="L3",
            title="Paper-table zero-shot classification protocol",
            dataset=DatasetSpec(
                name=dataset or "paper main zero-shot classification dataset",
                source="paper",
                size_estimate="large_or_external",
                public=None,
                notes=["Requires canonical class names/templates, split, and dataset access."],
            ),
            model=ModelSpec(name="paper zero-shot checkpoint", checkpoint_source="paper"),
            command=[],
            command_kind="manual_protocol",
            expected_metrics=metric_specs_for_family("zero_shot_classification"),
            feasibility={"runnable": False, "reason": "full classification benchmark data is not bundled"},
            fallback_reason="Use bundled README image-text example when full dataset is unavailable.",
            evidence=["paper_protocol_candidate"],
        )


def _clip_readme_image(context: AdapterContext) -> str | None:
    if (context.repo_dir / "CLIP.png").exists():
        return "CLIP.png"
    images = find_sample_files(context, {".png", ".jpg", ".jpeg", ".bmp"}, limit=1)
    return images[0] if images else None


def _clip_readme_script(image: str) -> str:
    return f'''
import json
from pathlib import Path

import clip
import torch
from PIL import Image


labels = ["a diagram", "a dog", "a cat"]
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
image = preprocess(Image.open({image!r})).unsqueeze(0).to(device)
text = clip.tokenize(labels).to(device)

with torch.no_grad():
    logits_per_image, _ = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

result = {{
    "device": device,
    "model": "ViT-B/32",
    "input_image": {image!r},
    "labels": labels,
    "label_probs": {{label: float(prob) for label, prob in zip(labels, probs)}},
    "top_label": labels[int(probs.argmax())],
    "top_probability": float(probs.max()),
}}
Path("paper_benchmark_zero_shot_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
'''.lstrip()


def _zero_shot_dataset_script(dataset_id: str, root: str, allow_download: bool = False) -> str:
    return f'''
import json
import os
from pathlib import Path

import clip
import torch
from PIL import Image


def load_dataset(dataset_id, root):
    if dataset_id == "cifar100":
        from torchvision.datasets import CIFAR100
        Path(root).mkdir(parents=True, exist_ok=True)
        ds = CIFAR100(root=root, train=False, download={allow_download!r})
        return ds, ds.classes
    if dataset_id == "imagenet":
        from torchvision.datasets import ImageFolder
        ds = ImageFolder(root=root)
        classes = [name.replace("_", " ") for name, _ in sorted(ds.class_to_idx.items(), key=lambda item: item[1])]
        return ds, classes
    raise SystemExit("unsupported dataset")


dataset_id = {dataset_id!r}
root = {root!r}
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = os.environ.get("PAPER_BENCH_CLIP_MODEL", "ViT-B/32")
max_examples = int(os.environ.get("PAPER_BENCH_MAX_EXAMPLES", "1000"))
templates = ["a photo of a {{}}.", "a blurry photo of a {{}}.", "a photo of the {{}}."]
model, preprocess = clip.load(model_name, device=device)
dataset, classes = load_dataset(dataset_id, root)

with torch.no_grad():
    text_features = []
    for class_name in classes:
        prompts = clip.tokenize([template.format(class_name) for template in templates]).to(device)
        features = model.encode_text(prompts)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.mean(dim=0)
        features = features / features.norm()
        text_features.append(features)
    text_features = torch.stack(text_features, dim=1)

correct1 = correct5 = total = 0
with torch.no_grad():
    for image, target in dataset:
        image_input = preprocess(image.convert("RGB") if isinstance(image, Image.Image) else image).unsqueeze(0).to(device)
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features
        topk = logits.topk(min(5, len(classes)), dim=1).indices[0].cpu().tolist()
        correct1 += int(topk[0] == target)
        correct5 += int(target in topk)
        total += 1
        if total >= max_examples:
            break

payload = {{
    "device": device,
    "model": model_name,
    "dataset": dataset_id,
    "dataset_root": root,
    "num_examples": total,
    "Top-1 Accuracy": 100.0 * correct1 / max(1, total),
    "Top-5 Accuracy": 100.0 * correct5 / max(1, total),
}}
Path(f"paper_benchmark_zero_shot_{{dataset_id}}_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
'''.lstrip()
