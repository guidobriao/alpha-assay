from __future__ import annotations

from app.benchmark.adapters.base import AdapterContext
from app.benchmark.adapters.downloads import download_helper_script
from app.benchmark.dataset_registry import data_root, dataset_entry, dataset_size_gb, expected_data_root, missing_data_reason
from app.benchmark.ontology import metric_specs_for_family
from app.benchmark.schema import BenchmarkSpec, DatasetSpec, ModelSpec


class SequenceLabelingAdapter:
    task_family = "sequence_labeling"

    def propose_benchmarks(self, context: AdapterContext) -> list[BenchmarkSpec]:
        specs = [self._paper_table_protocol(context)]
        specs.append(self._conll03_protocol(context))
        if _looks_like_flair_or_sequence_labeling(context):
            script_name = "paper_benchmark_sequence_labeling_sample.py"
            specs.append(BenchmarkSpec(
                id="sequence_labeling_pretrained_tagger_sample",
                task_family="sequence_labeling",
                level="L1",
                title="Pretrained sequence tagger sample inference",
                dataset=DatasetSpec(name="inline sentence sample", source="synthetic", size_estimate="tiny", public=True),
                model=ModelSpec(name="ner", checkpoint_source="official"),
                command=["python", script_name],
                command_kind="generated_runner",
                expected_metrics=metric_specs_for_family("sequence_labeling"),
                parser={"type": "json_file", "path": "paper_benchmark_sequence_labeling_results.json"},
                reference={
                    "source": "generated pretrained tagger inference run",
                    "scope": "single_sentence_demo_not_conll03_f1_benchmark",
                    "metrics": {},
                },
                generated_script_name=script_name,
                generated_script_body=_sequence_labeling_script(),
                evidence=["sequence_labeling_api_or_readme_signal"],
            ))
        return specs

    def _conll03_protocol(self, context: AdapterContext) -> BenchmarkSpec:
        entry = dataset_entry("conll03")
        existing_root = data_root("conll03", context.paper_slug, context.workspace_dir)
        size_gb = dataset_size_gb("conll03", existing_root)
        will_download = False
        root = existing_root
        if (
            root is None
            and entry.auto_download
            and entry.estimated_size_gb is not None
            and entry.estimated_size_gb <= context.budget.max_dataset_size_gb
        ):
            expected_root = expected_data_root("conll03", context.paper_slug, context.workspace_dir)
            if expected_root is not None:
                root = str(expected_root)
                will_download = True
        script_name = "paper_benchmark_sequence_labeling_conll.py"
        command = ["python", script_name] if root else []
        return BenchmarkSpec(
            id="sequence_labeling_conll03_eval",
            task_family="sequence_labeling",
            level="L3",
            title="CoNLL-style sequence labeling span-F1 benchmark",
            dataset=DatasetSpec(
                name=entry.name,
                source="official_repo",
                size_estimate=entry.size_estimate,
                size_gb=size_gb,
                public=entry.public,
                notes=[
                    "Generic CoNLL token/tag file reader. Looks for test.txt, eng.testb, testb.txt, *.conll, *.bio, or *.iob.",
                    "If no local file exists, the generated runner can materialize a small CoNLL-03 style split via datasets.",
                ],
            ),
            model=ModelSpec(name="env PAPER_BENCH_SEQUENCE_TAGGER_MODEL or ner", checkpoint_source="official"),
            command=command,
            command_kind="generated_runner" if root else "manual_protocol",
            expected_metrics=metric_specs_for_family("sequence_labeling"),
            parser={"type": "json_file", "path": "paper_benchmark_sequence_labeling_conll_results.json"},
            generated_script_name=script_name if root else None,
            generated_script_body=_conll03_script(root, will_download) if root else None,
            feasibility=(
                {"runnable": True, "data_source": root, "dataset_id": entry.dataset_id, "will_download": will_download}
                if root
                else {"runnable": False, "reason": missing_data_reason(entry.dataset_id, context.paper_slug, context.workspace_dir), "required_env": entry.env_var, "dataset_id": entry.dataset_id}
            ),
            fallback_reason="Use pretrained tagger sample inference when CoNLL-style benchmark data is unavailable.",
            evidence=[f"dataset:{entry.name}", f"required_env:{entry.env_var}"],
        )

    def _paper_table_protocol(self, context: AdapterContext) -> BenchmarkSpec:
        dataset = next((d for d in context.datasets if "conll" in d.lower()), None)
        return BenchmarkSpec(
            id="sequence_labeling_paper_table_protocol",
            task_family="sequence_labeling",
            level="L3",
            title="Paper-table sequence labeling protocol",
            dataset=DatasetSpec(
                name=dataset or "paper sequence labeling benchmark dataset",
                source="paper",
                size_estimate="external",
                public=None,
                notes=["Requires official train/dev/test split, tagging scheme, and span-level F1 protocol."],
            ),
            model=ModelSpec(name="paper sequence tagger checkpoint", checkpoint_source="paper"),
            command=[],
            command_kind="manual_protocol",
            expected_metrics=metric_specs_for_family("sequence_labeling"),
            feasibility={"runnable": False, "reason": "full sequence-labeling benchmark data/protocol is not bundled"},
            fallback_reason="Use pretrained tagger sample inference when full F1 benchmark is unavailable.",
            evidence=["paper_protocol_candidate"],
        )


def _looks_like_flair_or_sequence_labeling(context: AdapterContext) -> bool:
    text = " ".join([context.readme_text, " ".join(context.method_keywords), context.task or ""]).lower()
    return any(token in text for token in ("flair", "sequence labeling", "named entity", "ner", "tagger"))


def _sequence_labeling_script() -> str:
    script = '''
import importlib
import json
import os
import sys
from pathlib import Path

import torch

__EXTERNAL_DEPENDENCY_HELPER__

ensure_external_dependency("datasets")
import flair

try:
    if torch.cuda.is_available():
        flair.device = torch.device("cuda")
except Exception:
    pass

from flair.data import Sentence
from flair.models import SequenceTagger

sentence = Sentence("George Washington went to Washington.")
tagger = SequenceTagger.load("ner")
tagger.predict(sentence)

entities = []
for span in sentence.get_spans("ner"):
    entities.append({
        "text": span.text,
        "tag": span.tag,
        "score": float(span.score),
    })

payload = {
    "device": str(getattr(__import__("flair"), "device", "cpu")),
    "model": "ner",
    "sentence": sentence.to_plain_string(),
    "entities": entities,
    "num_entities": len(entities),
}
Path("paper_benchmark_sequence_labeling_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
'''
    return script.replace("__EXTERNAL_DEPENDENCY_HELPER__", _external_dependency_helper_script()).lstrip()


def _conll03_script(root: str, allow_download: bool = False) -> str:
    return f'''
import json
import importlib
import os
import signal
import sys
import zipfile
from pathlib import Path

import torch

{download_helper_script()}

{_external_dependency_helper_script()}

ensure_external_dependency("datasets")
import flair

try:
    if torch.cuda.is_available():
        flair.device = torch.device("cuda")
except Exception:
    pass

from flair.data import Sentence
from flair.models import SequenceTagger


def find_test_file(root, required=True):
    candidates = [
        "test.txt",
        "eng.testb",
        "eng.testb.txt",
        "testb.txt",
        "test.conll",
        "test.bio",
        "test.iob",
        "conll03_test.txt",
        "conll2003_test.txt",
    ]
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    ranked = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {{".txt", ".conll"}} and "test" in path.name.lower():
            ranked.append((0, path))
        elif path.is_file() and path.suffix.lower() in {{".bio", ".iob"}}:
            ranked.append((1, path))
    if ranked:
        return sorted(ranked, key=lambda item: (item[0], len(item[1].parts), item[1].name))[0][1]
    if required:
        raise SystemExit("No CoNLL-style test file found")
    return None


def timeout_call(label, seconds, fn):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return fn()

    def _raise_timeout(signum, frame):
        raise TimeoutError(f"{{label}} timed out after {{seconds:.0f}}s")

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(int(seconds))
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def safe_extract_zip(archive, dest):
    import os

    dest = Path(dest).resolve()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            target = (dest / name).resolve()
            if os.path.commonpath([str(dest), str(target)]) != str(dest):
                raise RuntimeError(f"blocked unsafe zip member: {{name}}")
        zf.extractall(dest)


def maybe_download_conll_url(root):
    url = os.environ.get("PAPER_BENCH_CONLL03_URL")
    if not url:
        return None
    target_name = url.rstrip("/").rsplit("/", 1)[-1] or "conll03_download"
    archive = root / target_name
    print(json.dumps({{"stage": "download_dataset", "dataset": "conll03", "target": str(root), "url": url}}), flush=True)
    download_with_progress(url, archive, "conll03")
    lower = archive.name.lower()
    if lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        print(json.dumps({{"stage": "extract_dataset", "archive": str(archive)}}), flush=True)
        safe_extract_tar(archive, root)
    elif lower.endswith(".zip"):
        print(json.dumps({{"stage": "extract_dataset", "archive": str(archive)}}), flush=True)
        safe_extract_zip(archive, root)
    elif archive.suffix.lower() in {{".txt", ".conll", ".bio", ".iob"}}:
        target = root / "test.txt"
        if archive != target:
            target.write_bytes(archive.read_bytes())
    return find_test_file(root, required=False)


def materialize_hf_conll03(root):
    datasets = ensure_external_dependency("datasets")
    if datasets is None:
        raise ImportError("Hugging Face datasets is required to materialize CoNLL-03")
    dataset_name = os.environ.get("PAPER_BENCH_CONLL03_DATASET", "conll2003")
    split_name = os.environ.get("PAPER_BENCH_CONLL03_SPLIT", "test")
    trust_remote_code = os.environ.get("PAPER_BENCH_HF_TRUST_REMOTE_CODE", "false").lower() in {{"1", "true", "yes", "on"}}
    print(json.dumps({{"stage": "download_dataset", "dataset": dataset_name, "split": split_name, "target": str(root)}}), flush=True)
    ds = datasets.load_dataset(dataset_name, trust_remote_code=trust_remote_code)
    if split_name not in ds:
        split_name = "validation" if "validation" in ds else list(ds.keys())[0]
    split = ds[split_name]
    label_feature = split.features.get("ner_tags")
    label_names = getattr(getattr(label_feature, "feature", None), "names", None)
    out_path = root / ("eng.testb" if split_name == "test" else f"{{split_name}}.conll")
    with out_path.open("w", encoding="utf-8") as out:
        for row in split:
            tokens = row.get("tokens") or row.get("words")
            tags = row.get("ner_tags") or row.get("tags")
            if not tokens or tags is None:
                continue
            for token, tag in zip(tokens, tags):
                tag_name = label_names[tag] if label_names and isinstance(tag, int) else str(tag)
                out.write(f"{{token}} X X {{tag_name}}\\n")
            out.write("\\n")
    print(json.dumps({{"stage": "download_complete", "dataset": dataset_name, "split": split_name, "target": str(out_path)}}), flush=True)
    return out_path


def read_conll(path, max_sentences):
    sentences = []
    tokens = []
    tags = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("-DOCSTART-"):
            if tokens:
                sentences.append((tokens, tags))
                tokens, tags = [], []
                if len(sentences) >= max_sentences:
                    break
            continue
        parts = line.split()
        if len(parts) >= 2:
            tokens.append(parts[0])
            tags.append(parts[-1])
    if tokens and len(sentences) < max_sentences:
        sentences.append((tokens, tags))
    return sentences


def spans(tags):
    out = set()
    start = None
    label = None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-") or tag == "O" or (tag.startswith("I-") and label and tag[2:] != label):
            if label is not None:
                out.add((start, i, label))
            label = None
            start = None
        if tag.startswith("B-"):
            label = tag[2:]
            start = i
        elif tag.startswith("I-") and label is None:
            label = tag[2:]
            start = i
    return out


root = Path({root!r})
ensure_external_dependency("datasets")
if {allow_download!r}:
    root.mkdir(parents=True, exist_ok=True)
    if find_test_file(root, required=False) is None:
        timeout_seconds = float(os.environ.get("PAPER_BENCH_DATA_DOWNLOAD_TIMEOUT_SECONDS", "900"))
        try:
            test_candidate = maybe_download_conll_url(root)
            if test_candidate is None:
                test_candidate = timeout_call("CoNLL-03 datasets materialization", timeout_seconds, lambda: materialize_hf_conll03(root))
            print(json.dumps({{"stage": "dataset_ready", "dataset": "conll03", "test_file": str(test_candidate)}}), flush=True)
        except Exception as exc:
            raise SystemExit(f"Failed to prepare CoNLL-03 data in {{root}}: {{exc}}")

max_sentences = int(os.environ.get("PAPER_BENCH_MAX_EXAMPLES", "500"))
model_name = os.environ.get("PAPER_BENCH_SEQUENCE_TAGGER_MODEL", "ner")
test_file = find_test_file(root)
data = read_conll(test_file, max_sentences)
tagger = SequenceTagger.load(model_name)

tp = pred_total = gold_total = 0
for tokens, gold_tags in data:
    sentence = Sentence(" ".join(tokens), use_tokenizer=False)
    tagger.predict(sentence)
    pred_tags = ["O"] * len(tokens)
    for span in sentence.get_spans("ner"):
        pred_tags[span.start_position:span.end_position]
        start_idx = max(0, span.tokens[0].idx - 1)
        end_idx = min(len(tokens), span.tokens[-1].idx)
        if start_idx < end_idx:
            pred_tags[start_idx] = "B-" + span.tag
            for idx in range(start_idx + 1, end_idx):
                pred_tags[idx] = "I-" + span.tag
    gold_spans = spans(gold_tags)
    pred_spans = spans(pred_tags)
    tp += len(gold_spans & pred_spans)
    pred_total += len(pred_spans)
    gold_total += len(gold_spans)

precision = 100.0 * tp / max(1, pred_total)
recall = 100.0 * tp / max(1, gold_total)
f1 = 2 * precision * recall / max(1e-12, precision + recall)
payload = {{
    "device": str(getattr(__import__("flair"), "device", "cpu")),
    "model": model_name,
    "dataset_root": str(root),
    "test_file": str(test_file),
    "num_sentences": len(data),
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
}}
Path("paper_benchmark_sequence_labeling_conll_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
'''.lstrip()


def _external_dependency_helper_script() -> str:
    return r'''
def ensure_external_dependency(module_name):
    """Import a site-packages dependency even when the target repo has a same-named package."""
    repo_root = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else repo_root

    def _under_repo(module):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return False
        try:
            path = Path(module_file).resolve()
            return os.path.commonpath([str(repo_root), str(path)]) == str(repo_root)
        except Exception:
            return False

    existing = sys.modules.get(module_name)
    if existing is not None and not _under_repo(existing):
        return existing

    removed_entries = []
    for entry in list(sys.path):
        candidate = Path(entry or ".").resolve()
        if candidate in {repo_root, script_dir}:
            removed_entries.append((sys.path.index(entry), entry))
            sys.path.remove(entry)

    old_module = sys.modules.pop(module_name, None) if existing is not None else None
    try:
        module = importlib.import_module(module_name)
        sys.modules[module_name] = module
        return module
    except Exception:
        if old_module is not None:
            sys.modules[module_name] = old_module
        return None
    finally:
        for index, entry in sorted(removed_entries):
            sys.path.insert(min(index, len(sys.path)), entry)
'''.strip()
