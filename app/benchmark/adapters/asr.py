from __future__ import annotations

from app.benchmark.adapters.base import AdapterContext, find_sample_files
from app.benchmark.adapters.downloads import download_helper_script
from app.benchmark.dataset_registry import data_root, dataset_entry, dataset_size_gb, expected_data_root, missing_data_reason
from app.benchmark.ontology import metric_specs_for_family
from app.benchmark.schema import BenchmarkSpec, DatasetSpec, ModelSpec


class ASRAdapter:
    task_family = "asr"

    def propose_benchmarks(self, context: AdapterContext) -> list[BenchmarkSpec]:
        specs = [self._paper_table_protocol(context)]
        specs.append(self._librispeech_protocol(context))
        audio = _audio_sample(context)
        if audio:
            script_name = "paper_benchmark_asr_sample.py"
            specs.append(BenchmarkSpec(
                id="asr_bundled_audio_transcription",
                task_family="asr",
                level="L1",
                title="Bundled audio transcription sanity benchmark",
                dataset=DatasetSpec(name=audio, source="bundled", size_estimate="tiny", public=True),
                model=ModelSpec(name="tiny", checkpoint_source="official"),
                command=["python", script_name],
                command_kind="generated_runner",
                expected_metrics=metric_specs_for_family("asr"),
                parser={"type": "json_file", "path": "paper_benchmark_asr_results.json"},
                reference={
                    "source": "generated bundled-audio transcription run",
                    "scope": "single_audio_demo_not_full_asr_benchmark",
                    "metrics": {},
                },
                generated_script_name=script_name,
                generated_script_body=_asr_script(audio),
                evidence=[f"sample_audio:{audio}", "metric_family:wer_cer"],
            ))
        return specs

    def _librispeech_protocol(self, context: AdapterContext) -> BenchmarkSpec:
        entry = dataset_entry("librispeech")
        existing_root = data_root("librispeech", context.paper_slug, context.workspace_dir)
        size_gb = dataset_size_gb("librispeech", existing_root)
        will_download = False
        root = existing_root
        if (
            root is None
            and entry.auto_download
            and entry.estimated_size_gb is not None
            and entry.estimated_size_gb <= context.budget.max_dataset_size_gb
        ):
            expected_root = expected_data_root("librispeech", context.paper_slug, context.workspace_dir)
            if expected_root is not None:
                root = str(expected_root)
                will_download = True
        script_name = "paper_benchmark_asr_librispeech.py"
        command = ["python", script_name] if root else []
        return BenchmarkSpec(
            id="asr_librispeech_eval",
            task_family="asr",
            level="L3",
            title="LibriSpeech ASR WER/CER benchmark",
            dataset=DatasetSpec(
                name=entry.name,
                source="official_repo",
                size_estimate=entry.size_estimate,
                size_gb=size_gb,
                public=entry.public,
                notes=["Generic ASR benchmark runner over LibriSpeech-style .trans.txt references."],
            ),
            model=ModelSpec(
                name="env PAPER_BENCH_ASR_MODEL or tiny",
                checkpoint_source="official",
                notes=["Model size is configurable to separate protocol wiring from resource budget."],
            ),
            command=command,
            command_kind="generated_runner" if root else "manual_protocol",
            expected_metrics=metric_specs_for_family("asr"),
            parser={"type": "json_file", "path": "paper_benchmark_asr_librispeech_results.json"},
            generated_script_name=script_name if root else None,
            generated_script_body=_librispeech_script(root, will_download) if root else None,
            feasibility=(
                {"runnable": True, "data_source": root, "dataset_id": entry.dataset_id, "will_download": will_download}
                if root
                else {"runnable": False, "reason": missing_data_reason(entry.dataset_id, context.paper_slug, context.workspace_dir), "required_env": entry.env_var, "dataset_id": entry.dataset_id}
            ),
            fallback_reason="Use bundled audio transcription when LibriSpeech references are unavailable.",
            evidence=[f"dataset:{entry.name}", f"required_env:{entry.env_var}"],
        )

    def _paper_table_protocol(self, context: AdapterContext) -> BenchmarkSpec:
        datasets = context.datasets or ["paper ASR benchmark dataset"]
        return BenchmarkSpec(
            id="asr_paper_table_protocol",
            task_family="asr",
            level="L3",
            title="Paper-table ASR benchmark protocol",
            dataset=DatasetSpec(
                name=", ".join(datasets[:4]),
                source="paper",
                size_estimate="large_or_external",
                public=None,
                notes=["Requires official ASR dataset split, references, normalization, and WER/CER protocol."],
            ),
            model=ModelSpec(name="paper ASR checkpoint", checkpoint_source="paper"),
            command=[],
            command_kind="manual_protocol",
            expected_metrics=metric_specs_for_family("asr"),
            feasibility={"runnable": False, "reason": "full ASR references/datasets are not bundled"},
            fallback_reason="Use bundled audio transcription when full WER/CER benchmark is unavailable.",
            evidence=["paper_protocol_candidate"],
        )


def _audio_sample(context: AdapterContext) -> str | None:
    samples = find_sample_files(context, {".wav", ".mp3", ".flac"}, limit=1)
    return samples[0] if samples else None


def _asr_script(audio: str) -> str:
    return f'''
import json
import os
import shutil
from pathlib import Path

import torch
import whisper

try:
    import imageio_ffmpeg
    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    shim_dir = Path(".paper_benchmark_bin")
    shim_dir.mkdir(exist_ok=True)
    shim = shim_dir / "ffmpeg"
    if not shim.exists():
        try:
            shim.symlink_to(ffmpeg_exe)
        except Exception:
            shutil.copy2(ffmpeg_exe, shim)
            shim.chmod(0o755)
    os.environ["PATH"] = str(shim_dir.resolve()) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("tiny", device=device)
result = model.transcribe({audio!r})
payload = {{
    "device": device,
    "model": "tiny",
    "audio": {audio!r},
    "text": result.get("text", "").strip(),
    "language": result.get("language"),
}}
Path("paper_benchmark_asr_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
'''.lstrip()


def _librispeech_script(root: str, allow_download: bool = False) -> str:
    return f'''
import json
import os
import shutil
import time
from pathlib import Path

import torch
import whisper

{download_helper_script()}

def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]


def normalize(text):
    import re
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


try:
    import imageio_ffmpeg
    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    shim_dir = Path(".paper_benchmark_bin")
    shim_dir.mkdir(exist_ok=True)
    shim = shim_dir / "ffmpeg"
    if not shim.exists():
        try:
            shim.symlink_to(ffmpeg_exe)
        except Exception:
            shutil.copy2(ffmpeg_exe, shim)
            shim.chmod(0o755)
    os.environ["PATH"] = str(shim_dir.resolve()) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

root = Path({root!r})
if {allow_download!r}:
    root.mkdir(parents=True, exist_ok=True)
    if not any(root.rglob("*.flac")):
        archive = root / "test-clean.tar.gz"
        url = os.environ.get("PAPER_BENCH_LIBRISPEECH_URL", "https://www.openslr.org/resources/12/test-clean.tar.gz")
        try:
            print(json.dumps({{"stage": "download_dataset", "dataset": "librispeech-test-clean", "target": str(root), "url": url}}), flush=True)
            download_with_progress(url, archive, "librispeech-test-clean")
            print(json.dumps({{"stage": "extract_dataset", "archive": str(archive)}}), flush=True)
            safe_extract_tar(archive, root)
        except Exception as exc:
            raise SystemExit(f"Failed to download LibriSpeech test-clean into {{root}}: {{exc}}")

max_items = int(os.environ.get("PAPER_BENCH_MAX_EXAMPLES", "200"))
model_name = os.environ.get("PAPER_BENCH_ASR_MODEL", "tiny")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(model_name, device=device)

references = {{}}
for trans_path in root.rglob("*.trans.txt"):
    for line in trans_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            references[parts[0]] = parts[1]

items = []
for audio in sorted(root.rglob("*.flac")):
    ref = references.get(audio.stem)
    if ref:
        items.append((audio, ref))
    if len(items) >= max_items:
        break

if not items:
    raise SystemExit("No LibriSpeech .flac files with matching .trans.txt references found")

word_err = word_total = char_err = char_total = 0
start = time.time()
for audio, ref in items:
    hyp = model.transcribe(str(audio)).get("text", "")
    ref_words = normalize(ref)
    hyp_words = normalize(hyp)
    word_err += edit_distance(ref_words, hyp_words)
    word_total += len(ref_words)
    ref_chars = list(" ".join(ref_words))
    hyp_chars = list(" ".join(hyp_words))
    char_err += edit_distance(ref_chars, hyp_chars)
    char_total += len(ref_chars)

elapsed = time.time() - start
payload = {{
    "device": device,
    "model": model_name,
    "dataset_root": str(root),
    "num_examples": len(items),
    "WER": 100.0 * word_err / max(1, word_total),
    "CER": 100.0 * char_err / max(1, char_total),
    "runtime_seconds": elapsed,
}}
Path("paper_benchmark_asr_librispeech_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
'''.lstrip()
