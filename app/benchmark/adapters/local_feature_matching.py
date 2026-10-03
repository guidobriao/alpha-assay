from __future__ import annotations

from app.benchmark.adapters.base import AdapterContext, first_existing_script
from app.benchmark.dataset_registry import (
    data_root,
    dataset_entry,
    dataset_size_gb,
    estimate_dataset_size_from_context,
    missing_data_reason,
)
from app.benchmark.ontology import metric_specs_for_family
from app.benchmark.schema import BenchmarkSpec, DatasetSpec, ModelSpec


class LocalFeatureMatchingAdapter:
    task_family = "local_feature_matching"

    def propose_benchmarks(self, context: AdapterContext) -> list[BenchmarkSpec]:
        specs: list[BenchmarkSpec] = [self._paper_table_protocol(context)]

        specs.extend(self._official_module_eval_protocols(context))

        if first_existing_script(context, ["match_pairs.py"]):
            specs.append(BenchmarkSpec(
                id="local_feature_matching_official_pair_eval",
                task_family="local_feature_matching",
                level="L2",
                title="Official bundled image-pair evaluation",
                dataset=DatasetSpec(
                    name="official bundled image pairs with ground truth",
                    source="bundled",
                    size_estimate="small",
                    public=True,
                    notes=["Discovered match_pairs.py with evaluation-style metrics."],
                ),
                model=ModelSpec(name="repo default matcher", checkpoint_source="official"),
                command=[
                    "python",
                    "match_pairs.py",
                    "--eval",
                    "--output_dir",
                    "paper_benchmark_local_feature_matching",
                ],
                expected_metrics=metric_specs_for_family("local_feature_matching"),
                parser={"type": "local_feature_table"},
                reference={
                    "source": "official repository bundled sample evaluation",
                    "scope": "sample_eval_not_full_paper_dataset",
                    "metrics": {
                        "AUC@5": 26.99,
                        "AUC@10": 48.40,
                        "AUC@20": 64.47,
                        "Prec": 73.52,
                        "MScore": 19.60,
                    },
                },
                evidence=["script:match_pairs.py", "metric_family:pose_auc_precision_matching_score"],
            ))

        if first_existing_script(context, ["benchmark.py"]):
            specs.append(BenchmarkSpec(
                id="local_feature_matching_speed_benchmark",
                task_family="local_feature_matching",
                level="L2",
                title="Official matcher speed benchmark",
                dataset=DatasetSpec(
                    name="official benchmark image pair(s)",
                    source="official_repo",
                    size_estimate="small",
                    public=True,
                    notes=["Discovered benchmark.py with speed/latency benchmark signal."],
                ),
                model=ModelSpec(name="repo default matcher", checkpoint_source="repo_default"),
                command=[
                    "python",
                    "benchmark.py",
                    "--device",
                    "cuda",
                    "--repeat",
                    "5",
                    "--num_keypoints",
                    "512",
                    "1024",
                    "--save",
                    "paper_benchmark_local_feature_matching.png",
                ],
                expected_metrics=metric_specs_for_family("local_feature_matching"),
                parser={"type": "local_feature_speed_table"},
                reference={
                    "source": "paper or README speed claims when available",
                    "scope": "hardware_dependent_speed_benchmark",
                    "metrics": {},
                    "notes": ["Speed metrics are compared with hardware/config metadata rather than strict parity by default."],
                },
                evidence=["script:benchmark.py", "metric_family:fps_latency"],
            ))

        script = first_existing_script(context, ["eval.py", "evaluate.py", "minimal_example.py", "demo.py"])
        if script:
            specs.append(BenchmarkSpec(
                id=f"local_feature_matching_{script.replace('/', '_').replace('.', '_')}",
                task_family="local_feature_matching",
                level="L1",
                title="Repository local-feature example or evaluation entry point",
                dataset=DatasetSpec(name="repo bundled or default sample", source="readme", size_estimate="small"),
                model=ModelSpec(name="repo default matcher", checkpoint_source="repo_default"),
                command=["python", script],
                expected_metrics=metric_specs_for_family("local_feature_matching"),
                parser={"type": "generic_metrics"},
                evidence=[f"script:{script}"],
            ))

        return specs

    def _official_module_eval_protocols(self, context: AdapterContext) -> list[BenchmarkSpec]:
        specs: list[BenchmarkSpec] = []
        module_specs = [
            (
                "modules/eval/megadepth1500.py",
                "official_megadepth1500_pose_eval",
                "Official MegaDepth-1500 pose evaluation module",
                "megadepth1500",
                ["python", "-m", "modules.eval.megadepth1500", "--matcher", "xfeat", "--ransac-thr", "2.5"],
            ),
            (
                "modules/eval/scannet1500.py",
                "official_scannet1500_pose_eval",
                "Official ScanNet-1500 pose evaluation module",
                "scannet1500",
                ["python", "-m", "modules.eval.scannet1500"],
            ),
        ]
        for path, spec_id, title, dataset_id, base_command in module_specs:
            if not (context.repo_dir / path).exists():
                continue
            entry = dataset_entry(dataset_id)
            dataset_root = data_root(dataset_id, context.paper_slug, context.workspace_dir)
            size_gb = dataset_size_gb(dataset_id, dataset_root)
            output_dir = f"paper_benchmark_{spec_id}"
            if dataset_root and spec_id == "official_megadepth1500_pose_eval":
                command = [*base_command, "--dataset-dir", dataset_root]
                feasibility = {
                    "runnable": True,
                    "data_source": dataset_root,
                    "dataset_id": dataset_id,
                }
            elif dataset_root and spec_id == "official_scannet1500_pose_eval":
                command = [*base_command, "--scannet_path", dataset_root, "--output", output_dir]
                feasibility = {
                    "runnable": True,
                    "data_source": dataset_root,
                    "dataset_id": dataset_id,
                }
            else:
                command = []
                feasibility = {
                    "runnable": False,
                    "reason": f"official eval module {path} requires external benchmark data. {missing_data_reason(dataset_id, context.paper_slug, context.workspace_dir)}",
                    "expected_command": _command_template(base_command, spec_id),
                    "required_env": entry.env_var,
                    "dataset_id": dataset_id,
                }
            specs.append(BenchmarkSpec(
                id=spec_id,
                task_family="local_feature_matching",
                level="L3",
                title=title,
                dataset=DatasetSpec(
                    name=entry.name,
                    source="official_repo",
                    size_estimate=entry.size_estimate,
                    size_gb=size_gb,
                    public=entry.public,
                    notes=["Official evaluation module discovered, but dataset path is not bundled."],
                ),
                model=ModelSpec(name="repo default matcher", checkpoint_source="official"),
                command=command,
                command_kind="official_script" if command else "manual_protocol",
                expected_metrics=metric_specs_for_family("local_feature_matching"),
                parser={"type": "xfeat_pose_eval"},
                feasibility=feasibility,
                fallback_reason="Use bundled/minimal local feature example until the official evaluation dataset is available.",
                evidence=[f"script:{path}", f"dataset:{entry.name}", f"required_env:{entry.env_var}"],
            ))
        return specs

    def _paper_table_protocol(self, context: AdapterContext) -> BenchmarkSpec:
        datasets = context.datasets or ["paper main benchmark dataset"]
        estimate = estimate_dataset_size_from_context(
            task_family="local_feature_matching",
            datasets=context.datasets,
            text="\n".join([context.readme_text, context.task or "", " ".join(context.method_keywords)]),
        )
        dataset_name = ", ".join(datasets[:4])
        size_estimate = "large_or_external"
        size_gb = None
        notes = ["Requires full paper dataset/split and exact protocol extraction."]
        evidence = ["paper_protocol_candidate"]
        if estimate:
            dataset_name = ", ".join(estimate.names)
            size_estimate = estimate.size_estimate
            size_gb = estimate.estimated_size_gb
            notes.append("Dataset size estimated from deterministic registry alias matching.")
            notes.extend(estimate.evidence)
            evidence.extend(f"dataset_size_estimate:{item}" for item in estimate.evidence)
        return BenchmarkSpec(
            id="local_feature_matching_paper_table_protocol",
            task_family="local_feature_matching",
            level="L3",
            title="Paper-table local feature matching protocol",
            dataset=DatasetSpec(
                name=dataset_name,
                source="paper",
                size_estimate=size_estimate,
                size_gb=size_gb,
                public=None,
                notes=notes,
            ),
            model=ModelSpec(name="paper checkpoint/config", checkpoint_source="paper"),
            command=[],
            command_kind="manual_protocol",
            expected_metrics=metric_specs_for_family("local_feature_matching"),
            feasibility={
                "runnable": False,
                "reason": "full paper-table protocol requires dataset and reference extraction beyond bundled assets",
            },
            fallback_reason="L3 target recorded; planner will use the highest runnable official/bundled protocol.",
            evidence=evidence,
        )


def _command_template(base_command: list[str], spec_id: str) -> str:
    if spec_id == "official_megadepth1500_pose_eval":
        return " ".join([*base_command, "--dataset-dir", "$PAPER_BENCH_MEGDEPTH1500_DIR"])
    if spec_id == "official_scannet1500_pose_eval":
        return " ".join([*base_command, "--scannet_path", "$PAPER_BENCH_SCANNET1500_DIR", "--output", "paper_benchmark_official_scannet1500_pose_eval"])
    return " ".join(base_command)
