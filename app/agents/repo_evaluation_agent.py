from __future__ import annotations

from pathlib import Path

from app.core.file_utils import save_json
from app.core.progress import emit_progress
from app.core.cancellation import PipelineCancelled
from app.core.state import TaskState, RepoEvaluation
from app.tools.llm import call_llm_json
from app.tools.repo_tool import (
    clone_repo,
    copy_local_repo,
    scan_repo_structure,
    compute_runnable_score,
    detect_risk_flags,
)


class RepoEvaluationAgent:
    def run(self, state: TaskState) -> TaskState:
        if not state.selected_repo:
            state.errors.append({"agent": "RepoEvaluationAgent", "error": "No selected repo"})
            state.status = "failed"
            emit_progress("Evaluate repo", "no selected repository", level="error")
            return state

        task_dir = Path(state.task_dir)
        repo_dir = task_dir / "repos" / "cloned_repo"

        try:
            if state.selected_repo.source == "local":
                emit_progress("Evaluate repo", "正在复制本地代码仓库",
                              detail=state.selected_repo.local_path,
                              repo_dir=str(repo_dir))
                copy_local_repo(Path(state.selected_repo.local_path), repo_dir)
            else:
                emit_progress("Evaluate repo", "正在克隆代码仓库",
                              detail=f"仓库地址：{state.selected_repo.url}\n目标目录：{repo_dir}",
                              repo_url=state.selected_repo.url,
                              repo_dir=str(repo_dir))
                clone_repo(state.selected_repo.url, repo_dir)

            emit_progress("Evaluate repo", "正在扫描仓库结构",
                          detail=f"仓库目录：{repo_dir}\n检查 README、requirements、入口脚本等文件……")
            scan = scan_repo_structure(repo_dir)

            # Build scan summary
            scan_summary: list[str] = []
            scan_summary.append(f"README：{'已发现' if scan.get('has_readme') else '未发现'}")
            req_files = scan.get("requirement_files", [])
            scan_summary.append(f"依赖文件：{', '.join(req_files[:5]) if req_files else '未发现'}")
            scan_summary.append(f"environment.yml：{'已发现' if scan.get('has_environment_yml') else '未发现'}")
            scan_summary.append(f"Dockerfile：{'已发现' if scan.get('has_dockerfile') else '未发现'}")
            scripts = scan.get("candidate_scripts", [])
            scan_summary.append(f"候选入口脚本：{', '.join(scripts[:8]) if scripts else '未发现'}")
            scan_summary.append(f"候选配置文件：{len(scan.get('candidate_configs', []))} 个")

            emit_progress(
                "Evaluate repo",
                "仓库结构扫描完成",
                detail="\n".join(f"  - {x}" for x in scan_summary),
                log_lines=scan_summary,
                candidate_script_count=len(scripts),
            )

            emit_progress("Evaluate repo", "检测风险标记",
                          detail="检查 GPU 需求、权重文件、数据集依赖等……")
            risk_flags = detect_risk_flags(repo_dir, scan)
            if risk_flags:
                emit_progress("Evaluate repo", "发现潜在风险",
                              level="warning",
                              detail="\n".join(f"  - {x}" for x in risk_flags[:5]),
                              log_lines=[f"风险：{x}" for x in risk_flags])
            else:
                emit_progress("Evaluate repo", "未发现明显风险标记")

            emit_progress("Evaluate repo", "正在调用 LLM 分析入口脚本",
                          detail="将仓库结构、README 摘要、候选脚本传给模型，提取推荐运行命令和风险点。")
            benchmark_surface = _llm_analyze_benchmark_surface(repo_dir, scan, risk_flags)
            if benchmark_surface:
                demo_cmds = benchmark_surface.get("demo_commands", [])
                eval_cmds = benchmark_surface.get("official_eval_commands", [])
                metrics = benchmark_surface.get("likely_metrics", [])
                conf = benchmark_surface.get("confidence", 0)
                llm_summary = (
                    f"LLM 仓库分析完成\n"
                    f"  - 推荐 demo 命令：{', '.join(demo_cmds[:3]) if demo_cmds else '未识别'}\n"
                    f"  - 推荐评估命令：{', '.join(eval_cmds[:3]) if eval_cmds else '未识别'}\n"
                    f"  - 可能产出指标：{', '.join(metrics[:5]) if metrics else '未识别'}\n"
                    f"  - 置信度：{conf:.2f}"
                )
                emit_progress("Evaluate repo", "LLM 分析完成", detail=llm_summary)

            evaluation = RepoEvaluation(
                repo_dir=str(repo_dir),
                **scan,
                runnable_score=compute_runnable_score(scan),
                risk_flags=risk_flags,
                benchmark_surface=benchmark_surface or {},
            )

            state.repo_evaluation = evaluation
            save_json(task_dir / "evaluation" / "repo_score.json", evaluation)
            state.status = "repo_evaluated"
            emit_progress(
                "Evaluate repo",
                "repo evaluation saved",
                detail=f"runnable_score={evaluation.runnable_score:.2f}",
                runnable_score=evaluation.runnable_score,
                repo_dir=str(repo_dir),
            )

        except PipelineCancelled:
            state.status = "cancelled"
            emit_progress("Evaluate repo", "cancelled", level="warning", phase="fail")
            raise
        except Exception as e:
            state.errors.append({"agent": "RepoEvaluationAgent", "error": str(e)})
            state.status = "failed"
            emit_progress("Evaluate repo", "repo evaluation error", level="error", detail=str(e))

        return state


_SURFACE_PROMPT = """\
You are a repository benchmark surface analysis assistant.

Given a research repository scan, README excerpt, and script excerpts, identify
official benchmark/evaluation surfaces without creating paper-specific recipes.

Return a JSON object with exactly these keys:
- "official_eval_commands": list of command strings likely to run official eval/benchmark
- "demo_commands": list of command strings likely to run demos/examples
- "dataset_requirements": list of required datasets or paths
- "weight_requirements": list of required weights/checkpoints
- "likely_metrics": list of metric names likely produced by eval scripts
- "benchmark_files": list of files that appear to define benchmark protocol
- "confidence": float between 0 and 1
"""


def _llm_analyze_benchmark_surface(repo_dir: Path, scan: dict, risk_flags: list[str]) -> dict | None:
    readme = _read_readme(repo_dir)
    script_excerpts = []
    for script in scan.get("candidate_scripts", [])[:12]:
        path = repo_dir / script
        if not path.exists() or not path.is_file():
            continue
        script_excerpts.append({
            "path": script,
            "excerpt": path.read_text(encoding="utf-8", errors="ignore")[:1600],
        })

    payload = {
        "scan": scan,
        "risk_flags": risk_flags,
        "readme_excerpt": readme[:6000],
        "script_excerpts": script_excerpts,
    }
    return call_llm_json(
        system_prompt=_SURFACE_PROMPT,
        user_prompt=json_dumps(payload),
        purpose="repo_benchmark_surface_analysis",
        max_tokens=3072,
    )


def _read_readme(repo_dir: Path) -> str:
    for name in ("README.md", "README.rst", "README.txt", "README"):
        path = repo_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def json_dumps(value: dict) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
