from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime

import jinja2

from app.core.file_utils import save_json
from app.core.progress import emit_progress
from app.core.state import TaskState, ReportResult
from app.tools.llm import call_llm_json, telemetry_records

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

_SYSTEM_PROMPT = """\
你是一个论文复现助手。根据下面的 smoke 测试结果，生成简洁的结论和可操作的建议。

必须使用简体中文回复，返回 JSON 对象：
- "conclusion": 1-3 句自然语言总结，说明整体结果，具体指出哪些步骤成功、哪些失败。
- "next_steps": 3-5 条具体的建议，告诉用户下一步应该做什么。尽量引用具体的文件路径或命令。

语言要求：
- 除文件路径、命令、环境变量、状态码、指标名和专有名词外，所有自然语言必须是简体中文。
- 不要输出英文段落、英文解释或中英混杂的建议。
- 如果原始日志或状态字段是英文，请先理解后用中文转述。

要求简洁实用，不要猜测。"""


_SMOKE_ENV_FAILURE_TYPES = {
    "missing_dependency",
    "cuda_error",
    "runtime_linker_error",
}


class ReportWriterAgent:
    def run(self, state: TaskState) -> TaskState:
        task_dir = Path(state.task_dir)
        report_dir = task_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)

        final_status = self._determine_final_status(state)
        emit_progress("Write report", "determined final status", detail=final_status, final_status=final_status)

        # Try LLM for conclusion and next steps
        emit_progress("Write report", "generating report insights", detail="LLM summary and next steps")
        llm_result = self._llm_generate_insights(state, final_status)
        if llm_result and _is_chinese_report_insight(llm_result):
            short_conclusion = llm_result.get("conclusion", self._short_conclusion(final_status))
            next_steps = llm_result.get("next_steps", self._next_steps(final_status))
            emit_progress("Write report", "LLM insights accepted", detail=short_conclusion)
        else:
            logger.info("LLM unavailable or failed, falling back to template conclusion")
            short_conclusion = self._short_conclusion(final_status)
            next_steps = self._next_steps(final_status)
            emit_progress("Write report", "using template report insights", level="warning", detail=short_conclusion)

        emit_progress("Write report", "rendering report template")
        template = jinja2.Template(
            (_TEMPLATES_DIR / "smoke_report.md.j2").read_text(encoding="utf-8")
        )

        report_content = template.render(
            timestamp=datetime.now().isoformat(),
            task_id=state.task_id,
            final_status=final_status,
            backend=state.backend,
            short_conclusion=short_conclusion,
            next_steps=next_steps,
            input=state.paper_input,
            paper=state.paper_metadata,
            brief=state.reproduction_brief,
            selected_repo=state.selected_repo,
            repo_candidates=state.repo_candidates[:5],
            repo_eval=state.repo_evaluation,
            env_build=state.env_build,
            smoke_run=state.smoke_run,
            reproduction_run=state.reproduction_run,
            benchmark_plan=state.benchmark_plan,
            benchmark_run=state.benchmark_run,
            errors=state.errors,
            step_timings=state.step_timings,
            api_calls=telemetry_records() or state.api_calls,
        )

        md_path = report_dir / "reproduction_smoke_report.md"
        json_path = report_dir / "reproduction_smoke_report.json"

        md_path.write_text(report_content, encoding="utf-8")
        emit_progress("Write report", "wrote markdown report", detail=str(md_path))

        report = ReportResult(
            final_status=final_status,
            report_markdown_path=str(md_path),
            report_json_path=str(json_path),
            short_conclusion=short_conclusion,
        )

        state.report = report
        save_json(json_path, report)
        emit_progress("Write report", "wrote report metadata", detail=str(json_path))

        state.status = "report_written"
        return state

    def _llm_generate_insights(self, state: TaskState, final_status: str) -> dict | None:
        lines = [
            "请基于以下结构化执行结果生成报告总结和下一步建议。",
            "请只输出中文自然语言；路径、命令、环境变量、状态码、指标名和专有名词可以保留原文。",
            f"最终状态: {final_status}",
        ]

        if state.paper_metadata and state.paper_metadata.title:
            lines.append(f"论文标题: {state.paper_metadata.title}")

        if state.selected_repo:
            lines.append(f"代码仓库: {state.selected_repo.url}")

        if state.repo_evaluation:
            lines.append(f"可运行分数: {state.repo_evaluation.runnable_score}")
            if state.repo_evaluation.risk_flags:
                lines.append(f"风险标记: {', '.join(state.repo_evaluation.risk_flags)}")
            if state.repo_evaluation.benchmark_surface:
                lines.append(f"Benchmark 入口分析: {state.repo_evaluation.benchmark_surface}")

        if state.env_build:
            lines.append(f"环境构建成功: {state.env_build.build_success}")
            if state.env_build.failure_summary:
                lines.append(f"环境构建失败摘要: {state.env_build.failure_summary}")

        if state.smoke_run:
            if state.smoke_run.command:
                lines.append(f"Smoke 执行命令: {state.smoke_run.command.display}")
            lines.append(f"Smoke 是否成功: {state.smoke_run.success}")
            if state.smoke_run.exit_code is not None:
                lines.append(f"退出码: {state.smoke_run.exit_code}")
            if state.smoke_run.summary:
                lines.append(f"Smoke 摘要: {state.smoke_run.summary}")
            if state.smoke_run.failure_type:
                lines.append(f"Smoke 失败类型: {state.smoke_run.failure_type}")
            if state.smoke_run.failure_evidence:
                lines.append(f"Smoke 失败证据: {state.smoke_run.failure_evidence}")

        if state.reproduction_run:
            if state.reproduction_run.command:
                lines.append(f"轻量复现命令: {state.reproduction_run.command.display}")
            lines.append(f"轻量复现是否适合尝试: {state.reproduction_run.eligible}")
            lines.append(f"轻量复现是否跳过: {state.reproduction_run.skipped}")
            lines.append(f"轻量复现是否成功: {state.reproduction_run.success}")
            if state.reproduction_run.skip_reason:
                lines.append(f"轻量复现跳过原因: {state.reproduction_run.skip_reason}")
            if state.reproduction_run.summary:
                lines.append(f"轻量复现摘要: {state.reproduction_run.summary}")
            if state.reproduction_run.failure_type:
                lines.append(f"轻量复现失败类型: {state.reproduction_run.failure_type}")
            if state.reproduction_run.failure_evidence:
                lines.append(f"轻量复现失败证据: {state.reproduction_run.failure_evidence}")
            if state.reproduction_run.output_artifacts:
                lines.append(f"轻量复现输出产物: {state.reproduction_run.output_artifacts[:10]}")
            if state.reproduction_run.metrics:
                lines.append(f"轻量复现指标: {state.reproduction_run.metrics}")
            if state.reproduction_run.reference_results:
                lines.append(f"参考结果: {state.reproduction_run.reference_results}")
            if state.reproduction_run.comparisons:
                lines.append(f"指标对比: {state.reproduction_run.comparisons}")

        if state.benchmark_run:
            if state.benchmark_run.selected_spec:
                lines.append(f"选中的 Benchmark: {state.benchmark_run.selected_spec.level} {state.benchmark_run.selected_spec.title}")
                lines.append(f"Benchmark 任务族: {state.benchmark_run.selected_spec.task_family}")
            lines.append(f"Benchmark 是否适合尝试: {state.benchmark_run.eligible}")
            lines.append(f"Benchmark 是否跳过: {state.benchmark_run.skipped}")
            lines.append(f"Benchmark 是否成功: {state.benchmark_run.success}")
            if state.benchmark_run.summary:
                lines.append(f"Benchmark 摘要: {state.benchmark_run.summary}")
            if state.benchmark_run.downgrade_reasons:
                lines.append(f"Benchmark 降级原因: {state.benchmark_run.downgrade_reasons}")
            if state.benchmark_run.metrics:
                lines.append(f"Benchmark 指标: {state.benchmark_run.metrics}")
            if state.benchmark_run.comparisons:
                lines.append(f"Benchmark 指标对比: {state.benchmark_run.comparisons}")
            if state.benchmark_run.parser_hints:
                lines.append(f"Benchmark 指标解析建议: {state.benchmark_run.parser_hints}")
            if state.benchmark_run.failure_diagnosis:
                lines.append(f"Benchmark 失败诊断: {state.benchmark_run.failure_diagnosis}")

        if state.errors:
            lines.append("错误记录:")
            for err in state.errors:
                lines.append(f"  - {err}")

        return call_llm_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt="\n".join(lines),
            purpose="report_generation",
        )

    def _determine_final_status(self, state: TaskState):
        if not state.paper_metadata:
            return "paper_parse_failed"

        if not state.selected_repo:
            return "repo_not_found"

        # backend=none: only static analysis, no execution
        if state.backend == "none" and state.repo_evaluation:
            return "repo_found_smoke_not_run"

        if state.env_build and not state.env_build.build_success and not state.env_build.skipped:
            return "repo_found_but_env_failed"

        if state.benchmark_run:
            if state.benchmark_run.success:
                if state.benchmark_run.metrics or state.benchmark_run.comparisons:
                    return "benchmark_success"
                if state.reproduction_run and state.reproduction_run.success:
                    return "reproduction_success"
                if state.smoke_run and state.smoke_run.success:
                    if state.smoke_run.command and state.smoke_run.command.kind == "help":
                        return "partial_success_help_only"
                    return "success"
                return "success"
            if state.benchmark_run.skipped:
                if not state.reproduction_run:
                    return "repo_found_benchmark_not_run"
            else:
                if state.reproduction_run and state.reproduction_run.success:
                    return "reproduction_success_benchmark_failed"
                return "repo_found_but_benchmark_failed"

        if state.reproduction_run:
            if state.reproduction_run.success:
                return "reproduction_success"
            if state.reproduction_run.skipped:
                if state.smoke_run and state.smoke_run.success:
                    if state.smoke_run.command and state.smoke_run.command.kind == "help":
                        return "partial_success_help_only"
                    return "success"
                return "repo_found_reproduction_not_run"
            return "repo_found_but_reproduction_failed"

        # Check smoke_run results first (applies to both local and docker backends)
        if state.smoke_run and state.smoke_run.success:
            if state.smoke_run.command and state.smoke_run.command.kind == "help":
                return "partial_success_help_only"
            return "success"

        if state.smoke_run and not state.smoke_run.success:
            if state.smoke_run.failure_type in _SMOKE_ENV_FAILURE_TYPES:
                return "repo_found_but_env_failed"
            return "repo_found_but_smoke_failed"

        if state.env_build and state.env_build.skipped:
            return "skipped_docker"

        return "failed"

    def _short_conclusion(self, status: str) -> str:
        mapping = {
            "reproduction_success": "已完成轻量端到端复现：仓库、环境和非 help 复现命令均执行成功。",
            "reproduction_success_benchmark_failed": "已完成轻量端到端复现，但协议化 benchmark 命令执行失败；复现成功和 benchmark 失败需要分开看待。",
            "benchmark_success": "已完成协议化 benchmark 复现，并生成结构化指标、参考结果和降级说明。",
            "success": "仓库已找到，Smoke 测试命令执行成功。",
            "partial_success_help_only": "--help 命令运行正常。这是一个部分成功的 Smoke 测试，并非完整复现。",
            "repo_found_but_env_failed": "已找到代码仓库，但 conda/venv/Docker 依赖或执行环境未准备好。",
            "repo_found_but_smoke_failed": "仓库已找到，但 Smoke 测试命令执行失败。",
            "repo_found_but_reproduction_failed": "仓库和环境已准备好，但轻量完整复现命令执行失败。",
            "repo_found_but_benchmark_failed": "仓库和环境已准备好，但协议化 benchmark 命令执行失败。",
            "repo_found_reproduction_not_run": "仓库已找到，但没有执行轻量完整复现命令。",
            "repo_found_benchmark_not_run": "仓库已找到，但没有可执行的协议化 benchmark plan。",
            "repo_found_smoke_not_run": "已找到代码仓库并完成静态评估，未执行代码（backend=none）。",
            "repo_not_found": "未找到或未提供合适的代码仓库。",
            "paper_parse_failed": "论文解析失败。",
            "skipped_docker": "用户跳过了 Docker 构建和 Smoke 测试。",
            "failed": "流水线在得到结论性结果之前失败。",
        }
        return mapping.get(status, "未知状态。")

    def _next_steps(self, status: str) -> list[str]:
        if status == "benchmark_success":
            return [
                "查看 runs/benchmark_001/stdout.log、stderr.log 和 benchmark_summary.json 确认协议、指标和降级原因。",
                "如果 Achieved 低于 L3，优先补齐报告中列出的数据集、split、权重或参考表格。",
                "用同一 BenchmarkSpec 扩展到全量数据集后重跑，保持指标 parser 和 comparator 不变。",
            ]

        if status == "reproduction_success":
            return [
                "查看 runs/reproduction_001/stdout.log、stderr.log 和输出文件列表确认复现产物。",
                "将轻量复现输出与论文或 README 中的示例结果进行人工比对。",
                "如果需要更强证据，再扩展到小规模真实数据或官方权重。",
            ]

        if status == "reproduction_success_benchmark_failed":
            return [
                "查看 runs/reproduction_001/stdout.log 确认轻量复现输出，这是当前已经跑通的部分。",
                "查看 runs/benchmark_001/benchmark_candidates.json 和 stderr.log，确认 benchmark planner 是否选择了正确任务族。",
                "如果 benchmark 失败来自缺依赖或缺数据，先补齐官方小数据集、权重或可选依赖后重跑 benchmark。",
            ]

        if status == "success":
            return [
                "手动运行仓库文档中描述的评估命令。",
                "根据需要下载所需的数据集或模型权重。",
                "将输出结果与论文报告的指标进行对比。",
            ]

        if status == "partial_success_help_only":
            return [
                "查看仓库 README 了解实际的 demo 或评估命令。",
                "如果仓库提供了示例输入，尝试运行。",
                "在确认数据集和权重需求后再进行更大规模的实验。",
            ]

        if status == "repo_found_but_env_failed":
            return [
                "打开 env/conda_build.log、env/venv_build.log、env/build.log 或 runs/smoke_001/stderr.log 查看第一个依赖错误。",
                "检查仓库是否需要特定版本的 Python、CUDA 或 PyTorch。",
                "优先使用 --backend conda 在本地隔离环境中重试；如有 Docker 权限也可使用 --backend docker。",
            ]

        if status == "repo_found_but_smoke_failed":
            return [
                "打开 runs/smoke_001/stderr.log 查看错误详情。",
                "检查命令是否需要数据集路径、权重文件或配置文件。",
                "尝试手动以 --help 模式运行命令。",
            ]

        if status == "repo_found_but_reproduction_failed":
            return [
                "打开 runs/reproduction_001/stderr.log 查看轻量复现命令失败原因。",
                "检查命令是否隐含需要权重、样例输入或配置文件。",
                "如果失败来自缺少小文件，优先补充官方示例资源后重跑。",
            ]

        if status == "repo_found_but_benchmark_failed":
            return [
                "打开 runs/benchmark_001/stderr.log 查看 benchmark 命令失败原因。",
                "查看 runs/benchmark_001/benchmark_candidates.json 确认 planner 是否选择了正确的任务族和 level。",
                "优先修复官方 eval/benchmark 脚本所需的小样例、权重或 CUDA 参数。",
            ]

        if status == "repo_found_benchmark_not_run":
            return [
                "查看 runs/benchmark_001/benchmark_candidates.json 了解 L3/L2/L1 候选协议。",
                "补充官方小数据集、样例文件或 README benchmark 入口后重跑。",
                "如果仓库是新任务族，新增对应 task-family adapter，而不是新增论文名分支。",
            ]

        if status == "repo_found_reproduction_not_run":
            return [
                "查看 runs/reproduction_001/command_candidates.json 了解为什么没有安全命令。",
                "优先选择 README 中使用内置样例输入的 demo/inference 命令。",
                "避免训练命令、大数据集评估命令和需要手动下载权重的命令。",
            ]

        if status == "repo_not_found":
            return [
                "使用 --repo 或 --repo-dir 手动指定仓库。",
                "在论文 PDF、项目主页、Papers with Code 或作者主页搜索代码仓库。",
                "找到仓库后重新运行。",
            ]

        if status == "paper_parse_failed":
            return [
                "确认输入是有效且可读取的本地 PDF 文件。",
                "如果原论文来自 arXiv，请先下载 PDF 后传入本地路径。",
                "检查 PDF 是否为扫描版或纯图片。",
            ]

        if status == "skipped_docker":
            return [
                "查看仓库的可运行分数评估。",
                "使用 --backend conda 重新运行以测试环境构建。",
                "使用 --repo-dir 进行更快的本地迭代。",
            ]

        if status == "repo_found_smoke_not_run":
            return [
                "使用 --backend local 在本地运行 smoke 命令。",
                "使用 --backend conda 构建本地隔离环境并测试。",
                "查看仓库可运行分数和候选脚本评估是否合理。",
            ]

        return [
            "检查 state.json 中失败的步骤。",
            "查看报告中记录的错误信息。",
            "使用 --repo-dir 重新运行以排除 GitHub 搜索和克隆问题。",
        ]


def _is_chinese_report_insight(result: dict) -> bool:
    conclusion = result.get("conclusion")
    next_steps = result.get("next_steps")
    if not isinstance(conclusion, str) or not conclusion.strip():
        return False
    if not isinstance(next_steps, list) or not next_steps:
        return False
    if not all(isinstance(step, str) and step.strip() for step in next_steps):
        return False

    natural_text = "\n".join([conclusion, *next_steps])
    return _has_meaningful_chinese(natural_text)


def _has_meaningful_chinese(text: str) -> bool:
    import re

    cleaned = re.sub(r"`[^`]*`", " ", text)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"[/\\][\w./\\:-]+", " ", cleaned)
    cleaned = re.sub(r"\b[A-Z0-9_./:-]{2,}\b", " ", cleaned)
    cjk_count = sum(1 for ch in cleaned if "\u4e00" <= ch <= "\u9fff")
    alpha_count = sum(1 for ch in cleaned if ("a" <= ch.lower() <= "z"))
    return cjk_count >= 8 and cjk_count >= alpha_count
