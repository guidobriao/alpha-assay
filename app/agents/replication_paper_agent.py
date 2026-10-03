"""Generate a replication / extension paper (LaTeX) from pipeline artifacts.

Collects the finance brief, data access, leakage, OOS and robustness
artifacts plus the benchmark reproduction results, drafts the narrative
sections with the LLM, renders the ICLR-style LaTeX template in
templates/finance_replication/latex/, then (best-effort) compiles the PDF
with pdflatex and runs the automated reviewer from paperlab.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.core.file_utils import save_json
from app.core.progress import emit_progress
from app.core.state import TaskState
from app.tools.llm import call_llm_json

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TEMPLATE_PATH = (
    _PROJECT_ROOT / "templates" / "finance_replication" / "latex" / "template.tex"
)

_TEX_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(value: Any) -> str:
    """Escape LaTeX special characters in arbitrary text."""
    return "".join(_TEX_MAP.get(ch, ch) for ch in str(value))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return tex_escape(value)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


_NARRATIVE_PROMPT = """\
You are a scientific writing assistant drafting the narrative sections of an \
automated replication report written in LaTeX.

Return a JSON object with exactly these keys:
- "abstract": string (60-150 words)
- "introduction": string (80-180 words)
- "discussion": string (80-180 words)
- "conclusion": string (50-120 words)

Rules:
- Plain text only: no markdown, no LaTeX commands, no bullet lists. Inline \
mathematical notation in words is fine.
- Be strictly factual: use only the provided execution results. Never invent \
numbers. Where results are missing, say so explicitly.
- The report documents an automated replication of a published paper, \
including a leakage audit, an out-of-sample extension and a robustness \
matrix. Interpret sign consistency as directional confirmation and be \
explicit about magnitude differences."""


class ReplicationPaperAgent:
    """Render the replication paper from pipeline artifacts."""

    def run(self, state: TaskState) -> TaskState:
        task_dir = Path(state.task_dir)
        out_dir = task_dir / "paper_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {"agent": "ReplicationPaperAgent"}

        if not _TEMPLATE_PATH.exists():
            meta["skipped"] = f"template not found: {_TEMPLATE_PATH}"
            save_json(out_dir / "paper_meta.json", meta)
            emit_progress("Replication paper", "skipped", level="warning", detail=meta["skipped"])
            return state

        emit_progress("Replication paper", "collecting pipeline artifacts")
        artifacts = self._collect(state)
        if not artifacts["has_content"]:
            meta["skipped"] = (
                "no benchmark, reproduction or finance artifacts to write about"
            )
            save_json(out_dir / "paper_meta.json", meta)
            emit_progress("Replication paper", "skipped", level="warning", detail=meta["skipped"])
            return state

        emit_progress("Replication paper", "drafting narrative sections")
        narrative = self._llm_narrative(state, artifacts)

        emit_progress("Replication paper", "rendering LaTeX")
        template_text = _TEMPLATE_PATH.read_text(encoding="utf-8")
        values = self._build_values(state, artifacts, narrative)
        tex_text = template_text
        for key, value in values.items():
            tex_text = tex_text.replace(f"%%{key}%%", value)

        tex_path = out_dir / "replication_paper.tex"
        tex_path.write_text(tex_text, encoding="utf-8")
        meta["tex_path"] = str(tex_path)
        emit_progress("Replication paper", "wrote LaTeX source", detail=str(tex_path))

        meta["compile"] = self._try_compile(out_dir)
        if meta["compile"].get("compiled"):
            emit_progress("Replication paper", "PDF compiled", detail=str(out_dir / "replication_paper.pdf"))

        emit_progress("Replication paper", "running automated review (best-effort)")
        meta["review"] = self._try_review(tex_text)

        save_json(out_dir / "paper_meta.json", meta)
        emit_progress("Replication paper", "done", level="success", detail=str(out_dir))
        return state

    # -- artifact collection ------------------------------------------------

    def _collect(self, state: TaskState) -> dict[str, Any]:
        task_dir = Path(state.task_dir)
        benchmark = state.benchmark_run.model_dump() if state.benchmark_run else None
        reproduction = state.reproduction_run.model_dump() if state.reproduction_run else None
        finance_brief = _load_json(task_dir / "paper" / "finance_brief.json")
        data_access = state.data_access.model_dump() if state.data_access else None
        leakage = _load_json(task_dir / "leakage" / "leakage_report.json")
        oos = _load_json(task_dir / "oos" / "oos_report.json") or (
            state.oos if isinstance(state.oos, dict) else None
        )
        robustness = _load_json(task_dir / "robustness" / "robustness_matrix.json") or (
            state.robustness if isinstance(state.robustness, dict) else None
        )
        has_content = bool(benchmark or reproduction or finance_brief)
        return {
            "benchmark": benchmark,
            "reproduction": reproduction,
            "finance_brief": finance_brief,
            "data_access": data_access,
            "leakage": leakage,
            "oos": oos,
            "robustness": robustness,
            "has_content": has_content,
        }

    # -- narrative ------------------------------------------------------------

    def _llm_narrative(self, state: TaskState, artifacts: dict[str, Any]) -> dict[str, str] | None:
        payload = {
            "paper_title": state.paper_metadata.title if state.paper_metadata else None,
            "benchmark": self._slim_benchmark(artifacts["benchmark"]),
            "reproduction": self._slim_reproduction(artifacts["reproduction"]),
            "finance_brief": artifacts["finance_brief"],
            "leakage_verdict": (artifacts["leakage"] or {}).get("verdict"),
            "leakage_counts": (artifacts["leakage"] or {}).get("counts"),
            "oos": self._slim_oos(artifacts["oos"]),
            "robustness": self._slim_robustness(artifacts["robustness"]),
            "data_access_summary": (artifacts["data_access"] or {}).get("summary"),
        }
        result = call_llm_json(
            system_prompt=_NARRATIVE_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
            purpose="replication_paper_narrative",
            max_tokens=2048,
        )
        if not isinstance(result, dict):
            return None
        wanted = ("abstract", "introduction", "discussion", "conclusion")
        if not all(isinstance(result.get(k), str) and result.get(k, "").strip() for k in wanted):
            return None
        return {k: result[k].strip() for k in wanted}

    def _slim_benchmark(self, benchmark: dict | None) -> dict | None:
        if not benchmark:
            return None
        return {
            "success": benchmark.get("success"),
            "skipped": benchmark.get("skipped"),
            "achieved_level": benchmark.get("achieved_level"),
            "task_family": (benchmark.get("selected_spec") or {}).get("task_family"),
            "title": (benchmark.get("selected_spec") or {}).get("title"),
            "metrics": benchmark.get("metrics"),
            "comparisons": benchmark.get("comparisons"),
            "downgrade_reasons": benchmark.get("downgrade_reasons"),
        }

    def _slim_reproduction(self, reproduction: dict | None) -> dict | None:
        if not reproduction:
            return None
        return {
            "success": reproduction.get("success"),
            "skipped": reproduction.get("skipped"),
            "metrics": reproduction.get("metrics"),
            "comparisons": reproduction.get("comparisons"),
        }

    def _slim_oos(self, oos: dict | None) -> dict | None:
        if not oos:
            return None
        return {
            "ran": oos.get("ran"),
            "verdict": oos.get("verdict"),
            "split": oos.get("split"),
            "comparisons": oos.get("comparisons"),
        }

    def _slim_robustness(self, robustness: dict | None) -> dict | None:
        if not robustness:
            return None
        return {
            "ran": robustness.get("ran"),
            "verdict": robustness.get("verdict"),
            "sign_consistency": robustness.get("sign_consistency"),
            "variation_labels": [
                v.get("label") for v in (robustness.get("variations") or [])
            ],
        }

    # -- LaTeX section builders ------------------------------------------------

    def _build_values(self, state: TaskState, artifacts: dict[str, Any], narrative: dict[str, str] | None) -> dict[str, str]:
        paper_title = state.paper_metadata.title if state.paper_metadata else None
        title = (
            f"Replication and Out-of-Sample Extension: {paper_title}"
            if paper_title
            else f"Automated Replication Report: {state.task_id}"
        )
        fallback = self._fallback_narrative(state, artifacts)
        return {
            "TITLE": tex_escape(title),
            "ABSTRACT": narrative["abstract"] if narrative else fallback["abstract"],
            "INTRO": narrative["introduction"] if narrative else fallback["introduction"],
            "HYPOTHESIS": self._hypothesis(artifacts),
            "METHODOLOGY": self._methodology(artifacts),
            "REPLICATION_NARRATIVE": self._replication_narrative(artifacts),
            "REPLICATION_TABLE": self._replication_table(artifacts),
            "LEAKAGE_SECTION": self._leakage_section(artifacts["leakage"]),
            "OOS_SECTION": self._oos_section(artifacts["oos"]),
            "ROBUSTNESS_SECTION": self._robustness_section(artifacts["robustness"]),
            "DISCUSSION": narrative["discussion"] if narrative else fallback["discussion"],
            "CONCLUSION": narrative["conclusion"] if narrative else fallback["conclusion"],
        }

    def _hypothesis(self, artifacts: dict[str, Any]) -> str:
        brief = artifacts["finance_brief"]
        if brief and brief.get("hypothesis"):
            return tex_escape(brief["hypothesis"])
        return "Not available: this run was not identified as an empirical asset-pricing study."

    def _methodology(self, artifacts: dict[str, Any]) -> str:
        brief = artifacts["finance_brief"]
        if not brief:
            return (
                "This run was not identified as an empirical asset-pricing study. "
                "See the benchmark reproduction section for the evaluated protocol."
            )
        uni = brief.get("universe") or {}
        period = brief.get("sample_period") or {}
        pf = brief.get("portfolio") or {}
        costs = brief.get("transaction_costs") or {}
        items = [
            f"Universe: {_fmt(uni.get('description'))}"
            + (f", exclusions: {tex_escape(', '.join(uni.get('exclusions') or []))}" if uni.get("exclusions") else "")
            + f"; survivorship: \\texttt{{{tex_escape(uni.get('survivorship') or 'unknown')}}}",
            f"Sample period: {_fmt(period.get('start'))} to {_fmt(period.get('end'))}"
            f" (frequency: \\texttt{{{tex_escape(period.get('frequency') or 'unknown')}}})",
            f"Portfolio: {_fmt(pf.get('signal'))}; sort: {_fmt(pf.get('sort_method'))}; "
            f"{_fmt(pf.get('n_portfolios'))} portfolios; weighting: "
            f"\\texttt{{{tex_escape(pf.get('weighting') or 'unknown')}}}; "
            f"rebalancing: {_fmt(pf.get('rebalancing'))}; formation lag: {_fmt(pf.get('formation_lag'))}",
            "Factor controls: " + (tex_escape(", ".join(brief.get("factor_controls") or [])) or "none"),
            "Transaction costs: "
            + (
                f"included ({_fmt(costs.get('bps_per_side'))} bps per side)"
                if costs.get("included")
                else "not included"
            ),
            "Winsorization: " + _fmt(brief.get("winsorization")),
            "Data sources: " + (tex_escape(", ".join(brief.get("data_sources") or [])) or "N/A"),
        ]
        body = "\n".join(f"\\item {line}" for line in items)
        return "\\begin{itemize}\n" + body + "\n\\end{itemize}"

    def _replication_narrative(self, artifacts: dict[str, Any]) -> str:
        benchmark = artifacts["benchmark"]
        if not benchmark:
            return "No protocolized benchmark was run in this pipeline execution."
        level = benchmark.get("achieved_level") or "N/A"
        if benchmark.get("success"):
            return (
                f"The automated benchmark reproduction completed at level "
                f"\\texttt{{{tex_escape(level)}}}. The table below compares the "
                "reproduced metrics against the reference values extracted from "
                "the paper. Status \\texttt{sign\\_matched} denotes directional "
                "confirmation with magnitude outside tolerance."
            )
        if benchmark.get("skipped"):
            return (
                "The protocolized benchmark was skipped: "
                + tex_escape(benchmark.get("skip_reason") or "reason unspecified")
                + "."
            )
        return (
            f"The protocolized benchmark failed (failure type: "
            f"\\texttt{{{tex_escape(benchmark.get('failure_type') or 'unknown')}}}). "
            "See the pipeline report for the failure diagnosis."
        )

    def _replication_table(self, artifacts: dict[str, Any]) -> str:
        sources = []
        benchmark = artifacts["benchmark"]
        if benchmark:
            sources.extend(benchmark.get("comparisons") or [])
        reproduction = artifacts["reproduction"]
        if reproduction:
            sources.extend(reproduction.get("comparisons") or [])
        if not sources:
            return "No metric comparisons were recorded for this run."
        rows = [
            "\\toprule",
            "Metric & Reproduced & Reference & $\\Delta$ & Status \\\\",
            "\\midrule",
        ]
        for c in sources:
            delta = c.get("delta")
            delta_s = _fmt(delta) if isinstance(delta, (int, float)) else "N/A"
            rows.append(
                f"{tex_escape(c.get('metric'))} & {_fmt(c.get('actual'))} & "
                f"{_fmt(c.get('expected'))} & {delta_s} & "
                f"\\texttt{{{tex_escape(c.get('status'))}}} \\\\"
            )
        rows.append("\\bottomrule")
        return (
            "\\begin{table}[h]\n\\centering\n"
            "\\begin{tabular}{lrrrl}\n" + "\n".join(rows) + "\n\\end{tabular}\n"
            "\\caption{Reproduced vs.\\ reference metrics.}\n\\end{table}"
        )

    def _leakage_section(self, leakage: dict | None) -> str:
        if not leakage:
            return "No leakage audit was run for this study."
        counts = leakage.get("counts") or {}
        lines = [
            f"Audit verdict: \\textbf{{{tex_escape(leakage.get('verdict'))}}} "
            f"(risk score {_fmt(leakage.get('risk_score'), 0)}; "
            f"critical {_fmt(counts.get('critical'), 0)}, "
            f"warnings {_fmt(counts.get('warning'), 0)}, "
            f"info {_fmt(counts.get('info'), 0)}).",
        ]
        findings = leakage.get("findings") or []
        if findings:
            items = []
            for f in findings[:20]:
                location = f.get("file") or "finance brief"
                if f.get("line"):
                    location += f":{f['line']}"
                items.append(
                    f"\\item [{tex_escape(f.get('severity'))}] "
                    f"\\texttt{{{tex_escape(f.get('rule_id'))}}} at "
                    f"{tex_escape(location)}: {tex_escape(f.get('message'))}"
                )
            lines.append("\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}")
        else:
            lines.append("No leakage findings were recorded.")
        return "\n\n".join(lines)

    def _oos_section(self, oos: dict | None) -> str:
        if not oos:
            return "The out-of-sample extension was not run."
        if not oos.get("ran"):
            return "The out-of-sample extension was skipped: " + tex_escape(
                oos.get("skip_reason") or "reason unspecified"
            ) + "."
        split = oos.get("split") or {}
        ins = split.get("in_sample") or {}
        oosw = split.get("out_of_sample") or {}
        lines = [
            f"Verdict: \\texttt{{{tex_escape(oos.get('verdict'))}}}.",
            f"In-sample window: {_fmt(ins.get('start'))} to {_fmt(ins.get('end'))}; "
            f"out-of-sample window: {_fmt(oosw.get('start'))} to {_fmt(oosw.get('end'))}.",
        ]
        if split.get("note"):
            lines.append(tex_escape(split["note"]))
        comparisons = oos.get("comparisons") or []
        if comparisons:
            rows = [
                "\\toprule",
                "Metric & In-sample & Out-of-sample & $\\Delta$ & Status \\\\",
                "\\midrule",
            ]
            for c in comparisons:
                rows.append(
                    f"{tex_escape(c.get('metric'))} & {_fmt(c.get('in_sample'))} & "
                    f"{_fmt(c.get('out_of_sample'))} & {_fmt(c.get('delta'))} & "
                    f"\\texttt{{{tex_escape(c.get('status'))}}} \\\\"
                )
            rows.append("\\bottomrule")
            lines.append(
                "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lrrrl}\n"
                + "\n".join(rows)
                + "\n\\end{tabular}\n\\caption{In-sample vs.\\ out-of-sample statistics.}\n\\end{table}"
            )
        if oos.get("caveat"):
            lines.append("\\emph{" + tex_escape(oos["caveat"]) + "}")
        return "\n\n".join(lines)

    def _robustness_section(self, robustness: dict | None) -> str:
        if not robustness:
            return "The robustness matrix was not run."
        if not robustness.get("ran"):
            return "The robustness matrix was skipped: " + tex_escape(
                robustness.get("skip_reason") or "reason unspecified"
            ) + "."
        lines = [
            f"Verdict: \\texttt{{{tex_escape(robustness.get('verdict'))}}} "
            f"(sign consistency: {tex_escape(robustness.get('sign_consistency'))})."
        ]
        variations = robustness.get("variations") or []
        if variations:
            rows = [
                "\\toprule",
                "Variation & Mean (\\%/month) & NW $t$-stat & Sharpe & Sign match \\\\",
                "\\midrule",
            ]
            for v in variations:
                m = v.get("metrics") or {}
                if isinstance(m, dict) and m.get("error"):
                    rows.append(
                        f"{tex_escape(v.get('label'))} & \\multicolumn{{3}}{{c}}{{run failed}} & --- \\\\"
                    )
                    continue
                sign = v.get("sign_match_vs_baseline")
                sign_s = "yes" if sign is True else ("no" if sign is False else "n/a")
                rows.append(
                    f"{tex_escape(v.get('label'))} & {_fmt(m.get('mean_pct_monthly'))} & "
                    f"{_fmt(m.get('nw_t_stat'), 3)} & {_fmt(m.get('sharpe_ratio'), 3)} & {sign_s} \\\\"
                )
            rows.append("\\bottomrule")
            lines.append(
                "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lrrrc}\n"
                + "\n".join(rows)
                + "\n\\end{tabular}\n\\caption{Robustness matrix over the open-access factor library.}\n\\end{table}"
            )
        if robustness.get("caveat"):
            lines.append("\\emph{" + tex_escape(robustness["caveat"]) + "}")
        return "\n\n".join(lines)

    def _fallback_narrative(self, state: TaskState, artifacts: dict[str, Any]) -> dict[str, str]:
        verdict = (artifacts["leakage"] or {}).get("verdict", "not run")
        oos_verdict = ((artifacts["oos"] or {}).get("verdict") if artifacts["oos"] else None) or "not run"
        rob = artifacts["robustness"] or {}
        return {
            "abstract": (
                "This report documents an automated replication of a published "
                "empirical study, executed end to end by a multi-agent pipeline. "
                "The pipeline extracted the study's empirical design, provisioned "
                "the obtainable data, reproduced the headline benchmark, audited "
                "the replicated code for look-ahead bias, extended the results "
                "out of sample, and ran a robustness matrix. The generated tables "
                "report the original versus replicated metrics, the audit "
                "findings, and the out-of-sample and robustness outcomes."
            ),
            "introduction": (
                "Replication is the strongest test an empirical claim can face, "
                "yet most published results are never independently reproduced. "
                "This report was produced by an automated pipeline that takes a "
                "paper and its associated repository and executes a full "
                "replication workflow: design extraction, data provisioning, "
                "environment construction, benchmark reproduction, a static "
                "leakage audit of the replicated code, an out-of-sample "
                "extension beyond the paper's sample, and a robustness matrix "
                "over sub-periods and alternative factor definitions."
            ),
            "discussion": (
                f"The leakage audit returned verdict '{verdict}'; findings, when "
                "present, are reported with file and line evidence and should be "
                "read as first-pass signals requiring human confirmation rather "
                "than proof of misconduct. The out-of-sample extension returned "
                f"verdict '{oos_verdict}': directional consistency across the "
                "sample split supports the claim's persistence, while magnitude "
                "differences are expected as effects decay after publication. "
                f"The robustness matrix returned '{rob.get('verdict', 'not run')}' "
                f"with sign consistency {rob.get('sign_consistency', 'n/a')}."
            ),
            "conclusion": (
                "The automated pipeline executed the full replication workflow "
                "and produced structured evidence for each stage. All artifacts "
                "(data access records, leakage findings, out-of-sample "
                "comparisons and the robustness matrix) are serialized alongside "
                "this document and can be re-examined independently. Residual "
                "limitations are documented in the respective sections and in "
                "the pipeline report."
            ),
        }

    # -- compile & review -------------------------------------------------------

    def _try_compile(self, out_dir: Path) -> dict[str, Any]:
        if not shutil.which("pdflatex"):
            return {"compiled": False, "reason": "pdflatex not found on PATH"}
        try:
            for _ in range(2):
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "replication_paper.tex"],
                    cwd=out_dir,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
        except Exception as exc:
            return {"compiled": False, "reason": f"pdflatex failed: {exc}"}
        pdf = out_dir / "replication_paper.pdf"
        return {"compiled": pdf.exists(), "pdf_path": str(pdf) if pdf.exists() else None}

    def _try_review(self, tex_text: str) -> dict[str, Any]:
        try:
            import openai  # noqa: F401
            from paperlab.perform_review import perform_review

            client = openai.OpenAI()
            model = os.environ.get("PAPER_REVIEW_MODEL", "gpt-4o-2024-05-13")
            review = perform_review(tex_text, model, client, num_reflections=1)
            if isinstance(review, dict):
                return {
                    "overall": review.get("Overall"),
                    "decision": review.get("Decision"),
                    "weaknesses": [str(w) for w in (review.get("Weaknesses") or [])[:5]],
                }
            return {"skipped": "review returned no structured output"}
        except Exception as exc:
            return {"skipped": f"review unavailable: {exc}"}
