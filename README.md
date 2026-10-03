# alpha-assay

AI-agent pipeline for **replication, verification and out-of-sample extension of published research**, specialized for quantitative finance.

Take a paper → reproduce it → verify the results → check for leakage → test robustness → validate out-of-sample → republish as a replication/extension paper.

## Status

**Phase 3 complete** — the full replication pipeline is wired end to end:

paper ingest → understanding (finance design extraction) → repo search →
environment build → smoke test → benchmark reproduction →
**data access → leakage audit → out-of-sample extension → robustness matrix →**
report (with Griffin-style sections) → **replication paper (LaTeX, best-effort
pdflatex + automated review)**

## Layout

| Path | Role |
|---|---|
| `orchestrator/`, `AGENTS.md` | Orchestrator skill + specialized sub-agents (paper → code → env → run → compare) |
| `app/`, `pyproject.toml` | Pipeline agents, benchmark framework, tools, TUI, reporting |
| `app/benchmark/finance_schema.py`, `finance_metrics.py` | Finance specification schema (universe, period, portfolio, costs) and stdlib-only statistics (Newey-West t-stat, Sharpe, drawdown, French CSV loader) |
| `app/benchmark/adapters/finance.py` | Finance benchmark adapter (L1 French-library runner, L2 repo scripts) |
| `app/agents/data_access_agent.py` | Dataset provisioning: open-access downloads (Kenneth French library), subscription sources flagged for manual extracts (CRSP/Compustat/IBES/WRDS) |
| `app/agents/leakage_detector.py` | Static look-ahead-bias audit of the replicated code + methodological audit of the extracted design |
| `app/agents/oos_agent.py` | Temporal split at the paper's sample end; in-sample vs out-of-sample comparison |
| `app/agents/robustness_agent.py` | Sub-period, factor-column and factor-file variation matrix with sign-consistency verdicts |
| `app/agents/replication_paper_agent.py` | LaTeX replication paper from pipeline artifacts |
| `templates/finance_replication/latex/` | ICLR-style LaTeX template for the replication paper |
| `contractgen/`, `run_contractgen.py` | Contract-guided paper-to-code pipeline (requirements + evidence channels) |
| `contractgen/pipeline/finance_contract.py` | Finance implementation contract: verifiable obligations (point-in-time universe, formation lag, weighting, factor controls, costs) + prompt-ready rendering |
| `paperlab/`, `launch_paperlab.py` | LLM abstraction, automated paper review |

## Roadmap

- Deep wiring of the finance contract into contractgen's prepare/IR stages (obligations currently flow via extracted units)
- `examples/gold_set_finance.json` — classic replication gold set with expected values
- Real-world end-to-end validation on published finance replications
- Decision on the internal PaperBench naming within `contractgen/`

## Licensing

Per-component licenses:

- `LICENSE` (MIT) — `app/`, `contractgen/`, project code
- `orchestrator/LICENSE.polyform-noncommercial` — the orchestrator skill and its agent guides (non-commercial use only; note that Part I of `AGENTS.md` is covered by this license)
- `LICENSE.ai_scientist` — `paperlab/` and `templates/`
