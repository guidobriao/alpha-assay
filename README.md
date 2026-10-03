# alpha-assay

AI-agent pipeline for **replication, verification and out-of-sample extension of published research**, specialized for quantitative finance.

Take a paper → reproduce it → verify the results → check for leakage → test robustness → validate out-of-sample → republish as a replication/extension paper.

## Status

🚧 Phase 1: components assembled, structure only — not yet integrated.

## Layout

| Path | Role |
|---|---|
| `orchestrator/`, `AGENTS.md` | Orchestrator skill + specialized sub-agents (paper → code → env → run → compare) |
| `app/`, `pyproject.toml` | Pipeline agents, benchmark framework, tools, TUI, reporting |
| `contractgen/`, `run_contractgen.py` | Contract-guided paper-to-code pipeline (requirements + evidence channels) |
| `paperlab/`, `launch_paperlab.py`, `templates/` | LLM abstraction, automated review, LaTeX paper generation |

## Planned (Phase 3)

Finance-specific components: finance ontology, data access agent, leakage detector, out-of-sample framework, robustness matrix, finance metrics library, replication paper generator.

## Licensing

Third-party components are included under their original licenses — see `LICENSE.*` files.
