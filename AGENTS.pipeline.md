# AGENTS.md — alpha-assay

## Project overview

Automated pipeline: PDF → find repo → build env → smoke test → benchmark → report.

Languages: Python 3.8+, TypeScript (docs site only). Package manager: **Poetry**.

## Development workflow (MANDATORY)

After EVERY code change, you MUST:

1. **Rebuild if needed** — if dependencies changed, reinstall:
   ```bash
   F:\Anaconda\envs\paper_smoke\python.exe -m pip install -e . --no-deps
   ```
2. **Commit** with a concise message describing the fix/feature.
3. **Push** to `git@github.com:guidobriao/alpha-assay.git` (SSH).
4. **Report** after every push: state the commit hash range, branch, and brief summary of what was pushed.

No exceptions. Never leave changes uncommitted or unpushed.

## Essential commands

```bash
# Install
poetry install

# Run pipeline on a local PDF (use @ prefix)
alpha-assay run --input @/path/to/paper.pdf --backend conda

# Run with manual repo override
alpha-assay run --input @paper.pdf --repo https://github.com/user/repo

# Evaluate a gold set of papers
alpha-assay eval-goldset --gold-set examples/gold_set.json --backend conda --max-items 5

# Run a single test
poetry run pytest tests/test_paper_ingest_agent.py -v

# Run all tests
poetry run pytest -v

# Lint and format (dev dependencies)
poetry run ruff check .
poetry run black --check .
```

## Environment setup

Copy `.env.example` to `.env` and fill in:

| Variable | Required? | Notes |
|---|---|---|
| `GITHUB_TOKEN` | Recommended | Avoids API rate limiting |
| `OPENAI_API_KEY` | Optional | LLM gracefully degrades if missing; used by `app/tools/llm.py` |
| `OPENAI_BASE_URL` | Optional | For custom OpenAI-compatible endpoints |
| `OPENAI_MODEL` | Optional | Default: `gpt-4o-mini` |
| `DEFAULT_BACKEND` | Optional | Default: `conda` |
| `DEFAULT_WORKSPACE` | Optional | Default: `./workspace` |

## Architecture

```
app/
├── cli.py              # Typer CLI: `alpha-assay run` and `alpha-assay eval-goldset`
├── agents/             # Pipeline stages, each agent is a class with .run(state) → state
├── benchmark/          # Task-family-aware benchmark subsystem
│   ├── adapters/       # Per-task-family runners (ASR, feat matching, ZS classif., seq label)
│   ├── planner.py      # Benchmark protocol planner
│   └── schema.py       # BenchmarkRunResult, BenchmarkSpec
├── core/
│   ├── config.py       # pydantic-settings from .env
│   ├── state.py        # TaskState (pydantic model) — the pipeline data carrier
│   ├── paths.py        # TaskPaths — file system layout for each run
│   └── naming.py       # stable_paper_slug() — deterministic naming for env reuse
├── runtime/            # Event-driven session engine for TUI
├── tools/              # Stateless utilities: GitHub search, PDF parse, LLM, arxiv, deps
├── templates/          # Jinja2 templates: Dockerfile, smoke report
└── tui/                # Textual-based interactive frontend
```

**Pipeline order** (linear): paper ingest → paper understanding → GitHub search (skipped if `--repo` provided) → repo evaluation → env build (conda/venv/docker/local/none) → smoke run → benchmark reproduction → simple reproduction → report write.

## Key architectural conventions

- **State object is the backbone**: `TaskState` (pydantic model in `app/core/state.py`) is passed through every agent. Each agent's `.run(state)` returns a **shallow copy** with new fields populated. Never mutate in place.
- **Config is one global**: `app.core.config.settings` — a pydantic-settings singleton loaded from `.env`. Access via `from app.core.config import settings`.
- **LLM is optional**: `app/tools/llm.py` wraps OpenAI. `call_llm()` returns `None` when no API key is set. All callers must handle `None`. The `openai` package import itself is try/except'd — works without it.
- **PDF inputs use `@` prefix** for local files: `@/path/to/paper.pdf`. Arxiv URLs are rejected directly — the pipeline needs a local PDF.
- **State persistence**: `save_state(state)` / `load_state(path)` in `app/core/file_utils.py` serializes TaskState to JSON at `state.json` in each task directory.
- **Task directories**: generated under `{workspace}/tasks/task_YYYYMMDD_HHMMSS_ffffff/` with subdirs for input, paper, repos, evaluation, env, runs, report.

## Testing conventions

- Uses **pytest** with `pytest.ini` at root. Test path: `tests/`.
- **Heavy monkeypatch usage**: nearly every test monkeypatches internal functions to avoid real network/filesystem/conda calls. Pattern:
  ```python
  monkeypatch.setattr(module_under_test, "function_name", lambda *a, **kw: fake_result)
  ```
- Fixtures live in `tests/fixtures/` (currently only `simple_demo_repo/`).
- Tests use **`tmp_path`** (pytest built-in) for all file system operations — never write to real paths.
- Test files mirror agent names: `test_paper_ingest_agent.py`, `test_conda_build_agent.py`, etc.
- Integration tests (TUI) are at `test_tui_integration.py` and `test_tui.py`.
- There is **no CI config**, **no mypy config**, **no pre-commit hooks**. Ruff and black are dev dependencies but not enforced automatically.

## The benchmark subsystem

Four task families with protocol adapters in `app/benchmark/adapters/`:
- `local_feature_matching.py` — SuperGlue, LightGlue, XFeat
- `zero_shot_classification.py` — CLIP
- `asr.py` — Whisper
- `sequence_labeling.py` — Flair

Each adapter has three levels (L1–L3). The planner auto-downgrades when higher levels aren't feasible. Results are stored as `BenchmarkRunResult` objects.

## Output and reporting

- All output goes to `workspace/` (gitignored) or custom `--workspace` dir.
- Report written to `{task_dir}/report/reproduction_smoke_report.md`.
- `eval-goldset` results go to `goldset_results/` (also gitignored).
- The `final_status` field in reports is the single most important output — see README for full status value reference.

## Gotchas

1. **No `.env` = no GitHub search**: `GITHUB_TOKEN` is technically optional via pydantic, but GitHub search will fail without it. Copy `.env.example` first.
2. **Backend validation**: backend must be one of `none | local | venv | conda | docker`. Default is `conda`.
3. **Workspace collisions**: task IDs are timestamp-based with a retry loop (100 attempts). Rarely conflicts.
4. **Import order**: `from app.core.config import settings` must come after `.env` is loaded (handled by pydantic-settings on first import).
5. **TUI is separate**: the Textual TUI (`app/tui/app.py`) has its own event loop and session management. The `alpha-assay` CLI and TUI share the same agents but different orchestration.

## 通用规则

1. **每次改动代码后必须重新构建、本地提交、推送 GitHub，然后在最后汇报。**
   - 依赖变更时：`F:\Anaconda\envs\paper_smoke\python.exe -m pip install -e . --no-deps` 或 `poetry install`
   - 提交：`git add -A && git commit -m "简明描述"`
   - 推送：`git push origin HEAD`
   - 汇报格式：commit hash range、branch、变更摘要
   - 不允许留下未提交或未推送的变更

2. **回答默认用中文。**

3. **网络代理**：如果遇到网络问题（如 `git push`、`pip install`、`poetry install` 失败），优先尝试本地 `http://127.0.0.1:7890` 代理：
   ```bash
   export https_proxy=http://127.0.0.1:7890
   export http_proxy=http://127.0.0.1:7890
   ```
   如果设置了代理仍无法连接，立即向用户汇报，不要自行跳过推送或提交步骤。
