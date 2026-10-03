from pathlib import Path
from datetime import datetime


class TaskPaths:
    def __init__(self, workspace_dir: str, task_id: str):
        self.workspace_dir = Path(workspace_dir)
        self.task_id = task_id
        self.task_dir = self.workspace_dir / "tasks" / task_id

        self.input_dir = self.task_dir / "input"
        self.paper_dir = self.task_dir / "paper"
        self.repos_dir = self.task_dir / "repos"
        self.evaluation_dir = self.task_dir / "evaluation"
        self.env_dir = self.task_dir / "env"
        self.runs_dir = self.task_dir / "runs"
        self.report_dir = self.task_dir / "report"

    @property
    def state_json_path(self) -> Path:
        return self.task_dir / "state.json"

    def create_all_dirs(self) -> None:
        self.task_dir.mkdir(parents=True, exist_ok=False)
        for path in [
            self.input_dir,
            self.paper_dir,
            self.repos_dir,
            self.evaluation_dir,
            self.env_dir,
            self.runs_dir,
            self.report_dir,
        ]:
            path.mkdir()


def generate_task_id() -> str:
    return datetime.now().strftime("task_%Y%m%d_%H%M%S_%f")


# ── Project root / workspace resolution ─────────────────────

def find_project_root(start: Path | None = None) -> Path:
    """Search upward from this file until pyproject.toml and app/ are found."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "app").exists():
            return parent
    # Fallback: app/core/paths.py → app/core → app → project root
    return Path(__file__).resolve().parents[2]


def default_project_workspace() -> Path:
    return find_project_root() / "workspace"


def project_pdf_dir() -> Path:
    return find_project_root() / "pdf"


def resolve_workspace_path(value: str | Path | None, *, explicit: bool = False) -> Path:
    """Resolve a workspace path.

    - None/empty → project_root/workspace
    - './workspace' (default, not explicit) → project_root/workspace
    - absolute → as-is
    - explicit relative → cwd-relative
    """
    if value is None or str(value).strip() == "":
        return default_project_workspace().resolve()

    raw = str(value).strip()
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()

    normalized = raw.replace("\\", "/").strip()
    if not explicit and normalized in {"workspace", "./workspace", ".\\workspace"}:
        return default_project_workspace().resolve()

    if explicit:
        return p.resolve()

    return (find_project_root() / p).resolve()
