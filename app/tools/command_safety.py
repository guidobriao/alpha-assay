from __future__ import annotations

from pathlib import PurePosixPath


ALLOWED_EXECUTABLES = {"python", "pytest"}


def is_safe_relative_script(path: str) -> bool:
    p = PurePosixPath(path)

    if p.is_absolute():
        return False
    if ".." in p.parts:
        return False
    if not path.endswith(".py"):
        return False
    if any(char in path for char in [";", "&", "|", "$", "`", ">", "<"]):
        return False

    return True


def is_safe_argv(argv: list[str], candidate_scripts: list[str]) -> tuple[bool, str | None]:
    if not argv:
        return False, "empty command"

    if any(any(char in part for char in [";", "&", "|", "$", "`", ">", "<"]) for part in argv):
        return False, "shell metacharacter blocked"

    if argv[0] == "python":
        if len(argv) >= 2 and argv[1] == "-m":
            if argv[:3] == ["python", "-m", "pytest"]:
                return True, None
            return False, "only python -m pytest is allowed"

        # python --version (no script, just version check)
        if len(argv) == 2 and argv[1] == "--version":
            return True, None

        if len(argv) < 2:
            return False, "missing script"

        script = argv[1]
        # Always verify the script path is safe, even if in candidate list
        if not is_safe_relative_script(script):
            return False, "unsafe script path"
        if script in candidate_scripts:
            pass  # Explicitly listed, allow
        elif script.startswith((
            "demo", "eval", "test", "main", "run", "train",
            "match", "infer", "predict", "bench",
        )):
            pass  # Recognized prefix, allow
        else:
            return False, "script is not in candidate_scripts"

        extra_args = argv[2:]
        allowed_extra_args = {"--help", "-h"}
        if extra_args and any(arg not in allowed_extra_args for arg in extra_args):
            return False, "only --help or -h arguments allowed for v0.1"

        return True, None

    if argv[0] == "pytest":
        allowed = {"pytest", "-q"}
        if all(part in allowed for part in argv):
            return True, None
        return False, "only pytest -q is allowed"

    return False, "executable not allowed"
