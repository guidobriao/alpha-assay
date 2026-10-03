from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import httpx
import yaml

from app.core.progress import emit_progress
from app.core.cancellation import is_cancelled, raise_if_cancelled, PipelineCancelled

logger = logging.getLogger(__name__)


_SKIPPED_ENV_DEPS = {
    "python",
    "pip",
    "cudatoolkit",
    "cuda",
    "cudnn",
    "pytorch-cuda",
    "cpuonly",
    "_libgcc_mutex",
    "_openmp_mutex",
    "mkl",
    "mkl-service",
    "setuptools",
}

_ENV_PACKAGE_RENAMES = {
    "pytorch": "torch",
}


def clone_repo(url: str, dest_dir: Path, max_retries: int = 3) -> Path:
    """Clone a git repo with retry, progress, cancel, and zip fallback."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    proxies = detect_git_proxy()
    if proxies:
        proxy_lines = [f"{k}={mask_proxy_url(v)}" for k, v in sorted(proxies.items())]
        emit_progress("Evaluate repo", "检测到 Git 代理配置",
                      detail="\n".join(f"  - {x}" for x in proxy_lines),
                      proxy_status="检测到代理", log_lines=proxy_lines)
    else:
        emit_progress("Evaluate repo", "未检测到代理", proxy_status="无代理")

    clone_env = {**os.environ}
    for k, v in proxies.items():
        clone_env[k] = v

    last_error = None
    for attempt in range(1, max_retries + 1):
        raise_if_cancelled()
        emit_progress("Evaluate repo",
                      f"正在克隆仓库，第 {attempt}/{max_retries} 次尝试",
                      detail=f"仓库地址：{url}", repo_url=url)
        try:
            _clone_subprocess(url, dest_dir, env=clone_env)
            return dest_dir
        except PipelineCancelled:
            _cleanup_clone(dest_dir)
            raise
        except Exception as e:
            last_error = e
            emit_progress("Evaluate repo", f"克隆失败（第 {attempt}/{max_retries} 次）",
                          level="warning", detail=str(e)[-500:])
            _cleanup_clone(dest_dir)
            time.sleep(1)

    raise_if_cancelled()
    emit_progress("Evaluate repo", "Git clone 失败，尝试 Zip 下载", level="warning")
    try:
        return _download_repo_zip(url, dest_dir)
    except PipelineCancelled:
        _cleanup_clone(dest_dir)
        raise
    except Exception as zip_err:
        _cleanup_clone(dest_dir)
        raise last_error


def _cleanup_clone(dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
    zip_path = dest_dir.parent / (dest_dir.name + ".zip")
    if zip_path.exists():
        zip_path.unlink(missing_ok=True)


def _clone_subprocess(url: str, dest_dir: Path, *, env: dict | None = None, timeout_seconds: int = 600) -> Path:
    env = env or os.environ.copy()
    cmd = ["git", "clone", "--depth", "1", "--progress", url, str(dest_dir)]

    emit_progress("Evaluate repo", "开始克隆仓库",
                  detail=f"git clone --depth 1 --progress {url}", repo_url=url)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="ignore", env=env, bufsize=1,
    )

    stderr_q: Queue[str] = Queue()
    stdout_q: Queue[str] = Queue()
    if proc.stderr is not None:
        Thread(target=_enqueue_lines, args=(proc.stderr, stderr_q), daemon=True).start()
    if proc.stdout is not None:
        Thread(target=_enqueue_lines, args=(proc.stdout, stdout_q), daemon=True).start()

    start = time.monotonic()
    last_emit = 0.0
    last_pct: int | None = None

    try:
        while True:
            raise_if_cancelled()

            now = time.monotonic()
            if now - start > timeout_seconds:
                proc.kill()
                raise TimeoutError(f"git clone timeout after {timeout_seconds}s")

            got_line = False
            for q in (stderr_q, stdout_q):
                while True:
                    try:
                        raw = q.get_nowait()
                    except Empty:
                        break
                    got_line = True
                    for part in raw.replace("\r", "\n").splitlines():
                        clean = normalize_git_line(part)
                        if not clean:
                            continue
                        pct = _extract_pct(clean)
                        if (pct is not None and pct != last_pct) or now - last_emit >= 1.0:
                            last_emit = now
                            if pct is not None:
                                last_pct = pct
                            bar = make_progress_bar(pct)
                            emit_progress("Evaluate repo", "正在克隆仓库",
                                          detail=None,
                                          progress_kind="git_clone",
                                          progress_percent=pct,
                                          progress_bar=bar,
                                          progress_text=clean)

            if proc.poll() is not None:
                break
            if not got_line:
                time.sleep(0.1)

        if proc.returncode != 0:
            tail = "".join(_drain_queue(q) for q in (stderr_q, stdout_q)).strip()
            msg = tail or f"git clone failed exit {proc.returncode}"
            emit_progress("Evaluate repo", "克隆仓库失败", level="warning", detail=msg[-1200:])
            raise RuntimeError(msg)

        emit_progress("Evaluate repo", "仓库克隆完成", detail=f"目标目录：{dest_dir}")
        return dest_dir

    except Exception:
        _cleanup_clone(dest_dir)
        raise


def _enqueue_lines(pipe, q: Queue[str]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if line:
                q.put(line)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _drain_queue(q: Queue[str]) -> list[str]:
    items: list[str] = []
    while True:
        try:
            items.append(q.get_nowait())
        except Empty:
            break
    return items


def detect_git_proxy() -> dict[str, str]:
    """Detect proxy from env vars and git global config."""
    proxies: dict[str, str] = {}
    for k in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        v = os.environ.get(k)
        if v:
            proxies[k] = v
    for cn in ("http.proxy", "https.proxy"):
        try:
            r = subprocess.run(["git", "config", "--global", "--get", cn],
                               capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                proxies[f"git_{cn.replace('.', '_')}"] = r.stdout.strip()
        except Exception:
            pass
    return proxies


def mask_proxy_url(value: str) -> str:
    """Mask credentials in proxy URLs. http://u:p@h:p → http://***@h:p"""
    return re.sub(r"(https?://)([^:@/]+):([^@/]+)@", r"\1***@", value)


def make_progress_bar(percent: int | None, width: int = 15) -> str:
    if percent is None or percent < 0:
        return ""
    pct = min(int(percent), 100)
    fill = int(width * pct / 100)
    return "[" + "█" * fill + "░" * (width - fill) + f"] {pct:3d}%"


def normalize_git_line(line: str) -> str:
    line = line.strip().replace("\r", "")
    if not line:
        return ""
    for src, dst in [
        ("Cloning into", "克隆到"), ("Enumerating objects", "枚举对象"),
        ("Counting objects", "统计对象"), ("Compressing objects", "压缩对象"),
        ("Receiving objects", "接收对象"), ("Resolving deltas", "解析增量"),
        ("Updating files", "更新文件"), ("remote:", "远端:"),
    ]:
        line = line.replace(src, dst)
    return line


def _extract_pct(line: str) -> int | None:
    m = re.search(r"(\d{1,3})%", line)
    if not m:
        return None
    try:
        v = int(m.group(1))
    except ValueError:
        return None
    return v if 0 <= v <= 100 else None


def _download_repo_zip(url: str, dest_dir: Path) -> Path:
    """Download a GitHub repo as zip with streaming progress and cancel support."""
    zip_url = url.rstrip("/")
    if zip_url.endswith(".git"):
        zip_url = zip_url[:-4]

    for branch in ("main", "master"):
        raise_if_cancelled()
        try:
            full_url = f"{zip_url}/archive/refs/heads/{branch}.zip"
            emit_progress("Evaluate repo", "正在下载仓库 ZIP", detail=f"URL：{full_url}")
            zip_path = dest_dir.parent / (dest_dir.name + ".zip")
            zip_path.parent.mkdir(parents=True, exist_ok=True)

            with httpx.stream("GET", full_url, follow_redirects=True, timeout=120) as resp:
                if resp.status_code != 200:
                    continue
                total = int(resp.headers.get("content-length", "0") or 0)
                downloaded = 0
                last_pct = -1
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_bytes():
                        raise_if_cancelled()
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = int(downloaded * 100 / total)
                            if pct != last_pct:
                                last_pct = pct
                                bar = make_progress_bar(pct)
                                emit_progress("Evaluate repo", "正在下载仓库 ZIP",
                                              detail=None,
                                              progress_kind="zip_download",
                                              progress_percent=pct,
                                              progress_bar=bar,
                                              progress_text=f"已下载 {downloaded / 1024 / 1024:.1f} MiB")

            emit_progress("Evaluate repo", "仓库 ZIP 下载完成，正在解压")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest_dir.parent)

            extracted = dest_dir.parent / f"{dest_dir.name}-{branch}"
            if not extracted.exists():
                for item in dest_dir.parent.iterdir():
                    if item.is_dir() and item.name != dest_dir.name:
                        extracted = item
                        break
            if extracted.exists():
                extracted.rename(dest_dir)

            zip_path.unlink(missing_ok=True)
            emit_progress("Evaluate repo", "仓库解压完成", detail=f"目标目录：{dest_dir}")
            return dest_dir

        except PipelineCancelled:
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)
            _cleanup_clone(dest_dir)
            raise
        except Exception as e:
            logger.warning("zip download for branch %s failed: %s", branch, e)
            continue

    raise RuntimeError(f"无法通过 git clone 或 zip 下载仓库: {url}")


def copy_local_repo(src_dir: Path, dest_dir: Path) -> Path:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    ignore = shutil.ignore_patterns(".git", "__pycache__", ".venv", "venv", "node_modules")
    shutil.copytree(src_dir, dest_dir, ignore=ignore)
    return dest_dir


def scan_repo_structure(repo_dir: Path) -> dict:
    has_readme = any(repo_dir.glob("README*"))
    has_requirements = bool(find_requirement_files(repo_dir))
    has_environment_yml = (repo_dir / "environment.yml").exists() or (repo_dir / "environment.yaml").exists()
    has_dockerfile = (repo_dir / "Dockerfile").exists()
    has_setup = (repo_dir / "setup.py").exists() or (repo_dir / "pyproject.toml").exists()

    # Exact-match priority names
    exact_names = [
        "demo.py", "eval.py", "evaluate.py", "test.py", "main.py", "run.py",
        "train.py", "app.py", "inference.py", "predict.py", "benchmark.py",
        "online_demo.py", "minimal_example.py",
    ]

    candidate_scripts = []

    # 1. Exact matches at root level
    for name in exact_names:
        if (repo_dir / name).exists():
            candidate_scripts.append(name)

    # 2. Prefix matches at root level
    for py_file in repo_dir.glob("*.py"):
        if py_file.name not in candidate_scripts:
            if _is_candidate_script(py_file):
                candidate_scripts.append(py_file.name)

    # 3. Scan common demo/tool subdirectories
    for subdir in ["scripts", "examples", "demo", "demos", "app", "tools"]:
        path = repo_dir / subdir
        if path.exists():
            for script in path.glob("*.py"):
                relative = str(script.relative_to(repo_dir))
                if relative not in candidate_scripts and _is_candidate_script(script):
                    candidate_scripts.append(relative)

    candidate_configs = []
    for subdir in ["config", "configs"]:
        path = repo_dir / subdir
        if path.exists():
            for config in list(path.glob("*.yaml")) + list(path.glob("*.yml")) + list(path.glob("*.json")):
                candidate_configs.append(str(config.relative_to(repo_dir)))

    return {
        "has_readme": has_readme,
        "has_requirements": has_requirements,
        "has_environment_yml": has_environment_yml,
        "has_dockerfile": has_dockerfile,
        "has_setup_py_or_pyproject": has_setup,
        "candidate_scripts": sorted(set(candidate_scripts), key=_script_sort_key),
        "candidate_configs": sorted(set(candidate_configs)),
    }


def find_requirement_files(repo_dir: Path, max_depth: int = 3) -> list[Path]:
    files: list[Path] = []

    for path in repo_dir.rglob("requirements*.txt"):
        if not path.is_file():
            continue
        if _is_ignored_requirement_path(repo_dir, path, max_depth):
            continue
        files.append(path)

    for path in repo_dir.rglob("requirements/*.txt"):
        if not path.is_file():
            continue
        if _is_ignored_requirement_path(repo_dir, path, max_depth):
            continue
        files.append(path)

    unique = {path.resolve(): path for path in files}
    return sorted(unique.values(), key=lambda path: _requirement_file_sort_key(repo_dir, path))


def _is_ignored_requirement_path(repo_dir: Path, path: Path, max_depth: int) -> bool:
    try:
        relative = path.relative_to(repo_dir)
    except ValueError:
        return True

    if len(relative.parts) > max_depth + 1:
        return True
    ignored_parts = {".git", ".venv", "venv", "__pycache__", "node_modules"}
    if any(part in ignored_parts or part.startswith(".") for part in relative.parts):
        return True

    lowered_parts = {part.lower() for part in relative.parts[:-1]}
    if lowered_parts.intersection({"docs", "doc", "tests", "test"}):
        return True

    lowered_name = relative.name.lower()
    return any(term in lowered_name for term in ("dev", "test", "doc", "lint"))


def _requirement_file_sort_key(repo_dir: Path, path: Path) -> tuple[int, str]:
    relative = path.relative_to(repo_dir)
    name = path.name.lower()
    rel = relative.as_posix().lower()

    if rel == "requirements.txt":
        return (0, rel)
    if rel == "app/requirements.txt":
        return (1, rel)
    if "runtime" in name or "install" in name:
        return (2, rel)
    if any(term in name for term in ("dev", "test", "doc", "lint")):
        return (20, rel)
    return (10, rel)


def _is_candidate_script(path: Path) -> bool:
    stem = path.stem.lower()
    name = path.name.lower()
    exact_stems = {
        "demo", "eval", "evaluate", "test", "main", "run", "train", "app",
        "inference", "predict", "benchmark", "online_demo", "amg",
    }
    prefix_patterns = (
        "demo", "eval", "test", "main", "run", "train", "match", "infer",
        "predict", "bench", "gradio", "app", "tool", "tutorial", "online",
    )

    if stem in exact_stems:
        return True
    if stem.startswith(prefix_patterns):
        return True
    if "demo" in stem or "example" in stem:
        return True
    if name in {"export_onnx_model.py"}:
        return True
    return False


def _script_sort_key(path: str) -> tuple[int, str]:
    name = Path(path).name.lower()
    stem = Path(path).stem.lower()
    priority = {
        "demo.py": 0,
        "online_demo.py": 1,
        "app.py": 2,
        "inference.py": 3,
        "predict.py": 4,
        "run.py": 5,
        "eval.py": 6,
        "evaluate.py": 7,
        "benchmark.py": 8,
        "minimal_example.py": 8,
        "main.py": 9,
        "test.py": 10,
        "train.py": 20,
    }
    if name in priority:
        return (priority[name], path)
    if stem.startswith("gradio"):
        return (11, path)
    if path.startswith("scripts/"):
        return (12, path)
    if stem.startswith("tool"):
        return (13, path)
    if stem.startswith("tutorial"):
        return (30, path)
    return (40, path)


def compute_runnable_score(scan: dict) -> float:
    score = 0.0

    if scan["has_readme"]:
        score += 0.2
    if scan["has_requirements"] or scan["has_environment_yml"]:
        score += 0.25
    if scan["has_dockerfile"]:
        score += 0.15
    if scan["has_setup_py_or_pyproject"]:
        score += 0.15
    if scan["candidate_scripts"]:
        score += 0.25

    return round(min(score, 1.0), 2)


def detect_risk_flags(repo_dir: Path, scan: dict) -> list[str]:
    flags = []

    if not scan["has_readme"]:
        flags.append("无 README")
    if not (scan["has_requirements"] or scan["has_environment_yml"] or scan["has_setup_py_or_pyproject"]):
        flags.append("无依赖声明文件")
    if not scan["candidate_scripts"]:
        flags.append("无入口脚本")

    readme_text = ""
    for readme in repo_dir.glob("README*"):
        readme_text += readme.read_text(encoding="utf-8", errors="ignore").lower()

    risk_terms = {
        "需要 GPU": ["cuda", "gpu", "nvidia"],
        "可能需要大数据集": ["imagenet", "coco", "download dataset", "large dataset"],
        "可能需要预训练权重": ["checkpoint", "pretrained weights", "model weights"],
    }

    for label, terms in risk_terms.items():
        if any(term in readme_text for term in terms):
            flags.append(label)

    return flags


def extract_pip_requirements_from_environment_file(repo_dir: Path) -> list[str]:
    env_path = _find_environment_file(repo_dir)
    if env_path is None:
        return []

    try:
        data = yaml.safe_load(env_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning("Failed to parse %s: %s", env_path, e)
        return []

    dependencies = data.get("dependencies", [])
    if not isinstance(dependencies, list):
        return []

    requirements: list[str] = []
    for dep in dependencies:
        if isinstance(dep, str):
            normalized = _normalize_environment_dependency(dep)
            if normalized:
                requirements.append(normalized)
        elif isinstance(dep, dict):
            pip_deps = dep.get("pip")
            if isinstance(pip_deps, list):
                for item in pip_deps:
                    if isinstance(item, str) and item.strip():
                        requirements.append(item.strip())

    return _dedupe_preserve_order(requirements)


def _find_environment_file(repo_dir: Path) -> Path | None:
    for name in ("environment.yml", "environment.yaml"):
        path = repo_dir / name
        if path.exists():
            return path
    return None


def _normalize_environment_dependency(spec: str) -> str | None:
    cleaned = spec.split("#", 1)[0].strip()
    if not cleaned:
        return None

    if "::" in cleaned:
        cleaned = cleaned.split("::", 1)[1].strip()
        if not cleaned:
            return None

    if cleaned.startswith(("git+", "http://", "https://")):
        return cleaned

    name = cleaned
    if any(op in cleaned for op in ("==", ">=", "<=", "!=", "~=", ">", "<")):
        for op in ("==", ">=", "<=", "!=", "~=", ">", "<"):
            if op in cleaned:
                name = cleaned.split(op, 1)[0].strip()
                break
        normalized_name = _normalize_environment_name(name)
        if not normalized_name:
            return None
        return cleaned.replace(name, normalized_name, 1)

    if "=" in cleaned:
        parts = [part.strip() for part in cleaned.split("=")]
        name = parts[0]
        normalized_name = _normalize_environment_name(name)
        if not normalized_name:
            return None
        if len(parts) >= 2 and parts[1]:
            return f"{normalized_name}=={parts[1]}"
        return normalized_name

    normalized_name = _normalize_environment_name(name)
    return normalized_name


def _normalize_environment_name(name: str) -> str | None:
    lowered = name.strip().lower()
    if not lowered:
        return None
    if lowered in _SKIPPED_ENV_DEPS:
        return None
    return _ENV_PACKAGE_RENAMES.get(lowered, name.strip())


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
