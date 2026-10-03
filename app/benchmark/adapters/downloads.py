from __future__ import annotations


def download_helper_script() -> str:
    return r'''
def download_with_progress(url, target, dataset_name):
    import json
    import os
    import time
    import urllib.request

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    timeout_seconds = float(os.environ.get("PAPER_BENCH_DATA_DOWNLOAD_TIMEOUT_SECONDS", "900"))
    progress_seconds = float(os.environ.get("PAPER_BENCH_DOWNLOAD_PROGRESS_SECONDS", "10"))
    chunk_size = int(os.environ.get("PAPER_BENCH_DOWNLOAD_CHUNK_BYTES", str(1024 * 1024)))
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    request = urllib.request.Request(url, headers={"User-Agent": "paper-benchmark-runner/1.0"})
    bytes_read = 0
    last_log = time.monotonic()
    start = last_log
    with urllib.request.urlopen(request, timeout=min(max(timeout_seconds, 1), 60)) as response:
        total_header = response.headers.get("Content-Length")
        total_bytes = int(total_header) if total_header and total_header.isdigit() else None
        with target.open("wb") as out:
            while True:
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"download timed out after {timeout_seconds:.0f}s: {url}")
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                bytes_read += len(chunk)
                now = time.monotonic()
                if now - last_log >= progress_seconds:
                    payload = {
                        "stage": "download_progress",
                        "dataset": dataset_name,
                        "target": str(target),
                        "bytes": bytes_read,
                        "total_bytes": total_bytes,
                        "elapsed_seconds": round(now - start, 1),
                    }
                    print(json.dumps(payload), flush=True)
                    last_log = now
    print(json.dumps({
        "stage": "download_complete",
        "dataset": dataset_name,
        "target": str(target),
        "bytes": bytes_read,
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }), flush=True)


def safe_extract_tar(archive, dest):
    import os
    import tarfile

    dest = Path(dest).resolve()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if os.path.commonpath([str(dest), str(target)]) != str(dest):
                raise RuntimeError(f"blocked unsafe tar member: {member.name}")
        tar.extractall(dest)
'''.strip()
