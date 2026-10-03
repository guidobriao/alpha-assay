from __future__ import annotations

import os


def sanitize_proxy_env() -> None:
    """Make common proxy env vars compatible with httpx/OpenAI clients.

    Some shells export ALL_PROXY=socks5h://... while httpx only accepts
    supported proxy schemes. If protocol-specific HTTP proxies are already
    configured, remove the broad ALL_PROXY fallback; otherwise downgrade the
    scheme to socks5 so environments with socks support can still use it.
    """
    protocol_proxy_exists = any(
        os.environ.get(key)
        for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
    )

    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if not value or not value.lower().startswith("socks5h://"):
            continue
        if protocol_proxy_exists:
            os.environ.pop(key, None)
        else:
            os.environ[key] = "socks5://" + value[len("socks5h://"):]

    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(key)
        if value and value.lower().startswith("socks5h://"):
            os.environ[key] = "socks5://" + value[len("socks5h://"):]
