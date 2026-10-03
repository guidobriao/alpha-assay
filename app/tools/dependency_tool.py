from __future__ import annotations

import re


MODULE_PACKAGE_RENAMES = {
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "OpenGL": "PyOpenGL",
    "open_clip": "open_clip_torch",
    "clip": "openai-clip",
    "gradio": "gradio",
}

PACKAGE_INSTALL_SPECS = {
    "gradio": "gradio==3.16.2",
}


def package_from_missing_dependency(stderr: str, stdout: str) -> str | None:
    text = stderr + "\n" + stdout
    module_name = _extract_missing_module(text)
    if module_name:
        return package_for_module(module_name)

    distribution = _extract_missing_distribution(text)
    if distribution:
        return normalize_package_name(distribution)

    return None


def package_for_module(module_name: str) -> str | None:
    top_level = module_name.split(".", 1)[0].strip()
    if not top_level:
        return None
    if top_level in MODULE_PACKAGE_RENAMES:
        return MODULE_PACKAGE_RENAMES[top_level]
    return normalize_package_name(top_level.replace("_", "-"))


def install_spec_for_package(package_name: str) -> str:
    return PACKAGE_INSTALL_SPECS.get(package_name, package_name)


def normalize_package_name(name: str) -> str | None:
    package = name.strip().strip("'\"")
    if not package:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", package):
        return None
    return package


def _extract_missing_module(text: str) -> str | None:
    patterns = [
        r"No module named ['\"]([^'\"]+)['\"]",
        r"ModuleNotFoundError:\s*No module named ['\"]([^'\"]+)['\"]",
        r"ImportError:\s*No module named ['\"]([^'\"]+)['\"]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def _extract_missing_distribution(text: str) -> str | None:
    patterns = [
        r"No package metadata was found for ([A-Za-z0-9_.-]+)",
        r"The ['\"]([^'\"]+)['\"] distribution was not found",
        r"DistributionNotFound:\s*The ['\"]([^'\"]+)['\"] distribution was not found",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None
