"""Small helpers used across the project: seeding, device choice, and record keeping."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every random number generator the project uses.

    Seeding makes runs repeatable: same code + same seed + same machine gives the
    same numbers. PyTorch's own docs warn that results can still differ across
    PyTorch versions, platforms, and CPU vs. GPU, so pinned package versions
    (requirements.txt) matter too.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds the CPU and all GPUs
    if deterministic:
        # Needed for deterministic matrix multiplies on NVIDIA GPUs. It only takes
        # effect if set before the GPU is first used, and is harmless on CPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Prefer deterministic algorithms; warn (instead of crashing) when an
        # operation has no deterministic version.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Pick where the CNN runs: an NVIDIA GPU, an Apple GPU, or the CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def timestamp() -> str:
    """UTC time as a sortable string, e.g. 20260924-153000."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    """The current git commit, so every result can be traced to exact code."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info() -> dict[str, Any]:
    """Record what the results were produced with: Python, OS, package versions, git commit."""
    packages = ["numpy", "pandas", "scikit-learn", "torch", "torchvision", "matplotlib", "medmnist"]
    return {
        "utc_time": timestamp(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in packages},
        "git_commit": _git_commit(),
    }
