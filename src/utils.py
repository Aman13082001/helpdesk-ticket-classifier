"""Shared helpers: config loading, project paths, seeding, logging, JSON I/O.

Every script imports from here so behaviour (paths, seeds, log format) is consistent.
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Project root = parent of the `src/` folder. Using this instead of the current working
# directory means scripts work no matter where they are launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# One process-wide lock for constructing torch models. Streamlit can run reruns on parallel
# threads; loading two Hugging Face models at the same time is not thread-safe.
MODEL_LOAD_LOCK = threading.RLock()


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a single, readable console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml (project root by default)."""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path: str | Path) -> Path:
    """Turn a config-relative path into an absolute path under the project root."""
    p = Path(rel_path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dir(path: str | Path) -> Path:
    p = resolve(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    """Seed every RNG we use so results are reproducible run-to-run."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for classical-only runs
        pass


def save_json(obj: Any, path: str | Path) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(resolve(path), encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def dir_size_mb(path: str | Path) -> float:
    """Size of a file or folder on disk in MB (used for the model comparison table)."""
    p = resolve(path)
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
