"""
Central configuration loader. All modules pull settings from config.yaml through
this module so paths and parameters stay in one place.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# Project root = two levels up from this file (src/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load and cache config.yaml. Safe to call from anywhere (pipeline, dashboard, tests)."""
    if config_path is None:
        config_path = os.environ.get("BTC_MONITOR_CONFIG", str(PROJECT_ROOT / "config.yaml"))
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative_path: str) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def get_paths() -> Dict[str, Path]:
    cfg = load_config()
    paths_cfg = cfg["paths"]
    return {k: resolve_path(v) for k, v in paths_cfg.items()}


def ensure_dirs():
    """Create all configured output/working directories if they don't exist."""
    paths = get_paths()
    for key in ["processed_dir", "geoip_dir", "models_dir", "outputs_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
