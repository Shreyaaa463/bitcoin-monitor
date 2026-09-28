"""
Central configuration loader. All modules pull settings from config.yaml through
this module so paths and parameters stay in one place.

Environment overrides (all optional):
    BTC_MONITOR_CONFIG     path to an alternative config.yaml
    BTC_MONITOR_DATASET    input dataset (file or directory) instead of paths.raw_data
    BTC_MONITOR_RUNS_ROOT  runs directory instead of runs.root
    BTC_MONITOR_RUN_ID     run the dashboard/scripts should open by default
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Project root = two levels up from this file (src/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=8)
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


def get_paths(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    cfg = cfg if cfg is not None else load_config()
    paths_cfg = cfg["paths"]
    return {k: resolve_path(v) for k, v in paths_cfg.items()}


def get_runs_root(cfg: Optional[Dict[str, Any]] = None, override: str | os.PathLike | None = None) -> Path:
    if override:
        return resolve_path(str(override))
    if os.environ.get("BTC_MONITOR_RUNS_ROOT"):
        return resolve_path(os.environ["BTC_MONITOR_RUNS_ROOT"])
    cfg = cfg if cfg is not None else load_config()
    return resolve_path((cfg.get("runs") or {}).get("root", "runs"))


def get_dataset_path(cfg: Optional[Dict[str, Any]] = None, override: str | os.PathLike | None = None) -> Path:
    if override:
        return resolve_path(str(override))
    if os.environ.get("BTC_MONITOR_DATASET"):
        return resolve_path(os.environ["BTC_MONITOR_DATASET"])
    return get_paths(cfg)["raw_data"]


def ensure_dirs(cfg: Optional[Dict[str, Any]] = None):
    """Create all configured output/working directories if they don't exist."""
    paths = get_paths(cfg)
    for key in ["processed_dir", "geoip_dir", "models_dir", "outputs_dir"]:
        if key in paths:
            paths[key].mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
