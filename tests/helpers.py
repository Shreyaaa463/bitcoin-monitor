import copy
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_dataset(path: Path, n: int, seed: int = 7, fmt: str = "json") -> Path:
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "generate_synthetic_dataset.py"),
                    "--n", str(n), "--seed", str(seed), "--format", fmt, "--out", str(path)],
                   check=True, capture_output=True)
    return path


def write_config(tmp_path: Path, name: str = "config.yaml", **overrides) -> Path:
    """Copy of the project config with paths pointed into tmp_path, plus dotted overrides
    such as {"model.isolation_forest.contamination": 0.1}."""
    cfg = copy.deepcopy(yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text()))
    cfg["paths"]["processed_dir"] = str(tmp_path / "processed")
    cfg["paths"]["geoip_dir"] = str(tmp_path / "geoip")
    cfg["runs"]["root"] = str(tmp_path / "runs")
    cfg["runs"]["lock_wait_seconds"] = 5
    for dotted, value in overrides.items():
        node = cfg
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg))
    return p
