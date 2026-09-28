"""
One-command launcher (invoked by ./run.sh):

    validate environment -> identify dataset -> SHA-256 it -> pipeline/config version
    -> reuse the identical COMPLETED run if one exists, else run the pipeline once
    -> start the Streamlit dashboard on that run  (http://localhost:8501)

Usage:
    ./run.sh                          # default dataset (config paths.raw_data)
    ./run.sh --input data/raw/other.json
    ./run.sh --force                  # new run even if an identical completed one exists
    ./run.sh --dashboard-only         # just open the latest completed run
    ./run.sh --pipeline-only          # run/reuse the pipeline, no dashboard
    ./run.sh --check                  # environment validation only

Nothing here needs network access. The dashboard is bound to 127.0.0.1 by default.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, get_dataset_path, get_runs_root, load_config  # noqa: E402

MIN_PYTHON = (3, 10)
REQUIRED_MODULES = [
    ("pandas", "pandas"), ("numpy", "numpy"), ("sklearn", "scikit-learn"), ("scipy", "scipy"),
    ("networkx", "networkx"), ("pyarrow", "pyarrow"), ("yaml", "PyYAML"), ("joblib", "joblib"),
    ("streamlit", "streamlit"), ("plotly", "plotly"), ("pyvis", "pyvis"),
]


def _say(msg: str):
    print(f"[launcher] {msg}", flush=True)


def validate_environment(cfg, dataset_path: Path | None, runs_root: Path, port: int | None,
                         host: str) -> list[str]:
    problems: list[str] = []
    if sys.version_info < MIN_PYTHON:
        problems.append(f"Python >= {'.'.join(map(str, MIN_PYTHON))} required, found {sys.version.split()[0]}")
    missing = []
    for mod, pkg in REQUIRED_MODULES:
        try:
            __import__(mod)
        except Exception as e:  # ImportError or a broken binary wheel
            missing.append(f"{pkg} ({type(e).__name__}: {e})")
    if missing:
        problems.append("Missing/broken packages: " + "; ".join(missing)
                        + ". Install offline with scripts/install_offline.sh (see README).")
    if dataset_path is not None and not dataset_path.exists():
        problems.append(f"Dataset not found: {dataset_path} (set paths.raw_data in config.yaml or pass --input)")
    try:
        runs_root.mkdir(parents=True, exist_ok=True)
        probe = runs_root / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        problems.append(f"Runs directory {runs_root} is not writable: {e}")
    free_gb = shutil.disk_usage(runs_root if runs_root.exists() else PROJECT_ROOT).free / 1e9
    if free_gb < 1.0:
        problems.append(f"Less than 1 GB free disk space under {runs_root} ({free_gb:.2f} GB)")
    if port is not None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host if host != "0.0.0.0" else "", port))
            except OSError:
                problems.append(f"Port {port} on {host} is already in use (another dashboard running? "
                                f"use --port)")
    return problems


def start_dashboard(run_id: str | None, runs_root: Path, host: str, port: int, config_path: str | None) -> int:
    env = dict(os.environ)
    env["BTC_MONITOR_RUNS_ROOT"] = str(runs_root)
    if run_id:
        env["BTC_MONITOR_RUN_ID"] = run_id
    if config_path:
        env["BTC_MONITOR_CONFIG"] = str(Path(config_path).resolve())
    cmd = [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "dashboard" / "app.py"),
           "--server.headless", "true",
           "--server.address", host,
           "--server.port", str(port),
           "--browser.gatherUsageStats", "false",   # no telemetry: fully offline
           "--server.fileWatcherType", "none",      # never re-run on file changes in production
           ]
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    _say(f"Dashboard: http://{shown}:{port}   (run {run_id or 'latest completed'}; Ctrl+C to stop)")
    try:
        return subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)
    except KeyboardInterrupt:
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate, run/reuse the pipeline, and start the dashboard.")
    ap.add_argument("--input", help="dataset file or directory (default: config paths.raw_data)")
    ap.add_argument("--config", help="alternative config.yaml")
    ap.add_argument("--runs-root", help="runs directory (default: config runs.root)")
    ap.add_argument("--run", help="open this completed RUN_ID in the dashboard (implies --dashboard-only)")
    ap.add_argument("--force", action="store_true", help="produce a new run even if an identical one is completed")
    ap.add_argument("--verify-deep", action="store_true", help="SHA-256-verify a reused run's artifacts")
    ap.add_argument("--dashboard-only", action="store_true")
    ap.add_argument("--pipeline-only", action="store_true")
    ap.add_argument("--check", action="store_true", help="validate the environment and exit")
    ap.add_argument("--host", help="dashboard bind address (default: launcher.host = 127.0.0.1)")
    ap.add_argument("--port", type=int, help="dashboard port (default: launcher.port = 8501)")
    args = ap.parse_args(argv)

    if args.config:
        os.environ["BTC_MONITOR_CONFIG"] = str(Path(args.config).resolve())
    cfg = load_config(args.config)
    lcfg = cfg.get("launcher") or {}
    host = args.host or os.environ.get("BTC_MONITOR_HOST") or lcfg.get("host", "127.0.0.1")
    port = args.port or int(os.environ.get("BTC_MONITOR_PORT") or lcfg.get("port", 8501))
    runs_root = get_runs_root(cfg, args.runs_root)
    dashboard_only = args.dashboard_only or bool(args.run)
    dataset_path = None if dashboard_only else get_dataset_path(cfg, args.input)

    _say(f"Python {sys.version.split()[0]} | project {PROJECT_ROOT.name} | runs dir {runs_root.name}/")
    problems = validate_environment(cfg, dataset_path, runs_root, None if args.pipeline_only else port, host)
    if problems:
        for p in problems:
            _say(f"ERROR: {p}")
        return 2
    _say("Environment OK (offline: no network access is used).")
    if args.check:
        return 0

    run_id = args.run
    if not dashboard_only:
        from src.pipeline import run_pipeline
        try:
            result = run_pipeline(dataset_path, args.config, runs_root, force=args.force,
                                  verify_deep=args.verify_deep)
        except Exception as e:
            _say(f"Pipeline FAILED: {type(e).__name__}: {e}")
            _say("The failed run is recorded as status=failed and will not be shown in the dashboard. "
                 "Start the dashboard on the previous completed run with: ./run.sh --dashboard-only")
            return 1
        run_id = result["run_id"]
        _say(f"Run {run_id}: {'REUSED (identical completed run, not recomputed)' if result['reused'] else 'COMPLETED'}")
    else:
        from src.runs.registry import RunRegistry
        registry = RunRegistry(runs_root)
        if run_id:
            m = registry.get(run_id)
            if m is None or not m.is_completed:
                _say(f"ERROR: run {run_id} does not exist or is not completed.")
                return 1
        elif registry.latest_completed() is None and not (PROJECT_ROOT / "outputs" / "alerts.parquet").exists():
            _say("ERROR: no completed run yet. Run ./run.sh without --dashboard-only first.")
            return 1

    if args.pipeline_only:
        return 0
    return start_dashboard(run_id, runs_root, host, port, args.config)


if __name__ == "__main__":
    sys.exit(main())
