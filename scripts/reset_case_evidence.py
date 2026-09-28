"""
Explicit, deliberate reset of one case's evidence in a run's local evidence store,
so it can be freshly re-derived from that run's (immutable) alerts table the next
time `python run.py` / `./run.sh` is executed for the same dataset + config.

This is intentionally a SEPARATE, manual step from running the pipeline: the
pipeline itself never silently overwrites evidence for a case that is already
present in the evidence store, even if that evidence was tampered with, because
"do not silently regenerate/re-anchor modified evidence" is a hard requirement
of this system. Use this script only when you deliberately want to discard the
current stored evidence for a case (e.g. resetting after a tamper demo).

The re-derived evidence is only restored if it hashes to exactly the Merkle root
already anchored for that case; nothing is re-anchored and the ledger is never edited.

Usage:
    python scripts/reset_case_evidence.py --case CASE-XXXXXXXXXXXX [--run RUN_ID]
    python run.py    # restores that case's evidence from the run's alerts table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, get_runs_root, load_config
from src.runs.fsutil import atomic_write_json
from src.runs.registry import LEGACY_RUN_ID, RunRegistry, resolve_artifact_dir
from src.serving.run_store import RunStore


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=str, required=True, help="Case ID to reset.")
    parser.add_argument("--run", type=str, default=None,
                        help="Completed RUN_ID to operate on (default: latest completed run).")
    args = parser.parse_args()

    runs_root = get_runs_root(cfg)
    run_id, run_dir = resolve_artifact_dir(PROJECT_ROOT, runs_root, args.run)
    if run_id is None:
        print("No completed run found (or --run is not a completed run).")
        sys.exit(1)
    evidence_store_path = RunStore(run_id, run_dir, cfg.get("forensics", {})).evidence_store_path

    if not evidence_store_path.exists():
        print(f"No evidence store found for run {run_id}.")
        sys.exit(1)

    lock = None if run_id == LEGACY_RUN_ID else RunRegistry(runs_root).lock_for(f"run-{run_id}")
    if lock is not None and not lock.acquire(wait_seconds=30):
        print(f"Run {run_id} is busy (another process is updating its evidence); try again.")
        sys.exit(1)
    try:
        with open(evidence_store_path, "r") as f:
            store = json.load(f)

        if args.case not in store:
            print(f"Case {args.case} not found in the evidence store of run {run_id} -- nothing to reset.")
            sys.exit(1)

        del store[args.case]
        atomic_write_json(evidence_store_path, store)
    finally:
        if lock is not None:
            lock.release()

    print(f"Removed stored evidence for {args.case} (run {run_id}). Note: the blockchain ledger still "
          f"retains its historical block(s) for this case -- the ledger is append-only "
          f"and is never edited by this script.")
    print("Run `python run.py` (or ./run.sh) to re-derive this case's evidence from the run's "
          "alerts and restore it (only if it matches the anchored Merkle root).")


if __name__ == "__main__":
    main()
