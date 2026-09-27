"""
Explicit, deliberate reset of one case's evidence in the local evidence store,
so it can be freshly re-derived (and re-anchored) from the current alerts table
the next time `python run.py` runs.

This is intentionally a SEPARATE, manual step from running the pipeline: the
pipeline itself never silently overwrites evidence for a case that is already
present in the evidence store, even if that evidence was tampered with, because
"do not silently regenerate/re-anchor modified evidence" is a hard requirement
of this system. Use this script only when you deliberately want to discard the
current stored evidence for a case (e.g. resetting after a tamper demo).

Usage:
    python scripts/reset_case_evidence.py --case CASE-XXXXXXXXXXXX
    python run.py    # will now freshly derive and re-anchor that case's evidence
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_paths, load_config


def main():
    cfg = load_config()
    paths = get_paths()
    forensics_cfg = cfg.get("forensics", {})
    evidence_store_path = paths["outputs_dir"] / forensics_cfg.get(
        "evidence_store_filename", "forensics/evidence_store.json")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=str, required=True, help="Case ID to reset.")
    args = parser.parse_args()

    if not evidence_store_path.exists():
        print(f"No evidence store found at {evidence_store_path}.")
        sys.exit(1)

    with open(evidence_store_path, "r") as f:
        store = json.load(f)

    if args.case not in store:
        print(f"Case {args.case} not found in the evidence store -- nothing to reset.")
        sys.exit(1)

    del store[args.case]

    with open(evidence_store_path, "w") as f:
        json.dump(store, f, indent=2)

    print(f"Removed stored evidence for {args.case}. Note: the blockchain ledger still "
          f"retains its historical block(s) for this case -- the ledger is append-only "
          f"and is never edited by this script.")
    print("Run `python run.py` to freshly derive this case's evidence from the current "
          "alerts and re-anchor it.")


if __name__ == "__main__":
    main()
