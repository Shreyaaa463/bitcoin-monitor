"""
Demo utility: deliberately modifies one field of one evidence record for a given
case, directly in the local evidence store JSON file, so you can show the
Forensic Blockchain dashboard tab go from VERIFIED to FAILED.

This does NOT touch the blockchain ledger -- only the evidence store. The whole
point of the demo is that the ledger's frozen hash for that evidence record no
longer matches what's now on disk.

Usage:
    python scripts/tamper_demo.py --list
    python scripts/tamper_demo.py --case CASE-XXXXXXXXXXXX
    python scripts/tamper_demo.py --case CASE-XXXXXXXXXXXX --field risk_score --value 0

To undo the tampering, use scripts/reset_case_evidence.py (an explicit, deliberate
action -- the pipeline itself never auto-repairs tampered evidence, by design).
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
    parser.add_argument("--list", action="store_true", help="List anchored case IDs and exit.")
    parser.add_argument("--case", type=str, help="Case ID to tamper with (e.g. CASE-ABCDEF123456).")
    parser.add_argument("--field", type=str, default="risk_score",
                         help="Evidence field to modify (default: risk_score).")
    parser.add_argument("--value", type=str, default="0",
                         help="New value to write into that field (default: 0).")
    parser.add_argument("--index", type=int, default=0,
                         help="Which evidence record within the case to modify (default: 0).")
    args = parser.parse_args()

    if not evidence_store_path.exists():
        print(f"No evidence store found at {evidence_store_path}. Run `python run.py` first.")
        sys.exit(1)

    with open(evidence_store_path, "r") as f:
        store = json.load(f)

    if args.list or not args.case:
        print(f"{len(store)} case(s) in the evidence store:")
        for case_id, entry in store.items():
            n = len(entry.get("evidence_records", []))
            print(f"  {case_id}  ({n} evidence record(s))")
        if not args.case:
            print("\nRe-run with --case <CASE_ID> to tamper with a specific case.")
        return

    if args.case not in store:
        print(f"Case {args.case} not found in the evidence store.")
        sys.exit(1)

    records = store[args.case]["evidence_records"]
    if args.index >= len(records):
        print(f"Case {args.case} only has {len(records)} evidence record(s); "
              f"--index {args.index} is out of range.")
        sys.exit(1)

    record = records[args.index]
    old_value = record.get(args.field, "<field not present>")

    # Try to preserve numeric types where it makes sense.
    new_value: object = args.value
    if isinstance(old_value, (int, float)) and not isinstance(old_value, bool):
        try:
            new_value = float(args.value) if "." in args.value else int(args.value)
        except ValueError:
            pass

    record[args.field] = new_value

    with open(evidence_store_path, "w") as f:
        json.dump(store, f, indent=2)

    print(f"Tampered case {args.case}, evidence record #{args.index} "
          f"(txid={record.get('txid')}): field '{args.field}' changed from "
          f"{old_value!r} to {new_value!r}.")
    print("Open the dashboard's Forensic Blockchain tab and click 'Verify Evidence' "
          f"for {args.case} -- it should now report FAILED.")


if __name__ == "__main__":
    main()
