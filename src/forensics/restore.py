"""
Restore evidence that was DELIBERATELY reset (scripts/reset_case_evidence.py) for a
completed run, without recomputing or re-anchoring anything.

This is what `python run.py` / `./run.sh` do when the requested run already exists
and is completed -- preserving the documented forensic workflow

    tamper -> Verify = FAILED -> reset_case_evidence.py -> run.py -> Verify = VERIFIED

without ever rewriting a completed run's analytical artifacts:
  - only cases ABSENT from the evidence store are touched (tampered-but-present
    evidence is never overwritten -- it keeps failing verification until reset);
  - the evidence is re-derived from the run's own immutable alerts table;
  - it is restored only if it hashes to EXACTLY the Merkle root already anchored in
    the ledger for that case; otherwise it is refused and reported. No block is ever
    appended here.
"""
from __future__ import annotations

import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.dataset as pads

from src.forensics.blockchain import Blockchain
from src.forensics.canonical import hash_record
from src.forensics.cases import Case
from src.forensics.evidence import EvidenceStore, build_case_evidence_records
from src.forensics.merkle import merkle_root
from src.storage.parquet_store import read_frame

logger = logging.getLogger(__name__)

EVIDENCE_COLUMNS = ["txid", "src_ip", "dst_ip", "timestamp_str", "timestamp_parsed", "timestamp",
                    "input_addresses", "output_addresses", "geo_country", "asn", "cluster_label",
                    "anomaly_score_0_100", "risk_score", "priority", "anomaly_type", "reasons"]

_CASE_FIELDS = {f.name for f in fields(Case)}


def case_from_dict(d: Dict[str, Any]) -> Case:
    return Case(**{k: v for k, v in d.items() if k in _CASE_FIELDS})


def restore_missing_evidence(run_dir: Path, evidence_store_path: Path, ledger_path: Path,
                             alerts_path: Path, run_id: str | None) -> Dict[str, Any]:
    cases_path = Path(run_dir) / "cases.json"
    if not cases_path.exists() or not ledger_path.exists():
        return {"restored": [], "refused": [], "not_anchored": [], "checked": 0}
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    store = EvidenceStore(evidence_store_path)
    missing = [c for c in cases if not store.has_case(c["case_id"])]
    result: Dict[str, List[str]] = {"restored": [], "refused": [], "not_anchored": []}
    if not missing:
        return {**result, "checked": len(cases)}

    chain = Blockchain(ledger_path)
    if run_id is not None and chain.run_id != run_id:
        raise RuntimeError(f"Ledger at {ledger_path.name} is bound to run {chain.run_id!r}, not {run_id!r}")

    needed = sorted({t for c in missing for t in c["linked_txids"]})
    schema_names = set(pads.dataset(str(alerts_path), format="parquet", partitioning="hive").schema.names) \
        if Path(alerts_path).is_dir() else None
    cols = [c for c in EVIDENCE_COLUMNS if schema_names is None or c in schema_names]
    alerts = read_frame(alerts_path, columns=cols, filter=pads.field("txid").isin(needed))
    alerts_indexed = alerts.set_index("txid", drop=False)

    for c in missing:
        case = case_from_dict(c)
        block = chain.find_latest_block_for_case(case.case_id)
        if block is None:
            result["not_anchored"].append(case.case_id)
            continue
        records = build_case_evidence_records(case, alerts_indexed)
        if merkle_root([hash_record(r) for r in records]) != block.merkle_root:
            result["refused"].append(case.case_id)
            logger.error("Refusing to restore evidence for %s: re-derived evidence does not match "
                         "the anchored Merkle root (run artifacts may have been modified).", case.case_id)
            continue
        store.set_case(case.case_id, case.to_dict(), records)
        result["restored"].append(case.case_id)

    if result["restored"]:
        store.save()
    return {**result, "checked": len(cases)}
