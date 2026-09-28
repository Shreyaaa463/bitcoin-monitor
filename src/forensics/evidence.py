"""
Canonical evidence records + the local evidence store.

For every transaction linked into a case, this module packages the SAME values
already produced by the existing pipeline (never recomputed) into one canonical,
deterministically-serializable dict -- the "evidence record". Each evidence
record's SHA-256 hash becomes one Merkle-tree leaf for its case
(src/forensics/merkle.py); the case's Merkle root is what gets anchored into the
local blockchain (src/forensics/blockchain.py).

The EvidenceStore persists the FULL evidence content (not just hashes) to a
local JSON file under outputs/forensics/, so the dashboard's Case -> transactions
-> entities -> evidence drill-down has something to display, and so verification
has a live, current copy of "what the evidence currently says" to re-hash and
compare against the frozen hashes anchored in the blockchain.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.forensics.canonical import hash_record
from src.forensics.cases import Case
from src.runs.fsutil import atomic_write_bytes

logger = logging.getLogger(__name__)


def build_evidence_record(row: pd.Series) -> Dict[str, Any]:
    """Build one canonical, JSON-serializable evidence record for a single
    transaction row from the alerts table. Only fields already produced by the
    existing pipeline are used -- nothing here is recomputed."""
    def _clean_list(v):
        if v is None:
            return []
        try:
            return sorted(str(x) for x in v if x)
        except TypeError:
            return []

    record = {
        "txid": str(row["txid"]),
        "src_ip": str(row["src_ip"]) if pd.notna(row.get("src_ip")) else None,
        "dst_ip": str(row["dst_ip"]) if pd.notna(row.get("dst_ip")) else None,
        "timestamp": str(row.get("timestamp_str") or row.get("timestamp_parsed") or row.get("timestamp")),
        "input_addresses": _clean_list(row.get("input_addresses")),
        "output_addresses": _clean_list(row.get("output_addresses")),
        "geo_country": str(row["geo_country"]) if pd.notna(row.get("geo_country")) else None,
        "asn": str(row["asn"]) if pd.notna(row.get("asn")) else None,
        "cluster_label": int(row["cluster_label"]) if pd.notna(row.get("cluster_label")) else None,
        "anomaly_score_0_100": float(row["anomaly_score_0_100"]) if pd.notna(row.get("anomaly_score_0_100")) else None,
        "risk_score": float(row["risk_score"]) if pd.notna(row.get("risk_score")) else None,
        "priority": str(row["priority"]) if pd.notna(row.get("priority")) else None,
        "anomaly_type": str(row["anomaly_type"]) if pd.notna(row.get("anomaly_type")) else None,
        "reasons": str(row["reasons"]) if pd.notna(row.get("reasons")) else "",
    }
    return record


def build_case_evidence_records(case: Case, alerts_indexed: pd.DataFrame) -> List[Dict[str, Any]]:
    """Evidence records for every txid in `case`, ordered by sorted txid (the same
    order as case.linked_txids) so hashing/Merkle construction is fully deterministic."""
    records = []
    for txid in case.linked_txids:  # already sorted by the case builder
        row = alerts_indexed.loc[txid]
        if isinstance(row, pd.DataFrame):  # duplicate txid guard -- take the first
            row = row.iloc[0]
        records.append(build_evidence_record(row))
    return records


class EvidenceStore:
    """Persists { case_id: {case_metadata, evidence_records: [...]} } to a single
    local JSON file. This is the ONLY place full evidence content lives -- the
    blockchain never stores it, only its hashes."""

    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        self.data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if self.store_path.exists():
            try:
                with open(self.store_path, "r") as f:
                    self.data = json.load(f)
                logger.info("Loaded evidence store with %d case(s) from %s",
                            len(self.data), self.store_path)
            except Exception as e:
                logger.error("Could not read evidence store at %s (%s); refusing to silently "
                              "discard it.", self.store_path, e)
                raise
        else:
            self.data = {}

    def save(self):
        # Atomic replace so a crash or a concurrent reader never sees a half-written store.
        atomic_write_bytes(self.store_path, json.dumps(self.data, indent=2).encode("utf-8"))

    def get_case_evidence(self, case_id: str) -> List[Dict[str, Any]]:
        entry = self.data.get(case_id)
        if not entry:
            return []
        return entry.get("evidence_records", [])

    def get_case_metadata(self, case_id: str) -> Dict[str, Any]:
        entry = self.data.get(case_id)
        return entry.get("case_metadata", {}) if entry else {}

    def has_case(self, case_id: str) -> bool:
        return case_id in self.data

    def set_case(self, case_id: str, case_metadata: Dict[str, Any], evidence_records: List[Dict[str, Any]]):
        self.data[case_id] = {
            "case_metadata": case_metadata,
            "evidence_records": evidence_records,
        }

    def all_case_ids(self) -> List[str]:
        return list(self.data.keys())

    def compute_current_leaf_hashes(self, case_id: str) -> List[str]:
        """Hash the CURRENT on-disk evidence for a case, in stored order. This is
        deliberately a fresh recomputation from whatever is on disk right now --
        if someone hand-edited the JSON file, this reflects that edited content,
        which is exactly what verification needs to compare against the frozen
        blockchain hashes."""
        records = self.get_case_evidence(case_id)
        return [hash_record(r) for r in records]
