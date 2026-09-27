"""
Ties cases -> evidence -> Merkle -> blockchain together.

Anchoring is idempotent and append-only:
  - If a case's evidence has never been anchored, a new block is appended.
  - If a case's freshly-derived evidence hashes to the SAME Merkle root as its
    most recent anchored block, nothing new is written (re-running the pipeline
    against unchanged data does not grow the chain).
  - If a case's freshly-derived evidence hashes to a DIFFERENT root than its most
    recent anchored block (e.g. the underlying dataset genuinely changed), a NEW
    block is appended on top -- the old block is never edited or removed. This
    keeps the ledger a true append-only history.

This module never "fixes" or "re-anchors" evidence found to be tampered at
VERIFY time -- see src/forensics/verification.py, which only ever reads and
compares. Anchoring only ever runs as part of a full pipeline run, driven by the
current alerts table, never by the evidence store's on-disk content.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from src.forensics.blockchain import Blockchain
from src.forensics.canonical import hash_record
from src.forensics.cases import Case
from src.forensics.evidence import EvidenceStore, build_case_evidence_records
from src.forensics.merkle import merkle_root

logger = logging.getLogger(__name__)


def anchor_cases(
    cases: List[Case],
    alerts: pd.DataFrame,
    evidence_store: EvidenceStore,
    chain: Blockchain,
) -> Dict[str, Any]:
    alerts_indexed = alerts.set_index("txid", drop=False)

    n_new = 0
    n_updated = 0
    n_unchanged = 0
    case_summaries: List[Dict[str, Any]] = []

    for case in cases:
        evidence_records = build_case_evidence_records(case, alerts_indexed)
        leaf_hashes = [hash_record(r) for r in evidence_records]
        new_root = merkle_root(leaf_hashes)

        existing_block = chain.find_latest_block_for_case(case.case_id)

        if existing_block is None:
            chain.add_block(case.case_id, leaf_hashes)
            evidence_store.set_case(case.case_id, case.to_dict(), evidence_records)
            action = "anchored_new"
            n_new += 1
        elif existing_block.merkle_root != new_root:
            chain.add_block(case.case_id, leaf_hashes)
            evidence_store.set_case(case.case_id, case.to_dict(), evidence_records)
            action = "re_anchored_updated"
            n_updated += 1
        else:
            action = "unchanged"
            n_unchanged += 1
            # Keep the evidence store populated even if the block already existed
            # (e.g. a fresh/rebuilt evidence store next to a pre-existing ledger).
            if not evidence_store.has_case(case.case_id):
                evidence_store.set_case(case.case_id, case.to_dict(), evidence_records)

        case_summaries.append({
            "case_id": case.case_id,
            "action": action,
            "merkle_root": new_root,
            "evidence_count": len(evidence_records),
            "block_index": chain.find_latest_block_for_case(case.case_id).index,
        })

    evidence_store.save()

    logger.info("Anchoring complete: %d new, %d updated, %d unchanged (of %d cases).",
                n_new, n_updated, n_unchanged, len(cases))

    return {
        "n_cases": len(cases),
        "n_new_blocks": n_new,
        "n_updated_blocks": n_updated,
        "n_unchanged": n_unchanged,
        "chain_length": len(chain.blocks),
        "case_summaries": case_summaries,
    }
