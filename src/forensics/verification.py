"""
Verification: the read-only half of the forensic layer.

verify_case() and generate_proof_for_txid() NEVER modify the evidence store or
the blockchain, and they NEVER "fix" or re-anchor anything they find wrong.
They only:
  1. read the CURRENT on-disk evidence for a case,
  2. re-hash it with the exact same canonicalization used at anchoring time,
  3. compare those hashes against the ones FROZEN inside the case's most recent
     anchored block, and
  4. re-validate the blockchain's own internal hash-links and Merkle roots.

If a single field of one evidence record was edited after anchoring, step 2
produces a different hash for that record, step 3's comparison fails, and the
Merkle root recomputed from current hashes will not match the root frozen in the
block -- this is what "verification fails" means here, and it is never silently
repaired.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.forensics.blockchain import Blockchain
from src.forensics.canonical import hash_record
from src.forensics.evidence import EvidenceStore
from src.forensics.merkle import generate_merkle_proof, merkle_root, verify_merkle_proof

logger = logging.getLogger(__name__)


def verify_case(case_id: str, evidence_store: EvidenceStore, chain: Blockchain,
                expected_run_id: Optional[str] = None) -> Dict[str, Any]:
    """expected_run_id: the run whose evidence is being verified. When given, the
    anchoring block (and the whole ledger) must be bound to that run."""
    block = chain.find_latest_block_for_case(case_id)
    if block is None:
        return {
            "case_id": case_id,
            "status": "NOT_ANCHORED",
            "overall_verified": False,
            "message": "No block in the forensic ledger anchors this case.",
        }

    stored_records = evidence_store.get_case_evidence(case_id)
    current_leaf_hashes = [hash_record(r) for r in stored_records]
    anchored_leaf_hashes = block.leaf_hashes

    count_mismatch = len(current_leaf_hashes) != len(anchored_leaf_hashes)
    n = max(len(current_leaf_hashes), len(anchored_leaf_hashes))

    per_txid_results: List[Dict[str, Any]] = []
    for i in range(n):
        txid = stored_records[i]["txid"] if i < len(stored_records) else None
        current_hash = current_leaf_hashes[i] if i < len(current_leaf_hashes) else None
        anchored_hash = anchored_leaf_hashes[i] if i < len(anchored_leaf_hashes) else None
        match = current_hash is not None and anchored_hash is not None and current_hash == anchored_hash
        per_txid_results.append({
            "index": i,
            "txid": txid,
            "current_hash": current_hash,
            "anchored_hash": anchored_hash,
            "match": match,
        })

    all_hashes_match = (not count_mismatch) and all(r["match"] for r in per_txid_results)
    recomputed_root = merkle_root(current_leaf_hashes)
    merkle_root_match = recomputed_root == block.merkle_root

    chain_validation = chain.validate_chain(expected_run_id=expected_run_id)
    run_binding_ok = expected_run_id is None or block.run_id == expected_run_id

    overall_verified = all_hashes_match and merkle_root_match and chain_validation["valid"] and run_binding_ok

    if overall_verified:
        message = "Evidence matches the anchored blockchain record exactly. Chain integrity confirmed."
    else:
        problems = []
        if count_mismatch:
            problems.append("evidence record count differs from what was anchored")
        if not all_hashes_match:
            n_bad = sum(1 for r in per_txid_results if not r["match"])
            problems.append(f"{n_bad} evidence record(s) do not match their anchored hash")
        if not merkle_root_match:
            problems.append("recomputed Merkle root does not match the anchored root")
        if not chain_validation["valid"]:
            problems.append("blockchain ledger itself failed internal validation")
        if not run_binding_ok:
            problems.append(f"anchoring block is bound to run {block.run_id!r}, not {expected_run_id!r}")
        message = "Verification FAILED: " + "; ".join(problems)

    return {
        "case_id": case_id,
        "status": "VERIFIED" if overall_verified else "FAILED",
        "overall_verified": overall_verified,
        "message": message,
        "block_index": block.index,
        "block_hash": block.hash,
        "block_timestamp": block.timestamp,
        "anchored_merkle_root": block.merkle_root,
        "recomputed_merkle_root": recomputed_root,
        "merkle_root_match": merkle_root_match,
        "evidence_count_match": not count_mismatch,
        "per_txid_results": per_txid_results,
        "chain_valid": chain_validation["valid"],
        "chain_errors": chain_validation["errors"],
        "run_id": block.run_id,
        "run_binding_ok": run_binding_ok,
    }


def generate_proof_for_txid(case_id: str, txid: str, evidence_store: EvidenceStore,
                             chain: Blockchain) -> Dict[str, Any]:
    """Full drill-down for one transaction's evidence: current hash, the hash frozen
    in the blockchain, a Merkle proof (built over the ANCHORED leaf hashes, which
    define the tree structure that was actually committed), and whether replaying
    that proof with the CURRENT evidence hash reconstructs the anchored root."""
    block = chain.find_latest_block_for_case(case_id)
    if block is None:
        return {"error": f"case {case_id} is not anchored in the forensic ledger."}

    stored_records = evidence_store.get_case_evidence(case_id)
    index: Optional[int] = None
    for i, rec in enumerate(stored_records):
        if rec.get("txid") == txid:
            index = i
            break
    if index is None:
        return {"error": f"txid {txid} not found in the current evidence for case {case_id}."}

    current_hash = hash_record(stored_records[index])
    anchored_hash = block.leaf_hashes[index] if index < len(block.leaf_hashes) else None

    try:
        proof = generate_merkle_proof(block.leaf_hashes, index)
        proof_reconstructs_root = verify_merkle_proof(current_hash, proof, block.merkle_root)
    except Exception as e:
        logger.warning("Could not generate/verify Merkle proof for %s/%s: %s", case_id, txid, e)
        proof = []
        proof_reconstructs_root = False

    return {
        "case_id": case_id,
        "txid": txid,
        "leaf_index": index,
        "current_hash": current_hash,
        "anchored_hash": anchored_hash,
        "hash_match": current_hash == anchored_hash,
        "proof": [{"sibling_hash": s, "side": side} for s, side in proof],
        "block_index": block.index,
        "anchored_merkle_root": block.merkle_root,
        "proof_reconstructs_anchored_root": proof_reconstructs_root,
    }
