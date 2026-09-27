"""
Lightweight, fully local, private blockchain ledger for anchoring case evidence.

There is no network, no mining, no consensus, and no external chain (Bitcoin,
Ethereum, or otherwise) involved anywhere in this module -- it is a plain
append-only, hash-linked JSON ledger persisted to disk under outputs/forensics/,
so it survives process restarts.

Each block anchors ONE case's evidence Merkle root plus metadata:
    index, timestamp, case_id, merkle_root, leaf_hashes, evidence_count,
    previous_hash, hash

Only hashes and small metadata are stored in a block -- never raw transaction
data, wallet addresses, or IPs. The full evidence content lives separately in
the local evidence store (src/forensics/evidence.py + EvidenceStore).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.forensics.canonical import canonical_json_bytes, sha256_hex
from src.forensics.merkle import merkle_root as compute_merkle_root

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64


@dataclass
class Block:
    index: int
    timestamp: str
    case_id: str
    merkle_root: str
    leaf_hashes: List[str]
    evidence_count: int
    previous_hash: str
    hash: str = ""

    def header_for_hashing(self) -> Dict[str, Any]:
        """Every field EXCEPT `hash` itself -- this is what gets hashed to produce it."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "case_id": self.case_id,
            "merkle_root": self.merkle_root,
            "leaf_hashes": self.leaf_hashes,
            "evidence_count": self.evidence_count,
            "previous_hash": self.previous_hash,
        }

    def compute_hash(self) -> str:
        return sha256_hex(canonical_json_bytes(self.header_for_hashing()))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Block":
        return Block(
            index=d["index"],
            timestamp=d["timestamp"],
            case_id=d["case_id"],
            merkle_root=d["merkle_root"],
            leaf_hashes=list(d.get("leaf_hashes", [])),
            evidence_count=d.get("evidence_count", len(d.get("leaf_hashes", []))),
            previous_hash=d["previous_hash"],
            hash=d.get("hash", ""),
        )


def _make_genesis_block(timestamp: str) -> Block:
    b = Block(
        index=0,
        timestamp=timestamp,
        case_id="GENESIS",
        merkle_root=compute_merkle_root([]),
        leaf_hashes=[],
        evidence_count=0,
        previous_hash=GENESIS_HASH,
    )
    b.hash = b.compute_hash()
    return b


class Blockchain:
    """Append-only local ledger backed by a single JSON file."""

    def __init__(self, ledger_path: Path):
        self.ledger_path = Path(ledger_path)
        self.blocks: List[Block] = []
        self._load_or_init()

    # ---------------------------------------------------------------- I/O
    def _load_or_init(self):
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, "r") as f:
                    raw = json.load(f)
                self.blocks = [Block.from_dict(b) for b in raw]
                logger.info("Loaded existing forensic ledger with %d block(s) from %s",
                            len(self.blocks), self.ledger_path)
                return
            except Exception as e:
                logger.error("Could not read existing ledger at %s (%s); a corrupt ledger is "
                              "NEVER auto-repaired or discarded -- raising.", self.ledger_path, e)
                raise
        # No ledger yet -- create one, seeded with a genesis block, and persist immediately.
        import datetime
        genesis_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.blocks = [_make_genesis_block(genesis_ts)]
        self._save()
        logger.info("Initialized new forensic ledger at %s with genesis block.", self.ledger_path)

    def _save(self):
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "w") as f:
            json.dump([b.to_dict() for b in self.blocks], f, indent=2)

    # ------------------------------------------------------------- Access
    def latest_block(self) -> Block:
        return self.blocks[-1]

    def find_latest_block_for_case(self, case_id: str) -> Optional[Block]:
        for b in reversed(self.blocks):
            if b.case_id == case_id:
                return b
        return None

    def all_blocks_for_case(self, case_id: str) -> List[Block]:
        return [b for b in self.blocks if b.case_id == case_id]

    # ------------------------------------------------------------- Writes
    def add_block(self, case_id: str, leaf_hashes: List[str]) -> Block:
        """Append a new block anchoring `leaf_hashes` (already-computed evidence
        leaf hashes, in canonical order) for `case_id`. The Merkle root is derived
        here, from the leaf hashes, using the same deterministic algorithm used
        everywhere else in the system."""
        import datetime

        prev = self.latest_block()
        root = compute_merkle_root(leaf_hashes)
        block = Block(
            index=prev.index + 1,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            case_id=case_id,
            merkle_root=root,
            leaf_hashes=list(leaf_hashes),
            evidence_count=len(leaf_hashes),
            previous_hash=prev.hash,
        )
        block.hash = block.compute_hash()
        self.blocks.append(block)
        self._save()
        logger.info("Anchored case %s in block #%d (root=%s...)", case_id, block.index, root[:12])
        return block

    # --------------------------------------------------------- Validation
    def validate_chain(self) -> Dict[str, Any]:
        """Recompute every block's hash and hash-link, and recheck every block's
        Merkle root against its own stored leaf_hashes. This detects tampering
        with the ledger file itself (not just with the separate evidence store)."""
        errors: List[str] = []

        if not self.blocks:
            return {"valid": False, "errors": ["chain is empty (missing genesis block)"], "length": 0}

        if self.blocks[0].previous_hash != GENESIS_HASH or self.blocks[0].case_id != "GENESIS":
            errors.append("block 0 is not a valid genesis block")

        for i, block in enumerate(self.blocks):
            recomputed_hash = block.compute_hash()
            if recomputed_hash != block.hash:
                errors.append(f"block #{block.index}: stored hash does not match recomputed hash "
                              f"(stored={block.hash[:12]}..., recomputed={recomputed_hash[:12]}...)")

            recomputed_root = compute_merkle_root(block.leaf_hashes)
            if recomputed_root != block.merkle_root:
                errors.append(f"block #{block.index}: stored merkle_root does not match root "
                              f"recomputed from its own leaf_hashes")

            if i > 0:
                prev_block = self.blocks[i - 1]
                if block.previous_hash != prev_block.hash:
                    errors.append(f"block #{block.index}: previous_hash does not match block "
                                  f"#{prev_block.index}'s actual hash -- chain link broken")

        return {"valid": len(errors) == 0, "errors": errors, "length": len(self.blocks)}

    def to_list_of_dicts(self) -> List[Dict[str, Any]]:
        return [b.to_dict() for b in self.blocks]
