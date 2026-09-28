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

Run binding: when a ledger is created for a pipeline run (runs/<RUN_ID>/forensics/),
every block -- including genesis -- carries that run_id inside its hashed header,
so a block (and the Merkle root it anchors) cannot be silently moved to, or
confused with, another run. Ledgers without run_id (legacy outputs/) hash exactly
as before.

What this ledger is: a local/private, append-only, tamper-EVIDENT log. What it is
not: a public blockchain, a consensus system, or protection against an attacker
with write access to the whole filesystem (who could rebuild a consistent ledger);
it detects modification, it cannot prove the underlying evidence is true.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.forensics.canonical import canonical_json_bytes, sha256_hex
from src.forensics.merkle import merkle_root as compute_merkle_root
from src.runs.fsutil import atomic_write_bytes

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
    run_id: Optional[str] = None

    def header_for_hashing(self) -> Dict[str, Any]:
        """Every field EXCEPT `hash` itself -- this is what gets hashed to produce it.
        run_id is only included when set, so legacy (unbound) blocks hash unchanged."""
        header = {
            "index": self.index,
            "timestamp": self.timestamp,
            "case_id": self.case_id,
            "merkle_root": self.merkle_root,
            "leaf_hashes": self.leaf_hashes,
            "evidence_count": self.evidence_count,
            "previous_hash": self.previous_hash,
        }
        if self.run_id is not None:
            header["run_id"] = self.run_id
        return header

    def compute_hash(self) -> str:
        return sha256_hex(canonical_json_bytes(self.header_for_hashing()))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("run_id") is None:
            d.pop("run_id", None)
        return d

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
            run_id=d.get("run_id"),
        )


def _make_genesis_block(timestamp: str, run_id: Optional[str] = None) -> Block:
    b = Block(
        index=0,
        timestamp=timestamp,
        case_id="GENESIS",
        merkle_root=compute_merkle_root([]),
        leaf_hashes=[],
        evidence_count=0,
        previous_hash=GENESIS_HASH,
        run_id=run_id,
    )
    b.hash = b.compute_hash()
    return b


class Blockchain:
    """Append-only local ledger backed by a single JSON file.

    run_id: bind a NEW ledger to a pipeline run (ignored for an existing ledger file,
    whose own genesis run_id is authoritative)."""

    def __init__(self, ledger_path: Path, run_id: Optional[str] = None):
        self.ledger_path = Path(ledger_path)
        self.blocks: List[Block] = []
        self._requested_run_id = run_id
        self._defer_depth = 0
        self._dirty = False
        self._latest_by_case: Dict[str, int] = {}
        self._load_or_init()
        self._reindex()

    @property
    def run_id(self) -> Optional[str]:
        return self.blocks[0].run_id if self.blocks else None

    def _reindex(self):
        self._latest_by_case = {b.case_id: i for i, b in enumerate(self.blocks)}

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
        self.blocks = [_make_genesis_block(genesis_ts, self._requested_run_id)]
        self._save()
        logger.info("Initialized new forensic ledger at %s with genesis block.", self.ledger_path)

    def _save(self):
        if self._defer_depth > 0:
            self._dirty = True
            return
        # Atomic replace: a crash mid-write can never leave a truncated ledger file.
        atomic_write_bytes(self.ledger_path,
                           json.dumps([b.to_dict() for b in self.blocks], indent=2).encode("utf-8"))
        self._dirty = False

    @contextmanager
    def deferred_save(self):
        """Batch many add_block() calls into ONE atomic write at the end (instead of
        rewriting the whole ledger file after every block, which is O(n^2) I/O).
        If the block raises, nothing is persisted -- the on-disk ledger is unchanged."""
        self._defer_depth += 1
        try:
            yield self
        except BaseException:
            self._defer_depth -= 1
            if self._defer_depth == 0 and self._dirty:
                self.blocks = []
                self._load_or_init()  # discard un-persisted blocks
                self._reindex()
                self._dirty = False
            raise
        self._defer_depth -= 1
        if self._defer_depth == 0 and self._dirty:
            self._save()

    # ------------------------------------------------------------- Access
    def latest_block(self) -> Block:
        return self.blocks[-1]

    def find_latest_block_for_case(self, case_id: str) -> Optional[Block]:
        i = self._latest_by_case.get(case_id)
        return self.blocks[i] if i is not None else None

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
            run_id=self.run_id,
        )
        block.hash = block.compute_hash()
        self.blocks.append(block)
        self._latest_by_case[case_id] = len(self.blocks) - 1
        self._save()
        logger.debug("Anchored case %s in block #%d (root=%s...)", case_id, block.index, root[:12])
        return block

    # --------------------------------------------------------- Validation
    def validate_chain(self, expected_run_id: Optional[str] = None) -> Dict[str, Any]:
        """Recompute every block's hash and hash-link, and recheck every block's
        Merkle root against its own stored leaf_hashes. This detects tampering
        with the ledger file itself (not just with the separate evidence store).
        With expected_run_id, also checks every block is bound to that run."""
        errors: List[str] = []

        if not self.blocks:
            return {"valid": False, "errors": ["chain is empty (missing genesis block)"], "length": 0}

        if self.blocks[0].previous_hash != GENESIS_HASH or self.blocks[0].case_id != "GENESIS":
            errors.append("block 0 is not a valid genesis block")

        genesis_run = self.blocks[0].run_id
        if expected_run_id is not None and genesis_run != expected_run_id:
            errors.append(f"ledger is bound to run {genesis_run!r}, expected {expected_run_id!r}")
        for block in self.blocks[1:]:
            if block.run_id != genesis_run:
                errors.append(f"block #{block.index}: run_id {block.run_id!r} differs from the "
                              f"ledger's genesis run_id {genesis_run!r}")

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

        return {"valid": len(errors) == 0, "errors": errors, "length": len(self.blocks),
                "run_id": genesis_run, "tip_hash": self.blocks[-1].hash}

    def to_list_of_dicts(self) -> List[Dict[str, Any]]:
        return [b.to_dict() for b in self.blocks]
