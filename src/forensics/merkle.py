"""
Deterministic binary Merkle tree over a list of leaf hashes.

Standard construction (same family as Bitcoin's): each internal node's hash is
SHA-256 of the concatenation of its two children's raw hash bytes. If a level has
an odd number of nodes, the last node is duplicated to pair with itself. Leaf
order is significant and must be fixed by the caller (the forensic case builder
orders evidence records by sorted txid before hashing) so the same case always
produces the same root.

A Merkle proof for leaf i is the list of sibling hashes needed to recompute the
root from that single leaf, without needing every other leaf. Verifying a proof
re-derives the root from (leaf_hash, proof) and compares it to the expected root:
if the evidence record backing that leaf was altered in any way, its leaf hash
changes, and the recomputed root will not match.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from src.forensics.canonical import sha256_hex

LEFT = "L"
RIGHT = "R"


def _combine(left_hex: str, right_hex: str) -> str:
    return sha256_hex(bytes.fromhex(left_hex) + bytes.fromhex(right_hex))


def build_merkle_levels(leaf_hashes: List[str]) -> List[List[str]]:
    """Return every level of the tree, leaves first, root last (as a 1-element list)."""
    if not leaf_hashes:
        # Empty-tree convention: root is the hash of an empty byte string.
        return [[sha256_hex(b"")]]

    levels: List[List[str]] = [list(leaf_hashes)]
    current = levels[0]
    while len(current) > 1:
        nxt: List[str] = []
        for i in range(0, len(current), 2):
            left = current[i]
            right = current[i + 1] if i + 1 < len(current) else current[i]  # duplicate last if odd
            nxt.append(_combine(left, right))
        levels.append(nxt)
        current = nxt
    return levels


def merkle_root(leaf_hashes: List[str]) -> str:
    levels = build_merkle_levels(leaf_hashes)
    return levels[-1][0]


def generate_merkle_proof(leaf_hashes: List[str], index: int) -> List[Tuple[str, str]]:
    """Proof for the leaf at `index`: a list of (sibling_hash, side) pairs, ordered
    from the leaf level up to the root, where side is 'L' or 'R' meaning the sibling
    sits to the left or right of the node being combined."""
    if index < 0 or index >= max(len(leaf_hashes), 1):
        raise IndexError(f"leaf index {index} out of range for {len(leaf_hashes)} leaves")

    levels = build_merkle_levels(leaf_hashes)
    proof: List[Tuple[str, str]] = []
    idx = index
    for level in levels[:-1]:  # exclude root level
        is_right_node = idx % 2 == 1
        sibling_idx = idx - 1 if is_right_node else idx + 1
        if sibling_idx >= len(level):
            sibling_idx = idx  # odd-length level: duplicate self
        sibling_hash = level[sibling_idx]
        side = LEFT if is_right_node else RIGHT  # side of the SIBLING relative to current node
        proof.append((sibling_hash, side))
        idx = idx // 2
    return proof


def verify_merkle_proof(leaf_hash: str, proof: List[Tuple[str, str]], expected_root: str) -> bool:
    """Recompute the root from a leaf hash and its proof; compare to expected_root.
    Returns False (never raises) on any mismatch or malformed proof, so callers can
    treat this as a plain tamper-check."""
    try:
        current = leaf_hash
        for sibling_hash, side in proof:
            if side == RIGHT:
                current = _combine(current, sibling_hash)
            elif side == LEFT:
                current = _combine(sibling_hash, current)
            else:
                return False
        return current == expected_root
    except Exception:
        return False


@dataclass
class MerkleTree:
    """Convenience wrapper bundling leaves, levels, and root together."""
    leaf_hashes: List[str]

    @property
    def levels(self) -> List[List[str]]:
        return build_merkle_levels(self.leaf_hashes)

    @property
    def root(self) -> str:
        return merkle_root(self.leaf_hashes)

    def proof(self, index: int) -> List[Tuple[str, str]]:
        return generate_merkle_proof(self.leaf_hashes, index)

    def verify(self, index: int, leaf_hash: str) -> bool:
        if index < 0 or index >= len(self.leaf_hashes):
            return False
        if self.leaf_hashes[index] != leaf_hash:
            return False
        return verify_merkle_proof(leaf_hash, self.proof(index), self.root)
