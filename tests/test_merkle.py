import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forensics.canonical import canonical_json_bytes, hash_record
from src.forensics.merkle import (
    MerkleTree,
    build_merkle_levels,
    generate_merkle_proof,
    merkle_root,
    verify_merkle_proof,
)


def test_canonical_json_is_key_order_independent():
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert canonical_json_bytes(a) == canonical_json_bytes(b)


def test_canonical_json_rounds_floats_consistently():
    a = {"x": 1.0000001}
    b = {"x": 1.00000009999}
    assert canonical_json_bytes(a) == canonical_json_bytes(b)


def test_hash_record_changes_on_any_field_change():
    r1 = {"txid": "a", "amount": 1.0}
    r2 = {"txid": "a", "amount": 1.0001}
    assert hash_record(r1) != hash_record(r2)


def test_merkle_root_single_leaf_equals_leaf_hash():
    leaves = [hash_record({"a": 1})]
    assert merkle_root(leaves) == leaves[0]


def test_merkle_root_deterministic_for_same_input():
    leaves = [hash_record({"i": i}) for i in range(5)]
    assert merkle_root(leaves) == merkle_root(list(leaves))


def test_merkle_proof_verifies_for_every_leaf_odd_and_even_counts():
    for n in (1, 2, 3, 4, 5, 7, 8):
        leaves = [hash_record({"i": i, "n": n}) for i in range(n)]
        root = merkle_root(leaves)
        for i, lh in enumerate(leaves):
            proof = generate_merkle_proof(leaves, i)
            assert verify_merkle_proof(lh, proof, root), f"proof failed for n={n}, i={i}"


def test_tampered_leaf_fails_verification_against_original_root():
    leaves = [hash_record({"i": i}) for i in range(6)]
    root = merkle_root(leaves)
    tampered_hash = hash_record({"i": 2, "extra_field": "tampered"})
    proof = generate_merkle_proof(leaves, 2)
    assert verify_merkle_proof(tampered_hash, proof, root) is False


def test_verify_merkle_proof_never_raises_on_garbage():
    assert verify_merkle_proof("not-a-hash", [("x", "bad-side")], "root") is False


def test_merkle_tree_wrapper():
    leaves = [hash_record({"i": i}) for i in range(4)]
    tree = MerkleTree(leaves)
    assert tree.root == merkle_root(leaves)
    assert tree.verify(1, leaves[1]) is True
    assert tree.verify(1, "wrong-hash") is False
    assert tree.verify(99, leaves[0]) is False
