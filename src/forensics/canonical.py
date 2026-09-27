"""
Canonical serialization and hashing utilities.

Every forensic hash in this system (evidence-record hashes, Merkle leaves,
block hashes) is computed over the SAME canonical byte representation, produced
here. Determinism requirements:

- dict keys are always sorted (json.dumps(..., sort_keys=True))
- separators are fixed (no incidental whitespace differences)
- floats are rounded to a fixed precision before serialization, so the same
  underlying value never serializes two different ways
- all text is encoded as UTF-8

If a single character of an evidence record changes, its canonical bytes change,
and therefore its SHA-256 hash changes. This is the property the whole
tamper-evidence layer (Merkle tree + blockchain) depends on.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

FLOAT_PRECISION = 6


def _round_floats(obj: Any) -> Any:
    """Recursively round every float in a nested structure to FLOAT_PRECISION
    decimal places, so numerically-equal values always serialize identically."""
    if isinstance(obj, float):
        return round(obj, FLOAT_PRECISION)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v) for v in obj]
    return obj


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic JSON serialization: sorted keys, fixed separators, rounded floats."""
    normalized = _round_floats(obj)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_record(obj: Any) -> str:
    """Canonicalize a JSON-serializable object and return its SHA-256 hex digest."""
    return sha256_hex(canonical_json_bytes(obj))
