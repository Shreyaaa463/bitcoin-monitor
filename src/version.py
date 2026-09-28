"""
Pipeline version + code fingerprint.

PIPELINE_VERSION is part of every run's identity (together with the dataset
SHA-256 and the analytical-config hash). Bump it whenever a change to src/ can
change analytical output; a completed run is then no longer reused for the same
dataset/config.

code_fingerprint() is recorded in each run manifest for provenance. The launcher
warns (but does not rerun) when a reused run was produced by different code under
the same PIPELINE_VERSION.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PIPELINE_VERSION = "2.0.0"

_SRC_DIR = Path(__file__).resolve().parent


def code_fingerprint() -> str:
    """SHA-256 over every src/**/*.py file (sorted relative path + content)."""
    h = hashlib.sha256()
    for p in sorted(_SRC_DIR.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(_SRC_DIR).as_posix().encode("utf-8") + b"\0")
        h.update(p.read_bytes() + b"\0")
    return h.hexdigest()
