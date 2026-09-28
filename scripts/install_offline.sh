#!/usr/bin/env bash
# Create .venv and install all dependencies WITHOUT Internet access, from wheelhouse/
# (produced by scripts/build_offline_bundle.sh on a connected machine).
#
# Usage:  scripts/install_offline.sh            (then: ./run.sh)
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

PY="${PYTHON:-python3}"
if ! ls wheelhouse/*.whl >/dev/null 2>&1; then
  echo "[install] ERROR: wheelhouse/ is empty. Build the bundle with scripts/build_offline_bundle.sh" >&2
  echo "          on a machine with Internet access, or install from requirements.txt if online." >&2
  exit 1
fi
"$PY" -m venv .venv
.venv/bin/python -m pip install --no-index --find-links wheelhouse --upgrade pip setuptools wheel >/dev/null
.venv/bin/python -m pip install --no-index --find-links wheelhouse -r requirements.lock.txt
.venv/bin/python -m src.launcher --check --pipeline-only || true
echo "[install] done. Start the system with: ./run.sh"
