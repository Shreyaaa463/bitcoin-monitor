#!/usr/bin/env bash
# Build a self-contained offline bundle ON A MACHINE WITH INTERNET ACCESS:
#   - downloads every wheel from requirements.lock.txt for the TARGET platform/Python
#     into wheelhouse/ (no compilers or Internet needed on the target)
#   - packs the project + wheelhouse into dist/bitcoin-monitor-offline-<version>.tar.gz
#
# Usage:
#   scripts/build_offline_bundle.sh                      # target: this machine's Python
#   PY_VERSION=3.12 PLATFORM=manylinux2014_x86_64 scripts/build_offline_bundle.sh
#   INCLUDE_DATA=1 scripts/build_offline_bundle.sh       # also ship data/raw/ datasets
#
# On the offline target:  tar xzf <bundle>.tar.gz && cd bitcoin-monitor
#                         scripts/install_offline.sh && ./run.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

PY="${PYTHON:-python3}"
PY_VERSION="${PY_VERSION:-$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')}"
PLATFORM="${PLATFORM:-manylinux2014_x86_64}"
VERSION="$("$PY" -c 'import re,pathlib; print(re.search(r"PIPELINE_VERSION = \"([^\"]+)\"", pathlib.Path("src/version.py").read_text()).group(1))')"

echo "[bundle] downloading wheels for Python ${PY_VERSION} / ${PLATFORM} ..."
rm -rf wheelhouse && mkdir -p wheelhouse
"$PY" -m pip download --only-binary=:all: --dest wheelhouse \
    --python-version "${PY_VERSION}" --platform "${PLATFORM}" --platform manylinux_2_17_x86_64 \
    --platform manylinux_2_28_x86_64 --platform any \
    -r requirements.lock.txt
# pip/setuptools/wheel so a fresh venv can install from the wheelhouse without Internet.
"$PY" -m pip download --only-binary=:all: --dest wheelhouse --python-version "${PY_VERSION}" \
    --platform "${PLATFORM}" --platform any pip setuptools wheel

mkdir -p dist
OUT="dist/bitcoin-monitor-offline-${VERSION}-py${PY_VERSION}.tar.gz"
EXCLUDES=(--exclude='./.venv' --exclude='./runs' --exclude='./dist' --exclude='./benchmarks'
          --exclude='*/__pycache__' --exclude='./.pytest_cache' --exclude='./data/processed/ingest_cache')
if [[ "${INCLUDE_DATA:-0}" != "1" ]]; then
  EXCLUDES+=(--exclude='./data/raw/*.json' --exclude='./data/raw/*.jsonl' --exclude='./data/raw/*.csv' --exclude='./data/raw/*.xml')
fi
tar czf "${OUT}" "${EXCLUDES[@]}" --transform 's,^\.,bitcoin-monitor,' .
echo "[bundle] wrote ${OUT} ($(du -h "${OUT}" | cut -f1))"
echo "[bundle] SHA-256: $(sha256sum "${OUT}" | cut -d' ' -f1)"
