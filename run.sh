#!/usr/bin/env bash
# TRACE-X one-command launcher (offline).
#
#   ./run.sh                     validate env -> reuse or run the pipeline -> start dashboard
#   ./run.sh --input <dataset>   use another dataset (file or directory)
#   ./run.sh --dashboard-only    open the latest completed run without touching the pipeline
#   ./run.sh --check             validate the environment only
#   ./run.sh --help              all options
#
# Dashboard: http://localhost:8501
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "[run.sh] ERROR: no Python found. Install Python 3.10+ or create .venv (scripts/install_offline.sh)." >&2
  exit 1
fi

exec "$PY" -m src.launcher "$@"
