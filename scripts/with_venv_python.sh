#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "expected virtualenv python at ${VENV_PYTHON}; run uv sync --frozen --dev first" >&2
  exit 1
fi

export PYO3_PYTHON="${VENV_PYTHON}"
export PYTHON_SYS_EXECUTABLE="${VENV_PYTHON}"

exec "$@"
