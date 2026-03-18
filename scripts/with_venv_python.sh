#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
VENV_BIN="${VENV_DIR}/bin"
VENV_PYTHON="${VENV_BIN}/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "expected virtualenv python at ${VENV_PYTHON}; run uv sync --frozen --dev first" >&2
  exit 1
fi

export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_BIN}${PATH:+:${PATH}}"
export PYO3_PYTHON="${VENV_PYTHON}"
export PYTHON_SYS_EXECUTABLE="${VENV_PYTHON}"

# Do NOT set PYTHONPATH — the venv python already knows its own site-packages.
# Setting PYTHONPATH can cause numpy source-directory import errors on CI.
unset PYTHONPATH

exec "$@"
