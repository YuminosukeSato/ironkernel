#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"

cd "${ROOT_DIR}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "expected virtualenv python at ${VENV_PYTHON}; run uv sync --frozen --dev first" >&2
  exit 1
fi

export PYO3_PYTHON="${VENV_PYTHON}"
export PYTHON_SYS_EXECUTABLE="${VENV_PYTHON}"

eval "$(cargo llvm-cov show-env --sh)"

cargo llvm-cov clean --workspace
bash scripts/with_venv_python.sh cargo test --locked --workspace
uv run maturin develop
uv run pytest tests/python/ -q
cargo llvm-cov report --json --summary-only --output-path coverage-rust.json
cargo llvm-cov report --text --show-missing-lines --output-path coverage-rust.txt
cargo llvm-cov report --cobertura --output-path coverage-rust.xml
uv run python scripts/check_rust_coverage.py coverage-rust.txt
