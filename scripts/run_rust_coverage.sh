#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

eval "$(cargo llvm-cov show-env --sh)"

cargo llvm-cov clean --workspace
cargo test --workspace
uv run maturin develop
uv run pytest tests/python/ -q
cargo llvm-cov report --json --summary-only --output-path coverage-rust.json
cargo llvm-cov report --text --show-missing-lines --output-path coverage-rust.txt
cargo llvm-cov report --cobertura --output-path coverage-rust.xml
uv run python scripts/check_rust_coverage.py coverage-rust.txt
