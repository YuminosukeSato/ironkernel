#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ITERATIONS="${1:-200}"

cd "${ROOT_DIR}"

for ((i = 1; i <= ITERATIONS; i++)); do
  echo "[stress] iteration ${i}/${ITERATIONS}"
  cargo test -q
  uv run pytest tests/python/test_goal_code_v1.py tests/python/test_readme_examples.py -q
done
