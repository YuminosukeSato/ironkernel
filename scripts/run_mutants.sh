#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"
rm -rf mutants.out

bash scripts/with_venv_python.sh cargo mutants \
  --baseline run \
  --jobs "${MUTANTS_JOBS:-2}" \
  --timeout "${MUTANTS_TIMEOUT:-120}" \
  --minimum-test-timeout "${MUTANTS_MIN_TEST_TIMEOUT:-10}" \
  -C --locked \
  --test-tool cargo \
  --cargo-test-arg mutation_guard \
  --cargo-test-arg '--' \
  --cargo-test-arg '--test-threads=1' \
  --file 'src/runtime/delivery.rs' \
  --file 'src/channel/select.rs' \
  --file 'src/python/py_runtime.rs'

uv run python scripts/check_mutation_results.py mutants.out/outcomes.json scripts/mutation-baseline.json
