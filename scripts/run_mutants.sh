#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

cargo mutants \
  --baseline run \
  --check \
  --jobs "${MUTANTS_JOBS:-2}" \
  --timeout "${MUTANTS_TIMEOUT:-300}" \
  --minimum-test-timeout "${MUTANTS_MIN_TEST_TIMEOUT:-20}" \
  --file 'src/runtime/**' \
  --file 'src/channel/**' \
  --file 'src/python/**'
