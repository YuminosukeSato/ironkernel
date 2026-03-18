# parsec - Project Rules

## Build
- `source .venv/bin/activate`
- `maturin develop --release` for release build
- `maturin develop` for debug build

## Test
- Rust: `cargo test`
- Python: `source .venv/bin/activate && pytest tests/python/ -v`
- All: `cargo test && source .venv/bin/activate && maturin develop && pytest tests/python/ -v`

## Lint
- Rust: `cargo clippy -- -D warnings && cargo fmt --check`
- Python: `source .venv/bin/activate && ruff check python/ tests/`
- Type: `source .venv/bin/activate && mypy python/parsec/ --strict`

## Architecture Rules
- PyO3 dependency confined to src/python/
- ir/, buffer/, runtime/, channel/ must NOT import pyo3
- unsafe requires SAFETY comment
- pub(crate) for inter-module visibility, re-export in lib.rs

## Two-Layer Principle
- Python = DSL (describes WHAT to execute)
- Rust = execution engine (controls HOW to parallelize)
- hot path never touches Python objects
- channels carry Buffer handles, not data payloads

## GIL Control
- Release GIL during Rust computation (py.allow_threads / Python.detach)
- Hold GIL only for Python<->Buffer conversion
- free-threaded build: all Rust types are Send+Sync

## Commit Convention
- Conventional Commits: feat(scope)/fix(scope)/test(scope)
- scope: ir, buffer, runtime, channel, python, docs, ci, bench
- Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

## TDD Required
Test -> Implement -> Refactor
