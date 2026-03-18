# ironkernel

Two-layer architecture: Rust (execution engine) + Python (PyO3 DSL)

## Architecture
- Python = DSL (WHAT) / Rust = engine (HOW)
- Hot path must never touch Python objects
- Channels carry Buffer handles, not data payloads
- Modules: ir/, buffer/, runtime/, channel/ (pyo3-free) + python/ (PyO3 boundary)

## Build & Test
- Rust: `cargo test`
- Python: `uv run maturin develop && uv run pytest tests/python/ -v`
- All: `cargo test && uv run maturin develop && uv run pytest tests/python/ -v`
- Release: `uv run maturin develop --release`

## Lint
- Rust: `cargo clippy -- -D warnings && cargo fmt --check`
- Python: `uv run ruff check python/ tests/ && uv run mypy python/ironkernel/ --strict`

## NEVER
- Change Rust Edition 2021
- Modify existing public API signatures without approval
- Import pyo3 outside src/python/

## IMPORTANT
- TDD required: Test -> Implement -> Refactor
- unsafe requires SAFETY comment
- pub(crate) for inter-module visibility, re-export only in lib.rs
## Commit Convention (MUST follow)
Follow Conventional Commits 1.0.0: https://www.conventionalcommits.org/en/v1.0.0/

Format: `<type>(<scope>): <description>`

Types: feat, fix, test, refactor, docs, ci, perf, chore, build
Scopes: ir, buffer, runtime, channel, python, docs, ci, bench

Breaking changes MUST include `!` after scope: `feat(ir)!: rename Expr variants`
Body and footer follow the spec when needed.

Examples:
- `feat(buffer): add f16 dtype support`
- `fix(python): release GIL before rayon parallel`
- `test(channel): add select timeout boundary tests`
- `refactor(ir)!: replace Opcode enum with trait-based dispatch`

## Dependencies (knowledge cutoff reference)
- pyo3 = "0.23", numpy = "0.23", rayon = "1.10", crossbeam-channel = "0.5"
- Dev: proptest = "1.5"
- Python: uv + maturin build system, Python 3.9+

## Plan Workflow
In Plan mode, output plans to z-ai/ and ensure quality via 3-stage pipeline:
1. ironkernel-planner: Requirements analysis -> generate z-ai/plan.md
2. ironkernel-architect: Design review -> revise z-ai/plan.md (architecture consistency)
3. ironkernel-e2e (Mode A): E2E test plan -> append test details to z-ai/plan.md
z-ai/ is gitignored. Proceed to implementation only after plan approval.
YOU MUST: Execute all 3 stages sequentially in Plan mode. No skipping.

## currentDate
Today's date is 2026-03-18.
