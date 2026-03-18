# Contributing to ironkernel

Thank you for your interest in contributing to ironkernel.

## Getting Started

### Prerequisites

- Rust 1.70+
- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Setup

```bash
git clone https://github.com/YuminosukeSato/ironkernel.git
cd ironkernel
uv sync
uv run maturin develop
```

### Verify

```bash
cargo test && uv run maturin develop && uv run pytest tests/python/ -v
```

## Architecture Rules

ironkernel uses a strict two-layer architecture. Understanding these rules is essential before contributing.

### Layer separation

```
src/ir/          Pure Rust. Expression tree, kernel specs, compiler.
src/buffer/      Pure Rust. Buffer storage, DType.
src/runtime/     Pure Rust. Rayon pool, task handles.
src/channel/     Pure Rust. Bounded channels, select.
src/python/      PyO3 bindings. The ONLY place that imports pyo3.
python/ironkernel/   Python re-exports. No logic here.
```

- `ir/`, `buffer/`, `runtime/`, `channel/` must NEVER import `pyo3` or `numpy` crates.
- All data passes between layers as `Buffer` handles (`Arc<BufferInner>`), not raw data.
- GIL must be released (`py.allow_threads()`) before calling any Rust computation.
- All `#[pyclass]` types must be `Send + Sync`.

### Visibility

- Use `pub(crate)` by default for inter-module items.
- Only `lib.rs` has public re-exports.

### Error handling

- All errors go through `ParsecError` enum in `src/error.rs`.
- Use `ParsecResult<T>` as the return type.
- No `unwrap()` or `expect()` in production code.

## Development Workflow

### TDD (required)

We follow strict Test-Driven Development:

1. Write a failing test
2. Confirm it fails for the expected reason
3. Write the minimum code to make it pass
4. Refactor while keeping tests green

### Adding a new feature

Here is the typical flow for adding a feature that spans all layers:

#### 1. Rust core (src/)

Add your core logic to the appropriate module. Write tests first.

```rust
// src/ir/your_module.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_feature() {
        // test goes here
    }
}
```

Run: `cargo test`

#### 2. PyO3 binding (src/python/)

Create a wrapper in `src/python/py_your_module.rs`.

Naming conventions:
- File: `py_{module}.rs`
- Class: `Py{Type}` (e.g., `PyBuffer`, `PyExpr`)

Register in `src/lib.rs`:
```rust
m.add_class::<python::py_your_module::PyYourType>()?;
```

#### 3. Python re-export (python/ironkernel/__init__.py)

```python
from ironkernel._ironkernel import YourType
```

#### 4. Python test (tests/python/)

```python
def test_your_feature():
    # test goes here
```

Run: `uv run maturin develop && uv run pytest tests/python/ -v`

### Running all checks

Before submitting a PR, run all checks:

```bash
# Rust
cargo fmt --check
cargo clippy -- -D warnings
cargo test

# Python
uv run maturin develop
uv run pytest tests/python/ -v
uv run ruff check python/ tests/
uv run mypy python/ironkernel/ --strict
```

## Commit Convention

We follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Use for |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `test` | Adding or updating tests |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `docs` | Documentation only |
| `perf` | Performance improvement |
| `ci` | CI/CD changes |
| `chore` | Maintenance tasks |
| `build` | Build system changes |

### Scopes

| Scope | Module |
|-------|--------|
| `ir` | `src/ir/` |
| `buffer` | `src/buffer/` |
| `runtime` | `src/runtime/` |
| `channel` | `src/channel/` |
| `python` | `src/python/` and `python/` |
| `docs` | Documentation |
| `ci` | CI/CD configuration |
| `bench` | Benchmarks |

### Examples

```
feat(buffer): add f16 dtype support
fix(python): release GIL before rayon parallel
test(channel): add select timeout boundary tests
refactor(ir)!: replace Opcode enum with trait-based dispatch
```

Breaking changes MUST include `!` after the scope:

```
feat(ir)!: rename Expr variants

BREAKING CHANGE: Expr::Var renamed to Expr::ArgRef
```

### What makes a good commit

- One logical change per commit
- The message explains WHY, not WHAT (the diff shows what)
- Tests included in the same commit as the feature/fix they cover

## Pull Requests

### PR checklist

- [ ] All tests pass (`cargo test && pytest tests/python/ -v`)
- [ ] Lint clean (`cargo clippy -- -D warnings && cargo fmt --check`)
- [ ] Python lint clean (`ruff check python/ tests/ && mypy python/ironkernel/ --strict`)
- [ ] No `pyo3` imports outside `src/python/`
- [ ] GIL released during Rust computation
- [ ] `unsafe` blocks have `SAFETY` comments
- [ ] `pub(crate)` used (not `pub`) except in `lib.rs`
- [ ] Commit messages follow Conventional Commits

### PR title

Follow the same format as commit messages:

```
feat(buffer): add f16 dtype support
```

## Reporting Issues

When reporting a bug, please include:

- ironkernel version (`uv run python -c "import ironkernel; print(ironkernel.__version__)"`)
- Python version (`uv run python --version`)
- Rust version (`rustc --version`)
- OS and architecture
- Minimal reproduction code

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
