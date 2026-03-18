# Python DSL Layer Rules

python/ironkernel/ is the user-facing DSL interface.

## __init__.py
- Re-export from _parsec (Rust binary) only
- No Python logic

## Types
- mypy --strict compliant
- py.typed marker file

## Tests
- Place pytest tests in tests/python/
- Shared fixtures in conftest.py
