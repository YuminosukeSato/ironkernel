# PyO3 Binding Layer Rules

src/python/ is the sole bridge between Rust core and Python.

## GIL Control (Critical)
- Release GIL via py.allow_threads() before Rust computation
- Hold GIL only during Python<->Buffer conversion
- Free-threaded build: all Rust types must be Send+Sync

## Patterns
- #[pyclass]: wrapper around Rust core type (e.g. PyBuffer -> Buffer)
- #[pymethods]: methods callable from Python
- Data passing: via Buffer handle (Arc<BufferInner>)
- numpy conversion: PyReadonlyArrayDyn -> Storage::Owned

## Error Conversion
- Unified ParsecError -> PyErr conversion
- Map to appropriate Python exceptions (ValueError/TypeError)

## Naming
- Files: py_{core_module}.rs
- Classes: Py{CoreType} (e.g. PyBuffer, PyExpr)
