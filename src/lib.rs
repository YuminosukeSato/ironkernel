use pyo3::prelude::*;

/// parsec: A Python parallel compute library backed by a Rust execution engine.
#[pymodule]
fn _parsec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
