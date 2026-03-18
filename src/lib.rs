use pyo3::prelude::*;

pub mod buffer;
pub mod error;

/// parsec: A Python parallel compute library backed by a Rust execution engine.
#[pymodule]
fn _parsec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
