use pyo3::prelude::*;

pub mod buffer;
pub mod error;
pub mod ir;
pub mod python;
pub mod runtime;

/// parsec: A Python parallel compute library backed by a Rust execution engine.
#[pymodule]
fn _parsec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<python::py_expr::PyExpr>()?;
    m.add_class::<python::py_buffer::PyBuffer>()?;
    m.add_class::<python::py_task::PyTaskHandle>()?;
    m.add_class::<python::py_kernel::PyKernelSpec>()?;
    m.add_class::<python::py_kernel::PyMapSpec>()?;
    m.add_class::<python::py_kernel::PyReduceSpec>()?;
    m.add_class::<python::py_kernel::PyKernelModule>()?;
    m.add_class::<python::py_runtime::PyRuntimeModule>()?;
    Ok(())
}
