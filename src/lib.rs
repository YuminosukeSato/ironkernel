use pyo3::prelude::*;

pub mod buffer;
pub mod channel;
pub mod error;
pub mod ir;
pub mod python;
pub mod runtime;

/// ironkernel: A Python parallel compute library backed by a Rust execution engine.
#[pymodule]
fn _ironkernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<python::py_expr::PyExpr>()?;
    m.add_class::<python::py_buffer::PyBuffer>()?;
    m.add_class::<python::py_task::PyTaskHandle>()?;
    m.add_class::<python::py_kernel::PyKernelSpec>()?;
    m.add_class::<python::py_kernel::PyMapSpec>()?;
    m.add_class::<python::py_kernel::PyReduceSpec>()?;
    m.add_class::<python::py_kernel::PyKernelModule>()?;
    m.add_class::<python::py_runtime::PyRuntimeModule>()?;
    m.add_class::<python::py_channel::PyChannel>()?;
    m.add_class::<python::py_channel::PyRecvCase>()?;
    m.add_function(wrap_pyfunction!(python::py_channel::py_select, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_registers_python_exports_why_import_surface_regressions_must_fail_in_rust_tests() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let module = PyModule::new(py, "_ironkernel_test").unwrap();
            _ironkernel(&module).unwrap();

            assert_eq!(
                module
                    .getattr("__version__")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                env!("CARGO_PKG_VERSION")
            );
            for name in [
                "Expr",
                "Buffer",
                "TaskHandle",
                "KernelSpec",
                "MapSpec",
                "ReduceSpec",
                "_KernelModule",
                "_RuntimeModule",
                "Channel",
                "RecvCase",
                "py_select",
            ] {
                assert!(module.getattr(name).is_ok(), "missing export: {name}");
            }
        });
    }
}
