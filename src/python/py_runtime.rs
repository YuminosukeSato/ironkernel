use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::runtime::task::{TaskHandle, TaskResult};

use super::py_buffer::{asarray_from_numpy, PyBuffer};
use super::py_channel::PyChannel;
use super::py_kernel::{execute_map, execute_reduce, PyMapSpec, PyReduceSpec};
use super::py_task::PyTaskHandle;

/// The runtime module exposed to Python.
#[pyclass(name = "_RuntimeModule")]
pub struct PyRuntimeModule;

#[pymethods]
impl PyRuntimeModule {
    #[new]
    fn new() -> Self {
        PyRuntimeModule
    }

    /// Create a Buffer from a numpy array.
    fn asarray(&self, array: PyReadonlyArray1<f64>) -> PyBuffer {
        PyBuffer::new(asarray_from_numpy(array))
    }

    /// Submit a computation and return a TaskHandle.
    /// Accepts either a MapSpec (elementwise) or a ReduceSpec.
    fn go(&self, py: Python<'_>, spec: &Bound<'_, PyAny>) -> PyResult<PyTaskHandle> {
        if let Ok(map_spec) = spec.extract::<PyMapSpec>() {
            let handle = TaskHandle::new();
            let h2 = handle.clone();

            // Release GIL, run on Rayon
            py.allow_threads(|| {
                h2.set_running();
                match execute_map(&map_spec) {
                    Ok(buf) => h2.complete(TaskResult::Buffer(buf)),
                    Err(e) => h2.fail(e),
                }
            });

            Ok(PyTaskHandle::new(handle))
        } else if let Ok(reduce_spec) = spec.extract::<PyReduceSpec>() {
            let handle = TaskHandle::new();
            let h2 = handle.clone();

            py.allow_threads(|| {
                h2.set_running();
                match execute_reduce(&reduce_spec) {
                    Ok(v) => h2.complete(TaskResult::Scalar(v)),
                    Err(e) => h2.fail(e),
                }
            });

            Ok(PyTaskHandle::new(handle))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "go() expects a MapSpec or ReduceSpec",
            ))
        }
    }

    /// Create a bounded channel.
    fn chan(&self, capacity: usize) -> PyChannel {
        PyChannel::new(capacity)
    }
}
