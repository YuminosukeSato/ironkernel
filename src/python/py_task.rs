use pyo3::prelude::*;

use crate::error::ParsecError;
use crate::runtime::task::{TaskHandle, TaskResult};

use super::py_buffer::PyBuffer;

/// Python-visible task handle.
#[pyclass(name = "TaskHandle")]
#[derive(Clone)]
pub struct PyTaskHandle {
    pub(crate) inner: TaskHandle,
}

impl PyTaskHandle {
    pub fn new(inner: TaskHandle) -> Self {
        PyTaskHandle { inner }
    }
}

fn parsec_err_to_py(e: ParsecError) -> PyErr {
    match e {
        ParsecError::TypeError(msg) => PyErr::new::<pyo3::exceptions::PyTypeError, _>(msg),
        ParsecError::ShapeError(msg) => PyErr::new::<pyo3::exceptions::PyValueError, _>(msg),
        ParsecError::ArgError(msg) => PyErr::new::<pyo3::exceptions::PyValueError, _>(msg),
        ParsecError::EmptyCollection(msg) => PyErr::new::<pyo3::exceptions::PyValueError, _>(msg),
        ParsecError::Cancelled => {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("task cancelled")
        }
        ParsecError::ChannelClosed => {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("channel closed")
        }
        ParsecError::Internal(msg) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg),
    }
}

#[pymethods]
impl PyTaskHandle {
    /// Block until completion and return result.
    fn result(&self, py: Python<'_>) -> PyResult<PyBuffer> {
        let task_result = py
            .allow_threads(|| self.inner.result())
            .map_err(parsec_err_to_py)?;
        match task_result {
            TaskResult::Buffer(buf) => Ok(PyBuffer::new(buf)),
            TaskResult::Scalar(v) => {
                use crate::buffer::inner::Buffer;
                Ok(PyBuffer::new(Buffer::from_f64_vec(vec![v])))
            }
        }
    }

    /// Check if the task is complete.
    fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    /// Cancel the task.
    fn cancel(&self) -> bool {
        self.inner.cancel()
    }

    fn __repr__(&self) -> String {
        format!("TaskHandle(state={:?})", self.inner.state())
    }
}
