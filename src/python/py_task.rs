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

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};

    #[test]
    fn task_result_maps_parsec_errors_to_python_exceptions_why_boundary_errors_must_stay_stable() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let type_failed = TaskHandle::new();
            type_failed.fail(ParsecError::TypeError("wrong type".into()));
            let result = PyTaskHandle::new(type_failed).result(py);
            assert!(result.is_err());
            let err = result.err().unwrap();
            assert!(err.is_instance_of::<PyTypeError>(py));

            let shape_failed = TaskHandle::new();
            shape_failed.fail(ParsecError::ShapeError("shape mismatch".into()));
            let result = PyTaskHandle::new(shape_failed).result(py);
            assert!(result.is_err());
            let err = result.err().unwrap();
            assert!(err.is_instance_of::<PyValueError>(py));

            let closed_failed = TaskHandle::new();
            closed_failed.fail(ParsecError::ChannelClosed);
            let result = PyTaskHandle::new(closed_failed).result(py);
            assert!(result.is_err());
            let err = result.err().unwrap();
            assert!(err.is_instance_of::<PyRuntimeError>(py));
        });
    }

    #[test]
    fn task_result_wraps_scalar_as_single_element_buffer_why_reduce_contract_must_stay_uniform() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let handle = TaskHandle::new();
            handle.complete(TaskResult::Scalar(49.5));

            let result = PyTaskHandle::new(handle).result(py).unwrap();

            assert_eq!(result.inner.as_f64_slice(), &[49.5]);
            assert_eq!(result.inner.len(), 1);
        });
    }

    #[test]
    fn task_result_wraps_buffer() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let handle = TaskHandle::new();
            let buf = crate::buffer::inner::Buffer::from_f64_vec(vec![1.0, 2.0, 3.0]);
            handle.complete(TaskResult::Buffer(buf));

            let result = PyTaskHandle::new(handle).result(py).unwrap();
            assert_eq!(result.inner.as_f64_slice(), &[1.0, 2.0, 3.0]);
        });
    }

    #[test]
    fn parsec_err_to_py_arg_error() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err = parsec_err_to_py(ParsecError::ArgError("missing x".into()));
            assert!(err.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn parsec_err_to_py_empty_collection() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err = parsec_err_to_py(ParsecError::EmptyCollection("empty buf".into()));
            assert!(err.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn parsec_err_to_py_cancelled() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err = parsec_err_to_py(ParsecError::Cancelled);
            assert!(err.is_instance_of::<PyRuntimeError>(py));
        });
    }

    #[test]
    fn parsec_err_to_py_internal() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err = parsec_err_to_py(ParsecError::Internal("boom".into()));
            assert!(err.is_instance_of::<PyRuntimeError>(py));
        });
    }

    #[test]
    fn pytaskhandle_is_done() {
        let handle = TaskHandle::new();
        let py_handle = PyTaskHandle::new(handle.clone());
        assert!(!py_handle.is_done());
        handle.complete(TaskResult::Scalar(1.0));
        assert!(py_handle.is_done());
    }

    #[test]
    fn pytaskhandle_cancel() {
        let handle = TaskHandle::new();
        let py_handle = PyTaskHandle::new(handle);
        assert!(py_handle.cancel());
        assert!(py_handle.is_done());
    }

    #[test]
    fn pytaskhandle_repr() {
        let handle = TaskHandle::new();
        let py_handle = PyTaskHandle::new(handle);
        let repr = py_handle.__repr__();
        assert!(repr.contains("Created"));
    }

    #[test]
    fn pytaskhandle_clone() {
        let handle = TaskHandle::new();
        let py_handle = PyTaskHandle::new(handle);
        let cloned = py_handle.clone();
        assert!(!cloned.is_done());
    }
}
