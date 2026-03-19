use pyo3::prelude::*;

use crate::error::ParsecError;
#[cfg(test)]
use crate::error::ParsecResult;
use crate::runtime::task::{TaskHandle, TaskResult};

use super::py_buffer::PyBuffer;
use super::py_callable::CallableTask;

#[derive(Clone)]
pub(crate) enum TaskInner {
    Native(TaskHandle),
    Callable(CallableTask),
}

#[cfg(test)]
impl TaskInner {
    pub(crate) fn is_done(&self) -> bool {
        match self {
            Self::Native(handle) => handle.is_done(),
            Self::Callable(handle) => handle.is_done(),
        }
    }

    pub(crate) fn result(&self) -> ParsecResult<TaskResult> {
        match self {
            Self::Native(handle) => handle.result(),
            Self::Callable(_) => Err(ParsecError::Internal(
                "callable task result requires Python object access".into(),
            )),
        }
    }
}

/// Python-visible task handle.
#[pyclass(name = "TaskHandle")]
#[derive(Clone)]
pub struct PyTaskHandle {
    pub(crate) inner: TaskInner,
}

impl PyTaskHandle {
    pub fn new(inner: TaskHandle) -> Self {
        PyTaskHandle {
            inner: TaskInner::Native(inner),
        }
    }

    pub(crate) fn from_callable(inner: CallableTask) -> Self {
        PyTaskHandle {
            inner: TaskInner::Callable(inner),
        }
    }

    pub(crate) fn result_object(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            TaskInner::Native(handle) => {
                let task_result = py
                    .allow_threads(|| handle.result())
                    .map_err(parsec_err_to_py)?;
                match task_result {
                    TaskResult::Buffer(buf) => {
                        Ok(PyBuffer::new(buf).into_pyobject(py)?.into_any().unbind())
                    }
                    TaskResult::Scalar(v) => {
                        use crate::buffer::inner::Buffer;
                        Ok(PyBuffer::new(Buffer::from_f64_vec(vec![v]))
                            .into_pyobject(py)?
                            .into_any()
                            .unbind())
                    }
                }
            }
            TaskInner::Callable(handle) => handle.result(py),
        }
    }

    pub(crate) fn done(&self) -> bool {
        match &self.inner {
            TaskInner::Native(handle) => handle.is_done(),
            TaskInner::Callable(handle) => handle.is_done(),
        }
    }

    pub(crate) fn cancel_inner(&self) -> bool {
        match &self.inner {
            TaskInner::Native(handle) => handle.cancel(),
            TaskInner::Callable(handle) => handle.cancel(),
        }
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
    fn result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.result_object(py)
    }

    /// Check if the task is complete.
    fn is_done(&self) -> bool {
        self.done()
    }

    /// Cancel the task.
    fn cancel(&self) -> bool {
        self.cancel_inner()
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            TaskInner::Native(handle) => format!("TaskHandle(state={:?})", handle.state()),
            TaskInner::Callable(handle) => format!("TaskHandle(state={})", handle.state_label()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::python::py_callable::CallableTask;
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
            let result = result.extract::<PyBuffer>(py).unwrap();

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
            let result = result.extract::<PyBuffer>(py).unwrap();
            assert_eq!(result.inner.as_f64_slice(), &[1.0, 2.0, 3.0]);
        });
    }

    #[test]
    fn task_result_returns_python_object_for_callable_why_callable_go_must_not_force_buffer_wrapping(
    ) {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let callable = CallableTask::new();
            let value = "callable".into_pyobject(py).unwrap().unbind().into_any();
            callable.complete(value);

            let result = PyTaskHandle::from_callable(callable).result(py).unwrap();

            assert_eq!(result.bind(py).extract::<String>().unwrap(), "callable");
        });
    }

    #[test]
    fn task_result_maps_callable_failure_to_runtime_error_why_python_callable_exceptions_must_reach_callers(
    ) {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let callable = CallableTask::new();
            callable.fail("callable boom".to_string());

            let err = PyTaskHandle::from_callable(callable)
                .result(py)
                .unwrap_err();

            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(err.to_string().contains("callable boom"));
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

    #[test]
    fn task_inner_callable_result_requires_python_access_why_test_only_rust_bridge_must_not_fake_python_objects(
    ) {
        let inner = TaskInner::Callable(CallableTask::new());

        assert_eq!(
            inner.result().unwrap_err(),
            ParsecError::Internal("callable task result requires Python object access".into())
        );
    }

    #[test]
    fn task_inner_callable_is_done_tracks_terminal_state_why_test_only_rust_bridge_must_match_callable_handles(
    ) {
        let callable = CallableTask::new();
        let inner = TaskInner::Callable(callable.clone());

        assert!(!inner.is_done());
        assert!(callable.cancel());
        assert!(inner.is_done());
    }

    #[test]
    fn pytaskhandle_clone_shares_callable_state_why_python_task_aliases_must_observe_same_completion(
    ) {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let callable = CallableTask::new();
            let py_handle = PyTaskHandle::from_callable(callable.clone());
            let cloned = py_handle.clone();
            let value = 11_i64.into_pyobject(py).unwrap().unbind().into_any();
            callable.complete(value);

            assert_eq!(
                cloned
                    .result(py)
                    .unwrap()
                    .bind(py)
                    .extract::<i64>()
                    .unwrap(),
                11
            );
        });
    }

    #[test]
    fn pytaskhandle_repr_for_callable_uses_callable_state_label_why_debug_output_must_distinguish_python_task_path(
    ) {
        let callable = CallableTask::new();
        let py_handle = PyTaskHandle::from_callable(callable.clone());

        assert!(py_handle.__repr__().contains("Pending"));
        assert!(callable.cancel());
        assert!(py_handle.__repr__().contains("Cancelled"));
    }
}
