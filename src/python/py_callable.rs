use std::sync::{Arc, Condvar, Mutex};

use pyo3::prelude::*;

#[derive(Clone)]
pub(crate) struct CallableTask {
    inner: Arc<(Mutex<CallableState>, Condvar)>,
}

pub(crate) enum CallableState {
    Pending,
    Completed(Py<PyAny>),
    Failed(String),
    Cancelled,
}

impl CallableState {
    fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed(_) | Self::Failed(_) | Self::Cancelled)
    }
}

impl CallableTask {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new((Mutex::new(CallableState::Pending), Condvar::new())),
        }
    }

    pub(crate) fn complete(&self, value: Py<PyAny>) {
        let (mutex, condvar) = &*self.inner;
        let mut state = mutex.lock().unwrap();
        if state.is_terminal() {
            return;
        }
        *state = CallableState::Completed(value);
        condvar.notify_all();
    }

    pub(crate) fn fail(&self, message: String) {
        let (mutex, condvar) = &*self.inner;
        let mut state = mutex.lock().unwrap();
        if state.is_terminal() {
            return;
        }
        *state = CallableState::Failed(message);
        condvar.notify_all();
    }

    pub(crate) fn cancel(&self) -> bool {
        let (mutex, condvar) = &*self.inner;
        let mut state = mutex.lock().unwrap();
        if state.is_terminal() {
            return false;
        }
        *state = CallableState::Cancelled;
        condvar.notify_all();
        true
    }

    pub(crate) fn is_done(&self) -> bool {
        self.inner.0.lock().unwrap().is_terminal()
    }

    pub(crate) fn state_label(&self) -> &'static str {
        match &*self.inner.0.lock().unwrap() {
            CallableState::Pending => "Pending",
            CallableState::Completed(_) => "Completed",
            CallableState::Failed(_) => "Failed",
            CallableState::Cancelled => "Cancelled",
        }
    }

    pub(crate) fn result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        py.allow_threads(|| {
            let (mutex, condvar) = &*self.inner;
            let mut state = mutex.lock().unwrap();
            while !state.is_terminal() {
                state = condvar.wait(state).unwrap();
            }
        });

        let state = self.inner.0.lock().unwrap();
        match &*state {
            CallableState::Completed(value) => Ok(value.clone_ref(py)),
            CallableState::Failed(message) => Err(
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(message.clone()),
            ),
            CallableState::Cancelled => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "task cancelled",
            )),
            CallableState::Pending => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyRuntimeError;
    use std::sync::Barrier;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn new_is_pending_why_async_callable_handle_must_start_in_non_terminal_state() {
        let task = CallableTask::new();

        assert!(!task.is_done());
    }

    #[test]
    fn complete_stores_result_why_callable_result_must_roundtrip_original_python_object() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let task = CallableTask::new();
            let value = "done".into_pyobject(py).unwrap().unbind().into_any();

            task.complete(value);
            let result = task.result(py).unwrap();

            assert_eq!(result.bind(py).extract::<String>().unwrap(), "done");
            assert!(task.is_done());
        });
    }

    #[test]
    fn fail_returns_error_why_callable_exception_text_must_surface_at_wait_boundary() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let task = CallableTask::new();

            task.fail("boom".to_string());
            let err = task.result(py).unwrap_err();

            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(err.to_string().contains("boom"));
            assert!(task.is_done());
        });
    }

    #[test]
    fn cancel_from_pending_returns_true_why_pre_start_cancellation_must_be_observable() {
        let task = CallableTask::new();

        assert!(task.cancel());
        assert!(task.is_done());
    }

    #[test]
    fn cancel_from_completed_returns_false_why_terminal_callable_tasks_must_not_reopen() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let task = CallableTask::new();
            let value = 7_i64.into_pyobject(py).unwrap().unbind().into_any();

            task.complete(value);

            assert!(!task.cancel());
        });
    }

    #[test]
    fn result_blocks_until_complete_why_waiters_must_sleep_until_background_callable_finishes() {
        pyo3::prepare_freethreaded_python();

        let task = CallableTask::new();
        let background = task.clone();

        let join = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            Python::with_gil(|py| {
                let value = 42_i64.into_pyobject(py).unwrap().unbind().into_any();
                background.complete(value);
            });
        });

        Python::with_gil(|py| {
            let result = task.result(py).unwrap();
            assert_eq!(result.bind(py).extract::<i64>().unwrap(), 42);
        });

        join.join().unwrap();
    }

    #[test]
    fn result_blocks_until_fail_why_waiters_must_observe_background_callable_errors() {
        pyo3::prepare_freethreaded_python();

        let task = CallableTask::new();
        let background = task.clone();

        let join = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            background.fail("bad callable".to_string());
        });

        Python::with_gil(|py| {
            let err = task.result(py).unwrap_err();
            assert!(err.to_string().contains("bad callable"));
        });

        join.join().unwrap();
    }

    #[test]
    fn result_blocks_until_cancel_why_waiters_must_observe_cooperative_cancellation() {
        pyo3::prepare_freethreaded_python();

        let task = CallableTask::new();
        let background = task.clone();

        let join = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            assert!(background.cancel());
        });

        Python::with_gil(|py| {
            let err = task.result(py).unwrap_err();
            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(err.to_string().contains("cancelled"));
        });

        join.join().unwrap();
    }

    #[test]
    fn clone_shares_state_why_python_task_clones_must_reference_the_same_async_result() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let task = CallableTask::new();
            let cloned = task.clone();
            let value = 3_i64.into_pyobject(py).unwrap().unbind().into_any();

            cloned.complete(value);

            assert_eq!(
                task.result(py).unwrap().bind(py).extract::<i64>().unwrap(),
                3
            );
        });
    }

    #[test]
    fn is_done_completed_why_completed_callable_must_report_terminal() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let task = CallableTask::new();
            assert!(!task.is_done());
            let value = 1_i64.into_pyobject(py).unwrap().unbind().into_any();
            task.complete(value);
            assert!(task.is_done());
        });
    }

    #[test]
    fn is_done_failed_why_failed_callable_must_report_terminal() {
        let task = CallableTask::new();
        assert!(!task.is_done());
        task.fail("oops".to_string());
        assert!(task.is_done());
    }

    #[test]
    fn is_done_cancelled_why_cancelled_callable_must_report_terminal() {
        let task = CallableTask::new();
        assert!(!task.is_done());
        task.cancel();
        assert!(task.is_done());
    }

    #[test]
    fn cancel_from_failed_returns_false_why_terminal_callable_tasks_must_not_reopen() {
        let task = CallableTask::new();
        task.fail("already failed".to_string());
        assert!(!task.cancel());
    }

    #[test]
    fn concurrent_complete_keeps_first_result_why_only_one_terminal_transition_may_win() {
        pyo3::prepare_freethreaded_python();

        let task = CallableTask::new();
        let start = Arc::new(Barrier::new(3));
        let left_task = task.clone();
        let right_task = task.clone();
        let left_start = start.clone();
        let right_start = start.clone();

        let left = thread::spawn(move || {
            left_start.wait();
            Python::with_gil(|py| {
                let value = 1_i64.into_pyobject(py).unwrap().unbind().into_any();
                left_task.complete(value);
            });
        });
        let right = thread::spawn(move || {
            right_start.wait();
            Python::with_gil(|py| {
                let value = 2_i64.into_pyobject(py).unwrap().unbind().into_any();
                right_task.complete(value);
            });
        });

        start.wait();
        left.join().unwrap();
        right.join().unwrap();

        Python::with_gil(|py| {
            let result = task.result(py).unwrap().bind(py).extract::<i64>().unwrap();
            assert!(result == 1 || result == 2);
        });
    }
}
