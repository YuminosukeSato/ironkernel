use std::sync::{Arc, Mutex};

use crate::buffer::inner::Buffer;
use crate::error::{ParsecError, ParsecResult};

/// State of a task in its lifecycle.
#[derive(Debug, Clone, PartialEq)]
pub enum TaskState {
    Created,
    Running,
    Completed,
    Failed(ParsecError),
    Cancelled,
}

/// Result of a completed task.
#[derive(Debug, Clone)]
pub enum TaskResult {
    Buffer(Buffer),
    Scalar(f64),
}

impl TaskResult {
    pub fn as_buffer(&self) -> ParsecResult<&Buffer> {
        match self {
            TaskResult::Buffer(b) => Ok(b),
            TaskResult::Scalar(_) => Err(ParsecError::TypeError(
                "expected Buffer result, got Scalar".into(),
            )),
        }
    }

    pub fn as_scalar(&self) -> ParsecResult<f64> {
        match self {
            TaskResult::Scalar(v) => Ok(*v),
            TaskResult::Buffer(_) => Err(ParsecError::TypeError(
                "expected Scalar result, got Buffer".into(),
            )),
        }
    }
}

/// Shared state inside a TaskHandle.
struct TaskInner {
    state: TaskState,
    result: Option<TaskResult>,
}

/// Handle to an asynchronously running task.
///
/// Created by `rt.go()`. The caller can poll `is_done()` or block on `result()`.
#[derive(Clone)]
pub struct TaskHandle {
    inner: Arc<Mutex<TaskInner>>,
}

impl TaskHandle {
    pub fn new() -> Self {
        TaskHandle {
            inner: Arc::new(Mutex::new(TaskInner {
                state: TaskState::Created,
                result: None,
            })),
        }
    }

    pub fn state(&self) -> TaskState {
        self.inner.lock().unwrap().state.clone()
    }

    pub fn is_done(&self) -> bool {
        matches!(
            self.inner.lock().unwrap().state,
            TaskState::Completed | TaskState::Failed(_) | TaskState::Cancelled
        )
    }

    pub fn set_running(&self) {
        self.inner.lock().unwrap().state = TaskState::Running;
    }

    pub fn complete(&self, result: TaskResult) {
        let mut inner = self.inner.lock().unwrap();
        inner.state = TaskState::Completed;
        inner.result = Some(result);
    }

    pub fn fail(&self, err: ParsecError) {
        self.inner.lock().unwrap().state = TaskState::Failed(err);
    }

    pub fn cancel(&self) -> bool {
        let mut inner = self.inner.lock().unwrap();
        match inner.state {
            TaskState::Created | TaskState::Running => {
                inner.state = TaskState::Cancelled;
                true
            }
            _ => false,
        }
    }

    /// Block until the task completes and return the result.
    /// If the task failed, returns the error.
    /// If cancelled, returns Cancelled error.
    pub fn result(&self) -> ParsecResult<TaskResult> {
        // In a real impl this would use a condvar. For now, spin-wait
        // (acceptable since tasks complete via rayon which is fast).
        loop {
            let inner = self.inner.lock().unwrap();
            match &inner.state {
                TaskState::Completed => return Ok(inner.result.clone().unwrap()),
                TaskState::Failed(e) => return Err(e.clone()),
                TaskState::Cancelled => return Err(ParsecError::Cancelled),
                TaskState::Created | TaskState::Running => {}
            }
            drop(inner);
            std::thread::yield_now();
        }
    }
}

impl Default for TaskHandle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_lifecycle_created_to_completed() {
        let handle = TaskHandle::new();
        assert_eq!(handle.state(), TaskState::Created);
        assert!(!handle.is_done());

        handle.set_running();
        assert_eq!(handle.state(), TaskState::Running);
        assert!(!handle.is_done());

        handle.complete(TaskResult::Scalar(42.0));
        assert_eq!(handle.state(), TaskState::Completed);
        assert!(handle.is_done());

        let result = handle.result().unwrap();
        assert_eq!(result.as_scalar().unwrap(), 42.0);
    }

    #[test]
    fn task_lifecycle_created_to_failed() {
        let handle = TaskHandle::new();
        handle.set_running();
        handle.fail(ParsecError::Internal("boom".into()));
        assert!(handle.is_done());

        let err = handle.result().unwrap_err();
        assert_eq!(err, ParsecError::Internal("boom".into()));
    }

    #[test]
    fn task_cancel_from_created() {
        let handle = TaskHandle::new();
        assert!(handle.cancel());
        assert_eq!(handle.state(), TaskState::Cancelled);
        assert!(handle.is_done());

        let err = handle.result().unwrap_err();
        assert_eq!(err, ParsecError::Cancelled);
    }

    #[test]
    fn task_cancel_from_running() {
        let handle = TaskHandle::new();
        handle.set_running();
        assert!(handle.cancel());
        assert_eq!(handle.state(), TaskState::Cancelled);
    }

    #[test]
    fn task_cancel_from_completed_fails() {
        let handle = TaskHandle::new();
        handle.complete(TaskResult::Scalar(1.0));
        assert!(!handle.cancel()); // already completed
        assert_eq!(handle.state(), TaskState::Completed);
    }

    #[test]
    fn task_result_buffer() {
        let handle = TaskHandle::new();
        let buf = Buffer::from_f64_vec(vec![1.0, 2.0, 3.0]);
        handle.complete(TaskResult::Buffer(buf));

        let result = handle.result().unwrap();
        let buf = result.as_buffer().unwrap();
        assert_eq!(buf.as_f64_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn task_result_type_mismatch_buffer_as_scalar() {
        let result = TaskResult::Buffer(Buffer::from_f64_vec(vec![1.0]));
        let err = result.as_scalar().unwrap_err();
        assert!(matches!(err, ParsecError::TypeError(_)));
    }

    #[test]
    fn task_result_type_mismatch_scalar_as_buffer() {
        let result = TaskResult::Scalar(42.0);
        let err = result.as_buffer().unwrap_err();
        assert!(matches!(err, ParsecError::TypeError(_)));
    }

    #[test]
    fn task_handle_clone_shares_state() {
        let h1 = TaskHandle::new();
        let h2 = h1.clone();
        h1.set_running();
        assert_eq!(h2.state(), TaskState::Running);
    }

    #[test]
    fn task_result_from_thread() {
        let handle = TaskHandle::new();
        let h2 = handle.clone();
        std::thread::spawn(move || {
            h2.set_running();
            h2.complete(TaskResult::Scalar(99.0));
        });
        let result = handle.result().unwrap();
        assert_eq!(result.as_scalar().unwrap(), 99.0);
    }
}
