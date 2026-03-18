use std::sync::{Arc, Condvar, Mutex};

use crate::buffer::inner::Buffer;
use crate::error::{ParsecError, ParsecResult};

/// State of a task in its lifecycle.
///
/// ```text
/// Created → RunningCompute → Completed          (no delivery)
/// Created → RunningCompute → DeliveryQueued → Delivering → Completed  (with out=channel)
///                                                        → Failed     (channel closed)
/// Any non-terminal → Cancelled                   (cooperative cancel)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum TaskState {
    Created,
    RunningCompute,
    DeliveryQueued,
    Delivering,
    Completed,
    Failed(ParsecError),
    Cancelled,
}

impl TaskState {
    /// Returns true if this is a terminal state (Completed/Failed/Cancelled).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TaskState::Completed | TaskState::Failed(_) | TaskState::Cancelled
        )
    }
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

/// Shared state inside a `TaskHandle`.
struct TaskInner {
    state: TaskState,
    result: Option<TaskResult>,
}

/// Handle to an asynchronously running task.
///
/// Created by `rt.go()`. The caller can poll `is_done()` or block on `result()`.
/// Uses `Condvar` for efficient blocking in `result()`.
#[derive(Clone)]
pub struct TaskHandle {
    inner: Arc<(Mutex<TaskInner>, Condvar)>,
}

impl TaskHandle {
    pub fn new() -> Self {
        TaskHandle {
            inner: Arc::new((
                Mutex::new(TaskInner {
                    state: TaskState::Created,
                    result: None,
                }),
                Condvar::new(),
            )),
        }
    }

    pub fn state(&self) -> TaskState {
        self.inner.0.lock().unwrap().state.clone()
    }

    pub fn is_done(&self) -> bool {
        self.inner.0.lock().unwrap().state.is_terminal()
    }

    pub fn set_running_compute(&self) {
        self.inner.0.lock().unwrap().state = TaskState::RunningCompute;
    }

    /// Transition to `DeliveryQueued`. Stores the result for later retrieval.
    /// Used when `out=channel` is specified: compute is done, delivery pending.
    pub fn set_delivery_queued(&self, result: TaskResult) {
        let mut inner = self.inner.0.lock().unwrap();
        if !inner.state.is_terminal() {
            inner.result = Some(result);
            inner.state = TaskState::DeliveryQueued;
        }
    }

    /// Transition to Delivering. Called when the delivery executor picks up the job.
    pub fn set_delivering(&self) {
        let mut inner = self.inner.0.lock().unwrap();
        if matches!(inner.state, TaskState::DeliveryQueued) {
            inner.state = TaskState::Delivering;
        }
    }

    /// Complete the task with a result.
    /// No-op if already in a terminal state (Completed/Failed/Cancelled).
    pub fn complete(&self, result: TaskResult) {
        let mut inner = self.inner.0.lock().unwrap();
        if !inner.state.is_terminal() {
            inner.state = TaskState::Completed;
            inner.result = Some(result);
            self.inner.1.notify_all();
        }
    }

    /// Complete the task after delivery. Uses the already-staged result.
    /// No-op if already in a terminal state.
    pub fn complete_delivery(&self) {
        let mut inner = self.inner.0.lock().unwrap();
        if !inner.state.is_terminal() {
            inner.state = TaskState::Completed;
            self.inner.1.notify_all();
        }
    }

    /// Mark the task as failed.
    /// No-op if already in a terminal state.
    pub fn fail(&self, err: ParsecError) {
        let mut inner = self.inner.0.lock().unwrap();
        if !inner.state.is_terminal() {
            inner.state = TaskState::Failed(err);
            self.inner.1.notify_all();
        }
    }

    /// Request cooperative cancellation.
    ///
    /// Cancellable from: `Created`, `RunningCompute`, `DeliveryQueued`, `Delivering`.
    /// Returns true if cancellation was applied, false if already terminal.
    pub fn cancel(&self) -> bool {
        let mut inner = self.inner.0.lock().unwrap();
        if inner.state.is_terminal() {
            return false;
        }
        inner.state = TaskState::Cancelled;
        self.inner.1.notify_all();
        true
    }

    /// Block until the task reaches a terminal state and return the result.
    /// Uses Condvar for efficient waiting (no spin-loop).
    pub fn result(&self) -> ParsecResult<TaskResult> {
        let (mutex, condvar) = &*self.inner;
        let mut inner = mutex.lock().unwrap();
        while !inner.state.is_terminal() {
            inner = condvar.wait(inner).unwrap();
        }
        if matches!(inner.state, TaskState::Completed) {
            return Ok(inner.result.clone().unwrap());
        }
        if let TaskState::Failed(e) = &inner.state {
            return Err(e.clone());
        }
        debug_assert!(matches!(inner.state, TaskState::Cancelled));
        Err(ParsecError::Cancelled)
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

        handle.set_running_compute();
        assert_eq!(handle.state(), TaskState::RunningCompute);
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
        handle.set_running_compute();
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
    fn task_cancel_from_running_compute() {
        let handle = TaskHandle::new();
        handle.set_running_compute();
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
        h1.set_running_compute();
        assert_eq!(h2.state(), TaskState::RunningCompute);
    }

    #[test]
    fn task_result_from_thread() {
        let handle = TaskHandle::new();
        let h2 = handle.clone();
        std::thread::spawn(move || {
            h2.set_running_compute();
            h2.complete(TaskResult::Scalar(99.0));
        });
        let result = handle.result().unwrap();
        assert_eq!(result.as_scalar().unwrap(), 99.0);
    }

    #[test]
    fn complete_after_cancel_is_noop() {
        let handle = TaskHandle::new();
        handle.set_running_compute();
        assert!(handle.cancel());
        handle.complete(TaskResult::Scalar(42.0));
        assert_eq!(handle.state(), TaskState::Cancelled);
        assert!(handle.result().is_err());
    }

    #[test]
    fn fail_after_cancel_is_noop() {
        let handle = TaskHandle::new();
        handle.set_running_compute();
        assert!(handle.cancel());
        handle.fail(ParsecError::Internal("late error".into()));
        assert_eq!(handle.state(), TaskState::Cancelled);
    }

    #[test]
    fn complete_after_complete_is_noop() {
        let handle = TaskHandle::new();
        handle.complete(TaskResult::Scalar(1.0));
        handle.complete(TaskResult::Scalar(2.0));
        assert_eq!(handle.result().unwrap().as_scalar().unwrap(), 1.0);
    }

    #[test]
    fn fail_after_fail_is_noop() {
        let handle = TaskHandle::new();
        handle.set_running_compute();
        handle.fail(ParsecError::Internal("first".into()));
        handle.fail(ParsecError::Internal("second".into()));
        assert_eq!(
            handle.state(),
            TaskState::Failed(ParsecError::Internal("first".into()))
        );
    }

    #[test]
    fn task_handle_default_is_created() {
        let handle = TaskHandle::default();
        assert_eq!(handle.state(), TaskState::Created);
    }

    // --- Delivery lifecycle ---

    #[test]
    fn delivery_lifecycle_state_transition() {
        let handle = TaskHandle::new();
        assert_eq!(handle.state(), TaskState::Created);

        handle.set_running_compute();
        assert_eq!(handle.state(), TaskState::RunningCompute);
        assert!(!handle.is_done());

        handle.set_delivery_queued(TaskResult::Buffer(Buffer::from_f64_vec(vec![1.0])));
        assert_eq!(handle.state(), TaskState::DeliveryQueued);
        assert!(!handle.is_done());

        handle.set_delivering();
        assert_eq!(handle.state(), TaskState::Delivering);
        assert!(!handle.is_done());

        handle.complete_delivery();
        assert_eq!(handle.state(), TaskState::Completed);
        assert!(handle.is_done());

        let result = handle.result().unwrap();
        assert_eq!(result.as_buffer().unwrap().as_f64_slice(), &[1.0]);
    }

    #[test]
    fn cancel_during_delivery_queued() {
        let handle = TaskHandle::new();
        handle.set_running_compute();
        handle.set_delivery_queued(TaskResult::Scalar(42.0));
        assert!(handle.cancel());
        assert_eq!(handle.state(), TaskState::Cancelled);
    }

    #[test]
    fn cancel_during_delivering() {
        let handle = TaskHandle::new();
        handle.set_running_compute();
        handle.set_delivery_queued(TaskResult::Scalar(42.0));
        handle.set_delivering();
        assert!(handle.cancel());
        assert_eq!(handle.state(), TaskState::Cancelled);
    }

    #[test]
    fn condvar_wakeup_on_complete() {
        let handle = TaskHandle::new();
        let h2 = handle.clone();

        let t = std::thread::spawn(move || {
            // result() blocks until terminal
            handle.result().unwrap()
        });

        std::thread::sleep(std::time::Duration::from_millis(10));
        h2.set_running_compute();
        h2.complete(TaskResult::Scalar(7.0));

        let result = t.join().unwrap();
        assert_eq!(result.as_scalar().unwrap(), 7.0);
    }

    #[test]
    fn condvar_wakeup_on_cancel() {
        let handle = TaskHandle::new();
        let h2 = handle.clone();

        let t = std::thread::spawn(move || handle.result());

        std::thread::sleep(std::time::Duration::from_millis(10));
        h2.cancel();

        let err = t.join().unwrap().unwrap_err();
        assert_eq!(err, ParsecError::Cancelled);
    }

    #[test]
    fn condvar_wakeup_on_fail() {
        let handle = TaskHandle::new();
        let h2 = handle.clone();

        let t = std::thread::spawn(move || handle.result());

        std::thread::sleep(std::time::Duration::from_millis(10));
        h2.fail(ParsecError::ChannelClosed);

        let err = t.join().unwrap().unwrap_err();
        assert_eq!(err, ParsecError::ChannelClosed);
    }

    #[test]
    fn task_state_is_terminal() {
        assert!(!TaskState::Created.is_terminal());
        assert!(!TaskState::RunningCompute.is_terminal());
        assert!(!TaskState::DeliveryQueued.is_terminal());
        assert!(!TaskState::Delivering.is_terminal());
        assert!(TaskState::Completed.is_terminal());
        assert!(TaskState::Failed(ParsecError::Cancelled).is_terminal());
        assert!(TaskState::Cancelled.is_terminal());
    }

    #[test]
    fn set_delivering_only_from_delivery_queued() {
        let handle = TaskHandle::new();
        handle.set_running_compute();
        handle.set_delivering(); // no-op: not DeliveryQueued
        assert_eq!(handle.state(), TaskState::RunningCompute);
    }

    #[test]
    fn mutation_guard_task_delivery_completion_why_terminal_visibility_must_follow_staged_result() {
        let handle = TaskHandle::new();
        let expected = Buffer::from_f64_vec(vec![3.0, 5.0, 8.0]);

        handle.set_running_compute();
        handle.set_delivery_queued(TaskResult::Buffer(expected.clone()));
        assert!(!handle.is_done());
        handle.set_delivering();
        assert!(!handle.is_done());
        handle.complete_delivery();

        let result = handle.result().unwrap();
        assert!(handle.is_done());
        assert_eq!(
            result.as_buffer().unwrap().as_f64_slice(),
            expected.as_f64_slice()
        );
        assert!(!handle.cancel());
    }

    #[test]
    fn mutation_guard_task_cancel_why_waiters_must_observe_cancelled_and_late_cancel_must_fail() {
        let handle = TaskHandle::new();

        handle.set_running_compute();
        assert!(handle.cancel());

        let err = handle.result().unwrap_err();
        assert_eq!(err, ParsecError::Cancelled);
        assert!(handle.is_done());
        assert!(!handle.cancel());
    }
}
