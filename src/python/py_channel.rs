use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::channel::bounded::Channel;
use crate::channel::select::{select_channels, SelectResult};

use super::py_buffer::PyBuffer;

/// Python-visible channel handle.
#[pyclass(name = "Channel")]
#[derive(Clone)]
pub struct PyChannel {
    pub(crate) inner: Channel,
}

impl PyChannel {
    pub fn new(capacity: usize) -> Self {
        PyChannel {
            inner: Channel::new(capacity),
        }
    }
}

fn channel_error_to_py<E: std::fmt::Display>(err: E) -> PyErr {
    PyErr::new::<PyRuntimeError, _>(err.to_string())
}

#[pymethods]
impl PyChannel {
    /// Send a buffer into the channel.
    fn send(&self, py: Python<'_>, buf: PyBuffer) -> PyResult<()> {
        py.allow_threads(|| self.inner.send(buf.inner))
            .map_err(channel_error_to_py)
    }

    /// Receive a buffer from the channel. Blocks until available.
    fn recv(&self, py: Python<'_>) -> PyResult<PyBuffer> {
        let buf = py
            .allow_threads(|| self.inner.recv())
            .map_err(channel_error_to_py)?;
        Ok(PyBuffer::new(buf))
    }

    /// Close the channel. Idempotent.
    fn close(&self) {
        self.inner.close();
    }

    /// Returns true if the channel has been closed.
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    fn __repr__(&self) -> String {
        "Channel()".to_string()
    }
}

/// Python-visible `RecvCase` for select.
#[pyclass(name = "RecvCase")]
#[derive(Clone)]
pub struct PyRecvCase {
    pub(crate) channel: PyChannel,
}

#[pymethods]
impl PyRecvCase {
    #[new]
    fn new(channel: PyChannel) -> Self {
        PyRecvCase { channel }
    }
}

/// Python-visible select function.
#[pyfunction]
#[pyo3(signature = (*cases, default=false))]
pub fn py_select(
    py: Python<'_>,
    cases: Vec<PyRecvCase>,
    default: bool,
) -> PyResult<(i64, PyObject)> {
    let channels: Vec<&Channel> = cases.iter().map(|c| &c.channel.inner).collect();

    let result = py
        .allow_threads(|| select_channels(&channels, default))
        .map_err(channel_error_to_py)?;

    match result {
        SelectResult::Received(idx, buf) => {
            let py_buf = PyBuffer::new(buf);
            Ok((idx as i64, py_buf.into_pyobject(py)?.into_any().unbind()))
        }
        SelectResult::Default => Ok((-1, py.None())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::bounded::Msg;
    use pyo3::exceptions::PyRuntimeError;
    use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    fn spawn_gil_progress_probe(
        phase: Arc<AtomicU8>,
        counter: Arc<AtomicUsize>,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            while phase.load(Ordering::SeqCst) == 1 {
                Python::with_gil(|_| {
                    counter.fetch_add(1, Ordering::SeqCst);
                });
                thread::yield_now();
            }
        })
    }

    #[test]
    fn select_default_returns_minus_one_and_none_why_public_contract_must_not_drift() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let channel = PyChannel::new(1);
            let case = PyRecvCase::new(channel);

            let (index, value) = py_select(py, vec![case], true).unwrap();

            assert_eq!(index, -1);
            assert!(value.is_none(py));
        });
    }

    #[test]
    fn send_releases_gil_why_blocking_boundary_calls_must_not_starve_python_threads() {
        pyo3::prepare_freethreaded_python();

        let channel = PyChannel::new(1);
        channel
            .inner
            .send(crate::buffer::inner::Buffer::from_f64_vec(vec![1.0]))
            .unwrap();

        let phase = Arc::new(AtomicU8::new(1));
        let counter = Arc::new(AtomicUsize::new(0));
        let probe = spawn_gil_progress_probe(phase.clone(), counter.clone());

        let receiver_channel = channel.clone();
        let unblock_phase = phase.clone();
        let receiver = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            unblock_phase.store(2, Ordering::SeqCst);
            let _ = receiver_channel.inner.recv().unwrap();
        });

        Python::with_gil(|py| {
            channel
                .send(
                    py,
                    crate::python::py_buffer::PyBuffer::new(
                        crate::buffer::inner::Buffer::from_f64_vec(vec![2.0]),
                    ),
                )
                .unwrap();
        });

        receiver.join().unwrap();
        probe.join().unwrap();

        assert!(counter.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn recv_releases_gil_why_blocking_boundary_calls_must_not_starve_python_threads() {
        pyo3::prepare_freethreaded_python();

        let channel = PyChannel::new(1);
        let phase = Arc::new(AtomicU8::new(1));
        let counter = Arc::new(AtomicUsize::new(0));
        let probe = spawn_gil_progress_probe(phase.clone(), counter.clone());

        let sender_channel = channel.clone();
        let unblock_phase = phase.clone();
        let sender = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            unblock_phase.store(2, Ordering::SeqCst);
            sender_channel
                .inner
                .send(crate::buffer::inner::Buffer::from_f64_vec(vec![7.0]))
                .unwrap();
        });

        Python::with_gil(|py| {
            let result = channel.recv(py).unwrap();
            assert_eq!(result.inner.as_f64_slice(), &[7.0]);
        });

        sender.join().unwrap();
        probe.join().unwrap();

        assert!(counter.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn select_releases_gil_why_blocking_boundary_calls_must_not_starve_python_threads() {
        pyo3::prepare_freethreaded_python();

        let channel = PyChannel::new(1);
        let phase = Arc::new(AtomicU8::new(1));
        let counter = Arc::new(AtomicUsize::new(0));
        let probe = spawn_gil_progress_probe(phase.clone(), counter.clone());

        let sender_channel = channel.clone();
        let unblock_phase = phase.clone();
        let sender = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            unblock_phase.store(2, Ordering::SeqCst);
            sender_channel
                .inner
                .send(crate::buffer::inner::Buffer::from_f64_vec(vec![42.0]))
                .unwrap();
        });

        Python::with_gil(|py| {
            let case = PyRecvCase::new(channel.clone());
            let (index, value) = py_select(py, vec![case], false).unwrap();
            assert_eq!(index, 0);
            assert!(!value.is_none(py));
        });

        sender.join().unwrap();
        probe.join().unwrap();

        assert!(counter.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn pychannel_repr() {
        let ch = PyChannel::new(10);
        assert_eq!(ch.__repr__(), "Channel()");
    }

    #[test]
    fn close_updates_public_state_why_python_code_must_observe_channel_shutdown() {
        let ch = PyChannel::new(1);

        assert!(!ch.is_closed());
        ch.close();
        assert!(ch.is_closed());
        ch.close();
        assert!(ch.is_closed());
    }

    #[test]
    fn send_reports_closed_channel_why_python_producers_must_see_backpressure_failures() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let channel = PyChannel::new(1);
            channel.close();

            let result = channel.send(
                py,
                PyBuffer::new(crate::buffer::inner::Buffer::from_f64_vec(vec![1.0])),
            );

            assert!(result.is_err());
            assert!(result.err().unwrap().is_instance_of::<PyRuntimeError>(py));
        });
    }

    #[test]
    fn recv_reports_closed_channel_why_python_consumers_must_see_terminal_channel_state() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let channel = PyChannel::new(1);
            channel.close();

            let result = channel.recv(py);

            assert!(result.is_err());
            assert!(result.err().unwrap().is_instance_of::<PyRuntimeError>(py));
        });
    }

    #[test]
    fn select_reports_closed_channel_why_muxed_receives_must_not_hide_runtime_errors() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let channel = PyChannel::new(1);
            channel.inner.sender().send(Msg::Closed).unwrap();
            let case = PyRecvCase::new(channel);

            let result = py_select(py, vec![case], true);

            assert!(result.is_err());
            assert!(result.err().unwrap().is_instance_of::<PyRuntimeError>(py));
        });
    }
}
