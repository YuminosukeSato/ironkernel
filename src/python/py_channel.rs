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

#[pymethods]
impl PyChannel {
    /// Send a buffer into the channel.
    fn send(&self, py: Python<'_>, buf: PyBuffer) -> PyResult<()> {
        py.allow_threads(|| self.inner.send(buf.inner))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Receive a buffer from the channel. Blocks until available.
    fn recv(&self, py: Python<'_>) -> PyResult<PyBuffer> {
        let buf = py
            .allow_threads(|| self.inner.recv())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyBuffer::new(buf))
    }

    fn __repr__(&self) -> String {
        "Channel()".to_string()
    }
}

/// Python-visible RecvCase for select.
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
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    match result {
        SelectResult::Received(idx, buf) => {
            let py_buf = PyBuffer::new(buf);
            Ok((idx as i64, py_buf.into_pyobject(py)?.into_any().unbind()))
        }
        SelectResult::Default => Ok((-1, py.None())),
    }
}
