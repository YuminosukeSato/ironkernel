use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::buffer::inner::Buffer;

/// Python-visible buffer handle wrapping a Rust Buffer.
#[pyclass(name = "Buffer")]
#[derive(Clone)]
pub struct PyBuffer {
    pub(crate) inner: Buffer,
}

impl PyBuffer {
    pub fn new(inner: Buffer) -> Self {
        PyBuffer { inner }
    }
}

#[pymethods]
impl PyBuffer {
    /// Convert to numpy array (copies data out).
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = self.inner.as_f64_slice();
        Ok(PyArray1::from_slice(py, slice))
    }

    /// Get a single scalar value (first element).
    fn scalar(&self) -> PyResult<f64> {
        let slice = self.inner.as_f64_slice();
        if slice.is_empty() {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "buffer is empty",
            ))
        } else {
            Ok(slice[0])
        }
    }

    /// Number of elements.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Buffer(dtype={}, shape={:?}, len={})",
            self.inner.dtype(),
            self.inner.shape(),
            self.inner.len()
        )
    }
}

/// Create a Buffer from a numpy array (zero-copy for contiguous f64).
pub fn asarray_from_numpy(array: PyReadonlyArray1<f64>) -> Buffer {
    let slice = array.as_slice().expect("contiguous array required");
    Buffer::from_f64_vec(slice.to_vec())
}
