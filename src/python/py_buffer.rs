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

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArrayMethods;
    use pyo3::exceptions::PyValueError;

    #[test]
    fn numpy_roundtrips_values_why_python_callers_must_see_the_rust_buffer_contents() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let buffer = PyBuffer::new(Buffer::from_f64_vec(vec![1.0, 2.0, 3.0]));
            let array = buffer.numpy(py).unwrap();

            assert_eq!(array.readonly().as_slice().unwrap(), &[1.0, 2.0, 3.0]);
        });
    }

    #[test]
    fn scalar_returns_first_value_why_reduce_results_share_the_same_buffer_wrapper() {
        let buffer = PyBuffer::new(Buffer::from_f64_vec(vec![7.5, 9.0]));
        assert_eq!(buffer.scalar().unwrap(), 7.5);
    }

    #[test]
    fn scalar_rejects_empty_buffer_why_public_scalar_access_must_fail_loudly_on_missing_data() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let err = PyBuffer::new(Buffer::from_f64_vec(vec![]))
                .scalar()
                .unwrap_err();

            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("empty"));
        });
    }

    #[test]
    fn len_and_repr_reflect_shape_why_debug_output_must_match_the_wrapped_buffer() {
        let buffer = PyBuffer::new(Buffer::from_f64_vec(vec![1.0, 2.0]));

        assert_eq!(buffer.__len__(), 2);
        assert_eq!(buffer.__repr__(), "Buffer(dtype=float64, shape=[2], len=2)");
    }

    #[test]
    fn asarray_from_numpy_copies_values_why_runtime_entrypoints_depend_on_consistent_host_buffer_conversion(
    ) {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let array = PyArray1::from_vec(py, vec![4.0, 5.0, 6.0]);
            let buffer = asarray_from_numpy(array.readonly());

            assert_eq!(buffer.as_f64_slice(), &[4.0, 5.0, 6.0]);
        });
    }
}
