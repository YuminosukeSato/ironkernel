use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::buffer::inner::Buffer;
use crate::runtime::delivery::{submit_delivery, DeliveryJob};
use crate::runtime::task::{TaskHandle, TaskResult};

use super::py_buffer::{asarray_from_numpy, PyBuffer};
use super::py_channel::PyChannel;
use super::py_kernel::{execute_map, execute_reduce, PyMapSpec, PyReduceSpec};
use super::py_task::PyTaskHandle;

/// The runtime module exposed to Python.
#[pyclass(name = "_RuntimeModule")]
pub struct PyRuntimeModule;

#[pymethods]
impl PyRuntimeModule {
    #[new]
    fn new() -> Self {
        PyRuntimeModule
    }

    /// Create a Buffer from a numpy array.
    fn asarray(&self, array: PyReadonlyArray1<f64>) -> PyBuffer {
        PyBuffer::new(asarray_from_numpy(array))
    }

    /// Submit a computation and return a `TaskHandle`.
    /// Accepts either a `MapSpec` (elementwise) or a `ReduceSpec`.
    /// If `out` is provided, the result is delivered to that channel asynchronously.
    #[pyo3(signature = (spec, out=None))]
    fn go(
        &self,
        py: Python<'_>,
        spec: &Bound<'_, PyAny>,
        out: Option<PyChannel>,
    ) -> PyResult<PyTaskHandle> {
        let out_channel = out.map(|c| c.inner);

        if let Ok(map_spec) = spec.extract::<PyMapSpec>() {
            let handle = TaskHandle::new();
            let h2 = handle.clone();

            py.allow_threads(|| {
                h2.set_running_compute();
                match execute_map(&map_spec) {
                    Ok(buf) => {
                        if let Some(ch) = out_channel {
                            h2.set_delivery_queued(TaskResult::Buffer(buf.clone()));
                            submit_delivery(DeliveryJob {
                                task: h2,
                                channel: ch,
                                buffer: buf,
                            });
                        } else {
                            h2.complete(TaskResult::Buffer(buf));
                        }
                    }
                    Err(e) => h2.fail(e),
                }
            });

            Ok(PyTaskHandle::new(handle))
        } else if let Ok(reduce_spec) = spec.extract::<PyReduceSpec>() {
            let handle = TaskHandle::new();
            let h2 = handle.clone();

            py.allow_threads(|| {
                h2.set_running_compute();
                match execute_reduce(&reduce_spec) {
                    Ok(v) => {
                        if let Some(ch) = out_channel {
                            h2.set_delivery_queued(TaskResult::Scalar(v));
                            submit_delivery(DeliveryJob {
                                task: h2,
                                channel: ch,
                                buffer: Buffer::from_f64_vec(vec![v]),
                            });
                        } else {
                            h2.complete(TaskResult::Scalar(v));
                        }
                    }
                    Err(e) => h2.fail(e),
                }
            });

            Ok(PyTaskHandle::new(handle))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "go() expects a MapSpec or ReduceSpec",
            ))
        }
    }

    /// Create a bounded channel.
    fn chan(&self, capacity: usize) -> PyChannel {
        PyChannel::new(capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyTypeError;

    #[test]
    fn go_rejects_non_spec_input_why_boundary_type_contract_must_fail_fast() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let runtime = PyRuntimeModule::new();
            let invalid_spec = 123i64.into_pyobject(py).unwrap();
            let result = runtime.go(py, invalid_spec.as_any(), None);
            assert!(result.is_err());
            let err = result.err().unwrap();

            assert!(err.is_instance_of::<PyTypeError>(py));
            assert!(err.to_string().contains("MapSpec or ReduceSpec"));
        });
    }

    #[test]
    fn go_map_failure_surfaces_via_task_result_why_async_execution_must_preserve_arg_errors() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            use crate::buffer::inner::{Buffer, DType};
            use crate::ir::compiler::ArgValue;
            use crate::ir::expr::Expr;
            use crate::ir::kernel::{ArgSpec, KernelKind, KernelSpec, TensorSpec};
            use crate::python::py_kernel::{PyKernelSpec, PyMapSpec};
            use std::collections::HashMap;

            let spec = PyKernelSpec {
                inner: KernelSpec {
                    kind: KernelKind::Elementwise,
                    args: vec![
                        ArgSpec {
                            name: "x".to_string(),
                            dtype: DType::F64,
                            is_scalar: false,
                        },
                        ArgSpec {
                            name: "y".to_string(),
                            dtype: DType::F64,
                            is_scalar: false,
                        },
                    ],
                    output: TensorSpec { dtype: DType::F64 },
                    expr: Expr::ArgRef(0),
                },
            };
            let mut args = HashMap::new();
            args.insert(
                "x".to_string(),
                ArgValue::Buffer(Buffer::from_f64_vec(vec![1.0, 2.0])),
            );

            let map_spec = PyMapSpec { spec, args };
            let py_map = map_spec.into_pyobject(py).unwrap();
            let task = PyRuntimeModule::new()
                .go(py, py_map.as_any(), None)
                .unwrap();
            let err = task.inner.result().unwrap_err();

            assert!(matches!(err, crate::error::ParsecError::ArgError(_)));
        });
    }

    #[test]
    fn go_reduce_failure_surfaces_via_task_result_why_empty_inputs_must_not_be_silently_accepted() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            use crate::ir::kernel::ReduceOp;
            use crate::python::py_kernel::PyReduceSpec;

            let reduce_spec = PyReduceSpec {
                op: ReduceOp::Sum,
                buffer: Buffer::from_f64_vec(vec![]),
            };
            let py_reduce = reduce_spec.into_pyobject(py).unwrap();
            let task = PyRuntimeModule::new()
                .go(py, py_reduce.as_any(), None)
                .unwrap();
            let err = task.inner.result().unwrap_err();

            assert!(matches!(err, crate::error::ParsecError::EmptyCollection(_)));
        });
    }

    #[test]
    fn asarray_roundtrip() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            use numpy::PyArrayMethods;
            let runtime = PyRuntimeModule::new();
            let np_array = numpy::PyArray1::from_vec(py, vec![1.0, 2.0, 3.0]);
            let buf = runtime.asarray(np_array.readonly());
            assert_eq!(buf.inner.as_f64_slice(), &[1.0, 2.0, 3.0]);
        });
    }

    #[test]
    fn go_with_map_spec() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            use crate::buffer::inner::{Buffer, DType};
            use crate::ir::compiler::ArgValue;
            use crate::ir::expr::Expr;
            use crate::ir::kernel::{ArgSpec, KernelKind, KernelSpec, TensorSpec};
            use crate::python::py_kernel::{PyKernelSpec, PyMapSpec};
            use std::collections::HashMap;

            let inner_spec = KernelSpec {
                kind: KernelKind::Elementwise,
                args: vec![ArgSpec {
                    name: "x".to_string(),
                    dtype: DType::F64,
                    is_scalar: false,
                }],
                output: TensorSpec { dtype: DType::F64 },
                expr: Expr::ArgRef(0),
            };
            let spec = PyKernelSpec { inner: inner_spec };
            let mut args = HashMap::new();
            args.insert(
                "x".to_string(),
                ArgValue::Buffer(Buffer::from_f64_vec(vec![1.0, 2.0])),
            );
            let map_spec = PyMapSpec { spec, args };
            let py_map = map_spec.into_pyobject(py).unwrap();

            let runtime = PyRuntimeModule::new();
            let task = runtime.go(py, py_map.as_any(), None).unwrap();
            assert!(task.inner.is_done());
        });
    }

    #[test]
    fn go_with_reduce_spec() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            use crate::ir::kernel::ReduceOp;
            use crate::python::py_kernel::PyReduceSpec;

            let reduce_spec = PyReduceSpec {
                op: ReduceOp::Sum,
                buffer: crate::buffer::inner::Buffer::from_f64_vec(vec![1.0, 2.0, 3.0]),
            };
            let py_reduce = reduce_spec.into_pyobject(py).unwrap();

            let runtime = PyRuntimeModule::new();
            let task = runtime.go(py, py_reduce.as_any(), None).unwrap();
            assert!(task.inner.is_done());
        });
    }

    #[test]
    fn chan_creates_channel() {
        pyo3::prepare_freethreaded_python();
        let runtime = PyRuntimeModule::new();
        let _ch = runtime.chan(10);
    }

    #[test]
    fn go_map_with_out_channel() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            use crate::buffer::inner::{Buffer, DType};
            use crate::ir::compiler::ArgValue;
            use crate::ir::expr::Expr;
            use crate::ir::kernel::{ArgSpec, KernelKind, KernelSpec, TensorSpec};
            use crate::python::py_kernel::{PyKernelSpec, PyMapSpec};
            use std::collections::HashMap;

            let inner_spec = KernelSpec {
                kind: KernelKind::Elementwise,
                args: vec![ArgSpec {
                    name: "x".to_string(),
                    dtype: DType::F64,
                    is_scalar: false,
                }],
                output: TensorSpec { dtype: DType::F64 },
                expr: Expr::ArgRef(0),
            };
            let spec = PyKernelSpec { inner: inner_spec };
            let mut args = HashMap::new();
            args.insert(
                "x".to_string(),
                ArgValue::Buffer(Buffer::from_f64_vec(vec![10.0, 20.0])),
            );
            let map_spec = PyMapSpec { spec, args };
            let py_map = map_spec.into_pyobject(py).unwrap();

            let runtime = PyRuntimeModule::new();
            let ch = PyChannel::new(10);
            let task = runtime.go(py, py_map.as_any(), Some(ch.clone())).unwrap();

            // Result delivered to channel.
            let received = ch.inner.recv().unwrap();
            assert_eq!(received.as_f64_slice(), &[10.0, 20.0]);

            // Task should be completed.
            assert!(task.inner.is_done());

            // task.result() returns the staged buffer.
            let result = task.inner.result().unwrap();
            assert_eq!(result.as_buffer().unwrap().as_f64_slice(), &[10.0, 20.0]);
        });
    }

    #[test]
    fn go_reduce_with_out_channel() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            use crate::ir::kernel::ReduceOp;
            use crate::python::py_kernel::PyReduceSpec;

            let reduce_spec = PyReduceSpec {
                op: ReduceOp::Sum,
                buffer: crate::buffer::inner::Buffer::from_f64_vec(vec![1.0, 2.0, 3.0]),
            };
            let py_reduce = reduce_spec.into_pyobject(py).unwrap();

            let runtime = PyRuntimeModule::new();
            let ch = PyChannel::new(10);
            let task = runtime
                .go(py, py_reduce.as_any(), Some(ch.clone()))
                .unwrap();

            // Buffer wrapping the scalar delivered to channel.
            let received = ch.inner.recv().unwrap();
            assert_eq!(received.as_f64_slice(), &[6.0]);

            // Task completed with Scalar result.
            let result = task.inner.result().unwrap();
            assert_eq!(result.as_scalar().unwrap(), 6.0);
        });
    }

    #[test]
    fn mutation_guard_py_runtime_go_map_why_boundary_execution_must_return_completed_buffer_task() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            use crate::buffer::inner::{Buffer, DType};
            use crate::ir::compiler::ArgValue;
            use crate::ir::expr::Expr;
            use crate::ir::kernel::{ArgSpec, KernelKind, KernelSpec, TensorSpec};
            use crate::python::py_kernel::{PyKernelSpec, PyMapSpec};
            use std::collections::HashMap;

            let spec = PyKernelSpec {
                inner: KernelSpec {
                    kind: KernelKind::Elementwise,
                    args: vec![ArgSpec {
                        name: "x".to_string(),
                        dtype: DType::F64,
                        is_scalar: false,
                    }],
                    output: TensorSpec { dtype: DType::F64 },
                    expr: Expr::ArgRef(0),
                },
            };
            let mut args = HashMap::new();
            args.insert(
                "x".to_string(),
                ArgValue::Buffer(Buffer::from_f64_vec(vec![2.0, 4.0, 8.0])),
            );
            let task = PyRuntimeModule::new()
                .go(
                    py,
                    PyMapSpec { spec, args }.into_pyobject(py).unwrap().as_any(),
                    None,
                )
                .unwrap();
            assert!(task.inner.is_done());
            let result = task.inner.result().unwrap();

            assert_eq!(result.as_buffer().unwrap().as_f64_slice(), &[2.0, 4.0, 8.0]);
        });
    }

    #[test]
    fn mutation_guard_py_runtime_chan_and_asarray_why_public_constructors_must_keep_data_intact() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            use numpy::PyArrayMethods;

            let runtime = PyRuntimeModule::new();
            let array = numpy::PyArray1::from_vec(py, vec![1.0, 3.0, 9.0]);
            let buf = runtime.asarray(array.readonly());
            let channel = runtime.chan(1);

            assert_eq!(buf.inner.as_f64_slice(), &[1.0, 3.0, 9.0]);
            channel.inner.send(buf.inner.clone()).unwrap();
            let received = channel.inner.recv().unwrap();
            assert_eq!(received.as_f64_slice(), &[1.0, 3.0, 9.0]);
        });
    }
}
