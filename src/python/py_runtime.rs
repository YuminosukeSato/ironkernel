use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::buffer::inner::Buffer;
use crate::channel::bounded::Channel;
use crate::runtime::delivery::{submit_delivery, DeliveryJob};
use crate::runtime::task::{TaskHandle, TaskResult};

use super::py_buffer::{asarray_from_numpy, PyBuffer};
use super::py_callable::CallableTask;
use super::py_channel::PyChannel;
use super::py_kernel::{execute_map, execute_reduce, PyMapSpec, PyReduceSpec};
use super::py_task::PyTaskHandle;

fn try_convert_to_buffer(value: &Bound<'_, PyAny>) -> PyResult<Buffer> {
    if let Ok(buffer) = value.extract::<PyBuffer>() {
        return Ok(buffer.inner);
    }
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(asarray_from_numpy(array));
    }
    if let Ok(values) = value.extract::<Vec<f64>>() {
        return Ok(Buffer::from_f64_vec(values));
    }
    if let Ok(value) = value.extract::<f64>() {
        return Ok(Buffer::from_f64_vec(vec![value]));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "callable result is not buffer-convertible",
    ))
}

fn reject_callable_args_for_spec(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if args.is_empty() && kwargs.is_none() {
        return Ok(());
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "go() only accepts *args and **kwargs for Python callables",
    ))
}

enum CallableOutcome {
    Cancelled,
    Ready {
        value: Py<PyAny>,
        delivery_buffer: Option<Buffer>,
    },
    Failed(String),
}

fn execute_callable_outcome(
    background: &CallableTask,
    callable: &Py<PyAny>,
    call_args: &Py<PyTuple>,
    call_kwargs: Option<&Py<PyDict>>,
    wants_delivery: bool,
) -> CallableOutcome {
    Python::with_gil(|py| {
        if background.is_done() {
            return CallableOutcome::Cancelled;
        }

        let kwargs = call_kwargs.map(|value| value.bind(py));
        let value = match callable.bind(py).call(call_args.bind(py), kwargs) {
            Ok(value) => value,
            Err(err) => return CallableOutcome::Failed(err.to_string()),
        };

        if background.is_done() {
            return CallableOutcome::Cancelled;
        }

        let delivery_buffer = if wants_delivery {
            match try_convert_to_buffer(value.as_any()) {
                Ok(buffer) => Some(buffer),
                Err(err) => return CallableOutcome::Failed(err.to_string()),
            }
        } else {
            None
        };

        CallableOutcome::Ready {
            value: value.unbind(),
            delivery_buffer,
        }
    })
}

fn finalize_callable_outcome(
    background: &CallableTask,
    out_channel: Option<Channel>,
    outcome: CallableOutcome,
) {
    match outcome {
        CallableOutcome::Cancelled => {}
        CallableOutcome::Failed(message) => {
            if !background.is_done() {
                background.fail(message);
            }
        }
        CallableOutcome::Ready {
            value,
            delivery_buffer,
        } => {
            if background.is_done() {
                return;
            }
            if let Some(channel) = out_channel {
                match channel.send(delivery_buffer.expect("buffer required for delivery")) {
                    Ok(()) => background.complete(value),
                    Err(err) => background.fail(err.to_string()),
                }
            } else {
                background.complete(value);
            }
        }
    }
}

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
    #[pyo3(signature = (spec, *args, out=None, **kwargs))]
    fn go(
        &self,
        py: Python<'_>,
        spec: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        out: Option<PyChannel>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyTaskHandle> {
        let out_channel = out.map(|c| c.inner);

        if let Ok(map_spec) = spec.extract::<PyMapSpec>() {
            reject_callable_args_for_spec(args, kwargs)?;
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
            reject_callable_args_for_spec(args, kwargs)?;
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
        } else if spec.is_callable() {
            let handle = CallableTask::new();
            let background = handle.clone();
            let callable = spec.clone().unbind();
            let call_args = args.clone().unbind();
            let call_kwargs = kwargs.map(|value| value.clone().unbind());
            let wants_delivery = out_channel.is_some();

            rayon::spawn(move || {
                if background.is_done() {
                    return;
                }

                let outcome = execute_callable_outcome(
                    &background,
                    &callable,
                    &call_args,
                    call_kwargs.as_ref(),
                    wants_delivery,
                );
                finalize_callable_outcome(&background, out_channel, outcome);
            });

            Ok(PyTaskHandle::from_callable(handle))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "go() expects a MapSpec, ReduceSpec, or callable",
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
    use pyo3::exceptions::{PyRuntimeError, PyTypeError};
    use std::ffi::CString;
    use std::thread;
    use std::time::Duration;

    fn empty_args<'py>(py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::empty(py)
    }

    fn eval_callable<'py>(py: Python<'py>, source: &str) -> Bound<'py, PyAny> {
        let source = CString::new(source).unwrap();
        py.eval(source.as_c_str(), None, None).unwrap()
    }

    fn unbound_callable_and_empty_args(source: &str) -> (Py<PyAny>, Py<PyTuple>) {
        Python::with_gil(|py| (eval_callable(py, source).unbind(), empty_args(py).unbind()))
    }

    #[test]
    fn go_rejects_non_spec_input_why_boundary_type_contract_must_fail_fast() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let runtime = PyRuntimeModule::new();
            let invalid_spec = 123i64.into_pyobject(py).unwrap();
            let args = empty_args(py);
            let result = runtime.go(py, invalid_spec.as_any(), &args, None, None);
            assert!(result.is_err());
            let err = result.err().unwrap();

            assert!(err.is_instance_of::<PyTypeError>(py));
            assert!(err.to_string().contains("MapSpec, ReduceSpec, or callable"));
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
            let call_args = empty_args(py);
            let task = PyRuntimeModule::new()
                .go(py, py_map.as_any(), &call_args, None, None)
                .unwrap();
            let err = task.inner.result().unwrap_err();

            assert!(matches!(err, crate::error::ParsecError::ArgError(_)));
        });
    }

    #[test]
    fn go_map_spec_rejects_callable_args_why_native_spec_execution_must_keep_its_existing_contract()
    {
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
                ArgValue::Buffer(Buffer::from_f64_vec(vec![1.0, 2.0])),
            );

            let py_map = PyMapSpec { spec, args }.into_pyobject(py).unwrap();
            let extra_args = PyTuple::new(py, [1_i64]).unwrap();

            let result = PyRuntimeModule::new().go(py, py_map.as_any(), &extra_args, None, None);
            assert!(result.is_err());
            let err = result.err().unwrap();

            assert!(err.is_instance_of::<PyTypeError>(py));
            assert!(err.to_string().contains("only accepts *args and **kwargs"));
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
            let call_args = empty_args(py);
            let task = PyRuntimeModule::new()
                .go(py, py_reduce.as_any(), &call_args, None, None)
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
    fn try_convert_to_buffer_accepts_buffer_scalar_list_and_numpy_why_callable_out_channel_must_cover_supported_shapes(
    ) {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let buffer = PyBuffer::new(Buffer::from_f64_vec(vec![1.0, 2.0]));
            let scalar = 3.5_f64.into_pyobject(py).unwrap();
            let list = vec![5.0_f64, 8.0_f64].into_pyobject(py).unwrap();
            let array = numpy::PyArray1::from_vec(py, vec![13.0_f64, 21.0_f64]);

            assert_eq!(
                try_convert_to_buffer(buffer.into_pyobject(py).unwrap().as_any())
                    .unwrap()
                    .as_f64_slice(),
                &[1.0, 2.0]
            );
            assert_eq!(
                try_convert_to_buffer(scalar.as_any())
                    .unwrap()
                    .as_f64_slice(),
                &[3.5]
            );
            assert_eq!(
                try_convert_to_buffer(list.as_any()).unwrap().as_f64_slice(),
                &[5.0, 8.0]
            );
            assert_eq!(
                try_convert_to_buffer(array.into_pyobject(py).unwrap().as_any())
                    .unwrap()
                    .as_f64_slice(),
                &[13.0, 21.0]
            );
        });
    }

    #[test]
    fn try_convert_to_buffer_rejects_unsupported_python_objects_why_channel_payload_type_errors_must_fail_fast(
    ) {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("bad", 1_i64).unwrap();

            let err = try_convert_to_buffer(dict.as_any()).unwrap_err();

            assert!(err.is_instance_of::<PyTypeError>(py));
            assert!(err.to_string().contains("buffer-convertible"));
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
            let call_args = empty_args(py);
            let task = runtime
                .go(py, py_map.as_any(), &call_args, None, None)
                .unwrap();
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
            let call_args = empty_args(py);
            let task = runtime
                .go(py, py_reduce.as_any(), &call_args, None, None)
                .unwrap();
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
            let call_args = empty_args(py);
            let task = runtime
                .go(py, py_map.as_any(), &call_args, Some(ch.clone()), None)
                .unwrap();

            // Result delivered to channel.
            let received = ch.inner.recv().unwrap();
            assert_eq!(received.as_f64_slice(), &[10.0, 20.0]);

            // task.result() waits for completion and returns the staged buffer.
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
            let call_args = empty_args(py);
            let task = runtime
                .go(py, py_reduce.as_any(), &call_args, Some(ch.clone()), None)
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
                    &empty_args(py),
                    None,
                    None,
                )
                .unwrap();
            assert!(task.inner.is_done());
            let result = task.inner.result().unwrap();

            assert_eq!(result.as_buffer().unwrap().as_f64_slice(), &[2.0, 4.0, 8.0]);
        });
    }

    #[test]
    fn go_callable_returns_immediately_why_submit_must_not_block_on_python_execution() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let runtime = PyRuntimeModule::new();
            let callable = eval_callable(py, "lambda: (__import__('time').sleep(0.05), 'done')[1]");
            let call_args = empty_args(py);

            let task = runtime
                .go(py, callable.as_any(), &call_args, None, None)
                .unwrap();

            assert!(!task.done());
            assert_eq!(
                task.result_object(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                "done"
            );
        });
    }

    #[test]
    fn go_callable_accepts_args_and_kwargs_why_runtime_must_match_threadpool_submit_ergonomics() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let runtime = PyRuntimeModule::new();
            let callable = eval_callable(py, "lambda x, y=0, scale=1: (x + y) * scale");
            let call_args = PyTuple::new(py, [3_i64, 4_i64]).unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("scale", 2_i64).unwrap();

            let task = runtime
                .go(py, callable.as_any(), &call_args, None, Some(&kwargs))
                .unwrap();

            assert_eq!(
                task.result_object(py).unwrap().extract::<i64>(py).unwrap(),
                14
            );
        });
    }

    #[test]
    fn go_callable_with_out_channel_converts_scalar_why_callable_results_must_enter_buffer_channels(
    ) {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let runtime = PyRuntimeModule::new();
            let callable = eval_callable(py, "lambda value: value * 2.0");
            let call_args = PyTuple::new(py, [3.5_f64]).unwrap();
            let channel = PyChannel::new(1);

            let task = runtime
                .go(
                    py,
                    callable.as_any(),
                    &call_args,
                    Some(channel.clone()),
                    None,
                )
                .unwrap();
            let received = py.allow_threads(|| channel.inner.recv()).unwrap();

            assert_eq!(received.as_f64_slice(), &[7.0]);
            assert_eq!(
                task.result_object(py).unwrap().extract::<f64>(py).unwrap(),
                7.0
            );
        });
    }

    #[test]
    fn go_callable_with_out_channel_rejects_non_buffer_result_why_channel_contract_must_stay_buffer_only(
    ) {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let runtime = PyRuntimeModule::new();
            let callable = eval_callable(py, "lambda: {'bad': 1}");
            let call_args = empty_args(py);
            let channel = PyChannel::new(1);

            let task = runtime
                .go(py, callable.as_any(), &call_args, Some(channel), None)
                .unwrap();
            let err = task.result_object(py).unwrap_err();

            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(err.to_string().contains("buffer-convertible"));
        });
    }

    #[test]
    fn execute_callable_outcome_returns_cancelled_before_python_call_when_task_was_already_cancelled_why_gil_handoff_races_must_not_run_user_code(
    ) {
        pyo3::prepare_freethreaded_python();

        let task = CallableTask::new();
        let (callable, args) = unbound_callable_and_empty_args("lambda: 1");

        assert!(task.cancel());
        assert!(matches!(
            execute_callable_outcome(&task, &callable, &args, None, false),
            CallableOutcome::Cancelled
        ));
    }

    #[test]
    fn execute_callable_outcome_returns_cancelled_after_python_call_when_cancellation_wins_mid_execution_why_completed_python_work_must_not_override_task_state(
    ) {
        pyo3::prepare_freethreaded_python();

        let task = CallableTask::new();
        let background = task.clone();
        let (callable, args) =
            unbound_callable_and_empty_args("lambda: (__import__('time').sleep(0.05), 1)[1]");
        let join = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            assert!(background.cancel());
        });

        assert!(matches!(
            execute_callable_outcome(&task, &callable, &args, None, false),
            CallableOutcome::Cancelled
        ));
        join.join().unwrap();
    }

    #[test]
    fn finalize_callable_outcome_keeps_cancelled_task_terminal_why_cancelled_outcomes_must_not_reopen_waiters(
    ) {
        pyo3::prepare_freethreaded_python();

        let task = CallableTask::new();
        assert!(task.cancel());

        finalize_callable_outcome(&task, None, CallableOutcome::Cancelled);

        Python::with_gil(|py| {
            let err = task.result(py).unwrap_err();
            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(err.to_string().contains("cancelled"));
        });
    }

    #[test]
    fn finalize_callable_outcome_skips_ready_value_when_task_was_cancelled_why_delivery_must_not_revive_cancelled_tasks(
    ) {
        pyo3::prepare_freethreaded_python();

        let task = CallableTask::new();
        assert!(task.cancel());
        let value = Python::with_gil(|py| 5_i64.into_pyobject(py).unwrap().unbind().into_any());

        finalize_callable_outcome(
            &task,
            None,
            CallableOutcome::Ready {
                value,
                delivery_buffer: None,
            },
        );

        Python::with_gil(|py| {
            let err = task.result(py).unwrap_err();
            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(err.to_string().contains("cancelled"));
        });
    }

    #[test]
    fn finalize_callable_outcome_fails_when_out_channel_is_closed_why_delivery_errors_must_reach_waiters(
    ) {
        pyo3::prepare_freethreaded_python();

        let task = CallableTask::new();
        let channel = crate::channel::bounded::Channel::new(1);
        let value = Python::with_gil(|py| 9_i64.into_pyobject(py).unwrap().unbind().into_any());
        channel.close();

        finalize_callable_outcome(
            &task,
            Some(channel),
            CallableOutcome::Ready {
                value,
                delivery_buffer: Some(Buffer::from_f64_vec(vec![9.0])),
            },
        );

        Python::with_gil(|py| {
            let err = task.result(py).unwrap_err();
            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(err.to_string().contains("ChannelClosed"));
        });
    }

    #[test]
    fn go_callable_exception_surfaces_via_task_result_why_background_python_errors_must_not_disappear(
    ) {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let runtime = PyRuntimeModule::new();
            let callable = eval_callable(
                py,
                "lambda: (_ for _ in ()).throw(RuntimeError('callable boom'))",
            );
            let call_args = empty_args(py);

            let task = runtime
                .go(py, callable.as_any(), &call_args, None, None)
                .unwrap();
            let err = task.result_object(py).unwrap_err();

            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(err.to_string().contains("callable boom"));
        });
    }

    #[test]
    fn go_callable_cancel_marks_task_cancelled_why_cooperative_stop_must_work_for_long_running_work(
    ) {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let runtime = PyRuntimeModule::new();
            let callable = eval_callable(py, "lambda: (__import__('time').sleep(0.05), 1)[1]");
            let call_args = empty_args(py);

            let task = runtime
                .go(py, callable.as_any(), &call_args, None, None)
                .unwrap();

            assert!(task.cancel_inner());
            let err = task.result_object(py).unwrap_err();
            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(err.to_string().contains("cancelled"));
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
