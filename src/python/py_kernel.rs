use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use pyo3::prelude::*;

use crate::buffer::inner::{Buffer, DType};
use crate::ir::compiler::{eval_elementwise, eval_reduce, ArgValue};
use crate::ir::expr::{BinaryOp, Expr, UnaryOp};
use crate::ir::kernel::{ArgSpec, KernelKind, KernelSpec, ReduceOp};

use super::py_buffer::PyBuffer;
use super::py_expr::{coerce_to_coerced, PyExpr};

/// Global counter for unique `temp_id` generation.
static NEXT_TEMP_ID: AtomicUsize = AtomicUsize::new(0);

fn next_temp_id() -> usize {
    NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
}

/// Python-visible kernel specification.
#[pyclass(name = "KernelSpec")]
#[derive(Clone)]
pub struct PyKernelSpec {
    pub(crate) inner: KernelSpec,
}

/// The kernel module exposed to Python.
/// Contains factory functions for creating expressions and kernel specs.
#[pyclass(name = "_KernelModule")]
pub struct PyKernelModule;

#[pymethods]
impl PyKernelModule {
    #[new]
    fn new() -> Self {
        PyKernelModule
    }

    /// Create a single named argument expression.
    #[pyo3(signature = (name))]
    fn arg(&self, name: &str) -> PyExpr {
        let temp_id = next_temp_id();
        PyExpr::named(Expr::ArgRef(temp_id), name.to_string(), temp_id)
    }

    /// Create multiple named argument expressions.
    #[pyo3(signature = (*names))]
    fn args(&self, names: Vec<String>) -> Vec<PyExpr> {
        names
            .into_iter()
            .map(|name| {
                let temp_id = next_temp_id();
                PyExpr::named(Expr::ArgRef(temp_id), name, temp_id)
            })
            .collect()
    }

    /// Create an elementwise kernel from an expression.
    /// Performs dense reindexing: `temp_ids` are mapped to 0, 1, 2, ... in sorted order.
    fn elementwise(&self, expr: PyExpr) -> PyResult<PyKernelSpec> {
        // Collect temp_ids referenced in the expression, sorted
        let referenced_temp_ids = expr.inner.referenced_args();
        let mut seen_names = HashSet::new();

        // Build dense mapping: temp_id -> dense_index
        let mapping: HashMap<usize, usize> = referenced_temp_ids
            .iter()
            .enumerate()
            .map(|(dense_idx, &temp_id)| (temp_id, dense_idx))
            .collect();

        // Rewrite ArgRefs to dense indices
        let rewritten_expr = expr.inner.rewrite_arg_refs(&mapping);

        // Build ArgSpecs with names from temp_args
        let arg_specs: Vec<ArgSpec> = referenced_temp_ids
            .iter()
            .map(|temp_id| -> PyResult<ArgSpec> {
                let name = expr.temp_args.get(temp_id).cloned().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "missing temp arg metadata for arg index {temp_id}"
                    ))
                })?;

                if !seen_names.insert(name.clone()) {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "duplicate arg name: {name}"
                    )));
                }

                Ok(ArgSpec {
                    name,
                    dtype: DType::F64,
                    is_scalar: false,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        let inner = KernelSpec {
            kind: KernelKind::Elementwise,
            args: arg_specs,
            output: crate::ir::kernel::TensorSpec { dtype: DType::F64 },
            expr: rewritten_expr,
        };
        Ok(PyKernelSpec { inner })
    }

    /// Map a kernel spec over named arguments, returning a `MapSpec` for `rt.go()`.
    #[pyo3(signature = (spec, **kwargs))]
    fn map(
        &self,
        spec: &PyKernelSpec,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyMapSpec> {
        let mut arg_map: HashMap<String, ArgValue> = HashMap::new();
        if let Some(kw) = kwargs {
            for (key, val) in kw {
                let name: String = key.extract()?;
                if let Ok(buf) = val.extract::<PyBuffer>() {
                    arg_map.insert(name, ArgValue::Buffer(buf.inner));
                } else if let Ok(v) = val.extract::<i64>() {
                    arg_map.insert(name, ArgValue::Scalar(v as f64));
                } else if let Ok(v) = val.extract::<f64>() {
                    arg_map.insert(name, ArgValue::Scalar(v));
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "argument '{name}' must be Buffer or numeric"
                    )));
                }
            }
        }
        Ok(PyMapSpec {
            spec: spec.clone(),
            args: arg_map,
        })
    }

    // --- Math functions (unary: inherit temp_args) ---

    fn sqrt(&self, expr: PyExpr) -> PyExpr {
        PyExpr::unary_with_temp_args(UnaryOp::Sqrt, &expr.inner, &expr.temp_args)
    }

    fn abs(&self, expr: PyExpr) -> PyExpr {
        PyExpr::unary_with_temp_args(UnaryOp::Abs, &expr.inner, &expr.temp_args)
    }

    fn log(&self, expr: PyExpr) -> PyExpr {
        PyExpr::unary_with_temp_args(UnaryOp::Log, &expr.inner, &expr.temp_args)
    }

    fn exp(&self, expr: PyExpr) -> PyExpr {
        PyExpr::unary_with_temp_args(UnaryOp::Exp, &expr.inner, &expr.temp_args)
    }

    fn log2(&self, expr: PyExpr) -> PyExpr {
        PyExpr::unary_with_temp_args(UnaryOp::Log2, &expr.inner, &expr.temp_args)
    }

    fn log10(&self, expr: PyExpr) -> PyExpr {
        PyExpr::unary_with_temp_args(UnaryOp::Log10, &expr.inner, &expr.temp_args)
    }

    fn floor(&self, expr: PyExpr) -> PyExpr {
        PyExpr::unary_with_temp_args(UnaryOp::Floor, &expr.inner, &expr.temp_args)
    }

    fn ceil(&self, expr: PyExpr) -> PyExpr {
        PyExpr::unary_with_temp_args(UnaryOp::Ceil, &expr.inner, &expr.temp_args)
    }

    fn round(&self, expr: PyExpr) -> PyExpr {
        PyExpr::unary_with_temp_args(UnaryOp::Round, &expr.inner, &expr.temp_args)
    }

    fn sin(&self, expr: PyExpr) -> PyExpr {
        PyExpr::unary_with_temp_args(UnaryOp::Sin, &expr.inner, &expr.temp_args)
    }

    fn cos(&self, expr: PyExpr) -> PyExpr {
        PyExpr::unary_with_temp_args(UnaryOp::Cos, &expr.inner, &expr.temp_args)
    }

    fn tan(&self, expr: PyExpr) -> PyExpr {
        PyExpr::unary_with_temp_args(UnaryOp::Tan, &expr.inner, &expr.temp_args)
    }

    // --- Math functions (binary: merge temp_args) ---

    fn pow(&self, base: PyExpr, exp: PyExpr) -> PyExpr {
        PyExpr::binop_with_temp_args(
            BinaryOp::Pow,
            &base.inner,
            &exp.inner,
            &base.temp_args,
            &exp.temp_args,
        )
    }

    fn atan2(&self, y: PyExpr, x: PyExpr) -> PyExpr {
        PyExpr::binop_with_temp_args(
            BinaryOp::Atan2,
            &y.inner,
            &x.inner,
            &y.temp_args,
            &x.temp_args,
        )
    }

    fn min(&self, a: PyExpr, b: PyExpr) -> PyExpr {
        PyExpr::binop_with_temp_args(
            BinaryOp::Min,
            &a.inner,
            &b.inner,
            &a.temp_args,
            &b.temp_args,
        )
    }

    fn max(&self, a: PyExpr, b: PyExpr) -> PyExpr {
        PyExpr::binop_with_temp_args(
            BinaryOp::Max,
            &a.inner,
            &b.inner,
            &a.temp_args,
            &b.temp_args,
        )
    }

    /// Conditional select: `where(cond, true_val, false_val)`.
    #[pyo3(name = "where_")]
    fn where_expr(
        &self,
        cond: PyExpr,
        true_val: &Bound<'_, PyAny>,
        false_val: &Bound<'_, PyAny>,
    ) -> PyResult<PyExpr> {
        let t = coerce_to_coerced(true_val)?;
        let f = coerce_to_coerced(false_val)?;
        let mut merged = cond.temp_args.clone();
        merged.extend(t.temp_args.iter().map(|(k, v)| (*k, v.clone())));
        merged.extend(f.temp_args.iter().map(|(k, v)| (*k, v.clone())));
        Ok(PyExpr::from_parts(
            Expr::Select(Box::new(cond.inner), Box::new(t.inner), Box::new(f.inner)),
            merged,
        ))
    }

    // --- Reduce ---

    fn sum(&self, buf: PyBuffer) -> PyReduceSpec {
        PyReduceSpec {
            op: ReduceOp::Sum,
            buffer: buf.inner,
        }
    }

    fn mean(&self, buf: PyBuffer) -> PyReduceSpec {
        PyReduceSpec {
            op: ReduceOp::Mean,
            buffer: buf.inner,
        }
    }

    #[pyo3(name = "min_reduce")]
    fn min_reduce(&self, buf: PyBuffer) -> PyReduceSpec {
        PyReduceSpec {
            op: ReduceOp::Min,
            buffer: buf.inner,
        }
    }

    #[pyo3(name = "max_reduce")]
    fn max_reduce(&self, buf: PyBuffer) -> PyReduceSpec {
        PyReduceSpec {
            op: ReduceOp::Max,
            buffer: buf.inner,
        }
    }
}

/// Specification for mapping a kernel over arguments.
#[pyclass(name = "MapSpec")]
#[derive(Clone)]
pub struct PyMapSpec {
    pub(crate) spec: PyKernelSpec,
    pub(crate) args: HashMap<String, ArgValue>,
}

/// Specification for a reduce operation.
#[pyclass(name = "ReduceSpec")]
#[derive(Clone)]
pub struct PyReduceSpec {
    pub(crate) op: ReduceOp,
    pub(crate) buffer: Buffer,
}

/// Execute a map spec: resolve named args to positional via KernelSpec.args[idx].name.
pub fn execute_map(map_spec: &PyMapSpec) -> crate::error::ParsecResult<Buffer> {
    let spec = &map_spec.spec.inner;
    let expr = &spec.expr;
    let mut missing: Vec<String> = spec
        .args
        .iter()
        .filter(|arg| !map_spec.args.contains_key(&arg.name))
        .map(|arg| arg.name.clone())
        .collect();
    missing.sort();
    if !missing.is_empty() {
        let message = if missing.len() == 1 {
            format!("missing arg: {}", missing[0])
        } else {
            format!("missing args: {}", missing.join(", "))
        };
        return Err(crate::error::ParsecError::ArgError(message));
    }

    let expected_names: HashSet<&str> = spec.args.iter().map(|arg| arg.name.as_str()).collect();
    let mut extra: Vec<String> = map_spec
        .args
        .keys()
        .filter(|name| !expected_names.contains(name.as_str()))
        .cloned()
        .collect();
    extra.sort();
    if !extra.is_empty() {
        return Err(crate::error::ParsecError::ArgError(format!(
            "unexpected args: {}",
            extra.join(", ")
        )));
    }

    let mut positional: Vec<ArgValue> = Vec::with_capacity(spec.args.len());
    positional.extend(spec.args.iter().map(|arg| map_spec.args[&arg.name].clone()));

    eval_elementwise(expr, &positional)
}

/// Execute a reduce spec.
pub fn execute_reduce(spec: &PyReduceSpec) -> crate::error::ParsecResult<f64> {
    eval_reduce(spec.op, &spec.buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- R1.3: arg() produces unique temp_ids ---

    #[test]
    fn arg_returns_unique_temp_ids() {
        let km = PyKernelModule::new();
        let x = km.arg("x");
        let y = km.arg("y");
        // temp_ids must differ
        let x_id = *x.temp_args.keys().next().unwrap();
        let y_id = *y.temp_args.keys().next().unwrap();
        assert_ne!(x_id, y_id);
    }

    #[test]
    fn arg_inner_matches_temp_id() {
        let km = PyKernelModule::new();
        let x = km.arg("x");
        let temp_id = *x.temp_args.keys().next().unwrap();
        assert_eq!(x.inner, Expr::ArgRef(temp_id));
    }

    #[test]
    fn args_returns_correct_count() {
        let km = PyKernelModule::new();
        let exprs = km.args(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(exprs.len(), 3);
    }

    #[test]
    fn args_all_unique_temp_ids() {
        let km = PyKernelModule::new();
        let exprs = km.args(vec!["a".into(), "b".into(), "c".into()]);
        let ids: Vec<usize> = exprs
            .iter()
            .map(|e| *e.temp_args.keys().next().unwrap())
            .collect();
        let mut deduped = ids.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(ids.len(), deduped.len());
    }

    // --- R1.6: elementwise dense reindex ---

    #[test]
    fn elementwise_dense_reindex() {
        let km = PyKernelModule::new();
        let x = km.arg("x");
        let y = km.arg("y");
        // x + y
        let expr = PyExpr::binop_with_temp_args(
            BinaryOp::Add,
            &x.inner,
            &y.inner,
            &x.temp_args,
            &y.temp_args,
        );
        let spec = km.elementwise(expr).unwrap();
        // After dense reindex, args should be [0, 1]
        assert_eq!(spec.inner.args.len(), 2);
        assert_eq!(spec.inner.args[0].name, "x");
        assert_eq!(spec.inner.args[1].name, "y");
        assert_eq!(spec.inner.expr.referenced_args(), vec![0, 1]);
    }

    #[test]
    fn elementwise_single_arg() {
        let km = PyKernelModule::new();
        let x = km.arg("x");
        let expr = PyExpr::unary_with_temp_args(UnaryOp::Neg, &x.inner, &x.temp_args);
        let spec = km.elementwise(expr).unwrap();
        assert_eq!(spec.inner.args.len(), 1);
        assert_eq!(spec.inner.args[0].name, "x");
        assert_eq!(spec.inner.expr.referenced_args(), vec![0]);
    }

    #[test]
    fn elementwise_dedup_same_arg() {
        let km = PyKernelModule::new();
        let x = km.arg("x");
        // x + x (same arg used twice)
        let expr = PyExpr::binop_with_temp_args(
            BinaryOp::Add,
            &x.inner,
            &x.inner,
            &x.temp_args,
            &x.temp_args,
        );
        let spec = km.elementwise(expr).unwrap();
        assert_eq!(spec.inner.args.len(), 1);
        assert_eq!(spec.inner.args[0].name, "x");
    }

    #[test]
    fn elementwise_three_args_order() {
        let km = PyKernelModule::new();
        let a = km.arg("a");
        let b = km.arg("b");
        let c = km.arg("c");
        // (a + b) * c
        let ab = PyExpr::binop_with_temp_args(
            BinaryOp::Add,
            &a.inner,
            &b.inner,
            &a.temp_args,
            &b.temp_args,
        );
        let expr = PyExpr::binop_with_temp_args(
            BinaryOp::Mul,
            &ab.inner,
            &c.inner,
            &ab.temp_args,
            &c.temp_args,
        );
        let spec = km.elementwise(expr).unwrap();
        assert_eq!(spec.inner.args.len(), 3);
        // Names should be in temp_id sorted order
        assert_eq!(spec.inner.args[0].name, "a");
        assert_eq!(spec.inner.args[1].name, "b");
        assert_eq!(spec.inner.args[2].name, "c");
    }

    // --- R1.7: execute_map name-based binding ---

    #[test]
    fn execute_map_name_based() {
        let km = PyKernelModule::new();
        let x = km.arg("x");
        let y = km.arg("y");
        // x + y
        let expr = PyExpr::binop_with_temp_args(
            BinaryOp::Add,
            &x.inner,
            &y.inner,
            &x.temp_args,
            &y.temp_args,
        );
        let spec = km.elementwise(expr).unwrap();

        // Create MapSpec manually with named args
        let mut args = HashMap::new();
        args.insert(
            "x".to_string(),
            ArgValue::Buffer(Buffer::from_f64_vec(vec![1.0, 2.0, 3.0])),
        );
        args.insert(
            "y".to_string(),
            ArgValue::Buffer(Buffer::from_f64_vec(vec![10.0, 20.0, 30.0])),
        );
        let map_spec = PyMapSpec { spec, args };

        let result = execute_map(&map_spec).unwrap();
        let data = result.as_f64_slice();
        assert_eq!(data, &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn execute_map_missing_arg() {
        let km = PyKernelModule::new();
        let x = km.arg("x");
        let spec = km.elementwise(x).unwrap();

        let args = HashMap::new(); // no args provided
        let map_spec = PyMapSpec { spec, args };
        let result = execute_map(&map_spec);
        assert!(result.is_err());
    }

    #[test]
    fn execute_map_multiple_missing_args_are_sorted_why_arg_validation_must_report_the_full_contract_gap(
    ) {
        let km = PyKernelModule::new();
        let x = km.arg("x");
        let y = km.arg("y");
        let expr = PyExpr::binop_with_temp_args(
            BinaryOp::Add,
            &x.inner,
            &y.inner,
            &x.temp_args,
            &y.temp_args,
        );
        let spec = km.elementwise(expr).unwrap();

        let err = execute_map(&PyMapSpec {
            spec,
            args: HashMap::new(),
        })
        .unwrap_err();

        assert_eq!(
            err,
            crate::error::ParsecError::ArgError("missing args: x, y".into())
        );
    }

    // --- Math functions: temp_args propagation ---

    #[test]
    fn math_unary_inherits_temp_args() {
        let km = PyKernelModule::new();
        let x = km.arg("x");
        let temp_id = *x.temp_args.keys().next().unwrap();
        let result = km.sqrt(x);
        assert_eq!(result.temp_args.len(), 1);
        assert_eq!(result.temp_args.get(&temp_id), Some(&"x".to_string()));
    }

    #[test]
    fn math_binary_merges_temp_args() {
        let km = PyKernelModule::new();
        let x = km.arg("base");
        let y = km.arg("exp");
        let result = km.pow(x.clone(), y.clone());
        assert_eq!(result.temp_args.len(), 2);
        let x_id = *x.temp_args.keys().next().unwrap();
        let y_id = *y.temp_args.keys().next().unwrap();
        assert_eq!(result.temp_args.get(&x_id), Some(&"base".to_string()));
        assert_eq!(result.temp_args.get(&y_id), Some(&"exp".to_string()));
    }

    // --- All math unary functions ---

    #[test]
    fn math_all_unary_functions() {
        let km = PyKernelModule::new();
        let funcs: Vec<fn(&PyKernelModule, PyExpr) -> PyExpr> = vec![
            PyKernelModule::sqrt,
            PyKernelModule::abs,
            PyKernelModule::log,
            PyKernelModule::exp,
            PyKernelModule::log2,
            PyKernelModule::log10,
            PyKernelModule::floor,
            PyKernelModule::ceil,
            PyKernelModule::round,
            PyKernelModule::sin,
            PyKernelModule::cos,
            PyKernelModule::tan,
        ];
        for f in funcs {
            let x = km.arg("x");
            let result = f(&km, x.clone());
            assert_eq!(result.temp_args.len(), 1);
            assert!(matches!(result.inner, Expr::Unary(_, _)));
        }
    }

    // --- Binary math functions ---

    #[test]
    fn math_atan2() {
        let km = PyKernelModule::new();
        let y = km.arg("y");
        let x = km.arg("x");
        let result = km.atan2(y, x);
        assert!(matches!(result.inner, Expr::Binary(BinaryOp::Atan2, _, _)));
    }

    #[test]
    fn math_min() {
        let km = PyKernelModule::new();
        let a = km.arg("a");
        let b = km.arg("b");
        let result = km.min(a, b);
        assert!(matches!(result.inner, Expr::Binary(BinaryOp::Min, _, _)));
    }

    #[test]
    fn math_max() {
        let km = PyKernelModule::new();
        let a = km.arg("a");
        let b = km.arg("b");
        let result = km.max(a, b);
        assert!(matches!(result.inner, Expr::Binary(BinaryOp::Max, _, _)));
    }

    // --- where_ ---

    #[test]
    fn where_expr_merges_temp_args() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let km = PyKernelModule::new();
            let cond = km.arg("cond");
            let t_val = 1.0f64.into_pyobject(py).unwrap();
            let f_val = 0.0f64.into_pyobject(py).unwrap();
            let result = km.where_expr(cond, t_val.as_any(), f_val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Select(_, _, _)));
            assert_eq!(result.temp_args.len(), 1);
        });
    }

    #[test]
    fn where_expr_with_three_args() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let km = PyKernelModule::new();
            let cond = km.arg("c");
            let t = km.arg("t");
            let f = km.arg("f");
            let t_obj = t.into_pyobject(py).unwrap();
            let f_obj = f.into_pyobject(py).unwrap();
            let result = km.where_expr(cond, t_obj.as_any(), f_obj.as_any()).unwrap();
            assert_eq!(result.temp_args.len(), 3);
        });
    }

    // --- Reduce specs ---

    #[test]
    fn reduce_sum_spec() {
        let km = PyKernelModule::new();
        let buf = PyBuffer::new(Buffer::from_f64_vec(vec![1.0, 2.0]));
        let spec = km.sum(buf);
        assert_eq!(spec.op, ReduceOp::Sum);
    }

    #[test]
    fn reduce_mean_spec() {
        let km = PyKernelModule::new();
        let buf = PyBuffer::new(Buffer::from_f64_vec(vec![1.0, 2.0]));
        let spec = km.mean(buf);
        assert_eq!(spec.op, ReduceOp::Mean);
    }

    #[test]
    fn reduce_min_spec() {
        let km = PyKernelModule::new();
        let buf = PyBuffer::new(Buffer::from_f64_vec(vec![1.0, 2.0]));
        let spec = km.min_reduce(buf);
        assert_eq!(spec.op, ReduceOp::Min);
    }

    #[test]
    fn reduce_max_spec() {
        let km = PyKernelModule::new();
        let buf = PyBuffer::new(Buffer::from_f64_vec(vec![1.0, 2.0]));
        let spec = km.max_reduce(buf);
        assert_eq!(spec.op, ReduceOp::Max);
    }

    // --- execute_reduce ---

    #[test]
    fn execute_reduce_sum() {
        let spec = PyReduceSpec {
            op: ReduceOp::Sum,
            buffer: Buffer::from_f64_vec(vec![1.0, 2.0, 3.0]),
        };
        assert_eq!(execute_reduce(&spec).unwrap(), 6.0);
    }

    // --- execute_map: extra args error ---

    #[test]
    fn execute_map_extra_arg_error() {
        let km = PyKernelModule::new();
        let x = km.arg("x");
        let spec = km.elementwise(x).unwrap();

        let mut args = HashMap::new();
        args.insert(
            "x".to_string(),
            ArgValue::Buffer(Buffer::from_f64_vec(vec![1.0])),
        );
        args.insert("extra".to_string(), ArgValue::Scalar(99.0));
        let map_spec = PyMapSpec { spec, args };
        let err = execute_map(&map_spec).unwrap_err();
        assert!(matches!(err, crate::error::ParsecError::ArgError(_)));
    }

    // --- map() via GIL ---

    #[test]
    fn map_with_kwargs() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let km = PyKernelModule::new();
            let x = km.arg("x");
            let spec = km.elementwise(x).unwrap();

            let kwargs = pyo3::types::PyDict::new(py);
            let buf = PyBuffer::new(Buffer::from_f64_vec(vec![1.0, 2.0]));
            kwargs
                .set_item("x", buf.into_pyobject(py).unwrap())
                .unwrap();
            let map_spec = km.map(&spec, Some(&kwargs.as_borrowed())).unwrap();
            assert_eq!(map_spec.args.len(), 1);
        });
    }

    #[test]
    fn map_with_scalar_kwarg() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let km = PyKernelModule::new();
            let a = km.arg("a");
            let x = km.arg("x");
            let expr = PyExpr::binop_with_temp_args(
                BinaryOp::Mul,
                &a.inner,
                &x.inner,
                &a.temp_args,
                &x.temp_args,
            );
            let spec = km.elementwise(expr).unwrap();

            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("a", 2.0f64).unwrap();
            let buf = PyBuffer::new(Buffer::from_f64_vec(vec![1.0]));
            kwargs
                .set_item("x", buf.into_pyobject(py).unwrap())
                .unwrap();
            let map_spec = km.map(&spec, Some(&kwargs.as_borrowed())).unwrap();
            assert_eq!(map_spec.args.len(), 2);
        });
    }

    #[test]
    fn map_with_int_kwarg() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let km = PyKernelModule::new();
            let a = km.arg("a");
            let x = km.arg("x");
            let expr = PyExpr::binop_with_temp_args(
                BinaryOp::Mul,
                &a.inner,
                &x.inner,
                &a.temp_args,
                &x.temp_args,
            );
            let spec = km.elementwise(expr).unwrap();

            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("a", 2i64).unwrap();
            let buf = PyBuffer::new(Buffer::from_f64_vec(vec![1.0]));
            kwargs
                .set_item("x", buf.into_pyobject(py).unwrap())
                .unwrap();
            let map_spec = km.map(&spec, Some(&kwargs.as_borrowed())).unwrap();
            assert_eq!(map_spec.args.len(), 2);
            let actual = map_spec.args.get("a");
            assert!(matches!(actual, Some(ArgValue::Scalar(v)) if *v == 2.0));
        });
    }

    #[test]
    fn map_with_invalid_kwarg_type() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let km = PyKernelModule::new();
            let x = km.arg("x");
            let spec = km.elementwise(x).unwrap();

            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("x", "invalid").unwrap();
            let err = km.map(&spec, Some(&kwargs.as_borrowed()));
            assert!(err.is_err());
        });
    }

    #[test]
    fn map_with_no_kwargs() {
        let km = PyKernelModule::new();
        let x = km.arg("x");
        let spec = km.elementwise(x).unwrap();
        let map_spec = km.map(&spec, None).unwrap();
        assert!(map_spec.args.is_empty());
    }

    // --- elementwise duplicate arg name ---

    #[test]
    fn elementwise_duplicate_name_error() {
        let km = PyKernelModule::new();
        let x1 = km.arg("x");
        let x2 = km.arg("x"); // same name, different temp_id
        let expr = PyExpr::binop_with_temp_args(
            BinaryOp::Add,
            &x1.inner,
            &x2.inner,
            &x1.temp_args,
            &x2.temp_args,
        );
        let result = km.elementwise(expr);
        assert!(result.is_err());
    }

    // --- elementwise missing temp arg metadata ---

    #[test]
    fn elementwise_missing_temp_metadata_error() {
        let km = PyKernelModule::new();
        // Create expr with ArgRef that has no temp_args entry
        let expr = PyExpr::new(Expr::ArgRef(999));
        let result = km.elementwise(expr);
        assert!(result.is_err());
    }
}
