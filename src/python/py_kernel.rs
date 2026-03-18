use std::collections::HashMap;

use pyo3::prelude::*;

use crate::buffer::inner::{Buffer, DType};
use crate::ir::compiler::{eval_elementwise, eval_reduce, ArgValue};
use crate::ir::expr::{BinaryOp, Expr, UnaryOp};
use crate::ir::kernel::{ArgSpec, KernelKind, KernelSpec, ReduceOp};

use super::py_buffer::PyBuffer;
use super::py_expr::PyExpr;

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
        PyExpr::named(Expr::ArgRef(0), name.to_string())
    }

    /// Create multiple named argument expressions.
    #[pyo3(signature = (*names))]
    fn args(&self, names: Vec<String>) -> Vec<PyExpr> {
        names
            .into_iter()
            .enumerate()
            .map(|(i, name)| PyExpr::named(Expr::ArgRef(i), name))
            .collect()
    }

    /// Create an elementwise kernel from an expression.
    fn elementwise(&self, expr: PyExpr) -> PyKernelSpec {
        let referenced = expr.inner.referenced_args();
        let arg_specs: Vec<ArgSpec> = referenced
            .iter()
            .map(|_| ArgSpec {
                name: String::new(),
                dtype: DType::F64,
                is_scalar: false,
            })
            .collect();
        let inner = KernelSpec {
            kind: KernelKind::Elementwise,
            args: arg_specs,
            output: crate::ir::kernel::TensorSpec { dtype: DType::F64 },
            expr: expr.inner,
        };
        PyKernelSpec { inner }
    }

    /// Map a kernel spec over named arguments, returning a MapSpec for rt.go().
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
                } else if let Ok(v) = val.extract::<f64>() {
                    arg_map.insert(name, ArgValue::Scalar(v));
                } else if let Ok(v) = val.extract::<i64>() {
                    arg_map.insert(name, ArgValue::Scalar(v as f64));
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

    // --- Math functions ---

    fn sqrt(&self, expr: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Unary(UnaryOp::Sqrt, Box::new(expr.inner)))
    }

    fn abs(&self, expr: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Unary(UnaryOp::Abs, Box::new(expr.inner)))
    }

    fn log(&self, expr: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Unary(UnaryOp::Log, Box::new(expr.inner)))
    }

    fn exp(&self, expr: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Unary(UnaryOp::Exp, Box::new(expr.inner)))
    }

    fn log2(&self, expr: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Unary(UnaryOp::Log2, Box::new(expr.inner)))
    }

    fn log10(&self, expr: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Unary(UnaryOp::Log10, Box::new(expr.inner)))
    }

    fn floor(&self, expr: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Unary(UnaryOp::Floor, Box::new(expr.inner)))
    }

    fn ceil(&self, expr: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Unary(UnaryOp::Ceil, Box::new(expr.inner)))
    }

    fn round(&self, expr: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Unary(UnaryOp::Round, Box::new(expr.inner)))
    }

    fn sin(&self, expr: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Unary(UnaryOp::Sin, Box::new(expr.inner)))
    }

    fn cos(&self, expr: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Unary(UnaryOp::Cos, Box::new(expr.inner)))
    }

    fn tan(&self, expr: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Unary(UnaryOp::Tan, Box::new(expr.inner)))
    }

    fn pow(&self, base: PyExpr, exp: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Binary(
            BinaryOp::Pow,
            Box::new(base.inner),
            Box::new(exp.inner),
        ))
    }

    fn atan2(&self, y: PyExpr, x: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Binary(
            BinaryOp::Atan2,
            Box::new(y.inner),
            Box::new(x.inner),
        ))
    }

    fn min(&self, a: PyExpr, b: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Binary(
            BinaryOp::Min,
            Box::new(a.inner),
            Box::new(b.inner),
        ))
    }

    fn max(&self, a: PyExpr, b: PyExpr) -> PyExpr {
        PyExpr::new(Expr::Binary(
            BinaryOp::Max,
            Box::new(a.inner),
            Box::new(b.inner),
        ))
    }

    /// Conditional select: where(cond, true_val, false_val).
    #[pyo3(name = "where_")]
    fn where_expr(
        &self,
        cond: PyExpr,
        true_val: &Bound<'_, PyAny>,
        false_val: &Bound<'_, PyAny>,
    ) -> PyResult<PyExpr> {
        let t = coerce_expr(true_val)?;
        let f = coerce_expr(false_val)?;
        Ok(PyExpr::new(Expr::Select(
            Box::new(cond.inner),
            Box::new(t),
            Box::new(f),
        )))
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

fn coerce_expr(obj: &Bound<'_, PyAny>) -> PyResult<Expr> {
    if let Ok(e) = obj.extract::<PyExpr>() {
        Ok(e.inner)
    } else if let Ok(v) = obj.extract::<f64>() {
        Ok(Expr::Const(v))
    } else if let Ok(v) = obj.extract::<i64>() {
        Ok(Expr::Const(v as f64))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "expected Expr or numeric value",
        ))
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

/// Execute a map spec: resolve named args to positional and call compiler.
pub fn execute_map(map_spec: &PyMapSpec) -> crate::error::ParsecResult<Buffer> {
    let expr = &map_spec.spec.inner.expr;

    // Collect all arg names from the map_spec.args keys
    let mut arg_names: Vec<String> = map_spec.args.keys().cloned().collect();
    arg_names.sort();

    // Build positional args from the expression's ArgRef indices
    let referenced = expr.referenced_args();
    let mut positional: Vec<ArgValue> = Vec::with_capacity(referenced.len());

    for idx in &referenced {
        if *idx < arg_names.len() {
            let name = &arg_names[*idx];
            positional.push(map_spec.args.get(name).cloned().ok_or_else(|| {
                crate::error::ParsecError::ArgError(format!("missing arg: {name}"))
            })?);
        } else {
            return Err(crate::error::ParsecError::ArgError(format!(
                "arg index {idx} out of range (have {} args)",
                arg_names.len()
            )));
        }
    }

    eval_elementwise(expr, &positional)
}

/// Execute a reduce spec.
pub fn execute_reduce(spec: &PyReduceSpec) -> crate::error::ParsecResult<f64> {
    eval_reduce(spec.op, &spec.buffer)
}
