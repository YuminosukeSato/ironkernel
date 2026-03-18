use pyo3::prelude::*;

use crate::ir::expr::{BinaryOp, CmpOp, Expr, UnaryOp};

/// Python-visible expression node.
/// Operator overloading builds an Expr tree without any computation.
#[pyclass(name = "Expr")]
#[derive(Clone)]
pub struct PyExpr {
    pub(crate) inner: Expr,
    pub(crate) name: Option<String>,
}

impl PyExpr {
    pub fn new(inner: Expr) -> Self {
        PyExpr { inner, name: None }
    }

    pub fn named(inner: Expr, name: String) -> Self {
        PyExpr {
            inner,
            name: Some(name),
        }
    }

    fn binop(op: BinaryOp, lhs: &Expr, rhs: &Expr) -> Self {
        PyExpr::new(Expr::Binary(
            op,
            Box::new(lhs.clone()),
            Box::new(rhs.clone()),
        ))
    }

    fn cmpop(op: CmpOp, lhs: &Expr, rhs: &Expr) -> Self {
        PyExpr::new(Expr::Compare(
            op,
            Box::new(lhs.clone()),
            Box::new(rhs.clone()),
        ))
    }
}

/// Convert a Python object to a PyExpr: either already a PyExpr, or a numeric constant.
fn coerce_to_expr(obj: &Bound<'_, PyAny>) -> PyResult<Expr> {
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

#[pymethods]
impl PyExpr {
    // --- Arithmetic: forward ---

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_expr(other)?;
        Ok(PyExpr::binop(BinaryOp::Add, &self.inner, &rhs))
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_expr(other)?;
        Ok(PyExpr::binop(BinaryOp::Sub, &self.inner, &rhs))
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_expr(other)?;
        Ok(PyExpr::binop(BinaryOp::Mul, &self.inner, &rhs))
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_expr(other)?;
        Ok(PyExpr::binop(BinaryOp::Div, &self.inner, &rhs))
    }

    fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyExpr> {
        let rhs = coerce_to_expr(other)?;
        Ok(PyExpr::binop(BinaryOp::Pow, &self.inner, &rhs))
    }

    fn __neg__(&self) -> PyExpr {
        PyExpr::new(Expr::Unary(UnaryOp::Neg, Box::new(self.inner.clone())))
    }

    fn __abs__(&self) -> PyExpr {
        PyExpr::new(Expr::Unary(UnaryOp::Abs, Box::new(self.inner.clone())))
    }

    // --- Arithmetic: reverse ---

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let lhs = coerce_to_expr(other)?;
        Ok(PyExpr::binop(BinaryOp::Add, &lhs, &self.inner))
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let lhs = coerce_to_expr(other)?;
        Ok(PyExpr::binop(BinaryOp::Sub, &lhs, &self.inner))
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let lhs = coerce_to_expr(other)?;
        Ok(PyExpr::binop(BinaryOp::Mul, &lhs, &self.inner))
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let lhs = coerce_to_expr(other)?;
        Ok(PyExpr::binop(BinaryOp::Div, &lhs, &self.inner))
    }

    fn __rpow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyExpr> {
        let lhs = coerce_to_expr(other)?;
        Ok(PyExpr::binop(BinaryOp::Pow, &lhs, &self.inner))
    }

    // --- Comparison ---

    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_expr(other)?;
        Ok(PyExpr::cmpop(CmpOp::Gt, &self.inner, &rhs))
    }

    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_expr(other)?;
        Ok(PyExpr::cmpop(CmpOp::Ge, &self.inner, &rhs))
    }

    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_expr(other)?;
        Ok(PyExpr::cmpop(CmpOp::Lt, &self.inner, &rhs))
    }

    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_expr(other)?;
        Ok(PyExpr::cmpop(CmpOp::Le, &self.inner, &rhs))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_expr(other)?;
        Ok(PyExpr::cmpop(CmpOp::Eq, &self.inner, &rhs))
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_expr(other)?;
        Ok(PyExpr::cmpop(CmpOp::Ne, &self.inner, &rhs))
    }

    // --- Repr ---

    fn __repr__(&self) -> String {
        if let Some(name) = &self.name {
            format!("Expr({name})")
        } else {
            format!("Expr({:?})", self.inner)
        }
    }
}
