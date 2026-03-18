use std::collections::{BTreeMap, HashMap};

use pyo3::prelude::*;

use crate::ir::expr::{BinaryOp, CmpOp, Expr, UnaryOp};

/// Python-visible expression node.
/// Operator overloading builds an `Expr` tree without any computation.
///
/// `temp_args` maps temporary argument IDs to their names.
/// Each `arg("x")` call assigns a unique `temp_id` via a global counter,
/// and operators merge both sides' `temp_args` via `BTreeMap` union.
#[pyclass(name = "Expr")]
#[derive(Clone)]
pub struct PyExpr {
    pub(crate) inner: Expr,
    pub(crate) name: Option<String>,
    pub(crate) temp_args: BTreeMap<usize, String>,
}

/// Result of coercing a Python value to an expression.
/// Carries `temp_args` alongside the Expr so operators can merge them.
pub(crate) struct CoercedExpr {
    pub(crate) inner: Expr,
    pub(crate) temp_args: BTreeMap<usize, String>,
}

impl CoercedExpr {
    pub(crate) fn from_const(val: f64) -> Self {
        CoercedExpr {
            inner: Expr::Const(val),
            temp_args: BTreeMap::new(),
        }
    }

    pub(crate) fn from_pyexpr(e: &PyExpr) -> Self {
        CoercedExpr {
            inner: e.inner.clone(),
            temp_args: e.temp_args.clone(),
        }
    }
}

fn merge_temp_args(
    a: &BTreeMap<usize, String>,
    b: &BTreeMap<usize, String>,
) -> BTreeMap<usize, String> {
    let mut merged = a.clone();
    merged.extend(b.iter().map(|(k, v)| (*k, v.clone())));
    merged
}

impl PyExpr {
    pub fn new(inner: Expr) -> Self {
        PyExpr {
            inner,
            name: None,
            temp_args: BTreeMap::new(),
        }
    }

    pub(crate) fn from_parts(inner: Expr, temp_args: BTreeMap<usize, String>) -> Self {
        PyExpr {
            inner,
            name: None,
            temp_args,
        }
    }

    pub fn named(inner: Expr, name: String, temp_id: usize) -> Self {
        let mut temp_args = BTreeMap::new();
        temp_args.insert(temp_id, name.clone());
        PyExpr {
            inner,
            name: Some(name),
            temp_args,
        }
    }

    pub(crate) fn binop_with_temp_args(
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
        lhs_temp: &BTreeMap<usize, String>,
        rhs_temp: &BTreeMap<usize, String>,
    ) -> Self {
        PyExpr {
            inner: Expr::Binary(op, Box::new(lhs.clone()), Box::new(rhs.clone())),
            name: None,
            temp_args: merge_temp_args(lhs_temp, rhs_temp),
        }
    }

    pub(crate) fn cmpop_with_temp_args(
        op: CmpOp,
        lhs: &Expr,
        rhs: &Expr,
        lhs_temp: &BTreeMap<usize, String>,
        rhs_temp: &BTreeMap<usize, String>,
    ) -> Self {
        PyExpr {
            inner: Expr::Compare(op, Box::new(lhs.clone()), Box::new(rhs.clone())),
            name: None,
            temp_args: merge_temp_args(lhs_temp, rhs_temp),
        }
    }

    pub(crate) fn unary_with_temp_args(
        op: UnaryOp,
        inner: &Expr,
        temp: &BTreeMap<usize, String>,
    ) -> Self {
        PyExpr {
            inner: Expr::Unary(op, Box::new(inner.clone())),
            name: None,
            temp_args: temp.clone(),
        }
    }

    pub(crate) fn rewrite_args_internal(
        &self,
        mapping: &std::collections::HashMap<usize, usize>,
    ) -> PyExpr {
        let new_inner = self.inner.rewrite_arg_refs(mapping);
        let new_temp_args = self
            .temp_args
            .iter()
            .map(|(old_id, name)| {
                let new_id = mapping.get(old_id).copied().unwrap_or(*old_id);
                (new_id, name.clone())
            })
            .collect();
        PyExpr {
            inner: new_inner,
            name: self.name.clone(),
            temp_args: new_temp_args,
        }
    }
}

/// Convert a Python object to a `CoercedExpr` (`Expr` + `temp_args`).
pub(crate) fn coerce_to_coerced(obj: &Bound<'_, PyAny>) -> PyResult<CoercedExpr> {
    if let Ok(e) = obj.extract::<PyExpr>() {
        Ok(CoercedExpr::from_pyexpr(&e))
    } else if let Ok(v) = obj.extract::<i64>() {
        Ok(CoercedExpr::from_const(v as f64))
    } else if let Ok(v) = obj.extract::<f64>() {
        Ok(CoercedExpr::from_const(v))
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
        let rhs = coerce_to_coerced(other)?;
        Ok(PyExpr::binop_with_temp_args(
            BinaryOp::Add,
            &self.inner,
            &rhs.inner,
            &self.temp_args,
            &rhs.temp_args,
        ))
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_coerced(other)?;
        Ok(PyExpr::binop_with_temp_args(
            BinaryOp::Sub,
            &self.inner,
            &rhs.inner,
            &self.temp_args,
            &rhs.temp_args,
        ))
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_coerced(other)?;
        Ok(PyExpr::binop_with_temp_args(
            BinaryOp::Mul,
            &self.inner,
            &rhs.inner,
            &self.temp_args,
            &rhs.temp_args,
        ))
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_coerced(other)?;
        Ok(PyExpr::binop_with_temp_args(
            BinaryOp::Div,
            &self.inner,
            &rhs.inner,
            &self.temp_args,
            &rhs.temp_args,
        ))
    }

    fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyExpr> {
        let rhs = coerce_to_coerced(other)?;
        Ok(PyExpr::binop_with_temp_args(
            BinaryOp::Pow,
            &self.inner,
            &rhs.inner,
            &self.temp_args,
            &rhs.temp_args,
        ))
    }

    fn __neg__(&self) -> PyExpr {
        PyExpr::unary_with_temp_args(UnaryOp::Neg, &self.inner, &self.temp_args)
    }

    fn __abs__(&self) -> PyExpr {
        PyExpr::unary_with_temp_args(UnaryOp::Abs, &self.inner, &self.temp_args)
    }

    // --- Arithmetic: reverse ---

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let lhs = coerce_to_coerced(other)?;
        Ok(PyExpr::binop_with_temp_args(
            BinaryOp::Add,
            &lhs.inner,
            &self.inner,
            &lhs.temp_args,
            &self.temp_args,
        ))
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let lhs = coerce_to_coerced(other)?;
        Ok(PyExpr::binop_with_temp_args(
            BinaryOp::Sub,
            &lhs.inner,
            &self.inner,
            &lhs.temp_args,
            &self.temp_args,
        ))
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let lhs = coerce_to_coerced(other)?;
        Ok(PyExpr::binop_with_temp_args(
            BinaryOp::Mul,
            &lhs.inner,
            &self.inner,
            &lhs.temp_args,
            &self.temp_args,
        ))
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let lhs = coerce_to_coerced(other)?;
        Ok(PyExpr::binop_with_temp_args(
            BinaryOp::Div,
            &lhs.inner,
            &self.inner,
            &lhs.temp_args,
            &self.temp_args,
        ))
    }

    fn __rpow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyExpr> {
        let lhs = coerce_to_coerced(other)?;
        Ok(PyExpr::binop_with_temp_args(
            BinaryOp::Pow,
            &lhs.inner,
            &self.inner,
            &lhs.temp_args,
            &self.temp_args,
        ))
    }

    // --- Comparison ---

    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_coerced(other)?;
        Ok(PyExpr::cmpop_with_temp_args(
            CmpOp::Gt,
            &self.inner,
            &rhs.inner,
            &self.temp_args,
            &rhs.temp_args,
        ))
    }

    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_coerced(other)?;
        Ok(PyExpr::cmpop_with_temp_args(
            CmpOp::Ge,
            &self.inner,
            &rhs.inner,
            &self.temp_args,
            &rhs.temp_args,
        ))
    }

    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_coerced(other)?;
        Ok(PyExpr::cmpop_with_temp_args(
            CmpOp::Lt,
            &self.inner,
            &rhs.inner,
            &self.temp_args,
            &rhs.temp_args,
        ))
    }

    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_coerced(other)?;
        Ok(PyExpr::cmpop_with_temp_args(
            CmpOp::Le,
            &self.inner,
            &rhs.inner,
            &self.temp_args,
            &rhs.temp_args,
        ))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_coerced(other)?;
        Ok(PyExpr::cmpop_with_temp_args(
            CmpOp::Eq,
            &self.inner,
            &rhs.inner,
            &self.temp_args,
            &rhs.temp_args,
        ))
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let rhs = coerce_to_coerced(other)?;
        Ok(PyExpr::cmpop_with_temp_args(
            CmpOp::Ne,
            &self.inner,
            &rhs.inner,
            &self.temp_args,
            &rhs.temp_args,
        ))
    }

    // --- Bool trap ---

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "cannot convert Expr to bool; use kernel.where(cond, t, f) for conditional expressions",
        ))
    }

    // --- Repr ---

    fn __repr__(&self) -> String {
        if let Some(name) = &self.name {
            format!("Expr({name})")
        } else {
            format!("Expr({:?})", self.inner)
        }
    }

    fn get_temp_args(&self) -> HashMap<usize, String> {
        self.temp_args
            .iter()
            .map(|(temp_id, name)| (*temp_id, name.clone()))
            .collect()
    }

    fn get_referenced_args(&self) -> Vec<usize> {
        self.inner.referenced_args()
    }

    fn rewrite_args(&self, mapping: HashMap<usize, usize>) -> PyExpr {
        self.rewrite_args_internal(&mapping)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- R1.3: temp_args field ---

    #[test]
    fn new_has_empty_temp_args() {
        let e = PyExpr::new(Expr::Const(1.0));
        assert!(e.temp_args.is_empty());
    }

    #[test]
    fn named_registers_temp_id() {
        let e = PyExpr::named(Expr::ArgRef(42), "x".to_string(), 42);
        assert_eq!(e.temp_args.get(&42), Some(&"x".to_string()));
        assert_eq!(e.name, Some("x".to_string()));
        assert_eq!(e.inner, Expr::ArgRef(42));
    }

    #[test]
    fn named_single_entry_in_temp_args() {
        let e = PyExpr::named(Expr::ArgRef(0), "y".to_string(), 0);
        assert_eq!(e.temp_args.len(), 1);
    }

    // --- R1.4: temp_args union merge ---

    #[test]
    fn binop_merges_temp_args() {
        let lhs = PyExpr::named(Expr::ArgRef(100), "x".to_string(), 100);
        let rhs = PyExpr::named(Expr::ArgRef(200), "y".to_string(), 200);
        let result = PyExpr::binop_with_temp_args(
            BinaryOp::Add,
            &lhs.inner,
            &rhs.inner,
            &lhs.temp_args,
            &rhs.temp_args,
        );
        assert_eq!(result.temp_args.len(), 2);
        assert_eq!(result.temp_args.get(&100), Some(&"x".to_string()));
        assert_eq!(result.temp_args.get(&200), Some(&"y".to_string()));
    }

    #[test]
    fn binop_deduplicates_same_temp_id() {
        let mut args = BTreeMap::new();
        args.insert(100, "x".to_string());
        let lhs_args = args.clone();
        let rhs_args = args;
        let result = PyExpr::binop_with_temp_args(
            BinaryOp::Add,
            &Expr::ArgRef(100),
            &Expr::ArgRef(100),
            &lhs_args,
            &rhs_args,
        );
        assert_eq!(result.temp_args.len(), 1);
        assert_eq!(result.temp_args.get(&100), Some(&"x".to_string()));
    }

    #[test]
    fn cmpop_merges_temp_args() {
        let lhs = PyExpr::named(Expr::ArgRef(10), "a".to_string(), 10);
        let rhs = PyExpr::named(Expr::ArgRef(20), "b".to_string(), 20);
        let result = PyExpr::cmpop_with_temp_args(
            CmpOp::Gt,
            &lhs.inner,
            &rhs.inner,
            &lhs.temp_args,
            &rhs.temp_args,
        );
        assert_eq!(result.temp_args.len(), 2);
        assert_eq!(result.temp_args.get(&10), Some(&"a".to_string()));
        assert_eq!(result.temp_args.get(&20), Some(&"b".to_string()));
    }

    #[test]
    fn unary_preserves_temp_args() {
        let operand = PyExpr::named(Expr::ArgRef(5), "z".to_string(), 5);
        let result = PyExpr::unary_with_temp_args(UnaryOp::Neg, &operand.inner, &operand.temp_args);
        assert_eq!(result.temp_args.len(), 1);
        assert_eq!(result.temp_args.get(&5), Some(&"z".to_string()));
    }

    #[test]
    fn const_coercion_has_empty_temp_args() {
        let coerced = CoercedExpr::from_const(42.0);
        assert!(coerced.temp_args.is_empty());
        assert_eq!(coerced.inner, Expr::Const(42.0));
    }

    #[test]
    fn coerced_from_pyexpr_carries_temp_args() {
        let e = PyExpr::named(Expr::ArgRef(7), "w".to_string(), 7);
        let coerced = CoercedExpr::from_pyexpr(&e);
        assert_eq!(coerced.temp_args.len(), 1);
        assert_eq!(coerced.temp_args.get(&7), Some(&"w".to_string()));
    }

    // --- R1.4: three-way merge for where_ / select ---

    #[test]
    fn three_way_merge_temp_args() {
        let a = PyExpr::named(Expr::ArgRef(1), "a".to_string(), 1);
        let b = PyExpr::named(Expr::ArgRef(2), "b".to_string(), 2);
        let c = PyExpr::named(Expr::ArgRef(3), "c".to_string(), 3);
        let mut merged = a.temp_args.clone();
        merged.extend(b.temp_args.iter().map(|(k, v)| (*k, v.clone())));
        merged.extend(c.temp_args.iter().map(|(k, v)| (*k, v.clone())));
        assert_eq!(merged.len(), 3);
    }

    // --- R1.5: get_referenced_args / rewrite_args ---

    #[test]
    fn get_referenced_arg_ids() {
        let mut temp_args = BTreeMap::new();
        temp_args.insert(100, "x".to_string());
        temp_args.insert(200, "y".to_string());
        let e = PyExpr {
            inner: Expr::Binary(
                BinaryOp::Add,
                Box::new(Expr::ArgRef(100)),
                Box::new(Expr::ArgRef(200)),
            ),
            name: None,
            temp_args,
        };
        let refs = e.inner.referenced_args();
        assert_eq!(refs, vec![100, 200]);
    }

    #[test]
    fn rewrite_args_remaps_and_rebuilds_temp_args() {
        let mut temp_args = BTreeMap::new();
        temp_args.insert(100, "x".to_string());
        temp_args.insert(200, "y".to_string());
        let e = PyExpr {
            inner: Expr::Binary(
                BinaryOp::Add,
                Box::new(Expr::ArgRef(100)),
                Box::new(Expr::ArgRef(200)),
            ),
            name: None,
            temp_args,
        };
        let mapping = std::collections::HashMap::from([(100, 0), (200, 1)]);
        let rewritten = e.rewrite_args_internal(&mapping);
        assert_eq!(rewritten.inner.referenced_args(), vec![0, 1]);
        assert_eq!(rewritten.temp_args.get(&0), Some(&"x".to_string()));
        assert_eq!(rewritten.temp_args.get(&1), Some(&"y".to_string()));
    }

    // --- from_parts ---

    #[test]
    fn from_parts_sets_inner_and_temp_args() {
        let mut ta = BTreeMap::new();
        ta.insert(5, "z".to_string());
        let e = PyExpr::from_parts(Expr::ArgRef(5), ta.clone());
        assert_eq!(e.inner, Expr::ArgRef(5));
        assert!(e.name.is_none());
        assert_eq!(e.temp_args, ta);
    }

    // --- rewrite_args_internal preserves name ---

    #[test]
    fn rewrite_args_preserves_name() {
        let e = PyExpr::named(Expr::ArgRef(10), "x".to_string(), 10);
        let mapping = HashMap::from([(10, 0)]);
        let rewritten = e.rewrite_args_internal(&mapping);
        assert_eq!(rewritten.name, Some("x".to_string()));
    }

    // --- #[pymethods] via GIL ---

    #[test]
    fn pymethods_add() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let lhs = PyExpr::new(Expr::ArgRef(0));
            let rhs_val = 2.0f64.into_pyobject(py).unwrap();
            let result = lhs.__add__(rhs_val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Binary(BinaryOp::Add, _, _)));
        });
    }

    #[test]
    fn pymethods_sub() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let lhs = PyExpr::new(Expr::ArgRef(0));
            let rhs_val = 1.0f64.into_pyobject(py).unwrap();
            let result = lhs.__sub__(rhs_val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Binary(BinaryOp::Sub, _, _)));
        });
    }

    #[test]
    fn pymethods_mul() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let lhs = PyExpr::new(Expr::ArgRef(0));
            let rhs_val = 3.0f64.into_pyobject(py).unwrap();
            let result = lhs.__mul__(rhs_val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Binary(BinaryOp::Mul, _, _)));
        });
    }

    #[test]
    fn pymethods_truediv() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let lhs = PyExpr::new(Expr::ArgRef(0));
            let rhs_val = 2.0f64.into_pyobject(py).unwrap();
            let result = lhs.__truediv__(rhs_val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Binary(BinaryOp::Div, _, _)));
        });
    }

    #[test]
    fn pymethods_pow() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let lhs = PyExpr::new(Expr::ArgRef(0));
            let rhs_val = 2.0f64.into_pyobject(py).unwrap();
            let result = lhs.__pow__(rhs_val.as_any(), None).unwrap();
            assert!(matches!(result.inner, Expr::Binary(BinaryOp::Pow, _, _)));
        });
    }

    #[test]
    fn pymethods_neg() {
        let e = PyExpr::new(Expr::ArgRef(0));
        let result = e.__neg__();
        assert!(matches!(result.inner, Expr::Unary(UnaryOp::Neg, _)));
    }

    #[test]
    fn pymethods_abs() {
        let e = PyExpr::new(Expr::ArgRef(0));
        let result = e.__abs__();
        assert!(matches!(result.inner, Expr::Unary(UnaryOp::Abs, _)));
    }

    // --- Reverse operators ---

    #[test]
    fn pymethods_radd() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let rhs = PyExpr::new(Expr::ArgRef(0));
            let lhs_val = 5.0f64.into_pyobject(py).unwrap();
            let result = rhs.__radd__(lhs_val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Binary(BinaryOp::Add, _, _)));
        });
    }

    #[test]
    fn pymethods_rsub() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let rhs = PyExpr::new(Expr::ArgRef(0));
            let lhs_val = 5.0f64.into_pyobject(py).unwrap();
            let result = rhs.__rsub__(lhs_val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Binary(BinaryOp::Sub, _, _)));
        });
    }

    #[test]
    fn pymethods_rmul() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let rhs = PyExpr::new(Expr::ArgRef(0));
            let lhs_val = 5.0f64.into_pyobject(py).unwrap();
            let result = rhs.__rmul__(lhs_val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Binary(BinaryOp::Mul, _, _)));
        });
    }

    #[test]
    fn pymethods_rtruediv() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let rhs = PyExpr::new(Expr::ArgRef(0));
            let lhs_val = 10.0f64.into_pyobject(py).unwrap();
            let result = rhs.__rtruediv__(lhs_val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Binary(BinaryOp::Div, _, _)));
        });
    }

    #[test]
    fn pymethods_rpow() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let rhs = PyExpr::new(Expr::ArgRef(0));
            let lhs_val = 2.0f64.into_pyobject(py).unwrap();
            let result = rhs.__rpow__(lhs_val.as_any(), None).unwrap();
            assert!(matches!(result.inner, Expr::Binary(BinaryOp::Pow, _, _)));
        });
    }

    // --- Comparison operators ---

    #[test]
    fn pymethods_gt() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let e = PyExpr::new(Expr::ArgRef(0));
            let val = 0.0f64.into_pyobject(py).unwrap();
            let result = e.__gt__(val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Compare(CmpOp::Gt, _, _)));
        });
    }

    #[test]
    fn pymethods_ge() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let e = PyExpr::new(Expr::ArgRef(0));
            let val = 0.0f64.into_pyobject(py).unwrap();
            let result = e.__ge__(val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Compare(CmpOp::Ge, _, _)));
        });
    }

    #[test]
    fn pymethods_lt() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let e = PyExpr::new(Expr::ArgRef(0));
            let val = 0.0f64.into_pyobject(py).unwrap();
            let result = e.__lt__(val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Compare(CmpOp::Lt, _, _)));
        });
    }

    #[test]
    fn pymethods_le() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let e = PyExpr::new(Expr::ArgRef(0));
            let val = 0.0f64.into_pyobject(py).unwrap();
            let result = e.__le__(val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Compare(CmpOp::Le, _, _)));
        });
    }

    #[test]
    fn pymethods_eq() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let e = PyExpr::new(Expr::ArgRef(0));
            let val = 0.0f64.into_pyobject(py).unwrap();
            let result = e.__eq__(val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Compare(CmpOp::Eq, _, _)));
        });
    }

    #[test]
    fn pymethods_ne() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let e = PyExpr::new(Expr::ArgRef(0));
            let val = 0.0f64.into_pyobject(py).unwrap();
            let result = e.__ne__(val.as_any()).unwrap();
            assert!(matches!(result.inner, Expr::Compare(CmpOp::Ne, _, _)));
        });
    }

    // --- Bool trap ---

    #[test]
    fn pymethods_bool_raises() {
        let e = PyExpr::new(Expr::ArgRef(0));
        assert!(e.__bool__().is_err());
    }

    // --- Repr ---

    #[test]
    fn pymethods_repr_named() {
        let e = PyExpr::named(Expr::ArgRef(0), "x".to_string(), 0);
        assert_eq!(e.__repr__(), "Expr(x)");
    }

    #[test]
    fn pymethods_repr_unnamed() {
        let e = PyExpr::new(Expr::Const(42.0));
        let repr = e.__repr__();
        assert!(repr.starts_with("Expr("));
        assert!(repr.contains("42.0"));
    }

    // --- get_temp_args / get_referenced_args / rewrite_args ---

    #[test]
    fn pymethods_get_temp_args() {
        let e = PyExpr::named(Expr::ArgRef(10), "x".to_string(), 10);
        let ta = e.get_temp_args();
        assert_eq!(ta.len(), 1);
        assert_eq!(ta.get(&10), Some(&"x".to_string()));
    }

    #[test]
    fn pymethods_get_referenced_args() {
        let e = PyExpr {
            inner: Expr::Binary(
                BinaryOp::Add,
                Box::new(Expr::ArgRef(0)),
                Box::new(Expr::ArgRef(1)),
            ),
            name: None,
            temp_args: BTreeMap::new(),
        };
        assert_eq!(e.get_referenced_args(), vec![0, 1]);
    }

    #[test]
    fn pymethods_rewrite_args() {
        let mut ta = BTreeMap::new();
        ta.insert(10, "x".to_string());
        let e = PyExpr {
            inner: Expr::ArgRef(10),
            name: None,
            temp_args: ta,
        };
        let mapping = HashMap::from([(10, 0)]);
        let result = e.rewrite_args(mapping);
        assert_eq!(result.inner, Expr::ArgRef(0));
        assert_eq!(result.temp_args.get(&0), Some(&"x".to_string()));
    }

    // --- Coerce int ---

    #[test]
    fn coerce_int_to_coerced() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let int_val = 42i64.into_pyobject(py).unwrap();
            let coerced = coerce_to_coerced(int_val.as_any()).unwrap();
            assert_eq!(coerced.inner, Expr::Const(42.0));
        });
    }

    #[test]
    fn coerce_invalid_type_raises() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let string_val = "invalid".into_pyobject(py).unwrap();
            let result = coerce_to_coerced(string_val.as_any());
            assert!(result.is_err());
        });
    }

    #[test]
    fn coerce_pyexpr_roundtrip() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let e = PyExpr::named(Expr::ArgRef(5), "x".to_string(), 5);
            let py_obj = e.clone().into_pyobject(py).unwrap();
            let coerced = coerce_to_coerced(py_obj.as_any()).unwrap();
            assert_eq!(coerced.inner, Expr::ArgRef(5));
            assert_eq!(coerced.temp_args.get(&5), Some(&"x".to_string()));
        });
    }
}
