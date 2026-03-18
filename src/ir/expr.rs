/// Unary operations on scalar values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Abs,
    Sqrt,
    Log,
    Exp,
    Log2,
    Log10,
    Floor,
    Ceil,
    Round,
    Sin,
    Cos,
    Tan,
}

/// Binary operations on scalar values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Atan2,
    Min,
    Max,
}

/// Comparison operations returning boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

/// Typed expression tree for elementwise computation.
///
/// This is the IR that Python's operator-overload DSL builds.
/// Rust's compiler/interpreter walks this tree to evaluate each element.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A constant scalar value.
    Const(f64),
    /// Reference to a kernel argument by index.
    ArgRef(usize),
    /// Unary operation on a sub-expression.
    Unary(UnaryOp, Box<Expr>),
    /// Binary operation on two sub-expressions.
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// Comparison producing a boolean (0.0 or 1.0).
    Compare(CmpOp, Box<Expr>, Box<Expr>),
    /// Conditional select: `where(cond, true_val, false_val)`.
    Select(Box<Expr>, Box<Expr>, Box<Expr>),
}

impl Expr {
    /// Shorthand to create a boxed `Const`.
    pub fn constant(val: f64) -> Self {
        Expr::Const(val)
    }

    /// Shorthand to create a boxed `ArgRef`.
    pub fn arg_ref(index: usize) -> Self {
        Expr::ArgRef(index)
    }

    /// Collect all `ArgRef` indices referenced in this expression.
    pub fn referenced_args(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        self.collect_args(&mut indices);
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    fn collect_args(&self, out: &mut Vec<usize>) {
        match self {
            Expr::Const(_) => {}
            Expr::ArgRef(i) => out.push(*i),
            Expr::Unary(_, inner) => inner.collect_args(out),
            Expr::Binary(_, lhs, rhs) | Expr::Compare(_, lhs, rhs) => {
                lhs.collect_args(out);
                rhs.collect_args(out);
            }
            Expr::Select(cond, t, f) => {
                cond.collect_args(out);
                t.collect_args(out);
                f.collect_args(out);
            }
        }
    }

    /// Rewrite `ArgRef` indices using the provided mapping.
    /// `ArgRef` indices not in the mapping are left unchanged.
    pub fn rewrite_arg_refs(&self, mapping: &std::collections::HashMap<usize, usize>) -> Expr {
        match self {
            Expr::Const(v) => Expr::Const(*v),
            Expr::ArgRef(i) => Expr::ArgRef(*mapping.get(i).unwrap_or(i)),
            Expr::Unary(op, inner) => Expr::Unary(*op, Box::new(inner.rewrite_arg_refs(mapping))),
            Expr::Binary(op, lhs, rhs) => Expr::Binary(
                *op,
                Box::new(lhs.rewrite_arg_refs(mapping)),
                Box::new(rhs.rewrite_arg_refs(mapping)),
            ),
            Expr::Compare(op, lhs, rhs) => Expr::Compare(
                *op,
                Box::new(lhs.rewrite_arg_refs(mapping)),
                Box::new(rhs.rewrite_arg_refs(mapping)),
            ),
            Expr::Select(cond, t, f) => Expr::Select(
                Box::new(cond.rewrite_arg_refs(mapping)),
                Box::new(t.rewrite_arg_refs(mapping)),
                Box::new(f.rewrite_arg_refs(mapping)),
            ),
        }
    }

    /// Count total nodes in the expression tree.
    pub fn node_count(&self) -> usize {
        match self {
            Expr::Const(_) | Expr::ArgRef(_) => 1,
            Expr::Unary(_, inner) => 1 + inner.node_count(),
            Expr::Binary(_, lhs, rhs) | Expr::Compare(_, lhs, rhs) => {
                1 + lhs.node_count() + rhs.node_count()
            }
            Expr::Select(cond, t, f) => 1 + cond.node_count() + t.node_count() + f.node_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashMap;

    // --- Leaf nodes ---

    #[test]
    fn const_creation() {
        let e = Expr::constant(std::f64::consts::PI);
        assert_eq!(e, Expr::Const(std::f64::consts::PI));
    }

    #[test]
    fn const_zero() {
        assert_eq!(Expr::constant(0.0), Expr::Const(0.0));
    }

    #[test]
    fn const_negative() {
        assert_eq!(Expr::constant(-1.0), Expr::Const(-1.0));
    }

    #[test]
    fn const_infinity() {
        let e = Expr::constant(f64::INFINITY);
        assert_eq!(e, Expr::Const(f64::INFINITY));
    }

    #[test]
    fn const_nan_not_equal() {
        let e1 = Expr::constant(f64::NAN);
        let e2 = Expr::constant(f64::NAN);
        // NaN != NaN by IEEE 754, so Expr::Const(NaN) != Expr::Const(NaN)
        assert_ne!(e1, e2);
    }

    #[test]
    fn arg_ref_creation() {
        let e = Expr::arg_ref(0);
        assert_eq!(e, Expr::ArgRef(0));
    }

    #[test]
    fn arg_ref_large_index() {
        let e = Expr::arg_ref(usize::MAX);
        assert_eq!(e, Expr::ArgRef(usize::MAX));
    }

    // --- Unary ---

    #[test]
    fn unary_neg() {
        let e = Expr::Unary(UnaryOp::Neg, Box::new(Expr::arg_ref(0)));
        assert_eq!(e.node_count(), 2);
    }

    #[test]
    fn unary_all_ops() {
        let ops = [
            UnaryOp::Neg,
            UnaryOp::Abs,
            UnaryOp::Sqrt,
            UnaryOp::Log,
            UnaryOp::Exp,
            UnaryOp::Log2,
            UnaryOp::Log10,
            UnaryOp::Floor,
            UnaryOp::Ceil,
            UnaryOp::Round,
            UnaryOp::Sin,
            UnaryOp::Cos,
            UnaryOp::Tan,
        ];
        for op in ops {
            let e = Expr::Unary(op, Box::new(Expr::constant(1.0)));
            assert_eq!(e.node_count(), 2);
        }
    }

    // --- Binary ---

    #[test]
    fn binary_add() {
        let e = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::arg_ref(0)),
            Box::new(Expr::constant(1.0)),
        );
        assert_eq!(e.node_count(), 3);
    }

    #[test]
    fn binary_all_ops() {
        let ops = [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Pow,
            BinaryOp::Atan2,
            BinaryOp::Min,
            BinaryOp::Max,
        ];
        for op in ops {
            let e = Expr::Binary(
                op,
                Box::new(Expr::constant(1.0)),
                Box::new(Expr::constant(2.0)),
            );
            assert_eq!(e.node_count(), 3);
        }
    }

    // --- Compare ---

    #[test]
    fn compare_gt() {
        let e = Expr::Compare(
            CmpOp::Gt,
            Box::new(Expr::arg_ref(0)),
            Box::new(Expr::constant(0.0)),
        );
        assert_eq!(e.node_count(), 3);
    }

    #[test]
    fn compare_all_ops() {
        let ops = [
            CmpOp::Gt,
            CmpOp::Ge,
            CmpOp::Lt,
            CmpOp::Le,
            CmpOp::Eq,
            CmpOp::Ne,
        ];
        for op in ops {
            let e = Expr::Compare(
                op,
                Box::new(Expr::constant(1.0)),
                Box::new(Expr::constant(2.0)),
            );
            assert_eq!(e.node_count(), 3);
        }
    }

    // --- Select ---

    #[test]
    fn select_where() {
        let cond = Expr::Compare(
            CmpOp::Gt,
            Box::new(Expr::arg_ref(0)),
            Box::new(Expr::constant(0.0)),
        );
        let e = Expr::Select(
            Box::new(cond),
            Box::new(Expr::arg_ref(0)),
            Box::new(Expr::constant(0.0)),
        );
        // cond(3) + true_val(1) + false_val(1) + select(1) = 6
        assert_eq!(e.node_count(), 6);
    }

    // --- Composite: a * x + y (saxpy) ---

    #[test]
    fn saxpy_expr() {
        // a=0, x=1, y=2 → a * x + y
        let a_times_x = Expr::Binary(
            BinaryOp::Mul,
            Box::new(Expr::arg_ref(0)),
            Box::new(Expr::arg_ref(1)),
        );
        let saxpy = Expr::Binary(
            BinaryOp::Add,
            Box::new(a_times_x),
            Box::new(Expr::arg_ref(2)),
        );
        assert_eq!(saxpy.node_count(), 5);
        assert_eq!(saxpy.referenced_args(), vec![0, 1, 2]);
    }

    // --- referenced_args ---

    #[test]
    fn referenced_args_const_only() {
        let e = Expr::constant(42.0);
        assert!(e.referenced_args().is_empty());
    }

    #[test]
    fn referenced_args_deduplication() {
        // x + x where x = arg(0)
        let e = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::arg_ref(0)),
            Box::new(Expr::arg_ref(0)),
        );
        assert_eq!(e.referenced_args(), vec![0]);
    }

    #[test]
    fn referenced_args_sorted() {
        // arg(2) + arg(0)
        let e = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::arg_ref(2)),
            Box::new(Expr::arg_ref(0)),
        );
        assert_eq!(e.referenced_args(), vec![0, 2]);
    }

    // --- Deep nesting ---

    #[test]
    fn deeply_nested_expr() {
        let mut e = Expr::constant(1.0);
        for _ in 0..100 {
            e = Expr::Unary(UnaryOp::Neg, Box::new(e));
        }
        assert_eq!(e.node_count(), 101);
    }

    // --- rewrite_arg_refs ---

    #[test]
    fn rewrite_basic() {
        let expr = Expr::arg_ref(0);
        let mapping = std::collections::HashMap::from([(0, 2)]);
        let rewritten = expr.rewrite_arg_refs(&mapping);
        assert_eq!(rewritten, Expr::ArgRef(2));
    }

    #[test]
    fn rewrite_preserves_const() {
        let expr = Expr::constant(42.0);
        let mapping = std::collections::HashMap::from([(0, 1)]);
        let rewritten = expr.rewrite_arg_refs(&mapping);
        assert_eq!(rewritten, Expr::Const(42.0));
    }

    #[test]
    fn rewrite_nested_select() {
        // where(arg(0) > 0, arg(1), arg(2)) → where(arg(10) > 0, arg(20), arg(30))
        let cond = Expr::Compare(
            CmpOp::Gt,
            Box::new(Expr::arg_ref(0)),
            Box::new(Expr::constant(0.0)),
        );
        let expr = Expr::Select(
            Box::new(cond),
            Box::new(Expr::arg_ref(1)),
            Box::new(Expr::arg_ref(2)),
        );
        let mapping = std::collections::HashMap::from([(0, 10), (1, 20), (2, 30)]);
        let rewritten = expr.rewrite_arg_refs(&mapping);

        let expected_cond = Expr::Compare(
            CmpOp::Gt,
            Box::new(Expr::arg_ref(10)),
            Box::new(Expr::constant(0.0)),
        );
        let expected = Expr::Select(
            Box::new(expected_cond),
            Box::new(Expr::arg_ref(20)),
            Box::new(Expr::arg_ref(30)),
        );
        assert_eq!(rewritten, expected);
    }

    #[test]
    fn rewrite_unmapped_passthrough() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::arg_ref(0)),
            Box::new(Expr::arg_ref(5)),
        );
        let mapping = std::collections::HashMap::from([(0, 1)]);
        let rewritten = expr.rewrite_arg_refs(&mapping);
        // arg(0) → arg(1), arg(5) unchanged
        let expected = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::arg_ref(1)),
            Box::new(Expr::arg_ref(5)),
        );
        assert_eq!(rewritten, expected);
    }

    // --- Clone ---

    #[test]
    fn expr_clone() {
        let e = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::arg_ref(0)),
            Box::new(Expr::constant(1.0)),
        );
        let e2 = e.clone();
        assert_eq!(e, e2);
    }

    // --- proptest: rewrite_arg_refs ---

    fn arb_expr(max_depth: u32) -> impl Strategy<Value = Expr> {
        let leaf = prop_oneof![
            any::<f64>().prop_map(Expr::Const),
            (0usize..10).prop_map(Expr::ArgRef),
        ];
        leaf.prop_recursive(max_depth, 64, 3, |inner| {
            prop_oneof![
                inner
                    .clone()
                    .prop_map(|e| Expr::Unary(UnaryOp::Neg, Box::new(e))),
                (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::Binary(
                    BinaryOp::Add,
                    Box::new(l),
                    Box::new(r)
                )),
                (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::Compare(
                    CmpOp::Gt,
                    Box::new(l),
                    Box::new(r)
                )),
                (inner.clone(), inner.clone(), inner).prop_map(|(c, t, f)| Expr::Select(
                    Box::new(c),
                    Box::new(t),
                    Box::new(f)
                )),
            ]
        })
    }

    proptest! {
        #[test]
        fn rewrite_preserves_tree_structure(expr in arb_expr(4)) {
            let mapping = HashMap::new(); // identity mapping
            let rewritten = expr.rewrite_arg_refs(&mapping);
            prop_assert_eq!(rewritten.node_count(), expr.node_count());
        }

        #[test]
        fn referenced_args_subset_of_rewritten(expr in arb_expr(4)) {
            // After rewriting all args to 0, referenced_args should be [0] or empty
            let refs = expr.referenced_args();
            let mapping: HashMap<usize, usize> = refs.iter().map(|&i| (i, 0)).collect();
            let rewritten = expr.rewrite_arg_refs(&mapping);
            let new_refs = rewritten.referenced_args();
            if refs.is_empty() {
                prop_assert!(new_refs.is_empty());
            } else {
                prop_assert_eq!(new_refs, vec![0]);
            }
        }
    }
}
