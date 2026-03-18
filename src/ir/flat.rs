//! `FlatExpr`: stack-machine representation of `Expr` trees.
//!
//! Converts an `Expr` tree into a linear `Vec<FlatOp>` for cache-friendly evaluation.
//! Post-order traversal ensures operands are on the stack before the operator.

use rayon::prelude::*;

use crate::buffer::inner::Buffer;
use crate::error::{ParsecError, ParsecResult};
use crate::ir::compiler::ArgValue;
use crate::ir::expr::{BinaryOp, CmpOp, Expr, UnaryOp};

/// A single instruction in the stack machine.
#[derive(Debug, Clone)]
pub(crate) enum FlatOp {
    PushConst(f64),
    PushArg(usize),
    Unary(UnaryOp),
    Binary(BinaryOp),
    Compare(CmpOp),
}

/// Compiled flat representation of an `Expr` tree.
#[derive(Debug, Clone)]
pub(crate) struct FlatExpr {
    ops: Vec<FlatOp>,
    max_stack_depth: usize,
}

/// Compile an `Expr` tree into a `FlatExpr` (stack-machine instruction sequence).
pub(crate) fn compile(expr: &Expr) -> FlatExpr {
    let mut ops = Vec::new();
    let mut max_depth: usize = 0;
    compile_recursive(expr, &mut ops, 0, &mut max_depth);
    FlatExpr {
        ops,
        max_stack_depth: max_depth,
    }
}

fn compile_recursive(
    expr: &Expr,
    ops: &mut Vec<FlatOp>,
    current_depth: usize,
    max_depth: &mut usize,
) {
    let new_depth = current_depth + 1;
    if new_depth > *max_depth {
        *max_depth = new_depth;
    }
    match expr {
        Expr::Const(v) => ops.push(FlatOp::PushConst(*v)),
        Expr::ArgRef(i) => ops.push(FlatOp::PushArg(*i)),
        Expr::Unary(op, inner) => {
            compile_recursive(inner, ops, current_depth, max_depth);
            ops.push(FlatOp::Unary(*op));
        }
        Expr::Binary(op, lhs, rhs) => {
            compile_recursive(lhs, ops, current_depth, max_depth);
            compile_recursive(rhs, ops, current_depth + 1, max_depth);
            ops.push(FlatOp::Binary(*op));
        }
        Expr::Compare(op, lhs, rhs) => {
            compile_recursive(lhs, ops, current_depth, max_depth);
            compile_recursive(rhs, ops, current_depth + 1, max_depth);
            ops.push(FlatOp::Compare(*op));
        }
        Expr::Select(..) => {
            unreachable!("Select nodes must be handled by the lazy tree evaluator")
        }
    }
}

/// Evaluate a compiled `FlatExpr` over input arguments, producing a new Buffer.
pub(crate) fn eval_flat_elementwise(flat: &FlatExpr, args: &[ArgValue]) -> ParsecResult<Buffer> {
    let len = infer_output_len(args)?;

    if len == 0 {
        return Ok(Buffer::empty_f64());
    }

    let arg_slices: Vec<Option<&[f64]>> = args
        .iter()
        .map(|a| match a {
            ArgValue::Scalar(_) => None,
            ArgValue::Buffer(b) => Some(b.as_f64_slice()),
        })
        .collect();

    const PAR_THRESHOLD: usize = 4096;

    let result: Vec<f64> = if len >= PAR_THRESHOLD {
        (0..len)
            .into_par_iter()
            .with_min_len(1024)
            .map_init(
                || Vec::with_capacity(flat.max_stack_depth),
                |scratch, i| eval_flat_at(flat, i, &arg_slices, args, scratch),
            )
            .collect()
    } else {
        let mut scratch = Vec::with_capacity(flat.max_stack_depth);
        (0..len)
            .map(|i| eval_flat_at(flat, i, &arg_slices, args, &mut scratch))
            .collect()
    };

    Ok(Buffer::from_f64_vec(result))
}

/// Evaluate the stack machine at element index `i`, reusing `scratch` as the operand stack.
#[inline]
fn eval_flat_at(
    flat: &FlatExpr,
    i: usize,
    slices: &[Option<&[f64]>],
    args: &[ArgValue],
    scratch: &mut Vec<f64>,
) -> f64 {
    scratch.clear();
    for op in &flat.ops {
        match op {
            FlatOp::PushConst(v) => scratch.push(*v),
            FlatOp::PushArg(idx) => {
                let val = match &args[*idx] {
                    ArgValue::Scalar(v) => *v,
                    ArgValue::Buffer(_) => slices[*idx].unwrap()[i],
                };
                scratch.push(val);
            }
            FlatOp::Unary(op) => {
                let a = scratch.pop().unwrap();
                scratch.push(apply_unary(*op, a));
            }
            FlatOp::Binary(op) => {
                let b = scratch.pop().unwrap();
                let a = scratch.pop().unwrap();
                scratch.push(apply_binary(*op, a, b));
            }
            FlatOp::Compare(op) => {
                let b = scratch.pop().unwrap();
                let a = scratch.pop().unwrap();
                scratch.push(if apply_cmp(*op, a, b) { 1.0 } else { 0.0 });
            }
        }
    }
    scratch[0]
}

/// Determine output length from arguments.
pub(crate) fn infer_output_len(args: &[ArgValue]) -> ParsecResult<usize> {
    let mut len: Option<usize> = None;
    for arg in args {
        if let ArgValue::Buffer(b) = arg {
            match len {
                None => len = Some(b.len()),
                Some(l) if l != b.len() => {
                    return Err(ParsecError::ShapeError(format!(
                        "buffer length mismatch: {l} vs {}",
                        b.len()
                    )));
                }
                Some(_) => {}
            }
        }
    }
    len.ok_or_else(|| ParsecError::ArgError("no buffer arguments provided".into()))
}

#[inline]
pub(crate) fn apply_unary(op: UnaryOp, v: f64) -> f64 {
    match op {
        UnaryOp::Neg => -v,
        UnaryOp::Abs => v.abs(),
        UnaryOp::Sqrt => v.sqrt(),
        UnaryOp::Log => v.ln(),
        UnaryOp::Exp => v.exp(),
        UnaryOp::Log2 => v.log2(),
        UnaryOp::Log10 => v.log10(),
        UnaryOp::Floor => v.floor(),
        UnaryOp::Ceil => v.ceil(),
        UnaryOp::Round => v.round(),
        UnaryOp::Sin => v.sin(),
        UnaryOp::Cos => v.cos(),
        UnaryOp::Tan => v.tan(),
    }
}

#[inline]
pub(crate) fn apply_binary(op: BinaryOp, l: f64, r: f64) -> f64 {
    match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div => l / r,
        BinaryOp::Pow => l.powf(r),
        BinaryOp::Atan2 => l.atan2(r),
        BinaryOp::Min => l.min(r),
        BinaryOp::Max => l.max(r),
    }
}

#[inline]
pub(crate) fn apply_cmp(op: CmpOp, l: f64, r: f64) -> bool {
    match op {
        CmpOp::Gt => l > r,
        CmpOp::Ge => l >= r,
        CmpOp::Lt => l < r,
        CmpOp::Le => l <= r,
        CmpOp::Eq => l == r,
        CmpOp::Ne => l != r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::kernel::ReduceOp;
    use proptest::prelude::*;

    fn buf(data: Vec<f64>) -> ArgValue {
        ArgValue::Buffer(Buffer::from_f64_vec(data))
    }

    fn scalar(v: f64) -> ArgValue {
        ArgValue::Scalar(v)
    }

    fn finite_vec_strategy(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(-1.0e6f64..1.0e6f64, 1..=max_len)
    }

    prop_compose! {
        fn same_length_vec_pair_strategy(max_len: usize)
            (len in 1usize..=max_len)
            (
                left in prop::collection::vec(-1.0e6f64..1.0e6f64, len),
                right in prop::collection::vec(-1.0e6f64..1.0e6f64, len),
            ) -> (Vec<f64>, Vec<f64>) {
                (left, right)
            }
    }

    // ─── compile: structure tests ───

    #[test]
    fn compile_const() {
        let flat = compile(&Expr::Const(42.0));
        assert_eq!(flat.ops.len(), 1);
        assert_eq!(flat.max_stack_depth, 1);
    }

    #[test]
    fn compile_argref() {
        let flat = compile(&Expr::ArgRef(0));
        assert_eq!(flat.ops.len(), 1);
        assert_eq!(flat.max_stack_depth, 1);
    }

    #[test]
    fn compile_binary_depth() {
        // x + y → [Arg(0), Arg(1), Add] depth=2
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(1)),
        );
        let flat = compile(&expr);
        assert_eq!(flat.ops.len(), 3);
        assert_eq!(flat.max_stack_depth, 2);
    }

    #[test]
    fn compile_saxpy_ops() {
        // a * x + y → [Arg(0), Arg(1), Mul, Arg(2), Add]
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::Binary(
                BinaryOp::Mul,
                Box::new(Expr::ArgRef(0)),
                Box::new(Expr::ArgRef(1)),
            )),
            Box::new(Expr::ArgRef(2)),
        );
        let flat = compile(&expr);
        assert_eq!(flat.ops.len(), 5);
    }

    #[test]
    fn compile_deeply_nested() {
        let mut expr = Expr::ArgRef(0);
        for _ in 0..50 {
            expr = Expr::Unary(UnaryOp::Neg, Box::new(expr));
        }
        let flat = compile(&expr);
        assert_eq!(flat.ops.len(), 51); // 1 push + 50 unary
                                        // Unary does not increase depth beyond 1
        assert_eq!(flat.max_stack_depth, 1);
    }

    #[test]
    #[should_panic(expected = "Select nodes must be handled by the lazy tree evaluator")]
    fn compile_select_panics_why_flat_compilation_must_reject_lazy_only_nodes() {
        let expr = Expr::Select(
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Const(2.0)),
            Box::new(Expr::Const(3.0)),
        );
        let _ = compile(&expr);
    }

    // ─── eval: basic operations ───

    #[test]
    fn eval_const() {
        let flat = compile(&Expr::Const(42.0));
        let args = vec![buf(vec![0.0])]; // dummy buffer for length
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        assert_eq!(result.as_f64_slice(), &[42.0]);
    }

    #[test]
    fn eval_argref() {
        let flat = compile(&Expr::ArgRef(0));
        let args = vec![buf(vec![1.0, 2.0, 3.0])];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        assert_eq!(result.as_f64_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn eval_add() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(1)),
        );
        let flat = compile(&expr);
        let args = vec![buf(vec![1.0, 2.0]), buf(vec![3.0, 4.0])];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        assert_eq!(result.as_f64_slice(), &[4.0, 6.0]);
    }

    #[test]
    fn eval_saxpy() {
        // a=2.0, x=[1,2,3], y=[10,20,30] → [12,24,36]
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::Binary(
                BinaryOp::Mul,
                Box::new(Expr::ArgRef(0)),
                Box::new(Expr::ArgRef(1)),
            )),
            Box::new(Expr::ArgRef(2)),
        );
        let flat = compile(&expr);
        let args = vec![
            scalar(2.0),
            buf(vec![1.0, 2.0, 3.0]),
            buf(vec![10.0, 20.0, 30.0]),
        ];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        assert_eq!(result.as_f64_slice(), &[12.0, 24.0, 36.0]);
    }

    #[test]
    fn eval_neg() {
        let expr = Expr::Unary(UnaryOp::Neg, Box::new(Expr::ArgRef(0)));
        let flat = compile(&expr);
        let args = vec![buf(vec![1.0, -2.0, 3.0])];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        assert_eq!(result.as_f64_slice(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn eval_sqrt() {
        let expr = Expr::Unary(UnaryOp::Sqrt, Box::new(Expr::ArgRef(0)));
        let flat = compile(&expr);
        let args = vec![buf(vec![4.0, 9.0, 16.0])];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        assert_eq!(result.as_f64_slice(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn eval_complex_sqrt_abs_sin() {
        // sqrt(abs(x)) + sin(x)
        let x = Expr::ArgRef(0);
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::Unary(
                UnaryOp::Sqrt,
                Box::new(Expr::Unary(UnaryOp::Abs, Box::new(x.clone()))),
            )),
            Box::new(Expr::Unary(UnaryOp::Sin, Box::new(x))),
        );
        let flat = compile(&expr);
        let args = vec![buf(vec![0.0, 1.0, 4.0])];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        let out = result.as_f64_slice();
        let expected_0 = 0.0_f64.abs().sqrt() + 0.0_f64.sin();
        let expected_1 = 1.0_f64.abs().sqrt() + 1.0_f64.sin();
        let expected_4 = 4.0_f64.abs().sqrt() + 4.0_f64.sin();
        assert!((out[0] - expected_0).abs() < 1e-10);
        assert!((out[1] - expected_1).abs() < 1e-10);
        assert!((out[2] - expected_4).abs() < 1e-10);
    }

    // ─── eval: matches tree evaluator ───

    #[test]
    fn flat_matches_tree_saxpy() {
        use crate::ir::compiler::eval_elementwise;

        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::Binary(
                BinaryOp::Mul,
                Box::new(Expr::ArgRef(0)),
                Box::new(Expr::ArgRef(1)),
            )),
            Box::new(Expr::ArgRef(2)),
        );
        let args = vec![
            scalar(2.0),
            buf(vec![0.0, 1.0, 2.0, 3.0, 4.0]),
            buf(vec![10.0, 20.0, 30.0, 40.0, 50.0]),
        ];

        let tree_result = eval_elementwise(&expr, &args).unwrap();
        let flat = compile(&expr);
        let flat_result = eval_flat_elementwise(&flat, &args).unwrap();
        assert_eq!(tree_result.as_f64_slice(), flat_result.as_f64_slice());
    }

    #[test]
    fn flat_matches_tree_complex() {
        use crate::ir::compiler::eval_elementwise;

        // sqrt(abs(x)) + sin(x) * cos(x)
        let x = Expr::ArgRef(0);
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::Unary(
                UnaryOp::Sqrt,
                Box::new(Expr::Unary(UnaryOp::Abs, Box::new(x.clone()))),
            )),
            Box::new(Expr::Binary(
                BinaryOp::Mul,
                Box::new(Expr::Unary(UnaryOp::Sin, Box::new(x.clone()))),
                Box::new(Expr::Unary(UnaryOp::Cos, Box::new(x))),
            )),
        );
        let data: Vec<f64> = (-50..50).map(|i| i as f64 * 0.1).collect();
        let args = vec![buf(data)];

        let tree_result = eval_elementwise(&expr, &args).unwrap();
        let flat = compile(&expr);
        let flat_result = eval_flat_elementwise(&flat, &args).unwrap();
        assert_eq!(tree_result.as_f64_slice(), flat_result.as_f64_slice());
    }

    // ─── boundary values ───

    #[test]
    fn eval_empty_buffer() {
        let flat = compile(&Expr::ArgRef(0));
        let args = vec![buf(vec![])];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn eval_single_element() {
        let flat = compile(&Expr::ArgRef(0));
        let args = vec![buf(vec![42.0])];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        assert_eq!(result.as_f64_slice(), &[42.0]);
    }

    #[test]
    fn eval_nan_propagation() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
        );
        let flat = compile(&expr);
        let args = vec![buf(vec![f64::NAN])];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        assert!(result.as_f64_slice()[0].is_nan());
    }

    #[test]
    fn eval_inf() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
        );
        let flat = compile(&expr);
        let args = vec![buf(vec![f64::INFINITY, f64::NEG_INFINITY])];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        assert_eq!(result.as_f64_slice()[0], f64::INFINITY);
        assert_eq!(result.as_f64_slice()[1], f64::NEG_INFINITY);
    }

    #[test]
    fn eval_zero_div() {
        let expr = Expr::Binary(
            BinaryOp::Div,
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::ArgRef(0)),
        );
        let flat = compile(&expr);
        let args = vec![buf(vec![0.0])];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        assert_eq!(result.as_f64_slice()[0], f64::INFINITY);
    }

    #[test]
    fn eval_deeply_nested_neg() {
        let depth = 50;
        let mut expr = Expr::ArgRef(0);
        for _ in 0..depth {
            expr = Expr::Unary(UnaryOp::Neg, Box::new(expr));
        }
        let flat = compile(&expr);
        let args = vec![buf(vec![1.0])];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        let expected = if depth % 2 == 0 { 1.0 } else { -1.0 };
        assert_eq!(result.as_f64_slice()[0], expected);
    }

    #[test]
    fn eval_at_par_threshold() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
        );
        let flat = compile(&expr);
        let data: Vec<f64> = (0..4096).map(|i| i as f64).collect();
        let args = vec![buf(data)];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        let out = result.as_f64_slice();
        assert_eq!(out[0], 1.0);
        assert_eq!(out[4095], 4096.0);
    }

    #[test]
    fn eval_at_par_threshold_minus_1() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
        );
        let flat = compile(&expr);
        let data: Vec<f64> = (0..4095).map(|i| i as f64).collect();
        let args = vec![buf(data)];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        let out = result.as_f64_slice();
        assert_eq!(out[0], 1.0);
        assert_eq!(out[4094], 4095.0);
    }

    #[test]
    fn eval_1m_elements() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::Binary(
                BinaryOp::Mul,
                Box::new(Expr::ArgRef(0)),
                Box::new(Expr::ArgRef(1)),
            )),
            Box::new(Expr::ArgRef(2)),
        );
        let flat = compile(&expr);
        let x: Vec<f64> = (0..1_000_000).map(|i| i as f64).collect();
        let y = vec![1.0; 1_000_000];
        let args = vec![scalar(2.0), buf(x), buf(y)];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        let out = result.as_f64_slice();
        assert_eq!(out[0], 1.0); // 2*0+1
        assert_eq!(out[999_999], 2.0 * 999_999.0 + 1.0);
    }

    #[test]
    fn flat_eval_length_boundaries_preserve_results_why_parallel_threshold_must_not_change_semantics(
    ) {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::Binary(
                BinaryOp::Mul,
                Box::new(Expr::ArgRef(0)),
                Box::new(Expr::ArgRef(1)),
            )),
            Box::new(Expr::ArgRef(2)),
        );
        let flat = compile(&expr);

        for len in [0usize, 1, 4095, 4096, 4097] {
            let x: Vec<f64> = (0..len).map(|i| i as f64).collect();
            let y = vec![1.0; len];
            let args = vec![scalar(2.0), buf(x), buf(y)];
            let result = eval_flat_elementwise(&flat, &args).unwrap();
            let expected: Vec<f64> = (0..len).map(|i| 2.0 * i as f64 + 1.0).collect();
            assert_eq!(
                result.as_f64_slice(),
                expected.as_slice(),
                "len={len} must preserve elementwise semantics across the threshold"
            );
        }
    }

    #[test]
    fn flat_eval_special_values_preserve_signals_why_numeric_regressions_must_surface_immediately()
    {
        let flat = compile(&Expr::ArgRef(0));
        let large_finite = f64::MAX / 2.0;
        let args = vec![buf(vec![
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            large_finite,
        ])];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        let out = result.as_f64_slice();

        assert!(out[0].is_nan());
        assert_eq!(out[1], f64::INFINITY);
        assert_eq!(out[2], f64::NEG_INFINITY);
        assert_eq!(out[3], 0.0);
        assert!(out[3].is_sign_negative());
        assert_eq!(out[4], large_finite);
    }

    #[test]
    fn eval_shape_mismatch() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(1)),
        );
        let flat = compile(&expr);
        let args = vec![buf(vec![1.0, 2.0]), buf(vec![1.0, 2.0, 3.0])];
        let err = eval_flat_elementwise(&flat, &args).unwrap_err();
        assert!(matches!(err, ParsecError::ShapeError(_)));
    }

    #[test]
    fn eval_no_buffer_args() {
        let flat = compile(&Expr::ArgRef(0));
        let args = vec![scalar(1.0)];
        let err = eval_flat_elementwise(&flat, &args).unwrap_err();
        assert!(matches!(err, ParsecError::ArgError(_)));
    }

    // ─── all unary ops ───

    #[test]
    fn eval_all_unary_ops() {
        let ops_and_expected: Vec<(UnaryOp, f64, f64)> = vec![
            (UnaryOp::Neg, 3.0, -3.0),
            (UnaryOp::Abs, -3.0, 3.0),
            (UnaryOp::Sqrt, 9.0, 3.0),
            (UnaryOp::Log, 1.0_f64.exp(), 1.0),
            (UnaryOp::Exp, 0.0, 1.0),
            (UnaryOp::Log2, 8.0, 3.0),
            (UnaryOp::Log10, 1000.0, 3.0),
            (UnaryOp::Floor, 1.7, 1.0),
            (UnaryOp::Ceil, 1.3, 2.0),
            (UnaryOp::Round, 1.5, 2.0),
            (UnaryOp::Sin, 0.0, 0.0),
            (UnaryOp::Cos, 0.0, 1.0),
            (UnaryOp::Tan, 0.0, 0.0),
        ];
        for (op, input, expected) in ops_and_expected {
            let expr = Expr::Unary(op, Box::new(Expr::ArgRef(0)));
            let flat = compile(&expr);
            let args = vec![buf(vec![input])];
            let result = eval_flat_elementwise(&flat, &args).unwrap();
            let actual = result.as_f64_slice()[0];
            assert!(
                (actual - expected).abs() < 1e-10,
                "{op:?}({input}) = {actual}, expected {expected}"
            );
        }
    }

    // ─── all binary ops ───

    #[test]
    fn eval_all_binary_ops() {
        let ops_and_expected: Vec<(BinaryOp, f64, f64, f64)> = vec![
            (BinaryOp::Add, 2.0, 3.0, 5.0),
            (BinaryOp::Sub, 5.0, 3.0, 2.0),
            (BinaryOp::Mul, 3.0, 4.0, 12.0),
            (BinaryOp::Div, 10.0, 2.0, 5.0),
            (BinaryOp::Pow, 2.0, 3.0, 8.0),
            (BinaryOp::Atan2, 0.0, 0.0, 0.0),
            (BinaryOp::Min, 3.0, 5.0, 3.0),
            (BinaryOp::Max, 3.0, 5.0, 5.0),
        ];
        for (op, l, r, expected) in ops_and_expected {
            let expr = Expr::Binary(op, Box::new(Expr::ArgRef(0)), Box::new(Expr::ArgRef(1)));
            let flat = compile(&expr);
            let args = vec![buf(vec![l]), buf(vec![r])];
            let result = eval_flat_elementwise(&flat, &args).unwrap();
            let actual = result.as_f64_slice()[0];
            assert!(
                (actual - expected).abs() < 1e-10,
                "{op:?}({l}, {r}) = {actual}, expected {expected}"
            );
        }
    }

    // ─── all compare ops ───

    #[test]
    fn eval_all_compare_ops() {
        let ops_and_expected: Vec<(CmpOp, f64, f64, f64)> = vec![
            (CmpOp::Gt, 3.0, 2.0, 1.0),
            (CmpOp::Gt, 2.0, 3.0, 0.0),
            (CmpOp::Ge, 3.0, 3.0, 1.0),
            (CmpOp::Ge, 2.0, 3.0, 0.0),
            (CmpOp::Lt, 2.0, 3.0, 1.0),
            (CmpOp::Lt, 3.0, 2.0, 0.0),
            (CmpOp::Le, 3.0, 3.0, 1.0),
            (CmpOp::Le, 4.0, 3.0, 0.0),
            (CmpOp::Eq, 3.0, 3.0, 1.0),
            (CmpOp::Eq, 3.0, 4.0, 0.0),
            (CmpOp::Ne, 3.0, 4.0, 1.0),
            (CmpOp::Ne, 3.0, 3.0, 0.0),
        ];
        for (op, l, r, expected) in ops_and_expected {
            let expr = Expr::Compare(op, Box::new(Expr::ArgRef(0)), Box::new(Expr::ArgRef(1)));
            let flat = compile(&expr);
            let args = vec![buf(vec![l]), buf(vec![r])];
            let result = eval_flat_elementwise(&flat, &args).unwrap();
            let actual = result.as_f64_slice()[0];
            assert_eq!(
                actual, expected,
                "{op:?}({l}, {r}) = {actual}, expected {expected}"
            );
        }
    }

    // ─── scratch stack reuse ───

    #[test]
    fn scratch_stack_reuse_sequential_produces_correct_results() {
        // Sequential path (< 4096 elements): scratch is reused across iterations.
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::Binary(
                BinaryOp::Mul,
                Box::new(Expr::ArgRef(0)),
                Box::new(Expr::ArgRef(1)),
            )),
            Box::new(Expr::ArgRef(2)),
        );
        let flat = compile(&expr);
        let len = 100;
        let x: Vec<f64> = (0..len).map(|i| i as f64).collect();
        let y = vec![1.0; len];
        let args = vec![scalar(3.0), buf(x), buf(y)];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        let out = result.as_f64_slice();
        for (i, value) in out.iter().enumerate() {
            assert_eq!(*value, 3.0 * i as f64 + 1.0, "mismatch at index {i}");
        }
    }

    #[test]
    fn scratch_stack_reuse_parallel_produces_correct_results() {
        // Parallel path (>= 4096 elements): each rayon task gets its own scratch via map_init.
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::Binary(
                BinaryOp::Mul,
                Box::new(Expr::ArgRef(0)),
                Box::new(Expr::ArgRef(1)),
            )),
            Box::new(Expr::ArgRef(2)),
        );
        let flat = compile(&expr);
        let len = 8192;
        let x: Vec<f64> = (0..len).map(|i| i as f64).collect();
        let y = vec![1.0; len];
        let args = vec![scalar(3.0), buf(x), buf(y)];
        let result = eval_flat_elementwise(&flat, &args).unwrap();
        let out = result.as_f64_slice();
        for (i, value) in out.iter().enumerate() {
            assert_eq!(*value, 3.0 * i as f64 + 1.0, "mismatch at index {i}");
        }
    }

    proptest! {
        #[test]
        fn reduce_sum_of_concatenated_buffers_matches_segment_sums_why_reduce_rounding_drift_must_stay_bounded(
            a in finite_vec_strategy(32),
            b in finite_vec_strategy(32),
        ) {
            let mut combined = a.clone();
            combined.extend_from_slice(&b);

            let combined_sum = crate::ir::compiler::eval_reduce(
                ReduceOp::Sum,
                &Buffer::from_f64_vec(combined),
            ).unwrap();
            let segmented_sum =
                crate::ir::compiler::eval_reduce(ReduceOp::Sum, &Buffer::from_f64_vec(a.clone())).unwrap()
                + crate::ir::compiler::eval_reduce(ReduceOp::Sum, &Buffer::from_f64_vec(b.clone())).unwrap();

            let tolerance = 1e-9 * (a.len() + b.len()) as f64;
            prop_assert!(
                (combined_sum - segmented_sum).abs() <= tolerance,
                "combined={combined_sum}, segmented={segmented_sum}, tolerance={tolerance}"
            );
        }

        #[test]
        fn identity_map_preserves_values_why_flat_eval_must_not_mutate_buffer_contents(
            data in finite_vec_strategy(64),
        ) {
            let flat = compile(&Expr::ArgRef(0));
            let result = eval_flat_elementwise(&flat, &[buf(data.clone())]).unwrap();
            prop_assert_eq!(result.as_f64_slice(), data.as_slice());
        }

        #[test]
        fn output_length_matches_input_length_when_shapes_agree_why_buffer_contract_must_stay_predictable(
            (left, right) in same_length_vec_pair_strategy(64),
        ) {
            let expr = Expr::Binary(
                BinaryOp::Add,
                Box::new(Expr::ArgRef(0)),
                Box::new(Expr::ArgRef(1)),
            );
            let flat = compile(&expr);
            let expected_len = left.len();
            let result = eval_flat_elementwise(&flat, &[buf(left), buf(right)]).unwrap();
            prop_assert_eq!(result.len(), expected_len);
        }
    }
}
