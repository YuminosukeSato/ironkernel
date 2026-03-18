use std::sync::OnceLock;

use crate::buffer::inner::Buffer;
use crate::error::ParsecResult;
use crate::ir::cache::CompileCache;
use crate::ir::expr::Expr;
use crate::ir::flat::FlatExpr;
use crate::ir::kernel::ReduceOp;

static GLOBAL_CACHE: OnceLock<CompileCache> = OnceLock::new();

fn global_cache() -> &'static CompileCache {
    GLOBAL_CACHE.get_or_init(CompileCache::new)
}

/// Scalar value: either f64 or a buffer arg reference.
#[derive(Debug, Clone)]
pub enum ArgValue {
    Scalar(f64),
    Buffer(Buffer),
}

#[derive(Debug, Clone)]
pub(crate) enum CompiledProgram {
    Flat(FlatExpr),
    LazyTree(Expr),
}

pub(crate) fn compile_program(expr: &Expr) -> CompiledProgram {
    if contains_select(expr) {
        CompiledProgram::LazyTree(expr.clone())
    } else {
        CompiledProgram::Flat(crate::ir::flat::compile(expr))
    }
}

/// Evaluate an elementwise expression over input arguments, producing a new Buffer.
///
/// Compilation results are cached by expression structure via [`CompileCache`].
/// Expressions without `Select` use the flat interpreter.
/// Expressions with `Select` use a lazy tree evaluator so dead branches are not evaluated.
pub fn eval_elementwise(expr: &Expr, args: &[ArgValue]) -> ParsecResult<Buffer> {
    let program = global_cache().get_or_compile(expr);
    match program.as_ref() {
        CompiledProgram::Flat(flat) => crate::ir::flat::eval_flat_elementwise(flat, args),
        CompiledProgram::LazyTree(tree) => crate::ir::evaluator::eval_lazy_elementwise(tree, args),
    }
}

fn contains_select(expr: &Expr) -> bool {
    match expr {
        Expr::Const(_) | Expr::ArgRef(_) => false,
        Expr::Unary(_, inner) => contains_select(inner),
        Expr::Binary(_, lhs, rhs) | Expr::Compare(_, lhs, rhs) => {
            contains_select(lhs) || contains_select(rhs)
        }
        Expr::Select(_, _, _) => true,
    }
}

/// Evaluate a reduce operation over a buffer.
pub fn eval_reduce(op: ReduceOp, buf: &Buffer) -> ParsecResult<f64> {
    buf.require_non_empty(&format!("{op:?}"))?;
    let data = buf.as_f64_slice();

    match op {
        ReduceOp::Sum => Ok(data.iter().sum()),
        ReduceOp::Mean => {
            let sum: f64 = data.iter().sum();
            Ok(sum / data.len() as f64)
        }
        ReduceOp::Min => Ok(data.iter().copied().fold(f64::INFINITY, |a, b| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        })),
        ReduceOp::Max => Ok(data.iter().copied().fold(f64::NEG_INFINITY, |a, b| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })),
        ReduceOp::Count => Ok(data.len() as f64),
        ReduceOp::Variance => {
            let n = data.len() as f64;
            let mean = data.iter().sum::<f64>() / n;
            let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            Ok(var)
        }
        ReduceOp::Std => {
            let n = data.len() as f64;
            let mean = data.iter().sum::<f64>() / n;
            let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            Ok(var.sqrt())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ParsecError;
    use crate::ir::expr::{BinaryOp, CmpOp, UnaryOp};
    use proptest::strategy::Strategy;

    fn buf(data: Vec<f64>) -> ArgValue {
        ArgValue::Buffer(Buffer::from_f64_vec(data))
    }

    fn scalar(v: f64) -> ArgValue {
        ArgValue::Scalar(v)
    }

    // --- Elementwise: a * x + y (saxpy) ---

    #[test]
    fn saxpy_basic() {
        // a=2.0, x=[0,1,2], y=[10,20,30] → [10,22,34]
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
            buf(vec![0.0, 1.0, 2.0]),
            buf(vec![10.0, 20.0, 30.0]),
        ];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert_eq!(result.as_f64_slice(), &[10.0, 22.0, 34.0]);
    }

    #[test]
    fn saxpy_1m_elements() {
        let x_data: Vec<f64> = (0..1_000_000).map(|i| i as f64).collect();
        let y_data: Vec<f64> = vec![1.0; 1_000_000];
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::Binary(
                BinaryOp::Mul,
                Box::new(Expr::ArgRef(0)),
                Box::new(Expr::ArgRef(1)),
            )),
            Box::new(Expr::ArgRef(2)),
        );
        let args = vec![scalar(2.0), buf(x_data), buf(y_data)];
        let result = eval_elementwise(&expr, &args).unwrap();
        let out = result.as_f64_slice();
        assert_eq!(out[0], 1.0); // 2.0 * 0.0 + 1.0
        assert_eq!(out[999_999], 2.0 * 999_999.0 + 1.0);
    }

    // --- Boundary: NaN, Inf ---

    #[test]
    fn elementwise_nan_propagation() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
        );
        let args = vec![buf(vec![f64::NAN])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert!(result.as_f64_slice()[0].is_nan());
    }

    #[test]
    fn elementwise_inf() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
        );
        let args = vec![buf(vec![f64::INFINITY])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert_eq!(result.as_f64_slice()[0], f64::INFINITY);
    }

    #[test]
    fn elementwise_zero_div_zero() {
        let expr = Expr::Binary(
            BinaryOp::Div,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(1)),
        );
        let args = vec![buf(vec![0.0]), buf(vec![0.0])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert!(result.as_f64_slice()[0].is_nan());
    }

    #[test]
    fn elementwise_one_div_zero() {
        let expr = Expr::Binary(
            BinaryOp::Div,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(1)),
        );
        let args = vec![buf(vec![1.0]), buf(vec![0.0])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert_eq!(result.as_f64_slice()[0], f64::INFINITY);
    }

    #[test]
    fn elementwise_sqrt_neg() {
        let expr = Expr::Unary(UnaryOp::Sqrt, Box::new(Expr::ArgRef(0)));
        let args = vec![buf(vec![-1.0])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert!(result.as_f64_slice()[0].is_nan());
    }

    // --- Shape mismatch ---

    #[test]
    fn elementwise_shape_mismatch() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(1)),
        );
        let args = vec![buf(vec![1.0, 2.0]), buf(vec![1.0, 2.0, 3.0])];
        let err = eval_elementwise(&expr, &args).unwrap_err();
        assert!(matches!(err, ParsecError::ShapeError(_)));
    }

    #[test]
    fn elementwise_no_buffer_args() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(1)),
        );
        let args = vec![scalar(1.0), scalar(2.0)];
        let err = eval_elementwise(&expr, &args).unwrap_err();
        assert!(matches!(err, ParsecError::ArgError(_)));
    }

    #[test]
    fn elementwise_empty_buffer() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
        );
        let args = vec![buf(vec![])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert!(result.is_empty());
    }

    // --- Select (where) ---

    #[test]
    fn select_relu() {
        // where(x > 0, x, 0)
        let x = Expr::ArgRef(0);
        let cond = Expr::Compare(CmpOp::Gt, Box::new(x.clone()), Box::new(Expr::Const(0.0)));
        let expr = Expr::Select(Box::new(cond), Box::new(x), Box::new(Expr::Const(0.0)));
        let args = vec![buf(vec![-2.0, -1.0, 0.0, 1.0, 2.0])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert_eq!(result.as_f64_slice(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn select_lazy_does_not_evaluate_false_branch() {
        let x = Expr::ArgRef(0);
        let cond = Expr::Compare(CmpOp::Gt, Box::new(x.clone()), Box::new(Expr::Const(0.0)));
        let expr = Expr::Select(
            Box::new(cond),
            Box::new(Expr::Unary(UnaryOp::Sqrt, Box::new(x))),
            Box::new(Expr::Const(0.0)),
        );
        let args = vec![buf(vec![-1.0, -4.0])];

        let result = eval_elementwise(&expr, &args).unwrap();

        assert_eq!(result.as_f64_slice(), &[0.0, 0.0]);
    }

    #[test]
    fn select_nan_in_unselected_branch_no_contamination() {
        let cond = Expr::ArgRef(0);
        let expr = Expr::Select(
            Box::new(cond),
            Box::new(Expr::ArgRef(1)),
            Box::new(Expr::ArgRef(2)),
        );
        let args = vec![
            buf(vec![1.0, 1.0]),
            buf(vec![5.0, 6.0]),
            buf(vec![f64::NAN, f64::NAN]),
        ];

        let result = eval_elementwise(&expr, &args).unwrap();

        assert_eq!(result.as_f64_slice(), &[5.0, 6.0]);
    }

    #[test]
    fn select_mixed_elements_lazy() {
        let cond = Expr::ArgRef(0);
        let expr = Expr::Select(
            Box::new(cond),
            Box::new(Expr::ArgRef(1)),
            Box::new(Expr::ArgRef(2)),
        );
        let args = vec![
            buf(vec![1.0, 0.0, 1.0, 0.0]),
            buf(vec![10.0, 20.0, 30.0, 40.0]),
            buf(vec![100.0, 200.0, 300.0, 400.0]),
        ];

        let result = eval_elementwise(&expr, &args).unwrap();

        assert_eq!(result.as_f64_slice(), &[10.0, 200.0, 30.0, 400.0]);
    }

    #[test]
    fn select_empty_buffer() {
        let cond = Expr::ArgRef(0);
        let expr = Expr::Select(
            Box::new(cond),
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Const(0.0)),
        );
        let args = vec![buf(vec![])];

        let result = eval_elementwise(&expr, &args).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn select_single_element() {
        let cond = Expr::ArgRef(0);
        let expr = Expr::Select(
            Box::new(cond),
            Box::new(Expr::Const(7.0)),
            Box::new(Expr::Const(9.0)),
        );
        let args = vec![buf(vec![0.0])];

        let result = eval_elementwise(&expr, &args).unwrap();

        assert_eq!(result.as_f64_slice(), &[9.0]);
    }

    // --- Reduce ---

    #[test]
    fn reduce_sum() {
        let b = Buffer::from_f64_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(eval_reduce(ReduceOp::Sum, &b).unwrap(), 10.0);
    }

    #[test]
    fn reduce_sum_large() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let b = Buffer::from_f64_vec(data);
        assert_eq!(eval_reduce(ReduceOp::Sum, &b).unwrap(), 4950.0);
    }

    #[test]
    fn reduce_mean() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let b = Buffer::from_f64_vec(data);
        assert_eq!(eval_reduce(ReduceOp::Mean, &b).unwrap(), 49.5);
    }

    #[test]
    fn reduce_min() {
        let b = Buffer::from_f64_vec(vec![3.0, 1.0, 4.0, 1.5]);
        assert_eq!(eval_reduce(ReduceOp::Min, &b).unwrap(), 1.0);
    }

    #[test]
    fn reduce_max() {
        let b = Buffer::from_f64_vec(vec![3.0, 1.0, 4.0, 1.5]);
        assert_eq!(eval_reduce(ReduceOp::Max, &b).unwrap(), 4.0);
    }

    #[test]
    fn reduce_count() {
        let b = Buffer::from_f64_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(eval_reduce(ReduceOp::Count, &b).unwrap(), 3.0);
    }

    #[test]
    fn reduce_variance() {
        let b = Buffer::from_f64_vec(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let var = eval_reduce(ReduceOp::Variance, &b).unwrap();
        assert!((var - 4.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_std() {
        let b = Buffer::from_f64_vec(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let std = eval_reduce(ReduceOp::Std, &b).unwrap();
        assert!((std - 2.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_empty_error() {
        let b = Buffer::empty_f64();
        let err = eval_reduce(ReduceOp::Sum, &b).unwrap_err();
        assert!(matches!(err, ParsecError::EmptyCollection(_)));
    }

    #[test]
    fn reduce_single_element() {
        let b = Buffer::from_f64_vec(vec![42.0]);
        assert_eq!(eval_reduce(ReduceOp::Sum, &b).unwrap(), 42.0);
        assert_eq!(eval_reduce(ReduceOp::Mean, &b).unwrap(), 42.0);
        assert_eq!(eval_reduce(ReduceOp::Min, &b).unwrap(), 42.0);
        assert_eq!(eval_reduce(ReduceOp::Max, &b).unwrap(), 42.0);
        assert_eq!(eval_reduce(ReduceOp::Variance, &b).unwrap(), 0.0);
        assert_eq!(eval_reduce(ReduceOp::Std, &b).unwrap(), 0.0);
    }

    #[test]
    fn reduce_sum_with_nan() {
        let b = Buffer::from_f64_vec(vec![f64::NAN, 1.0]);
        assert!(eval_reduce(ReduceOp::Sum, &b).unwrap().is_nan());
    }

    #[test]
    fn reduce_min_with_nan() {
        let b = Buffer::from_f64_vec(vec![f64::NAN, 1.0]);
        // f64::min propagates NaN
        assert!(eval_reduce(ReduceOp::Min, &b).unwrap().is_nan());
    }

    #[test]
    fn reduce_max_with_nan() {
        let b = Buffer::from_f64_vec(vec![f64::NAN, 1.0]);
        assert!(eval_reduce(ReduceOp::Max, &b).unwrap().is_nan());
    }

    #[test]
    fn reduce_sum_with_inf() {
        let b = Buffer::from_f64_vec(vec![f64::INFINITY, 1.0]);
        assert_eq!(eval_reduce(ReduceOp::Sum, &b).unwrap(), f64::INFINITY);
    }

    // --- Math functions ---

    #[test]
    fn math_log_zero() {
        let expr = Expr::Unary(UnaryOp::Log, Box::new(Expr::ArgRef(0)));
        let args = vec![buf(vec![0.0])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert_eq!(result.as_f64_slice()[0], f64::NEG_INFINITY);
    }

    #[test]
    fn math_log_neg() {
        let expr = Expr::Unary(UnaryOp::Log, Box::new(Expr::ArgRef(0)));
        let args = vec![buf(vec![-1.0])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert!(result.as_f64_slice()[0].is_nan());
    }

    #[test]
    fn math_sin_inf() {
        let expr = Expr::Unary(UnaryOp::Sin, Box::new(Expr::ArgRef(0)));
        let args = vec![buf(vec![f64::INFINITY])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert!(result.as_f64_slice()[0].is_nan());
    }

    #[test]
    fn math_atan2_zero_zero() {
        let expr = Expr::Binary(
            BinaryOp::Atan2,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(1)),
        );
        let args = vec![buf(vec![0.0]), buf(vec![0.0])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert_eq!(result.as_f64_slice()[0], 0.0);
    }

    #[test]
    fn math_pow_zero_zero() {
        let expr = Expr::Binary(
            BinaryOp::Pow,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(1)),
        );
        let args = vec![buf(vec![0.0]), buf(vec![0.0])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert_eq!(result.as_f64_slice()[0], 1.0);
    }

    #[test]
    fn math_pow_neg_half() {
        let expr = Expr::Binary(
            BinaryOp::Pow,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(1)),
        );
        let args = vec![buf(vec![-1.0]), buf(vec![0.5])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert!(result.as_f64_slice()[0].is_nan());
    }

    #[test]
    fn math_exp_large() {
        let expr = Expr::Unary(UnaryOp::Exp, Box::new(Expr::ArgRef(0)));
        let args = vec![buf(vec![1000.0])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert_eq!(result.as_f64_slice()[0], f64::INFINITY);
    }

    #[test]
    fn math_floor_ceil_round() {
        let test_val = vec![1.3, 1.5, 1.7, -1.3, -1.5, -1.7];

        let expr = Expr::Unary(UnaryOp::Floor, Box::new(Expr::ArgRef(0)));
        let result = eval_elementwise(&expr, &[buf(test_val.clone())]).unwrap();
        assert_eq!(result.as_f64_slice(), &[1.0, 1.0, 1.0, -2.0, -2.0, -2.0]);

        let expr = Expr::Unary(UnaryOp::Ceil, Box::new(Expr::ArgRef(0)));
        let result = eval_elementwise(&expr, &[buf(test_val.clone())]).unwrap();
        assert_eq!(result.as_f64_slice(), &[2.0, 2.0, 2.0, -1.0, -1.0, -1.0]);

        let expr = Expr::Unary(UnaryOp::Round, Box::new(Expr::ArgRef(0)));
        let result = eval_elementwise(&expr, &[buf(test_val)]).unwrap();
        assert_eq!(result.as_f64_slice(), &[1.0, 2.0, 2.0, -1.0, -2.0, -2.0]);
    }

    // --- Select (where): proptest ---

    #[test]
    fn select_all_true() {
        let cond = Expr::ArgRef(0);
        let expr = Expr::Select(
            Box::new(cond),
            Box::new(Expr::Const(7.0)),
            Box::new(Expr::Const(9.0)),
        );
        let args = vec![buf(vec![1.0, 2.0, 3.0])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert_eq!(result.as_f64_slice(), &[7.0, 7.0, 7.0]);
    }

    #[test]
    fn select_all_false() {
        let cond = Expr::ArgRef(0);
        let expr = Expr::Select(
            Box::new(cond),
            Box::new(Expr::Const(7.0)),
            Box::new(Expr::Const(9.0)),
        );
        let args = vec![buf(vec![0.0, 0.0, 0.0])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert_eq!(result.as_f64_slice(), &[9.0, 9.0, 9.0]);
    }

    #[test]
    fn math_abs() {
        let expr = Expr::Unary(UnaryOp::Abs, Box::new(Expr::ArgRef(0)));
        let args = vec![buf(vec![-3.0, 0.0, 3.0])];
        let result = eval_elementwise(&expr, &args).unwrap();
        assert_eq!(result.as_f64_slice(), &[3.0, 0.0, 3.0]);
    }

    #[test]
    fn math_composite_sqrt_abs_plus_sin() {
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
        let args = vec![buf(vec![4.0])];
        let result = eval_elementwise(&expr, &args).unwrap();
        let expected = 4.0_f64.sqrt() + 4.0_f64.sin();
        assert!((result.as_f64_slice()[0] - expected).abs() < 1e-10);
    }

    // --- compile_program dispatch ---

    #[test]
    fn compile_program_flat_for_non_select() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
        );
        assert!(matches!(compile_program(&expr), CompiledProgram::Flat(_)));
    }

    #[test]
    fn compile_program_lazy_for_select() {
        let expr = Expr::Select(
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Const(0.0)),
        );
        assert!(matches!(
            compile_program(&expr),
            CompiledProgram::LazyTree(_)
        ));
    }

    #[test]
    fn compile_program_lazy_for_nested_select() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::Select(
                Box::new(Expr::ArgRef(0)),
                Box::new(Expr::ArgRef(1)),
                Box::new(Expr::Const(0.0)),
            )),
            Box::new(Expr::Const(1.0)),
        );
        assert!(matches!(
            compile_program(&expr),
            CompiledProgram::LazyTree(_)
        ));
    }

    // --- proptest ---

    proptest::prop_compose! {
        fn same_length_vec_pair_strategy(max_len: usize)
            (len in 1usize..=max_len)
            (
                left in proptest::collection::vec(-1.0e6f64..1.0e6f64, len),
                right in proptest::collection::vec(-1.0e6f64..1.0e6f64, len),
            ) -> (Vec<f64>, Vec<f64>) {
                (left, right)
            }
    }

    proptest::proptest! {
        #[test]
        fn where_selects_correct_branch(
            (cond_data, true_data, false_data) in proptest::collection::vec(0.0f64..=1.0f64, 1..=64usize).prop_flat_map(|cond: Vec<f64>| {
                let len = cond.len();
                (
                    proptest::strategy::Just(cond),
                    proptest::collection::vec(-1.0e6f64..1.0e6f64, len),
                    proptest::collection::vec(-1.0e6f64..1.0e6f64, len),
                )
            }),
        ) {
            let cond_bools: Vec<f64> = cond_data.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect();
            let expr = Expr::Select(
                Box::new(Expr::ArgRef(0)),
                Box::new(Expr::ArgRef(1)),
                Box::new(Expr::ArgRef(2)),
            );
            let result = eval_elementwise(&expr, &[buf(cond_bools.clone()), buf(true_data.clone()), buf(false_data.clone())]).unwrap();
            let out = result.as_f64_slice();
            for i in 0..cond_bools.len() {
                let expected = if cond_bools[i] != 0.0 { true_data[i] } else { false_data[i] };
                proptest::prop_assert!(
                    (out[i] - expected).abs() < 1e-10,
                    "index={i}, cond={}, expected={expected}, got={}",
                    cond_bools[i], out[i]
                );
            }
        }

        #[test]
        fn where_with_identical_branches_returns_original_values(
            (cond, values) in same_length_vec_pair_strategy(64),
        ) {
            let expr = Expr::Select(
                Box::new(Expr::ArgRef(0)),
                Box::new(Expr::ArgRef(1)),
                Box::new(Expr::ArgRef(1)),
            );
            let result = eval_elementwise(&expr, &[buf(cond), buf(values.clone())]).unwrap();
            proptest::prop_assert_eq!(result.as_f64_slice(), values.as_slice());
        }
    }
}
