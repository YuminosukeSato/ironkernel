use rayon::prelude::*;

use crate::buffer::inner::Buffer;
use crate::error::{ParsecError, ParsecResult};
use crate::ir::expr::{BinaryOp, CmpOp, Expr, UnaryOp};
use crate::ir::kernel::ReduceOp;

/// Scalar value: either f64 or a buffer arg reference.
#[derive(Debug, Clone)]
pub enum ArgValue {
    Scalar(f64),
    Buffer(Buffer),
}

/// Evaluate an elementwise expression over input arguments, producing a new Buffer.
///
/// Each element of the output is computed by walking the Expr tree with the
/// corresponding element values from the input arrays.
pub fn eval_elementwise(expr: &Expr, args: &[ArgValue]) -> ParsecResult<Buffer> {
    let len = infer_output_len(args)?;

    if len == 0 {
        return Ok(Buffer::empty_f64());
    }

    // Extract f64 slices for buffer args, scalars as-is
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
            .map(|i| eval_expr_at(expr, i, &arg_slices, args))
            .collect()
    } else {
        (0..len)
            .map(|i| eval_expr_at(expr, i, &arg_slices, args))
            .collect()
    };

    Ok(Buffer::from_f64_vec(result))
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

/// Determine output length from arguments.
/// All buffer args must have the same length. Scalar-only → error.
fn infer_output_len(args: &[ArgValue]) -> ParsecResult<usize> {
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

/// Evaluate expression at element index `i`.
#[inline]
fn eval_expr_at(expr: &Expr, i: usize, slices: &[Option<&[f64]>], args: &[ArgValue]) -> f64 {
    match expr {
        Expr::Const(v) => *v,
        Expr::ArgRef(idx) => match &args[*idx] {
            ArgValue::Scalar(v) => *v,
            ArgValue::Buffer(_) => slices[*idx].unwrap()[i],
        },
        Expr::Unary(op, inner) => {
            let v = eval_expr_at(inner, i, slices, args);
            eval_unary(*op, v)
        }
        Expr::Binary(op, lhs, rhs) => {
            let l = eval_expr_at(lhs, i, slices, args);
            let r = eval_expr_at(rhs, i, slices, args);
            eval_binary(*op, l, r)
        }
        Expr::Compare(op, lhs, rhs) => {
            let l = eval_expr_at(lhs, i, slices, args);
            let r = eval_expr_at(rhs, i, slices, args);
            if eval_cmp(*op, l, r) {
                1.0
            } else {
                0.0
            }
        }
        Expr::Select(cond, t, f) => {
            let c = eval_expr_at(cond, i, slices, args);
            if c != 0.0 {
                eval_expr_at(t, i, slices, args)
            } else {
                eval_expr_at(f, i, slices, args)
            }
        }
    }
}

#[inline]
fn eval_unary(op: UnaryOp, v: f64) -> f64 {
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
fn eval_binary(op: BinaryOp, l: f64, r: f64) -> f64 {
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
fn eval_cmp(op: CmpOp, l: f64, r: f64) -> bool {
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
}
