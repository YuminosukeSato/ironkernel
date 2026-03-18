use rayon::prelude::*;

use crate::buffer::inner::Buffer;
use crate::error::ParsecResult;
use crate::ir::compiler::ArgValue;
use crate::ir::expr::Expr;
use crate::ir::flat::{apply_binary, apply_cmp, apply_unary, infer_output_len};

const PAR_THRESHOLD: usize = 4096;

pub(crate) fn eval_lazy_elementwise(expr: &Expr, args: &[ArgValue]) -> ParsecResult<Buffer> {
    let len = infer_output_len(args)?;

    if len == 0 {
        return Ok(Buffer::empty_f64());
    }

    let arg_slices: Vec<Option<&[f64]>> = args
        .iter()
        .map(|arg| match arg {
            ArgValue::Scalar(_) => None,
            ArgValue::Buffer(buf) => Some(buf.as_f64_slice()),
        })
        .collect();

    let result = if len >= PAR_THRESHOLD {
        (0..len)
            .into_par_iter()
            .map(|index| eval_lazy_at(expr, index, &arg_slices, args))
            .collect()
    } else {
        (0..len)
            .map(|index| eval_lazy_at(expr, index, &arg_slices, args))
            .collect()
    };

    Ok(Buffer::from_f64_vec(result))
}

fn eval_lazy_at(expr: &Expr, index: usize, slices: &[Option<&[f64]>], args: &[ArgValue]) -> f64 {
    match expr {
        Expr::Const(value) => *value,
        Expr::ArgRef(arg_index) => match &args[*arg_index] {
            ArgValue::Scalar(value) => *value,
            ArgValue::Buffer(_) => slices[*arg_index].unwrap()[index],
        },
        Expr::Unary(op, inner) => apply_unary(*op, eval_lazy_at(inner, index, slices, args)),
        Expr::Binary(op, lhs, rhs) => apply_binary(
            *op,
            eval_lazy_at(lhs, index, slices, args),
            eval_lazy_at(rhs, index, slices, args),
        ),
        Expr::Compare(op, lhs, rhs) => {
            let left = eval_lazy_at(lhs, index, slices, args);
            let right = eval_lazy_at(rhs, index, slices, args);
            if apply_cmp(*op, left, right) {
                1.0
            } else {
                0.0
            }
        }
        Expr::Select(cond, true_branch, false_branch) => {
            let predicate = eval_lazy_at(cond, index, slices, args);
            if predicate != 0.0 {
                eval_lazy_at(true_branch, index, slices, args)
            } else {
                eval_lazy_at(false_branch, index, slices, args)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::compiler::ArgValue;
    use crate::ir::expr::{BinaryOp, CmpOp, UnaryOp};

    fn buf(data: Vec<f64>) -> ArgValue {
        ArgValue::Buffer(Buffer::from_f64_vec(data))
    }

    fn scalar(v: f64) -> ArgValue {
        ArgValue::Scalar(v)
    }

    // --- Basic node types ---

    #[test]
    fn lazy_const() {
        let result = eval_lazy_elementwise(&Expr::Const(42.0), &[buf(vec![0.0])]).unwrap();
        assert_eq!(result.as_f64_slice(), &[42.0]);
    }

    #[test]
    fn lazy_argref_buffer() {
        let result = eval_lazy_elementwise(&Expr::ArgRef(0), &[buf(vec![1.0, 2.0, 3.0])]).unwrap();
        assert_eq!(result.as_f64_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn lazy_argref_scalar() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(1)),
        );
        let result = eval_lazy_elementwise(&expr, &[scalar(10.0), buf(vec![1.0, 2.0])]).unwrap();
        assert_eq!(result.as_f64_slice(), &[11.0, 12.0]);
    }

    #[test]
    fn lazy_unary() {
        let expr = Expr::Unary(UnaryOp::Neg, Box::new(Expr::ArgRef(0)));
        let result = eval_lazy_elementwise(&expr, &[buf(vec![1.0, -2.0])]).unwrap();
        assert_eq!(result.as_f64_slice(), &[-1.0, 2.0]);
    }

    #[test]
    fn lazy_binary() {
        let expr = Expr::Binary(
            BinaryOp::Mul,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(1)),
        );
        let result =
            eval_lazy_elementwise(&expr, &[buf(vec![2.0, 3.0]), buf(vec![4.0, 5.0])]).unwrap();
        assert_eq!(result.as_f64_slice(), &[8.0, 15.0]);
    }

    #[test]
    fn lazy_compare() {
        let expr = Expr::Compare(
            CmpOp::Gt,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.5)),
        );
        let result = eval_lazy_elementwise(&expr, &[buf(vec![1.0, 2.0, 3.0])]).unwrap();
        assert_eq!(result.as_f64_slice(), &[0.0, 1.0, 1.0]);
    }

    #[test]
    fn lazy_compare_false() {
        let expr = Expr::Compare(
            CmpOp::Lt,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(0.0)),
        );
        let result = eval_lazy_elementwise(&expr, &[buf(vec![1.0])]).unwrap();
        assert_eq!(result.as_f64_slice(), &[0.0]);
    }

    #[test]
    fn lazy_select_true_branch() {
        let cond = Expr::Compare(
            CmpOp::Gt,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(0.0)),
        );
        let expr = Expr::Select(
            Box::new(cond),
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(-1.0)),
        );
        let result = eval_lazy_elementwise(&expr, &[buf(vec![5.0])]).unwrap();
        assert_eq!(result.as_f64_slice(), &[5.0]);
    }

    #[test]
    fn lazy_select_false_branch() {
        let cond = Expr::Compare(
            CmpOp::Gt,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(0.0)),
        );
        let expr = Expr::Select(
            Box::new(cond),
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(-1.0)),
        );
        let result = eval_lazy_elementwise(&expr, &[buf(vec![-5.0])]).unwrap();
        assert_eq!(result.as_f64_slice(), &[-1.0]);
    }

    // --- Empty buffer ---

    #[test]
    fn lazy_empty_buffer() {
        let result = eval_lazy_elementwise(&Expr::ArgRef(0), &[buf(vec![])]).unwrap();
        assert!(result.is_empty());
    }

    // --- Parallel threshold ---

    #[test]
    fn lazy_sequential_path() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
        );
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = eval_lazy_elementwise(&expr, &[buf(data)]).unwrap();
        assert_eq!(result.as_f64_slice()[0], 1.0);
        assert_eq!(result.as_f64_slice()[99], 100.0);
    }

    #[test]
    fn lazy_parallel_path() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
        );
        let data: Vec<f64> = (0..5000).map(|i| i as f64).collect();
        let result = eval_lazy_elementwise(&expr, &[buf(data)]).unwrap();
        assert_eq!(result.as_f64_slice()[0], 1.0);
        assert_eq!(result.as_f64_slice()[4999], 5000.0);
    }

    #[test]
    fn lazy_at_threshold_boundary() {
        let expr = Expr::Select(
            Box::new(Expr::Compare(
                CmpOp::Gt,
                Box::new(Expr::ArgRef(0)),
                Box::new(Expr::Const(0.0)),
            )),
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(0.0)),
        );
        for len in [4095, 4096, 4097] {
            let data: Vec<f64> = (0..len).map(|i| i as f64 - 2.0).collect();
            let result = eval_lazy_elementwise(&expr, &[buf(data.clone())]).unwrap();
            let out = result.as_f64_slice();
            for (i, &v) in data.iter().enumerate() {
                let expected = if v > 0.0 { v } else { 0.0 };
                assert_eq!(out[i], expected, "mismatch at index {i} for len={len}");
            }
        }
    }

    // --- Shape mismatch ---

    #[test]
    fn lazy_shape_mismatch() {
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(1)),
        );
        let err = eval_lazy_elementwise(&expr, &[buf(vec![1.0, 2.0]), buf(vec![1.0])]).unwrap_err();
        assert!(matches!(err, crate::error::ParsecError::ShapeError(_)));
    }

    #[test]
    fn lazy_no_buffer_args() {
        let err = eval_lazy_elementwise(&Expr::ArgRef(0), &[scalar(1.0)]).unwrap_err();
        assert!(matches!(err, crate::error::ParsecError::ArgError(_)));
    }
}
