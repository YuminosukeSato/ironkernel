use crate::buffer::inner::DType;
use crate::ir::expr::Expr;

/// Kind of kernel computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    Elementwise,
    Reduce,
}

/// Specification for a single kernel argument.
#[derive(Debug, Clone, PartialEq)]
pub struct ArgSpec {
    pub name: String,
    pub dtype: DType,
    pub is_scalar: bool,
}

/// Specification for the output tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSpec {
    pub dtype: DType,
}

/// Reduce operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Min,
    Max,
    Count,
    Variance,
    Std,
}

/// Complete kernel specification: what to compute and on what arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelSpec {
    pub kind: KernelKind,
    pub args: Vec<ArgSpec>,
    pub output: TensorSpec,
    pub expr: Expr,
}

impl KernelSpec {
    /// Create an elementwise kernel from an expression and argument specs.
    pub fn elementwise(args: Vec<ArgSpec>, expr: Expr) -> Self {
        KernelSpec {
            kind: KernelKind::Elementwise,
            args,
            output: TensorSpec { dtype: DType::F64 },
            expr,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::expr::BinaryOp;

    #[test]
    fn kernel_kind_eq() {
        assert_eq!(KernelKind::Elementwise, KernelKind::Elementwise);
        assert_ne!(KernelKind::Elementwise, KernelKind::Reduce);
    }

    #[test]
    fn arg_spec_scalar() {
        let arg = ArgSpec {
            name: "a".into(),
            dtype: DType::F64,
            is_scalar: true,
        };
        assert!(arg.is_scalar);
        assert_eq!(arg.dtype, DType::F64);
    }

    #[test]
    fn arg_spec_array() {
        let arg = ArgSpec {
            name: "x".into(),
            dtype: DType::F64,
            is_scalar: false,
        };
        assert!(!arg.is_scalar);
    }

    #[test]
    fn tensor_spec_dtype() {
        let spec = TensorSpec { dtype: DType::F32 };
        assert_eq!(spec.dtype, DType::F32);
    }

    #[test]
    fn reduce_op_all_variants() {
        let ops = [
            ReduceOp::Sum,
            ReduceOp::Mean,
            ReduceOp::Min,
            ReduceOp::Max,
            ReduceOp::Count,
            ReduceOp::Variance,
            ReduceOp::Std,
        ];
        for op in ops {
            let op2 = op;
            assert_eq!(op, op2);
        }
    }

    #[test]
    fn kernel_spec_elementwise_saxpy() {
        let args = vec![
            ArgSpec {
                name: "a".into(),
                dtype: DType::F64,
                is_scalar: true,
            },
            ArgSpec {
                name: "x".into(),
                dtype: DType::F64,
                is_scalar: false,
            },
            ArgSpec {
                name: "y".into(),
                dtype: DType::F64,
                is_scalar: false,
            },
        ];
        // a * x + y
        let a_times_x = Expr::Binary(
            BinaryOp::Mul,
            Box::new(Expr::arg_ref(0)),
            Box::new(Expr::arg_ref(1)),
        );
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(a_times_x),
            Box::new(Expr::arg_ref(2)),
        );
        let spec = KernelSpec::elementwise(args.clone(), expr);
        assert_eq!(spec.kind, KernelKind::Elementwise);
        assert_eq!(spec.args.len(), 3);
        assert_eq!(spec.output.dtype, DType::F64);
    }

    #[test]
    fn kernel_spec_clone() {
        let spec = KernelSpec::elementwise(vec![], Expr::constant(1.0));
        let spec2 = spec.clone();
        assert_eq!(spec, spec2);
    }

    #[test]
    fn kernel_spec_no_args() {
        let spec = KernelSpec::elementwise(vec![], Expr::constant(42.0));
        assert!(spec.args.is_empty());
        assert_eq!(spec.kind, KernelKind::Elementwise);
    }
}
