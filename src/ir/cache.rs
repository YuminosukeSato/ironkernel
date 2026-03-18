use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

use crate::ir::compiler::{compile_program, CompiledProgram};
use crate::ir::expr::{BinaryOp, CmpOp, Expr, UnaryOp};

/// Tag bytes for each Expr variant, used to serialize the tree structure.
mod tag {
    pub const CONST: u8 = 0;
    pub const ARG_REF: u8 = 1;

    // UnaryOp: 10..22
    pub const NEG: u8 = 10;
    pub const ABS: u8 = 11;
    pub const SQRT: u8 = 12;
    pub const LOG: u8 = 13;
    pub const EXP: u8 = 14;
    pub const LOG2: u8 = 15;
    pub const LOG10: u8 = 16;
    pub const FLOOR: u8 = 17;
    pub const CEIL: u8 = 18;
    pub const ROUND: u8 = 19;
    pub const SIN: u8 = 20;
    pub const COS: u8 = 21;
    pub const TAN: u8 = 22;

    // BinaryOp: 30..37
    pub const ADD: u8 = 30;
    pub const SUB: u8 = 31;
    pub const MUL: u8 = 32;
    pub const DIV: u8 = 33;
    pub const POW: u8 = 34;
    pub const ATAN2: u8 = 35;
    pub const MIN: u8 = 36;
    pub const MAX: u8 = 37;

    // CmpOp: 50..55
    pub const GT: u8 = 50;
    pub const GE: u8 = 51;
    pub const LT: u8 = 52;
    pub const LE: u8 = 53;
    pub const EQ: u8 = 54;
    pub const NE: u8 = 55;

    pub const SELECT: u8 = 70;
}

fn unary_op_tag(op: &UnaryOp) -> u8 {
    match op {
        UnaryOp::Neg => tag::NEG,
        UnaryOp::Abs => tag::ABS,
        UnaryOp::Sqrt => tag::SQRT,
        UnaryOp::Log => tag::LOG,
        UnaryOp::Exp => tag::EXP,
        UnaryOp::Log2 => tag::LOG2,
        UnaryOp::Log10 => tag::LOG10,
        UnaryOp::Floor => tag::FLOOR,
        UnaryOp::Ceil => tag::CEIL,
        UnaryOp::Round => tag::ROUND,
        UnaryOp::Sin => tag::SIN,
        UnaryOp::Cos => tag::COS,
        UnaryOp::Tan => tag::TAN,
    }
}

fn binary_op_tag(op: &BinaryOp) -> u8 {
    match op {
        BinaryOp::Add => tag::ADD,
        BinaryOp::Sub => tag::SUB,
        BinaryOp::Mul => tag::MUL,
        BinaryOp::Div => tag::DIV,
        BinaryOp::Pow => tag::POW,
        BinaryOp::Atan2 => tag::ATAN2,
        BinaryOp::Min => tag::MIN,
        BinaryOp::Max => tag::MAX,
    }
}

fn cmp_op_tag(op: &CmpOp) -> u8 {
    match op {
        CmpOp::Gt => tag::GT,
        CmpOp::Ge => tag::GE,
        CmpOp::Lt => tag::LT,
        CmpOp::Le => tag::LE,
        CmpOp::Eq => tag::EQ,
        CmpOp::Ne => tag::NE,
    }
}

/// Structural key for an expression tree.
///
/// Built by pre-order traversal: each node appends its tag to `op_stream`,
/// `Const` values store `f64::to_bits()` in `const_bits`, and `ArgRef`
/// indices go into `arg_indices`.
///
/// Using `to_bits()` means `+0.0` and `-0.0` produce different keys,
/// while same-bit NaN values produce the same key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ExprKey {
    op_stream: Vec<u8>,
    const_bits: Vec<u64>,
    arg_indices: Vec<usize>,
}

impl ExprKey {
    fn build(expr: &Expr, key: &mut ExprKey) {
        match expr {
            Expr::Const(v) => {
                key.op_stream.push(tag::CONST);
                key.const_bits.push(v.to_bits());
            }
            Expr::ArgRef(i) => {
                key.op_stream.push(tag::ARG_REF);
                key.arg_indices.push(*i);
            }
            Expr::Unary(op, inner) => {
                key.op_stream.push(unary_op_tag(op));
                Self::build(inner, key);
            }
            Expr::Binary(op, lhs, rhs) => {
                key.op_stream.push(binary_op_tag(op));
                Self::build(lhs, key);
                Self::build(rhs, key);
            }
            Expr::Compare(op, lhs, rhs) => {
                key.op_stream.push(cmp_op_tag(op));
                Self::build(lhs, key);
                Self::build(rhs, key);
            }
            Expr::Select(cond, t, f) => {
                key.op_stream.push(tag::SELECT);
                Self::build(cond, key);
                Self::build(t, key);
                Self::build(f, key);
            }
        }
    }
}

impl From<&Expr> for ExprKey {
    fn from(expr: &Expr) -> Self {
        let mut key = ExprKey {
            op_stream: Vec::new(),
            const_bits: Vec::new(),
            arg_indices: Vec::new(),
        };
        ExprKey::build(expr, &mut key);
        key
    }
}

/// A hash bucket: list of (key, compiled program) pairs sharing the same hash.
type Bucket = Vec<(ExprKey, Arc<CompiledProgram>)>;

#[cfg(test)]
type AfterCompileHook = Arc<dyn Fn() + Send + Sync>;

/// Thread-safe compile cache using hash-bucket collision resolution.
///
/// Keyed by `ExprKey` (structural equality). Uses `RwLock` for
/// concurrent read access with exclusive write on miss.
pub(crate) struct CompileCache {
    buckets: RwLock<HashMap<u64, Bucket>>,
    #[cfg(test)]
    after_compile_hook: Option<AfterCompileHook>,
}

impl CompileCache {
    pub(crate) fn new() -> Self {
        CompileCache {
            buckets: RwLock::new(HashMap::new()),
            #[cfg(test)]
            after_compile_hook: None,
        }
    }

    #[cfg(test)]
    fn with_after_compile_hook(hook: AfterCompileHook) -> Self {
        CompileCache {
            buckets: RwLock::new(HashMap::new()),
            after_compile_hook: Some(hook),
        }
    }

    /// Look up or compile an expression. Returns a shared handle to the
    /// compiled program. Thread-safe: concurrent callers with the same
    /// expression will share a single compilation result.
    pub(crate) fn get_or_compile(&self, expr: &Expr) -> Arc<CompiledProgram> {
        let key = ExprKey::from(expr);
        let hash = Self::hash_key(&key);

        // Fast path: read lock
        {
            let buckets = self.buckets.read().expect("cache read lock poisoned");
            let cached = buckets
                .get(&hash)
                .and_then(|bucket| bucket.iter().find(|(k, _)| *k == key));
            if let Some((_, program)) = cached {
                return program.clone();
            }
        }

        // Slow path: compile then insert
        let program = Arc::new(compile_program(expr));
        #[cfg(test)]
        if let Some(hook) = &self.after_compile_hook {
            hook();
        }

        {
            let mut buckets = self.buckets.write().expect("cache write lock poisoned");
            let bucket = buckets.entry(hash).or_default();
            // Double-check: another thread may have inserted while we compiled
            let existing = bucket.iter().find(|(k, _)| *k == key);
            if let Some((_, existing)) = existing {
                return existing.clone();
            }
            bucket.push((key, program.clone()));
        }

        program
    }

    fn hash_key(key: &ExprKey) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Barrier;

    // --- ExprKey construction ---

    #[test]
    fn expr_key_const() {
        let key = ExprKey::from(&Expr::Const(1.0));
        assert_eq!(key.op_stream, vec![tag::CONST]);
        assert_eq!(key.const_bits, vec![1.0_f64.to_bits()]);
        assert!(key.arg_indices.is_empty());
    }

    #[test]
    fn expr_key_arg_ref() {
        let key = ExprKey::from(&Expr::ArgRef(3));
        assert_eq!(key.op_stream, vec![tag::ARG_REF]);
        assert!(key.const_bits.is_empty());
        assert_eq!(key.arg_indices, vec![3]);
    }

    // RT3.2: +0.0 vs -0.0
    #[test]
    fn positive_zero_negative_zero_different_keys() {
        let key_pos = ExprKey::from(&Expr::Const(0.0));
        let key_neg = ExprKey::from(&Expr::Const(-0.0));
        assert_ne!(key_pos, key_neg);
    }

    // RT3.3: NaN same bits same key
    #[test]
    fn nan_same_bits_same_key() {
        let key1 = ExprKey::from(&Expr::Const(f64::NAN));
        let key2 = ExprKey::from(&Expr::Const(f64::NAN));
        assert_eq!(key1, key2);
    }

    // RT3.6: different arg indices
    #[test]
    fn cache_different_arg_count() {
        let expr1 = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(1)),
        );
        let expr2 = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::ArgRef(2)),
        );
        assert_ne!(ExprKey::from(&expr1), ExprKey::from(&expr2));
    }

    // RT3.9: all unary ops distinct
    #[test]
    fn expr_key_all_unary_ops_distinct() {
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
        let keys: Vec<ExprKey> = ops
            .iter()
            .map(|op| ExprKey::from(&Expr::Unary(*op, Box::new(Expr::ArgRef(0)))))
            .collect();
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j], "ops {ops:?} collide at {i} and {j}");
            }
        }
    }

    // RT3.10: all binary ops distinct
    #[test]
    fn expr_key_all_binary_ops_distinct() {
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
        let keys: Vec<ExprKey> = ops
            .iter()
            .map(|op| {
                ExprKey::from(&Expr::Binary(
                    *op,
                    Box::new(Expr::ArgRef(0)),
                    Box::new(Expr::ArgRef(1)),
                ))
            })
            .collect();
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j], "ops {ops:?} collide at {i} and {j}");
            }
        }
    }

    // RT3.11: all cmp ops distinct
    #[test]
    fn expr_key_all_cmp_ops_distinct() {
        let ops = [
            CmpOp::Gt,
            CmpOp::Ge,
            CmpOp::Lt,
            CmpOp::Le,
            CmpOp::Eq,
            CmpOp::Ne,
        ];
        let keys: Vec<ExprKey> = ops
            .iter()
            .map(|op| {
                ExprKey::from(&Expr::Compare(
                    *op,
                    Box::new(Expr::ArgRef(0)),
                    Box::new(Expr::ArgRef(1)),
                ))
            })
            .collect();
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j], "ops {ops:?} collide at {i} and {j}");
            }
        }
    }

    // --- CompileCache tests ---

    // RT3.1: same expr hits cache
    #[test]
    fn same_expr_hits_cache() {
        let cache = CompileCache::new();
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
        );
        let p1 = cache.get_or_compile(&expr);
        let p2 = cache.get_or_compile(&expr);
        assert!(Arc::ptr_eq(&p1, &p2));
    }

    // RT3.4: different structure misses
    #[test]
    fn same_hash_different_key_misses() {
        let cache = CompileCache::new();
        let expr1 = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
        );
        let expr2 = Expr::Binary(
            BinaryOp::Sub,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
        );
        let p1 = cache.get_or_compile(&expr1);
        let p2 = cache.get_or_compile(&expr2);
        assert!(!Arc::ptr_eq(&p1, &p2));
    }

    // RT3.5: concurrent access
    #[test]
    fn cache_concurrent_access() {
        let cache = Arc::new(CompileCache::new());
        let expr = Expr::Binary(
            BinaryOp::Mul,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(2.0)),
        );

        let mut handles = vec![];
        for _ in 0..4 {
            let cache = Arc::clone(&cache);
            let expr = expr.clone();
            handles.push(std::thread::spawn(move || cache.get_or_compile(&expr)));
        }

        let results: Vec<Arc<CompiledProgram>> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();
        for r in &results[1..] {
            assert!(Arc::ptr_eq(&results[0], r));
        }
    }

    #[test]
    fn cache_double_check_reuses_inserted_program_why_concurrent_misses_must_coalesce_to_one_entry()
    {
        let expr = Expr::Binary(
            BinaryOp::Mul,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(2.0)),
        );
        let first_hook = Arc::new(AtomicBool::new(true));
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let cache = Arc::new(CompileCache::with_after_compile_hook({
            let first_hook = Arc::clone(&first_hook);
            let entered = Arc::clone(&entered);
            let release = Arc::clone(&release);
            Arc::new(move || {
                if first_hook.swap(false, Ordering::SeqCst) {
                    entered.wait();
                    release.wait();
                }
            })
        }));

        let blocked_cache = Arc::clone(&cache);
        let blocked_expr = expr.clone();
        let blocked = std::thread::spawn(move || blocked_cache.get_or_compile(&blocked_expr));

        entered.wait();

        let inserted_cache = Arc::clone(&cache);
        let inserted_expr = expr.clone();
        let inserted = std::thread::spawn(move || inserted_cache.get_or_compile(&inserted_expr))
            .join()
            .unwrap();

        release.wait();

        let resumed = blocked.join().unwrap();
        assert!(Arc::ptr_eq(&resumed, &inserted));
    }

    // Cache with Select (LazyTree path)
    #[test]
    fn cache_select_expr() {
        let cache = CompileCache::new();
        let expr = Expr::Select(
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Const(0.0)),
        );
        let p1 = cache.get_or_compile(&expr);
        let p2 = cache.get_or_compile(&expr);
        assert!(Arc::ptr_eq(&p1, &p2));
        assert!(matches!(p1.as_ref(), CompiledProgram::LazyTree(_)));
    }

    // Cache with flat expr (Flat path)
    #[test]
    fn cache_flat_expr() {
        let cache = CompileCache::new();
        let expr = Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::ArgRef(0)),
            Box::new(Expr::Const(1.0)),
        );
        let p1 = cache.get_or_compile(&expr);
        assert!(matches!(p1.as_ref(), CompiledProgram::Flat(_)));
    }

    // --- proptest ---

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

    // RT3.7: key equality is reflexive
    proptest! {
        #[test]
        fn expr_key_equality_is_reflexive(expr in arb_expr(4)) {
            let key1 = ExprKey::from(&expr);
            let key2 = ExprKey::from(&expr);
            prop_assert_eq!(key1, key2);
        }
    }

    // RT3.8: const bits roundtrip
    proptest! {
        #[test]
        fn const_bits_roundtrip(v in any::<f64>()) {
            let bits = v.to_bits();
            let restored = f64::from_bits(bits);
            prop_assert_eq!(v.to_bits(), restored.to_bits());
        }
    }

    // Cache returns same Arc for structurally identical expressions
    proptest! {
        #[test]
        fn cache_hit_for_identical_exprs(expr in arb_expr(3)) {
            let cache = CompileCache::new();
            let p1 = cache.get_or_compile(&expr);
            let p2 = cache.get_or_compile(&expr);
            prop_assert!(Arc::ptr_eq(&p1, &p2));
        }
    }
}
