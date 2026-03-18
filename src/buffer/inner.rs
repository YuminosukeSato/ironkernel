use std::fmt;

/// Supported data types for buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
}

impl DType {
    /// Size in bytes of a single element.
    pub fn byte_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::Bool => 1,
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::Bool => "bool",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_byte_sizes() {
        assert_eq!(DType::F32.byte_size(), 4);
        assert_eq!(DType::F64.byte_size(), 8);
        assert_eq!(DType::I32.byte_size(), 4);
        assert_eq!(DType::I64.byte_size(), 8);
        assert_eq!(DType::Bool.byte_size(), 1);
    }

    #[test]
    fn dtype_names() {
        assert_eq!(DType::F32.name(), "float32");
        assert_eq!(DType::F64.name(), "float64");
        assert_eq!(DType::I32.name(), "int32");
        assert_eq!(DType::I64.name(), "int64");
        assert_eq!(DType::Bool.name(), "bool");
    }

    #[test]
    fn dtype_display() {
        assert_eq!(format!("{}", DType::F64), "float64");
        assert_eq!(format!("{}", DType::Bool), "bool");
    }

    #[test]
    fn dtype_clone_and_eq() {
        let d1 = DType::F64;
        let d2 = d1;
        assert_eq!(d1, d2);
    }

    #[test]
    fn dtype_ne() {
        assert_ne!(DType::F32, DType::F64);
        assert_ne!(DType::I32, DType::I64);
        assert_ne!(DType::Bool, DType::F32);
    }

    #[test]
    fn dtype_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DType::F64);
        set.insert(DType::F64);
        assert_eq!(set.len(), 1);
        set.insert(DType::F32);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn all_variants_covered() {
        let all = [DType::F32, DType::F64, DType::I32, DType::I64, DType::Bool];
        for d in all {
            assert!(d.byte_size() > 0);
            assert!(!d.name().is_empty());
        }
    }
}
