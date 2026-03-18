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

use std::sync::Arc;

use crate::error::{ParsecError, ParsecResult};

/// Runtime-owned buffer storage.
#[derive(Debug)]
pub enum Storage {
    /// Owned contiguous memory.
    Owned(Vec<u8>),
}

/// Inner buffer data: dtype, shape, and storage.
#[derive(Debug)]
pub struct BufferInner {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub storage: Storage,
}

impl BufferInner {
    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total bytes of storage.
    pub fn byte_len(&self) -> usize {
        self.len() * self.dtype.byte_size()
    }

    /// Get a slice of the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        match &self.storage {
            Storage::Owned(v) => v,
        }
    }

    /// Interpret storage as a slice of f64 values.
    ///
    /// # Panics
    /// Panics if dtype is not F64.
    pub fn as_f64_slice(&self) -> &[f64] {
        assert_eq!(self.dtype, DType::F64, "as_f64_slice requires F64 dtype");
        let bytes = self.as_bytes();
        // SAFETY: Vec<u8> was created from f64 data with correct alignment and length.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, self.len()) }
    }
}

/// A reference-counted buffer handle.
#[derive(Debug, Clone)]
pub struct Buffer {
    pub(crate) inner: Arc<BufferInner>,
}

impl Buffer {
    /// Create a new Buffer from owned f64 data.
    pub fn from_f64_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        let byte_vec = {
            let mut v = std::mem::ManuallyDrop::new(data);
            let ptr = v.as_mut_ptr() as *mut u8;
            let byte_len = len * 8;
            let byte_cap = v.capacity() * 8;
            // SAFETY: f64 vec reinterpreted as u8 vec with correct ptr/len/cap.
            unsafe { Vec::from_raw_parts(ptr, byte_len, byte_cap) }
        };
        Buffer {
            inner: Arc::new(BufferInner {
                dtype: DType::F64,
                shape: vec![len],
                storage: Storage::Owned(byte_vec),
            }),
        }
    }

    /// Create an empty buffer.
    pub fn empty_f64() -> Self {
        Buffer::from_f64_vec(vec![])
    }

    pub fn dtype(&self) -> DType {
        self.inner.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn as_f64_slice(&self) -> &[f64] {
        self.inner.as_f64_slice()
    }

    /// Validate buffer is non-empty, returning error if empty.
    pub fn require_non_empty(&self, op: &str) -> ParsecResult<()> {
        if self.is_empty() {
            Err(ParsecError::EmptyCollection(format!(
                "cannot {op} empty buffer"
            )))
        } else {
            Ok(())
        }
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

    // --- Buffer tests ---

    #[test]
    fn buffer_from_f64_vec() {
        let buf = Buffer::from_f64_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(buf.dtype(), DType::F64);
        assert_eq!(buf.shape(), &[3]);
        assert_eq!(buf.len(), 3);
        assert!(!buf.is_empty());
        assert_eq!(buf.as_f64_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn buffer_empty() {
        let buf = Buffer::empty_f64();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        let empty: &[f64] = &[];
        assert_eq!(buf.as_f64_slice(), empty);
    }

    #[test]
    fn buffer_single_element() {
        let buf = Buffer::from_f64_vec(vec![42.0]);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.as_f64_slice(), &[42.0]);
    }

    #[test]
    fn buffer_large() {
        let data: Vec<f64> = (0..1_000_000).map(|i| i as f64).collect();
        let buf = Buffer::from_f64_vec(data);
        assert_eq!(buf.len(), 1_000_000);
        assert_eq!(buf.as_f64_slice()[0], 0.0);
        assert_eq!(buf.as_f64_slice()[999_999], 999_999.0);
    }

    #[test]
    fn buffer_special_values() {
        let buf = Buffer::from_f64_vec(vec![
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            f64::EPSILON,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
        ]);
        assert_eq!(buf.len(), 9);
        assert!(buf.as_f64_slice()[0].is_nan());
        assert!(buf.as_f64_slice()[1].is_infinite());
        assert!(buf.as_f64_slice()[2].is_infinite());
    }

    #[test]
    fn buffer_byte_len() {
        let buf = Buffer::from_f64_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(buf.inner.byte_len(), 24); // 3 * 8
    }

    #[test]
    fn buffer_clone_shares_data() {
        let buf1 = Buffer::from_f64_vec(vec![1.0, 2.0]);
        let buf2 = buf1.clone();
        assert_eq!(buf1.as_f64_slice(), buf2.as_f64_slice());
        // Arc refcount increased
        assert_eq!(Arc::strong_count(&buf1.inner), 2);
    }

    #[test]
    fn buffer_require_non_empty_ok() {
        let buf = Buffer::from_f64_vec(vec![1.0]);
        assert!(buf.require_non_empty("sum").is_ok());
    }

    #[test]
    fn buffer_require_non_empty_err() {
        let buf = Buffer::empty_f64();
        let err = buf.require_non_empty("sum").unwrap_err();
        assert_eq!(
            err,
            ParsecError::EmptyCollection("cannot sum empty buffer".into())
        );
    }
}
