use std::fmt;

/// All errors that can occur in parsec.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsecError {
    /// Type mismatch (expected, got)
    TypeError(String),
    /// Invalid shape or dimension
    ShapeError(String),
    /// Argument error (missing, extra, wrong type)
    ArgError(String),
    /// Operation on empty collection where not allowed
    EmptyCollection(String),
    /// Task was cancelled
    Cancelled,
    /// Channel is closed
    ChannelClosed,
    /// Generic internal error
    Internal(String),
}

impl fmt::Display for ParsecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParsecError::TypeError(msg) => write!(f, "TypeError: {msg}"),
            ParsecError::ShapeError(msg) => write!(f, "ShapeError: {msg}"),
            ParsecError::ArgError(msg) => write!(f, "ArgError: {msg}"),
            ParsecError::EmptyCollection(msg) => write!(f, "EmptyCollection: {msg}"),
            ParsecError::Cancelled => write!(f, "Cancelled"),
            ParsecError::ChannelClosed => write!(f, "ChannelClosed"),
            ParsecError::Internal(msg) => write!(f, "Internal: {msg}"),
        }
    }
}

impl std::error::Error for ParsecError {}

pub type ParsecResult<T> = Result<T, ParsecError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_type_error() {
        let e = ParsecError::TypeError("expected f64, got i32".into());
        assert_eq!(e.to_string(), "TypeError: expected f64, got i32");
    }

    #[test]
    fn error_display_shape_error() {
        let e = ParsecError::ShapeError("shape mismatch: [3] vs [4]".into());
        assert_eq!(e.to_string(), "ShapeError: shape mismatch: [3] vs [4]");
    }

    #[test]
    fn error_display_arg_error() {
        let e = ParsecError::ArgError("missing argument: x".into());
        assert_eq!(e.to_string(), "ArgError: missing argument: x");
    }

    #[test]
    fn error_display_empty_collection() {
        let e = ParsecError::EmptyCollection("cannot sum empty array".into());
        assert_eq!(e.to_string(), "EmptyCollection: cannot sum empty array");
    }

    #[test]
    fn error_display_cancelled() {
        let e = ParsecError::Cancelled;
        assert_eq!(e.to_string(), "Cancelled");
    }

    #[test]
    fn error_display_channel_closed() {
        let e = ParsecError::ChannelClosed;
        assert_eq!(e.to_string(), "ChannelClosed");
    }

    #[test]
    fn error_display_internal() {
        let e = ParsecError::Internal("unexpected state".into());
        assert_eq!(e.to_string(), "Internal: unexpected state");
    }

    #[test]
    fn error_clone_and_eq() {
        let e1 = ParsecError::Cancelled;
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(ParsecError::Internal("test".into()));
        assert!(e.to_string().contains("Internal"));
    }
}
