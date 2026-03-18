# Rust Core Layer Rules

ir/, buffer/, runtime/, channel/ are pure Rust modules.

## Dependency Constraints
- NEVER import pyo3 or numpy crates
- Allowed external deps: rayon (parallelism), crossbeam-channel (channels)

## Visibility
- pub(crate) by default. pub re-export only in lib.rs
- Use pub(crate) types directly across modules

## Error Handling
- All errors consolidated in ParsecError enum (src/error.rs)
- ParsecResult<T> = Result<T, ParsecError>
- No unwrap()/expect() in production code (test code exempt)

## Testing
- Each module has #[cfg(test)] mod tests
- proptest for boundary and property-based tests recommended
- Test around rayon parallelism threshold (4K+ elements)

## unsafe
- SAFETY comment required (document why the invariant holds)
- Consider safe alternatives first
