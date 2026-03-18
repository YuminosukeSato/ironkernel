use rayon::ThreadPoolBuilder;

/// Initialize the global Rayon thread pool with the specified number of threads.
/// Must be called at most once. Subsequent calls are no-ops (Rayon's default behavior).
pub fn init_pool(num_threads: usize) {
    let _ = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_pool_does_not_panic() {
        init_pool(2);
    }

    #[test]
    fn init_pool_zero_uses_default() {
        // Rayon treats 0 as "use all CPUs"
        init_pool(0);
    }
}
