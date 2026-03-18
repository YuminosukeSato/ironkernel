# parsec

A Python parallel compute library backed by a Rust execution engine.

Write NumPy-like expressions in Python. Execute them in parallel on Rust, outside the GIL.

## Why parsec?

Python numeric libraries face a fundamental tension: Python is great for expressing computations, but the GIL prevents true parallelism. parsec resolves this with a two-layer architecture:

- Python layer: a DSL that builds expression trees using standard operators (`+`, `*`, `>`, etc.)
- Rust layer: a parallel execution engine that evaluates those trees across all CPU cores via rayon

The key difference from other approaches:

| | parsec | NumPy | Numba | JAX |
|---|---|---|---|---|
| GIL during compute | Released | Held (mostly) | Released (nopython) | Released |
| Parallelism | Automatic (rayon) | Limited BLAS | Manual parallel | XLA compiler |
| Expression building | Python operators | Eager | JIT decorator | Tracing |
| Channel/select | Built-in | No | No | No |
| Dependencies | Rust + maturin | C/Fortran | LLVM | XLA/CUDA |

parsec is not a replacement for NumPy or JAX. It targets a specific niche: CPU-bound elementwise/reduce workloads that benefit from automatic parallelism and Go-style concurrency, with zero GIL contention.

## Quick Start

### Requirements

- Python 3.9+
- Rust 1.70+ (for building from source)
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Install

```bash
git clone https://github.com/your-org/parsec.git
cd parsec
uv sync
uv run maturin develop
```

Verify the installation:

```bash
uv run python -c "import parsec; print(parsec.__version__)"
```

### Your First Kernel (30 seconds)

```python
import numpy as np
from parsec import kernel, rt

# 1. Define expressions using Python operators
a, x, y = kernel.args("a", "x", "y")
saxpy = kernel.elementwise(a * x + y)

# 2. Create buffers from NumPy arrays
bx = rt.asarray(np.arange(1_000_000, dtype=np.float64))
by = rt.asarray(np.ones(1_000_000, dtype=np.float64))

# 3. Launch computation (runs in parallel on Rust, GIL released)
task = rt.go(kernel.map(saxpy, a=2.0, x=bx, y=by))

# 4. Get result as NumPy array
result = task.result().numpy()
print(result[:5])  # [1. 3. 5. 7. 9.]
```

## Core Concepts

### Expression Tree IR

Python operators build an intermediate representation (IR) tree. No computation happens until `rt.go()` is called.

```python
x = kernel.arg("x")

# These build IR nodes, not compute results
expr = kernel.sqrt(kernel.abs(x)) + kernel.sin(x)
spec = kernel.elementwise(expr)
```

Available operations:
- Arithmetic: `+`, `-`, `*`, `/`, `**` (and reverse: `2.0 * x`)
- Unary math: `sqrt`, `abs`, `log`, `exp`, `log2`, `log10`, `sin`, `cos`, `tan`, `floor`, `ceil`, `round`
- Binary math: `pow`, `atan2`, `min`, `max`
- Comparison: `>`, `>=`, `<`, `<=`, `==`, `!=`
- Conditional: `kernel.where_(cond, true_val, false_val)`

### Reductions

```python
buf = rt.asarray(np.arange(100, dtype=np.float64))

total = rt.go(kernel.sum(buf)).result().scalar()      # 4950.0
avg   = rt.go(kernel.mean(buf)).result().scalar()      # 49.5
```

### Channels and Select (Go-style concurrency)

parsec provides bounded channels for passing buffers between concurrent tasks, with a `select` function for multiplexing.

```python
from parsec import RecvCase, rt, select

# Create bounded channels
ch_a = rt.chan(10)
ch_b = rt.chan(10)

# Send data
ch_a.send(rt.asarray(np.array([42.0])))

# Select across multiple channels (non-blocking with default=True)
idx, val = select(RecvCase(ch_a), RecvCase(ch_b), default=True)
# idx=0, val contains [42.0]
```

## Architecture

```
Python (DSL)              Rust (Engine)
─────────────             ─────────────
kernel.arg("x")     →     Expr::ArgRef(0)
x + y               →     Expr::Binary(Add, ...)
kernel.elementwise() →    KernelSpec { kind, expr, args }
rt.go(map_spec)      →    rayon parallel eval (GIL released)
                     →    Buffer (Arc<BufferInner>)
task.result()        ←
.numpy()             ←    zero-copy to ndarray
```

Modules:
- `ir/` — Expression tree, kernel specs, interpreter/compiler
- `buffer/` — Typed buffer storage with Arc-based sharing
- `runtime/` — Rayon thread pool, task handles
- `channel/` — Bounded channels, select multiplexing
- `python/` — PyO3 bindings (the only module that touches pyo3)

## Development

### Build

```bash
uv run maturin develop          # debug build
uv run maturin develop --release  # optimized build
```

### Test

```bash
# Rust tests (110 tests)
cargo test

# Python tests (20 tests)
uv run maturin develop
uv run pytest tests/python/ -v

# All
cargo test && uv run maturin develop && uv run pytest tests/python/ -v
```

### Lint

```bash
# Rust
cargo clippy -- -D warnings
cargo fmt --check

# Python
uv run ruff check python/ tests/
uv run mypy python/parsec/ --strict
```

## License

MIT
