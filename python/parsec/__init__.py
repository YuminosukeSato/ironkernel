"""parsec: A Python parallel compute library backed by a Rust execution engine."""

from parsec._parsec import (
    Buffer,
    Expr,
    KernelSpec,
    MapSpec,
    ReduceSpec,
    TaskHandle,
    __version__,
    _KernelModule,
    _RuntimeModule,
)

# Singleton module instances
kernel = _KernelModule()
rt = _RuntimeModule()

__all__ = [
    "__version__",
    "Buffer",
    "Expr",
    "KernelSpec",
    "MapSpec",
    "ReduceSpec",
    "TaskHandle",
    "kernel",
    "rt",
]
