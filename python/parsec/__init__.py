"""parsec: A Python parallel compute library backed by a Rust execution engine."""

from parsec._parsec import (
    Buffer,
    Channel,
    Expr,
    KernelSpec,
    MapSpec,
    RecvCase,
    ReduceSpec,
    TaskHandle,
    __version__,
    _KernelModule,
    _RuntimeModule,
)
from parsec._parsec import (
    py_select as select,
)

# Singleton module instances
kernel = _KernelModule()
rt = _RuntimeModule()

__all__ = [
    "__version__",
    "Buffer",
    "Channel",
    "Expr",
    "KernelSpec",
    "MapSpec",
    "RecvCase",
    "ReduceSpec",
    "TaskHandle",
    "kernel",
    "rt",
    "select",
]
