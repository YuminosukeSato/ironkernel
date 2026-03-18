"""ironkernel: A Python parallel compute library backed by a Rust execution engine."""

from ironkernel._facade import chan, kernel, rt
from ironkernel._ironkernel import (
    Buffer,
    Channel,
    Expr,
    KernelSpec,
    MapSpec,
    RecvCase,
    ReduceSpec,
    TaskHandle,
    __version__,
)
from ironkernel._ironkernel import (
    py_select as select,
)

__all__ = [
    "Buffer",
    "Channel",
    "Expr",
    "KernelSpec",
    "MapSpec",
    "RecvCase",
    "ReduceSpec",
    "TaskHandle",
    "__version__",
    "chan",
    "kernel",
    "rt",
    "select",
]
