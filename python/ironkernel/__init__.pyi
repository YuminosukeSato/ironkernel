from __future__ import annotations

from ironkernel._facade import KernelFacade, RuntimeFacade
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

kernel: KernelFacade
rt: RuntimeFacade

def chan(capacity: int) -> Channel: ...
