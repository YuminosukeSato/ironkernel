from __future__ import annotations

from typing import Any

from ironkernel._decorator import lower_function_to_spec
from ironkernel._ironkernel import (
    Buffer,
    Channel,
    Expr,
    KernelSpec,
    MapSpec,
    ReduceSpec,
    TaskHandle,
)
from ironkernel._ironkernel import (
    _KernelModule as _BackendKernelModule,
)
from ironkernel._ironkernel import (
    _RuntimeModule as _BackendRuntimeModule,
)


class KernelFacade:
    __slots__ = ("_backend",)

    def __init__(self) -> None:
        self._backend = _BackendKernelModule()

    def arg(self, name: str) -> Expr:
        return self._backend.arg(name)

    def args(self, *names: str) -> list[Expr]:
        if not names:
            raise ValueError("at least one arg name is required")
        return list(self._backend.args(*names))

    def elementwise(self, expr_or_fn: Any) -> KernelSpec:
        if callable(expr_or_fn) and not isinstance(expr_or_fn, Expr):
            return lower_function_to_spec(expr_or_fn, self)
        return self._backend.elementwise(expr_or_fn)

    def map(self, spec: KernelSpec, **kwargs: Any) -> MapSpec:
        return self._backend.map(spec, **kwargs)

    def where(self, cond: Expr, true_val: Expr | int | float, false_val: Expr | int | float) -> Expr:
        return self._backend.where_(cond, true_val, false_val)

    def sqrt(self, expr: Expr) -> Expr:
        return self._backend.sqrt(expr)

    def abs(self, expr: Expr) -> Expr:
        return self._backend.abs(expr)

    def log(self, expr: Expr) -> Expr:
        return self._backend.log(expr)

    def exp(self, expr: Expr) -> Expr:
        return self._backend.exp(expr)

    def log2(self, expr: Expr) -> Expr:
        return self._backend.log2(expr)

    def log10(self, expr: Expr) -> Expr:
        return self._backend.log10(expr)

    def floor(self, expr: Expr) -> Expr:
        return self._backend.floor(expr)

    def ceil(self, expr: Expr) -> Expr:
        return self._backend.ceil(expr)

    def sin(self, expr: Expr) -> Expr:
        return self._backend.sin(expr)

    def cos(self, expr: Expr) -> Expr:
        return self._backend.cos(expr)

    def tan(self, expr: Expr) -> Expr:
        return self._backend.tan(expr)

    def pow(self, base: Expr, exp: Expr) -> Expr:
        return self._backend.pow(base, exp)

    def atan2(self, y: Expr, x: Expr) -> Expr:
        return self._backend.atan2(y, x)

    def min(self, left: Expr, right: Expr) -> Expr:
        return self._backend.min(left, right)

    def max(self, left: Expr, right: Expr) -> Expr:
        return self._backend.max(left, right)

    def sum(self, buf: Buffer) -> ReduceSpec:
        return self._backend.sum(buf)

    def mean(self, buf: Buffer) -> ReduceSpec:
        return self._backend.mean(buf)

    def min_reduce(self, buf: Buffer) -> ReduceSpec:
        return self._backend.min_reduce(buf)

    def max_reduce(self, buf: Buffer) -> ReduceSpec:
        return self._backend.max_reduce(buf)


class RuntimeFacade:
    __slots__ = ("_backend",)

    def __init__(self) -> None:
        self._backend = _BackendRuntimeModule()

    def asarray(self, array: Any) -> Buffer:
        return self._backend.asarray(array)

    def go(
        self,
        spec: MapSpec | ReduceSpec | Any,
        *args: Any,
        out: Channel | None = None,
        **kwargs: Any,
    ) -> TaskHandle:
        return self._backend.go(spec, *args, out=out, **kwargs)

    def chan(self, capacity: int) -> Channel:
        return self._backend.chan(capacity)


kernel = KernelFacade()
rt = RuntimeFacade()


def chan(capacity: int) -> Channel:
    return rt.chan(capacity)
