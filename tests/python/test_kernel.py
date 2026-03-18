"""Tests for kernel module: args, elementwise, map, math, where, reduce."""

import math

import numpy as np
import pytest
from parsec import kernel, rt


class TestArgs:
    def test_single_arg(self) -> None:
        x = kernel.arg("x")
        assert repr(x) == "Expr(x)"

    def test_multiple_args(self) -> None:
        a, x, y = kernel.args("a", "x", "y")
        assert repr(a) == "Expr(a)"
        assert repr(x) == "Expr(x)"
        assert repr(y) == "Expr(y)"


class TestElementwise:
    def test_saxpy(self) -> None:
        a, x, y = kernel.args("a", "x", "y")
        saxpy = kernel.elementwise(a * x + y)
        bx = rt.asarray(np.arange(100, dtype=np.float64))
        by = rt.asarray(np.ones(100, dtype=np.float64))
        task = rt.go(kernel.map(saxpy, a=2.0, x=bx, y=by))
        out = task.result().numpy()
        assert out[0] == 1.0
        assert out[99] == 199.0

    def test_saxpy_1m(self) -> None:
        a, x, y = kernel.args("a", "x", "y")
        saxpy = kernel.elementwise(a * x + y)
        bx = rt.asarray(np.arange(1_000_000, dtype=np.float64))
        by = rt.asarray(np.ones(1_000_000, dtype=np.float64))
        task = rt.go(kernel.map(saxpy, a=2.0, x=bx, y=by))
        out = task.result().numpy()
        assert out[0] == 1.0
        assert out[-1] == 2.0 * 999999.0 + 1.0

    def test_negation(self) -> None:
        x = kernel.arg("x")
        neg = kernel.elementwise(-x)
        buf = rt.asarray(np.array([1.0, -2.0, 0.0]))
        out = rt.go(kernel.map(neg, x=buf)).result().numpy()
        np.testing.assert_array_equal(out, [-1.0, 2.0, 0.0])

    def test_subtraction(self) -> None:
        x, y = kernel.args("x", "y")
        sub = kernel.elementwise(x - y)
        bx = rt.asarray(np.array([5.0, 3.0]))
        by = rt.asarray(np.array([2.0, 1.0]))
        out = rt.go(kernel.map(sub, x=bx, y=by)).result().numpy()
        np.testing.assert_array_equal(out, [3.0, 2.0])

    def test_division(self) -> None:
        x, y = kernel.args("x", "y")
        div = kernel.elementwise(x / y)
        bx = rt.asarray(np.array([6.0, 9.0]))
        by = rt.asarray(np.array([2.0, 3.0]))
        out = rt.go(kernel.map(div, x=bx, y=by)).result().numpy()
        np.testing.assert_array_equal(out, [3.0, 3.0])

    def test_reverse_ops(self) -> None:
        x = kernel.arg("x")
        expr = 2.0 * x + 1.0
        spec = kernel.elementwise(expr)
        buf = rt.asarray(np.array([3.0, 4.0]))
        out = rt.go(kernel.map(spec, x=buf)).result().numpy()
        np.testing.assert_array_equal(out, [7.0, 9.0])


class TestMathFunctions:
    def test_sqrt_abs_sin(self) -> None:
        x = kernel.arg("x")
        transform = kernel.elementwise(kernel.sqrt(kernel.abs(x)) + kernel.sin(x))
        bx = rt.asarray(np.array([4.0, 9.0]))
        out = rt.go(kernel.map(transform, x=bx)).result().numpy()
        assert abs(out[0] - (math.sqrt(4.0) + math.sin(4.0))) < 1e-10
        assert abs(out[1] - (math.sqrt(9.0) + math.sin(9.0))) < 1e-10

    def test_log_exp(self) -> None:
        x = kernel.arg("x")
        spec = kernel.elementwise(kernel.exp(kernel.log(x)))
        buf = rt.asarray(np.array([1.0, 2.0, math.e]))
        out = rt.go(kernel.map(spec, x=buf)).result().numpy()
        np.testing.assert_allclose(out, [1.0, 2.0, math.e], rtol=1e-10)

    def test_floor_ceil_round(self) -> None:
        x = kernel.arg("x")
        buf = rt.asarray(np.array([1.3, 1.5, 1.7]))

        out = rt.go(kernel.map(kernel.elementwise(kernel.floor(x)), x=buf)).result().numpy()
        np.testing.assert_array_equal(out, [1.0, 1.0, 1.0])

        out = rt.go(kernel.map(kernel.elementwise(kernel.ceil(x)), x=buf)).result().numpy()
        np.testing.assert_array_equal(out, [2.0, 2.0, 2.0])


class TestWhere:
    def test_relu(self) -> None:
        x = kernel.arg("x")
        relu = kernel.elementwise(kernel.where_(x > 0, x, 0))
        data = rt.asarray(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        out = rt.go(kernel.map(relu, x=data)).result().numpy()
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0, 1.0, 2.0])


class TestReduce:
    def test_sum(self) -> None:
        buf = rt.asarray(np.arange(100, dtype=np.float64))
        total = rt.go(kernel.sum(buf)).result().scalar()
        assert total == 4950.0

    def test_mean(self) -> None:
        buf = rt.asarray(np.arange(100, dtype=np.float64))
        avg = rt.go(kernel.mean(buf)).result().scalar()
        assert avg == 49.5

    def test_empty_sum_raises(self) -> None:
        buf = rt.asarray(np.array([], dtype=np.float64))
        with pytest.raises(ValueError):
            rt.go(kernel.sum(buf)).result()
