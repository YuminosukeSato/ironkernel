"""README example regression tests.

Each test reproduces a code block from README.md.
If a test breaks, the README example is lying to readers.
"""

import numpy as np
import numpy.testing as npt
from ironkernel import RecvCase, chan, kernel, rt, select


def test_example1_saxpy_decorator() -> None:
    """README Examples §1: SAXPY with decorator syntax."""

    @kernel.elementwise
    def saxpy(a, x, y):  # type: ignore[no-untyped-def]
        return a * x + y

    bx = rt.asarray(np.arange(1_000_000, dtype=np.float64))
    by = rt.asarray(np.ones(1_000_000, dtype=np.float64))

    result = rt.go(kernel.map(saxpy, a=2.0, x=bx, y=by)).result().numpy()

    expected = 2.0 * np.arange(1_000_000, dtype=np.float64) + np.ones(1_000_000)
    npt.assert_array_equal(result, expected)


def test_example2_math_functions() -> None:
    """README Examples §2: sqrt(abs(x)) + sin(x) with decorator."""

    @kernel.elementwise
    def transform(x):  # type: ignore[no-untyped-def]
        return kernel.sqrt(kernel.abs(x)) + kernel.sin(x)

    buf = rt.asarray(np.arange(1_000_000, dtype=np.float64))
    out = rt.go(kernel.map(transform, x=buf)).result().numpy()

    x_data = np.arange(1_000_000, dtype=np.float64)
    npt.assert_allclose(out, np.sqrt(np.abs(x_data)) + np.sin(x_data), rtol=1e-10)


def test_example3_manual_expression_tree() -> None:
    """README Examples §3: manual operator-based expression tree."""
    x, y = kernel.args("x", "y")
    spec = kernel.elementwise(x + y)

    left = rt.asarray(np.array([1.0, 2.0, 3.0]))
    right = rt.asarray(np.array([10.0, 20.0, 30.0]))

    result = rt.go(kernel.map(spec, x=left, y=right)).result().numpy()
    npt.assert_array_equal(result, np.array([11.0, 22.0, 33.0]))


def test_example4_reductions() -> None:
    """README Examples §4: sum, mean, min_reduce, max_reduce."""
    buf = rt.asarray(np.arange(100, dtype=np.float64))

    assert rt.go(kernel.sum(buf)).result().scalar() == 4950.0
    assert rt.go(kernel.mean(buf)).result().scalar() == 49.5
    assert rt.go(kernel.min_reduce(buf)).result().scalar() == 0.0
    assert rt.go(kernel.max_reduce(buf)).result().scalar() == 99.0


def test_example5_relu_where() -> None:
    """README Examples §5: ReLU via kernel.where."""

    @kernel.elementwise
    def relu(x):  # type: ignore[no-untyped-def]
        return kernel.where(x > 0, x, 0)

    buf = rt.asarray(np.arange(100, dtype=np.float64) - 50)
    out = rt.go(kernel.map(relu, x=buf)).result().numpy()

    x_data = np.arange(100, dtype=np.float64) - 50
    npt.assert_array_equal(out, np.where(x_data > 0, x_data, 0.0))


def test_example6_channel_select() -> None:
    """README Examples §6: channel and select."""
    ch_a = chan(10)
    ch_b = chan(10)

    ch_a.send(rt.asarray(np.array([42.0])))

    idx, val = select(RecvCase(ch_a), RecvCase(ch_b), default=True)
    assert idx == 0
    assert val is not None
    assert val.scalar() == 42.0


def test_example7_channel_handoff() -> None:
    """README Examples §7: out=channel handoff."""

    @kernel.elementwise
    def double(x):  # type: ignore[no-untyped-def]
        return x * 2.0

    buf = rt.asarray(np.arange(1_000_000, dtype=np.float64))

    c = chan(10)
    task = rt.go(kernel.map(double, x=buf), out=c)
    result = c.recv()
    assert task.is_done()
    npt.assert_array_equal(result.numpy()[:5], np.array([0.0, 2.0, 4.0, 6.0, 8.0]))
