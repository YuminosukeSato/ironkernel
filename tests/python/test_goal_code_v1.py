"""Goal code v1 reproduction tests (design.md Section 3.1)."""

import numpy as np
import numpy.testing as npt
from ironkernel import RecvCase, chan, kernel, rt, select


def test_saxpy_goal_code() -> None:
    """design.md L84-91: SAXPY with 1M elements."""
    a, x, y = kernel.args("a", "x", "y")
    saxpy = kernel.elementwise(a * x + y)

    bx = rt.asarray(np.arange(1_000_000, dtype=np.float64))
    by = rt.asarray(np.ones(1_000_000, dtype=np.float64))

    task = rt.go(kernel.map(saxpy, a=2.0, x=bx, y=by))
    out = task.result().numpy()

    expected = 2.0 * np.arange(1_000_000, dtype=np.float64) + np.ones(1_000_000)
    npt.assert_array_equal(out, expected)


def test_math_goal_code() -> None:
    """design.md L93-95: sqrt(abs(x)) + sin(x)."""
    x = kernel.arg("x")
    transform = kernel.elementwise(kernel.sqrt(kernel.abs(x)) + kernel.sin(x))
    bx = rt.asarray(np.arange(1_000_000, dtype=np.float64))
    out = rt.go(kernel.map(transform, x=bx)).result().numpy()

    x_data = np.arange(1_000_000, dtype=np.float64)
    npt.assert_allclose(out, np.sqrt(np.abs(x_data)) + np.sin(x_data), rtol=1e-10)


def test_reduce_goal_code() -> None:
    """design.md L97-99: sum and mean."""
    buf = rt.asarray(np.arange(100, dtype=np.float64))
    total = rt.go(kernel.sum(buf)).result().scalar()
    avg = rt.go(kernel.mean(buf)).result().scalar()

    assert total == 4950.0
    assert avg == 49.5


def test_channel_handoff_goal_code() -> None:
    """design.md L101-105: rt.go with out=channel."""
    a, x, y = kernel.args("a", "x", "y")
    saxpy = kernel.elementwise(a * x + y)

    bx = rt.asarray(np.arange(1_000_000, dtype=np.float64))
    by = rt.asarray(np.ones(1_000_000, dtype=np.float64))

    c = chan(10)
    task = rt.go(kernel.map(saxpy, a=2.0, x=bx, y=by), out=c)
    result_buf = c.recv()
    task_result = task.result()
    out = result_buf.numpy()
    npt.assert_array_equal(task_result.numpy(), out)
    assert task.is_done()

    expected = 2.0 * np.arange(1_000_000, dtype=np.float64) + np.ones(1_000_000)
    npt.assert_array_equal(out, expected)


def test_select_goal_code() -> None:
    """design.md L107-111: select with data in ch_a."""
    ch_a = chan(10)
    ch_b = chan(10)
    ch_a.send(rt.asarray(np.array([42.0])))
    idx, _val = select(RecvCase(ch_a), RecvCase(ch_b), default=True)
    assert idx == 0


def test_relu_goal_code() -> None:
    """design.md L113-115: relu via kernel.where."""
    x = kernel.arg("x")
    relu = kernel.elementwise(kernel.where(x > 0, x, 0))
    buf = rt.asarray(np.arange(100, dtype=np.float64) - 50)
    out = rt.go(kernel.map(relu, x=buf)).result().numpy()

    x_data = np.arange(100, dtype=np.float64) - 50
    expected = np.where(x_data > 0, x_data, 0.0)
    npt.assert_array_equal(out, expected)
