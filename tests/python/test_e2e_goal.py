"""Goal-code smoke tests for the current public API."""

import numpy as np
from ironkernel import RecvCase, kernel, rt, select


def test_goal_story_runs_end_to_end_why_public_regressions_should_break_one_smoke_path() -> None:
    a, x, y = kernel.args("a", "x", "y")
    saxpy = kernel.elementwise(a * x + y)

    x_data = np.arange(1024, dtype=np.float64)
    y_data = np.ones(1024, dtype=np.float64)
    bx = rt.asarray(x_data)
    by = rt.asarray(y_data)

    task = rt.go(kernel.map(saxpy, a=2.0, x=bx, y=by))
    mapped = task.result().numpy()
    assert task.is_done() is True
    np.testing.assert_array_equal(mapped, 2.0 * x_data + y_data)

    arg_x = kernel.arg("x")
    transform = kernel.elementwise(kernel.sqrt(kernel.abs(arg_x)) + kernel.sin(arg_x))
    transformed = rt.go(kernel.map(transform, x=bx)).result().numpy()
    np.testing.assert_allclose(transformed, np.sqrt(np.abs(x_data)) + np.sin(x_data), rtol=1e-10)

    reduce_input = rt.asarray(np.arange(100, dtype=np.float64))
    assert rt.go(kernel.sum(reduce_input)).result().scalar() == 4950.0
    assert rt.go(kernel.mean(reduce_input)).result().scalar() == 49.5

    channel = rt.chan(2)
    channel.send(rt.asarray(np.array([42.0, 43.0], dtype=np.float64)))
    np.testing.assert_array_equal(channel.recv().numpy(), np.array([42.0, 43.0], dtype=np.float64))

    empty_a = rt.chan(1)
    empty_b = rt.chan(1)
    index, value = select(RecvCase(empty_a), RecvCase(empty_b), default=True)
    assert index == -1
    assert value is None

    relu = kernel.elementwise(kernel.where(arg_x > 0, arg_x, 0.0))
    relu_input = rt.asarray(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64))
    relu_out = rt.go(kernel.map(relu, x=relu_input)).result().numpy()
    np.testing.assert_array_equal(relu_out, np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float64))
