from __future__ import annotations

import numpy as np
from ironkernel import RecvCase, chan, kernel, rt, select


def test_callable_pipeline_between_channels_why_out_channel_results_must_feed_follow_up_stages() -> None:
    stage_one = chan(3)
    stage_two = chan(3)

    upstream = [rt.go(lambda value=value: float(value), out=stage_one) for value in (1, 2, 3)]
    downstream = [
        rt.go(lambda value: value * 10.0, stage_one.recv().scalar(), out=stage_two) for _ in range(3)
    ]

    assert sorted(stage_two.recv().scalar() for _ in range(3)) == [10.0, 20.0, 30.0]
    assert sorted(task.result() for task in upstream) == [1.0, 2.0, 3.0]
    assert sorted(task.result() for task in downstream) == [10.0, 20.0, 30.0]


def test_mixed_rust_python_pipeline_why_native_specs_and_python_callables_should_compose() -> None:
    middle = chan(1)
    final = chan(1)
    x = kernel.arg("x")
    doubled = kernel.elementwise(x * 2.0)
    source = rt.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float64))

    native_task = rt.go(kernel.map(doubled, x=source), out=middle)
    python_task = rt.go(lambda values: values + 1.0, middle.recv().numpy(), out=final)

    np.testing.assert_array_equal(final.recv().numpy(), np.array([3.0, 5.0, 7.0], dtype=np.float64))
    np.testing.assert_array_equal(native_task.result().numpy(), np.array([2.0, 4.0, 6.0], dtype=np.float64))
    np.testing.assert_array_equal(python_task.result(), np.array([3.0, 5.0, 7.0], dtype=np.float64))


def test_callable_select_receives_channel_output_why_csp_primitives_should_work_with_callable_tasks() -> None:
    left = chan(1)
    right = chan(1)

    task = rt.go(lambda: 9.0, out=right)
    idx, value = select(RecvCase(left), RecvCase(right), default=False)

    assert idx == 1
    assert value.scalar() == 9.0
    assert task.result() == 9.0
