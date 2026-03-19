from __future__ import annotations

import time

import numpy as np
import pytest
from ironkernel import chan, rt


def test_go_callable_lambda_with_args_and_kwargs_returns_python_value_why_runtime_should_accept_plain_python_submit(
) -> None:
    task = rt.go(lambda x, y=0, scale=1: (x + y) * scale, 3, y=4, scale=2)

    assert task.result() == 14


def test_go_callable_builtin_returns_python_value_why_threadpool_like_usage_should_work_without_specs() -> None:
    task = rt.go(sum, [1, 2, 3, 4])

    assert task.result() == 10


def test_go_callable_numpy_result_stays_python_object_why_callable_result_should_not_be_forced_into_buffer() -> None:
    task = rt.go(lambda: np.array([1.0, 2.0, 3.0], dtype=np.float64))

    np.testing.assert_array_equal(task.result(), np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_go_callable_with_channel_scalar_delivers_buffer_and_keeps_python_result_why_dual_contract_must_hold() -> None:
    output = chan(1)

    task = rt.go(lambda value: value * 2.0, 3.5, out=output)

    assert output.recv().scalar() == 7.0
    assert task.result() == 7.0


def test_go_callable_with_channel_numpy_delivers_buffer_and_keeps_numpy_result_why_array_outputs_must_bridge(
) -> None:
    output = chan(1)

    task = rt.go(
        lambda: np.array([2.0, 4.0, 8.0], dtype=np.float64),
        out=output,
    )

    np.testing.assert_array_equal(output.recv().numpy(), np.array([2.0, 4.0, 8.0], dtype=np.float64))
    np.testing.assert_array_equal(task.result(), np.array([2.0, 4.0, 8.0], dtype=np.float64))


def test_go_callable_with_channel_non_buffer_result_raises_why_channels_must_remain_buffer_only() -> None:
    output = chan(1)
    task = rt.go(lambda: {"bad": 1}, out=output)

    with pytest.raises(RuntimeError, match="buffer-convertible"):
        task.result()


def test_go_callable_exception_surfaces_why_background_failures_must_not_be_swallowed() -> None:
    task = rt.go(lambda: (_ for _ in ()).throw(RuntimeError("callable boom")))

    with pytest.raises(RuntimeError, match="callable boom"):
        task.result()


def test_go_callable_cancel_raises_runtime_error_why_cooperative_cancellation_should_be_visible() -> None:
    task = rt.go(lambda: (time.sleep(0.05), 1)[1])

    assert task.cancel() is True
    with pytest.raises(RuntimeError, match="cancelled"):
        task.result()


def test_go_callable_concurrent_results_complete_why_multiple_python_submissions_should_not_share_state() -> None:
    tasks = [rt.go(lambda value=value: value * value) for value in range(10)]

    assert sorted(task.result() for task in tasks) == [value * value for value in range(10)]
