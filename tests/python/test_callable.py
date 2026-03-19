from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
from ironkernel import chan, kernel, rt


def test_go_callable_with_args_kwargs_why_runtime_accepts_plain_python_submit() -> None:
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


def test_go_callable_with_channel_numpy_delivers_buffer_and_keeps_numpy_result_why_array_outputs_must_bridge() -> None:
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


def test_go_custom_function_why_def_style_callables_must_work_not_just_lambdas() -> None:
    def add(a: int, b: int) -> int:
        return a + b

    task = rt.go(add, 10, 20)

    assert task.result() == 30


def test_go_callable_returns_immediately_why_submit_must_not_block_caller() -> None:
    task = rt.go(lambda: (time.sleep(0.1), "slow")[1])

    assert task.is_done() is False


def test_go_callable_is_done_transitions_why_state_must_reflect_background_completion() -> None:
    task = rt.go(lambda: 42)

    task.result()

    assert task.is_done() is True


def test_go_callable_returns_dict_why_arbitrary_python_objects_must_roundtrip() -> None:
    task = rt.go(lambda: {"key": "value", "count": 3})

    result = task.result()
    assert result == {"key": "value", "count": 3}


def test_go_callable_returns_string_why_non_numeric_objects_must_pass_through() -> None:
    task = rt.go(lambda: "hello world")

    assert task.result() == "hello world"


def test_go_callable_returns_list_why_collection_objects_must_not_be_coerced() -> None:
    task = rt.go(lambda: [1, "two", 3.0])

    assert task.result() == [1, "two", 3.0]


def test_go_callable_returns_none_why_void_callables_must_not_raise() -> None:
    task = rt.go(lambda: None)

    assert task.result() is None


def test_go_callable_with_channel_list_of_floats_delivers_buffer_why_float_list_is_buffer_convertible() -> None:
    output = chan(1)

    task = rt.go(lambda: [1.0, 2.0, 3.0], out=output)

    np.testing.assert_array_equal(output.recv().numpy(), np.array([1.0, 2.0, 3.0], dtype=np.float64))
    assert task.result() == [1.0, 2.0, 3.0]


def test_go_callable_with_channel_string_raises_why_string_is_not_buffer_convertible() -> None:
    output = chan(1)
    task = rt.go(lambda: "not a buffer", out=output)

    with pytest.raises(RuntimeError, match="buffer-convertible"):
        task.result()


def test_backward_compat_map_spec_why_existing_native_kernel_api_must_not_break() -> None:
    x = kernel.arg("x")
    spec = kernel.elementwise(x * 2.0)
    source = rt.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    mapped = kernel.map(spec, x=source)

    task = rt.go(mapped)
    result: Any = task.result()

    np.testing.assert_array_equal(result.numpy(), np.array([2.0, 4.0, 6.0], dtype=np.float64))


def test_backward_compat_reduce_spec_why_existing_native_reduce_api_must_not_break() -> None:
    source = rt.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    reduce = kernel.sum(source)

    task = rt.go(reduce)
    result: Any = task.result()

    assert result.scalar() == 6.0


def test_backward_compat_map_with_channel_why_native_channel_delivery_must_still_work() -> None:
    x = kernel.arg("x")
    spec = kernel.elementwise(x + 1.0)
    source = rt.asarray(np.array([10.0, 20.0], dtype=np.float64))
    mapped = kernel.map(spec, x=source)
    output = chan(1)

    task = rt.go(mapped, out=output)

    np.testing.assert_array_equal(output.recv().numpy(), np.array([11.0, 21.0], dtype=np.float64))
    np.testing.assert_array_equal(task.result().numpy(), np.array([11.0, 21.0], dtype=np.float64))


def test_go_non_callable_non_spec_raises_type_error_why_invalid_first_arg_must_fail_fast() -> None:
    with pytest.raises(TypeError, match="MapSpec, ReduceSpec, or callable"):
        rt.go(42)
