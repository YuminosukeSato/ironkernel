from __future__ import annotations

import time

import numpy as np
import pytest
from ironkernel import RecvCase, chan, kernel, rt, select


def test_callable_pipeline_between_channels_why_out_channel_results_must_feed_follow_up_stages() -> None:
    stage_one = chan(3)
    stage_two = chan(3)

    upstream = [rt.go(lambda value=value: float(value), out=stage_one) for value in (1, 2, 3)]
    downstream = [rt.go(lambda value: value * 10.0, stage_one.recv().scalar(), out=stage_two) for _ in range(3)]

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


def test_fan_out_fan_in_why_multiple_producers_must_feed_single_consumer_channel() -> None:
    collector = chan(10)

    tasks = [rt.go(lambda i=i: float(i * 10), out=collector) for i in range(5)]

    results = sorted(collector.recv().scalar() for _ in range(5))
    assert results == [0.0, 10.0, 20.0, 30.0, 40.0]
    for task in tasks:
        task.result()


def test_backpressure_bounded_channel_why_producers_must_not_overflow_slow_consumers() -> None:
    narrow = chan(2)

    t1 = rt.go(lambda: 1.0, out=narrow)
    t2 = rt.go(lambda: 2.0, out=narrow)
    t1.result()
    t2.result()

    v1 = narrow.recv().scalar()
    v2 = narrow.recv().scalar()
    assert sorted([v1, v2]) == [1.0, 2.0]


def test_error_propagation_in_pipeline_why_upstream_failures_must_not_silently_drop() -> None:
    output = chan(1)

    task = rt.go(
        lambda: (_ for _ in ()).throw(ValueError("stage1 fail")),
        out=output,
    )

    with pytest.raises(RuntimeError, match="stage1 fail"):
        task.result()


def test_cancel_in_pipeline_why_cancelled_tasks_must_not_deliver_to_channels() -> None:
    output = chan(1)

    task = rt.go(lambda: (time.sleep(0.1), 999.0)[1], out=output)
    task.cancel()

    with pytest.raises(RuntimeError, match="cancelled"):
        task.result()


def test_100_tasks_throughput_why_concurrent_load_must_not_deadlock_or_lose_results() -> None:
    tasks = [rt.go(lambda i=i: i * i) for i in range(100)]

    results = sorted(task.result() for task in tasks)
    expected = sorted(i * i for i in range(100))
    assert results == expected


def test_three_stage_pipeline_why_multi_hop_channel_chains_must_preserve_data_integrity() -> None:
    ch1 = chan(5)
    ch2 = chan(5)
    ch3 = chan(5)

    stage1_tasks = [rt.go(lambda v=v: float(v), out=ch1) for v in range(5)]

    stage2_tasks = [rt.go(lambda x: x * 2.0, ch1.recv().scalar(), out=ch2) for _ in range(5)]

    stage3_tasks = [rt.go(lambda x: x + 100.0, ch2.recv().scalar(), out=ch3) for _ in range(5)]

    results = sorted(ch3.recv().scalar() for _ in range(5))
    assert results == [100.0, 102.0, 104.0, 106.0, 108.0]

    for task in stage1_tasks + stage2_tasks + stage3_tasks:
        task.result()


def test_mixed_native_callable_fan_in_why_heterogeneous_producers_must_coexist_on_same_channel() -> None:
    collector = chan(2)
    x = kernel.arg("x")
    spec = kernel.elementwise(x * 3.0)
    source = rt.asarray(np.array([1.0], dtype=np.float64))

    native_task = rt.go(kernel.map(spec, x=source), out=collector)
    callable_task = rt.go(lambda: 99.0, out=collector)

    results = sorted([collector.recv().scalar(), collector.recv().scalar()])
    assert results == [3.0, 99.0]

    native_task.result()
    callable_task.result()
