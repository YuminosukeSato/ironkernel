"""Regression matrix for public API boundary and error behavior."""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from ironkernel import kernel, rt


@pytest.mark.parametrize(
    "size",
    [0, 1, 4095, 4096],
    ids=[
        "empty_input_why_zero_length_must_stay_safe",
        "single_element_why_scalar_like_shapes_must_stay_correct",
        "threshold_minus_one_why_sequential_path_must_match_public_contract",
        "threshold_exact_why_parallel_path_must_match_public_contract",
    ],
)
def test_map_boundaries_preserve_results_why_empty_single_and_threshold_inputs_must_stay_safe(
    size: int,
) -> None:
    x = kernel.arg("x")
    spec = kernel.elementwise(x + 1.0)
    data = np.arange(size, dtype=np.float64)

    result = rt.go(kernel.map(spec, x=rt.asarray(data))).result().numpy()

    np.testing.assert_array_equal(result, data + 1.0)


def test_shape_mismatch_raises_value_error_why_argument_shape_contract_must_not_silently_change() -> None:
    x, y = kernel.args("x", "y")
    spec = kernel.elementwise(x + y)

    with pytest.raises(ValueError, match="buffer length mismatch"):
        rt.go(
            kernel.map(
                spec,
                x=rt.asarray(np.array([1.0, 2.0], dtype=np.float64)),
                y=rt.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            )
        ).result()


def test_wrong_argument_type_raises_type_error_why_python_boundary_must_fail_fast() -> None:
    x = kernel.arg("x")
    spec = kernel.elementwise(x + 1.0)

    with pytest.raises(TypeError, match="Buffer or numeric"):
        kernel.map(spec, x="not-a-buffer")


def test_multiple_python_threads_can_finish_independent_tasks_why_public_api_must_stay_thread_robust() -> None:
    a, x, y = kernel.args("a", "x", "y")
    spec = kernel.elementwise(a * x + y)
    offsets = [0.0, 100.0, 1000.0, 10_000.0]
    barrier = threading.Barrier(len(offsets))

    def run_task(offset: float) -> np.ndarray:
        x_data = np.arange(256, dtype=np.float64) + offset
        y_data = np.ones(256, dtype=np.float64)
        barrier.wait()
        return (
            rt.go(
                kernel.map(
                    spec,
                    a=2.0,
                    x=rt.asarray(x_data),
                    y=rt.asarray(y_data),
                )
            )
            .result()
            .numpy()
        )

    with ThreadPoolExecutor(max_workers=len(offsets)) as executor:
        results = list(executor.map(run_task, offsets))

    for offset, result in zip(offsets, results):
        expected = 2.0 * (np.arange(256, dtype=np.float64) + offset) + 1.0
        np.testing.assert_array_equal(result, expected)
