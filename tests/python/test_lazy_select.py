import numpy as np
from ironkernel import kernel, rt


def test_sqrt_negative_not_evaluated_in_false_branch_why_where_must_not_touch_dead_branch_values() -> None:
    x = kernel.arg("x")
    spec = kernel.elementwise(kernel.where(x > 0, kernel.sqrt(x), 0.0))
    data = rt.asarray(np.array([-1.0, -4.0], dtype=np.float64))

    out = rt.go(kernel.map(spec, x=data)).result().numpy()

    np.testing.assert_array_equal(out, np.array([0.0, 0.0], dtype=np.float64))


def test_nan_in_unselected_branch_does_not_contaminate_why_dead_branch_payloads_must_not_leak() -> None:
    cond = kernel.arg("cond")
    left = kernel.arg("left")
    right = kernel.arg("right")
    spec = kernel.elementwise(kernel.where(cond, left, right))

    out = (
        rt.go(
            kernel.map(
                spec,
                cond=rt.asarray(np.array([1.0, 1.0], dtype=np.float64)),
                left=rt.asarray(np.array([5.0, 6.0], dtype=np.float64)),
                right=rt.asarray(np.array([np.nan, np.nan], dtype=np.float64)),
            )
        )
        .result()
        .numpy()
    )

    np.testing.assert_array_equal(out, np.array([5.0, 6.0], dtype=np.float64))


def test_where_identical_branches_equals_original_why_select_should_reduce_to_identity_when_both_sides_match() -> None:
    cond = kernel.arg("cond")
    x = kernel.arg("x")
    spec = kernel.elementwise(kernel.where(cond, x, x))
    values = np.array([3.0, -2.0, 0.0, 8.0], dtype=np.float64)

    out = (
        rt.go(
            kernel.map(
                spec,
                cond=rt.asarray(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)),
                x=rt.asarray(values),
            )
        )
        .result()
        .numpy()
    )

    np.testing.assert_array_equal(out, values)


def test_where_nested_why_nested_selects_must_choose_branches_independently_per_element() -> None:
    c1 = kernel.arg("c1")
    c2 = kernel.arg("c2")
    a = kernel.arg("a")
    b = kernel.arg("b")
    c = kernel.arg("c")
    nested = kernel.where(c2, a, b)
    spec = kernel.elementwise(kernel.where(c1, nested, c))

    out = (
        rt.go(
            kernel.map(
                spec,
                c1=rt.asarray(np.array([1.0, 1.0, 0.0], dtype=np.float64)),
                c2=rt.asarray(np.array([1.0, 0.0, 1.0], dtype=np.float64)),
                a=rt.asarray(np.array([10.0, 20.0, 30.0], dtype=np.float64)),
                b=rt.asarray(np.array([100.0, 200.0, 300.0], dtype=np.float64)),
                c=rt.asarray(np.array([7.0, 8.0, 9.0], dtype=np.float64)),
            )
        )
        .result()
        .numpy()
    )

    np.testing.assert_array_equal(out, np.array([10.0, 200.0, 9.0], dtype=np.float64))


def test_where_parallel_vs_sequential_match_why_threshold_switch_must_not_change_semantics() -> None:
    x = kernel.arg("x")
    spec = kernel.elementwise(kernel.where(x > 0, x, 0.0))

    for size in (4095, 4096, 4097):
        data = np.linspace(-2.0, 2.0, size, dtype=np.float64)
        expected = np.where(data > 0, data, 0.0)

        out = rt.go(kernel.map(spec, x=rt.asarray(data))).result().numpy()

        np.testing.assert_array_equal(out, expected)
