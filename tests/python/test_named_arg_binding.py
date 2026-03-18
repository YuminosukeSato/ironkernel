import numpy as np
import pytest
from ironkernel import kernel, rt


def test_separate_arg_calls_compose_correctly_why_independent_arg_builders_must_not_alias_index_zero() -> None:
    x = kernel.arg("x")
    y = kernel.arg("y")
    spec = kernel.elementwise(x + y)

    x_buf = rt.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    y_buf = rt.asarray(np.array([10.0, 20.0, 30.0], dtype=np.float64))

    out = rt.go(kernel.map(spec, x=x_buf, y=y_buf)).result().numpy()

    np.testing.assert_array_equal(out, np.array([11.0, 22.0, 33.0], dtype=np.float64))


def test_kwargs_order_does_not_matter_why_named_binding_should_follow_kernel_spec_names_not_dict_order() -> None:
    z = kernel.arg("z")
    a = kernel.arg("a")
    m = kernel.arg("m")
    spec = kernel.elementwise(z + a * m)

    z_buf = rt.asarray(np.array([100.0, 200.0], dtype=np.float64))
    a_buf = rt.asarray(np.array([2.0, 3.0], dtype=np.float64))
    m_buf = rt.asarray(np.array([4.0, 5.0], dtype=np.float64))

    forward = rt.go(kernel.map(spec, z=z_buf, a=a_buf, m=m_buf)).result().numpy()
    reverse = rt.go(kernel.map(spec, m=m_buf, z=z_buf, a=a_buf)).result().numpy()

    np.testing.assert_array_equal(forward, np.array([108.0, 215.0], dtype=np.float64))
    np.testing.assert_array_equal(reverse, forward)


def test_missing_arg_raises_value_error_why_map_execution_should_fail_fast_on_incomplete_bindings() -> None:
    x = kernel.arg("x")
    y = kernel.arg("y")
    spec = kernel.elementwise(x + y)
    x_buf = rt.asarray(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ValueError, match="missing arg: y"):
        rt.go(kernel.map(spec, x=x_buf)).result()


def test_extra_arg_raises_value_error_why_unknown_kwargs_should_not_be_silently_ignored() -> None:
    x = kernel.arg("x")
    spec = kernel.elementwise(x + 1.0)
    x_buf = rt.asarray(np.array([1.0, 2.0], dtype=np.float64))
    y_buf = rt.asarray(np.array([3.0, 4.0], dtype=np.float64))

    with pytest.raises(ValueError, match="unexpected args: y"):
        rt.go(kernel.map(spec, x=x_buf, y=y_buf)).result()


def test_duplicate_arg_name_raises_value_error_why_ambiguous_name_binding_cannot_be_resolved_safely() -> None:
    duplicate_left = kernel.arg("x")
    duplicate_right = kernel.arg("x")

    with pytest.raises(ValueError, match="duplicate arg name: x"):
        kernel.elementwise(duplicate_left + duplicate_right)


def test_non_alphabetical_args_bind_by_name_why_arg_resolution_should_ignore_lexicographic_sorting() -> None:
    z, a, m = kernel.args("z", "a", "m")
    spec = kernel.elementwise(z - a + m)

    out = rt.go(
        kernel.map(
            spec,
            m=rt.asarray(np.array([7.0, 8.0], dtype=np.float64)),
            z=rt.asarray(np.array([10.0, 20.0], dtype=np.float64)),
            a=rt.asarray(np.array([1.0, 2.0], dtype=np.float64)),
        )
    ).result().numpy()

    np.testing.assert_array_equal(out, np.array([16.0, 26.0], dtype=np.float64))


def test_args_with_single_name_why_plural_constructor_should_still_return_a_python_list_for_uniformity() -> None:
    exprs = kernel.args("x")

    assert isinstance(exprs, list)
    assert len(exprs) == 1
    assert repr(exprs[0]) == "Expr(x)"


def test_args_empty_raises_why_empty_argument_declarations_do_not_define_a_valid_kernel_contract() -> None:
    with pytest.raises(ValueError, match="at least one arg name is required"):
        kernel.args()


def test_scalar_broadcast_with_named_args_why_scalar_bindings_should_follow_same_name_resolution_as_buffers() -> None:
    scale = kernel.arg("scale")
    x = kernel.arg("x")
    spec = kernel.elementwise(scale * x + 1.0)
    x_buf = rt.asarray(np.array([2.0, 4.0], dtype=np.float64))

    out = rt.go(kernel.map(spec, x=x_buf, scale=3.0)).result().numpy()

    np.testing.assert_array_equal(out, np.array([7.0, 13.0], dtype=np.float64))
