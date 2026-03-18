import importlib

import ironkernel
import numpy as np
import pytest
from ironkernel import kernel, rt


def test_kernel_where_exists_why_public_api_should_not_expose_pyo3_escape_hack() -> None:
    assert hasattr(kernel, "where")


def test_kernel_args_requires_name_why_placeholder_creation_without_names_would_break_binding_contract() -> None:
    with pytest.raises(ValueError, match="at least one arg name is required"):
        kernel.args()


def test_kernel_where_computes_correctly_why_facade_alias_must_preserve_where_semantics() -> None:
    x = kernel.arg("x")
    relu = kernel.elementwise(kernel.where(x > 0, x, 0.0))
    data = rt.asarray(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64))

    out = rt.go(kernel.map(relu, x=data)).result().numpy()

    np.testing.assert_array_equal(out, np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float64))


def test_round_not_exposed_why_frozen_v1_api_should_hide_unapproved_operations() -> None:
    assert not hasattr(kernel, "round")


def test_chan_at_root_why_runtime_channel_factory_should_be_available_from_public_package() -> None:
    channel = ironkernel.chan(1)

    assert channel.__class__.__name__ == "Channel"


def test_kernel_is_kernel_facade_why_public_singleton_should_be_python_facade_not_raw_pyo3_type() -> None:
    assert kernel.__class__.__name__ == "KernelFacade"


def test_rt_is_runtime_facade_why_public_singleton_should_be_python_facade_not_raw_pyo3_type() -> None:
    assert rt.__class__.__name__ == "RuntimeFacade"


def test_root_imports_all_why_public_entrypoint_should_export_supported_runtime_types() -> None:
    module = importlib.import_module("ironkernel")

    for name in (
        "Buffer",
        "Channel",
        "Expr",
        "KernelSpec",
        "MapSpec",
        "RecvCase",
        "ReduceSpec",
        "TaskHandle",
        "kernel",
        "rt",
        "select",
        "chan",
    ):
        assert hasattr(module, name), name


def test_internal_modules_not_exported_why_raw_backend_construction_should_stay_private() -> None:
    assert "_KernelModule" not in dir(ironkernel)
    assert "_RuntimeModule" not in dir(ironkernel)


def test_where_underscore_not_in_root_why_escape_hatch_name_should_not_leak_into_public_api() -> None:
    assert "where_" not in dir(ironkernel)


def test_public_math_facade_methods_execute_why_python_wrappers_must_not_drift_from_backend_contracts() -> None:
    x = kernel.arg("x")
    y = kernel.arg("y")

    log2_input = rt.asarray(np.array([1.0, 2.0, 8.0], dtype=np.float64))
    log10_input = rt.asarray(np.array([1.0, 10.0, 100.0], dtype=np.float64))
    trig_input = rt.asarray(np.array([0.0, np.pi / 4.0], dtype=np.float64))
    atan_y = rt.asarray(np.array([0.0, 1.0], dtype=np.float64))
    atan_x = rt.asarray(np.array([1.0, 1.0], dtype=np.float64))
    min_left = rt.asarray(np.array([5.0, -1.0, 3.0], dtype=np.float64))
    min_right = rt.asarray(np.array([2.0, -2.0, 4.0], dtype=np.float64))
    pow_base = rt.asarray(np.array([2.0, 3.0, 4.0], dtype=np.float64))
    pow_exp = rt.asarray(np.array([3.0, 2.0, 1.0], dtype=np.float64))

    log2_out = rt.go(kernel.map(kernel.elementwise(kernel.log2(x)), x=log2_input)).result().numpy()
    log10_out = rt.go(kernel.map(kernel.elementwise(kernel.log10(x)), x=log10_input)).result().numpy()
    cos_out = rt.go(kernel.map(kernel.elementwise(kernel.cos(x)), x=trig_input)).result().numpy()
    tan_out = rt.go(kernel.map(kernel.elementwise(kernel.tan(x)), x=trig_input)).result().numpy()
    pow_out = rt.go(kernel.map(kernel.elementwise(kernel.pow(x, y)), x=pow_base, y=pow_exp)).result().numpy()
    atan2_out = rt.go(kernel.map(kernel.elementwise(kernel.atan2(y, x)), x=atan_x, y=atan_y)).result().numpy()
    min_out = rt.go(kernel.map(kernel.elementwise(kernel.min(x, y)), x=min_left, y=min_right)).result().numpy()
    max_out = rt.go(kernel.map(kernel.elementwise(kernel.max(x, y)), x=min_left, y=min_right)).result().numpy()

    np.testing.assert_allclose(log2_out, np.array([0.0, 1.0, 3.0], dtype=np.float64))
    np.testing.assert_allclose(log10_out, np.array([0.0, 1.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(cos_out, np.cos(np.array([0.0, np.pi / 4.0], dtype=np.float64)))
    np.testing.assert_allclose(tan_out, np.tan(np.array([0.0, np.pi / 4.0], dtype=np.float64)))
    np.testing.assert_allclose(pow_out, np.array([8.0, 9.0, 4.0], dtype=np.float64))
    np.testing.assert_allclose(atan2_out, np.arctan2(np.array([0.0, 1.0]), np.array([1.0, 1.0])))
    np.testing.assert_array_equal(min_out, np.array([2.0, -2.0, 3.0], dtype=np.float64))
    np.testing.assert_array_equal(max_out, np.array([5.0, -1.0, 4.0], dtype=np.float64))


def test_public_reduce_facade_methods_execute_why_wrapper_entrypoints_must_cover_all_supported_reductions() -> None:
    buf = rt.asarray(np.array([3.0, -1.0, 9.0], dtype=np.float64))

    assert rt.go(kernel.min_reduce(buf)).result().scalar() == -1.0
    assert rt.go(kernel.max_reduce(buf)).result().scalar() == 9.0
