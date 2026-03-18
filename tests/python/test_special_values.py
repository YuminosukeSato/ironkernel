"""Special value tests: NaN, Inf, -0.0 propagation across PyO3 boundary."""

import math

import numpy as np
from ironkernel import chan, kernel, rt


def test_nan_propagation_through_map() -> None:
    """NaN input propagates through elementwise map."""
    x = kernel.arg("x")
    spec = kernel.elementwise(x + 1.0)
    buf = rt.asarray(np.array([1.0, float("nan"), 3.0]))
    out = rt.go(kernel.map(spec, x=buf)).result().numpy()

    assert not math.isnan(out[0])
    assert math.isnan(out[1])
    assert not math.isnan(out[2])


def test_inf_propagation_through_map() -> None:
    """Inf input propagates through elementwise map."""
    x = kernel.arg("x")
    spec = kernel.elementwise(x * 2.0)
    buf = rt.asarray(np.array([float("inf"), float("-inf"), 1.0]))
    out = rt.go(kernel.map(spec, x=buf)).result().numpy()

    assert out[0] == float("inf")
    assert out[1] == float("-inf")
    assert out[2] == 2.0


def test_negative_zero_preservation() -> None:
    """-0.0 is preserved through identity map (copysign check)."""
    x = kernel.arg("x")
    spec = kernel.elementwise(x)
    buf = rt.asarray(np.array([-0.0, 0.0, 1.0]))
    out = rt.go(kernel.map(spec, x=buf)).result().numpy()

    assert math.copysign(1.0, out[0]) == -1.0  # -0.0
    assert math.copysign(1.0, out[1]) == 1.0  # +0.0


def test_nan_reduce_sum() -> None:
    """NaN in array makes sum return NaN."""
    buf = rt.asarray(np.array([1.0, float("nan"), 3.0]))
    result = rt.go(kernel.sum(buf)).result().scalar()
    assert math.isnan(result)


def test_nan_through_channel() -> None:
    """NaN buffer survives channel send/recv."""
    c = chan(1)
    buf = rt.asarray(np.array([float("nan"), 1.0]))
    c.send(buf)
    received = c.recv().numpy()
    assert math.isnan(received[0])
    assert received[1] == 1.0


def test_special_values_in_where() -> None:
    """NaN/Inf/-0.0 in where branch selection."""
    x = kernel.arg("x")
    # where(x > 0, x, -0.0)
    spec = kernel.elementwise(kernel.where(x > 0, x, 0))
    buf = rt.asarray(np.array([float("nan"), float("inf"), -1.0, 1.0]))
    out = rt.go(kernel.map(spec, x=buf)).result().numpy()

    # NaN > 0 is false → 0.0
    assert out[0] == 0.0
    # Inf > 0 is true → Inf
    assert out[1] == float("inf")
    # -1.0 > 0 is false → 0.0
    assert out[2] == 0.0
    # 1.0 > 0 is true → 1.0
    assert out[3] == 1.0
