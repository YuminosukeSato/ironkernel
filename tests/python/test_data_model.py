"""Data model boundary and edge case tests."""

import numpy as np
import numpy.testing as npt
import pytest
from ironkernel import chan, kernel, rt


def test_empty_mean_raises() -> None:
    """mean() on empty buffer raises ValueError."""
    buf = rt.asarray(np.array([], dtype=np.float64))
    with pytest.raises(ValueError, match="empty"):
        rt.go(kernel.mean(buf)).result()


def test_single_element_reduce() -> None:
    """sum and mean on single-element buffer."""
    buf = rt.asarray(np.array([42.0]))
    assert rt.go(kernel.sum(buf)).result().scalar() == 42.0
    assert rt.go(kernel.mean(buf)).result().scalar() == 42.0


def test_channel_send_empty_buffer() -> None:
    """Empty buffer can be sent and received through channel."""
    c = chan(1)
    buf = rt.asarray(np.array([], dtype=np.float64))
    c.send(buf)
    received = c.recv()
    assert len(received.numpy()) == 0


def test_channel_capacity_1_fifo() -> None:
    """Channel with capacity 1 maintains FIFO order."""
    c = chan(1)

    for i in range(5):
        buf = rt.asarray(np.array([float(i)]))
        c.send(buf)
        received = c.recv()
        assert received.scalar() == float(i)


def test_par_threshold_boundary() -> None:
    """Results match at PAR_THRESHOLD boundary: 4095, 4096, 4097."""
    x = kernel.arg("x")
    spec = kernel.elementwise(x * 2.0 + 1.0)

    for size in [4095, 4096, 4097]:
        data = np.arange(size, dtype=np.float64)
        buf = rt.asarray(data)
        out = rt.go(kernel.map(spec, x=buf)).result().numpy()
        expected = data * 2.0 + 1.0
        npt.assert_array_equal(out, expected)
