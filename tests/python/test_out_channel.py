"""Tests for rt.go(spec, out=channel) delivery pipeline."""

import numpy as np
import numpy.testing as npt
import pytest
from ironkernel import Buffer, chan, kernel, rt


def _make_identity_map(buf: Buffer):
    """Create a MapSpec that returns the input buffer unchanged."""
    x = kernel.arg("x")
    spec = kernel.elementwise(x)
    return kernel.map(spec, x=buf)


class TestGoWithOutChannel:
    def test_go_with_out_channel(self):
        """Result is delivered to the channel and task completes."""
        buf = rt.asarray(np.array([1.0, 2.0, 3.0]))
        c = chan(10)
        map_spec = _make_identity_map(buf)

        task = rt.go(map_spec, out=c)
        received = c.recv()

        npt.assert_array_equal(received.numpy(), np.array([1.0, 2.0, 3.0]))
        assert task.is_done()

    def test_reduce_out_channel(self):
        """Reduce result is delivered to the channel as a single-element buffer."""
        buf = rt.asarray(np.array([1.0, 2.0, 3.0]))
        c = chan(10)
        reduce_spec = kernel.sum(buf)

        task = rt.go(reduce_spec, out=c)
        received = c.recv()

        npt.assert_array_equal(received.numpy(), np.array([6.0]))
        assert task.is_done()

    def test_channel_closed_fails_task(self):
        """Sending to a closed channel fails the task."""
        buf = rt.asarray(np.array([1.0]))
        c = chan(10)
        c.close()
        map_spec = _make_identity_map(buf)

        task = rt.go(map_spec, out=c)

        with pytest.raises(RuntimeError, match="channel closed"):
            task.result()

    def test_delivery_completion_marks_task_completed(self):
        """Task transitions to completed after successful delivery."""
        buf = rt.asarray(np.array([42.0]))
        c = chan(10)
        map_spec = _make_identity_map(buf)

        task = rt.go(map_spec, out=c)

        # Recv to let delivery complete.
        c.recv()
        assert task.is_done()

    def test_go_out_none_is_default(self):
        """out=None (default) behaves like go() without out."""
        buf = rt.asarray(np.array([5.0, 6.0]))
        map_spec = _make_identity_map(buf)

        task = rt.go(map_spec)
        result = task.result()

        npt.assert_array_equal(result.numpy(), np.array([5.0, 6.0]))

    def test_task_result_after_channel_recv(self):
        """task.result() works after channel.recv() completes."""
        buf = rt.asarray(np.array([7.0, 8.0]))
        c = chan(10)
        map_spec = _make_identity_map(buf)

        task = rt.go(map_spec, out=c)
        _received = c.recv()
        result = task.result()

        npt.assert_array_equal(result.numpy(), np.array([7.0, 8.0]))

    def test_cancelled_task_result_raises(self):
        """cancel() followed by result() raises RuntimeError."""
        buf = rt.asarray(np.array([1.0]))
        map_spec = _make_identity_map(buf)

        task = rt.go(map_spec)
        # Task completes synchronously when out=None, so cancel may fail.
        # This tests the contract: if cancel succeeds, result raises.
        if task.cancel():
            with pytest.raises(RuntimeError):
                task.result()

    def test_go_out_channel_large_buffer(self):
        """1M elements delivered through out=channel."""
        data = np.arange(1_000_000, dtype=np.float64)
        buf = rt.asarray(data)
        c = chan(1)
        map_spec = _make_identity_map(buf)

        task = rt.go(map_spec, out=c)
        received = c.recv()

        npt.assert_array_equal(received.numpy(), data)
        assert task.is_done()

    def test_multiple_go_out_same_channel(self):
        """Multiple go() calls can deliver to the same channel."""
        c = chan(10)
        results = []

        for i in range(5):
            buf = rt.asarray(np.array([float(i)]))
            map_spec = _make_identity_map(buf)
            rt.go(map_spec, out=c)

        for _ in range(5):
            received = c.recv()
            results.append(received.scalar())

        assert sorted(results) == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_channel_close_is_closed(self):
        """Channel.close() and is_closed() work correctly."""
        c = chan(10)
        assert not c.is_closed()
        c.close()
        assert c.is_closed()

    def test_channel_close_idempotent(self):
        """Calling close() multiple times does not raise."""
        c = chan(10)
        c.close()
        c.close()
        assert c.is_closed()
