"""Tests for channel and select."""

import numpy as np
from ironkernel import RecvCase, rt, select


class TestChannel:
    def test_send_recv(self) -> None:
        c = rt.chan(10)
        c.send(rt.asarray(np.array([1.0, 2.0, 3.0])))
        buf = c.recv()
        np.testing.assert_array_equal(buf.numpy(), [1.0, 2.0, 3.0])

    def test_multiple_send_recv(self) -> None:
        c = rt.chan(10)
        for i in range(5):
            c.send(rt.asarray(np.array([float(i)])))
        for i in range(5):
            buf = c.recv()
            assert buf.numpy()[0] == float(i)


class TestSelect:
    def test_select_receives(self) -> None:
        ch_a = rt.chan(10)
        ch_b = rt.chan(10)
        ch_a.send(rt.asarray(np.array([42.0])))
        idx, val = select(RecvCase(ch_a), RecvCase(ch_b), default=True)
        assert idx == 0
        assert val.numpy()[0] == 42.0

    def test_select_default(self) -> None:
        ch_a = rt.chan(10)
        ch_b = rt.chan(10)
        idx, val = select(RecvCase(ch_a), RecvCase(ch_b), default=True)
        assert idx == -1
        assert val is None

    def test_select_second_channel(self) -> None:
        ch_a = rt.chan(10)
        ch_b = rt.chan(10)
        ch_b.send(rt.asarray(np.array([99.0])))
        idx, val = select(RecvCase(ch_a), RecvCase(ch_b), default=True)
        assert idx == 1
        assert val.numpy()[0] == 99.0
