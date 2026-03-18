"""Reference example for the current ironkernel public API."""

from __future__ import annotations

from pprint import pprint

import numpy as np
from ironkernel import RecvCase, kernel, rt, select


def main() -> dict[str, object]:
    a, x, y = kernel.args("a", "x", "y")
    saxpy = kernel.elementwise(a * x + y)

    x_data = np.arange(4, dtype=np.float64)
    y_data = np.ones(4, dtype=np.float64)
    bx = rt.asarray(x_data)
    by = rt.asarray(y_data)
    mapped = rt.go(kernel.map(saxpy, a=2.0, x=bx, y=by)).result().numpy()

    arg_x = kernel.arg("x")
    transform = kernel.elementwise(kernel.sqrt(kernel.abs(arg_x)) + kernel.sin(arg_x))
    transformed = rt.go(kernel.map(transform, x=bx)).result().numpy()

    reduce_input = rt.asarray(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
    total = rt.go(kernel.sum(reduce_input)).result().scalar()
    average = rt.go(kernel.mean(reduce_input)).result().scalar()

    channel = rt.chan(1)
    channel.send(rt.asarray(np.array([42.0, 43.0], dtype=np.float64)))
    received = channel.recv().numpy()

    ready = rt.chan(1)
    waiting = rt.chan(1)
    ready.send(rt.asarray(np.array([99.0], dtype=np.float64)))
    selected_index, selected_buffer = select(RecvCase(ready), RecvCase(waiting), default=True)
    assert selected_buffer is not None
    selected_value = selected_buffer.numpy()

    relu = kernel.elementwise(kernel.where(arg_x > 0, arg_x, 0.0))
    relu_input = rt.asarray(np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float64))
    relu = rt.go(kernel.map(relu, x=relu_input)).result().numpy()

    return {
        "mapped": mapped,
        "transformed": transformed,
        "total": total,
        "average": average,
        "received": received,
        "selected_index": selected_index,
        "selected_value": selected_value,
        "relu": relu,
    }


if __name__ == "__main__":
    pprint(main())
