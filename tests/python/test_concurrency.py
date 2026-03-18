"""Concurrency tests for channel operations and task execution."""

import threading

import numpy as np
from ironkernel import chan, kernel, rt


def test_concurrent_channel_operations() -> None:
    """4 threads send to the same channel concurrently."""
    c = chan(100)
    n_threads = 4
    n_per_thread = 25
    threads = []

    def sender(thread_id: int) -> None:
        for i in range(n_per_thread):
            val = float(thread_id * n_per_thread + i)
            c.send(rt.asarray(np.array([val])))

    for t in range(n_threads):
        th = threading.Thread(target=sender, args=(t,))
        threads.append(th)
        th.start()

    results = []
    for _ in range(n_threads * n_per_thread):
        results.append(c.recv().scalar())

    for th in threads:
        th.join()

    assert sorted(results) == [float(i) for i in range(n_threads * n_per_thread)]


def test_concurrent_go_out_channel() -> None:
    """4 tasks submit to the same out=channel concurrently."""
    c = chan(100)
    x = kernel.arg("x")
    spec = kernel.elementwise(x)
    tasks = []

    for i in range(4):
        buf = rt.asarray(np.array([float(i)]))
        task = rt.go(kernel.map(spec, x=buf), out=c)
        tasks.append(task)

    results = []
    for _ in range(4):
        results.append(c.recv().scalar())

    task_results = [task.result().scalar() for task in tasks]
    assert sorted(results) == [0.0, 1.0, 2.0, 3.0]
    assert sorted(task_results) == [0.0, 1.0, 2.0, 3.0]


def test_result_does_not_hold_gil() -> None:
    """task.result() releases GIL, allowing other threads to run."""
    x = kernel.arg("x")
    spec = kernel.elementwise(x)
    buf = rt.asarray(np.array([1.0]))
    task = rt.go(kernel.map(spec, x=buf))

    progress = {"count": 0}

    def background() -> None:
        for _ in range(100):
            progress["count"] += 1

    th = threading.Thread(target=background)
    th.start()

    # result() should not block the background thread.
    _result = task.result()
    th.join()

    assert progress["count"] == 100
