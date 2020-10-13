import time


class BenchmarkTimer:
    """Provides a context manager that runs a code block `reps` times
    and records results to the instance variable `timings`. Use like:
    .. code-block:: python
        timer = BenchmarkTimer(rep=5)
        for _ in timer.benchmark_runs():
            ... do something ...
        print(np.min(timer.timings))
    """

    def __init__(self, reps=1):
        self.reps = reps
        self.timings = []

    def benchmark_runs(self):
        for r in range(self.reps):
            t0 = time.time()
            yield r
            t1 = time.time()
            self.timings.append(t1 - t0)
