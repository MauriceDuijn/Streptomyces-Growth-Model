from time import perf_counter
from contextlib import contextmanager
from collections import defaultdict


class Timer:
    def __init__(self):
        self.line_times: dict[str, float] = defaultdict(float)
        self._prev_times: dict[str, float] = defaultdict(float)

    @property
    def total_time(self):
        return sum(self.line_times.values())

    def print_times(self):
        indent_size = max(map(len, self.line_times.keys())) + 1
        print(f"total time {self.total_time:>20}")
        for label, time in self.line_times.items():
            print(f"{label:>{indent_size}} took in total {time:<25}, perc. {time/self.total_time:.3f}")

    def measure_start(self, label: str):
        """
        Hold the start time when function is called.

        :param label: Each label can hold a independent time, only correlates with an end measure with same label.
        """
        self._prev_times[label] = perf_counter()

    def measure_end(self, label: str):
        """
        Benchmark the total between the start and end of a given label.

        :param label: Calculate the time between start and end measure.
        """
        self.line_times[label] += perf_counter() - self._prev_times[label]

    @contextmanager
    def measure(self, label: str):
        """
        Measures how much a set of code takes in given context window.

        Use case:
        with timer_obj.measure(label):
            benchmark code

        other code that is not benchmarked
        """
        start = perf_counter()
        yield
        total = perf_counter() - start
        self.line_times[label] += total

    def measure_decorator(self, label: str):
        """
        Decorator version of the measure context manager.

        Use case:
        @timer_obj.measure_decorator(label)
        def benchmarked_function():
            benchmark code
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.measure(label):
                    return func(*args, **kwargs)

            return wrapper

        return decorator


if __name__ == '__main__':
    timer_1 = Timer()
    timer_2 = Timer()

    @timer_1.measure_decorator("Function decorator")
    def add_one(i):
        """Dummy function"""
        return i + 1

    for i in range(100_000):
        with timer_2.measure("add one"):
            add_one(i)

    for i in range(100_000):
        with timer_2.measure("add one v2"):
            i += 1

    timer_1.print_times()
    timer_2.print_times()
