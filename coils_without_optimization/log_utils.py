"""Simple logging and timing utilities.

The functions in this module provide lightweight print–based logging and
context management for timing code blocks.  They flush immediately to
ensure output appears in the correct order when running in parallel
processes.
"""

import time
from contextlib import contextmanager


def log(msg: str) -> None:
    """Log a message to stdout and flush immediately."""
    print(msg, flush=True)


@contextmanager
def time_block(label: str):
    """Context manager to time a block of code.

    Usage:

    >>> with time_block("my operation"):
    ...     do_work()
    ...
    my operation took 0.42 s
    """
    t0 = time.time()
    yield
    dt = time.time() - t0
    log(f"{label} took {dt:.2f} s")