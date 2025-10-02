import time
from contextlib import contextmanager

def log(msg: str) -> None:
    print(msg, flush=True)

@contextmanager
def time_block(label: str):
    t0 = time.time()
    yield
    dt = time.time() - t0
    log(f"{label} took {dt:.2f} s")
