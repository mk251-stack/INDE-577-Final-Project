# src/rice_ml/utils/timing.py

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional, Dict


@contextmanager
def time_block(
    label: str,
    store: Optional[Dict[str, float]] = None,
    verbose: bool = True,
):
    """
    Context manager to measure wall-clock time for a code block.

    Example
    -------
    >>> with time_block("graph_build", store=timings):
    ...     W = build_graph(...)
    >>> print(timings["graph_build"])

    Parameters
    ----------
    label : str
        Name of the block (key used in `store`).

    store : dict, optional
        If provided, elapsed time will be stored as store[label] = seconds.

    verbose : bool, default=True
        If True, prints a message when the block completes.
    """
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    dt = t1 - t0

    if store is not None:
        store[label] = dt

    if verbose:
        print(f"[Timing] {label}: {dt:.3f} s")
