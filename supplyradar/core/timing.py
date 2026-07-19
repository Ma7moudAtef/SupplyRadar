"""Lightweight wall-clock timing harness for performance work.

Usage:
    from supplyradar.core.timing import stage

    with stage("build_stock_projection"):
        ...

Timing is OFF by default and costs nothing. Export SUPPLYRADAR_TIMING=1 to
turn it on; each stage then logs its wall-clock duration to stderr as
    [supplyradar-timing] <name>: <ms> ms
so before/after numbers in a performance pass are measured, never guessed.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

# Read once at import: the harness must add zero overhead when disabled.
ENABLED = os.environ.get("SUPPLYRADAR_TIMING", "") not in ("", "0", "false")

@contextmanager
def stage(name: str):
    """Context manager that logs the wall-clock time of the enclosed block
    (only when SUPPLYRADAR_TIMING is set — otherwise a no-op)."""
    if not ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[supplyradar-timing] {name}: {dt_ms:.1f} ms", file=sys.stderr)
