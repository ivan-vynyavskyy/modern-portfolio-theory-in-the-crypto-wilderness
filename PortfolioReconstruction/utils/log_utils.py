"""
log_utils.py

Purpose
-------
Lightweight timestamped logging helpers shared across pipeline scripts.

Notes
-----
- All output is flushed immediately for visibility in long-running jobs.
"""

import sys
from datetime import datetime


def log(msg: str, *, end: str = "\n") -> None:
    """Write a timestamped line to stdout."""
    print(f"{datetime.now():%H:%M:%S} | {msg}", end=end, flush=True)


def log_exc(msg: str) -> None:
    """Write a timestamped error line to stderr."""
    print(f"{datetime.now():%H:%M:%S} | ERROR: {msg}", file=sys.stderr, flush=True)
