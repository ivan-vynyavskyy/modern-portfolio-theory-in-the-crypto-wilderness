"""
http_utils.py

Purpose
-------
Reusable HTTP session factory with configurable retry and backoff.

What it does
------------
- Builds a requests.Session with retry, exponential backoff, and connection pooling.

Notes
-----
- The session mounts retry handling only on HTTPS.
- Default status_forcelist includes 429 (rate-limit); callers can override.
"""

from typing import Optional, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_retry_session(
    retries: int = 5,
    backoff_factor: float = 0.6,
    status_forcelist: Sequence[int] = (429, 500, 502, 503, 504),
    pool_connections: int = 20,
    pool_maxsize: int = 20,
    headers: Optional[dict] = None,
) -> requests.Session:
    """Create a requests session with retry handling for transient errors."""
    session = requests.Session()
    retry_config = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=tuple(status_forcelist),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry_config,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )
    session.mount("https://", adapter)
    if headers:
        session.headers.update(headers)
    return session
