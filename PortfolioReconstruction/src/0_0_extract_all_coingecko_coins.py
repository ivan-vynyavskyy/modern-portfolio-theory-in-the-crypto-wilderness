#!/usr/bin/env python3
"""
0_0_extract_all_coingecko_coins.py

Purpose
-------
Download the full CoinGecko coin list (including platform metadata) to a local CSV.

What it does
------------
- Calls the CoinGecko Pro coins list endpoint with platform metadata enabled.
- Expands the nested platform mapping into one column per chain.
- Writes the resulting table to a CSV (optionally gzipped).

Inputs
------
- CoinGecko Pro API key via --api-key.
- CoinGecko Pro /coins/list?include_platform=true endpoint (JSON array).

Outputs
-------
- data/all_coins_coingecko.csv.gz (CSV; columns: id, symbol, name, platforms + per-chain columns)

CLI
---
Examples:
  $ python 0_0_extract_all_coingecko_coins.py --api-key "$COINGECKO_KEY"
  $ python 0_0_extract_all_coingecko_coins.py --api-key "$COINGECKO_KEY" --output /tmp/all_coins.csv.gz

Notes
-----
- The CoinGecko response includes a nested "platforms" dict; this script expands it into columns.
- Empty platform addresses are stored as missing values.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from requests import Session

# Ensure helpers/ is importable when running from src/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.http_utils import build_retry_session
from utils.log_utils import log

# ----------------------------- Constants -----------------------------
COINGECKO_URL = "https://pro-api.coingecko.com/api/v3/coins/list?include_platform=true"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "all_coins_coingecko.csv.gz"


def fetch_coin_list(session: Session, api_key: str, timeout: int) -> Iterable[dict[str, Any]]:
    """Fetch the raw CoinGecko coins list payload."""
    headers = {
        "accept": "application/json",
        "x-cg-pro-api-key": api_key,
    }
    response = session.get(COINGECKO_URL, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def expand_platform_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the nested platforms mapping into per-chain columns."""
    if "platforms" not in df.columns:
        return df

    chains_to_create: list[str] = []
    for _, platforms in df["platforms"].items():
        if not isinstance(platforms, dict) or len(platforms) == 0:
            continue
        for chain in platforms.keys():
            if chain not in df.columns and chain not in chains_to_create:
                chains_to_create.append(chain)

    if chains_to_create:
        new_columns = pd.DataFrame({chain: pd.NA for chain in chains_to_create}, index=df.index)
        df = pd.concat([df, new_columns], axis=1)

    for idx, platforms in df["platforms"].items():
        if not isinstance(platforms, dict) or len(platforms) == 0:
            continue
        for chain, address in platforms.items():
            df.at[idx, chain] = address if address else pd.NA
    return df


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Write the dataframe to CSV at the requested location."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, compression="infer")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Download the CoinGecko coins list and save it to a CSV (optionally gzipped)."
    )
    parser.add_argument("--api-key", required=True, help="CoinGecko Pro API key.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to store the CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP request timeout in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries for transient HTTP errors (default: %(default)s).",
    )
    parser.add_argument(
        "--backoff-factor",
        type=float,
        default=1.0,
        help="Backoff factor between retries (default: %(default)s).",
    )
    return parser

def main() -> int:
    """Run the CLI entrypoint and return an exit code."""
    args = _build_arg_parser().parse_args()
    session = build_retry_session(
        retries=args.retries,
        backoff_factor=args.backoff_factor,
        status_forcelist=(500, 502, 503, 504),
    )
    coins_payload = fetch_coin_list(session=session, api_key=args.api_key, timeout=args.timeout)
    df = pd.DataFrame(coins_payload)
    df = expand_platform_columns(df)
    save_dataframe(df, args.output)
    log(f"Saved {len(df):,} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
