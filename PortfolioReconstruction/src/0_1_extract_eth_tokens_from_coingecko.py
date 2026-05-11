#!/usr/bin/env python3
"""
0_1_extract_eth_tokens_from_coingecko.py

Purpose
-------
Extract Ethereum contract addresses from a CoinGecko coins export and save a
clean token list for downstream analysis.

What it does
------------
- Reads the CoinGecko coins CSV and selects core columns.
- Parses the mixed-format "platforms" field for Ethereum contracts.
- Writes a filtered CSV of tokens with non-empty contract addresses.

Inputs
------
- ../data/all_coins_coingecko.csv (CSV; columns: symbol, name, platforms)

Outputs
-------
- ../data/derived_coingecko_eth_tokens.csv (CSV; columns: symbol, name, contract_address)

CLI
---
Examples:
  $ python 0_1_extract_eth_tokens_from_coingecko.py --input ../data/all_coins_coingecko.csv --output ../data/derived_coingecko_eth_tokens.csv
  $ python 0_1_extract_eth_tokens_from_coingecko.py --input ../data/all_coins_coingecko.csv

Notes
-----
- The "platforms" column may be a dict or a stringified dict; JSON parsing is
  attempted before falling back to ast.literal_eval.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Ensure helpers/ is importable when running from src/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.log_utils import log

# ----------------------------- Constants -----------------------------
DEFAULT_INPUT_CSV = "../data/all_coins_coingecko.csv"
DEFAULT_OUTPUT_CSV = "../data/derived_coingecko_eth_tokens.csv"
INPUT_COLUMNS = ["symbol", "name", "platforms"]
INPUT_DTYPES = {"symbol": "string", "name": "string", "platforms": "string"}


def _parse_platforms_to_dict(value: str) -> Optional[dict]:
    """Parse a stringified platforms value into a dict when possible."""
    # Most rows look like "{'ethereum': '0xABCD...', 'solana': ...}".
    # Try JSON parsing first (swap single→double quotes), then fall back to ast.literal_eval.
    try:
        parsed = json.loads(value.replace("'", '"'))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def extract_eth_contract(value: Any) -> Optional[str]:
    """Return the Ethereum contract address (or None) from the platforms column."""
    if pd.isna(value):
        return None
    if isinstance(value, dict):
        return value.get("ethereum")
    if isinstance(value, str):
        parsed = _parse_platforms_to_dict(value)
        return parsed.get("ethereum") if parsed else None
    return None


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract Ethereum contract addresses from CoinGecko CSV."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT_CSV,
        help=f"Path to input CoinGecko CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path to output CSV (default: {DEFAULT_OUTPUT_CSV})",
    )
    return parser


def main() -> int:
    """Run the CLI and return the process exit code."""
    args = _build_arg_parser().parse_args()

    input_csv = Path(args.input)
    output_csv = Path(args.output)

    df = pd.read_csv(
        str(input_csv),
        usecols=INPUT_COLUMNS,
        dtype=INPUT_DTYPES,
        low_memory=False,
    )

    df["contract_address"] = df["platforms"].apply(extract_eth_contract)

    out = (
        df[df["contract_address"].notna() & (df["contract_address"] != "")]
        [["symbol", "name", "contract_address"]]
    )

    out.to_csv(str(output_csv), index=False)

    log(f"Saved {len(out)} Ethereum-based tokens with contracts to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
