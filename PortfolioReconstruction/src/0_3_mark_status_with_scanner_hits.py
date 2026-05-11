#!/usr/bin/env python3
"""
0_3_mark_status_with_scanner_hits.py

Purpose
-------
Enrich a CoinGecko-derived token list with ERC-20 status based on a scanner
TSV of known contracts and their creation blocks.

What it does
------------
- Loads an address-to-creation_block index from a TSV/CSV/whitespace file.
- Normalizes contract addresses for reliable matching.
- Inserts is_erc_20 and creation_block columns after contract_address.

Inputs
------
- --input CSV (must include column: contract_address)
- --tsv (two columns: contract_address<TAB>creation_block; comma/whitespace accepted)

Outputs
-------
- --output CSV (same columns as input plus is_erc_20 and creation_block)

CLI
---
Examples:
  $ python 0_3_mark_status_with_scanner_hits.py --input ../data/derived_coingecko_eth_tokens.csv --tsv ../data/contract_addresses.tsv --output ../data/status_eth_tokens.csv

Notes
-----
- Duplicate addresses in the TSV are deduped by keeping the last occurrence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# Ensure helpers/ is importable when running from src/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.ethereum_codec_utils import normalize_hex_address
from utils.log_utils import log

# ----------------------------- Constants -----------------------------
TSV_COLUMNS = ["contract_address", "creation_block"]


def _read_erc20_index(tsv_path: Path) -> pd.DataFrame:
    """Load the ERC-20 index file with a tolerant delimiter strategy."""
    try:
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            header=None,
            names=TSV_COLUMNS,
            dtype=str,
            engine="python",
        )
        if df.shape[1] == 1:
            raise ValueError("Delimiter detection failed; fallback to whitespace/comma.")
        return df
    except Exception:
        return pd.read_csv(
            tsv_path,
            sep=r"[,\s]+",
            header=None,
            names=TSV_COLUMNS,
            dtype=str,
            engine="python",
        )


def _load_erc20_map(tsv_path: Path) -> Dict[str, str]:
    """Return a dict mapping normalized address to creation_block (string)."""
    df = _read_erc20_index(tsv_path)

    df["contract_address"] = df["contract_address"].map(normalize_hex_address)
    df = df[df["contract_address"] != ""].copy()
    df["creation_block"] = df["creation_block"].astype(str).str.strip()

    if df["contract_address"].duplicated(keep=False).any():
        dup_count = int(df["contract_address"].duplicated(keep=False).sum())
        log(
            f"Note: duplicate addresses detected in {tsv_path.name} (deduped {dup_count} rows); "
            f"keeping the last occurrence per address."
        )
        df = df.drop_duplicates(subset=["contract_address"], keep="last")

    return dict(zip(df["contract_address"], df["creation_block"]))


def update_erc20_columns(input_csv: Path, tsv_path: Path) -> pd.DataFrame:
    """Return a new DataFrame with is_erc_20 and creation_block inserted."""
    df = pd.read_csv(input_csv, dtype=str)
    if "contract_address" not in df.columns:
        raise ValueError("Input CSV must contain a 'contract_address' column.")

    addr_key = df["contract_address"].map(normalize_hex_address)
    erc20_map = _load_erc20_map(tsv_path)

    creation_block = addr_key.map(lambda addr: erc20_map.get(addr, ""))
    is_erc_20 = creation_block.map(lambda block: "true" if block != "" else "false")

    for col in ("is_erc_20", "creation_block"):
        if col in df.columns:
            df = df.drop(columns=[col])

    insert_at = df.columns.get_loc("contract_address") + 1
    df.insert(insert_at, "is_erc_20", is_erc_20)
    df.insert(insert_at + 1, "creation_block", creation_block)

    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Add is_erc_20 and creation_block columns to eth_token_status.csv "
            "based on addresses present in erc20_candidates_scanner.tsv"
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to eth_token_status.csv",
    )
    parser.add_argument(
        "--tsv",
        required=True,
        type=Path,
        help="Path to erc20_candidates_scanner.tsv (addr<TAB>block)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path",
    )
    return parser


def main() -> int:
    """Run the CLI and return the process exit code."""
    args = _build_arg_parser().parse_args()

    updated = update_erc20_columns(args.input, args.tsv)
    updated.to_csv(args.output, index=False)

    total = len(updated)
    matched = int((updated["is_erc_20"] == "true").sum())
    log(
        f"Wrote {args.output} ({total} rows). "
        f"Matched {matched} ERC-20 addresses; {total - matched} not found."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
