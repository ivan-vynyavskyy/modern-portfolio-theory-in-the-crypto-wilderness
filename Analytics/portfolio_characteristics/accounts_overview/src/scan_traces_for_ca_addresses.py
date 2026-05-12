#!/usr/bin/env python3
# =============================================================================
# Script: scan_traces_for_ca_addresses.py
#
# Updated semantics:
#   - Scan Delta traces and output EVERY successful CREATE / CREATE2 deployment.
#   - No ERC-20 bytecode filtering (no REQUIRED_METHOD_SELECTORS / REQUIRED_EVENT_PREFIXES).
#
# Output TSV format:
#   <address>\t<block_id>
# =============================================================================
import os
import sys
import argparse
from pathlib import Path
from typing import Set, Iterable, Tuple

import polars as pl
from deltalake import DeltaTable

# Import shared utilities from Analytics/shared/.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))

from ethereum_codec_utils import (
    load_existing_addresses,
    to_hex_address,
)

# ---------- Configuration ----------
PARQUET_COLUMNS = [
    "block_id", "tx_hash", "transaction_index",
    "trace_index", "trace_type", "error", "status",
    "to_address",
]
# -----------------------------------------------------------------------


def append_result(tsv_path: str, address: str, block_id: int) -> None:
    """Write exactly: <address>\t<block_id>"""
    os.makedirs(os.path.dirname(tsv_path) or ".", exist_ok=True)
    with open(tsv_path, "a", encoding="utf-8") as f:
        f.write(f"{address}\t{block_id}\n")
        f.flush()


def delta_active_files(traces_dir: str) -> Iterable[str]:
    """
    Yield absolute paths to active Parquet files for the current Delta snapshot.
    """
    table = DeltaTable(traces_dir)
    base = Path(traces_dir)
    for rel in table.files():
        if str(rel).lower().endswith(".parquet"):
            yield str(base / rel)


def scan_parquet_file_polars(
    parquet_path: str,
    out_tsv: str,
    seen_addresses: Set[str],
) -> Tuple[int, int]:
    """
    Scan a single Parquet file with Polars.
    Returns (files_scanned=1, matches_found_in_this_file).
    """
    try:
        lf = pl.scan_parquet(parquet_path)
    except Exception as e:
        print(f"[warn] Could not scan {parquet_path}: {e}", file=sys.stderr)
        return (1, 0)

    # Check required columns exist
    schema = lf.collect_schema()
    available_cols = set(schema.names())
    missing = set(PARQUET_COLUMNS) - available_cols
    if missing:
        print(f"[warn] {parquet_path}: missing columns {missing}; skipping.", file=sys.stderr)
        return (1, 0)

    # Filter to successful CREATE/CREATE2 traces
    trace_type_lower = pl.col("trace_type").cast(pl.Utf8).str.to_lowercase()
    cond = trace_type_lower.is_in(["create", "create2"])

    if "status" in available_cols:
        cond &= (pl.col("status") == 1) | (pl.col("status") == True)

    if "error" in available_cols:
        err = pl.col("error")
        cond &= err.is_null() | (err == "")

    lf_filtered = (
        lf.filter(cond)
          .select(["block_id", "transaction_index", "trace_index", "to_address"])
    )

    try:
        df = lf_filtered.collect()
    except Exception as e:
        print(f"[warn] {parquet_path}: collect failed: {e}", file=sys.stderr)
        return (1, 0)

    if df.is_empty():
        return (1, 0)

    # Sort deterministically: block_id / tx_index / trace_index descending
    df = (
        df.with_columns([
            pl.col("transaction_index").fill_null(-1).cast(pl.Int64),
            pl.col("trace_index").fill_null(-1).cast(pl.Int64),
        ])
        .sort(
            by=["block_id", "transaction_index", "trace_index"],
            descending=[True, True, True],
        )
    )

    matches = 0

    for row in df.iter_rows(named=True):
        block_id = int(row["block_id"])
        addr = to_hex_address(row["to_address"])
        if not addr:
            continue

        addr_lc = addr.lower()

        # Deduplicate across all files
        if addr_lc in seen_addresses:
            continue

        append_result(out_tsv, addr, block_id)
        seen_addresses.add(addr_lc)
        matches += 1

    return (1, matches)


def main():
    ap = argparse.ArgumentParser(
        description="Scan Delta traces and extract ALL successful CREATE/CREATE2 deployed contract addresses."
    )
    ap.add_argument(
        "--traces-dir",
        default="traces",
        help="Root folder of the Delta table (must contain _delta_log/).",
    )
    ap.add_argument(
        "--out-tsv",
        default="contract_accounts_created.tsv",
        help="Where to append created contract addresses and block_id.",
    )
    args = ap.parse_args()

    print(f"[info] traces Delta table root: {args.traces_dir}")
    print(f"[info] writing results to: {args.out_tsv} (append mode)")

    seen_addresses = load_existing_addresses(args.out_tsv)
    if seen_addresses:
        print(f"[info] loaded {len(seen_addresses)} existing addresses from {args.out_tsv} (will skip duplicates)")

    # Delta snapshot file list; sort for deterministic processing order
    file_list = sorted(delta_active_files(args.traces_dir))
    print(f"[info] active data files in Delta snapshot: {len(file_list)}")

    files_scanned = 0
    total_matches = 0

    for i, parquet_path in enumerate(file_list, start=1):
        f_count, f_matches = scan_parquet_file_polars(
            parquet_path=parquet_path,
            out_tsv=args.out_tsv,
            seen_addresses=seen_addresses,
        )
        files_scanned += f_count
        total_matches += f_matches

        if f_matches:
            print(f"[ok] {i}/{len(file_list)} {parquet_path}: {f_matches} new contracts")
        elif i % 100 == 0:
            print(f"[..] scanned {i}/{len(file_list)} files…")

    print(f"[done] files scanned: {files_scanned}, new contracts written: {total_matches}")


if __name__ == "__main__":
    main()
