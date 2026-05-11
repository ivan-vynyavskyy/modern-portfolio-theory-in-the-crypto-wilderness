#!/usr/bin/env python3
"""
0_2_scan_traces_for_erc20_candidates.py

Purpose
-------
Scan a Delta Lake traces table to find ERC-20-like contracts created via
CREATE/CREATE2 and append candidates to a TSV file.

What it does
------------
- Enumerates active Parquet files from the current Delta snapshot.
- Filters successful CREATE/CREATE2 traces and orders them by recency.
- Uses bytecode selectors/event signatures to identify ERC-20 candidates.

Inputs
------
- traces (Delta table; Parquet files via _delta_log snapshot)
- erc20_candidates_scanner.tsv (optional; used for dedup on append)

Outputs
-------
- erc20_candidates_scanner.tsv (TSV; columns: token_address, block_id)

CLI
---
Examples:
  $ python 0_2_scan_traces_for_erc20_candidates.py --traces-dir traces --out-tsv erc20_candidates_scanner.tsv
  $ python 0_2_scan_traces_for_erc20_candidates.py --traces-dir traces --no-events

Notes
-----
- Delta snapshot semantics are preserved via deltalake.DeltaTable(...).files().
- Bytecode checks use the original helpers and per-row loop for exact behavior.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import polars as pl
from deltalake import DeltaTable

# Ensure helpers/ is importable when running from src/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.ethereum_codec_utils import (
    erc20_selectors_present,
    load_existing_addresses,
    to_hex_address,
    to_hex_bytes,
)
from utils.log_utils import log, log_exc

# ----------------------------- Constants -----------------------------
DEFAULT_TRACES_DIR = "traces"
DEFAULT_OUT_TSV = "erc20_candidates_scanner.tsv"

PARQUET_COLUMNS = [
    "block_id",
    "tx_hash",
    "transaction_index",
    "trace_index",
    "trace_type",
    "error",
    "status",
    "to_address",
    "output",
]

CREATE_TRACE_TYPES = ["create", "create2"]

def append_result(tsv_path: str, address: str, block_id: int) -> None:
    """Append one result line formatted as <address>\\t<block_id>."""
    os.makedirs(os.path.dirname(tsv_path) or ".", exist_ok=True)
    with open(tsv_path, "a", encoding="utf-8") as handle:
        handle.write(f"{address}\t{block_id}\n")
        handle.flush()


def delta_active_files(traces_dir: str, version: Optional[int] = None) -> Iterable[str]:
    """Yield absolute paths to active Parquet files in the Delta snapshot.

    If `version` is provided, the table is loaded at that specific commit
    version. This is useful when the `_delta_log/` chain has gaps above the
    latest checkpoint (e.g. archived commit JSONs): pin to the last
    contiguous version to bypass the broken tail of the log.
    """
    table = DeltaTable(traces_dir) if version is None else DeltaTable(traces_dir, version=version)
    base = Path(traces_dir)
    for rel in table.files():
        if str(rel).lower().endswith(".parquet"):
            yield str(base / rel)


def scan_parquet_file_polars(
    parquet_path: str,
    out_tsv: str,
    seen_addresses: Set[str],
    require_events: bool = True,
) -> Tuple[int, int]:
    """Scan one Parquet file and return (files_scanned=1, matches_found)."""
    try:
        lf = pl.scan_parquet(parquet_path)
    except Exception as exc:
        log_exc(f"[warn] Could not scan {parquet_path}: {exc}")
        return (1, 0)

    schema = lf.collect_schema()
    available_cols = set(schema.names())
    missing = set(PARQUET_COLUMNS) - available_cols
    if missing:
        log_exc(f"[warn] {parquet_path}: missing columns {missing}; skipping.")
        return (1, 0)

    trace_type_lower = pl.col("trace_type").cast(pl.Utf8).str.to_lowercase()
    cond = trace_type_lower.is_in(CREATE_TRACE_TYPES)

    if "status" in available_cols:
        cond &= (pl.col("status") == 1) | (pl.col("status") == True)

    if "error" in available_cols:
        err = pl.col("error")
        cond &= err.is_null() | (err == "")

    lf_filtered = lf.filter(cond).select(
        ["block_id", "transaction_index", "trace_index", "to_address", "output"]
    )

    try:
        df = lf_filtered.collect()
    except Exception as exc:
        log_exc(f"[warn] {parquet_path}: collect failed: {exc}")
        return (1, 0)

    if df.is_empty():
        return (1, 0)

    # Fill missing indices and sort to preserve original ordering semantics.
    df = (
        df.with_columns(
            [
                pl.col("transaction_index").fill_null(-1).cast(pl.Int64),
                pl.col("trace_index").fill_null(-1).cast(pl.Int64),
            ]
        )
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
        if addr_lc in seen_addresses:
            continue

        bytecode_hex = to_hex_bytes(row["output"])
        if not bytecode_hex:
            continue

        if not erc20_selectors_present(bytecode_hex, require_events=require_events):
            continue

        append_result(out_tsv, addr, block_id)
        seen_addresses.add(addr_lc)
        matches += 1

    return (1, matches)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Scan Delta traces for ERC-20-compatible contracts deployed via CREATE/CREATE2."
    )
    parser.add_argument(
        "--traces-dir",
        default=DEFAULT_TRACES_DIR,
        help="Root folder of the Delta table (must contain _delta_log/).",
    )
    parser.add_argument(
        "--out-tsv",
        default=DEFAULT_OUT_TSV,
        help="Where to append token_address and block_id.",
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Do not require Transfer/Approval event hashes in bytecode.",
    )
    parser.add_argument(
        "--delta-version",
        type=int,
        default=None,
        help=(
            "Pin the Delta table to this commit version (use the latest "
            "contiguous version when the _delta_log/ chain has gaps). "
            "Default: read the latest snapshot."
        ),
    )
    return parser


def main() -> int:
    """Run the CLI and return the process exit code."""
    args = _build_arg_parser().parse_args()

    traces_dir = Path(args.traces_dir)
    out_tsv = Path(args.out_tsv)
    require_events = not args.no_events

    log(f"traces Delta table root: {traces_dir}")
    log(f"writing results to: {out_tsv} (append mode)")
    log(f"requiring events: {require_events}")
    if args.delta_version is not None:
        log(f"pinned Delta version: {args.delta_version}")

    seen_addresses = load_existing_addresses(str(out_tsv))
    if seen_addresses:
        log(
            f"loaded {len(seen_addresses)} existing addresses from {out_tsv} (will skip duplicates)"
        )

    file_list = sorted(delta_active_files(str(traces_dir), version=args.delta_version))
    log(f"active data files in Delta snapshot: {len(file_list)}")

    files_scanned = 0
    total_matches = 0

    for i, parquet_path in enumerate(file_list, start=1):
        f_count, f_matches = scan_parquet_file_polars(
            parquet_path=parquet_path,
            out_tsv=str(out_tsv),
            seen_addresses=seen_addresses,
            require_events=require_events,
        )
        files_scanned += f_count
        total_matches += f_matches

        if f_matches:
            log(f"{i}/{len(file_list)} {parquet_path}: {f_matches} matches")
        elif i % 100 == 0:
            log(f"scanned {i}/{len(file_list)} files…")

    log(f"files scanned: {files_scanned}, new matches written: {total_matches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
