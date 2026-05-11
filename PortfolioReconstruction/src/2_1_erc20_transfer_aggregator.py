#!/usr/bin/env python3
"""
2_1_erc20_transfer_aggregator.py

Purpose
-------
Merge per-token ERC-20 transfer shards into a single ordered ledger per token.

What it does
------------
- Discovers shard Parquet files under token_address=* directories.
- Builds debit and credit rows per transfer and sorts them by address, block, log index.
- Writes one ledger Parquet per token to the output directory.

Inputs
------
- base_dir (directory; expects token_address=*/**/*.parquet with columns: sender, recipient, value, block, log_index)
- --token-status-csv (CSV; columns include contract_address,is_erc_20,price_data_status,trading_volume_status,fdv_mcap_lt_eth)
- DATA (env var; default output base directory when --output is not provided)
- SLURM_CPUS_PER_TASK (env var; optional worker-count override)

Outputs
-------
- <output>/token_address=<token>/<token>.parquet (Parquet; columns: address, block, log_index, value)

CLI
---
Examples:
  $ python 2_1_erc20_transfer_aggregator.py /data/shards --token-status-csv token_status.csv
  $ python 2_1_erc20_transfer_aggregator.py /data/shards --tokens 0xabc... 0xdef... --output /data/output

Notes
-----
- Token discovery uses directory names like token_address=<lowercase_0x...>.
- The allow-list keeps ERC-20 tokens with successful price data, good volume, and fdv_mcap_lt_eth != false.
- Output rows are sorted by address, then block, then log_index.
"""

from __future__ import annotations

# ----------------------------- Standard library -----------------------------
import argparse
import decimal
import multiprocessing as mp
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# ------------------------------ Third-party ---------------------------------
import pyarrow as pa  # pylint: disable=import-error
import pyarrow.parquet as pq  # pylint: disable=import-error
import psutil  # pylint: disable=import-error

# Ensure helpers/ is importable when running from src/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from helpers.token_filters import load_token_allow_list
from utils.log_utils import log

# ----------------------------- Constants ------------------------------------
decimal.getcontext().prec = 80  # lossless uint256 math

TOKEN_WHITELIST: Set[str] = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
}

READ_COLS: Sequence[str] = ["sender", "recipient", "value", "block", "log_index"]

HEX_RE = re.compile(r"^0x[0-9a-f]{40}$")


# ----------------------- Arrow value column helpers -------------------------
def _as_large_string(arr: pa.Array) -> pa.Array:
    """Return arr as pa.large_string() if it is string-like."""
    if pa.types.is_large_string(arr.type):
        return arr
    if pa.types.is_string(arr.type):
        if hasattr(pa, "compute"):
            try:
                return pa.compute.cast(arr, pa.large_string())  # type: ignore
            except Exception:
                pass
        return pa.array(arr.to_pylist(), pa.large_string())
    return arr


def _make_value_columns(arr: pa.Array) -> Tuple[pa.Array, pa.Array]:
    """Return (neg_arr, pos_arr) with identical Arrow types."""
    try:
        # Fast numeric path.
        neg_py = [(-x if x is not None else None) for x in arr.to_pylist()]
        neg_arr = pa.array(neg_py, type=arr.type, safe=False)
        return neg_arr, arr
    except Exception:
        # Fallback for string-like values.
        pos_py = [str(x) if x is not None else None for x in arr.to_pylist()]
        neg_py = [("-" + s.lstrip("-")) if s else None for s in pos_py]
        pos_arr = pa.array(pos_py, pa.large_string())
        neg_arr = pa.array(neg_py, pa.large_string())
        return neg_arr, pos_arr


# ----------------------------- Per-token worker -----------------------------
def _process_token_impl(token: str, parquet_paths: List[Path], output_dir: Path) -> None:
    """Merge shards for a token and write the ordered ledger parquet."""
    proc = mp.current_process().name
    t0 = time.time()
    token = token.strip().lower()
    log(f"[{proc:>10}] start {token} - {len(parquet_paths):,} shard(s)")

    tables: List[pa.Table] = []
    raw_rows = 0

    for p in parquet_paths:
        try:
            pf = pq.ParquetFile(p, memory_map=True)
        except Exception as exc:
            log(f"warn: cannot open {p}: {exc}")
            continue
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=READ_COLS, use_threads=True)
            raw_rows += tbl.num_rows
            tables.append(tbl)

    if not tables:
        return

    merged = pa.concat_tables(tables, promote_options="default")

    value = merged["value"]
    block = merged["block"]
    log_index = merged["log_index"]

    neg_value, pos_value = _make_value_columns(value)

    sender_arr = _as_large_string(merged["sender"])
    recipient_arr = _as_large_string(merged["recipient"])

    debits = pa.table(
        {
            "address": sender_arr,
            "block": block,
            "log_index": log_index,
            "value": neg_value,
        }
    )
    credits = pa.table(
        {
            "address": recipient_arr,
            "block": block,
            "log_index": log_index,
            "value": pos_value,
        }
    )

    ledger = pa.concat_tables([debits, credits]).sort_by(
        [
            ("address", "ascending"),
            ("block", "ascending"),
            ("log_index", "ascending"),
        ]
    )

    token_dir = output_dir / f"token_address={token}"
    token_dir.mkdir(parents=True, exist_ok=True)
    out_path = token_dir / f"{token}.parquet"

    pq.write_table(
        ledger,
        out_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=False,
    )

    log(
        f"[{proc:>10}] done {token} raw={raw_rows:,} "
        f"-> out={ledger.num_rows:,} rows | {time.time() - t0:.1f}s"
    )


def process_token(token: str, parquet_paths: List[Path], output_dir: Path) -> None:
    """Run token processing with a contextual error message on failure."""
    try:
        _process_token_impl(token, parquet_paths, output_dir)
    except Exception as exc:
        raise RuntimeError(f"token {token}: {exc}") from exc


# ----------------------------- Discovery helper -----------------------------
def _discover(base: Path, whitelist: Optional[Set[str]] = None) -> Dict[str, List[Path]]:
    """Return mapping of token -> list of parquet shard paths."""
    wl = {t.lower() for t in whitelist} if whitelist else None
    result: Dict[str, List[Path]] = defaultdict(list)

    for tok_dir in base.glob("token_address=*"):
        if not tok_dir.is_dir():
            continue
        token = tok_dir.name.split("=", 1)[1].lower()
        if wl and token not in wl:
            continue
        result[token].extend(tok_dir.rglob("*.parquet"))

    log(
        f"discovered {sum(len(v) for v in result.values()):,} parquet files "
        f"for {len(result):,} token(s)"
    )
    if wl and (missing := wl.difference(result)):
        log("warn: whitelisted tokens not found: " + ", ".join(sorted(missing)))
    return result


# ----------------------------- Worker sizing --------------------------------
def _pool_size() -> int:
    """Return a worker count based on CPU availability and load."""
    total = int(os.getenv("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    busy = psutil.cpu_stats().ctx_switches > 1_000_000
    n = max(1, total // 2) if busy else total
    log(f"using {n} worker processes (CPUs={total}{', busy' if busy else ''})")
    return n


# ----------------------------- CLI / main -----------------------------------

def _resolve_output_dir(output_arg: Optional[Path]) -> Path:
    """Return the output directory path, creating it if needed."""
    out_dir = output_arg.expanduser().resolve() if output_arg else None
    if out_dir is None:
        out_dir = Path(os.getenv("DATA", str(Path.home()))) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _resolve_whitelist(args: argparse.Namespace) -> Tuple[Optional[Set[str]], str]:
    """Return (whitelist, mode) based on CLI arguments."""
    if args.tokens:
        return {t.lower() for t in args.tokens}, "CLI whitelist"

    if args.token_status_csv:
        csv_path = args.token_status_csv.expanduser().resolve()
        if not csv_path.is_file():
            raise SystemExit(f"x {csv_path} is not a file")
        allow = load_token_allow_list(str(csv_path), log_fn=log)
        return allow, f"token-status CSV: {csv_path.name}"

    if args.all_tokens:
        return None, "ALL tokens"

    return TOKEN_WHITELIST, "default whitelist"

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    ap = argparse.ArgumentParser(
        prog="erc20_transfer_aggregator",
        description="Merge ERC-20 transfer shards into one ledger Parquet per token",
    )
    ap.add_argument(
        "base_dir",
        type=Path,
        help="root directory with token_address=*/shard-*.parquet",
    )

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--tokens", nargs="+", help="whitelist of token contracts to process")
    g.add_argument(
        "--token-status-csv",
        type=Path,
        help=(
            "CSV with columns like: symbol,name,contract_address,is_erc_20,creation_block,"
            "price_data_status,trading_volume_status,fdv_mcap_lt_eth (uses allow-list filters)"
        ),
    )
    g.add_argument(
        "--all-tokens",
        action="store_true",
        help="process every token found (ignore whitelist)",
    )

    ap.add_argument(
        "--output",
        type=Path,
        help="output directory (default $DATA/output)",
    )
    ap.add_argument(
        "-j",
        "--workers",
        type=int,
        help="number of worker processes (default = heuristic)",
    )
    return ap.parse_args()

def main() -> int:
    """CLI entrypoint."""
    args = _parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.is_dir():
        raise SystemExit(f"x {base_dir} is not a directory")

    whitelist, mode = _resolve_whitelist(args)
    log(f"discovery mode: {mode}")

    token_map = _discover(base_dir, whitelist)
    if not token_map:
        raise SystemExit("x no matching parquet files found")

    out_dir = _resolve_output_dir(args.output)
    log(f"output directory: {out_dir}")

    ctx = mp.get_context("fork")
    n_workers = args.workers if (args.workers or 0) > 0 else _pool_size()
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(process_token, tok, paths, out_dir): tok for tok, paths in token_map.items()}
        for fut in as_completed(futures):
            tok = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                log(f"error: worker failed for {tok}: {exc}")
                raise

    log(f"\nALL DONE in {timedelta(seconds=time.time() - t0)}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
