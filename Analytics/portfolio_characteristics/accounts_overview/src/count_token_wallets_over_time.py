#!/usr/bin/env python3
"""
count_token_wallets_over_time.py

Description:
    Per asset and snapshot, the cumulative number of wallets that have
    (a) appeared in the asset's ledger and (b) acquired the asset (a credit
    row with value > 0) at or before the snapshot block.

    Both counts come from one pass per asset: the first-seen and
    first-acquired block of every address is histogrammed and summed
    cumulatively at the snapshot blocks. The zero address (mint/burn
    counterparty, not an account) is excluded, as in the two sibling
    scripts. The native-ETH ledger (pseudo-token
    0xee..ee) is by far the largest partition and is NOT scanned unless
    --include-eth is given; it then appears as one more asset row set.

Input:
    --data-root      structured transfers dump: token_address=<addr>/*.parquet
                     with columns address, block, log_index, value (signed)
    --status-csv     token status CSV (whitelist; see shared/token_whitelist.py)
    --snapshots-csv  snapshot list, normally the 4_2 _manifest.csv;
                     --block-to-date-csv resolves or cross-checks dates

Output (in --out-dir, default ../data; same columns as the notebook wrote):
    output_wallet_counts.csv            token_address, block, date, wallets
                                        (wallets ever seen in the ledger)
    output_wallet_counts_acquired.csv   token_address, block, date, wallets
                                        (wallets that ever received the asset)
    --resume skips assets already present in both files.

Usage (cluster):
    python count_token_wallets_over_time.py \\
        --data-root /path/to/eth-erc20-transfers-20251208-structured \\
        --status-csv /path/to/data/status_eth_tokens_3_1.csv \\
        --snapshots-csv /path/to/recon-eth-erc20-partitioned/_manifest.csv \\
        --threads 32 --mem 60GB --temp-dir /path/to/duckdb_tmp --resume
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Set

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))

from snapshots import cumulative_counts, find_token_parquet_glob, load_snapshots  # noqa: E402
from token_whitelist import ETH_PSEUDO_TOKEN, ZERO_ADDRESS, load_token_whitelist  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_SEEN = "output_wallet_counts.csv"
OUT_ACQUIRED = "output_wallet_counts_acquired.csv"
HEADER = ["token_address", "block", "date", "wallets"]


def sql_quote(p: str) -> str:
    return "'" + str(p).replace("'", "''") + "'"


def tokens_done(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path, usecols=["token_address"], dtype=str)["token_address"].unique())
    except Exception:
        return set()


def first_blocks(con: duckdb.DuckDBPyConnection, parquet_glob: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Histograms (block, n) of first-seen and first-acquired blocks."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fb AS
        SELECT lower(trim(address)) AS addr,
               MIN(block) AS first_seen,
               MIN(CASE WHEN TRY_CAST(value AS DECIMAL(38, 0)) > 0 THEN block END) AS first_acq
        FROM read_parquet({sql_quote(parquet_glob)})
        WHERE address IS NOT NULL AND address <> ''
          AND lower(trim(address)) <> '{ZERO_ADDRESS}'
        GROUP BY 1
    """)
    seen = con.execute("SELECT first_seen AS block, COUNT(*) AS n FROM fb GROUP BY 1").df()
    acq = con.execute("SELECT first_acq AS block, COUNT(*) AS n FROM fb WHERE first_acq IS NOT NULL GROUP BY 1").df()
    return seen, acq


def append_rows(path: Path, token: str, counts: pd.Series, date_map: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(HEADER)
        for b, n in counts.items():
            w.writerow([token, int(b), date_map[int(b)], int(n)])


def main() -> int:
    ap = argparse.ArgumentParser(description="Cumulative wallets per asset and snapshot (seen / acquired).")
    ap.add_argument("--data-root")
    ap.add_argument("--status-csv")
    ap.add_argument("--snapshots-csv")
    ap.add_argument("--block-to-date-csv", default=None)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--mem", default="32GB")
    ap.add_argument("--temp-dir", default=None)
    ap.add_argument("--include-eth", action="store_true",
                    help="Also scan the native-ETH ledger (one more asset row set; slow).")
    ap.add_argument("--resume", action="store_true", help="Skip assets already present in both output files.")
    ap.add_argument("--max-tokens", type=int, default=None, help="Only the first N assets (smoke test).")
    args = ap.parse_args()

    for label, path in (("--data-root", args.data_root), ("--status-csv", args.status_csv),
                        ("--snapshots-csv", args.snapshots_csv), ("--block-to-date-csv", args.block_to_date_csv)):
        if path and not Path(path).exists():
            raise SystemExit(f"[preflight] {label}: {path} does not exist")

    snaps = load_snapshots(args.snapshots_csv, args.block_to_date_csv)
    target_blocks: List[int] = snaps["block"].tolist()
    date_map = dict(zip(snaps["block"], snaps["date"]))

    whitelist = sorted(load_token_whitelist(args.status_csv, include_eth=args.include_eth))
    if args.max_tokens:
        whitelist = whitelist[: args.max_tokens]
    print(f"[config] {len(whitelist):,} assets | native ETH {'INCLUDED' if args.include_eth else 'excluded'}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_seen, p_acq = out_dir / OUT_SEEN, out_dir / OUT_ACQUIRED
    if args.resume:
        done = tokens_done(p_seen) & tokens_done(p_acq)
        print(f"[resume] {len(done)} asset(s) already in the outputs")
    else:
        done = set()
        p_seen.unlink(missing_ok=True)
        p_acq.unlink(missing_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads={int(args.threads)}")
    con.execute(f"SET memory_limit='{args.mem}'")
    con.execute("SET preserve_insertion_order=false")
    if args.temp_dir:
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory={sql_quote(args.temp_dir)}")

    n_ok = n_skip = n_err = 0
    t0 = time.time()
    for i, token in enumerate(whitelist, start=1):
        if token in done:
            continue
        g = find_token_parquet_glob(args.data_root, token)
        if not g:
            n_skip += 1
            if token == ETH_PSEUDO_TOKEN:
                print("[warn] native-ETH partition 0xee..ee not found under the data root")
            continue
        try:
            seen, acq = first_blocks(con, g)
        except Exception as e:  # keep going; the asset can be redone with --resume
            n_err += 1
            print(f"[err] {token}: {e}")
            continue
        append_rows(p_seen, token, cumulative_counts(seen, target_blocks), date_map)
        append_rows(p_acq, token, cumulative_counts(acq, target_blocks), date_map)
        n_ok += 1
        if token == ETH_PSEUDO_TOKEN or n_ok % 100 == 0:
            print(f"[{i}/{len(whitelist)}] {n_ok} assets done, {time.time() - t0:,.0f}s"
                  f"{' (native ETH)' if token == ETH_PSEUDO_TOKEN else ''}", flush=True)

    con.close()
    print(f"[done] {n_ok} assets written now, {len(done)} from earlier runs, "
          f"{n_skip} without parquet, {n_err} errors")
    print(f"  - {p_seen}\n  - {p_acq}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
