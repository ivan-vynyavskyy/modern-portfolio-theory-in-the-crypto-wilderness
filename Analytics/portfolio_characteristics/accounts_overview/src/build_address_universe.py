#!/usr/bin/env python3
"""
build_address_universe.py

Description:
    Builds the address universe of the study and counts the ledger rows.

    - Writes the distinct addresses appearing in the ledgers of all
      whitelisted assets to a single-column Parquet file (column `address`).
      This is the --input-wallets-parquet of
      4_0_portfolio_reconstruction_pipeline.py, i.e. the set of accounts for
      which portfolios are reconstructed: every address that ever moved a
      whitelisted ERC-20 token. The native-ETH ledger (pseudo-token
      0xee..ee) is NOT scanned unless --include-eth is given; ETH-only
      addresses cannot form a multi-asset portfolio and stay outside the
      universe (their ETH is still priced for the addresses that are in).
    - Counts the rows of those ledgers. Each transfer is stored as a debit
      and a credit row, so transfers = rows / 2 (the ReconTransferEvents
      figure of the paper's computational appendix).

    The zero address is dropped from the universe by default: it is the
    mint/burn counterparty of ERC-20 ledgers, not an account
    (--keep-zero-address keeps it).

Input:
    --data-root     structured transfers dump: token_address=<addr>/*.parquet
    --status-csv    token status CSV (whitelist; see shared/token_whitelist.py)

Output:
    --out-parquet   distinct addresses (default: Location.WHITELIST_ADDRESSES_PARQUET)
    --stats-csv     optional one-row CSV: assets, rows, transfers, unique
                    addresses (total / in token ledgers / only in the ETH ledger)

Usage (cluster):
    python build_address_universe.py \\
        --data-root /path/to/eth-erc20-transfers-20251208-structured \\
        --status-csv /path/to/data/eth_tokens_status_new_new_sorted.csv \\
        --out-parquet /path/to/data/whitelist_unique_addresses.parquet \\
        --stats-csv ../data/address_universe_stats.csv \\
        --threads 32 --mem 100GB --temp-dir /path/to/duckdb_tmp
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from locations import Location  # noqa: E402
from snapshots import find_token_parquet_glob  # noqa: E402
from token_whitelist import ETH_PSEUDO_TOKEN, ZERO_ADDRESS, load_token_whitelist  # noqa: E402


def sql_quote(p: str) -> str:
    return "'" + str(p).replace("'", "''") + "'"


def main() -> int:
    ap = argparse.ArgumentParser(description="Distinct addresses and row counts over the whitelisted ledgers.")
    ap.add_argument("--data-root", default=str(Location.ETH_TRANSFERS_DATA))
    ap.add_argument("--status-csv", default=str(Location.TOKEN_STATUS_CSV))
    ap.add_argument("--out-parquet", default=str(Location.WHITELIST_ADDRESSES_PARQUET))
    ap.add_argument("--stats-csv", default=None)
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--mem", default="32GB")
    ap.add_argument("--temp-dir", default=None)
    ap.add_argument("--include-eth", action="store_true",
                    help="Also scan the native-ETH ledger (ETH-only addresses enter the universe).")
    ap.add_argument("--keep-zero-address", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None, help="Only the first N assets (smoke test).")
    args = ap.parse_args()

    for label, path in (("--data-root", args.data_root), ("--status-csv", args.status_csv)):
        if not Path(path).exists():
            raise SystemExit(f"[preflight] {label}: {path} does not exist")

    whitelist = sorted(load_token_whitelist(args.status_csv, include_eth=args.include_eth))
    print(f"[config] {len(whitelist):,} assets | native ETH {'INCLUDED' if args.include_eth else 'excluded'}")
    if args.max_tokens:
        whitelist = whitelist[: args.max_tokens]

    con = duckdb.connect()
    con.execute(f"SET threads={int(args.threads)}")
    con.execute(f"SET memory_limit='{args.mem}'")
    con.execute("SET preserve_insertion_order=false")
    if args.temp_dir:
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory={sql_quote(args.temp_dir)}")
    con.execute("CREATE TEMP TABLE addrs (address VARCHAR, in_eth BOOLEAN)")

    rows_total = rows_eth = 0
    n_ok = n_skip = 0
    t0 = time.time()
    for i, token in enumerate(whitelist, start=1):
        g = find_token_parquet_glob(args.data_root, token)
        if not g:
            n_skip += 1
            if token == ETH_PSEUDO_TOKEN:
                print("[warn] native-ETH partition 0xee..ee not found under the data root")
            continue
        is_eth = token == ETH_PSEUDO_TOKEN
        n_rows = int(con.execute(f"SELECT COUNT(*) FROM read_parquet({sql_quote(g)})").fetchone()[0])
        con.execute(f"""
            INSERT INTO addrs
            SELECT DISTINCT lower(trim(address)) AS address, {'true' if is_eth else 'false'}
            FROM read_parquet({sql_quote(g)})
            WHERE address IS NOT NULL AND address <> ''
        """)
        rows_total += n_rows
        rows_eth += n_rows if is_eth else 0
        n_ok += 1
        if is_eth or n_ok % 250 == 0:
            print(f"[{i}/{len(whitelist)}] {n_ok} assets, {rows_total:,} rows, {time.time() - t0:,.0f}s"
                  f"{' (native ETH: ' + f'{n_rows:,}' + ' rows)' if is_eth else ''}", flush=True)

    zero_filter = "" if args.keep_zero_address else f"WHERE address <> '{ZERO_ADDRESS}'"
    con.execute(f"""
        CREATE TEMP TABLE universe AS
        SELECT address, BOOL_OR(NOT in_eth) AS in_tokens, BOOL_OR(in_eth) AS in_eth
        FROM addrs {zero_filter} GROUP BY address
    """)
    n_all, n_tok, n_eth_only = con.execute("""
        SELECT COUNT(*), COUNT(*) FILTER (WHERE in_tokens), COUNT(*) FILTER (WHERE in_eth AND NOT in_tokens)
        FROM universe
    """).fetchone()

    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT address FROM universe ORDER BY address) TO {sql_quote(args.out_parquet)} (FORMAT PARQUET)")

    stats = {
        "assets_scanned": n_ok, "assets_without_parquet": n_skip,
        "rows": rows_total, "rows_native_eth": rows_eth,
        "transfers": rows_total // 2,
        "unique_addresses": int(n_all), "unique_in_token_ledgers": int(n_tok),
        "unique_only_in_eth_ledger": int(n_eth_only),
    }
    print("\n[universe]")
    for k, v in stats.items():
        print(f"  {k:28s} {v:,}")
    print(f"[ok] wrote {args.out_parquet}")
    if args.stats_csv:
        Path(args.stats_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([stats]).to_csv(args.stats_csv, index=False)
        print(f"[ok] wrote {args.stats_csv}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
