#!/usr/bin/env python3
"""
count_accounts_eoa_vs_ca.py

Description:
    Number of accounts observed in the data set at each snapshot, split into
    externally owned accounts (EOA) and contract accounts (CA).

    An address is "observed" at snapshot block B if it appears, as sender or
    recipient, in the ledger of any whitelisted TOKEN at a block <= B, so the
    counts are cumulative over the chain history. This is the account
    universe of the study (addresses that ever moved an ERC-20 token; their
    ETH balances are added later, ETH-only addresses stay outside). The
    native-ETH ledger (pseudo-token 0xee..ee in the status CSV) is therefore
    NOT scanned unless --include-eth is given, which yields the larger
    "ever moved ETH or a token" population instead.
    A CA is an observed address that also appears in the contract-creation
    TSV; it counts from ca_effective_block = max(first seen, created).

    First-seen blocks are accumulated asset by asset, tagged with the asset,
    in a persistent DuckDB file, so a crashed run can continue with --resume
    (assets already scanned are skipped) and one scan can serve both views:
    the counts are always computed over the assets selected for the current
    run, whatever else the file contains. The native-ETH ledger is by far
    the largest partition; give DuckDB a --temp-dir on the cluster.

Input:
    --data-root        structured transfers dump: token_address=<addr>/*.parquet
                       with columns address, block, log_index, value
    --status-csv       token status CSV (whitelist; see shared/token_whitelist.py)
    --snapshots-csv    snapshot list, normally the 4_2 _manifest.csv
                       (block_number, date); --block-to-date-csv resolves or
                       cross-checks dates
    --ca-created-tsv   <address>\\t<creation_block> per line, no header
                       (scan_traces_for_ca_addresses.py)

Output:
    --out-csv          block, date, eoa_accounts, ca_accounts
                       (default ../data/output_accounts_eoa_vs_ca.csv, read by
                       generate_all_macros.py and plot_eoa_ca_over_time.R)
    --ca-out-tsv       addr, creation_block, first_seen_block, ca_effective_block
                       for every CA observed in the data set: the
                       contract_accounts_in_dataset.tsv that the
                       distribution_of_wealth and portfolio_size scripts take
                       as --ca-tsv (written only when the flag is given)

Usage (cluster):
    python count_accounts_eoa_vs_ca.py \\
        --data-root /path/to/eth-erc20-transfers-20251208-structured \\
        --status-csv /path/to/data/status_eth_tokens_3_1.csv \\
        --snapshots-csv /path/to/recon-eth-erc20-partitioned/_manifest.csv \\
        --ca-created-tsv /path/to/data/contract_accounts_created.tsv \\
        --ca-out-tsv /path/to/data/contract_accounts_in_dataset.tsv \\
        --duckdb /path/to/duckdb_tmp/first_seen_tokens.duckdb \\
        --threads 32 --mem 100GB --temp-dir /path/to/duckdb_tmp --resume

    Add --include-eth for the population that also covers ETH-only addresses.
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
from snapshots import cumulative_counts, find_token_parquet_glob, load_snapshots  # noqa: E402
from token_whitelist import ETH_PSEUDO_TOKEN, ZERO_ADDRESS, load_token_whitelist  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data"


def sql_quote(p: str) -> str:
    return "'" + str(p).replace("'", "''") + "'"


def get_duck(db_path: str, threads: int, mem: str, temp_dir: str | None) -> duckdb.DuckDBPyConnection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit='{mem}'")
    con.execute("SET preserve_insertion_order=false")
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory={sql_quote(temp_dir)}")
    return con


def init_tables(con: duckdb.DuckDBPyConnection, resume: bool, db_path: str) -> None:
    if not resume:
        con.execute("DROP TABLE IF EXISTS first_seen_raw")
        con.execute("DROP TABLE IF EXISTS scanned_assets")
    con.execute("CREATE TABLE IF NOT EXISTS first_seen_raw (token VARCHAR, addr VARCHAR, first_seen_block BIGINT)")
    con.execute("CREATE TABLE IF NOT EXISTS scanned_assets (token VARCHAR, n_addrs BIGINT, seconds DOUBLE)")
    cols = {r[1] for r in con.execute("PRAGMA table_info('first_seen_raw')").fetchall()}
    if "token" not in cols:
        raise SystemExit(f"[resume] {db_path} was scanned by an older version without the per-asset tag; "
                         "its rows cannot be filtered by asset. Use a new --duckdb path.")


def scan_asset(con: duckdb.DuckDBPyConnection, token: str, parquet_glob: str, max_block: int) -> int:
    """Append (addr, first block) of every address in one ledger to first_seen_raw."""
    t0 = time.time()
    con.execute("BEGIN")
    try:
        n_before = int(con.execute("SELECT COUNT(*) FROM first_seen_raw").fetchone()[0])
        con.execute(f"""
            INSERT INTO first_seen_raw
            SELECT {sql_quote(token)} AS token, lower(trim(address)) AS addr, MIN(block) AS first_seen_block
            FROM read_parquet({sql_quote(parquet_glob)})
            WHERE address IS NOT NULL AND address <> '' AND block <= {int(max_block)}
            GROUP BY 1, 2
        """)
        n_after = int(con.execute("SELECT COUNT(*) FROM first_seen_raw").fetchone()[0])
        con.execute("INSERT INTO scanned_assets VALUES (?, ?, ?)", [token, n_after - n_before, time.time() - t0])
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return n_after


def main() -> int:
    ap = argparse.ArgumentParser(description="EOA vs CA accounts observed per snapshot (cumulative).")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--status-csv", required=True)
    ap.add_argument("--snapshots-csv", required=True)
    ap.add_argument("--block-to-date-csv", default=None)
    ap.add_argument("--ca-created-tsv", required=True)
    ap.add_argument("--out-csv", default=str(DEFAULT_OUT_DIR / "output_accounts_eoa_vs_ca.csv"))
    ap.add_argument("--ca-out-tsv", default=None,
                    help="Also write the observed-CA list (contract_accounts_in_dataset.tsv) here.")
    ap.add_argument("--duckdb", default="first_seen.duckdb",
                    help="Persistent DuckDB file holding the per-asset first-seen rows (enables --resume).")
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--mem", default="32GB")
    ap.add_argument("--temp-dir", default=None)
    ap.add_argument("--include-eth", action="store_true",
                    help="Also scan the native-ETH ledger (addresses that only ever moved ETH enter the counts).")
    ap.add_argument("--resume", action="store_true", help="Keep the DuckDB file and skip assets already scanned.")
    ap.add_argument("--max-tokens", type=int, default=None, help="Scan only the first N assets (smoke test).")
    args = ap.parse_args()

    # fail on missing inputs BEFORE the multi-hour scan
    for label, path in (("--data-root", args.data_root), ("--status-csv", args.status_csv),
                        ("--snapshots-csv", args.snapshots_csv), ("--ca-created-tsv", args.ca_created_tsv),
                        ("--block-to-date-csv", args.block_to_date_csv)):
        if path and not Path(path).exists():
            raise SystemExit(f"[preflight] {label}: {path} does not exist")
    if args.ca_out_tsv:
        Path(args.ca_out_tsv).parent.mkdir(parents=True, exist_ok=True)

    snaps = load_snapshots(args.snapshots_csv, args.block_to_date_csv)
    target_blocks = snaps["block"].tolist()
    date_map = dict(zip(snaps["block"], snaps["date"]))
    max_block = max(target_blocks)

    whitelist = sorted(load_token_whitelist(args.status_csv, include_eth=args.include_eth))
    if args.max_tokens:
        whitelist = whitelist[: args.max_tokens]
    print(f"[config] {len(whitelist):,} assets to scan | native ETH {'INCLUDED' if args.include_eth else 'excluded'}"
          f" | duckdb={args.duckdb}")

    con = get_duck(args.duckdb, args.threads, args.mem, args.temp_dir)
    init_tables(con, args.resume, args.duckdb)
    done = {r[0] for r in con.execute("SELECT token FROM scanned_assets").fetchall()}
    if done:
        print(f"[resume] {len(done)} asset(s) already scanned in {args.duckdb}")

    # ---- 1. first-seen block per address, asset by asset --------------------
    n_ok = n_skip = 0
    for i, token in enumerate(whitelist, start=1):
        if token in done:
            continue
        g = find_token_parquet_glob(args.data_root, token)
        if not g:
            n_skip += 1
            if token == ETH_PSEUDO_TOKEN:
                print("[warn] native-ETH partition 0xee..ee not found under the data root")
            continue
        tag = " (native ETH, large)" if token == ETH_PSEUDO_TOKEN else ""
        print(f"[{i}/{len(whitelist)}] {token}{tag}", flush=True)
        total = scan_asset(con, token, g, max_block)
        n_ok += 1
        if n_ok % 100 == 0:
            print(f"  ... {n_ok} assets scanned, {total:,} raw rows so far", flush=True)
    print(f"[scan] done: {n_ok} scanned now, {len(done)} from earlier runs, {n_skip} without parquet")

    # ---- 2. global first-seen and CA join ------------------------------------
    # only the assets of THIS run count (the file may hold more, e.g. an
    # earlier --include-eth scan); the zero address is the mint/burn
    # counterparty of ERC-20 ledgers, not an account
    con.register("run_assets_df", pd.DataFrame({"token": whitelist}))
    con.execute("CREATE OR REPLACE TEMP TABLE run_assets AS SELECT token FROM run_assets_df")
    con.unregister("run_assets_df")
    con.execute(f"""
        CREATE OR REPLACE TABLE global_first_seen AS
        SELECT addr, MIN(first_seen_block) AS first_seen_block
        FROM first_seen_raw
        WHERE addr <> '{ZERO_ADDRESS}' AND token IN (SELECT token FROM run_assets)
        GROUP BY addr
    """)
    n_addr = int(con.execute("SELECT COUNT(*) FROM global_first_seen").fetchone()[0])
    print(f"[addresses] {n_addr:,} distinct addresses observed up to block {max_block:,} "
          f"({'ETH ledger included' if args.include_eth else 'token ledgers only'})")

    con.execute(f"""
        CREATE OR REPLACE TABLE contract_creation AS
        SELECT lower(trim(column0)) AS addr, MIN(TRY_CAST(column1 AS BIGINT)) AS creation_block
        FROM read_csv({sql_quote(args.ca_created_tsv)}, delim='\\t', header=false,
                      names=['column0', 'column1'], all_varchar=true, ignore_errors=true)
        WHERE column0 IS NOT NULL AND lower(trim(column0)) LIKE '0x%' AND length(trim(column0)) = 42
          AND TRY_CAST(column1 AS BIGINT) IS NOT NULL
        GROUP BY 1
    """)
    n_ca_created = int(con.execute("SELECT COUNT(*) FROM contract_creation").fetchone()[0])
    con.execute("""
        CREATE OR REPLACE TABLE ca_observed AS
        SELECT g.addr, cc.creation_block, g.first_seen_block,
               GREATEST(g.first_seen_block, cc.creation_block) AS ca_effective_block
        FROM global_first_seen g JOIN contract_creation cc USING (addr)
    """)
    n_ca_obs = int(con.execute("SELECT COUNT(*) FROM ca_observed").fetchone()[0])
    print(f"[contracts] {n_ca_created:,} created contracts in the TSV, {n_ca_obs:,} of them observed in the data set")

    # ---- 3. cumulative counts per snapshot (histogram + cumsum) --------------
    h_all = con.execute("SELECT first_seen_block AS block, COUNT(*) AS n FROM global_first_seen GROUP BY 1").df()
    h_ca = con.execute("SELECT ca_effective_block AS block, COUNT(*) AS n FROM ca_observed GROUP BY 1").df()
    total = cumulative_counts(h_all, target_blocks)
    ca = cumulative_counts(h_ca, target_blocks)

    out = pd.DataFrame({
        "block": target_blocks,
        "date": [date_map[b] for b in target_blocks],
        "eoa_accounts": (total - ca).values,
        "ca_accounts": ca.values,
    })
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {args.out_csv}")
    print(out.iloc[[0, len(out) // 2, -1]].to_string(index=False))
    last = out.iloc[-1]
    print(f"[end] accounts {int(last['eoa_accounts'] + last['ca_accounts']):,}"
          f" | EOA {int(last['eoa_accounts']):,} | CA {int(last['ca_accounts']):,}"
          f" ({100 * last['ca_accounts'] / (last['eoa_accounts'] + last['ca_accounts']):.2f}%)")

    if args.ca_out_tsv:
        Path(args.ca_out_tsv).parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"""
            COPY (SELECT addr, creation_block, first_seen_block, ca_effective_block
                  FROM ca_observed ORDER BY addr)
            TO {sql_quote(args.ca_out_tsv)} (FORMAT CSV, DELIMITER '\t', HEADER true)
        """)
        print(f"[ok] wrote {args.ca_out_tsv} ({n_ca_obs:,} contract accounts)")

    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
