#!/usr/bin/env python3
"""
portfolio_size_by_wealth_bucket.py

Description:
    For each Ethereum block snapshot, compute the distribution of
    portfolio size (number of distinct assets held) across wealth
    buckets, separately for EOAs, contract accounts (CAs), and all
    wallets pooled.

    Input is either a single 4_0 parquet file or a 4_2-style hive root
    (block_number=N/ subdirs, block_number NOT in the file bodies).

    Native ETH is a position like any other token (pseudo-token
    0xee..ee), so `num_tokens` in the reconstruction already counts it.
    WETH is NOT folded into ETH by the reconstruction (4_0); by default
    this script folds it, so a wallet holding both ETH and WETH counts
    them as ONE asset. That matches the optimisation pipeline (5_0) and
    the paper, which treats ETH and WETH as a single asset throughout.
    Pass --keep-weth-separate to reproduce the raw `num_tokens`.

    Wealth buckets (USD, upper bound inclusive):
        0-1, 1-100, 100-1,000, 1,000-10,000, 10,000-100,000, >100,000

    Portfolio-size buckets (number of assets):
        0, 1, 2, 3, 4, 5, 6-10, 11-20, 21-50, 51-100, >100

    Wallets listed in --ca-tsv are classified as CA; everything else as
    EOA.

Input:
    - Parquet with wallet_address, num_tokens, total_value_usd and (for
      WETH folding) the nested holdings column; block_number either in
      the file or in the hive directory names (--parquet).
    - TSV with contract-account addresses (first column, --ca-tsv; a
      header row is tolerated).

Output (in --out-dir, appended per block):
    - portfolio_size_stats_by_wealth_wallet_type.csv
          block, wallet_type, wealth_bucket, n_wallets, mean, median,
          p25, p75, p90, p95, p99, max
    - portfolio_size_hist_by_wealth_wallet_type.csv
          size_bucket, wallet_count, wallet_pct, wallet_type,
          wealth_bucket, block
    - portfolio_size_stats_by_wealth.csv   (all wallets pooled, no
          wallet_type column)
    - portfolio_size_hist_by_wealth.csv    (all wallets pooled; read by
          generate_all_macros.py)

Usage:
    python portfolio_size_by_wealth_bucket.py \\
        --parquet /path/to/recon-eth-erc20-partitioned \\
        --ca-tsv  /path/to/data/contract_accounts_in_dataset.tsv \\
        --out-dir ./analytics_out \\
        --threads 32 --mem 100GB --resume

    Optional: --blocks 21525891 21747950 ...   --max-blocks 12
              --keep-weth-separate
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set

import duckdb
import pandas as pd


ETH_PSEUDO_TOKEN = "0x" + "ee" * 20
WETH_TOKEN = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"

# The beacon-chain deposit contract holds every ETH ever staked (73 M ETH by
# December 2025) on behalf of validators whose claims live on the consensus
# layer. That is custody, not an account's wealth, so the address is excluded
# as an ETH holder from every value-based statistic. Token universe unchanged.
BEACON_DEPOSIT_CONTRACT = "0x00000000219ab540356cbb839cbe05303d7705fa"

OUT_STATS_WT = "portfolio_size_stats_by_wealth_wallet_type.csv"
OUT_HIST_WT = "portfolio_size_hist_by_wealth_wallet_type.csv"
OUT_STATS_ALL = "portfolio_size_stats_by_wealth.csv"
OUT_HIST_ALL = "portfolio_size_hist_by_wealth.csv"

STATS_COLS = ["n_wallets", "mean", "median", "p25", "p75", "p90", "p95", "p99", "max"]


# -----------------------------------------------------------------------------
# Buckets (labels must stay byte-identical: downstream code matches on them)
# -----------------------------------------------------------------------------
WEALTH_ORDER = {
    "0–1": 0,
    "1–100": 1,
    "100–1,000": 2,
    "1,000–10,000": 3,
    "10,000–100,000": 4,
    ">100,000": 5,
}

WEALTH_CASE_SQL = """
CASE
  WHEN v <=       1 THEN '0–1'
  WHEN v <=     100 THEN '1–100'
  WHEN v <=    1000 THEN '100–1,000'
  WHEN v <=   10000 THEN '1,000–10,000'
  WHEN v <=  100000 THEN '10,000–100,000'
  ELSE '>100,000'
END
"""

SIZE_ORDER = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6–10": 6,
    "11–20": 7,
    "21–50": 8,
    "51–100": 9,
    ">100": 10,
}

SIZE_CASE_SQL = """
CASE
  WHEN n_assets <=   0 THEN '0'
  WHEN n_assets  =   1 THEN '1'
  WHEN n_assets  =   2 THEN '2'
  WHEN n_assets  =   3 THEN '3'
  WHEN n_assets  =   4 THEN '4'
  WHEN n_assets  =   5 THEN '5'
  WHEN n_assets <=  10 THEN '6–10'
  WHEN n_assets <=  20 THEN '11–20'
  WHEN n_assets <=  50 THEN '21–50'
  WHEN n_assets <= 100 THEN '51–100'
  ELSE '>100'
END
"""


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def parquet_scan_target(path: str | Path) -> str:
    """Accept either a single parquet file or a 4_2-style hive root
    (block_number=N/ subdirs, block_number NOT in file bodies). For a
    directory, return a glob the queries read with hive_partitioning=1
    so block_number materializes from the directory names."""
    p = Path(path)
    if p.is_dir():
        return str(p.resolve() / "*" / "*.parquet")
    return str(p.resolve())


def get_duck(threads: int, mem: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads = ?", [int(threads)])
    con.execute("PRAGMA memory_limit = ?", [str(mem)])
    con.execute("PRAGMA preserve_insertion_order=false;")
    return con


def parquet_columns(con: duckdb.DuckDBPyConnection, parquet: str) -> Set[str]:
    rows = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?, hive_partitioning=1) LIMIT 0",
        [parquet],
    ).fetchall()
    return {r[0] for r in rows}


def get_blocks(con: duckdb.DuckDBPyConnection, parquet: str) -> List[int]:
    rows = con.execute(
        "SELECT DISTINCT block_number::BIGINT AS b "
        "FROM read_parquet(?, hive_partitioning=1) ORDER BY b",
        [parquet],
    ).fetchall()
    return [int(r[0]) for r in rows]


def normalize_address(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", "ignore")
        except Exception:
            return ""
    return str(x).strip().lower()


def load_ca_table(con: duckdb.DuckDBPyConnection, tsv_path: str | Path) -> int:
    """Load the first column of the TSV into a temp table `ca(address)`.
    A header row is harmless: it never matches a real address."""
    addrs: Set[str] = set()
    with Path(tsv_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            addr = normalize_address(line.split("\t")[0])
            if addr:
                addrs.add(addr)
    ca_df = pd.DataFrame({"address": sorted(addrs)})
    con.register("ca_df", ca_df)
    con.execute("CREATE OR REPLACE TEMP TABLE ca AS SELECT address FROM ca_df")
    con.unregister("ca_df")
    return len(addrs)


def append_csv(df: pd.DataFrame, path: Path, columns: List[str]) -> None:
    if df.empty:
        return
    header = not path.exists()
    df[columns].to_csv(path, mode="a", header=header, index=False)


def done_blocks_from(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    try:
        prev = pd.read_csv(path, usecols=["block"])
        return set(int(b) for b in prev["block"].unique())
    except Exception as e:  # pragma: no cover
        print(f"[resume] could not read {path} ({e}); recomputing all blocks")
        return set()


# -----------------------------------------------------------------------------
# Per-block computation (all heavy lifting in DuckDB)
# -----------------------------------------------------------------------------
def build_sized_table(
    con: duckdb.DuckDBPyConnection, parquet: str, block: int, fold_weth: bool
) -> int:
    """Materialise one block as temp table `sized(wallet_type, wealth_bucket,
    n_assets, size_bucket)`; return its row count."""
    if fold_weth:
        pos = "COALESCE(TRY_CAST(h.balance AS DOUBLE), 0) > 0"
        fold_sql = f"""
        CASE WHEN len(list_filter(holdings, h -> lower(h.token_address) = '{ETH_PSEUDO_TOKEN}' AND {pos})) > 0
              AND len(list_filter(holdings, h -> lower(h.token_address) = '{WETH_TOKEN}' AND {pos})) > 0
             THEN 1 ELSE 0 END
        """
    else:
        fold_sql = "0"

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE sized AS
    WITH base AS (
      SELECT
        lower(trim(wallet_address)) AS addr,
        GREATEST(COALESCE(TRY_CAST(total_value_usd AS DOUBLE), 0.0), 0.0) AS v,
        GREATEST(COALESCE(TRY_CAST(num_tokens AS BIGINT), 0), 0) AS num_tokens,
        {fold_sql} AS fold
      FROM read_parquet(?, hive_partitioning=1)
      WHERE block_number = ?
        AND wallet_address IS NOT NULL
        AND lower(trim(wallet_address)) <> '{BEACON_DEPOSIT_CONTRACT}'
    ),
    typed AS (
      SELECT
        CASE WHEN addr IN (SELECT address FROM ca) THEN 'CA' ELSE 'EOA' END AS wallet_type,
        {WEALTH_CASE_SQL} AS wealth_bucket,
        GREATEST(num_tokens - fold, 0) AS n_assets
      FROM base
    )
    SELECT wallet_type, wealth_bucket, n_assets, {SIZE_CASE_SQL} AS size_bucket
    FROM typed
    """, [parquet, int(block)])
    return int(con.execute("SELECT COUNT(*) FROM sized").fetchone()[0])


STATS_SELECT = """
  COUNT(*)                        AS n_wallets,
  AVG(n_assets)                   AS mean,
  quantile_cont(n_assets, 0.50)   AS median,
  quantile_cont(n_assets, 0.25)   AS p25,
  quantile_cont(n_assets, 0.75)   AS p75,
  quantile_cont(n_assets, 0.90)   AS p90,
  quantile_cont(n_assets, 0.95)   AS p95,
  quantile_cont(n_assets, 0.99)   AS p99,
  MAX(n_assets)                   AS max
"""


def stats_for_block(con: duckdb.DuckDBPyConnection, block: int) -> pd.DataFrame:
    df = con.execute(f"""
    SELECT wallet_type, wealth_bucket, {STATS_SELECT}
    FROM sized GROUP BY 1, 2
    UNION ALL
    SELECT 'ALL' AS wallet_type, wealth_bucket, {STATS_SELECT}
    FROM sized GROUP BY 2
    """).df()
    df["block"] = int(block)
    df["wealth_order"] = df["wealth_bucket"].map(WEALTH_ORDER).fillna(999)
    df = df.sort_values(["block", "wallet_type", "wealth_order"]).drop(columns=["wealth_order"])
    for c in ("mean", "median", "p25", "p75", "p90", "p95", "p99", "max"):
        df[c] = df[c].astype("float64")
    df["n_wallets"] = df["n_wallets"].astype("int64")
    return df.reset_index(drop=True)


def hist_for_block(con: duckdb.DuckDBPyConnection, block: int) -> pd.DataFrame:
    df = con.execute("""
    SELECT wallet_type, wealth_bucket, size_bucket, COUNT(*) AS wallet_count
    FROM sized GROUP BY 1, 2, 3
    UNION ALL
    SELECT 'ALL' AS wallet_type, wealth_bucket, size_bucket, COUNT(*) AS wallet_count
    FROM sized GROUP BY 2, 3
    """).df()
    df["block"] = int(block)
    df["wallet_count"] = df["wallet_count"].astype("int64")
    denom = df.groupby(["wallet_type", "wealth_bucket"])["wallet_count"].transform("sum")
    df["wallet_pct"] = (100.0 * df["wallet_count"] / denom).round(4)
    df["wealth_order"] = df["wealth_bucket"].map(WEALTH_ORDER).fillna(999)
    df["size_order"] = df["size_bucket"].map(SIZE_ORDER).fillna(999)
    df = (
        df.sort_values(["block", "wallet_type", "wealth_order", "size_order"])
        .drop(columns=["wealth_order", "size_order"])
        .reset_index(drop=True)
    )
    return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Portfolio size distribution (number of assets) by wealth bucket with EOA/CA split."
    )
    ap.add_argument("--parquet", required=True,
                    help="Reconstructed portfolio parquet: single file or 4_2 hive root.")
    ap.add_argument("--ca-tsv", required=True,
                    help="TSV with contract addresses (CA), first column is address.")
    ap.add_argument("--out-dir", default="analytics_out", help="Output directory.")
    ap.add_argument("--threads", type=int, default=32, help="DuckDB threads.")
    ap.add_argument("--mem", type=str, default="100GB", help="DuckDB memory limit, e.g. 100GB.")
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="Optional limit on number of blocks processed.")
    ap.add_argument("--blocks", nargs="*", type=int, default=None,
                    help="Optional explicit list of blocks to process.")
    ap.add_argument("--keep-weth-separate", action="store_true",
                    help="Do not fold WETH into ETH (use raw num_tokens).")
    ap.add_argument("--resume", action="store_true",
                    help="Keep existing outputs and skip blocks already present in them.")
    args = ap.parse_args()

    parquet = parquet_scan_target(args.parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_stats_wt = out_dir / OUT_STATS_WT
    out_hist_wt = out_dir / OUT_HIST_WT
    out_stats_all = out_dir / OUT_STATS_ALL
    out_hist_all = out_dir / OUT_HIST_ALL
    outputs = [out_stats_wt, out_hist_wt, out_stats_all, out_hist_all]

    done: Set[int] = set()
    if args.resume:
        # Only blocks present in EVERY output count as done, so a run that
        # died between the four appends is redone for that block.
        sets = [done_blocks_from(p) for p in outputs]
        done = set.intersection(*sets) if all(sets) else set()
        partial = set.union(*sets) - done
        if partial:
            print(f"[resume] {len(partial)} block(s) only partially written; "
                  f"they will be recomputed (duplicates must be dropped downstream: {sorted(partial)})")
        print(f"[resume] {len(done)} block(s) already complete")
    else:
        for p in outputs:
            p.unlink(missing_ok=True)

    con = get_duck(args.threads, args.mem)

    n_ca = load_ca_table(con, args.ca_tsv)
    print(f"[info] loaded CA addresses: {n_ca:,}")
    print(f"[info] parquet: {parquet}")

    cols = parquet_columns(con, parquet)
    missing = {"wallet_address", "num_tokens", "total_value_usd", "block_number"} - cols
    if missing:
        raise SystemExit(f"[error] parquet lacks required columns: {sorted(missing)}")

    fold_weth = not args.keep_weth_separate
    if fold_weth and "holdings" not in cols:
        print("[warn] no `holdings` column: cannot detect ETH+WETH pairs, using raw num_tokens")
        fold_weth = False
    print(f"[info] WETH folded into ETH: {fold_weth}")

    if args.blocks:
        blocks = [int(b) for b in args.blocks]
    else:
        blocks = get_blocks(con, parquet)
        if args.max_blocks is not None:
            blocks = blocks[: args.max_blocks]
    print(f"[info] blocks: {len(blocks):,}")

    for i, block in enumerate(blocks, start=1):
        if block in done:
            print(f"[{i}/{len(blocks)}] skip block {block} (already done)")
            continue
        print(f"\n==== block {block} ({i}/{len(blocks)}) ====")

        n_rows = build_sized_table(con, parquet, block, fold_weth)
        if n_rows == 0:
            print("[warn] no rows")
            continue

        stats = stats_for_block(con, block)
        hist = hist_for_block(con, block)

        is_all_s = stats["wallet_type"] == "ALL"
        is_all_h = hist["wallet_type"] == "ALL"

        append_csv(stats[~is_all_s], out_stats_wt,
                   ["block", "wallet_type", "wealth_bucket", *STATS_COLS])
        append_csv(stats[is_all_s], out_stats_all,
                   ["block", "wealth_bucket", *STATS_COLS])
        append_csv(hist[~is_all_h], out_hist_wt,
                   ["size_bucket", "wallet_count", "wallet_pct", "wallet_type", "wealth_bucket", "block"])
        append_csv(hist[is_all_h], out_hist_all,
                   ["size_bucket", "wallet_count", "wallet_pct", "wealth_bucket", "block"])

        n_by_type = con.execute(
            "SELECT wallet_type, COUNT(*) FROM sized GROUP BY 1"
        ).fetchall()
        n_by_type = {k: int(v) for k, v in n_by_type}
        print(f"[info] wallets: EOA={n_by_type.get('EOA', 0):,} CA={n_by_type.get('CA', 0):,} "
              f"| stats_rows={len(stats)} hist_rows={len(hist)}")

    con.execute("DROP TABLE IF EXISTS sized")
    print("\n[done] wrote:")
    for p in outputs:
        print(f"  - {p.resolve()}")


if __name__ == "__main__":
    main()
