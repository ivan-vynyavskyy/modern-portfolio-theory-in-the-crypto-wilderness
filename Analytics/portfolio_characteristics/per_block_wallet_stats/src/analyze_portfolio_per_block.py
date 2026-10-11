#!/usr/bin/env python3
"""
analyze_portfolio_per_block.py

Description:
    Per-snapshot wallet aggregates over the reconstructed portfolios, split
    by wallet type:
        CA  = contract accounts (addresses listed in --ca-tsv)
        EOA = everything else
        ALL = CA + EOA

    Input is either a single 4_0 parquet file or a 4_2-style hive root
    (block_number=N/ subdirectories, block_number only in the directory
    names). Every metric is computed in DuckDB, so the script scales to the
    tens of millions of accounts per snapshot of the ETH-inclusive dataset.

    Native ETH is the pseudo-token 0xee..ee and counts as an asset like any
    token; `num_tokens` in the reconstruction already includes it. WETH is
    NOT folded into ETH by 4_0; by default this script folds it, so a
    wallet holding both ETH and WETH counts them as ONE asset. That matches
    the optimisation pipeline (5_0) and the paper. Pass --keep-weth-separate
    to use the raw `num_tokens`.

    For each block and wallet_type the script writes (appended per block):

    1) wallet_value_distribution_per_block.csv
         wallets_distinct, wallets_val_eq0, wallets_val_lt1, total_value_usd,
         top_{1,5,10}_share_pct (share of total value held by the top
         1/5/10 % of wallets by value), top_{1,5,10}_num_wallets
    2) averages_per_block.csv
         avg/med num_tokens and total_value_usd, over all wallets and over
         wallets with value > 1 USD (num_tokens = assets after WETH folding)
    3) asset_buckets_per_block.csv
         bucket (#assets: 0, 1, 2–5, 5–10, 10–20, 20–50, 50–100, >100),
         wallet_count, wallet_pct, value_sum_usd, value_pct

Input:
    - --parquet   reconstructed portfolios: single file or 4_2 hive root
    - --ca-tsv    TSV with contract addresses in the first column (a header
                  row is tolerated)

Usage (cluster):
    python analyze_portfolio_per_block.py \\
        --parquet /path/to/recon-eth-erc20-partitioned \\
        --ca-tsv  /path/to/data/contract_accounts_in_dataset.tsv \\
        --out-dir ../data --threads 32 --mem 100GB \\
        --temp-dir /path/to/duckdb_tmp --resume

    Optional: --blocks 9193266 9393154 ...   --max-blocks 3
              --keep-weth-separate
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))

ETH_PSEUDO_TOKEN = "0x" + "ee" * 20
WETH_TOKEN = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"

# The beacon-chain deposit contract holds every ETH ever staked (73 M ETH by
# December 2025) on behalf of validators whose claims live on the consensus
# layer. That is custody, not an account's wealth, so the address is excluded
# as an ETH holder from every value-based statistic. Token universe unchanged.
BEACON_DEPOSIT_CONTRACT = "0x00000000219ab540356cbb839cbe05303d7705fa"

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data"
WALLET_TYPES = ["ALL", "EOA", "CA"]

OUT_FILES = {
    "value_dist": "wallet_value_distribution_per_block.csv",
    "averages": "averages_per_block.csv",
    "asset_buckets": "asset_buckets_per_block.csv",
}

# Labels must stay byte-identical to earlier outputs (downstream code matches on them).
ASSET_BUCKETS = ["0", "1", "2–5", "5–10", "10–20", "20–50", "50–100", ">100"]
ASSET_BUCKET_SQL = """
CASE
  WHEN n_assets <=   0 THEN '0'
  WHEN n_assets  =   1 THEN '1'
  WHEN n_assets <=   5 THEN '2–5'
  WHEN n_assets <=  10 THEN '5–10'
  WHEN n_assets <=  20 THEN '10–20'
  WHEN n_assets <=  50 THEN '20–50'
  WHEN n_assets <= 100 THEN '50–100'
  ELSE '>100'
END
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def sql_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def normalize_address(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", "ignore")
        except Exception:
            return ""
    return str(x).strip().lower()


def get_duck(threads: int, mem: str, temp_dir: Optional[str]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit='{mem}'")
    con.execute("SET preserve_insertion_order=false")
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory={sql_quote(temp_dir)}")
    return con


def parquet_scan_target(path: str | Path) -> str:
    """Single file, or a 4_2 hive root read as <root>/*/*.parquet with
    hive_partitioning=1 so block_number materialises from the directory names."""
    p = Path(path)
    if p.is_dir():
        return str(p.resolve() / "*" / "*.parquet")
    return str(p.resolve())


def discover_blocks(con: duckdb.DuckDBPyConnection, path: str | Path, scan: str) -> List[int]:
    p = Path(path)
    if p.is_dir():
        blocks = []
        for d in p.iterdir():
            m = re.fullmatch(r"block_number=(\d+)", d.name)
            if d.is_dir() and m:
                blocks.append(int(m.group(1)))
        if blocks:
            return sorted(blocks)
    rows = con.execute(
        "SELECT DISTINCT block_number::BIGINT FROM read_parquet(?, hive_partitioning=1) ORDER BY 1",
        [scan],
    ).fetchall()
    return [int(r[0]) for r in rows]


def load_ca_table(con: duckdb.DuckDBPyConnection, tsv_path: str | Path) -> int:
    """First column of the TSV -> temp table ca(address). A header row never matches an address."""
    addrs: Set[str] = set()
    with Path(tsv_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = normalize_address(line.split("\t")[0])
            if a:
                addrs.add(a)
    con.register("ca_df", pd.DataFrame({"address": sorted(addrs)}))
    con.execute("CREATE OR REPLACE TEMP TABLE ca AS SELECT address FROM ca_df")
    con.unregister("ca_df")
    return len(addrs)


def append_csv(df: pd.DataFrame, path: Path, columns: List[str]) -> None:
    if df.empty:
        return
    header = not path.exists()
    df[columns].to_csv(path, mode="a", header=header, index=False)


def done_blocks(paths: Dict[str, Path]) -> Set[int]:
    """Blocks present in every output file (a block is written to all files at once)."""
    done: Optional[Set[int]] = None
    for p in paths.values():
        if not p.exists():
            return set()
        try:
            got = set(int(b) for b in pd.read_csv(p, usecols=["block"])["block"].unique())
        except Exception as e:  # pragma: no cover
            print(f"[resume] could not read {p} ({e}); recomputing all blocks")
            return set()
        done = got if done is None else done & got
    return done or set()


def assert_bucket_totals(df: pd.DataFrame) -> None:
    """ALL must equal EOA + CA, overall and within every bucket."""
    if df.empty:
        return
    for bucket in [None] + list(df["bucket"].unique()):
        sub = df if bucket is None else df[df["bucket"] == bucket]
        tag = "totals" if bucket is None else f"bucket {bucket}"
        for col in ("wallet_count", "value_sum_usd"):
            vals = {wt: float(sub.loc[sub["wallet_type"] == wt, col].sum()) for wt in WALLET_TYPES}
            target = vals["EOA"] + vals["CA"]
            if abs(vals["ALL"] - target) > 1e-6 * max(1.0, abs(target)):
                raise AssertionError(f"{tag} {col}: ALL ({vals['ALL']}) != EOA+CA ({target})")


# -----------------------------------------------------------------------------
# Per-block base table (all heavy lifting in DuckDB)
# -----------------------------------------------------------------------------
def build_block_table(con: duckdb.DuckDBPyConnection, scan: str, block: int, fold_weth: bool) -> int:
    """Materialise one block as temp table w2(wallet_type, v, n_assets): one
    row per wallet with its own type plus a copy typed 'ALL'. Returns the
    wallet count."""
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
    CREATE OR REPLACE TEMP TABLE w AS
    WITH base AS (
      SELECT lower(trim(wallet_address)) AS addr,
             GREATEST(COALESCE(TRY_CAST(total_value_usd AS DOUBLE), 0.0), 0.0) AS v,
             GREATEST(COALESCE(TRY_CAST(num_tokens AS BIGINT), 0), 0) AS num_tokens,
             {fold_sql} AS fold
      FROM read_parquet({sql_quote(scan)}, hive_partitioning=1)
      WHERE block_number = {int(block)} AND wallet_address IS NOT NULL
        AND lower(trim(wallet_address)) <> '{BEACON_DEPOSIT_CONTRACT}'
    )
    SELECT CASE WHEN addr IN (SELECT address FROM ca) THEN 'CA' ELSE 'EOA' END AS wallet_type,
           v,
           GREATEST(num_tokens - fold, 0) AS n_assets
    FROM base
    """)
    con.execute("""
    CREATE OR REPLACE TEMP TABLE w2 AS
    SELECT wallet_type, v, n_assets FROM w
    UNION ALL
    SELECT 'ALL' AS wallet_type, v, n_assets FROM w
    """)
    n = int(con.execute("SELECT COUNT(*) FROM w").fetchone()[0])
    con.execute("DROP TABLE w")
    return n


# -----------------------------------------------------------------------------
# 1) Wallet value distribution + top shares
# -----------------------------------------------------------------------------
VALUE_DIST_COLS = [
    "block", "wallet_type", "wallets_distinct", "wallets_val_eq0", "wallets_val_lt1",
    "total_value_usd", "top_1_share_pct", "top_5_share_pct", "top_10_share_pct",
    "top_1_num_wallets", "top_5_num_wallets", "top_10_num_wallets",
]


def value_distribution(con: duckdb.DuckDBPyConnection, block: int) -> pd.DataFrame:
    df = con.execute("""
    WITH r AS (
      SELECT wallet_type, v,
             row_number() OVER (PARTITION BY wallet_type ORDER BY v DESC) AS rn,
             COUNT(*) OVER (PARTITION BY wallet_type) AS n
      FROM w2
    ),
    k AS (
      SELECT wallet_type, v, rn, n,
             GREATEST(1, CAST(CEIL(0.01 * n) AS BIGINT)) AS k1,
             GREATEST(1, CAST(CEIL(0.05 * n) AS BIGINT)) AS k5,
             GREATEST(1, CAST(CEIL(0.10 * n) AS BIGINT)) AS k10
      FROM r
    )
    SELECT wallet_type,
           COUNT(*)                                     AS wallets_distinct,
           COUNT(*) FILTER (WHERE v = 0)                AS wallets_val_eq0,
           COUNT(*) FILTER (WHERE v < 1)                AS wallets_val_lt1,
           SUM(v)                                       AS total_value_usd,
           SUM(v) FILTER (WHERE rn <= k1)               AS top_1_value,
           SUM(v) FILTER (WHERE rn <= k5)               AS top_5_value,
           SUM(v) FILTER (WHERE rn <= k10)              AS top_10_value,
           MAX(k1) AS k1, MAX(k5) AS k5, MAX(k10) AS k10
    FROM k GROUP BY wallet_type
    """).df()
    rows = []
    for wt in WALLET_TYPES:
        sub = df[df["wallet_type"] == wt]
        row = {"block": int(block), "wallet_type": wt, "wallets_distinct": 0, "wallets_val_eq0": 0,
               "wallets_val_lt1": 0, "total_value_usd": 0.0, "top_1_share_pct": 0.0,
               "top_5_share_pct": 0.0, "top_10_share_pct": 0.0, "top_1_num_wallets": 0,
               "top_5_num_wallets": 0, "top_10_num_wallets": 0}
        if not sub.empty:
            r = sub.iloc[0]
            tot = float(r["total_value_usd"] or 0.0)
            row.update({
                "wallets_distinct": int(r["wallets_distinct"]),
                "wallets_val_eq0": int(r["wallets_val_eq0"]),
                "wallets_val_lt1": int(r["wallets_val_lt1"]),
                "total_value_usd": tot,
            })
            if tot > 0:
                for p in ("1", "5", "10"):
                    row[f"top_{p}_share_pct"] = 100.0 * float(r[f"top_{p}_value"] or 0.0) / tot
                    row[f"top_{p}_num_wallets"] = int(r[f"k{p}"])
        rows.append(row)
    return pd.DataFrame(rows)[VALUE_DIST_COLS]


# -----------------------------------------------------------------------------
# 2) Averages and medians
# -----------------------------------------------------------------------------
AVERAGES_COLS = [
    "block", "wallet_type", "avg_num_tokens", "med_num_tokens", "avg_value_usd", "med_value_usd",
    "avg_num_tokens_gt1", "med_num_tokens_gt1", "avg_value_usd_gt1", "med_value_usd_gt1",
]


def averages(con: duckdb.DuckDBPyConnection, block: int) -> pd.DataFrame:
    df = con.execute("""
    SELECT wallet_type,
           AVG(n_assets)                                       AS avg_num_tokens,
           quantile_cont(n_assets, 0.5)                        AS med_num_tokens,
           AVG(v)                                              AS avg_value_usd,
           quantile_cont(v, 0.5)                               AS med_value_usd,
           AVG(n_assets)          FILTER (WHERE v > 1)         AS avg_num_tokens_gt1,
           quantile_cont(n_assets, 0.5) FILTER (WHERE v > 1)   AS med_num_tokens_gt1,
           AVG(v)                 FILTER (WHERE v > 1)         AS avg_value_usd_gt1,
           quantile_cont(v, 0.5)  FILTER (WHERE v > 1)         AS med_value_usd_gt1
    FROM w2 GROUP BY wallet_type
    """).df()
    df = df.set_index("wallet_type").reindex(WALLET_TYPES).reset_index()
    df.insert(0, "block", int(block))
    return df[AVERAGES_COLS]


# -----------------------------------------------------------------------------
# 3) Asset-count buckets
# -----------------------------------------------------------------------------
ASSET_BUCKET_COLS = ["block", "wallet_type", "bucket", "wallet_count", "wallet_pct", "value_sum_usd", "value_pct"]


def asset_buckets(con: duckdb.DuckDBPyConnection, block: int) -> pd.DataFrame:
    df = con.execute(f"""
    SELECT wallet_type, {ASSET_BUCKET_SQL} AS bucket,
           COUNT(*) AS wallet_count, SUM(v) AS value_sum_usd
    FROM w2 GROUP BY 1, 2
    """).df()
    if df.empty:
        return pd.DataFrame(columns=ASSET_BUCKET_COLS)
    tw = df.groupby("wallet_type")["wallet_count"].transform("sum")
    tv = df.groupby("wallet_type")["value_sum_usd"].transform("sum")
    df["wallet_pct"] = (100.0 * df["wallet_count"] / tw).round(2)
    df["value_pct"] = (100.0 * df["value_sum_usd"] / tv.where(tv != 0, 1e-12)).round(2)
    df["wallet_count"] = df["wallet_count"].astype("int64")
    df["value_sum_usd"] = df["value_sum_usd"].astype(float)
    df["block"] = int(block)
    order = {b: i for i, b in enumerate(ASSET_BUCKETS)}
    df["_o"] = df["bucket"].map(order)
    df["_t"] = df["wallet_type"].map({wt: i for i, wt in enumerate(WALLET_TYPES)})
    df = df.sort_values(["_t", "_o"]).drop(columns=["_o", "_t"]).reset_index(drop=True)
    return df[ASSET_BUCKET_COLS]


def print_block_summary(vdist: pd.DataFrame, avg: pd.DataFrame, abuck: pd.DataFrame, block: int) -> None:
    for _, r in vdist.iterrows():
        a = avg[avg["wallet_type"] == r["wallet_type"]].iloc[0]
        print(f"  {r['wallet_type']:<3} wallets={int(r['wallets_distinct']):,} (=0: {int(r['wallets_val_eq0']):,}, "
              f"<$1: {int(r['wallets_val_lt1']):,}) | value=${r['total_value_usd']:,.0f} | "
              f"top1%={r['top_1_share_pct']:.2f}% top5%={r['top_5_share_pct']:.2f}% top10%={r['top_10_share_pct']:.2f}% "
              f"| avg#={a['avg_num_tokens']:.3f} med#={a['med_num_tokens']:.1f} "
              f"avg$={a['avg_value_usd']:,.2f} med$={a['med_value_usd']:,.2f}")
    for wt in WALLET_TYPES:
        sub = abuck[abuck["wallet_type"] == wt]
        if sub.empty:
            continue
        print(f"\n  # assets per wallet (block={block}, type={wt})")
        for _, r in sub.iterrows():
            print(f"    {r['bucket']:>7} | {int(r['wallet_count']):12,d} | {r['wallet_pct']:6.2f}% "
                  f"| {r['value_sum_usd']:18,.2f} | {r['value_pct']:6.2f}%")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-block wallet aggregates (value distribution, averages, asset buckets) with EOA/CA split."
    )
    ap.add_argument("--parquet", help="Reconstructed portfolios: single 4_0 parquet file or 4_2 hive root.")
    ap.add_argument("--ca-tsv", help="TSV with contract addresses (first column).")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--mem", default="100GB")
    ap.add_argument("--temp-dir", default=None, help="DuckDB spill directory.")
    ap.add_argument("--blocks", type=int, nargs="*", default=None, help="Only these snapshot blocks.")
    ap.add_argument("--max-blocks", type=int, default=None, help="Only the first N blocks (smoke test).")
    ap.add_argument("--keep-weth-separate", action="store_true",
                    help="Do not fold WETH into native ETH (raw num_tokens).")
    ap.add_argument("--resume", action="store_true", help="Skip blocks already present in all output files.")
    args = ap.parse_args()

    scan = parquet_scan_target(args.parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / v for k, v in OUT_FILES.items()}
    if args.resume:
        done = done_blocks(paths)
        print(f"[resume] {len(done)} block(s) already complete in {out_dir}")
    else:
        done = set()
        for p in paths.values():
            p.unlink(missing_ok=True)

    con = get_duck(args.threads, args.mem, args.temp_dir)
    n_ca = load_ca_table(con, args.ca_tsv)
    print(f"[ca] {n_ca:,} contract addresses from {args.ca_tsv}")
    fold = not args.keep_weth_separate
    print(f"[config] parquet={scan} | fold_weth={fold}")

    blocks = args.blocks or discover_blocks(con, args.parquet, scan)
    if args.max_blocks:
        blocks = blocks[: args.max_blocks]
    todo = [b for b in blocks if b not in done]
    print(f"[blocks] {len(blocks)} found, {len(todo)} to process")

    t0 = time.time()
    for i, block in enumerate(todo, start=1):
        tb = time.time()
        print(f"\n==================== block {block} ({i}/{len(todo)}) ====================")
        n_wallets = build_block_table(con, scan, block, fold)
        if n_wallets == 0:
            print(f"[warn] block {block}: no wallets, skipped")
            continue

        vdist = value_distribution(con, block)
        avg = averages(con, block)
        abuck = asset_buckets(con, block)
        assert_bucket_totals(abuck)
        print_block_summary(vdist, avg, abuck, block)

        append_csv(vdist, paths["value_dist"], VALUE_DIST_COLS)
        append_csv(avg, paths["averages"], AVERAGES_COLS)
        append_csv(abuck, paths["asset_buckets"], ASSET_BUCKET_COLS)
        print(f"[block {block}] {n_wallets:,} wallets in {time.time() - tb:,.1f}s "
              f"(total {time.time() - t0:,.0f}s)", flush=True)

    con.close()
    print(f"\n[done] outputs in {out_dir.resolve()}:")
    for p in paths.values():
        print(f"  - {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
