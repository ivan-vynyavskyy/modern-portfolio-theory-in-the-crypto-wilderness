#!/usr/bin/env python3
"""
inequality_measures.py

Description:
    Per-token holder-concentration metrics for every reconstruction
    snapshot. The snapshot list is read from --snapshots-csv, normally the
    _manifest.csv that 4_2_partition_holdings.py writes into the holdings
    root, so the metrics cover exactly the blocks for which portfolios were
    reconstructed (no separately curated dates). Script conversion of
    inequality_measures.ipynb.

    For each (date, token_address) pair, in one DuckDB query over the
    structured transfers dump (token_address=*/ hive partitions with
    address/value/block columns):

    - holders          — number of positive-balance holders
    - gini, hhi        — classical inequality / concentration measures
    - top1/10/100      — share held by the 1st / 10th / 100th largest holders
    - top1p/5p/10p     — share held by the top 1% / 5% / 10% holders
    - nakamoto         — minimum holders needed to control >= 50% of supply
    - supply           — decimals-adjusted total supply (sum of positive
                         holder balances)

    Native ETH: the ETH pseudo-token partition (token_address=0xee..ee,
    present in the combined eth-erc20 transfers dump) is included by
    default — it is injected into the whitelist as 'ETH' since the token
    status CSV only covers ERC-20 contracts. Its `supply` rows are what
    5_0_portfolio_backtest_pipeline.py needs in --supply-csv for the
    Market-Cap Weight strategy on ETH-holding wallets. Disable with
    --no-include-eth to reproduce the old ERC-20-only output.

    The beacon deposit contract (0x00000000219ab540356cbb839cbe05303d7705fa)
    is excluded as an ETH holder: it custodies all staked ETH for validators
    whose claims live on the consensus layer. Consequently the ETH `supply`
    column is the execution-layer balance sum WITHOUT the deposit contract
    and is not ETH's circulating supply; 5_0 must take ETH supply from
    --eth-supply-csv (Coingecko), which overrides this file for ETH.

Snapshots (--snapshots-csv):
    Any CSV with a block column (block_number or block) and, optionally, a
    date column (date, bound_date or block_time_utc). Without a date column
    the dates are looked up in --block-to-date-csv; with both, the two are
    cross-checked. Duplicates are dropped and the list is sorted by block.

Output:
    CSV with one row per token per snapshot (columns: date, block,
    token_address, token_name, holders, gini, hhi, top1, top10, top100,
    top1p, top5p, top10p, nakamoto, supply), streamed snapshot by
    snapshot; --resume skips (date, block) pairs already in the file.
    After computing (or with --summary-only on an existing file) a summary
    of the results is printed: coverage, per-year holder and concentration
    statistics, the native-ETH rows, sanity checks, and the most / least
    concentrated tokens at the last snapshot. No plots are produced.

Usage (cluster):
    python inequality_measures.py \\
        --data-root /path/to/eth-erc20-transfers-20251208-structured \\
        --status-csv /path/to/data/eth_tokens_status_new_new_sorted.csv \\
        --snapshots-csv /path/to/recon-eth-erc20-partitioned/_manifest.csv \\
        --decimals-cache /path/to/data/token_decimals_cache.json \\
        --out-csv token_inequality_per_block.csv \\
        --threads 32 --mem 60GB \\
        --temp-dir /path/to/duckdb_tmp \\
        --resume

    Summary only (from an existing CSV, no computation):
    python inequality_measures.py --summary-only --out-csv token_inequality_per_block.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import duckdb
import pandas as pd


ETH_PSEUDO_TOKEN = "0x" + "ee" * 20  # native ETH inside the combined dump

# The beacon-chain deposit contract holds every ETH ever staked (73 M ETH by
# December 2025) on behalf of validators whose claims live on the consensus
# layer. That is custody, not an account's wealth, so the address is excluded
# as an ETH holder from every value-based statistic. Token universe unchanged.
BEACON_DEPOSIT_CONTRACT = "0x00000000219ab540356cbb839cbe05303d7705fa"
ETH_TOKEN_NAME = "ETH"

# Parquet schema columns of the transfers dump
ADDR_COL = "address"
VAL_COL = "value"
BLK_COL = "block"

_id_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_ident(name: str) -> str:
    if not _id_re.match(name):
        raise ValueError(f"Illegal identifier: {name!r}")
    return name


# -----------------------------------------------------------------------------
# Snapshot dates and block resolution
# -----------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: List[str], what: str, path: str) -> Optional[str]:
    col = next((c for c in candidates if c in df.columns), None)
    if col is None and what == "block":
        raise ValueError(f"No block column ({candidates}) in {path}. Got {list(df.columns)}")
    return col


def load_snapshots(snapshots_csv: str, block_to_date_csv: Optional[str] = None) -> pd.DataFrame:
    """Read and check the (date, block) snapshot list.

    `snapshots_csv` needs a block column and may carry a date column (the
    4_2 _manifest.csv has both). Missing dates are resolved through
    `block_to_date_csv`; if both sources carry a date they are cross-checked.
    Returns a DataFrame with columns ['date' (YYYY-MM-DD), 'block'] sorted
    by block, duplicates removed.
    """
    snap = pd.read_csv(snapshots_csv)
    bcol = _find_col(snap, ["block_number", "block"], "block", snapshots_csv)
    dcol = _find_col(snap, ["date", "bound_date", "block_time_utc"], "date", snapshots_csv)

    out = pd.DataFrame({"block": pd.to_numeric(snap[bcol], errors="coerce")})
    out["date"] = pd.to_datetime(snap[dcol], errors="coerce").dt.normalize() if dcol else pd.NaT
    n_bad = int(out["block"].isna().sum())
    out = out.dropna(subset=["block"]).astype({"block": "int64"})

    if block_to_date_csv:
        m = pd.read_csv(block_to_date_csv)
        mb = _find_col(m, ["block_number", "block"], "block", block_to_date_csv)
        md = _find_col(m, ["date", "bound_date", "block_time_utc"], "date", block_to_date_csv)
        if md is None:
            raise ValueError(f"No date column in {block_to_date_csv}. Got {list(m.columns)}")
        m = pd.DataFrame({"block": pd.to_numeric(m[mb], errors="coerce"),
                          "map_date": pd.to_datetime(m[md], errors="coerce").dt.normalize()})
        m = m.dropna().astype({"block": "int64"}).drop_duplicates("block")
        out = out.merge(m, on="block", how="left")
        if dcol:
            both = out["date"].notna() & out["map_date"].notna()
            n_mismatch = int((out.loc[both, "date"] != out.loc[both, "map_date"]).sum())
            if n_mismatch:
                print(f"[warn] {n_mismatch} snapshot(s) where the date in {snapshots_csv} differs "
                      f"from {block_to_date_csv}; keeping the snapshots file's date")
        out["date"] = out["date"].fillna(out["map_date"])
        out = out.drop(columns=["map_date"])

    n_nodate = int(out["date"].isna().sum())
    if n_nodate:
        if dcol is None and not block_to_date_csv:
            raise ValueError(f"{snapshots_csv} has no date column; pass --block-to-date-csv to resolve dates")
        print(f"[warn] {n_nodate} snapshot(s) without a resolvable date are skipped: "
              f"{out.loc[out['date'].isna(), 'block'].tolist()[:10]}")
        out = out.dropna(subset=["date"])

    n_before = len(out)
    out = out.drop_duplicates(subset=["block"]).sort_values("block").reset_index(drop=True)
    n_dup = n_before - len(out)
    dup_dates = int(out["date"].duplicated().sum())
    if dup_dates:
        print(f"[warn] {dup_dates} date(s) map to more than one block; all blocks are kept")

    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    print(f"[snapshots] {len(out)} from {snapshots_csv}"
          f" | dropped: {n_bad} unparseable block(s), {n_dup} duplicate block(s)"
          f" | first: {out['date'].iloc[0]} (block {out['block'].iloc[0]:,})"
          f" | last: {out['date'].iloc[-1]} (block {out['block'].iloc[-1]:,})")
    return out[["date", "block"]]


def load_decimals_cache(path: str) -> Dict[str, int]:
    """Load {token_address: decimals} from a JSON cache file, if present."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, int] = {}
    for k, v in data.items():
        try:
            out[str(k).strip().lower()] = int(v)
        except Exception:
            pass
    return out


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------

def compute_metrics_for_block_duckdb(
    max_block: int,
    date_str: str,
    *,
    data_root: str,
    status_csv: str,
    decimals_map: Dict[str, int],
    threads: int,
    mem: str,
    temp_dir: Optional[str],
    include_eth: bool,
    addr_col: str = ADDR_COL,
    val_col: str = VAL_COL,
    blk_col: str = BLK_COL,
    parquet_glob: Optional[str] = None,
) -> pd.DataFrame:
    """Compute per-token concentration metrics at `max_block` in one DuckDB query.

    Output columns:
        date, block, token_address, token_name, holders, gini, hhi,
        top1, top10, top100, top1p, top5p, top10p, nakamoto, supply
    """
    addr_col = _safe_ident(addr_col)
    val_col = _safe_ident(val_col)
    blk_col = _safe_ident(blk_col)

    if parquet_glob is None:
        parquet_glob = f"{data_root}/token_address=*/**/*.parquet"

    if decimals_map:
        dec_df = pd.DataFrame(
            {"token_address": list(decimals_map.keys()), "decimals": list(decimals_map.values())}
        )
    else:
        # typed, otherwise an empty frame defaults to float64 and lower() fails
        dec_df = pd.DataFrame({"token_address": pd.Series(dtype="string"), "decimals": pd.Series(dtype="int64")})

    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA threads={int(threads)};")
        con.execute(f"PRAGMA memory_limit='{mem}';")
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            con.execute(f"PRAGMA temp_directory='{temp_dir}';")

        # Whitelist + labels inside DuckDB (fast, no pandas filtering).
        con.execute(f"""
            CREATE TEMP TABLE wl AS
            SELECT
                lower(trim(contract_address)) AS token_address,
                CASE
                  WHEN coalesce(nullif(trim(symbol), ''), '') != '' THEN trim(symbol)
                  ELSE trim(name)
                END AS token_name
            FROM read_csv_auto('{status_csv}', all_varchar=true)
            WHERE lower(trim(local_status)) = 'pass'
              AND contract_address IS NOT NULL
              AND length(trim(contract_address)) = 42
              AND left(lower(trim(contract_address)), 2) = '0x';
        """)

        if include_eth:
            # Native ETH is not an ERC-20, so the status CSV never lists it;
            # inject the pseudo-token so its partition passes the whitelist
            # join. Its decimals fall through to the COALESCE(..., 18) below,
            # which is correct for ETH.
            con.execute(f"""
                INSERT INTO wl
                SELECT '{ETH_PSEUDO_TOKEN}', '{ETH_TOKEN_NAME}'
                WHERE NOT EXISTS (
                    SELECT 1 FROM wl WHERE token_address = '{ETH_PSEUDO_TOKEN}'
                );
            """)

        con.register("dec_df", dec_df)
        con.execute("""
            CREATE TEMP TABLE dec AS
            SELECT lower(token_address) AS token_address, CAST(decimals AS INTEGER) AS decimals
            FROM dec_df;
        """)

        sql = f"""
        WITH raw AS (
            SELECT
                lower(token_address) AS token_address,
                lower({addr_col})    AS address,
                TRY_CAST({val_col} AS DOUBLE) AS v,
                TRY_CAST({blk_col} AS BIGINT) AS blk
            FROM parquet_scan('{parquet_glob}', hive_partitioning=1, union_by_name=1, hive_types_autocast=0)
            WHERE TRY_CAST({val_col} AS DOUBLE) IS NOT NULL
              AND TRY_CAST({blk_col} AS BIGINT) IS NOT NULL
              AND TRY_CAST({blk_col} AS BIGINT) <= {int(max_block)}
        ),
        raw_w AS (
            SELECT r.*
            FROM raw r
            JOIN wl USING (token_address)
        ),
        -- sum per address, skip addresses where SUM < 0
        bal AS (
            SELECT
                token_address,
                address,
                SUM(v) AS balance
            FROM raw_w
            GROUP BY token_address, address
            HAVING SUM(v) >= 0
        ),
        -- only positive holders for concentration stats
        holders AS (
            SELECT token_address, address, balance
            FROM bal
            WHERE balance > 0
              -- custody, not wealth: the beacon deposit contract is not an ETH holder
              AND NOT (token_address = '{ETH_PSEUDO_TOKEN}' AND address = '{BEACON_DEPOSIT_CONTRACT}')
        ),
        stats AS (
            SELECT
                token_address,
                COUNT(*)::BIGINT AS n_holders,
                SUM(balance)     AS supply_raw
            FROM holders
            GROUP BY token_address
        ),
        ranked AS (
            SELECT
                h.token_address,
                h.address,
                h.balance,
                s.n_holders,
                s.supply_raw,
                ROW_NUMBER() OVER (PARTITION BY h.token_address ORDER BY h.balance DESC) AS rn_desc,
                ROW_NUMBER() OVER (PARTITION BY h.token_address ORDER BY h.balance ASC)  AS rn_asc,
                SUM(h.balance) OVER (
                    PARTITION BY h.token_address
                    ORDER BY h.balance DESC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) / s.supply_raw AS cum_share_desc
            FROM holders h
            JOIN stats s USING (token_address)
        ),
        metrics AS (
            SELECT
                token_address,
                MAX(n_holders) AS holders,
                MAX(supply_raw) AS supply_raw,

                -- HHI
                SUM( (balance / supply_raw) * (balance / supply_raw) ) AS hhi,

                -- Top-K shares
                SUM(CASE WHEN rn_desc <= 1   THEN balance ELSE 0 END) / MAX(supply_raw) AS top1,
                SUM(CASE WHEN rn_desc <= 10  THEN balance ELSE 0 END) / MAX(supply_raw) AS top10,
                SUM(CASE WHEN rn_desc <= 100 THEN balance ELSE 0 END) / MAX(supply_raw) AS top100,

                -- Top-% shares (k = ceil(pct * n_holders))
                SUM(CASE WHEN rn_desc <= CEIL(0.01 * n_holders) THEN balance ELSE 0 END) / MAX(supply_raw) AS top1p,
                SUM(CASE WHEN rn_desc <= CEIL(0.05 * n_holders) THEN balance ELSE 0 END) / MAX(supply_raw) AS top5p,
                SUM(CASE WHEN rn_desc <= CEIL(0.10 * n_holders) THEN balance ELSE 0 END) / MAX(supply_raw) AS top10p,

                -- Nakamoto(50%)
                MIN(CASE WHEN cum_share_desc >= 0.5 THEN rn_desc ELSE NULL END) AS nakamoto,

                -- Gini (ascending ranks)
                (
                    (2.0 * SUM(rn_asc * balance) / (MAX(n_holders) * MAX(supply_raw)))
                    - (MAX(n_holders) + 1.0) / MAX(n_holders)
                ) AS gini
            FROM ranked
            GROUP BY token_address
        )
        SELECT
            '{date_str}' AS date,
            {int(max_block)} AS block,
            m.token_address AS token_address,
            wl.token_name   AS token_name,
            m.holders       AS holders,
            m.gini          AS gini,
            m.hhi           AS hhi,
            m.top1          AS top1,
            m.top10         AS top10,
            m.top100        AS top100,
            m.top1p         AS top1p,
            m.top5p         AS top5p,
            m.top10p        AS top10p,
            m.nakamoto      AS nakamoto,
            (m.supply_raw / POWER(10.0, COALESCE(dec.decimals, 18))) AS supply
        FROM metrics m
        JOIN wl USING (token_address)
        LEFT JOIN dec USING (token_address)
        WHERE m.holders > 1
        ;
        """

        return con.execute(sql).df()

    finally:
        con.close()


OUTPUT_COLUMNS = [
    "date", "block", "token_address", "token_name", "holders",
    "gini", "hhi", "top1", "top10", "top100", "top1p", "top5p", "top10p",
    "nakamoto", "supply",
]


def compute_metrics_for_all_snapshots_to_csv(
    date_block_df: pd.DataFrame,
    *,
    out_csv: str,
    data_root: str,
    status_csv: str,
    decimals_map: Dict[str, int],
    threads: int,
    mem: str,
    temp_dir: Optional[str],
    include_eth: bool,
    resume: bool,
) -> None:
    """Run compute_metrics_for_block_duckdb over every (date, block) pair.

    Writes results incrementally to `out_csv` (one pass per snapshot).
    With --resume, (date, block) pairs already present in the output are
    skipped; without it, an existing output file is removed first.
    """
    if date_block_df.empty:
        print("[warn] No (date, block) pairs. Nothing to do.")
        return

    date_block_df = date_block_df.drop_duplicates(subset=["date", "block"]).sort_values(["date", "block"])

    done_pairs: set = set()
    if os.path.exists(out_csv):
        if resume:
            try:
                prev = pd.read_csv(out_csv, usecols=["date", "block"])
                done_pairs = set(zip(prev["date"].astype(str), prev["block"].astype(int)))
                print(f"[resume] {len(done_pairs)} (date, block) pairs already in {out_csv}")
            except Exception as e:
                print(f"[resume] could not read existing output ({e}); recomputing all snapshots")
        else:
            print(f"[info] Removing existing {out_csv}")
            os.remove(out_csv)

    header_written = os.path.exists(out_csv)
    n_total = len(date_block_df)

    for i, row in enumerate(date_block_df.itertuples(index=False), start=1):
        date_str = str(row.date)
        blk = int(row.block)

        if (date_str, blk) in done_pairs:
            print(f"[{i}/{n_total}] skip {date_str} | block {blk} (already done)")
            continue

        print(f"\n=== [{i}/{n_total}] {date_str} | block {blk} ===")
        df_b = compute_metrics_for_block_duckdb(
            max_block=blk,
            date_str=date_str,
            data_root=data_root,
            status_csv=status_csv,
            decimals_map=decimals_map,
            threads=threads,
            mem=mem,
            temp_dir=temp_dir,
            include_eth=include_eth,
        )

        if df_b.empty:
            print("  -> no rows (maybe no whitelist tokens found in shards?)")
            continue

        df_b = df_b[OUTPUT_COLUMNS]

        mode = "a" if header_written else "w"
        df_b.to_csv(out_csv, mode=mode, index=False, header=not header_written)
        header_written = True

        if include_eth:
            eth_rows = df_b[df_b["token_address"] == ETH_PSEUDO_TOKEN]
            if eth_rows.empty:
                print("  [warn] ETH pseudo-token produced NO row for this snapshot "
                      "(is the 0xee..ee partition present under the data root?)")
            else:
                r = eth_rows.iloc[0]
                print(f"  [eth] holders={int(r['holders']):,} supply={r['supply']:,.0f} ETH "
                      f"gini={r['gini']:.4f}")
        print(f"  -> appended {len(df_b)} token rows")

        del df_b

    print(f"\nDone. Wrote: {out_csv}")


# -----------------------------------------------------------------------------
# Summary of the results file (printed; no plots)
# -----------------------------------------------------------------------------

def print_summary(out_csv: str) -> None:
    """Print coverage, per-year statistics, the native-ETH rows, sanity
    checks, and the most / least concentrated tokens at the last snapshot."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    df = pd.read_csv(out_csv, dtype={"token_address": str}, low_memory=False)
    n_raw = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("block", "holders", "gini", "hhi", "top1", "top10", "top100",
              "top1p", "top5p", "top10p", "nakamoto", "supply"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "block", "token_address"])
    df["token_address"] = df["token_address"].str.lower().str.strip()

    is_eth = df["token_address"] == ETH_PSEUDO_TOKEN
    tok = df[~is_eth]
    eth = df[is_eth].sort_values("date")

    print("\n" + "=" * 78)
    print(f"SUMMARY of {out_csv}")
    print("=" * 78)
    print(f"rows: {n_raw:,} (dropped {n_raw - len(df):,} unparseable)"
          f" | snapshots: {df['block'].nunique()}"
          f" | {df['date'].min().date()} -> {df['date'].max().date()}"
          f" | ERC-20 tokens: {tok['token_address'].nunique():,}"
          f" | native-ETH rows: {len(eth)}")

    per_snap = tok.groupby("block").agg(date=("date", "first"), tokens=("token_address", "nunique"))
    med_tokens = per_snap["tokens"].median()
    print(f"tokens per snapshot: min {per_snap['tokens'].min()} | median {med_tokens:.0f} | max {per_snap['tokens'].max()}")

    # per-year statistics (ERC-20 tokens only; ETH is reported separately)
    tok = tok.assign(year=tok["date"].dt.year)
    yearly = tok.groupby("year").agg(
        snapshots=("block", "nunique"), tokens=("token_address", "nunique"),
        rows=("holders", "size"), mean_holders=("holders", "mean"),
        median_holders=("holders", "median"), mean_gini=("gini", "mean"),
        median_gini=("gini", "median"), mean_top1p=("top1p", "mean"),
        median_nakamoto=("nakamoto", "median"),
    )
    yearly["mean_holders"] = yearly["mean_holders"].round(0)
    yearly["mean_top1p"] = (yearly["mean_top1p"] * 100).round(1)
    print("\nPer year (ERC-20 tokens; mean_top1p in %):")
    print(yearly.round(3).to_string())

    # native ETH
    if not eth.empty:
        first, last = eth.iloc[0], eth.iloc[-1]
        print("\nNative ETH (pseudo-token 0xee..ee):")
        print(f"  first {first['date'].date()}: holders {int(first['holders']):,} | supply {first['supply']:,.0f} ETH"
              f" | gini {first['gini']:.4f} | top1p {first['top1p']*100:.1f}%")
        print(f"  last  {last['date'].date()}: holders {int(last['holders']):,} | supply {last['supply']:,.0f} ETH"
              f" | gini {last['gini']:.4f} | top1p {last['top1p']*100:.1f}%")
        missing = per_snap.index.difference(eth["block"])
        if len(missing):
            print(f"  [check] ETH row missing for {len(missing)} snapshot(s): {list(missing)[:10]}")
    else:
        print("\nNative ETH: no rows (ERC-20-only run, or the 0xee..ee partition was absent)")

    # sanity checks
    print("\nSanity checks:")
    n_dup = int(df.duplicated(subset=["block", "token_address"]).sum())
    print(f"  duplicate (block, token) rows: {n_dup:,}")
    thin = per_snap[per_snap["tokens"] < 0.5 * med_tokens]
    print(f"  snapshots with < 50% of the median token count: {len(thin)}"
          + (f" -> {[(d.date().isoformat(), int(b)) for b, d in zip(thin.index, thin['date'])][:10]}" if len(thin) else ""))
    for c, lo, hi in (("gini", 0, 1), ("hhi", 0, 1), ("top1", 0, 1), ("top1p", 0, 1), ("top10p", 0, 1)):
        if c in df.columns:
            bad = int(((df[c] < lo) | (df[c] > hi)).sum())
            nan = int(df[c].isna().sum())
            print(f"  {c}: {bad:,} out of [{lo}, {hi}], {nan:,} NaN")
    if "supply" in df.columns:
        print(f"  supply <= 0: {int((df['supply'] <= 0).sum()):,} | supply NaN: {int(df['supply'].isna().sum()):,}")
    print(f"  holders <= 1 (should be 0, filtered in SQL): {int((df['holders'] <= 1).sum()):,}")

    # last snapshot
    last_block = int(df["block"].max())
    snap = tok[tok["block"] == last_block]
    cols = ["token_name", "token_address", "holders", "gini", "hhi", "top1p", "top10p", "nakamoto"]
    print(f"\nLast snapshot {df.loc[df['block'] == last_block, 'date'].iloc[0].date()} (block {last_block:,}), {len(snap):,} ERC-20 tokens")
    print("\n  Top-10 tokens by holders:")
    print(snap.nlargest(10, "holders")[cols].to_string(index=False))
    print("\n  Top-10 most concentrated (by Gini):")
    print(snap.nlargest(10, "gini")[cols].to_string(index=False))
    print("\n  Top-10 least concentrated (by Gini):")
    print(snap.nsmallest(10, "gini")[cols].to_string(index=False))
    print("=" * 78)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Per-token holder-concentration metrics per snapshot.")
    ap.add_argument("--data-root", help="Transfers dump root (token_address=*/ hive partitions).")
    ap.add_argument("--status-csv", help="Token status CSV (whitelist: local_status='pass').")
    ap.add_argument("--snapshots-csv",
                    help="Snapshot list: CSV with a block column and optionally a date column "
                         "(e.g. the _manifest.csv of the 4_2 holdings root).")
    ap.add_argument("--block-to-date-csv", default=None,
                    help="Optional blocks_closest_to_midnight.csv: resolves dates for a snapshots "
                         "file without a date column, or cross-checks the dates it has.")
    ap.add_argument("--decimals-cache", default="", help="Optional token_decimals_cache.json")
    ap.add_argument("--out-csv", default="token_inequality_per_block.csv")
    ap.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--mem", type=str, default="32GB", help="DuckDB memory limit.")
    ap.add_argument("--temp-dir", default=None, help="DuckDB spill dir (recommended on cluster).")
    ap.add_argument("--no-include-eth", action="store_true",
                    help="Exclude the native-ETH pseudo-token (reproduces the old ERC-20-only output).")
    ap.add_argument("--max-snapshots", type=int, default=None,
                    help="Only the first N (date, block) pairs — for smoke tests.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (date, block) pairs already present in --out-csv.")
    ap.add_argument("--summary-only", action="store_true",
                    help="Skip computation; only print the summary of an existing --out-csv.")
    args = ap.parse_args()

    if args.summary_only:
        print_summary(args.out_csv)
        return 0

    for req in ("data_root", "status_csv", "snapshots_csv"):
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} is required unless --summary-only")

    date_block_df = load_snapshots(args.snapshots_csv, args.block_to_date_csv)
    if args.max_snapshots is not None:
        date_block_df = date_block_df.head(int(args.max_snapshots))
        print(f"[info] limited to first {len(date_block_df)} snapshots (--max-snapshots)")

    decimals_map = load_decimals_cache(args.decimals_cache)
    print("Decimals loaded:", len(decimals_map))

    compute_metrics_for_all_snapshots_to_csv(
        date_block_df=date_block_df,
        out_csv=args.out_csv,
        data_root=args.data_root,
        status_csv=args.status_csv,
        decimals_map=decimals_map,
        threads=args.threads,
        mem=args.mem,
        temp_dir=args.temp_dir,
        include_eth=not args.no_include_eth,
        resume=args.resume,
    )

    if os.path.exists(args.out_csv):
        print_summary(args.out_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
