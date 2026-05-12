#!/usr/bin/env python3
"""
portfolio_size_by_wealth_bucket.py

Description:
    For each Ethereum block snapshot, compute the distribution of
    portfolio size (num_tokens = number of distinct assets held) across
    wealth buckets, separately for EOAs and contract accounts (CAs).

    Wealth buckets (USD):
        0-1, 1-100, 100-1,000, 1,000-10,000, 10,000-100,000, >100,000

    Portfolio-size buckets (num_tokens):
        0, 1, 2, 3, 4, 5, 6-10, 11-20, 21-50, 51-100, >100

    Wallets listed in --ca-tsv are classified as CA; everything else as
    EOA.

Input:
    - Parquet with block_number, wallet_address, num_tokens,
      total_value_usd (--parquet).
    - TSV with contract-account addresses (first column, --ca-tsv).

Output (in --out-dir, one row per block x wallet_type x wealth_bucket):
    - portfolio_size_stats_by_wealth_wallet_type.csv
          summary stats of num_tokens: n_wallets, mean, median,
          p25, p75, p90, p95, p99, max.
    - portfolio_size_hist_by_wealth_wallet_type.csv
          counts and percentages across the portfolio-size buckets.

Usage:
    python portfolio_size_by_wealth_bucket.py \\
        --parquet /path/recon.parquet \\
        --ca-tsv  /path/contract_addresses.tsv \\
        --out-dir ./analytics_out \\
        --threads 8 --mem 12GB

    Optional: --blocks 21525891 21747950 ...   --max-blocks 12
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set

import duckdb
import pandas as pd


# -----------------------------------------------------------------------------
# Wealth buckets (USD)
# -----------------------------------------------------------------------------
WEALTH_ORDER = {
    "0–1": 0,
    "1–100": 1,
    "100–1,000": 2,
    "1,000–10,000": 3,
    "10,000–100,000": 4,
    ">100,000": 5,
}


def wealth_bucket(v: float) -> str:
    """Map a USD value to its wealth-bucket label."""
    if v is None:
        v = 0.0
    try:
        v = float(v)
    except Exception:
        v = 0.0

    if v <= 1.0:
        return "0–1"
    if v <= 100.0:
        return "1–100"
    if v <= 1_000.0:
        return "100–1,000"
    if v <= 10_000.0:
        return "1,000–10,000"
    if v <= 100_000.0:
        return "10,000–100,000"
    return ">100,000"


# -----------------------------------------------------------------------------
# Portfolio-size buckets (num_tokens)
# -----------------------------------------------------------------------------
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


def size_bucket(n: int) -> str:
    """Map a num_tokens count to its portfolio-size-bucket label."""
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    if n == 4:
        return "4"
    if n == 5:
        return "5"
    if 6 <= n <= 10:
        return "6–10"
    if 11 <= n <= 20:
        return "11–20"
    if 21 <= n <= 50:
        return "21–50"
    if 51 <= n <= 100:
        return "51–100"
    return ">100"


# -----------------------------------------------------------------------------
# Stats / IO helpers
# -----------------------------------------------------------------------------
def quantile_stats(s: pd.Series) -> dict:
    """Return n_wallets, mean, median and p25/75/90/95/99/max of `s`."""
    if s.empty:
        return {
            "n_wallets": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    s = s.astype("float64")
    q = s.quantile([0.25, 0.5, 0.75, 0.90, 0.95, 0.99], interpolation="linear")
    return {
        "n_wallets": int(s.shape[0]),
        "mean": float(s.mean()),
        "median": float(q.loc[0.50]),
        "p25": float(q.loc[0.25]),
        "p75": float(q.loc[0.75]),
        "p90": float(q.loc[0.90]),
        "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]),
        "max": float(s.max()),
    }


def append_csv(df: pd.DataFrame, path: Path, sort_cols: List[str]) -> None:
    """Append `df` to `path`, writing a header only when the file is new."""
    if df.empty:
        return
    df = df.sort_values(sort_cols)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def get_duck(threads: int, mem: str) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with the given thread count and memory cap."""
    con = duckdb.connect()
    con.execute("PRAGMA threads = ?", [int(threads)])
    con.execute("PRAGMA memory_limit = ?", [str(mem)])
    return con


def get_blocks(con: duckdb.DuckDBPyConnection, parquet: str) -> List[int]:
    """Return the sorted distinct block_numbers present in the parquet."""
    rows = con.execute(
        "SELECT DISTINCT block_number::BIGINT AS b FROM read_parquet(?) ORDER BY b",
        [parquet],
    ).fetchall()
    return [int(r[0]) for r in rows]


def normalize_address(x) -> str:
    """Return the lowercased, whitespace-stripped string form of `x`."""
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", "ignore")
        except Exception:
            return ""
    return str(x).strip().lower()


def load_contract_addresses(tsv_path: str | Path) -> Set[str]:
    """Load a set of CA addresses from the first column of a TSV file.

    Header rows are handled implicitly: any non-address token is just
    kept as-is after normalisation (the caller only uses this set for
    membership checks, so non-addresses will never match).
    """
    ca: Set[str] = set()
    p = Path(tsv_path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            addr = normalize_address(line.split("\t")[0])
            if addr:
                ca.add(addr)
    return ca


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Portfolio size distribution (num_tokens) by wealth bucket with EOA/CA split."
    )
    ap.add_argument("--parquet", required=True, help="Reconstructed portfolio parquet.")
    ap.add_argument("--ca-tsv", required=True, help="TSV with contract addresses (CA), first column is address.")
    ap.add_argument("--out-dir", default="analytics_out", help="Output directory.")
    ap.add_argument("--threads", type=int, default=8, help="DuckDB threads.")
    ap.add_argument("--mem", type=str, default="12GB", help="DuckDB memory limit, e.g. 12GB.")
    ap.add_argument("--max-blocks", type=int, default=None, help="Optional limit on number of blocks processed.")
    ap.add_argument("--blocks", nargs="*", type=int, default=None, help="Optional explicit list of blocks to process.")
    args = ap.parse_args()

    parquet = str(Path(args.parquet).resolve())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_stats = out_dir / "portfolio_size_stats_by_wealth_wallet_type.csv"
    out_hist = out_dir / "portfolio_size_hist_by_wealth_wallet_type.csv"
    out_stats.unlink(missing_ok=True)
    out_hist.unlink(missing_ok=True)

    ca_set = load_contract_addresses(args.ca_tsv)
    print(f"[info] loaded CA addresses: {len(ca_set):,}")

    con = get_duck(args.threads, args.mem)

    if args.blocks:
        blocks = [int(b) for b in args.blocks]
    else:
        blocks = get_blocks(con, parquet)
        if args.max_blocks is not None:
            blocks = blocks[: args.max_blocks]

    print(f"[info] blocks: {len(blocks):,}")

    q = """
    SELECT
      wallet_address,
      COALESCE(TRY_CAST(num_tokens AS BIGINT), 0) AS num_tokens,
      COALESCE(TRY_CAST(total_value_usd AS DOUBLE), 0.0) AS total_value_usd
    FROM read_parquet(?)
    WHERE block_number = ?
    """

    for i, block in enumerate(blocks, start=1):
        print(f"\n==== block {block} ({i}/{len(blocks)}) ====")

        df = con.execute(q, [parquet, int(block)]).df()
        if df.empty:
            print("[warn] no rows")
            continue

        df["wallet_norm"] = df["wallet_address"].map(normalize_address)
        df["wallet_type"] = df["wallet_norm"].apply(lambda a: "CA" if a in ca_set else "EOA")

        df["num_tokens"] = df["num_tokens"].astype("int64").clip(lower=0)
        df["wealth_bucket"] = df["total_value_usd"].map(wealth_bucket)
        df["size_bucket"] = df["num_tokens"].map(size_bucket)

        # Summary stats per wallet_type x wealth_bucket.
        stats_rows = []
        for (wt, wb), sub in df.groupby(["wallet_type", "wealth_bucket"], sort=False):
            st = quantile_stats(sub["num_tokens"])
            stats_rows.append(
                {
                    "block": int(block),
                    "wallet_type": wt,
                    "wealth_bucket": wb,
                    "wealth_order": WEALTH_ORDER.get(wb, 999),
                    **st,
                }
            )

        stats_df = (
            pd.DataFrame(stats_rows)
            .sort_values(["block", "wallet_type", "wealth_order"])
            .drop(columns=["wealth_order"])
        )
        append_csv(stats_df, out_stats, sort_cols=["block", "wallet_type", "wealth_bucket"])

        # Histogram buckets per wallet_type x wealth_bucket.
        hist_rows = []
        for (wt, wb), sub in df.groupby(["wallet_type", "wealth_bucket"], sort=False):
            n = int(len(sub))
            if n == 0:
                continue
            g = sub.groupby("size_bucket", as_index=False).agg(wallet_count=("size_bucket", "count"))
            g["wallet_pct"] = 100.0 * g["wallet_count"] / n
            g["size_order"] = g["size_bucket"].map(lambda x: SIZE_ORDER.get(x, 999))
            g["wallet_type"] = wt
            g["wealth_bucket"] = wb
            g["wealth_order"] = WEALTH_ORDER.get(wb, 999)
            g["block"] = int(block)
            hist_rows.append(g)

        if hist_rows:
            hist_df = pd.concat(hist_rows, ignore_index=True)
            hist_df = hist_df.sort_values(["block", "wallet_type", "wealth_order", "size_order"])
            hist_df = hist_df.drop(columns=["wealth_order", "size_order"])
            hist_df["wallet_pct"] = hist_df["wallet_pct"].round(4)
            append_csv(hist_df, out_hist, sort_cols=["block", "wallet_type", "wealth_bucket", "size_bucket"])
            hist_n = len(hist_df)
        else:
            hist_n = 0

        n_eoa = int((df["wallet_type"] == "EOA").sum())
        n_ca = int((df["wallet_type"] == "CA").sum())
        print(f"[info] wallets: EOA={n_eoa:,} CA={n_ca:,} | wrote: stats_rows={len(stats_df)} hist_rows={hist_n}")

    print(f"\n[done] wrote:\n  - {out_stats.resolve()}\n  - {out_hist.resolve()}")


if __name__ == "__main__":
    main()
