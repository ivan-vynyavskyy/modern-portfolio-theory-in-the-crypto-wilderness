#!/usr/bin/env python3
"""
analyze_holders_distribution_by_year.py

Analyze token holder counts over time (from --start-year, default 2018), aggregated by year:
  - overall avg/median holders
  - bucket distribution: 0–100, 100–1K, 1K–10K, >10K
  - same metrics by primary_category (optional join)

Input:
  1) token_inequality_per_block.csv produced by
     distribution_of_wealth/src/inequality_measures.py, with columns
     including date, holders, token_address (or token), token_name.
     Since the ETH-inclusive run the file also carries one row per
     snapshot for native ETH (pseudo-token 0xee..ee, token_name 'ETH').
     ETH is not a token and its holder count (tens of millions) would
     dominate every average, so it is EXCLUDED by default; pass
     --include-eth to keep it. The file also carries D-1/D+1 snapshots
     around a few market events; only first-of-month snapshots are used
     by default (--all-dates keeps everything).
  2) (optional) token metadata CSV with columns:
       contract_address, primary_category

Outputs (in --out-dir, default ../data next to this script; names carry
--prefix, default 'holders_', which is what generate_all_macros.py reads):
  - holders_overall_yearly_summary.csv
  - holders_overall_yearly_buckets.csv
  - holders_category_yearly_summary.csv (if categories provided)
  - holders_category_yearly_buckets.csv (if categories provided)
  No figures are produced; the LaTeX macros derived from the two overall
  CSVs live in token_holder_counts_macros.py.

Notes:
  - By default, metrics are computed over *snapshot rows* (each token-date row).
  - If you prefer token-level aggregation within each year, use --aggregation token_year
    (mean holders per token within each year, then bucket/average those values).
  - Duplicate (date, token) rows (possible after --resume runs of the
    producer) and rows with a non-numeric holders value are dropped.

Usage:
  python analyze_holders_distribution_by_year.py \\
      ../../distribution_of_wealth/data/token_inequality_per_block.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


BUCKET_LABELS = ["0–100", "100–1K", "1K–10K", ">10K"]
ETH_PSEUDO_TOKEN = "0x" + "ee" * 20  # native ETH row in the inequality CSV
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze holders distribution by year, overall and by primary_category."
    )
    p.add_argument(
        "csv",
        help="Path to token_inequality_per_block.csv (must include date, holders, token_address or token).",
    )
    p.add_argument(
        "--categories-csv",
        help="Optional: metadata CSV containing contract_address and primary_category.",
    )
    p.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="Start year (inclusive). Default: 2018 (the published data files start there).",
    )
    p.add_argument(
        "--min-holders",
        type=float,
        default=0.0,
        help="Drop rows with holders < this value. Default: 0.",
    )
    p.add_argument(
        "--aggregation",
        choices=["snapshot_rows", "token_year"],
        default="snapshot_rows",
        help=(
            "snapshot_rows: treat each token-date row as an observation (default). "
            "token_year: within each year, average holders per token first, then analyze."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for the CSVs. Default: the data/ folder next to this script.",
    )
    p.add_argument(
        "--prefix",
        default="holders_",
        help="Prefix for output file names. Default: 'holders_' (expected by generate_all_macros.py).",
    )
    p.add_argument(
        "--include-eth",
        action="store_true",
        help="Keep the native-ETH pseudo-token row (0xee..ee / 'ETH'). Excluded by default.",
    )
    p.add_argument(
        "--all-dates",
        action="store_true",
        help="Keep every snapshot date. By default only first-of-month snapshots are used, "
             "dropping the D-1/D+1 event-neighbour snapshots of the producer.",
    )
    return p.parse_args()


def ensure_out_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def detect_token_col(df: pd.DataFrame) -> str:
    if "token_address" in df.columns:
        return "token_address"
    if "token" in df.columns:
        return "token"
    raise KeyError("Expected token address column 'token_address' or 'token'.")


def clean_base_df(df: pd.DataFrame, token_col: str, include_eth: bool, all_dates: bool) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)

    df[token_col] = df[token_col].astype(str).str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["holders"] = pd.to_numeric(df["holders"], errors="coerce")

    df = df.dropna(subset=["date", "holders", token_col]).copy()
    n_bad = n0 - len(df)

    n_dup = int(df.duplicated(subset=["date", token_col]).sum())
    df = df.drop_duplicates(subset=["date", token_col], keep="last")

    n_eth = 0
    if not include_eth:
        # match on the pseudo-token address only: a real ERC-20 token with the
        # symbol 'ETH' exists in the universe and must stay in
        is_eth = df[token_col] == ETH_PSEUDO_TOKEN
        n_eth = int(is_eth.sum())
        df = df.loc[~is_eth].copy()

    n_event = 0
    if not all_dates:
        keep = df["date"].dt.day == 1
        n_event = int((~keep).sum())
        df = df.loc[keep].copy()

    print(f"[info] rows: {n0:,} | dropped: {n_bad:,} unparseable, {n_dup:,} duplicate (date, token), "
          f"{n_eth:,} native-ETH, {n_event:,} non-monthly snapshot | kept: {len(df):,}")
    return df


def filter_from_year(df: pd.DataFrame, start_year: int, min_holders: float) -> pd.DataFrame:
    start_dt = pd.Timestamp(year=start_year, month=1, day=1)
    df = df.loc[df["date"] >= start_dt].copy()
    df = df.loc[df["holders"] >= float(min_holders)].copy()
    df["year"] = df["date"].dt.year.astype(int)
    return df


def add_buckets(df: pd.DataFrame, holders_col: str = "holders") -> pd.DataFrame:
    df = df.copy()
    # Buckets: [0,100], (100,1000], (1000,10000], (10000, inf)
    bins = [-np.inf, 100, 1000, 10000, np.inf]
    df["holders_bucket"] = pd.cut(
        df[holders_col],
        bins=bins,
        labels=BUCKET_LABELS,
        right=True,
        include_lowest=True,
    )
    return df


def aggregate_snapshot_rows(df: pd.DataFrame, token_col: str) -> pd.DataFrame:
    # Each row is an observation already
    return df


def aggregate_token_year(df: pd.DataFrame, token_col: str) -> pd.DataFrame:
    # Within each year & token: average holders across snapshots in that year
    g = (
        df.groupby(["year", token_col], as_index=False)
          .agg(holders=("holders", "mean"))
    )
    return g


def yearly_summary(df: pd.DataFrame, token_col: str) -> pd.DataFrame:
    g = (
        df.groupby("year", as_index=False)
          .agg(
              n_obs=("holders", "size"),
              n_tokens=(token_col, "nunique"),
              avg_holders=("holders", "mean"),
              median_holders=("holders", "median"),
          )
          .sort_values("year")
          .reset_index(drop=True)
    )
    return g


def yearly_buckets_wide(df: pd.DataFrame) -> pd.DataFrame:
    # counts + percentages by year and bucket
    counts = (
        df.groupby(["year", "holders_bucket"], as_index=False, observed=False)
          .size()
          .rename(columns={"size": "count"})
    )
    totals = df.groupby("year", as_index=False).size().rename(columns={"size": "total"})
    out = counts.merge(totals, on="year", how="left")
    out["pct"] = out["count"] / out["total"] * 100.0

    # Ensure all buckets appear per year
    years = sorted(df["year"].unique().tolist())
    full = (
        pd.MultiIndex.from_product([years, BUCKET_LABELS], names=["year", "holders_bucket"])
        .to_frame(index=False)
    )
    out = full.merge(out, on=["year", "holders_bucket"], how="left")
    out["count"] = out["count"].fillna(0).astype(int)
    out["total"] = out["total"].fillna(0).astype(int)
    out["pct"] = out["pct"].fillna(0.0)

    # Wide format for easier plotting
    wide_pct = out.pivot(index="year", columns="holders_bucket", values="pct").reset_index()
    wide_cnt = out.pivot(index="year", columns="holders_bucket", values="count").reset_index()

    # Merge wide pct + wide count with suffixes
    wide_pct.columns = ["year"] + [f"pct_{c}" for c in wide_pct.columns[1:]]
    wide_cnt.columns = ["year"] + [f"count_{c}" for c in wide_cnt.columns[1:]]
    wide = wide_pct.merge(wide_cnt, on="year", how="left").sort_values("year").reset_index(drop=True)
    return wide


def join_categories(df: pd.DataFrame, token_col: str, categories_csv: str) -> pd.DataFrame:
    meta = pd.read_csv(categories_csv, dtype=str)
    if "contract_address" not in meta.columns or "primary_category" not in meta.columns:
        raise KeyError("categories-csv must contain 'contract_address' and 'primary_category' columns.")

    meta["contract_address"] = meta["contract_address"].astype(str).str.lower().str.strip()
    meta["primary_category"] = (
        meta["primary_category"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
    )

    out = df.merge(
        meta[["contract_address", "primary_category"]],
        how="left",
        left_on=token_col,
        right_on="contract_address",
    )
    out = out.drop(columns=["contract_address"], errors="ignore")
    out = out.dropna(subset=["primary_category"]).copy()
    return out


def category_yearly_summary(df_cat: pd.DataFrame, token_col: str) -> pd.DataFrame:
    g = (
        df_cat.groupby(["year", "primary_category"], as_index=False)
              .agg(
                  n_obs=("holders", "size"),
                  n_tokens=(token_col, "nunique"),
                  avg_holders=("holders", "mean"),
                  median_holders=("holders", "median"),
              )
              .sort_values(["year", "primary_category"])
              .reset_index(drop=True)
    )
    return g


def category_yearly_buckets(df_cat: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df_cat.groupby(["year", "primary_category", "holders_bucket"], as_index=False, observed=False)
              .size()
              .rename(columns={"size": "count"})
    )
    totals = (
        df_cat.groupby(["year", "primary_category"], as_index=False)
              .size()
              .rename(columns={"size": "total"})
    )
    out = counts.merge(totals, on=["year", "primary_category"], how="left")
    out["pct"] = out["count"] / out["total"] * 100.0
    return out


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")

    # Load
    df = pd.read_csv(args.csv, low_memory=False)
    token_col = detect_token_col(df)
    df = clean_base_df(df, token_col, include_eth=args.include_eth, all_dates=args.all_dates)
    df = filter_from_year(df, args.start_year, args.min_holders)

    if df.empty:
        raise RuntimeError(f"No data left after filtering from year {args.start_year} and min_holders {args.min_holders}.")

    # Aggregate choice
    if args.aggregation == "snapshot_rows":
        obs = aggregate_snapshot_rows(df, token_col)
    else:
        obs = aggregate_token_year(df, token_col)

    # Buckets + overall summaries
    obs = add_buckets(obs, holders_col="holders")
    overall_sum = yearly_summary(obs, token_col=token_col)
    overall_buckets = yearly_buckets_wide(obs)

    # Save overall
    px = args.prefix
    overall_sum_path = out_dir / f"{px}overall_yearly_summary.csv"
    overall_bucket_path = out_dir / f"{px}overall_yearly_buckets.csv"
    overall_sum.to_csv(overall_sum_path, index=False)
    overall_buckets.to_csv(overall_bucket_path, index=False)

    print(f"[ok] Wrote: {overall_sum_path}")
    print(f"[ok] Wrote: {overall_bucket_path}")

    # Category join + category summaries (optional)
    if args.categories_csv:
        if not os.path.exists(args.categories_csv):
            raise FileNotFoundError(f"Categories CSV not found: {args.categories_csv}")

        df_cat = join_categories(obs, token_col=token_col, categories_csv=args.categories_csv)
        if df_cat.empty:
            raise RuntimeError("After joining categories, no rows have a non-null primary_category.")

        cat_sum = category_yearly_summary(df_cat, token_col=token_col)
        cat_bucket_long = category_yearly_buckets(df_cat)

        cat_sum_path = out_dir / f"{px}category_yearly_summary.csv"
        cat_bucket_path = out_dir / f"{px}category_yearly_buckets.csv"
        cat_sum.to_csv(cat_sum_path, index=False)
        cat_bucket_long.to_csv(cat_bucket_path, index=False)

        print(f"[ok] Wrote: {cat_sum_path}")
        print(f"[ok] Wrote: {cat_bucket_path}")
    else:
        print("[warn] --categories-csv not provided, skipping primary_category analysis.")


if __name__ == "__main__":
    main()
