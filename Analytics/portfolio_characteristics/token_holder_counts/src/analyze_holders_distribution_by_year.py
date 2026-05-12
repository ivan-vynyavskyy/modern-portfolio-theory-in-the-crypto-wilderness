#!/usr/bin/env python3
"""
analyze_holders_distribution_by_year.py

Analyze token holder counts over time (from 2020), aggregated by year:
  - overall avg/median holders
  - bucket distribution: 0–100, 100–1K, 1K–10K, >10K
  - same metrics by primary_category (optional join)

Input:
  1) token_inequality_per_block.csv with columns including:
       date, holders, token_address (or token)
  2) (optional) token metadata CSV with columns:
       contract_address, primary_category

Outputs (in --out-dir):
  - overall_yearly_summary.csv
  - overall_yearly_buckets.csv
  - category_yearly_summary.csv (if categories provided)
  - category_yearly_buckets.csv (if categories provided)
  - plots:
      overall_bucket_distribution.png
      overall_avg_holders.png
      category_bucket_distribution_topK.png (if categories provided)
      category_avg_holders_topK.png (if categories provided)

Notes:
  - By default, metrics are computed over *snapshot rows* (each token-date row).
  - If you prefer token-level aggregation within each year, use --aggregation token_year
    (mean holders per token within each year, then bucket/average those values).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BUCKET_LABELS = ["0–100", "100–1K", "1K–10K", ">10K"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze holders distribution by year (from 2020), overall and by primary_category."
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
        default=2020,
        help="Start year (inclusive). Default: 2020.",
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
        "--top-k-categories",
        type=int,
        default=10,
        help="For category plots: plot only top K categories by row count. Default: 10.",
    )
    p.add_argument(
        "--out-dir",
        default="holders_analysis_out",
        help="Output directory for CSVs/plots. Default: holders_analysis_out",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots interactively (useful on headless servers).",
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


def clean_base_df(df: pd.DataFrame, token_col: str) -> pd.DataFrame:
    df = df.copy()

    df[token_col] = df[token_col].astype(str).str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["holders"] = pd.to_numeric(df["holders"], errors="coerce")

    df = df.dropna(subset=["date", "holders", token_col]).copy()
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
        df.groupby(["year", "holders_bucket"], as_index=False)
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
        df_cat.groupby(["year", "primary_category", "holders_bucket"], as_index=False)
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


def plot_overall_bucket_distribution(wide: pd.DataFrame, out_path: Path) -> None:
    years = wide["year"].astype(int).tolist()
    pct_cols = [f"pct_{b}" for b in BUCKET_LABELS]

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(years), dtype=float)

    for col, label in zip(pct_cols, BUCKET_LABELS):
        vals = wide[col].to_numpy(dtype=float)
        ax.bar(years, vals, bottom=bottom, label=label)
        bottom += vals

    ax.set_title("Holders bucket distribution by year (percent)", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Percent of observations (%)", fontsize=12)
    ax.set_xticks(years)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def plot_overall_avg_holders(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        summary["year"],
        summary["avg_holders"],
        linestyle="-",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Average holders",
    )
    ax.plot(
        summary["year"],
        summary["median_holders"],
        linestyle="-",
        linewidth=2,
        marker="s",
        markersize=4,
        label="Median holders",
    )
    ax.set_title("Average / Median holders by year", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Holders", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def plot_category_topk_bucket_distribution(
    df_cat_bucket: pd.DataFrame, top_categories: List[str], out_path: Path
) -> None:
    # Plot: for each bucket, line over years for the combined topK categories (separate lines)
    # To keep it readable: we plot "% in >10K bucket" (often most interesting) + maybe "0–100"
    # Here: create two plots in one figure is not allowed by your style preference, so do just one:
    # We'll plot the >10K bucket pct over time for topK categories.
    target_bucket = ">10K"
    sub = df_cat_bucket.loc[
        (df_cat_bucket["primary_category"].isin(top_categories)) &
        (df_cat_bucket["holders_bucket"] == target_bucket)
    ].copy()

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    for cat in top_categories:
        s = sub.loc[sub["primary_category"] == cat].sort_values("year")
        if s.empty:
            continue
        ax.plot(
            s["year"],
            s["pct"],
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=3,
            label=cat,
        )

    ax.set_title(f"Share of observations in '{target_bucket}' bucket (top categories)", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Percent within category (%)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def plot_category_topk_avg_holders(
    df_cat_summary: pd.DataFrame, top_categories: List[str], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for cat in top_categories:
        s = df_cat_summary.loc[df_cat_summary["primary_category"] == cat].sort_values("year")
        if s.empty:
            continue
        ax.plot(
            s["year"],
            s["avg_holders"],
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=3,
            label=cat,
        )

    ax.set_title("Average holders by year (top categories)", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Average holders", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")

    # Load
    df = pd.read_csv(args.csv, low_memory=False)
    token_col = detect_token_col(df)
    df = clean_base_df(df, token_col)
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
    overall_sum_path = out_dir / "overall_yearly_summary.csv"
    overall_bucket_path = out_dir / "overall_yearly_buckets.csv"
    overall_sum.to_csv(overall_sum_path, index=False)
    overall_buckets.to_csv(overall_bucket_path, index=False)

    print(f"[ok] Wrote: {overall_sum_path}")
    print(f"[ok] Wrote: {overall_bucket_path}")

    # Plots overall
    plot_overall_bucket_distribution(overall_buckets, out_dir / "overall_bucket_distribution.png")
    plot_overall_avg_holders(overall_sum, out_dir / "overall_avg_holders.png")
    print(f"[ok] Saved plots in: {out_dir}")

    # Category join + category summaries (optional)
    if args.categories_csv:
        if not os.path.exists(args.categories_csv):
            raise FileNotFoundError(f"Categories CSV not found: {args.categories_csv}")

        df_cat = join_categories(obs, token_col=token_col, categories_csv=args.categories_csv)
        if df_cat.empty:
            raise RuntimeError("After joining categories, no rows have a non-null primary_category.")

        cat_sum = category_yearly_summary(df_cat, token_col=token_col)
        cat_bucket_long = category_yearly_buckets(df_cat)

        cat_sum_path = out_dir / "category_yearly_summary.csv"
        cat_bucket_path = out_dir / "category_yearly_buckets.csv"
        cat_sum.to_csv(cat_sum_path, index=False)
        cat_bucket_long.to_csv(cat_bucket_path, index=False)

        print(f"[ok] Wrote: {cat_sum_path}")
        print(f"[ok] Wrote: {cat_bucket_path}")

        # Determine topK categories by observation count (across all years)
        cat_counts = (
            df_cat.groupby("primary_category", as_index=False)
                  .size()
                  .rename(columns={"size": "n_obs"})
                  .sort_values("n_obs", ascending=False)
        )
        topk = cat_counts.head(int(args.top_k_categories))["primary_category"].tolist()

        plot_category_topk_bucket_distribution(
            cat_bucket_long, topk, out_dir / "category_bucket_distribution_topK.png"
        )
        plot_category_topk_avg_holders(
            cat_sum, topk, out_dir / "category_avg_holders_topK.png"
        )
        print(f"[ok] Saved category plots (top {len(topk)} categories) in: {out_dir}")
    else:
        print("[warn] --categories-csv not provided, skipping primary_category analysis.")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
