#!/usr/bin/env python3
"""
compute_topshare_yearly_table.py

Description:
    Aggregate per-block top-holder concentration statistics into a yearly
    summary table (mean / min / max of top 1%, 5%, 10% share, plus the
    average number of wallets considered each year).

Input:
    CSV with columns (per block, per wallet_type):
        block_number, wallet_type, wallets_considered, total_value_usd,
        top_1_value_usd, top_5_value_usd, top_10_value_usd,
        top_1_share_pct, top_5_share_pct, top_10_share_pct
    Typically the output of compute_top_value_concentration_per_block.py.

Output:
    - Printed summary table (one row per year).
    - Optional CSV with the yearly aggregates (--out-csv).
    - Optional LaTeX-tabular rows on stdout (--latex).

Usage:
    python compute_topshare_yearly_table.py \\
        --csv ../data/top_held_percent_gr_1.csv \\
        --type-value ALL \\
        --out-csv yearly_topshare.csv \\
        --latex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# Hardcoded block -> date mapping for monthly snapshots (2020-01 .. 2025-12).
# Mirrors Analytics/portfolio_characteristics/distribution_of_wealth/plot/
# block_date_mapping.R; keep the two in sync if blocks are added.
DATES = [
  "2020-01-01","2020-02-01","2020-03-01","2020-04-01","2020-05-01","2020-06-01",
  "2020-07-01","2020-08-01","2020-09-01","2020-10-01","2020-11-01","2020-12-01",
  "2021-01-01","2021-02-01","2021-03-01","2021-04-01","2021-05-01","2021-06-01",
  "2021-07-01","2021-08-01","2021-09-01","2021-10-01","2021-11-01","2021-12-01",
  "2022-01-01","2022-02-01","2022-03-01","2022-04-01","2022-05-01","2022-06-01",
  "2022-07-01","2022-08-01","2022-09-01","2022-10-01","2022-11-01","2022-12-01",
  "2023-01-01","2023-02-01","2023-03-01","2023-04-01","2023-05-01","2023-06-01",
  "2023-07-01","2023-08-01","2023-09-01","2023-10-01","2023-11-01","2023-12-01",
  "2024-01-01","2024-02-01","2024-03-01","2024-04-01","2024-05-01","2024-06-01",
  "2024-07-01","2024-08-01","2024-09-01","2024-10-01","2024-11-01","2024-12-01",
  "2025-01-01","2025-02-01","2025-03-01","2025-04-01","2025-05-01","2025-06-01",
  "2025-07-01","2025-08-01","2025-09-01","2025-10-01","2025-11-01","2025-12-01"
]

BLOCKS = [
  9193266, 9393154, 9581792, 9782602, 9976964, 10176690, 10370274, 10570485, 10771925, 10966874, 11167817, 11363270,
  11565019, 11766939, 11948960, 12150245, 12344945, 12545219, 12738509, 12936340, 13136427, 13330090, 13527859, 13717847,
  13916166, 14116761, 14297759, 14497034, 14688630, 14881677, 15053226, 15253306, 15449618, 15649595, 15871480, 16086234,
  16308190, 16530248, 16730072, 16950603, 17162287, 17382266, 17595510, 17816434, 18037988, 18251965, 18473543, 18687851,
  18908895, 19129889, 19336607, 19557289, 19771560, 19993250, 20207949, 20429973, 20651994, 20866919, 21089069, 21303934,
  21525891, 21747950, 21948292, 22170335, 22385294, 22606143, 22820674, 23042514, 23264566, 23479244, 23700767, 23914921
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to top_help_percent_gr_1.csv")
    p.add_argument("--type-value", default="ALL", help="wallet_type value to keep (default: ALL)")
    p.add_argument("--out-csv", default=None, help="Optional: write yearly output CSV")
    p.add_argument("--latex", action="store_true", help="Print LaTeX rows for tabular")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.csv)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    b2d = pd.DataFrame(
        {"block_number": pd.Series(BLOCKS, dtype="int64"), "date": pd.to_datetime(DATES)}
    )

    df = pd.read_csv(in_path)

    required = {
        "block_number",
        "wallet_type",
        "wallets_considered",
        "top_1_share_pct",
        "top_5_share_pct",
        "top_10_share_pct",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in CSV: {sorted(missing)}")

    df["block_number"] = pd.to_numeric(df["block_number"], errors="coerce").astype("Int64")
    df["wallet_type"] = df["wallet_type"].astype(str).str.strip()
    df["wallets_considered"] = pd.to_numeric(df["wallets_considered"], errors="coerce")

    for c in ["top_1_share_pct", "top_5_share_pct", "top_10_share_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["block_number", "wallet_type", "wallets_considered",
                           "top_1_share_pct", "top_5_share_pct", "top_10_share_pct"]).copy()

    before = len(df)
    df = df[df["wallet_type"].str.upper() == args.type_value.strip().upper()].copy()
    print(f"[info] wallet_type filter: kept {len(df)}/{before} rows where wallet_type == {args.type_value}")
    if df.empty:
        raise RuntimeError("No rows left after wallet_type filter.")

    before = len(df)
    df = df.merge(b2d, how="left", on="block_number")
    df = df.dropna(subset=["date"]).copy()
    print(f"[info] block->date join: kept {len(df)}/{before} rows matched to mapping")
    if df.empty:
        raise RuntimeError("No rows matched the hardcoded block->date mapping.")

    df["year"] = df["date"].dt.year.astype(int)

    # Shares in [0, 1] are rescaled to percent so downstream formatting is
    # uniform regardless of the convention in the source CSV.
    max_share = df[["top_1_share_pct", "top_5_share_pct", "top_10_share_pct"]].max().max()
    if max_share <= 1.5:
        df[["top_1_share_pct", "top_5_share_pct", "top_10_share_pct"]] *= 100.0
        print("[info] detected shares in [0,1]; rescaled to percent")

    yearly = (
        df.groupby("year", as_index=False)
        .agg(
            avg_wallets=("wallets_considered", "mean"),
            top1_avg=("top_1_share_pct", "mean"),
            top1_min=("top_1_share_pct", "min"),
            top1_max=("top_1_share_pct", "max"),
            top5_avg=("top_5_share_pct", "mean"),
            top10_avg=("top_10_share_pct", "mean"),
            n_snapshots=("block_number", "count"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    yearly["avg_wallets_m"] = yearly["avg_wallets"] / 1_000_000.0
    yearly["top1_minmax"] = yearly.apply(lambda r: f"{r['top1_min']:.2f}--{r['top1_max']:.2f}", axis=1)

    show = yearly[["year", "avg_wallets_m", "top1_avg", "top1_minmax", "top5_avg", "top10_avg", "n_snapshots"]].copy()
    show = show.rename(columns={
        "year": "Year",
        "avg_wallets_m": "Avg wallets (M)",
        "top1_avg": "Top 1% (avg)",
        "top1_minmax": "Top 1% (min--max)",
        "top5_avg": "Top 5% (avg)",
        "top10_avg": "Top 10% (avg)",
        "n_snapshots": "N snapshots",
    })
    show["Avg wallets (M)"] = show["Avg wallets (M)"].map(lambda x: f"{x:.3f}")
    show["Top 1% (avg)"] = show["Top 1% (avg)"].map(lambda x: f"{x:.2f}")
    show["Top 5% (avg)"] = show["Top 5% (avg)"].map(lambda x: f"{x:.2f}")
    show["Top 10% (avg)"] = show["Top 10% (avg)"].map(lambda x: f"{x:.2f}")

    print("\n[table] Yearly top-holder concentration (ALL wallets):")
    print(show.to_string(index=False))

    if args.latex:
        print("\n[latex] Paste these rows into your Overleaf tabular:")
        for _, r in yearly.iterrows():
            print(
                f"{int(r['year'])} & {r['avg_wallets_m']:.3f} & {r['top1_avg']:.2f} & "
                f"{r['top1_min']:.2f}--{r['top1_max']:.2f} & {r['top5_avg']:.2f} & {r['top10_avg']:.2f} \\\\"
            )

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        yearly.to_csv(args.out_csv, index=False)
        print(f"\n[ok] wrote: {args.out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        raise
