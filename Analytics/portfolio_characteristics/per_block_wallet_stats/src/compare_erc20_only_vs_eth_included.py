#!/usr/bin/env python3
"""
compare_erc20_only_vs_eth_included.py

Description:
    Side-by-side comparison of the per-block wallet statistics from the
    ERC-20-only reconstruction (April 2026; files *_erc20_only.csv) and the
    ETH-inclusive reconstruction (September 2026; files without suffix),
    both produced by analyze_portfolio_per_block.py.

    Reports, per wallet type (ALL, EOA, CA), as means over the common
    snapshots:
      1. accounts per snapshot and the ratio new/old
      2. averages_per_block: mean / median number of assets and account
         value, all accounts and accounts worth more than 1 USD
      3. asset_buckets_per_block: % of accounts and % of value per asset
         bucket, recomputed from the per-block counts (bucket 0 stays in
         the denominator), and the change in percentage points
      4. wallet_value_distribution_per_block: accounts below 1 USD and the
         top-1/5/10 % value shares; compared with the ERC-20-only file when
         it exists, otherwise reported for the new data only

Input (in --data-dir):
    averages_per_block[_erc20_only].csv
    asset_buckets_per_block[_erc20_only].csv
    wallet_value_distribution_per_block[_erc20_only].csv (old file optional)

Usage:
    python compare_erc20_only_vs_eth_included.py --data-dir ../data
    python compare_erc20_only_vs_eth_included.py --data-dir ../data --out ../data/compare_erc20_only_vs_eth_included.txt
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd

OLD_SUFFIX = "_erc20_only"
OLD_LABEL = "erc20_only"
NEW_LABEL = "eth_incl"

WALLET_TYPES = ["ALL", "EOA", "CA"]
BUCKET_ORDER = ["0", "1", "2–5", "5–10", "10–20", "20–50", "50–100", ">100"]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def load_pair(data_dir: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(data_dir / f"{stem}{OLD_SUFFIX}.csv", dtype={"bucket": str})
    new = pd.read_csv(data_dir / f"{stem}.csv", dtype={"bucket": str})
    return old, new


def restrict_to_common_blocks(old: pd.DataFrame, new: pd.DataFrame, name: str, out: io.StringIO):
    b_old, b_new = set(old["block"]), set(new["block"])
    common = b_old & b_new
    if b_old != b_new:
        out.write(f"[note] {name}: {len(b_old)} old vs {len(b_new)} new blocks; "
                  f"comparing the {len(common)} common ones\n")
    return old[old["block"].isin(common)], new[new["block"].isin(common)], len(common)


def side_by_side(old: pd.DataFrame, new: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    """Interleave old / new / delta columns for every column of `old`."""
    cols = old.columns.union(new.columns, sort=False)
    old = old.reindex(columns=cols, fill_value=0.0)
    new = new.reindex(columns=cols, fill_value=0.0)
    parts = {}
    for c in cols:
        parts[f"{c} {OLD_LABEL}"] = old[c]
        parts[f"{c} {NEW_LABEL}"] = new[c]
        parts[f"{c} Δ"] = new[c] - old[c]
    return pd.DataFrame(parts).round(digits)


def accounts_per_snapshot(ab: pd.DataFrame) -> pd.Series:
    return ab.groupby(["block", "wallet_type"])["wallet_count"].sum().groupby("wallet_type").mean().reindex(WALLET_TYPES)


def averages_table(av: pd.DataFrame) -> pd.DataFrame:
    return av.groupby("wallet_type").mean(numeric_only=True).drop(columns=["block"]).reindex(WALLET_TYPES)


def bucket_shares(ab: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean over blocks of the per-block share (% of accounts or % of value)
    per asset bucket; rows wallet_type, columns buckets."""
    tot = ab.groupby(["block", "wallet_type"])[col].transform("sum")
    s = ab.assign(share=100.0 * ab[col] / tot.where(tot != 0, float("nan")))
    per_block = s.pivot_table(index=["block", "wallet_type"], columns="bucket", values="share",
                              aggfunc="sum", fill_value=0.0)
    m = per_block.groupby(level="wallet_type").mean()
    return m.reindex(index=WALLET_TYPES, columns=[b for b in BUCKET_ORDER if b in m.columns], fill_value=0.0)


def value_dist_table(vd: pd.DataFrame) -> pd.DataFrame:
    d = vd.copy()
    d["below_1usd_pct"] = 100.0 * d["wallets_val_lt1"] / d["wallets_distinct"]
    d["zero_value_pct"] = 100.0 * d["wallets_val_eq0"] / d["wallets_distinct"]
    cols = ["wallets_distinct", "zero_value_pct", "below_1usd_pct",
            "top_1_share_pct", "top_5_share_pct", "top_10_share_pct"]
    return d.groupby("wallet_type")[cols].mean().reindex(WALLET_TYPES)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare ERC-20-only vs ETH-inclusive per-block wallet statistics.")
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--out", default=None, help="Optional path to also write the report as text.")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    out = io.StringIO()
    w = out.write

    av_old, av_new = load_pair(data_dir, "averages_per_block")
    av_old, av_new, n_blocks = restrict_to_common_blocks(av_old, av_new, "averages", out)
    ab_old, ab_new = load_pair(data_dir, "asset_buckets_per_block")
    ab_old, ab_new, _ = restrict_to_common_blocks(ab_old, ab_new, "asset buckets", out)

    w(f"Per-block wallet statistics: {OLD_LABEL} (April 2026) vs {NEW_LABEL} (Sept 2026), "
      f"{n_blocks} monthly snapshots\n")
    w("=" * 110 + "\n")

    # 1. accounts per snapshot ------------------------------------------------
    w("\n1. Accounts per snapshot (mean over blocks)\n")
    acc = pd.DataFrame({OLD_LABEL: accounts_per_snapshot(ab_old), NEW_LABEL: accounts_per_snapshot(ab_new)})
    acc["ratio new/old"] = (acc[NEW_LABEL] / acc[OLD_LABEL]).round(2)
    acc[[OLD_LABEL, NEW_LABEL]] = acc[[OLD_LABEL, NEW_LABEL]].round(0).astype("int64")
    w(acc.to_string() + "\n")

    # 2. averages -------------------------------------------------------------
    w("\n2. Mean / median number of assets and account value (USD), mean of the per-block values\n")
    w("   (gt1 = accounts worth more than 1 USD; assets = ETH included, ETH/WETH folded)\n")
    sbs = side_by_side(averages_table(av_old), averages_table(av_new), 3)
    for c in sbs.columns:
        if "value" in c:
            sbs[c] = sbs[c].round(2)
    w(sbs.T.to_string() + "\n")

    # 3. asset buckets --------------------------------------------------------
    w("\n3. Share of accounts by number of assets, % (per-block share, mean over blocks; bucket 0 in the denominator)\n")
    for wt in WALLET_TYPES:
        w(f"\n3a. {wt}\n")
        w(side_by_side(bucket_shares(ab_old, "wallet_count").loc[[wt]],
                       bucket_shares(ab_new, "wallet_count").loc[[wt]]).T.to_string() + "\n")
    w("\n3b. Share of total value held by accounts in each asset bucket, %\n")
    for wt in WALLET_TYPES:
        w(f"\n    {wt}\n")
        w(side_by_side(bucket_shares(ab_old, "value_sum_usd").loc[[wt]],
                       bucket_shares(ab_new, "value_sum_usd").loc[[wt]]).T.to_string() + "\n")

    # 4. value distribution ---------------------------------------------------
    w("\n4. Value distribution: accounts, share below 1 USD, top-1/5/10 % value shares (no value filter)\n")
    new_vd_p = data_dir / "wallet_value_distribution_per_block.csv"
    old_vd_p = data_dir / f"wallet_value_distribution_per_block{OLD_SUFFIX}.csv"
    if not new_vd_p.exists():
        w("   (wallet_value_distribution_per_block.csv not found)\n")
    else:
        vd_new = pd.read_csv(new_vd_p)
        if old_vd_p.exists():
            vd_old = pd.read_csv(old_vd_p)
            vd_old, vd_new, _ = restrict_to_common_blocks(vd_old, vd_new, "value distribution", out)
            t = side_by_side(value_dist_table(vd_old), value_dist_table(vd_new))
            for c in t.columns:
                if c.startswith("wallets_distinct"):
                    t[c] = t[c].round(0).astype("int64")
            w(t.T.to_string() + "\n")
        else:
            w(f"   (no {old_vd_p.name}: the ERC-20-only run did not produce this file; new data only)\n")
            t = value_dist_table(vd_new).round(2)
            t["wallets_distinct"] = t["wallets_distinct"].round(0).astype("int64")
            w(t.T.to_string() + "\n")

    report = out.getvalue()
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"[ok] wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
