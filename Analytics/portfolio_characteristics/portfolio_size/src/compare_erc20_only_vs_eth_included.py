#!/usr/bin/env python3
"""
compare_erc20_only_vs_eth_included.py

Description:
    Basic side-by-side comparison of the portfolio-size statistics from the
    ERC-20-only reconstruction (April 2026; files *_erc20_only.csv) and the
    ETH-inclusive reconstruction (September 2026; files without suffix),
    both produced by portfolio_size_by_wealth_bucket.py.

    Reports, for all wallets pooled and per wallet type (EOA, CA):
      1. accounts per snapshot (mean over blocks) and the ratio new/old
      2. share of accounts by coarse portfolio size (1, 2, 3-4, 5+),
         computed exactly like the PortSize* macros in generate_all_macros.py
         (per-block share, then mean across blocks; size-0 wallets stay in
         the denominator)
      3. avg % of accounts in each size bucket within each wealth bucket,
         old vs new, and the change in percentage points
      4. mean and median number of assets per wealth bucket (mean of the
         per-block values)

Input (in --data-dir):
    portfolio_size_hist_by_wealth[_wallet_type][_erc20_only].csv
    portfolio_size_stats_by_wealth[_wallet_type][_erc20_only].csv

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

WEALTH_ORDER = ["0–1", "1–100", "100–1,000", "1,000–10,000", "10,000–100,000", ">100,000"]
SIZE_ORDER = ["0", "1", "2", "3", "4", "5", "6–10", "11–20", "21–50", "51–100", ">100"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def coarse_cat(b: str) -> str:
    """Same mapping as generate_all_macros.emit_portfolio_size_distribution."""
    if b == "1":
        return "1"
    if b == "2":
        return "2"
    if b in ("3", "4"):
        return "3-4"
    if b in ("5", "6–10", "11–20", "21–50", "51–100", ">100"):
        return "5+"
    if b == "0":
        return "0"
    return "other"


def load_pair(data_dir: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(data_dir / f"{stem}{OLD_SUFFIX}.csv", dtype={"size_bucket": str, "wealth_bucket": str})
    new = pd.read_csv(data_dir / f"{stem}.csv", dtype={"size_bucket": str, "wealth_bucket": str})
    return old, new


def restrict_to_common_blocks(old: pd.DataFrame, new: pd.DataFrame, name: str, out: io.StringIO):
    b_old, b_new = set(old["block"]), set(new["block"])
    common = b_old & b_new
    if b_old != b_new:
        out.write(f"[note] {name}: {len(b_old)} old vs {len(b_new)} new blocks; "
                  f"comparing the {len(common)} common ones\n")
    return old[old["block"].isin(common)], new[new["block"].isin(common)], len(common)


def accounts_per_block(stats: pd.DataFrame, by_type: bool) -> pd.Series:
    keys = ["block", "wallet_type"] if by_type else ["block"]
    per_block = stats.groupby(keys)["n_wallets"].sum()
    if by_type:
        return per_block.groupby("wallet_type").mean()
    return pd.Series({"ALL": per_block.mean()})


def coarse_shares(hist: pd.DataFrame, by_type: bool) -> pd.DataFrame:
    """Per-block share of accounts in 1 / 2 / 3-4 / 5+ (size 0 stays in the
    denominator), then mean across blocks. Rows: wallet_type (or ALL)."""
    h = hist.copy()
    h["cat"] = h["size_bucket"].map(coarse_cat)
    keys = ["block", "wallet_type"] if by_type else ["block"]
    pb = h.groupby(keys + ["cat"])["wallet_count"].sum().unstack(fill_value=0)
    total = pb.sum(axis=1)
    shares = pb.div(total, axis=0) * 100.0
    if by_type:
        res = shares.groupby(level="wallet_type").mean()
    else:
        res = shares.mean().to_frame("ALL").T
    # The ETH-inclusive run has no size-0 wallets, the ERC-20-only run does;
    # keep a fixed column set so both sides line up.
    return res.reindex(columns=["0", "1", "2", "3-4", "5+"], fill_value=0.0)


def bucket_matrix(hist: pd.DataFrame) -> pd.DataFrame:
    """Avg % of accounts per (wealth_bucket x size_bucket), size 0 dropped
    and shares renormalised, i.e. what the stacked plots show."""
    h = hist[hist["size_bucket"] != "0"].copy()
    denom = h.groupby(["block", "wealth_bucket"])["wallet_count"].transform("sum")
    h["pct"] = 100.0 * h["wallet_count"] / denom
    m = h.groupby(["wealth_bucket", "size_bucket"])["pct"].mean().unstack(fill_value=0.0)
    m = m.reindex(index=WEALTH_ORDER, columns=[s for s in SIZE_ORDER if s in m.columns], fill_value=0.0)
    return m


def moments(stats: pd.DataFrame) -> pd.DataFrame:
    m = stats.groupby("wealth_bucket")[["n_wallets", "mean", "median"]].mean()
    return m.reindex(WEALTH_ORDER)


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare ERC-20-only vs ETH-inclusive portfolio-size outputs.")
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--out", default=None, help="Optional path to also write the report as text.")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    out = io.StringIO()
    w = out.write

    hist_old, hist_new = load_pair(data_dir, "portfolio_size_hist_by_wealth")
    hist_old, hist_new, n_blocks = restrict_to_common_blocks(hist_old, hist_new, "hist pooled", out)
    histwt_old, histwt_new = load_pair(data_dir, "portfolio_size_hist_by_wealth_wallet_type")
    histwt_old, histwt_new, _ = restrict_to_common_blocks(histwt_old, histwt_new, "hist by type", out)
    stats_old, stats_new = load_pair(data_dir, "portfolio_size_stats_by_wealth")
    stats_old, stats_new, _ = restrict_to_common_blocks(stats_old, stats_new, "stats pooled", out)
    statswt_old, statswt_new = load_pair(data_dir, "portfolio_size_stats_by_wealth_wallet_type")
    statswt_old, statswt_new, _ = restrict_to_common_blocks(statswt_old, statswt_new, "stats by type", out)

    w(f"Portfolio size: {OLD_LABEL} (April 2026) vs {NEW_LABEL} (Sept 2026), {n_blocks} monthly snapshots\n")
    w("=" * 100 + "\n")

    # 1. accounts per snapshot ------------------------------------------------
    w("\n1. Accounts per snapshot (mean over blocks)\n")
    acc = pd.concat([
        pd.DataFrame({OLD_LABEL: accounts_per_block(stats_old, False), NEW_LABEL: accounts_per_block(stats_new, False)}),
        pd.DataFrame({OLD_LABEL: accounts_per_block(statswt_old, True), NEW_LABEL: accounts_per_block(statswt_new, True)}),
    ])
    acc["ratio new/old"] = acc[NEW_LABEL] / acc[OLD_LABEL]
    acc[[OLD_LABEL, NEW_LABEL]] = acc[[OLD_LABEL, NEW_LABEL]].round(0).astype("int64")
    w(acc.round(2).to_string() + "\n")

    # 2. coarse size shares (macro definition) --------------------------------
    w("\n2. Share of accounts by portfolio size, % (per-block share, mean over blocks; "
      "definition of the PortSize* macros)\n")
    cs = pd.concat([
        side_by_side(coarse_shares(hist_old, False), coarse_shares(hist_new, False)),
        side_by_side(coarse_shares(histwt_old, True), coarse_shares(histwt_new, True)),
    ])
    w(cs.to_string() + "\n")

    # 3. size bucket x wealth bucket -----------------------------------------
    w("\n3. Avg % of accounts per size bucket within each wealth bucket (size 0 dropped, as in the plots)\n")
    m_old, m_new = bucket_matrix(hist_old), bucket_matrix(hist_new)
    w(f"\n3a. {OLD_LABEL}\n" + m_old.round(1).to_string() + "\n")
    w(f"\n3b. {NEW_LABEL}\n" + m_new.round(1).to_string() + "\n")
    w("\n3c. change in percentage points (new - old)\n" + (m_new - m_old).round(1).to_string() + "\n")

    w("\n3d. single-asset share per wealth bucket by wallet type, % (size 0 dropped)\n")
    rows = {}
    for wt in ("EOA", "CA"):
        mo = bucket_matrix(histwt_old[histwt_old["wallet_type"] == wt])["1"]
        mn = bucket_matrix(histwt_new[histwt_new["wallet_type"] == wt])["1"]
        rows[f"{wt} {OLD_LABEL}"] = mo
        rows[f"{wt} {NEW_LABEL}"] = mn
        rows[f"{wt} Δ"] = mn - mo
    w(pd.DataFrame(rows).round(1).to_string() + "\n")

    # 4. moments --------------------------------------------------------------
    w("\n4. Accounts, mean and median number of assets per wealth bucket (mean of per-block values)\n")
    mo, mn = moments(stats_old), moments(stats_new)
    mm = side_by_side(mo, mn)
    for c in [f"n_wallets {OLD_LABEL}", f"n_wallets {NEW_LABEL}", "n_wallets Δ"]:
        mm[c] = mm[c].round(0).astype("int64")
    w(mm.to_string() + "\n")

    report = out.getvalue()
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"[ok] wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
