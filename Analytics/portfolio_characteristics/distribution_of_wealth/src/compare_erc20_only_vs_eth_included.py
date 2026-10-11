#!/usr/bin/env python3
"""
compare_erc20_only_vs_eth_included.py

Description:
    Side-by-side comparison of the wealth-distribution outputs from the
    ERC-20-only reconstruction (April 2026; files *_erc20_only.csv) and the
    ETH-inclusive reconstruction with the beacon deposit contract excluded
    (September 2026; files without suffix). Tag-coverage files are ignored.

    Reports, over the common snapshots:
      1. accounts per snapshot by wallet type (ALL, EOA, CA)
      2. wealth buckets: mean % of accounts and % of value per USD bucket,
         with the change in percentage points
      3. total value: last snapshot, peak and its date, CA share of value,
         and the total-value / DeFiLlama-TVL ratio
      4. top-share concentration (accounts above 1 USD): the yearly table of
         top-1/5/10 % shares for ALL accounts, old and new, and the overall
         means per wallet type
      5. top-1 % and top-0.1 % breakdown: share of the top accounts that are
         CAs, CA share of the top value, entry thresholds
      6. token inequality: per-year means across the ERC-20 tokens present
         in both runs (holders, Gini, HHI, top-1/10 shares, Nakamoto), which
         should agree up to floating-point noise, and the native-ETH row of
         the new run by year

Input (in --data-dir):
    value_buckets_minimal[_erc20_only].csv
    top_held_percent_gr_1[_erc20_only].csv
    top_1_pct_summary[_erc20_only].csv, top_1_pct_ca_eoa_breakdown[_erc20_only].csv
    top_0_1_pct_summary[_erc20_only].csv, top_0_1_pct_ca_eoa_breakdown[_erc20_only].csv
    token_inequality_per_block[_erc20_only].csv
    defillama_eth_tvl.csv

Usage:
    python compare_erc20_only_vs_eth_included.py --data-dir ../data
    python compare_erc20_only_vs_eth_included.py --data-dir ../data --out ../data/compare_erc20_only_vs_eth_included.txt
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distribution_of_wealth_macros import (  # noqa: E402
    ERC_ONLY_SUFFIX, load_block_dates, load_buckets, load_topshare, load_tvl,
    topshare_yearly, wealth_series, with_tvl,
)

OLD_LABEL, NEW_LABEL = "erc20_only", "eth_incl"
WALLET_TYPES = ["ALL", "EOA", "CA"]
BUCKET_ORDER = ["0–1", "1–100", "100–1K", "1K–10K", "10K–100K", ">100K"]
ETH_PSEUDO_TOKEN = "0x" + "ee" * 20

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def side_by_side(old: pd.DataFrame, new: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    cols = old.columns.union(new.columns, sort=False)
    old = old.reindex(columns=cols, fill_value=0.0)
    new = new.reindex(columns=cols, fill_value=0.0)
    parts = {}
    for c in cols:
        parts[f"{c} {OLD_LABEL}"] = old[c]
        parts[f"{c} {NEW_LABEL}"] = new[c]
        parts[f"{c} Δ"] = new[c] - old[c]
    return pd.DataFrame(parts).round(digits)


def bucket_means(vb: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rows wallet_type, columns buckets: mean over blocks of the per-block share (%)."""
    tot = vb.groupby(["block", "wallet_type"])[col].transform("sum")
    s = vb.assign(share=100.0 * vb[col] / tot.where(tot != 0))
    m = s.pivot_table(index=["block", "wallet_type"], columns="bucket", values="share", aggfunc="sum").groupby(level="wallet_type").mean()
    return m.reindex(index=WALLET_TYPES, columns=[b for b in BUCKET_ORDER if b in m.columns])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare ERC-20-only vs ETH-inclusive wealth-distribution outputs.")
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--out", default=None, help="Optional path to also write the report as text.")
    args = ap.parse_args()
    dd = Path(args.data_dir)

    out = io.StringIO()
    w = out.write

    d_old, d_new = load_block_dates(dd, ERC_ONLY_SUFFIX), load_block_dates(dd, "")
    vb_old, vb_new = load_buckets(dd, ERC_ONLY_SUFFIX), load_buckets(dd, "")
    common = sorted(set(vb_old["block"]) & set(vb_new["block"]))
    vb_old, vb_new = vb_old[vb_old["block"].isin(common)], vb_new[vb_new["block"].isin(common)]
    w(f"Wealth distribution: {OLD_LABEL} (April 2026) vs {NEW_LABEL} (Sept 2026, beacon deposit contract excluded), "
      f"{len(common)} common monthly snapshots\n" + "=" * 110 + "\n")

    # 1. accounts ---------------------------------------------------------------
    w("\n1. Accounts per snapshot (mean over blocks)\n")
    acc = pd.DataFrame({
        OLD_LABEL: vb_old.groupby(["block", "wallet_type"])["wallet_count"].sum().groupby("wallet_type").mean(),
        NEW_LABEL: vb_new.groupby(["block", "wallet_type"])["wallet_count"].sum().groupby("wallet_type").mean(),
    }).reindex(WALLET_TYPES)
    acc["ratio new/old"] = (acc[NEW_LABEL] / acc[OLD_LABEL]).round(2)
    acc[[OLD_LABEL, NEW_LABEL]] = acc[[OLD_LABEL, NEW_LABEL]].round(0).astype("int64")
    w(acc.to_string() + "\n")

    # 2. wealth buckets -----------------------------------------------------------
    w("\n2. Share of accounts by USD value bucket, % (per-block share, mean over blocks)\n")
    for wt in WALLET_TYPES:
        w(f"\n2a. {wt}\n")
        w(side_by_side(bucket_means(vb_old, "wallet_count").loc[[wt]], bucket_means(vb_new, "wallet_count").loc[[wt]]).T.to_string() + "\n")
    w("\n2b. Share of total value held by the accounts of each bucket, %\n")
    for wt in WALLET_TYPES:
        w(f"\n    {wt}\n")
        w(side_by_side(bucket_means(vb_old, "value_sum_usd").loc[[wt]], bucket_means(vb_new, "value_sum_usd").loc[[wt]]).T.to_string() + "\n")

    # 3. total value, CA share, TVL ------------------------------------------------
    w("\n3. Total value, CA share of value, and total value over DeFiLlama TVL\n")
    tvl = load_tvl(dd)
    rows = {}
    for lab, vb, dates in ((OLD_LABEL, vb_old, d_old), (NEW_LABEL, vb_new, d_new)):
        ws = with_tvl(wealth_series(vb, dates), tvl)
        share = 100.0 * ws["ca_holdings_usd"] / ws["overall_wealth_usd"]
        ratio = ws["overall_wealth_usd"] / ws["defillama_eth_tvl_usd"]
        peak = ws.loc[ws["overall_wealth_usd"].idxmax()]
        rows[lab] = {
            "total value, last snapshot (B USD)": ws["overall_wealth_usd"].iloc[-1] / 1e9,
            "total value, peak (B USD)": peak["overall_wealth_usd"] / 1e9,
            "peak date": peak["date"].strftime("%Y-%m"),
            "CA value, last snapshot (B USD)": ws["ca_holdings_usd"].iloc[-1] / 1e9,
            "CA share of value, mean %": share.mean(),
            "CA share of value, min %": share.min(),
            "CA share of value, max %": share.max(),
            "CA share of value, last %": share.iloc[-1],
            "total value / TVL, median": ratio.median(),
            "total value / TVL, min": ratio.min(),
            "total value / TVL, max": ratio.max(),
        }
    t3 = pd.DataFrame(rows)
    t3["Δ"] = pd.to_numeric(t3[NEW_LABEL], errors="coerce") - pd.to_numeric(t3[OLD_LABEL], errors="coerce")
    w(t3.apply(lambda c: c.map(lambda v: f"{v:,.2f}" if isinstance(v, float) else v)).to_string() + "\n")

    # 4. top-share concentration ---------------------------------------------------
    w("\n4. Top-share concentration, accounts above 1 USD (top_held_percent_gr_1.csv)\n")
    th_old, th_new = load_topshare(dd, ERC_ONLY_SUFFIX), load_topshare(dd, "")
    th_old, th_new = th_old[th_old["block_number"].isin(common)], th_new[th_new["block_number"].isin(common)]
    w("\n4a. Yearly table, ALL accounts (avg accounts M | top-1 % avg | top-1 % min--max | top-5 % avg | top-10 % avg)\n")
    yo = topshare_yearly(th_old, d_old, "ALL").set_index("year")
    yn = topshare_yearly(th_new, d_new, "ALL").set_index("year")
    tab = pd.DataFrame({
        f"accounts M {OLD_LABEL}": yo["avg_wallets_m"].round(3), f"accounts M {NEW_LABEL}": yn["avg_wallets_m"].round(3),
        f"top1 {OLD_LABEL}": yo["top1_avg"].round(2), f"top1 {NEW_LABEL}": yn["top1_avg"].round(2),
        f"top1 min--max {OLD_LABEL}": [f"{a:.2f}--{b:.2f}" for a, b in zip(yo["top1_min"], yo["top1_max"])],
        f"top1 min--max {NEW_LABEL}": [f"{a:.2f}--{b:.2f}" for a, b in zip(yn["top1_min"], yn["top1_max"])],
        f"top5 {OLD_LABEL}": yo["top5_avg"].round(2), f"top5 {NEW_LABEL}": yn["top5_avg"].round(2),
        f"top10 {OLD_LABEL}": yo["top10_avg"].round(2), f"top10 {NEW_LABEL}": yn["top10_avg"].round(2),
    })
    w(tab.to_string() + "\n")
    w("\n4b. Means over all snapshots per wallet type\n")
    cols = ["wallets_considered", "top_1_share_pct", "top_5_share_pct", "top_10_share_pct"]
    w(side_by_side(th_old.groupby("wallet_type")[cols].mean().reindex(WALLET_TYPES),
                   th_new.groupby("wallet_type")[cols].mean().reindex(WALLET_TYPES)).T.to_string() + "\n")

    # 5. top-1 % / top-0.1 % breakdown ------------------------------------------
    w("\n5. Top-1 % and top-0.1 % accounts: CA / EOA split and entry thresholds (means over blocks)\n")
    for pct, f_sum in (("1", "top_1_pct_summary"), ("0_1", "top_0_1_pct_summary")):
        rows = {}
        for lab, suffix in ((OLD_LABEL, ERC_ONLY_SUFFIX), (NEW_LABEL, "")):
            br = pd.read_csv(dd / f"top_{pct}_pct_ca_eoa_breakdown{suffix}.csv")
            sm = pd.read_csv(dd / f"{f_sum}{suffix}.csv")
            br, sm = br[br["block"].isin(common)], sm[sm["block"].isin(common)]
            ca = br[br["wallet_type"] == "CA"]
            rows[lab] = {
                "CA % of the top accounts": ca["n_wallets_pct_of_top1"].mean(),
                "CA % of the top value": ca["value_pct_of_top1"].mean(),
                "top value % of all value": sm["value_pct_top1"].mean(),
                "entry threshold, mean (USD)": sm["threshold_usd"].mean(),
                "entry threshold, last (USD)": sm["threshold_usd"].iloc[-1],
                "top accounts, last snapshot": sm["n_wallets_top1"].iloc[-1],
            }
        t5 = pd.DataFrame(rows); t5["Δ"] = t5[NEW_LABEL] - t5[OLD_LABEL]
        w(f"\n5{'a' if pct == '1' else 'b'}. Top {pct.replace('_', '.')} %\n")
        w(t5.round(2).to_string() + "\n")

    # 6. token inequality --------------------------------------------------------
    w("\n6. Token inequality (token_inequality_per_block.csv)\n")
    use = ["date", "block", "token_address", "holders", "gini", "hhi", "top1", "top10", "nakamoto", "supply"]
    iq_old = pd.read_csv(dd / f"token_inequality_per_block{ERC_ONLY_SUFFIX}.csv", usecols=use, dtype={"block": str})
    iq_new = pd.read_csv(dd / "token_inequality_per_block.csv", usecols=use, dtype={"block": str})
    for iq in (iq_old, iq_new):
        iq["block"] = pd.to_numeric(iq["block"], errors="coerce")
        for c in use[3:]:
            iq[c] = pd.to_numeric(iq[c], errors="coerce")
    iq_old = iq_old[iq_old["block"].isin(common)].copy(); iq_new = iq_new[iq_new["block"].isin(common)].copy()
    tokens_common = set(iq_old["token_address"]) & set(iq_new["token_address"]) - {ETH_PSEUDO_TOKEN}
    w(f"   tokens: {iq_old['token_address'].nunique():,} old, {iq_new['token_address'].nunique():,} new "
      f"(incl. ETH), {len(tokens_common):,} ERC-20 tokens in both\n")
    w("\n6a. Per-year means across the common ERC-20 tokens (identical ledgers; differences are floating-point noise)\n")
    metrics = ["holders", "gini", "hhi", "top1", "top10", "nakamoto"]
    def yearly(iq):
        d = iq[iq["token_address"].isin(tokens_common)].copy(); d["year"] = pd.to_datetime(d["date"]).dt.year
        return d.groupby("year")[metrics].mean()
    w(side_by_side(yearly(iq_old), yearly(iq_new), 4).to_string() + "\n")
    eth = iq_new[iq_new["token_address"] == ETH_PSEUDO_TOKEN].copy()
    if len(eth):
        w("\n6b. Native ETH (new run only; deposit contract excluded), per-year means\n")
        eth["year"] = pd.to_datetime(eth["date"]).dt.year
        e = eth.groupby("year").agg(holders_M=("holders", lambda s: s.mean() / 1e6), gini=("gini", "mean"), hhi=("hhi", "mean"),
                                    top1=("top1", "mean"), top10=("top10", "mean"), nakamoto=("nakamoto", "mean"),
                                    supply_M=("supply", lambda s: s.mean() / 1e6))
        w(e.round(4).to_string() + "\n")

    report = out.getvalue()
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"[ok] wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
