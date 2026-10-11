#!/usr/bin/env python3
"""
per_block_wallet_stats_macros.py

Description:
    Single owner of every LaTeX macro derived from the per-block wallet
    statistics in ../data (produced by analyze_portfolio_per_block.py).
    Called by Analytics/generate_all_macros.py; can also be run standalone
    to print the macros.

    Every value is the mean over the monthly snapshots of the per-block
    statistic (shares are recomputed from the per-block counts, not from the
    rounded percentage columns). "Assets" are distinct assets held, native
    ETH included and ETH/WETH folded into one asset.

Macros (all wallets unless a wallet-type suffix EOA / CA is given):

    From averages_per_block.csv
        \\AcctAvgAssets/[EOA|CA]     mean number of assets per account
        \\AcctMedAssets/             median number of assets (mean of the
                                    per-block medians)
        \\AcctAvgValueUsd/[EOA|CA]   mean account value, USD
        \\AcctMedValueUsd/[EOA|CA]   median account value, USD
        \\Acct...GtOne/              the four ALL quantities restricted to
                                    accounts worth more than 1 USD

    From asset_buckets_per_block.csv (bucket-0 accounts stay in the
    denominator, as for the PortSize* macros)
        \\AcctSingleAssetPct/[EOA|CA]       % of accounts holding 1 asset
        \\AcctTwoFiveAssetsPct/             2-5 assets
        \\AcctAboveFiveAssetsPct/           more than 5 assets
        \\AcctSingleAssetValuePct/[EOA|CA]  % of total value held by
                                           single-asset accounts
        \\AcctTwoFiveAssetsValuePct/, \\AcctAboveFiveAssetsValuePct/

    From wallet_value_distribution_per_block.csv
        \\AcctPerSnapshotM/          accounts per snapshot, millions
        \\AcctZeroValuePct/          % of accounts with zero USD value
        \\AcctBelowOneUsdPct/        % of accounts worth less than 1 USD
        \\AcctTopOneSharePct/[EOA|CA], \\AcctTopFiveSharePct/, \\AcctTopTenSharePct/
                                    % of value held by the top 1 / 5 / 10 %
                                    of accounts (no value filter)
        \\AcctTopOneShareMinPct/, \\AcctTopOneShareMaxPct/
                                    range of the top-1 % share across snapshots
        \\AcctTopOneShareY<Year>Pct/ yearly means of the top-1 % share for
                                    the first and last year (needs a
                                    block-to-date map; see --block-to-date-csv)

    Comparison with the ERC-20-only reconstruction (emitted only when the
    *_erc20_only.csv files are present):
        \\Acct...ErcOnly             the averages and asset-bucket macros
                                    recomputed on the token-only data
        \\AcctAvgAssetsDelta/        change in the mean number of assets
        \\AcctAvgValueUsdDeltaPct/   relative change in the mean account value, %
        \\AcctSingleAssetPctDeltaPP/, \\AcctSingleAssetValuePctDeltaPP/,
        \\AcctAboveFiveAssetsValuePctDeltaPP/
                                    changes in percentage points (ETH-inclusive
                                    minus ERC-20-only)

Input (in --data-dir):
    averages_per_block.csv, asset_buckets_per_block.csv,
    wallet_value_distribution_per_block.csv (optional),
    optional: the same files with suffix _erc20_only

Usage:
    python per_block_wallet_stats_macros.py                 # prints the macros
    python per_block_wallet_stats_macros.py --data-dir ../data
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# block -> date mapping for the yearly top-1% macros: the account-count table
# of accounts_overview carries one (block, date) row per snapshot.
DEFAULT_BLOCK_DATES = (
    Path(__file__).resolve().parents[2] / "accounts_overview" / "data" / "output_accounts_eoa_vs_ca.csv"
)

F_AVERAGES = "averages_per_block"
F_BUCKETS = "asset_buckets_per_block"
F_VALUE_DIST = "wallet_value_distribution_per_block"
ERC_ONLY_SUFFIX = "_erc20_only"

WALLET_TYPES = ["ALL", "EOA", "CA"]
ABOVE_FIVE = ["5–10", "10–20", "20–50", "50–100", ">100"]
YEAR_WORDS = {2020: "Ytwenty", 2021: "Ytwentyone", 2022: "Ytwentytwo", 2023: "Ytwentythree",
              2024: "Ytwentyfour", 2025: "Ytwentyfive", 2026: "Ytwentysix"}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def fmt_dec(v, dp: int) -> str:
    return f"{round(float(v), dp):.{dp}f}"


def suffix_of(wt: str) -> str:
    return "" if wt == "ALL" else wt


def read_csv(path: Path, required: set) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"bucket": str})
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"[per_block_wallet_stats_macros] {path} lacks columns {sorted(missing)}")
    return df


def common_blocks(*frames: pd.DataFrame) -> set:
    blocks = None
    for f in frames:
        b = set(f["block"])
        blocks = b if blocks is None else blocks & b
    return blocks or set()


# -----------------------------------------------------------------------------
# quantities (plain floats), one dict per data set
# -----------------------------------------------------------------------------
def compute_averages(av: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    means = av.groupby("wallet_type").mean(numeric_only=True)
    for wt in WALLET_TYPES:
        if wt not in means.index:
            raise SystemExit(f"[per_block_wallet_stats_macros] wallet_type {wt!r} missing in averages")
        m = means.loc[wt]
        out[f"avg_assets_{wt}"] = m["avg_num_tokens"]
        out[f"med_assets_{wt}"] = m["med_num_tokens"]
        out[f"avg_value_{wt}"] = m["avg_value_usd"]
        out[f"med_value_{wt}"] = m["med_value_usd"]
    m = means.loc["ALL"]
    out["avg_assets_gt1"] = m["avg_num_tokens_gt1"]
    out["med_assets_gt1"] = m["med_num_tokens_gt1"]
    out["avg_value_gt1"] = m["avg_value_usd_gt1"]
    out["med_value_gt1"] = m["med_value_usd_gt1"]
    return out


def compute_buckets(ab: pd.DataFrame) -> Dict[str, float]:
    """Per-block shares of accounts and of value by asset bucket (bucket 0 in
    the denominator), then mean across blocks."""
    out: Dict[str, float] = {}
    for wt in WALLET_TYPES:
        sub = ab[ab["wallet_type"] == wt]
        if sub.empty:
            raise SystemExit(f"[per_block_wallet_stats_macros] wallet_type {wt!r} missing in asset buckets")
        tot_w = sub.groupby("block")["wallet_count"].transform("sum")
        tot_v = sub.groupby("block")["value_sum_usd"].transform("sum")
        s = sub.assign(w_share=100.0 * sub["wallet_count"] / tot_w,
                       v_share=100.0 * sub["value_sum_usd"] / tot_v.where(tot_v != 0, float("nan")))

        def share(buckets, col):
            per_block = s[s["bucket"].isin(buckets)].groupby("block")[col].sum()
            per_block = per_block.reindex(sorted(s["block"].unique()), fill_value=0.0)
            return float(per_block.mean())

        out[f"single_pct_{wt}"] = share(["1"], "w_share")
        out[f"single_value_pct_{wt}"] = share(["1"], "v_share")
        if wt == "ALL":
            out["two_five_pct"] = share(["2–5"], "w_share")
            out["above_five_pct"] = share(ABOVE_FIVE, "w_share")
            out["two_five_value_pct"] = share(["2–5"], "v_share")
            out["above_five_value_pct"] = share(ABOVE_FIVE, "v_share")
    return out


def compute_value_dist(vd: pd.DataFrame, block_dates: Optional[pd.Series]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for wt in WALLET_TYPES:
        sub = vd[vd["wallet_type"] == wt]
        if sub.empty:
            raise SystemExit(f"[per_block_wallet_stats_macros] wallet_type {wt!r} missing in value distribution")
        out[f"top1_{wt}"] = float(sub["top_1_share_pct"].mean())
    a = vd[vd["wallet_type"] == "ALL"].sort_values("block")
    out["accounts_per_snapshot"] = float(a["wallets_distinct"].mean())
    out["zero_value_pct"] = float((100.0 * a["wallets_val_eq0"] / a["wallets_distinct"]).mean())
    out["below_one_pct"] = float((100.0 * a["wallets_val_lt1"] / a["wallets_distinct"]).mean())
    out["top5"] = float(a["top_5_share_pct"].mean())
    out["top10"] = float(a["top_10_share_pct"].mean())
    out["top1_min"] = float(a["top_1_share_pct"].min())
    out["top1_max"] = float(a["top_1_share_pct"].max())
    if block_dates is not None:
        years = a["block"].map(block_dates).dt.year
        if years.notna().all():
            by_year = a.groupby(years)["top_1_share_pct"].mean()
            out["first_year"], out["last_year"] = int(by_year.index.min()), int(by_year.index.max())
            out["top1_first_year"] = float(by_year.iloc[0])
            out["top1_last_year"] = float(by_year.iloc[-1])
    return out


def load_block_dates(path: Optional[Path]) -> Optional[pd.Series]:
    if path is None or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    bcol = next((c for c in ("block", "block_number") if c in df.columns), None)
    dcol = next((c for c in ("date", "bound_date", "block_time_utc") if c in df.columns), None)
    if bcol is None or dcol is None:
        return None
    s = pd.Series(pd.to_datetime(df[dcol], errors="coerce").values, index=df[bcol].astype("int64"))
    return s[~s.index.duplicated()]


# -----------------------------------------------------------------------------
# emitters
# -----------------------------------------------------------------------------
def emit_averages(emit, v: Dict[str, float], suffix: str = "") -> None:
    for wt in WALLET_TYPES:
        s = suffix_of(wt) + suffix
        emit.macro(f"AcctAvgAssets{s}", fmt_dec(v[f"avg_assets_{wt}"], 2))
        emit.macro(f"AcctAvgValueUsd{s}", fmt_dec(v[f"avg_value_{wt}"], 0))
        emit.macro(f"AcctMedValueUsd{s}", fmt_dec(v[f"med_value_{wt}"], 2))
    emit.macro(f"AcctMedAssets{suffix}", fmt_dec(v["med_assets_ALL"], 1))
    emit.macro(f"AcctAvgAssetsGtOne{suffix}", fmt_dec(v["avg_assets_gt1"], 2))
    emit.macro(f"AcctMedAssetsGtOne{suffix}", fmt_dec(v["med_assets_gt1"], 1))
    emit.macro(f"AcctAvgValueUsdGtOne{suffix}", fmt_dec(v["avg_value_gt1"], 0))
    emit.macro(f"AcctMedValueUsdGtOne{suffix}", fmt_dec(v["med_value_gt1"], 2))


def emit_buckets(emit, v: Dict[str, float], suffix: str = "") -> None:
    for wt in WALLET_TYPES:
        s = suffix_of(wt) + suffix
        emit.macro(f"AcctSingleAssetPct{s}", fmt_dec(v[f"single_pct_{wt}"], 2))
        emit.macro(f"AcctSingleAssetValuePct{s}", fmt_dec(v[f"single_value_pct_{wt}"], 1))
    emit.macro(f"AcctTwoFiveAssetsPct{suffix}", fmt_dec(v["two_five_pct"], 2))
    emit.macro(f"AcctAboveFiveAssetsPct{suffix}", fmt_dec(v["above_five_pct"], 2))
    emit.macro(f"AcctTwoFiveAssetsValuePct{suffix}", fmt_dec(v["two_five_value_pct"], 1))
    emit.macro(f"AcctAboveFiveAssetsValuePct{suffix}", fmt_dec(v["above_five_value_pct"], 1))


def emit_value_dist(emit, v: Dict[str, float], suffix: str = "") -> None:
    emit.macro(f"AcctPerSnapshotM{suffix}", fmt_dec(v["accounts_per_snapshot"] / 1e6, 1))
    emit.macro(f"AcctZeroValuePct{suffix}", fmt_dec(v["zero_value_pct"], 2))
    emit.macro(f"AcctBelowOneUsdPct{suffix}", fmt_dec(v["below_one_pct"], 1))
    for wt in WALLET_TYPES:
        emit.macro(f"AcctTopOneSharePct{suffix_of(wt)}{suffix}", fmt_dec(v[f"top1_{wt}"], 2))
    emit.macro(f"AcctTopFiveSharePct{suffix}", fmt_dec(v["top5"], 2))
    emit.macro(f"AcctTopTenSharePct{suffix}", fmt_dec(v["top10"], 2))
    emit.macro(f"AcctTopOneShareMinPct{suffix}", fmt_dec(v["top1_min"], 2))
    emit.macro(f"AcctTopOneShareMaxPct{suffix}", fmt_dec(v["top1_max"], 2))
    if "first_year" in v:
        for key, year in (("top1_first_year", v["first_year"]), ("top1_last_year", v["last_year"])):
            word = YEAR_WORDS.get(int(year), f"Y{int(year)}")
            emit.macro(f"AcctTopOneShare{word}Pct{suffix}", fmt_dec(v[key], 2))


def emit_per_block_wallet_stats_macros(emit, data_dir: Optional[Path] = None,
                                       block_dates_csv: Optional[Path] = DEFAULT_BLOCK_DATES) -> None:
    """Write all per-block wallet-statistics macros through `emit`, which must
    provide .comment(text), .macro(name, value) and .blank() (MacroEmitter in
    generate_all_macros.py, or _PrintEmitter below)."""
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    av = read_csv(data_dir / f"{F_AVERAGES}.csv", {"block", "wallet_type", "avg_num_tokens"})
    ab = read_csv(data_dir / f"{F_BUCKETS}.csv", {"block", "wallet_type", "bucket", "wallet_count", "value_sum_usd"})
    n_blocks = av["block"].nunique()

    emit.comment(f"% --- Per-block wallet statistics (mean over {n_blocks} monthly snapshots; "
                 "per_block_wallet_stats/src/per_block_wallet_stats_macros.py) ---")
    new_av = compute_averages(av)
    new_ab = compute_buckets(ab)
    emit_averages(emit, new_av)
    emit_buckets(emit, new_ab)

    vd_p = data_dir / f"{F_VALUE_DIST}.csv"
    if vd_p.exists():
        vd = read_csv(vd_p, {"block", "wallet_type", "wallets_distinct", "wallets_val_eq0",
                             "wallets_val_lt1", "top_1_share_pct", "top_5_share_pct", "top_10_share_pct"})
        emit_value_dist(emit, compute_value_dist(vd, load_block_dates(block_dates_csv)))
    emit.blank()

    # optional comparison with the ERC-20-only reconstruction
    old_av_p = data_dir / f"{F_AVERAGES}{ERC_ONLY_SUFFIX}.csv"
    old_ab_p = data_dir / f"{F_BUCKETS}{ERC_ONLY_SUFFIX}.csv"
    if not (old_av_p.exists() and old_ab_p.exists()):
        return
    old_av_df = read_csv(old_av_p, {"block", "wallet_type", "avg_num_tokens"})
    old_ab_df = read_csv(old_ab_p, {"block", "wallet_type", "bucket", "wallet_count", "value_sum_usd"})
    blocks = common_blocks(av, ab, old_av_df, old_ab_df)
    emit.comment(f"% --- Same quantities on the ERC-20-only reconstruction (April 2026, "
                 f"{len(blocks)} common snapshots), plus the change when native ETH is included ---")
    old_av = compute_averages(old_av_df[old_av_df["block"].isin(blocks)])
    old_ab = compute_buckets(old_ab_df[old_ab_df["block"].isin(blocks)])
    emit_averages(emit, old_av, "ErcOnly")
    emit_buckets(emit, old_ab, "ErcOnly")
    cmp_av = compute_averages(av[av["block"].isin(blocks)])
    cmp_ab = compute_buckets(ab[ab["block"].isin(blocks)])
    emit.macro("AcctAvgAssetsDelta", fmt_dec(cmp_av["avg_assets_ALL"] - old_av["avg_assets_ALL"], 2))
    emit.macro("AcctAvgValueUsdDeltaPct",
               fmt_dec(100.0 * (cmp_av["avg_value_ALL"] / old_av["avg_value_ALL"] - 1.0), 1))
    emit.macro("AcctSingleAssetPctDeltaPP", fmt_dec(cmp_ab["single_pct_ALL"] - old_ab["single_pct_ALL"], 1))
    emit.macro("AcctSingleAssetValuePctDeltaPP",
               fmt_dec(cmp_ab["single_value_pct_ALL"] - old_ab["single_value_pct_ALL"], 1))
    emit.macro("AcctAboveFiveAssetsValuePctDeltaPP",
               fmt_dec(cmp_ab["above_five_value_pct"] - old_ab["above_five_value_pct"], 1))

    old_vd_p = data_dir / f"{F_VALUE_DIST}{ERC_ONLY_SUFFIX}.csv"
    if old_vd_p.exists():
        old_vd = read_csv(old_vd_p, {"block", "wallet_type", "wallets_distinct", "wallets_val_eq0",
                                     "wallets_val_lt1", "top_1_share_pct", "top_5_share_pct", "top_10_share_pct"})
        emit_value_dist(emit, compute_value_dist(old_vd, load_block_dates(block_dates_csv)), "ErcOnly")
    emit.blank()


# -----------------------------------------------------------------------------
# standalone use
# -----------------------------------------------------------------------------
class _PrintEmitter:
    """Minimal stand-in for generate_all_macros.MacroEmitter."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def comment(self, text: str = "") -> None:
        self.lines.extend(text.split("\n"))

    def blank(self) -> None:
        self.lines.append("")

    def macro(self, name: str, value: str, slash: bool = True) -> None:
        sep = "/" if slash else ""
        self.lines.append(rf"\def\{name}{sep}{{{value}}}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Print the per-block wallet-statistics LaTeX macros.")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--block-to-date-csv", default=str(DEFAULT_BLOCK_DATES),
                    help="CSV with block and date columns for the yearly top-1%% macros "
                         "(default: accounts_overview/data/output_accounts_eoa_vs_ca.csv).")
    args = ap.parse_args()
    e = _PrintEmitter()
    emit_per_block_wallet_stats_macros(e, Path(args.data_dir), Path(args.block_to_date_csv))
    print("\n".join(e.lines))


if __name__ == "__main__":
    main()
