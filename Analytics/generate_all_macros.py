#!/usr/bin/env python3
"""
generate_all_macros.py

Description:
    Rebuild every LaTeX macro used in the paper from pre-computed
    CSV / Markdown artefacts under Analytics/. Values that cannot be
    derived from data are loaded from manual_constants.py.

    The generated file mirrors the section ordering of the paper's
    macro.tex so that diffs stay legible.

Output:
    Analytics/macro.tex

Usage:
    python Analytics/generate_all_macros.py            # write macro.tex
    python Analytics/generate_all_macros.py --verify   # compare against target
    python Analytics/generate_all_macros.py --verbose  # debug output

Verification target:
    --verify compares the generated file against a reference macro.tex.
    The path is resolved in this order:
        1. --target-tex CLI flag
        2. MACRO_TARGET_TEX env var
        3. ../BlockchainPortfolio/macro.tex (default, the paper's LaTeX
           source — not included in this public repo)
    Verification is skipped with a clear message if the target file does
    not exist.
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import re
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
BASE = pathlib.Path(__file__).resolve().parent.parent
ANALYTICS = BASE / "Analytics"
OUT_TEX = ANALYTICS / "macro.tex"
DEFAULT_TARGET_TEX = BASE / "BlockchainPortfolio" / "macro.tex"
REPORT_TXT = ANALYTICS / "macro_verification_report.txt"


def resolve_target_tex(cli_arg: Optional[str]) -> pathlib.Path:
    """Return the verification target path from CLI arg, env var, or default."""
    if cli_arg:
        return pathlib.Path(cli_arg).expanduser()
    env_val = os.environ.get("MACRO_TARGET_TEX")
    if env_val:
        return pathlib.Path(env_val).expanduser()
    return DEFAULT_TARGET_TEX

# Data folders
DRA_DATA = ANALYTICS / "distance_return_analytics" / "data"
RFSHAP_DATA = DRA_DATA / "rf_shap_data"   # 3-feat & 6-feat RF importances + SHAP values + report
PC = ANALYTICS / "portfolio_characteristics"
ACC_DATA = PC / "accounts_overview" / "data"
DOW_DATA = PC / "distribution_of_wealth" / "data"
PSZ_DATA = PC / "portfolio_size" / "data"
PBS_DATA = PC / "per_block_wallet_stats" / "data"
THC_DATA = PC / "token_holder_counts" / "data"

# ---------------------------------------------------------------------------
# Macro emitter helper
# ---------------------------------------------------------------------------
class MacroEmitter:
    """Collect raw text lines (macros and comments) in order, then assemble."""

    def __init__(self):
        self.lines: List[str] = []
        self.macro_names: "OrderedDict[str, int]" = OrderedDict()
        self.macro_seen: Dict[str, str] = {}  # name -> raw value text

    def comment(self, text: str = "") -> None:
        for line in text.split("\n"):
            self.lines.append(line)

    def blank(self) -> None:
        self.lines.append("")

    def macro(self, name: str, value: str, slash: bool = True) -> None:
        """Emit `\def\<name>/{<value>}`.

        If `slash=False`, emit `\def\<name>{<value>}` (no trailing slash —
        a few legacy macros like \HoldersAvgYtwenty use this form).
        """
        sep = "/" if slash else ""
        line = rf"\def\{name}{sep}{{{value}}}"
        self.lines.append(line)
        # Remember last-seen value (handle redefines like \LOneSmall*).
        if name not in self.macro_seen:
            self.macro_names[name] = len(self.lines) - 1
        self.macro_seen[name] = value

    def macro_raw(self, line: str) -> None:
        """Emit a raw \def line verbatim (used for tricky cases).

        Strips any trailing inline comment.
        """
        m = re.match(r"\\def\\([A-Za-z][A-Za-z0-9]*)/?\{(.*)\}", line)
        # Drop any trailing "  % comment" so generated output stays clean.
        clean = re.sub(r"\}\s*%.*$", "}", line)
        self.lines.append(clean)
        if m:
            name = m.group(1)
            if name not in self.macro_seen:
                self.macro_names[name] = len(self.lines) - 1
            self.macro_seen[name] = m.group(2)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


# ---------------------------------------------------------------------------
# Number formatters
# ---------------------------------------------------------------------------
def fmt_num_int(v) -> str:
    """`\num{12345}`."""
    return rf"\num{{{int(round(float(v)))}}}"


def fmt_si_pct(v, dp: int = 2) -> str:
    """`\SI{12.34}{\percent}`."""
    return rf"\SI{{{round(float(v), dp):.{dp}f}}}{{\percent}}"


def fmt_pct(v, dp: int = 1) -> str:
    """`12.3\%` or similar."""
    return rf"\num{{{round(float(v), dp):.{dp}f}}}\%"


def fmt_dec(v, dp: int) -> str:
    return f"{round(float(v), dp):.{dp}f}"


# ---------------------------------------------------------------------------
def _read_md(path: pathlib.Path) -> str:
    return path.read_text()


def parse_capm_md() -> Dict[str, float]:
    """Pull every numerical macro source value from analyze_extended_capm_full.md."""
    txt = _read_md(DRA_DATA / "analyze_extended_capm_full.md")
    out: Dict[str, float] = {}

    # --- L1 distances overall (lines 8..18 in the report) ---
    # Format: index   l1_metric  mean (%)  median (%)  std (%)
    for line in txt.splitlines():
        m = re.match(
            r"^\s*\d+\s+(l1_\S+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s*$",
            line,
        )
        if m:
            metric, mean, median, std = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
            out[f"{metric}_mean"] = mean
            out[f"{metric}_median"] = median
            out[f"{metric}_std"] = std

    # --- Returns overall (lines after "Returns — Overall") ---
    for line in txt.splitlines():
        m = re.match(r"^\s*\d+\s+(ret_\S+|market_return)\s+([-+0-9.]+)\s*$", line)
        if m:
            out[f"{m.group(1)}_median"] = float(m.group(2))

    # --- Beta summary ---
    for line in txt.splitlines():
        m = re.match(
            r"^\s*\d+\s+(baseline|better_return|safer_risk|max_sharpe|equal_weight|mcap_weight)\s+"
            r"([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s*$",
            line,
        )
        if m:
            strat = m.group(1)
            mean, median, std, p5, p25, p75, p95, frac_above_1, frac_neg = [float(m.group(i)) for i in range(2, 11)]
            out[f"beta_{strat}_mean"] = mean
            out[f"beta_{strat}_median"] = median
            out[f"beta_{strat}_std"] = std
            out[f"beta_{strat}_p25"] = p25
            out[f"beta_{strat}_p75"] = p75
            out[f"beta_{strat}_frac_above_1"] = frac_above_1 * 100
            out[f"beta_{strat}_frac_neg"] = frac_neg * 100

    # --- Alpha summary by strategy: cols mean median std p5 p25 p75 p95 frac_positive ---
    in_alpha_section = False
    for line in txt.splitlines():
        if "Alpha Summary by Strategy" in line:
            in_alpha_section = True
            continue
        if in_alpha_section:
            m = re.match(
                r"^\s*\d+\s+(baseline|better_return|safer_risk|max_sharpe|equal_weight|mcap_weight)\s+"
                r"([-+0-9.eE+]+)\s+([-+0-9.]+)\s+([-+0-9.eE+]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s*$",
                line,
            )
            if m:
                strat = m.group(1)
                median = float(m.group(3))
                frac_positive = float(m.group(9))
                out[f"alpha_{strat}_median"] = median
                out[f"alpha_{strat}_frac_positive"] = frac_positive * 100
            elif line.startswith("=========") or "Beta Distribution" in line:
                in_alpha_section = False

    # --- Spearman correlations (sample) ---
    spearman_re = re.compile(
        r"\s*([a-zA-Z0-9_]+)\s+vs\s+([a-zA-Z0-9_]+)\s+ρ\s*=\s*([-+0-9.]+)"
    )
    for line in txt.splitlines():
        m = spearman_re.search(line)
        if m:
            a, b, rho = m.group(1), m.group(2), float(m.group(3))
            out[f"spearman_{a}_vs_{b}"] = rho

    # --- Optimizer improvement: strategy improvement over baseline ---
    in_strat_imp = False
    for line in txt.splitlines():
        if "Strategy Improvement Over Baseline" in line:
            in_strat_imp = True
            continue
        if in_strat_imp:
            # strategy mean_ret_diff_pp median_ret_diff_pp mean_alpha_diff_pp median_alpha_diff_pp ret_hit_rate_pct alpha_hit_rate_pct
            m = re.match(
                r"^\s*\d+\s+(better_return|safer_risk|max_sharpe|equal_weight|mcap_weight)\s+"
                r"([-+0-9.eE+]+)\s+([-+0-9.]+)\s+([-+0-9.eE+]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s*$",
                line,
            )
            if m:
                strat = m.group(1)
                median_ret_diff = float(m.group(3))
                median_alpha_diff = float(m.group(5))
                ret_hit = float(m.group(6))
                alpha_hit = float(m.group(7))
                out[f"strat_{strat}_median_ret_diff"] = median_ret_diff
                out[f"strat_{strat}_median_alpha_diff"] = median_alpha_diff
                out[f"strat_{strat}_ret_hit_rate"] = ret_hit
                out[f"strat_{strat}_alpha_hit_rate"] = alpha_hit
            elif line.startswith("==========") or "Optimizer" in line:
                in_strat_imp = False

    # --- Beta change ---
    in_beta_change = False
    for line in txt.splitlines():
        if "Beta Change (strategy − baseline)" in line:
            in_beta_change = True
            continue
        if in_beta_change:
            m = re.match(
                r"^\s*\d+\s+(better_return|safer_risk|max_sharpe|equal_weight|mcap_weight)\s+"
                r"([-+0-9.eE]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s*$",
                line,
            )
            if m:
                strat = m.group(1)
                mean_dbeta = float(m.group(2))
                pct_inc = float(m.group(5))
                out[f"dbeta_{strat}_mean"] = mean_dbeta
                out[f"dbeta_{strat}_pct_inc"] = pct_inc
            elif line.startswith("==========") or "Cross-Strategy" in line:
                in_beta_change = False

    # --- pct_best_<strategy> values ---
    for line in txt.splitlines():
        m = re.match(r"^\s*pct_best_(\w+)\s+([-+0-9.]+)\s*$", line)
        if m:
            out[f"pct_best_{m.group(1)}"] = float(m.group(2))

    # --- MPT vs Naive direction (lines after "MPT vs Naive Direction") ---
    in_naive = False
    for line in txt.splitlines():
        if "MPT vs Naive Direction" in line:
            in_naive = True
            continue
        if in_naive:
            m = re.match(
                r"^\s*\d+\s+(better_return|safer_risk|max_sharpe)\s+"
                r"(equal_weight|mcap_weight)\s+"
                r"([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s*$",
                line,
            )
            if m:
                mpt = m.group(1)
                naive = m.group(2)
                mean_delta = float(m.group(3))
                pct_closer = float(m.group(5))
                pct_farther = float(m.group(6))
                pct_unchanged = float(m.group(7))
                out[f"naive_{mpt}_{naive}_mean"] = mean_delta
                out[f"naive_{mpt}_{naive}_closer"] = pct_closer
                out[f"naive_{mpt}_{naive}_farther"] = pct_farther
                out[f"naive_{mpt}_{naive}_unchanged"] = pct_unchanged
            elif line.startswith("=========="):
                in_naive = False

    return out


# ---------------------------------------------------------------------------
# Section emitters (in order of appearance in macro.tex)
# ---------------------------------------------------------------------------

def emit_token_metadata(emit: MacroEmitter, manual: Dict[str, str]) -> None:
    """\Coingecko* macros + total accounts head."""
    emit.comment("% --- Token metadata (Coingecko universe) ---")
    emit.macro("CoingeckoInitTokensN", manual["CoingeckoInitTokensN"])
    emit.macro("CoingeckoERCTokensN", manual["CoingeckoERCTokensN"])
    emit.macro("CoingeckoCleanedTokensN", manual["CoingeckoCleanedTokensN"])
    emit.macro("CoingeckoRemovedTokensN", manual["CoingeckoRemovedTokensN"])
    emit.blank()


def emit_total_accounts_raw(emit: MacroEmitter) -> None:
    emit.comment("% --- Total accounts at end-of-study (millions) ---")
    df = pd.read_csv(ACC_DATA / "output_accounts_eoa_vs_ca.csv")
    last = df.iloc[-1]
    total = int(last["eoa_accounts"] + last["ca_accounts"])
    emit.macro("TotalAccountsRaw", str(total))
    emit.macro_raw(r"\def\TotalAccountsRounded/{\the\numexpr\TotalAccountsRaw//1000000\relax\,M}")
    emit.blank()


def emit_ca_vs_eoa(emit: MacroEmitter) -> None:
    """EOA / CA accounts at end-of-study, CAGR, ratios."""
    emit.comment("% --- Account counts: EOA vs CA growth (CAGR + ratios) ---")
    df = pd.read_csv(ACC_DATA / "output_accounts_eoa_vs_ca.csv")
    df["date"] = pd.to_datetime(df["date"])

    last = df.iloc[-1]
    first = df.iloc[0]

    eoa_end = int(last["eoa_accounts"])
    ca_end = int(last["ca_accounts"])
    total_end = eoa_end + ca_end

    emit.macro("EOAAccountsEnd", fmt_num_int(eoa_end))
    emit.macro("CAAccountsEnd", fmt_num_int(ca_end))
    emit.blank()

    # CAGR: (end/start)^(1/years) - 1
    years = (last["date"] - first["date"]).days / 365.25
    eoa_cagr = ((last["eoa_accounts"] / first["eoa_accounts"]) ** (1 / years) - 1) * 100
    ca_cagr = ((last["ca_accounts"] / first["ca_accounts"]) ** (1 / years) - 1) * 100
    # Avg increase per year = total increase / total elapsed years.
    eoa_avg_inc = (last["eoa_accounts"] - first["eoa_accounts"]) / years
    ca_avg_inc = (last["ca_accounts"] - first["ca_accounts"]) / years

    emit.macro("EOACAGR", fmt_si_pct(eoa_cagr, 2))
    emit.macro("EOAAvgIncreaseYear", fmt_num_int(eoa_avg_inc))
    emit.blank()
    emit.macro("CACAGR", fmt_si_pct(ca_cagr, 2))
    emit.macro("CAAvgIncreaseYear", fmt_num_int(ca_avg_inc))
    emit.blank()

    # Aggregate macros (using \TotalAccountsRaw refs)
    emit.macro_raw(r"\def\TotalAccountsEnd/{\num{\TotalAccountsRaw/}} % EOA + CA")
    ca_share = ca_end / total_end * 100
    emit.macro("CAShareEndPct", fmt_si_pct(ca_share, 2))
    emit.blank()


def emit_holders(emit: MacroEmitter) -> None:
    emit.comment("% --- Token holder counts: distribution per year (2020 vs 2025) ---")
    summary = pd.read_csv(THC_DATA / "holders_overall_yearly_summary.csv")
    bk = pd.read_csv(THC_DATA / "holders_overall_yearly_buckets.csv")

    def get(year: int):
        s = summary[summary["year"] == year].iloc[0]
        b = bk[bk["year"] == year].iloc[0]
        return s, b

    s20, b20 = get(2020)
    _,   b24 = get(2024)
    s25, b25 = get(2025)

    # Note: Holders macros use no trailing slash in macro.tex.
    emit.macro("HoldersAvgYtwenty", fmt_num_int(s20["avg_holders"]), slash=False)
    emit.macro("HoldersMedYtwenty", fmt_num_int(s20["median_holders"]), slash=False)
    emit.blank()
    emit.macro("HoldersAvgYtwentyfive", fmt_num_int(s25["avg_holders"]), slash=False)
    emit.macro("HoldersMedYtwentyfive", fmt_num_int(s25["median_holders"]), slash=False)
    emit.blank()

    emit.macro("HoldersPctGtTenKYtwenty", fmt_pct(b20["pct_>10K"], 1), slash=False)
    emit.macro("HoldersPctGtTenKYtwentyfive", fmt_pct(b25["pct_>10K"], 1), slash=False)
    emit.blank()
    emit.macro("HoldersPctOneKTenKYtwenty", fmt_pct(b20["pct_1K–10K"], 1), slash=False)
    emit.macro("HoldersPctOneKTenKYtwentyfour", fmt_pct(b24["pct_1K–10K"], 1), slash=False)
    emit.macro("HoldersPctOneKTenKYtwentyfive", fmt_pct(b25["pct_1K–10K"], 1), slash=False)
    emit.blank()
    emit.macro("HoldersPctHundredOneKYtwenty", fmt_pct(b20["pct_100–1K"], 1), slash=False)
    emit.macro("HoldersPctHundredOneKYtwentyfive", fmt_pct(b25["pct_100–1K"], 1), slash=False)
    emit.blank()


def emit_wealth_buckets(emit: MacroEmitter) -> None:
    emit.comment("% --- Wealth buckets: average $-bucket allocation by wallet type ---")
    df = pd.read_csv(DOW_DATA / "value_buckets_minimal.csv")
    # Buckets in macro: 0-1, 1-100, 100-1K, 1K-10K, 10K-100K, >100K
    BUCKET_MAP = [
        ("ZeroOneP",      "0–1"),
        ("OneHundredP",   "1–100"),
        ("HundredOneKP",  "100–1K"),
        ("OneKTenKP",     "1K–10K"),
        ("TenKHundredKP", "10K–100K"),
        ("OverHundredKP", ">100K"),
    ]

    def bucket_means(wt: str) -> Dict[str, float]:
        sub = df[df["wallet_type"] == wt]
        return sub.groupby("bucket")["wallet_pct"].mean().to_dict()

    ca_means = bucket_means("CA")
    eoa_means = bucket_means("EOA")

    emit.comment("% --- CA ---")
    for suffix, bucket in BUCKET_MAP:
        emit.macro(f"WealthBucketCA{suffix}", rf"\num{{{ca_means[bucket]:.2f}}}")
    emit.blank()

    emit.comment("% --- EOA ---")
    for suffix, bucket in BUCKET_MAP:
        emit.macro(f"WealthBucketEOA{suffix}", rf"\num{{{eoa_means[bucket]:.2f}}}")
    emit.blank()


def emit_wealth_share_stats(emit: MacroEmitter) -> None:
    emit.comment("% --- Wealth share: CA vs EOA fraction of total value over time ---")
    df = pd.read_csv(DOW_DATA / "value_buckets_minimal.csv")
    ca_sum = df[df["wallet_type"] == "CA"].groupby("block")["value_sum_usd"].sum()
    eoa_sum = df[df["wallet_type"] == "EOA"].groupby("block")["value_sum_usd"].sum()
    total = ca_sum + eoa_sum
    ca_share = ca_sum / total * 100
    eoa_share = 100 - ca_share

    def stats(s: pd.Series) -> Dict[str, float]:
        return dict(
            mean=s.mean(),
            median=s.median(),
        )

    sca = stats(ca_share)
    seoa = stats(eoa_share)

    emit.comment("% --- CA share stats ---")
    emit.macro("WealthShareCAMeanP", rf"\num{{{sca['mean']:.2f}}}")
    emit.blank()

    emit.comment("% --- EOA share stats ---")
    emit.macro("WealthShareEOAMeanP", rf"\num{{{seoa['mean']:.2f}}}")
    emit.blank()


def emit_l1_distance_main(emit: MacroEmitter, manual: Dict[str, str]) -> None:
    emit.comment("% --- L1 distance: observed vs MPT-optimal portfolios ---")
    emit.macro("LOneObservationsRaw", manual["LOneObservationsRaw"])
    emit.macro_raw(r"\def\LOneObservationsN/{\num{\LOneObservationsRaw/}}")
    emit.macro_raw(r"\def\LOneObservationsM/{\the\numexpr\LOneObservationsRaw//1000000\relax\,M}")
    emit.blank()

    # Time-averaged means / medians from per_block_distance_stats.csv (token_group=all)
    pb = pd.read_csv(DRA_DATA / "per_block_distance_stats.csv")
    pb_all = pb[pb["token_group"] == "all"]

    def opt_stats(opt: str) -> Tuple[float, float, float, float]:
        sub = pb_all[pb_all["optimisation"] == opt]
        return sub["mean_dist"].mean(), sub["median_dist"].mean(), sub["mean_dist"].min(), sub["mean_dist"].max()

    sr_mean, sr_med, _, _ = opt_stats("safer_risk")
    br_mean, br_med, br_lo, br_hi = opt_stats("better_return")
    ms_mean, ms_med, ms_lo, ms_hi = opt_stats("max_sharpe")

    emit.comment("% Time-averaged means and medians per strategy (72 monthly snapshots)")
    emit.macro("LOneMeanSaferRisk", fmt_dec(sr_mean, 2))
    emit.macro("LOneMeanBetterReturn", fmt_dec(br_mean, 2))
    emit.macro("LOneMeanMaxSharpe", fmt_dec(ms_mean, 2))
    emit.macro("LOneMedianBetterReturn", fmt_dec(br_med, 2))
    emit.macro("LOneMedianMaxSharpe", fmt_dec(ms_med, 2))
    n_blocks = pb["block_number"].nunique()
    emit.macro("LOneMonthsN", str(n_blocks))
    emit.blank()

    emit.comment("% Distribution buckets per strategy (share with L1 <= 1)")
    # Combined <=1% share comes from zero_distance_counts.csv (exact_0 + (0,1]).
    zd = pd.read_csv(DRA_DATA / "distance_analysis_results" / "zero_distance_counts.csv")
    zd = zd.set_index("optimisation")
    emit.macro("LOneSrCombinedLeOnePct", fmt_dec(zd.at["safer_risk",    "combined_<=1%_pct"], 2))
    emit.macro("LOneBrCombinedLeOnePct", fmt_dec(zd.at["better_return", "combined_<=1%_pct"], 2))
    emit.macro("LOneMsCombinedLeOnePct", fmt_dec(zd.at["max_sharpe",    "combined_<=1%_pct"], 2))
    # MaxSR-specific distance bucket shares from distance_distribution_pct.csv.
    dd = pd.read_csv(DRA_DATA / "distance_analysis_results" / "distance_distribution_pct.csv")
    dd = dd.set_index("bucket")
    emit.macro("LOneMsHighBinPct", fmt_dec(dd.at["(80,100]", "max_sharpe"], 2))
    emit.macro("LOneMs60to80Pct",  fmt_dec(dd.at["(60,80]",  "max_sharpe"], 2))
    emit.blank()

    emit.comment("% Cross-strategy Pearson correlations (full data)")
    # Strategy×strategy Pearson on the full 239M parquet — output of
    # mpt_distance_analysis.py SECTION 7a (CORR(...) over the parquet).
    ct = pd.read_csv(DRA_DATA / "distance_analysis_results" / "cross_target_correlation.csv",
                     index_col=0)
    emit.macro("LOneCorrBrSr", fmt_dec(ct.at["corr_br_sr", "pearson_r"], 3))
    emit.macro("LOneCorrBrMs", fmt_dec(ct.at["corr_br_ms", "pearson_r"], 3))
    emit.macro("LOneCorrSrMs", fmt_dec(ct.at["corr_sr_ms", "pearson_r"], 3))
    emit.blank()

    emit.comment("% Joint optimality across strategies")
    # Counts of wallets simultaneously at L1=0 under multiple objectives,
    # from mpt_distance_analysis.py SECTION 8b. CSV: 4 rows × (count, pct).
    mw = pd.read_csv(DRA_DATA / "distance_analysis_results" / "multi_optimal_wallets.csv")
    mw = mw.set_index("metric")
    emit.macro("LOneJointAllThreeN",   fmt_num_int(mw.at["opt_all_3",     "count"]))
    emit.macro("LOneJointAllThreePct", fmt_dec(mw.at["opt_all_3",     "pct"], 3))
    emit.macro("LOneJointBrSrPct",     fmt_dec(mw.at["opt_br_and_sr", "pct"], 2))
    emit.macro("LOneJointBrMsPct",     fmt_dec(mw.at["opt_br_and_ms", "pct"], 2))
    emit.macro("LOneJointSrMsPct",     fmt_dec(mw.at["opt_sr_and_ms", "pct"], 2))
    emit.blank()

    emit.comment("% Near-optimal (L1 <= 1%)")
    two_token_pcts, median_values = [], []
    for opt in ("better_return", "safer_risk", "max_sharpe"):
        td = pd.read_csv(DRA_DATA / "distance_analysis_results" / f"low_dist_tokens_dist_{opt}.csv")
        two_token_pcts.append(float(td.loc[td["num_tokens"] == 2, "pct"].iloc[0]))
        st = pd.read_csv(DRA_DATA / "distance_analysis_results" / f"low_dist_stats_{opt}.csv",
                         index_col=0)
        median_values.append(float(st.at["median_value", "value"]))

    emit.macro("LOneNearOptTwoTokenPctLo", str(round(min(two_token_pcts))))
    emit.macro("LOneNearOptTwoTokenPctHi", str(round(max(two_token_pcts))))
    emit.macro("LOneNearOptMedianValueLo", str(math.ceil(min(median_values))))
    emit.macro("LOneNearOptMedianValueHi", str(math.ceil(max(median_values))))

    zd = pd.read_csv(DRA_DATA / "distance_analysis_results" / "zero_distance_counts.csv")
    zd = zd.set_index("optimisation")
    STRAT_MAP = {"better_return": "Br", "safer_risk": "Sr", "max_sharpe": "Ms"}
    for opt, suf in STRAT_MAP.items():
        n_millions = zd.at[opt, "combined_<=1%_count"] / 1_000_000
        pct = zd.at[opt, "combined_<=1%_pct"]
        emit.macro(f"LOneNearOpt{suf}N",   fmt_dec(n_millions, 1))
        emit.macro(f"LOneNearOpt{suf}Pct", fmt_dec(pct, 1))
    emit.blank()

    emit.comment("% OLS R-squared (base + extended)")
    ols_summary = pd.read_csv(DRA_DATA / "distance_analysis_results" / "ols_summary.csv")
    ols_r2 = ols_summary.set_index("target")["R2"]
    emit.macro("LOneOlsBrRsq", fmt_dec(ols_r2["better_return"], 3))
    emit.macro("LOneOlsSrRsq", fmt_dec(ols_r2["safer_risk"], 3))
    emit.macro("LOneOlsMsRsq", fmt_dec(ols_r2["max_sharpe"], 3))
    for k in ["LOneOlsExtBrRsq", "LOneOlsExtSrRsq", "LOneOlsExtMsRsq"]:
        emit.macro(k, manual[k])
    emit.blank()

    emit.comment("% Random Forest fit (R^2 + MAE) and feature importances")
    rf = pd.read_csv(DRA_DATA / "distance_analysis_results" / "random_forest_importance.csv")

    # Per-strategy R² and MAE (constant within each target).
    rf_by_target = rf.groupby("target").first()
    STRAT_MAP = {"better_return": "Br", "safer_risk": "Sr", "max_sharpe": "Ms"}
    for target, suf in STRAT_MAP.items():
        emit.macro(f"LOneRf{suf}Rsq", fmt_dec(rf_by_target.at[target, "R2_test"], 3))
        emit.macro(f"LOneRf{suf}Mae", fmt_dec(rf_by_target.at[target, "MAE_test"], 2))

    # Single-strategy importance: safer_risk num_tokens dominance (× 100).
    sr_tokens = rf[(rf["target"] == "safer_risk") & (rf["feature"] == "num_tokens")]["importance"].iloc[0]
    emit.macro("LOneRfSrTokensImp", fmt_dec(sr_tokens * 100, 1))

    # Cross-strategy ranges across {better_return, max_sharpe} (the "BrMs" pair).
    def _imp_pair(feat: str) -> Tuple[float, float]:
        v = rf[rf["target"].isin(["better_return", "max_sharpe"]) & (rf["feature"] == feat)]["importance"] * 100
        return float(v.min()), float(v.max())

    val_lo, val_hi = _imp_pair("log_value_usd")
    tok_lo, tok_hi = _imp_pair("num_tokens")
    mon_lo, mon_hi = _imp_pair("month")
    emit.macro("LOneRfBrMsValueImp",      fmt_dec(val_hi, 0))
    emit.macro("LOneRfBrMsTokensImpLo",   fmt_dec(tok_lo, 0))
    emit.macro("LOneRfBrMsTokensImpHi",   fmt_dec(tok_hi, 0))
    emit.macro("LOneRfBrMsMonthImpLo",    fmt_dec(mon_lo, 0))
    emit.macro("LOneRfBrMsMonthImpHi",    fmt_dec(mon_hi, 0))
    emit.blank()

    emit.comment("% Spearman/Pearson correlations on 50M-row sample")
    # Long-format CSV: (feature, target, pearson_r, spearman_r). We only
    # need the num_tokens × {safer_risk, better_return, max_sharpe} rows
    # plus the Pearson on num_tokens × safer_risk.
    cs = pd.read_csv(DRA_DATA / "distance_analysis_results" / "correlation_sample.csv")
    def _corr(feat: str, target: str, col: str) -> float:
        row = cs[(cs["feature"] == feat) & (cs["target"] == target)].iloc[0]
        return float(row[col])
    emit.macro("LOneSpearmanSrTokens", fmt_dec(_corr("num_tokens", "safer_risk",    "spearman_r"), 3))
    emit.macro("LOnePearsonSrTokens",  fmt_dec(_corr("num_tokens", "safer_risk",    "pearson_r"),  3))
    emit.macro("LOneSpearmanBrTokens", fmt_dec(_corr("num_tokens", "better_return", "spearman_r"), 3))
    emit.macro("LOneSpearmanMsTokens", fmt_dec(_corr("num_tokens", "max_sharpe",    "spearman_r"), 3))
    emit.blank()

    emit.comment("% --- 5+ token robustness check ---")
    GR5 = DRA_DATA / "distance_analysis_results_gr_5"

    # ≤1% combined share, per strategy.
    zd5 = pd.read_csv(GR5 / "zero_distance_counts.csv").set_index("optimisation")
    emit.macro("LOneFivePlusBrLeOnePct", fmt_dec(zd5.at["better_return", "combined_<=1%_pct"], 2))
    emit.macro("LOneFivePlusSrLeOnePct", fmt_dec(zd5.at["safer_risk",    "combined_<=1%_pct"], 2))
    emit.macro("LOneFivePlusMsLeOnePct", fmt_dec(zd5.at["max_sharpe",    "combined_<=1%_pct"], 2))

    # max_sharpe high-distance bucket (80,100] share.
    dd5 = pd.read_csv(GR5 / "distance_distribution_pct.csv")
    high_bin = float(dd5.loc[dd5["bucket"] == "(80,100]", "max_sharpe"].iloc[0])
    emit.macro("LOneFivePlusMsHighBinPct", fmt_dec(high_bin, 1))

    # RF safer_risk R² + max_sharpe month importance (× 100).
    rf5 = pd.read_csv(GR5 / "random_forest_importance.csv")
    sr_r2  = rf5[rf5["target"] == "safer_risk"]["R2_test"].iloc[0]
    ms_mon = rf5[(rf5["target"] == "max_sharpe") & (rf5["feature"] == "month")]["importance"].iloc[0]
    emit.macro("LOneFivePlusRfSrRsq",     fmt_dec(sr_r2, 3))
    emit.macro("LOneFivePlusMsMonthImp",  fmt_dec(ms_mon * 100, 1))

    # Spearman correlation: safer_risk distance vs num_tokens (sample).
    cs5 = pd.read_csv(GR5 / "correlation_sample.csv")
    sr_tok_rho = cs5[(cs5["feature"] == "num_tokens") & (cs5["target"] == "safer_risk")]["spearman_r"].iloc[0]
    emit.macro("LOneFivePlusSpearmanSrTokens", fmt_dec(sr_tok_rho, 3))

    # Cross-strategy l1 correlation (BR × MS) on the 5+ subset.
    ct5 = pd.read_csv(GR5 / "cross_target_correlation.csv", index_col=0)
    emit.macro("LOneFivePlusCorrBrMs", fmt_dec(ct5.at["corr_br_ms", "pearson_r"], 3))
    emit.blank()
def emit_wealth_composition(emit: MacroEmitter, manual: Dict[str, str]) -> None:
    emit.comment("% --- Wealth composition: peak total-value snapshots ---")
    for k in ["WealthPeakTwentyone", "WealthPeakTwentyfiveB"]:
        emit.macro(k, manual[k])
    emit.blank()


def emit_top_wallet_concentration(emit: MacroEmitter, manual: Dict[str, str]) -> None:
    emit.comment("% --- Top wallet wealth concentration ---")
    emit.macro("TopOneWealthTwenty", manual["TopOneWealthTwenty"])
    emit.macro("TopOneWealthTwentyfive", manual["TopOneWealthTwentyfive"])
    emit.macro("TopFiveTenWealthMin", manual["TopFiveTenWealthMin"])
    emit.blank()


def emit_portfolio_size_distribution(emit: MacroEmitter) -> None:
    emit.comment("% --- Portfolio size distribution (avg across 72 monthly snapshots) ---")
    df = pd.read_csv(PSZ_DATA / "portfolio_size_hist_by_wealth.csv")

    def cat(b):
        if b == "1": return "1"
        if b == "2": return "2"
        if b in ("3", "4"): return "3-4"
        if b in ("5", "6–10", "11–20", "21–50", "51–100", ">100"): return "5+"
        if b == "0": return "0"
        return None

    df["cat"] = df["size_bucket"].apply(cat)
    pb = df.groupby(["block", "cat"])["wallet_count"].sum().unstack(fill_value=0)
    pb["total"] = pb.sum(axis=1)
    pcts = {}
    for c in ("1", "2", "3-4", "5+"):
        pcts[c] = (pb[c] / pb["total"] * 100).mean()

    emit.macro("PortSizeOneTokenPct", fmt_dec(pcts["1"], 2))
    emit.macro("PortSizeTwoTokenPct", fmt_dec(pcts["2"], 2))
    emit.macro("PortSizeThreeFourTokenPct", fmt_dec(pcts["3-4"], 2))
    emit.macro("PortSizeFivePlusPct", fmt_dec(pcts["5+"], 2))

    # 'two of multi' (per-block ratio averaged)
    multi = pb["2"] + pb["3-4"] + pb["5+"]
    two_of_multi = (pb["2"] / multi * 100).mean()
    emit.macro("PortSizeTwoOfMultiPct", fmt_dec(two_of_multi, 1))
    emit.blank()


def emit_naive_benchmark(emit: MacroEmitter, capm: Dict[str, float], manual: Dict[str, str]) -> None:
    emit.comment("% --- L1 distance: baseline portfolio vs naive (equal/mcap) benchmarks ---")
    emit.comment("% L1 distance: baseline vs naive (overall)")
    emit.macro("LOneNaiveEqualMean", fmt_dec(capm["l1_baseline_vs_equal_mean"], 1))
    emit.macro("LOneNaiveEqualMedian", fmt_dec(capm["l1_baseline_vs_equal_median"], 1))
    emit.macro("LOneNaiveMcapMean", fmt_dec(capm["l1_baseline_vs_mcap_mean"], 1))
    emit.macro("LOneNaiveMcapMedian", fmt_dec(capm["l1_baseline_vs_mcap_median"], 1))
    emit.blank()

    # MPT vs naive direction
    def naive(mpt, naive_name, key, dp=3):
        return fmt_dec(capm[f"naive_{mpt}_{naive_name}_{key}"], dp)

    def naive_signed(mpt, naive_name, dp=3):
        v = capm[f"naive_{mpt}_{naive_name}_mean"]
        if v >= 0:
            return f"+{round(v, dp):.{dp}f}"
        return f"{round(v, dp):.{dp}f}"

    emit.comment("% MPT vs Naive direction: Equal-weight")
    emit.macro("LOneDeltaBrEqualMean", naive_signed("better_return", "equal_weight"))
    emit.macro("LOneDeltaBrEqualCloserPct", fmt_dec(capm["naive_better_return_equal_weight_closer"], 1))
    emit.macro("LOneDeltaBrEqualFartherPct", fmt_dec(capm["naive_better_return_equal_weight_farther"], 1))
    emit.macro("LOneDeltaBrEqualUnchangedPct", fmt_dec(capm["naive_better_return_equal_weight_unchanged"], 1))
    emit.macro("LOneDeltaSrEqualMean", naive_signed("safer_risk", "equal_weight"))
    emit.macro("LOneDeltaSrEqualCloserPct", fmt_dec(capm["naive_safer_risk_equal_weight_closer"], 1))
    emit.macro("LOneDeltaSrEqualFartherPct", fmt_dec(capm["naive_safer_risk_equal_weight_farther"], 1))
    emit.macro("LOneDeltaSrEqualUnchangedPct", fmt_dec(capm["naive_safer_risk_equal_weight_unchanged"], 1))
    emit.macro("LOneDeltaMsEqualMean", naive_signed("max_sharpe", "equal_weight"))
    emit.macro("LOneDeltaMsEqualCloserPct", fmt_dec(capm["naive_max_sharpe_equal_weight_closer"], 1))
    emit.macro("LOneDeltaMsEqualFartherPct", fmt_dec(capm["naive_max_sharpe_equal_weight_farther"], 1))
    emit.macro("LOneDeltaMsEqualUnchangedPct", fmt_dec(capm["naive_max_sharpe_equal_weight_unchanged"], 1))
    emit.blank()

    emit.comment("% MPT vs Naive direction: MCap-weight")
    emit.macro("LOneDeltaBrMcapMean", naive_signed("better_return", "mcap_weight"))
    emit.macro("LOneDeltaBrMcapCloserPct", fmt_dec(capm["naive_better_return_mcap_weight_closer"], 1))
    emit.macro("LOneDeltaBrMcapFartherPct", fmt_dec(capm["naive_better_return_mcap_weight_farther"], 1))
    emit.macro("LOneDeltaBrMcapUnchangedPct", fmt_dec(capm["naive_better_return_mcap_weight_unchanged"], 1))
    emit.macro("LOneDeltaSrMcapMean", naive_signed("safer_risk", "mcap_weight"))
    emit.macro("LOneDeltaSrMcapCloserPct", fmt_dec(capm["naive_safer_risk_mcap_weight_closer"], 1))
    emit.macro("LOneDeltaSrMcapFartherPct", fmt_dec(capm["naive_safer_risk_mcap_weight_farther"], 1))
    emit.macro("LOneDeltaSrMcapUnchangedPct", fmt_dec(capm["naive_safer_risk_mcap_weight_unchanged"], 1))
    emit.macro("LOneDeltaMsMcapMean", naive_signed("max_sharpe", "mcap_weight"))
    emit.macro("LOneDeltaMsMcapCloserPct", fmt_dec(capm["naive_max_sharpe_mcap_weight_closer"], 1))
    emit.macro("LOneDeltaMsMcapFartherPct", fmt_dec(capm["naive_max_sharpe_mcap_weight_farther"], 1))
    emit.macro("LOneDeltaMsMcapUnchangedPct", fmt_dec(capm["naive_max_sharpe_mcap_weight_unchanged"], 1))
    emit.blank()

    emit.comment("% Safer Risk closer-to-farther ratio vs equal")
    emit.macro("LOneDeltaSrEqualRatio", manual["LOneDeltaSrEqualRatio"])
    emit.blank()
def emit_capm(emit: MacroEmitter, capm: Dict[str, float], manual: Dict[str, str]) -> None:
    emit.comment("% --- CAPM: alpha/beta of strategies vs market benchmark ---")
    emit.comment("% Beta summary (baseline median + IQR)")
    emit.macro("CAPMBetaBaselineMedian", fmt_dec(capm["beta_baseline_median"], 3))
    emit.macro("CAPMBetaBaselinePtwentyfive", fmt_dec(capm["beta_baseline_p25"], 2))
    emit.macro("CAPMBetaBaselinePseventyfive", fmt_dec(capm["beta_baseline_p75"], 2))
    emit.blank()

    emit.comment("% Alpha medians per strategy")
    emit.macro("CAPMAlphaBaselineMedian", fmt_dec(capm["alpha_baseline_median"], 2))
    emit.macro("CAPMAlphaMcapMedian", fmt_dec(capm["alpha_mcap_weight_median"], 2))
    emit.macro("CAPMAlphaEqualMedian", fmt_dec(capm["alpha_equal_weight_median"], 2))
    emit.macro("CAPMAlphaBetterMedian", fmt_dec(capm["alpha_better_return_median"], 2))
    emit.macro("CAPMAlphaSaferMedian", fmt_dec(capm["alpha_safer_risk_median"], 2))
    emit.macro("CAPMAlphaMaxSharpeMedian", fmt_dec(capm["alpha_max_sharpe_median"], 2))
    emit.blank()

    emit.comment("% Share of blocks with positive alpha (per strategy)")
    emit.macro("CAPMFracPositiveBaseline", fmt_dec(capm["alpha_baseline_frac_positive"], 1))
    emit.macro("CAPMFracPositiveMcap", fmt_dec(capm["alpha_mcap_weight_frac_positive"], 1))
    emit.macro("CAPMFracPositiveEqual", fmt_dec(capm["alpha_equal_weight_frac_positive"], 1))
    emit.macro("CAPMFracPositiveBetter", fmt_dec(capm["alpha_better_return_frac_positive"], 1))
    emit.macro("CAPMFracPositiveSafer", fmt_dec(capm["alpha_safer_risk_frac_positive"], 1))
    emit.macro("CAPMFracPositiveMaxSharpe", fmt_dec(capm["alpha_max_sharpe_frac_positive"], 1))
    emit.blank()

    emit.comment("% Median realized returns per strategy + market benchmark")
    emit.macro("CAPMRetBaselineMedian", fmt_dec(capm["ret_baseline_median"], 2))
    emit.macro("CAPMRetBetterMedian", fmt_dec(capm["ret_better_return_median"], 2))
    emit.macro("CAPMRetSaferMedian", fmt_dec(capm["ret_safer_risk_median"], 2))
    emit.macro("CAPMRetMaxSharpeMedian", fmt_dec(capm["ret_max_sharpe_median"], 2))
    emit.macro("CAPMRetEqualMedian", fmt_dec(capm["ret_equal_weight_median"], 2))
    emit.macro("CAPMRetMcapMedian", fmt_dec(capm["ret_mcap_weight_median"], 2))
    rets = pd.read_csv(DRA_DATA / "camp_analysis_results_full" / "returns_per_block.csv")
    market_med = float(rets["median_market_return"].median()) * 100
    emit.macro("CAPMRetMarketMedian", fmt_dec(market_med, 2))
    emit.blank()

    emit.comment("% Return hit rates vs baseline (share of blocks beating baseline)")
    emit.macro("CAPMRetHitBetter", fmt_dec(capm["strat_better_return_ret_hit_rate"], 1))
    emit.macro("CAPMRetHitSafer", fmt_dec(capm["strat_safer_risk_ret_hit_rate"], 1))
    emit.macro("CAPMRetHitMaxSharpe", fmt_dec(capm["strat_max_sharpe_ret_hit_rate"], 1))
    emit.macro("CAPMRetHitEqual", fmt_dec(capm["strat_equal_weight_ret_hit_rate"], 1))
    emit.macro("CAPMRetHitMcap", fmt_dec(capm["strat_mcap_weight_ret_hit_rate"], 1))
    emit.blank()

    emit.comment("% Spearman: L1 gap and beta vs alpha (signed)")
    def signed(v, dp=3):
        if v >= 0:
            return f"+{round(v, dp):.{dp}f}"
        return f"{round(v, dp):.{dp}f}"

    emit.macro("CAPMSpearmanSrAlpha", signed(capm["spearman_l1_gap_safer_risk_vs_alpha_baseline"], 3))
    emit.macro("CAPMSpearmanMsAlpha", signed(capm["spearman_l1_gap_max_sharpe_vs_alpha_baseline"], 3))
    emit.macro("CAPMSpearmanBetaAlpha", signed(capm["spearman_beta_baseline_vs_alpha_baseline"], 3))
    emit.blank()

    emit.comment("% Beta delta: mcap-weight vs baseline (mean)")
    emit.macro("CAPMDeltaBetaMcapMean", fmt_dec(capm["dbeta_mcap_weight_mean"], 3))
    emit.blank()

    emit.comment("% Dataset size: observations, blocks, horizon")
    emit.macro("CAPMObsRaw", manual["CAPMObsRaw"])
    emit.macro("CAPMBlocksN", manual["CAPMBlocksN"])
    emit.macro("CAPMHorizonDays", manual["CAPMHorizonDays"])
    emit.blank()


def emit_pct_beat_block_aggregates(emit: MacroEmitter) -> None:
    """Per-block aggregates of pct-beat-market and pct-positive-alpha.

    Source: Analytics/distance_return_analytics/data/pct_beat_market_per_block.csv
    Columns: pct_beat_market_ret_<strategy>, pct_positive_alpha_<strategy>
    Strategies: baseline, better_return, safer_risk, max_sharpe, equal_weight, mcap_weight

    Convention:
      * BlockMedian / BlockMean → median / mean across the 71 blocks
      * MedianStrategySpread    → max(median) − min(median) across strategies
        - PctBeatRet excludes baseline
        - PctPosAlpha includes baseline
      * BlocksBelowFifty / TwentyFive → % of blocks where baseline pct_beat < 50 / 25
      * CrossStratCorrLo / Hi → min / max pairwise Pearson corr of the
        six pct_beat columns (excluding the diagonal).
    """
    emit.comment("% --- Per-block aggregates: pct beating market / pct positive alpha ---")
    df = pd.read_csv(DRA_DATA / "pct_beat_market_per_block.csv")

    STRAT_MAP = {
        "baseline":      "Baseline",
        "better_return": "Better",
        "safer_risk":    "Safer",
        "max_sharpe":    "MaxSharpe",
        "equal_weight":  "Equal",
        "mcap_weight":   "Mcap",
    }

    # Per-strategy block-level median + mean for both ret-beat and alpha-positive.
    for strat, suf in STRAT_MAP.items():
        beat = df[f"pct_beat_market_ret_{strat}"]
        alph = df[f"pct_positive_alpha_{strat}"]
        emit.macro(f"PctBeatRet{suf}BlockMedian",   fmt_dec(beat.median(), 2))
        emit.macro(f"PctBeatRet{suf}BlockMean",     fmt_dec(beat.mean(), 2))
        emit.macro(f"PctPosAlpha{suf}BlockMedian",  fmt_dec(alph.median(), 2))
        emit.macro(f"PctPosAlpha{suf}BlockMean",    fmt_dec(alph.mean(), 2))
    emit.blank()

    # Baseline-strategy block extremes (used in narrative as "best/worst block").
    baseline_beat = df["pct_beat_market_ret_baseline"]
    emit.macro("PctBeatRetBaselineBlockMin", fmt_dec(baseline_beat.min(), 2))
    emit.macro("PctBeatRetBaselineBlockMax", fmt_dec(baseline_beat.max(), 2))
    emit.blank()

    # Median-strategy spread:
    #   PctBeat excludes baseline (paper uses spread of MPT vs naive strategies).
    #   PctPosAlpha includes baseline.
    beat_medians_no_base = [df[f"pct_beat_market_ret_{s}"].median()
                            for s in STRAT_MAP if s != "baseline"]
    alpha_medians_all = [df[f"pct_positive_alpha_{s}"].median() for s in STRAT_MAP]
    emit.macro("PctBeatRetMedianStrategySpread",
               fmt_dec(max(beat_medians_no_base) - min(beat_medians_no_base), 1))
    emit.macro("PctPosAlphaMedianStrategySpread",
               fmt_dec(max(alpha_medians_all) - min(alpha_medians_all), 1))
    emit.blank()

    # Cross-strategy Pearson correlations of the six pct_beat columns.
    beat_cols = [f"pct_beat_market_ret_{s}" for s in STRAT_MAP]
    corr = df[beat_cols].corr().to_numpy()
    # Off-diagonal entries only.
    n = corr.shape[0]
    off = [corr[i, j] for i in range(n) for j in range(n) if i != j]
    emit.macro("PctBeatCrossStratCorrLo", fmt_dec(min(off), 3))
    emit.blank()

    # Share of blocks where baseline pct_beat falls below 50.
    n_blocks = len(df)
    pct_below_50 = (baseline_beat < 50).sum() / n_blocks * 100
    emit.macro("PctBlocksBelowFifty", fmt_dec(pct_below_50, 1))
    emit.blank()


def emit_beat_market_quantiles(emit: MacroEmitter) -> None:
    """Cross-sectional beat-market margin quantiles per strategy + metric.

    Source: excess_return_quantiles_per_block.csv (output of
    analyze_pct_beat_market.py). For each (strategy, metric, quantile)
    triple we take the median of the per-block percentile values
    across all blocks and convert to percentage points.

    Only macros actually referenced in section 6 / Table 4 are emitted:
      - {BeatExcRet,BeatAlpha}{Strategy}{BotOne,Med,TopOne}  (36 macros)
      - {BeatExcRet,BeatAlpha}{Strategy}Skew = Q99+Q01       (12 macros)
      - BeatExcRetBaselineBotOneAbs = |Q01|                  (1 macro)

    Naming convention: ALPHABETIC quantile suffixes (BotOne, Med,
    TopOne, ...). Digits are catcode 12 in TeX so a name like
    \\FooQ01/ defines \\FooQ with delimiter pattern "01/" — sibling
    definitions silently collide. The alphabetic-suffix scheme avoids
    the bug entirely. See QUANTILE_TOK below for the full mapping.

    Negative-value rendering: values use ``$-$`` so the macro emits a
    proper math minus when expanded in text mode. The macro itself
    must NOT be placed inside ``$...$`` (the trailing-slash delimiter
    would clash with a slash inside math mode).
    """
    emit.comment("% --- Beat-market margin: cross-sectional quantiles per strategy ---")
    df = pd.read_csv(DRA_DATA / "excess_return_quantiles_per_block.csv")

    g = (df.groupby(["strategy", "metric", "quantile"])["value"]
            .median()
            .reset_index())
    g["value_pct"] = g["value"] * 100.0

    STRAT_TOK = {
        "baseline":      "Baseline",
        "better_return": "Better",
        "safer_risk":    "Safer",
        "max_sharpe":    "MaxSharpe",
        "equal_weight":  "Equal",
        "mcap_weight":   "Mcap",
    }
    METR_TOK = {
        "excess_return": "BeatExcRet",
        "alpha":         "BeatAlpha",
    }
    # Alphabetic-only suffixes — digit suffixes collide in TeX (catcode 12).
    QUANTILE_TOK = {
        0.01: "BotOne",
        0.50: "Med",
        0.99: "TopOne",
    }

    def fmt_signed(x: float) -> str:
        """`$-$X.XX` for negative, `X.XX` for non-negative — math-minus
        compatible when used in text mode."""
        if x < 0:
            return f"$-${abs(x):.2f}"
        return f"{x:.2f}"

    for metric_csv, m_tok in METR_TOK.items():
        for strat_csv, s_tok in STRAT_TOK.items():
            prefix = f"{m_tok}{s_tok}"
            sub = g[(g["strategy"] == strat_csv) & (g["metric"] == metric_csv)]
            for q, q_tok in QUANTILE_TOK.items():
                v = sub.loc[sub["quantile"] == q, "value_pct"].iloc[0]
                emit.macro(f"{prefix}{q_tok}", fmt_signed(v))
            q01 = sub.loc[sub["quantile"] == 0.01, "value_pct"].iloc[0]
            q99 = sub.loc[sub["quantile"] == 0.99, "value_pct"].iloc[0]
            emit.macro(f"{prefix}Skew", fmt_signed(q99 + q01))

    # Single BotOneAbs convenience macro — only used for Baseline ExcRet
    # in the prose ("the bottom one percent return X% below it").
    base_q01 = g[(g["strategy"] == "baseline") &
                 (g["metric"] == "excess_return") &
                 (g["quantile"] == 0.01)]["value_pct"].iloc[0]
    emit.macro("BeatExcRetBaselineBotOneAbs", f"{abs(base_q01):.2f}")
    emit.blank()



def emit_return_analysis(emit: MacroEmitter, manual: Dict[str, str]) -> None:
    emit.comment("% --- Return analysis: realized 20-day returns ---")

    emit.comment("% Distance bucket median returns (safer_risk, baseline cohort)")
    # SAFER_RISK distance bucket median returns
    # (computed offline from mpt_return_analysis_full.md; see manual_constants.py).
    emit.macro("RetDistZeroMedianBaseline", manual["RetDistZeroMedianBaseline"])
    emit.macro("RetDistHighMedianBaseline", manual["RetDistHighMedianBaseline"])
    emit.blank()

    emit.comment("% Return distribution: share of negative-return blocks")
    emit.macro("RetNegativePctLo", manual["RetNegativePctLo"])
    emit.macro("RetNegativePctHi", manual["RetNegativePctHi"])
    emit.blank()


def emit_l1_fitting(emit: MacroEmitter) -> None:
    emit.comment("% --- L1 fitting: power-decay parametric fits per strategy ---")
    df = pd.read_csv(DRA_DATA / "l1_formula_fits.csv")
    df = df[df["statistic"] == "mean"]
    # In macro.tex: FitPsiMRV, FitGammaMRV, FitDinfMRV, FitRsqMRV, FitMaeMRV
    # The CSV cols are a, b, c, r2, mae and `a` is psi-base (113.18 for MRV).
    # macro values:  FitPsiMRV/=1.1319 corresponds to a/100; FitGammaMRV/=0.4803 corresponds to b
    STRAT_MAP = {"better_return": "MRV", "safer_risk": "MVR", "max_sharpe": "MSR"}
    for opt, suf in STRAT_MAP.items():
        row = df[df["strategy"] == opt].iloc[0]
        # ψ (psi) = a / c (d_inf); macro displays the rescaled coefficient.
        psi   = row["a"] / row["c"]
        gamma = row["b"]
        dinf  = row["c"]
        r2    = row["r2"]
        mae   = row["mae"]
        emit.macro(f"FitPsi{suf}",   f"{psi:.4f}")
        emit.macro(f"FitGamma{suf}", f"{gamma:.4f}")
        emit.macro(f"FitDinf{suf}",  f"{dinf:.2f}")
        emit.macro(f"FitRsq{suf}",   f"{r2:.4f}")
        emit.macro(f"FitMae{suf}",   f"{mae:.2f}")
    emit.blank()
def emit_top1_breakdown(emit: MacroEmitter) -> None:
    emit.comment("% --- Top-1% CA/EOA wealth concentration breakdown ---")
    df = pd.read_csv(DOW_DATA / "top_1_pct_ca_eoa_breakdown.csv")
    df01 = pd.read_csv(DOW_DATA / "top_0_1_pct_ca_eoa_breakdown.csv")
    s1 = pd.read_csv(DOW_DATA / "top_1_pct_summary.csv")
    s01 = pd.read_csv(DOW_DATA / "top_0_1_pct_summary.csv")

    # Top 1% – averages across blocks
    ca = df[df["wallet_type"] == "CA"]
    eoa = df[df["wallet_type"] == "EOA"]

    emit.comment("% Top 1%")
    emit.macro("TopOneCAPctWallets", fmt_dec(ca["n_wallets_pct_of_top1"].mean(), 2))
    emit.macro("TopOneEOAPctWallets", fmt_dec(eoa["n_wallets_pct_of_top1"].mean(), 2))
    emit.macro("TopOneCAPctValue", fmt_dec(ca["value_pct_of_top1"].mean(), 2))
    emit.macro("TopOneEOAPctValue", fmt_dec(eoa["value_pct_of_top1"].mean(), 2))
    emit.blank()

    # Top 0.1%
    ca01 = df01[df01["wallet_type"] == "CA"]
    eoa01 = df01[df01["wallet_type"] == "EOA"]

    emit.comment("% Top 0.1%")
    emit.macro("TopPointOneCAPctWallets", fmt_dec(ca01["n_wallets_pct_of_top1"].mean(), 2))
    emit.macro("TopPointOneCAPctValue", fmt_dec(ca01["value_pct_of_top1"].mean(), 2))
    # \TopPointOnePctWealth = mean of top-0.1%'s value_pct_top1 (share of total ecosystem) ≈ 93.45
    emit.blank()
def emit_graphsense_tags(emit: MacroEmitter, manual: Dict[str, str]) -> None:
    emit.comment("% --- GraphSense tag coverage of top-0.1% addresses ---")

    # Coverage stats (per-block aggregates, not in any CSV — kept manual).
    for key in ["TagCoveragePctMean", "TagCoveragePctMin", "TagCoveragePctMax"]:
        emit.macro(key, manual[key])

    # \TagUnlabeledPct/ — share of all top-0.1% addresses with no tag info.
    eff = pd.read_csv(DOW_DATA / "effective_label_distribution.csv")
    unlabeled = eff.loc[eff["effective_label"] == "UNLABELED", "pct"].iloc[0]
    emit.macro("TagUnlabeledPct", fmt_dec(unlabeled, 2))
    emit.blank()

    # Category-level shares (within categorised addresses only).
    cat = pd.read_csv(DOW_DATA / "category_distribution.csv").set_index("category")
    cat_map = {
        "TagCatUserPct":       "user",
        "TagCatDexPairPct":    "defi_dex_pair",
        "TagCatServicePct":    "service",
        "TagCatExchangePct":   "exchange",
        "TagCatDefiTokenPct":  "defi_token",
    }
    for macro_name, cat_key in cat_map.items():
        emit.macro(macro_name, fmt_dec(cat.at[cat_key, "pct_of_categorized"], 2))
    emit.blank()

    # Concept-level shares (across the GraphSense concept hierarchy).
    con = pd.read_csv(DOW_DATA / "concept_distribution.csv").set_index("concept")
    con_map = {
        "TagConDefiPct":      "defi",
        "TagConCustodyPct":   "defi_custody",
        "TagConExchangePct":  "exchange",
        "TagConDexPct":       "defi_dex",
    }
    for macro_name, con_key in con_map.items():
        emit.macro(macro_name, fmt_dec(con.at[con_key, "pct_of_concept_addr"], 2))
    emit.blank()


def emit_rf_shap_extended(emit: MacroEmitter) -> None:
    emit.comment("% --- Random Forest + SHAP (extended feature set) ---")
    ext  = pd.read_csv(RFSHAP_DATA / "rf_shap_importance.csv", index_col=0)
    base = pd.read_csv(RFSHAP_DATA / "rf_3feat_importance.csv", index_col=0)

    def _imp_range(df, target_prefix: str, feat: str) -> Tuple[float, float]:
        """Min/max importance (in %) for `feat` across targets matching prefix."""
        sub = df[df["target"].str.startswith(target_prefix) & (df["feature"] == feat)]
        v = sub["importance"] * 100
        return float(v.min()), float(v.max())

    def _r2_range(df, target_prefix: str) -> Tuple[float, float]:
        sub = df[df["target"].str.startswith(target_prefix)]
        r = sub["R2_test"].drop_duplicates()
        return float(r.min()), float(r.max())

    # 6-feature extended model — return targets
    r2_lo, r2_hi = _r2_range(ext, "ret_")
    emit.macro("RfExtRetRsqLo", fmt_dec(r2_lo, 2))
    emit.macro("RfExtRetRsqHi", fmt_dec(r2_hi, 2))
    for feat, prefix in [("month", "Month"), ("beta_baseline", "Beta"),
                         ("log_value_usd", "Value")]:
        lo, hi = _imp_range(ext, "ret_", feat)
        emit.macro(f"RfExtRet{prefix}ImpLo", fmt_dec(lo, 0))
        emit.macro(f"RfExtRet{prefix}ImpHi", fmt_dec(hi, 0))

    # 6-feature extended model — alpha targets (only month and beta are reported)
    for feat, prefix in [("month", "Month"), ("beta_baseline", "Beta")]:
        lo, hi = _imp_range(ext, "alpha_", feat)
        emit.macro(f"RfExtAlpha{prefix}ImpLo", fmt_dec(lo, 0))
        emit.macro(f"RfExtAlpha{prefix}ImpHi", fmt_dec(hi, 0))

    # ΔR² (extended − 3-feat) per target, then min/max over each target group.
    ext_r2  = ext.groupby("target")["R2_test"].first()
    base_r2 = base.groupby("target")["R2_test"].first()
    delta = ext_r2 - base_r2
    ret_d   = delta[delta.index.str.startswith("ret_")]
    alpha_d = delta[delta.index.str.startswith("alpha_")]

    def _signed(v: float, dp: int = 2) -> str:
        return f"{round(v, dp):+.{dp}f}"

    emit.macro("RfDeltaRsqRetLo",   _signed(ret_d.min()))
    emit.macro("RfDeltaRsqRetHi",   _signed(ret_d.max()))
    emit.macro("RfDeltaRsqAlphaLo", _signed(alpha_d.min()))
    emit.macro("RfDeltaRsqAlphaHi", _signed(alpha_d.max()))
    emit.blank()

    # Compute SHAP from CSV when available
    shap = pd.read_csv(RFSHAP_DATA / "rf_shap_values_baseline.csv")
    shap_means = {col: shap[col].abs().mean() for col in
                  ["month", "beta_baseline"]}

    emit.macro("ShapMonthPp",  fmt_dec(shap_means["month"], 1))
    emit.macro("ShapBetaPp",   fmt_dec(shap_means["beta_baseline"], 1))
    emit.blank()


def emit_rf_and_robustness(emit: MacroEmitter, manual: Dict[str, str]) -> None:
    """Risk-free rate parameter, on-chain stablecoin yield evidence, and the
    rf=0 vs rf=5% robustness check. All values are pinned constants —
    sourced offline (DefiLlama for stablecoin APYs; a 5,000-wallet × 72-block
    re-run of MSR for the rf-sensitivity comparison) and not derived from any
    CSV in the Analytics/ tree.
    """
    emit.comment("% --- Risk-free rate (used for MSR optimisation) ---")
    emit.macro("RfDefault", manual["RfDefault"])
    emit.blank()

    emit.comment("% --- On-chain stablecoin yield evidence (DefiLlama, Ethereum-only) ---")
    # Derived directly from the CSV outputs of
    # distance_return_analytics/src/fetch_stablecoin_lending_rates.py.
    # `overall_avg_apy`           = mean of monthly avg_apy values
    # `overall_median_monthly_apy` = median of monthly median_apy values
    by_asset = pd.read_csv(DRA_DATA / "stablecoin_overall_avg_apy_by_asset_ethereum.csv")
    by_asset = by_asset.set_index("asset")
    combined = pd.read_csv(DRA_DATA / "stablecoin_overall_avg_apy_combined_ethereum.csv").iloc[0]

    for asset, tok in (("USDC", "USDC"), ("USDT", "USDT")):
        row = by_asset.loc[asset]
        emit.macro(f"Stablecoin{tok}AvgAPY",    fmt_dec(row["overall_avg_apy"], 2))
        emit.macro(f"Stablecoin{tok}MedianAPY", fmt_dec(row["overall_median_monthly_apy"], 2))
        emit.macro(f"Stablecoin{tok}Months",    str(int(row["months_observed"])))

    emit.macro("StablecoinPooledAvgAPY",    fmt_dec(combined["overall_avg_apy"], 2))
    emit.macro("StablecoinPooledMedianAPY", fmt_dec(combined["overall_median_monthly_apy"], 2))
    emit.macro("StablecoinPooledMonths",    str(int(combined["months_observed"])))
    emit.blank()

    emit.comment("% --- Robustness check sample (rf=0 vs rf=5%) ---")
    for k in ["RfRobustBlocks", "RfRobustWalletsPerBlock", "RfRobustTotalWallets"]:
        emit.macro(k, manual[k])
    emit.blank()

    emit.comment("% MSR weight L1 distance, rf=0 vs rf=5%, normalised 0--100")
    for k in ["RfRobustWeightLOneMedianPct", "RfRobustWeightLOneMeanPct",
              "RfRobustWeightLOneTopFivePct"]:
        emit.macro(k, manual[k])
    emit.blank()

    emit.comment("% Differences in L1-style gap metrics (pp on 0--100 scale)")
    for k in ["RfRobustGapDiffMedianPP", "RfRobustGapDiffMeanPP", "RfRobustGapDiffTopFivePP",
              "RfRobustVsEqualDiffMedianPP", "RfRobustVsEqualDiffMeanPP", "RfRobustVsEqualDiffTopFivePP",
              "RfRobustVsMcapDiffMedianPP", "RfRobustVsMcapDiffMeanPP", "RfRobustVsMcapDiffTopFivePP"]:
        emit.macro(k, manual[k])
    emit.blank()


def emit_recon_and_mpt_config(emit: MacroEmitter, manual: Dict[str, str]) -> None:
    emit.comment("% --- Portfolio Reconstruction pipeline: data sizes + cluster config ---")
    keys_recon = [
        "ReconTransferEvents", "ReconTokensDiscovered", "ReconParquetSizeGB",
        "ReconTopTokenTransfers", "ReconTopTokenSharePct", "ReconMedianTokenTransfers",
        "ReconTokensAfterFilter",
        "ReconPipelineWorkers", "ReconBatchSize",
        "ReconOutputRowsFirst", "ReconOutputRowsSecond", "ReconOutputRows",
        "ReconOutputSizeGB",
    ]
    for k in keys_recon:
        emit.macro(k, manual[k])
    emit.macro("ReconSnapshotBlocks", manual["ReconSnapshotBlocks"])
    emit.blank()

    emit.comment("% --- MPT pipeline: observations, blocks, wallets-per-block stats ---")
    df = pd.read_csv(DRA_DATA / "per_block_distance_stats.csv")
    pb = df[df["token_group"] == "all"]
    # Wallets per block come from per_block_distance_by_token_group counts
    # (sum of lt5 + gte5 per block, fixed across optimisations).
    bg = pd.read_csv(DRA_DATA / "per_block_distance_by_token_group.csv")
    nw = bg[bg["optimisation"] == "better_return"].groupby("block_number")["count"].sum()
    emit.macro("MptObservations", manual["MptObservations"])
    emit.macro("MptObservationsM", manual["MptObservationsM"])
    n_blocks = pb["block_number"].nunique()
    emit.macro("MptBlocks", str(n_blocks))
    emit.macro("MptWorkers", manual["MptWorkers"])
    emit.macro("MptCores", manual["MptCores"])
    emit.macro("MptOutputSizeGB", manual["MptOutputSizeGB"])
    emit.macro("MptStrategies", manual["MptStrategies"])
    emit.macro("MptHorizonDays", manual["MptHorizonDays"])

    def commafmt(v: float) -> str:
        return f"{int(round(v)):,}"

    emit.macro("MptMedianWalletsPerBlock", commafmt(nw.median()))
    emit.macro("MptMinWalletsPerBlock", commafmt(nw.min()))
    emit.macro("MptMaxWalletsPerBlock", commafmt(nw.max()))
    emit.blank()
def emit_l1_token_group_autogen(emit: MacroEmitter) -> None:
    """Auto-generated L1 token-group section — the second instance of
    \\LOneSmall*/\\LOneLarge*."""
    emit.comment("% --- Auto-generated: L1 distance by token-group threshold (N=5) ---")

    df = pd.read_csv(DRA_DATA / "per_block_distance_by_token_group.csv")
    n_blocks = df["block_number"].nunique()
    emit.macro("TotalBlocks", str(n_blocks))
    emit.blank()

    STRAT_MAP = [("better_return", "MRV"), ("safer_risk", "MVR"), ("max_sharpe", "MSR")]
    GROUP_MAP = [("lt5", "Small"), ("gte5", "Large")]

    emit.comment("% --- Per-group means (time-series averages) ---")

    for opt_int, opt_suf in STRAT_MAP:
        for grp, grp_label in GROUP_MAP:
            name = f"LOne{grp_label}{opt_suf}Mean"
            sub = df[(df["optimisation"] == opt_int) & (df["token_group"] == grp)]
            avg_mean = sub["mean_dist"].mean()
            emit.macro(name, f"{avg_mean:.1f}")
        emit.blank()

    emit.comment("% --- Hypothesis test results (paired Wilcoxon, N=5 threshold) ---")
    tests = pd.read_csv(DRA_DATA / "hypothesis_tests_l1_by_token_group.csv")
    for opt_int, opt_suf in STRAT_MAP:
        row = tests[tests["optimisation"] == opt_int].iloc[0]
        delta = row["mean_diff_gte5_minus_lt5"]
        cohens_d = row["cohens_d_paired"]
        emit.macro(f"LOneDelta{opt_suf}", f"{delta:.1f}")
        emit.macro(f"LOneCohen{opt_suf}", f"{cohens_d:.2f}")
    emit.blank()

    emit.comment("% --- Quick reference ---")
    emit.comment(f"% Blocks: {n_blocks}")
    for opt_int, opt_suf in STRAT_MAP:
        lt5 = df[(df["optimisation"] == opt_int) & (df["token_group"] == "lt5")]["mean_dist"].mean()
        gte5 = df[(df["optimisation"] == opt_int) & (df["token_group"] == "gte5")]["mean_dist"].mean()
        emit.comment(f"% {opt_suf}: <5 = {lt5:.1f}%,  >=5 = {gte5:.1f}%,  Δ = {gte5 - lt5:.1f} pp")
    emit.blank()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class _ManualLookup(dict):
    """Dict subclass that returns "" for missing keys.

    manual_constants.py only carries the entries we actually emit; this
    keeps the lookup tolerant if a key was pruned without breaking the
    rest of the build.
    """

    def __getitem__(self, key):
        return super().get(key, "")


def build(verbose: bool = False) -> str:
    sys.path.insert(0, str(ANALYTICS))
    from manual_constants import MANUAL_CONSTANTS as _RAW_MC  # local import
    MC = _ManualLookup(_RAW_MC)

    capm = parse_capm_md()
    if verbose:
        print(f"[capm csvs] loaded {len(capm)} numerical entries", file=sys.stderr)

    e = MacroEmitter()
    emit_token_metadata(e, MC)
    emit_total_accounts_raw(e)
    emit_ca_vs_eoa(e)
    emit_holders(e)
    emit_wealth_buckets(e)
    emit_wealth_share_stats(e)
    emit_l1_distance_main(e, MC)
    emit_wealth_composition(e, MC)
    emit_top_wallet_concentration(e, MC)
    emit_portfolio_size_distribution(e)
    emit_naive_benchmark(e, capm, MC)
    emit_capm(e, capm, MC)
    emit_pct_beat_block_aggregates(e)
    emit_beat_market_quantiles(e)
    emit_return_analysis(e, MC)
    emit_l1_fitting(e)
    emit_top1_breakdown(e)
    emit_graphsense_tags(e, MC)
    emit_rf_shap_extended(e)
    emit_rf_and_robustness(e, MC)
    emit_recon_and_mpt_config(e, MC)
    emit_l1_token_group_autogen(e)

    return e.text()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
# Allow digits in macro names (e.g. \LOneMs60to80Pct/, \WealthShareCAP10P/).
# Use greedy (.*) for the value so nested braces (\num{X}) are captured.
DEF_RE = re.compile(r"\\def\\([A-Za-z][A-Za-z0-9]*)/?\{(.*)\}\s*(?:%.*)?$")


def parse_macros(text: str) -> "OrderedDict[str, str]":
    """Last-wins dict of macro_name -> raw value text, plus a list of duplicates."""
    out: "OrderedDict[str, str]" = OrderedDict()
    for line in text.splitlines():
        m = DEF_RE.match(line.strip())
        if not m:
            continue
        name, raw = m.group(1), m.group(2)
        out[name] = raw
    return out


def normalise_value(raw: str) -> Optional[str]:
    """Strip wrappers and return a normalised string for byte-equal comparison."""
    s = raw.strip()
    # Strip \num{...}
    while True:
        m = re.match(r"^\\num\{(.*)\}$", s)
        if not m:
            break
        s = m.group(1).strip()
    # Strip \SI{X}{\percent} → "X" (mark percent)
    m = re.match(r"^\\SI\{([^}]+)\}\{\\percent\}$", s)
    is_pct = False
    if m:
        s = m.group(1).strip()
        is_pct = True
    # Strip trailing \% / %
    if s.endswith(r"\%"):
        s = s[:-2].strip()
        is_pct = True
    elif s.endswith("%"):
        s = s[:-1].strip()
        is_pct = True
    return s + ("%" if is_pct else "")


def try_float(s: str) -> Optional[float]:
    s2 = s.rstrip("%").replace(",", "").replace("+", "").strip()
    if s2 == "":
        return None
    try:
        return float(s2)
    except ValueError:
        return None


def compare_values(expected: str, generated: str) -> Tuple[str, str, str]:
    """Return (status, exp_norm, gen_norm).
    Status: EXACT, NUMERIC_MATCH, ROUND_MATCH, MISMATCH."""
    en = normalise_value(expected)
    gn = normalise_value(generated)
    if en is None or gn is None:
        return ("MISMATCH", str(en), str(gn))
    if en == gn:
        return ("EXACT", en, gn)
    # Numeric tolerance
    ef = try_float(en)
    gf = try_float(gn)
    if ef is not None and gf is not None:
        if math.isclose(ef, gf, rel_tol=1e-3, abs_tol=1e-9):
            return ("NUMERIC_MATCH", en, gn)
        # ROUND_MATCH: same rounded display at some dp ∈ {0,1,2,3},
        # AND both the absolute and relative diffs are small. The
        # relative cap prevents pathological cases like 2.33 vs 1.89
        # (both round to 2 at dp=0 but the drift is 19%) from being
        # classified as "rounding noise".
        rel = abs(ef - gf) / max(abs(ef), 1e-9)
        for dp in (0, 1, 2, 3):
            if round(ef, dp) == round(gf, dp):
                if abs(ef - gf) <= max(0.5, abs(ef) * 0.02) and rel <= 0.05:
                    return ("ROUND_MATCH", en, gn)
        return ("MISMATCH", en, gn)
    # Strings: case-insensitive whitespace-collapsed
    en2 = re.sub(r"\s+", " ", en.strip())
    gn2 = re.sub(r"\s+", " ", gn.strip())
    if en2 == gn2:
        return ("EXACT", en, gn)
    return ("MISMATCH", en, gn)


def verify(generated_text: str, target_tex: pathlib.Path) -> Tuple[List[str], Dict[str, int]]:
    target_text = target_tex.read_text()
    expected = parse_macros(target_text)
    actual = parse_macros(generated_text)

    counts = {
        "EXACT": 0, "NUMERIC_MATCH": 0, "ROUND_MATCH": 0,
        "MISMATCH": 0, "MISSING_FROM_GENERATED": 0,
        "EXTRA_IN_GENERATED": 0,
    }
    lines: List[str] = []
    seen_names = set()

    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GREY = "\033[90m"
    RESET = "\033[0m"

    for name, exp_raw in expected.items():
        seen_names.add(name)
        if name not in actual:
            counts["MISSING_FROM_GENERATED"] += 1
            lines.append(f"{YELLOW}[MISSING]{RESET}     \\{name}/  expected={exp_raw}")
            continue
        gen_raw = actual[name]
        status, en, gn = compare_values(exp_raw, gen_raw)
        counts[status] += 1
        if status == "EXACT":
            lines.append(f"{GREEN}[EXACT  ]{RESET}     \\{name}/  = {en}")
        elif status == "NUMERIC_MATCH":
            lines.append(f"{YELLOW}[NUMERIC]{RESET}     \\{name}/  expected={en}  generated={gn}")
        elif status == "ROUND_MATCH":
            lines.append(f"{YELLOW}[ROUND  ]{RESET}     \\{name}/  expected={en}  generated={gn}")
        else:
            lines.append(f"{RED}[MISMATCH]{RESET}    \\{name}/  expected={en}  generated={gn}")

    for name in actual:
        if name not in seen_names:
            counts["EXTRA_IN_GENERATED"] += 1
            lines.append(f"{GREY}[EXTRA  ]{RESET}     \\{name}/  generated={actual[name]}")

    total = sum(counts.values())
    summary = []
    summary.append("")
    summary.append("=" * 78)
    summary.append("VERIFICATION SUMMARY")
    summary.append("=" * 78)
    summary.append(f"  Total macros (expected):      {len(expected)}")
    summary.append(f"  Total macros (generated):     {len(actual)}")
    summary.append(f"  EXACT:                        {counts['EXACT']}")
    summary.append(f"  NUMERIC_MATCH (rel-tol 1e-3): {counts['NUMERIC_MATCH']}")
    summary.append(f"  ROUND_MATCH:                  {counts['ROUND_MATCH']}")
    summary.append(f"  MISMATCH:                     {counts['MISMATCH']}")
    summary.append(f"  MISSING_FROM_GENERATED:       {counts['MISSING_FROM_GENERATED']}")
    summary.append(f"  EXTRA_IN_GENERATED:           {counts['EXTRA_IN_GENERATED']}")

    fail_n = counts["MISMATCH"] + counts["MISSING_FROM_GENERATED"]
    if fail_n == 0:
        summary.append(f"  RESULT: PASS (every macro present and within tolerance)")
    else:
        summary.append(f"  RESULT: FAIL ({fail_n} issue(s) need fixing)")
    summary.append("=" * 78)

    return lines + summary, counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true", help="run verification against target macro.tex")
    p.add_argument(
        "--target-tex",
        default=None,
        help=(
            "Path to reference macro.tex for --verify. Overrides the "
            "MACRO_TARGET_TEX env var. Defaults to ../BlockchainPortfolio/macro.tex."
        ),
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    txt = build(verbose=args.verbose)
    OUT_TEX.write_text(txt)
    print(f"[generate_all_macros] wrote {OUT_TEX} ({len(txt.splitlines())} lines)")

    if args.verify:
        target_tex = resolve_target_tex(args.target_tex)
        if not target_tex.exists():
            print(
                f"[generate_all_macros] --verify skipped: target file not found at {target_tex}\n"
                "  Set MACRO_TARGET_TEX or pass --target-tex to point at your reference macro.tex.",
                file=sys.stderr,
            )
            return 0
        report, counts = verify(txt, target_tex)
        report_str = "\n".join(report) + "\n"
        # Strip ANSI for the file copy
        ansi_re = re.compile(r"\x1b\[[0-9;]*m")
        REPORT_TXT.write_text(ansi_re.sub("", report_str))
        # Print to stdout with ANSI
        print(report_str)
        print(f"[generate_all_macros] verification report → {REPORT_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
