#!/usr/bin/env python3
"""
distance_return_macros.py

Description:
    Single owner of every LaTeX macro derived from the distance and return
    analytics (paper sections 5 and 6, appendices C to E). Every value is
    computed from the files in ../data; there are no manual constants.
    Called by Analytics/generate_all_macros.py; run standalone to print the
    macros, and with --tables to print the rows of the threshold-sweep table
    (paper App. C) and of the two Random Forest tables (App. D), which are
    pasted into the LaTeX source.

    Aggregation bases, as in the paper: per-block (snapshot) statistics are
    averaged or taken as medians across the 72 snapshots; "pooled" values
    use all account-snapshot rows. Shares are in percent, L1 distances in
    percent of the maximum distance (l1 / 2 * 100) unless noted.

Input (../data keeps each cluster output folder as it comes):
    distance_analysis_results/, distance_analysis_results_gr_5/
                                    mpt_distance_analysis.py (also analysis_log.txt)
    distance_per_block_results/     distance_per_block_analysis.py
    l1_formula_fits.csv             fit_l1_formulas.py
    return_analysis_results/        mpt_return_analysis.py, analyze_pct_beat_market.py
                                    (pct_beat_baseline_per_block.csv: hit rates)
    camp_analysis_results_full/     analyze_extended_capm.py
    analyze_extended_capm_output.txt  console log of analyze_extended_capm.py
                                    (the pooled tables are only printed there)
    rf_shap_results/                rf_shap_analysis_extended.py
    stablecoin_*_ethereum.csv       fetch_stablecoin_lending_rates.py
    Optional, skipped with a warning until they exist:
    return_analysis_results/pct_beat_eth_per_block.csv,
    return_analysis_results/excess_vs_eth_quantiles_per_block.csv
                                    analyze_pct_beat_market.py --prices-dir
                                    (also the source of the forward horizon)
    rf_test_eth/comparison/overall_summary.csv
                                    rf_zero_test.py

Macros (names kept from the former sections of generate_all_macros.py):
    L1 distance      \\LOneObservations{Raw,N,M}/, \\LOneMean*/, \\LOneMedian*/,
                     \\LOneMonthsN/, \\LOneNTwoObsM/, \\LOne*CombinedLeOnePct/,
                     \\LOneMs*Pct/, \\LOneCorr*/, \\LOneJoint*/, \\LOneNearOpt*/,
                     \\LOneOls*Rsq/, \\LOneOlsExt*Rsq/, \\LOneRf*/ (per-strategy
                     importance shares), \\LOneSpearman*/, \\LOnePearson*/,
                     \\LOneFivePlus*/
    Naive benchmarks \\LOneNaive*/, \\LOneDelta{Br,Sr,Ms}{Equal,Mcap}*/ (mean change
                     in pp of the 0-100 distance scale),
                     \\LOneDeltaSrEqualRatio/ (closer-to-farther, MinVar vs equal)
    CAPM             \\CAPMBeta*/, \\CAPMAlpha*Median/ and \\CAPMRet*Median/ (median
                     across snapshots of per-snapshot medians), \\CAPMFracPositive*/
                     (median across snapshots of per-snapshot shares), \\CAPMRetHit*/
                     (wins among accounts whose strategy portfolio lies more than
                     1 % from their own), \\CAPMRetTie*/ (share within 1 %),
                     \\CAPMSpearman*/, \\CAPMDeltaBeta*Median/ (shift of the median beta), \\CAPMObsRaw/,
                     \\CAPMBlocksN/, \\CAPMHorizonDays/
    Beat the market  \\PctBeat*/, \\PctPosAlpha*/ (medians across snapshots; ranges and
                     spreads leave out mcap-weight, whose per-snapshot shares are
                     bimodal: \\Pct*McapBlocks{BelowTen,AboveNinety}/), \\PctBlocksBelowFifty/,
                     \\PctBeatRetBaselineEth{Up,Down}Median/, \\BeatExcRet*/, \\BeatAlpha*/
    Return paths     \\CumExcess*Pp/ (cumulative median excess over the market),
                     \\RetStratSpreadMedianPp/, \\RetMarketSwingPp/, \\RetHoldsEthPct/,
                     \\Ret{Mcap,Baseline}EthCorr/
    ETH benchmark    \\EthBeatMarketBlocksPct/, \\PctBeatEth*Block{Median,Mean}/,
                     \\BeatExcEth*{BotOne,Med,TopOne}/
    Return analysis  \\RetDist{Zero,High}MedianBaseline/, \\RetNegativePct{Lo,Hi}/
    Power-decay fit  \\Fit{Psi,Gamma,Dinf,Rsq,Mae}{MRV,MVR,MSR}/
    RF + SHAP        \\RfExt*/, \\RfDeltaRsq*/ (ranges over every strategy but
                     mcap-weight, which RfExtMcap*/ report on their own),
                     \\RfExt{Ret,Alpha}{Month,Beta,Value}Perm*/ (permutation shares),
                     \\RfExtOtherPermHi/, \\RfSample*/, \\RfRandom*Rsq/, \\Shap*/
    Month-grouped RF \\VarShare*/, \\RfGrouped*Rsq/, \\RfState*Rsq/,
                     \\RfWithin*Rsq/, \\RfForward*Rsq{Lo,Hi}/
    Risk-free rate   \\Stablecoin*/, \\RfDefault/ (the non-zero rate of the re-run;
                     production MSR uses r_f = 0), \\RfRobust*/
    MPT run          \\MptObservations{,M}/, \\MptBlocks/, \\MptStrategies/,
                     \\MptHorizonDays/, \\Mpt{Median,Min,Max}WalletsPerBlock/
    Token groups     \\TotalBlocks/, \\LOne{Small,Large}{MRV,MVR,MSR}Mean/,
                     \\LOneDelta{MRV,MVR,MSR}/, \\LOneCohen{MRV,MVR,MSR}/

Usage:
    python distance_return_macros.py
    python distance_return_macros.py --data-dir ../data
    python distance_return_macros.py --tables        # appendix table rows
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

STRATEGIES = ["baseline", "better_return", "safer_risk", "max_sharpe", "equal_weight", "mcap_weight"]
STRAT_TOK = {"baseline": "Baseline", "better_return": "Better", "safer_risk": "Safer",
             "max_sharpe": "MaxSharpe", "equal_weight": "Equal", "mcap_weight": "Mcap"}
BR_SR_MS = {"better_return": "Br", "safer_risk": "Sr", "max_sharpe": "Ms"}
MRV_MVR_MSR = {"better_return": "MRV", "safer_risk": "MVR", "max_sharpe": "MSR"}

# Strategies left out of the RF importance / R2 ranges: the mcap-weight portfolio
# is close to ETH for nearly every account, so its return is almost the same for
# everyone in a month and the month alone predicts it (R2 ~ 1). Its values are
# emitted separately as \RfExtMcap*/.
RF_RANGE_EXCLUDE = {"mcap_weight"}

# Portfolio-size threshold of the small / large split in the paper (SPLIT in
# distance_per_block_analysis.py); its row of the threshold sweep uses macros.
SPLIT_N = 5

# Alphabetic quantile suffixes: digits are catcode 12 in TeX, so \FooQ01/ would
# define \FooQ with delimiter "01/" and sibling definitions would collide.
QUANTILE_TOK = {0.01: "BotOne", 0.50: "Med", 0.99: "TopOne"}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def fmt_dec(v, dp: int) -> str:
    # + 0.0 turns a rounded -0.0 into 0.0
    return f"{round(float(v), dp) + 0.0:.{dp}f}"


def fmt_num_int(v) -> str:
    return rf"\num{{{int(round(float(v)))}}}"


def fmt_signed(v, dp: int = 3) -> str:
    """Explicit sign (+0.115, -0.046); a value that rounds to zero prints unsigned."""
    x = round(float(v), dp) + 0.0
    return f"{x:.{dp}f}" if x == 0 else f"{x:+.{dp}f}"


def fmt_math_minus(x: float) -> str:
    """`$-$X.XX` for negative values (text mode), `X.XX` otherwise."""
    return f"$-${abs(x):.2f}" if x < 0 else f"{x:.2f}"


def fmt_text_signed(v, dp: int) -> str:
    """Text-mode number with a real minus sign: `$-$0.2`, `0.0`, `3.7`."""
    x = round(float(v), dp) + 0.0
    return f"$-${abs(x):.{dp}f}" if x < 0 else f"{x:.{dp}f}"


def fmt_text_pm(v, dp: int) -> str:
    """Text-mode number with an explicit sign: `$-$35.5`, `$+$39.3`."""
    x = round(float(v), dp) + 0.0
    return f"$-${abs(x):.{dp}f}" if x < 0 else f"$+${x:.{dp}f}"


def fmt_thousands_tex(v) -> str:
    """5000 -> 5{,}000 (TeX-safe thousands separator)."""
    return f"{int(round(float(v))):,}".replace(",", "{,}")


def read(path: Path, **kw) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[distance_return_macros] missing input {path}")
    return pd.read_csv(path, **kw)


def warn(msg: str) -> None:
    print(f"[distance_return_macros] {msg}", file=sys.stderr)


# -----------------------------------------------------------------------------
# console log of analyze_extended_capm.py (pooled tables are printed only)
# -----------------------------------------------------------------------------
def parse_capm_log(path: Path) -> Dict[str, float]:
    """Every pooled value the macros need from the analyze_extended_capm.py log."""
    if not path.exists():
        raise SystemExit(f"[distance_return_macros] missing input {path}")
    lines = path.read_text().splitlines()
    out: Dict[str, float] = {}
    num = r"([-+0-9.eE]+)"
    strat = r"(baseline|better_return|safer_risk|max_sharpe|equal_weight|mcap_weight)"

    for line in lines:
        # L1 distances overall: index  l1_metric  mean (%)  median (%)  std (%)
        m = re.match(rf"^\s*\d+\s+(l1_\S+)\s+{num}\s+{num}\s+{num}\s*$", line)
        if m:
            out[f"{m.group(1)}_mean"], out[f"{m.group(1)}_median"], out[f"{m.group(1)}_std"] = (
                float(m.group(2)), float(m.group(3)), float(m.group(4)))
        # Beta summary: mean median std p5 p25 p75 p95 frac_above_1 frac_negative
        m = re.match(rf"^\s*\d+\s+{strat}\s+" + r"\s+".join([num] * 9) + r"\s*$", line)
        if m:
            s = m.group(1)
            v = [float(m.group(i)) for i in range(2, 11)]
            out.update({f"beta_{s}_mean": v[0], f"beta_{s}_median": v[1], f"beta_{s}_std": v[2],
                        f"beta_{s}_p25": v[4], f"beta_{s}_p75": v[5],
                        f"beta_{s}_frac_above_1": v[7] * 100, f"beta_{s}_frac_neg": v[8] * 100})
        # Spearman correlations of the sample
        m = re.search(r"\s*([a-zA-Z0-9_]+)\s+vs\s+([a-zA-Z0-9_]+)\s+ρ\s*=\s*([-+0-9.]+)", line)
        if m:
            out[f"spearman_{m.group(1)}_vs_{m.group(2)}"] = float(m.group(3))

    def section(title: str, row_re: str, stop) -> None:
        inside = False
        for line in lines:
            if title in line:
                inside = True
                continue
            if inside:
                m = re.match(row_re, line)
                if m:
                    yield m
                elif stop(line):
                    return

    # Beta change (strategy - baseline): mean median std pct_increased
    for m in section("Beta Change (strategy − baseline)",
                     rf"^\s*\d+\s+{strat}\s+{num}\s+{num}\s+{num}\s+{num}\s*$",
                     lambda l: l.startswith("==========") or "Cross-Strategy" in l):
        out[f"dbeta_{m.group(1)}_mean"] = float(m.group(2))
        out[f"dbeta_{m.group(1)}_pct_inc"] = float(m.group(5))
    # MPT vs naive direction: mean median closer farther unchanged
    for m in section("MPT vs Naive Direction",
                     rf"^\s*\d+\s+(better_return|safer_risk|max_sharpe)\s+(equal_weight|mcap_weight)\s+"
                     + r"\s+".join([num] * 5) + r"\s*$",
                     lambda l: l.startswith("==========")):
        key = f"naive_{m.group(1)}_{m.group(2)}"
        out[f"{key}_mean"] = float(m.group(3))
        out[f"{key}_closer"] = float(m.group(5))
        out[f"{key}_farther"] = float(m.group(6))
        out[f"{key}_unchanged"] = float(m.group(7))
    return out


# -----------------------------------------------------------------------------
# sections
# -----------------------------------------------------------------------------
def emit_l1_distance_main(emit, d: Path) -> None:
    dar, gr5 = d / "distance_analysis_results", d / "distance_analysis_results_gr_5"
    pb = read(d / "distance_per_block_results" / "per_block_distance_stats.csv")
    tg = read(d / "distance_per_block_results" / "per_block_distance_by_token_group.csv")

    emit.comment("% --- L1 distance: observed vs MPT-optimal portfolios ---")
    n_obs = int(tg[tg["optimisation"] == "better_return"]["count"].sum())
    emit.macro("LOneObservationsRaw", str(n_obs))
    emit.macro_raw(r"\def\LOneObservationsN/{\num{\LOneObservationsRaw/}}")
    emit.macro_raw(r"\def\LOneObservationsM/{\the\numexpr\LOneObservationsRaw//1000000\relax\,M}")
    emit.blank()

    # Time-averaged per-block means / medians (token_group = all)
    pb_all = pb[pb["token_group"] == "all"]

    def opt_stats(opt: str) -> Tuple[float, float]:
        sub = pb_all[pb_all["optimisation"] == opt]
        return sub["mean_dist"].mean(), sub["median_dist"].mean()

    emit.comment("% Time-averaged means and medians per strategy (monthly snapshots)")
    emit.macro("LOneMeanSaferRisk", fmt_dec(opt_stats("safer_risk")[0], 2))
    emit.macro("LOneMeanBetterReturn", fmt_dec(opt_stats("better_return")[0], 2))
    emit.macro("LOneMeanMaxSharpe", fmt_dec(opt_stats("max_sharpe")[0], 2))
    emit.macro("LOneMedianSaferRisk", fmt_dec(opt_stats("safer_risk")[1], 2))
    emit.macro("LOneMedianBetterReturn", fmt_dec(opt_stats("better_return")[1], 2))
    emit.macro("LOneMedianMaxSharpe", fmt_dec(opt_stats("max_sharpe")[1], 2))
    emit.macro("LOneMonthsN", str(pb["block_number"].nunique()))
    emit.blank()

    emit.comment("% Two-asset account-snapshots (pooled), millions")
    agg = read(d / "distance_per_block_results" / "tokens_vs_dist_agg.csv")
    n_two = agg.loc[(agg["strategy"] == "better_return") & (agg["num_tokens"] == 2), "n"].iloc[0]
    emit.macro("LOneNTwoObsM", fmt_dec(n_two / 1e6, 0))
    emit.blank()

    emit.comment("% Distribution buckets per strategy (share with L1 <= 1)")
    zd = read(dar / "zero_distance_counts.csv").set_index("optimisation")
    emit.macro("LOneSrCombinedLeOnePct", fmt_dec(zd.at["safer_risk",    "combined_<=1%_pct"], 2))
    emit.macro("LOneBrCombinedLeOnePct", fmt_dec(zd.at["better_return", "combined_<=1%_pct"], 2))
    emit.macro("LOneMsCombinedLeOnePct", fmt_dec(zd.at["max_sharpe",    "combined_<=1%_pct"], 2))
    dd = read(dar / "distance_distribution_pct.csv").set_index("bucket")
    emit.macro("LOneMsHighBinPct", fmt_dec(dd.at["(80,100]", "max_sharpe"], 2))
    emit.macro("LOneMs60to80Pct",  fmt_dec(dd.at["(60,80]",  "max_sharpe"], 2))
    emit.blank()

    emit.comment("% Cross-strategy Pearson correlations (full data)")
    ct = read(dar / "cross_target_correlation.csv", index_col=0)
    emit.macro("LOneCorrBrSr", fmt_dec(ct.at["corr_br_sr", "pearson_r"], 3))
    emit.macro("LOneCorrBrMs", fmt_dec(ct.at["corr_br_ms", "pearson_r"], 3))
    emit.macro("LOneCorrSrMs", fmt_dec(ct.at["corr_sr_ms", "pearson_r"], 3))
    emit.blank()

    emit.comment("% Joint optimality across strategies")
    mw = read(dar / "multi_optimal_wallets.csv").set_index("metric")
    emit.macro("LOneJointAllThreeN",   fmt_num_int(mw.at["opt_all_3",     "count"]))
    emit.macro("LOneJointAllThreePct", fmt_dec(mw.at["opt_all_3",     "pct"], 3))
    emit.macro("LOneJointBrSrPct",     fmt_dec(mw.at["opt_br_and_sr", "pct"], 2))
    emit.macro("LOneJointBrMsPct",     fmt_dec(mw.at["opt_br_and_ms", "pct"], 2))
    emit.macro("LOneJointSrMsPct",     fmt_dec(mw.at["opt_sr_and_ms", "pct"], 2))
    emit.blank()

    emit.comment("% Near-optimal (L1 <= 1%)")
    two_token_pcts, median_values = [], []
    for opt in BR_SR_MS:
        td = read(dar / f"low_dist_tokens_dist_{opt}.csv")
        two_token_pcts.append(float(td.loc[td["num_tokens"] == 2, "pct"].iloc[0]))
        st = read(dar / f"low_dist_stats_{opt}.csv", index_col=0)
        median_values.append(float(st.at["median_value", "value"]))
    emit.macro("LOneNearOptTwoTokenPctLo", str(round(min(two_token_pcts))))
    emit.macro("LOneNearOptTwoTokenPctHi", str(round(max(two_token_pcts))))
    emit.macro("LOneNearOptMedianValueLo", str(math.ceil(min(median_values))))
    emit.macro("LOneNearOptMedianValueHi", str(math.ceil(max(median_values))))
    for opt, suf in BR_SR_MS.items():
        emit.macro(f"LOneNearOpt{suf}N",   fmt_dec(zd.at[opt, "combined_<=1%_count"] / 1_000_000, 1))
        emit.macro(f"LOneNearOpt{suf}Pct", fmt_dec(zd.at[opt, "combined_<=1%_pct"], 1))
    emit.blank()

    emit.comment("% OLS R-squared (base from ols_summary.csv, extended from analysis_log.txt)")
    ols_r2 = read(dar / "ols_summary.csv").set_index("target")["R2"]
    for opt, suf in BR_SR_MS.items():
        emit.macro(f"LOneOls{suf}Rsq", fmt_dec(ols_r2[opt], 3))
    log = (dar / "analysis_log.txt")
    if not log.exists():
        raise SystemExit(f"[distance_return_macros] missing input {log}")
    ext = dict(re.findall(r"Extended OLS: (\w+)\s+R² = ([0-9.]+)", log.read_text()))
    for opt, suf in BR_SR_MS.items():
        emit.macro(f"LOneOlsExt{suf}Rsq", fmt_dec(ext[opt], 3))
    emit.blank()

    emit.comment("% Random Forest fit (R^2 + MAE) and impurity importances")
    rf = read(dar / "random_forest_importance.csv")
    rf_by_target = rf.groupby("target").first()
    for target, suf in BR_SR_MS.items():
        emit.macro(f"LOneRf{suf}Rsq", fmt_dec(rf_by_target.at[target, "R2_test"], 3))
        emit.macro(f"LOneRf{suf}Mae", fmt_dec(rf_by_target.at[target, "MAE_test"], 2))
    def imp(target: str, feat: str) -> str:
        v = rf[(rf["target"] == target) & (rf["feature"] == feat)]["importance"].iloc[0]
        return fmt_dec(v * 100, 1)

    emit.comment("% Importance shares, %: {Sr,Br}Tokens = num_tokens; Ms{Value,Month,Tokens} = MaxSR")
    emit.macro("LOneRfSrTokensImp", imp("safer_risk", "num_tokens"))
    emit.macro("LOneRfBrTokensImp", imp("better_return", "num_tokens"))
    emit.macro("LOneRfMsValueImp",  imp("max_sharpe", "log_value_usd"))
    emit.macro("LOneRfMsMonthImp",  imp("max_sharpe", "month"))
    emit.macro("LOneRfMsTokensImp", imp("max_sharpe", "num_tokens"))
    emit.blank()

    emit.comment("% Spearman/Pearson correlations on the 50M-row sample")
    cs = read(dar / "correlation_sample.csv")

    def corr(target: str, col: str) -> float:
        return float(cs[(cs["feature"] == "num_tokens") & (cs["target"] == target)].iloc[0][col])

    emit.macro("LOneSpearmanSrTokens", fmt_dec(corr("safer_risk",    "spearman_r"), 3))
    emit.macro("LOnePearsonSrTokens",  fmt_dec(corr("safer_risk",    "pearson_r"),  3))
    emit.macro("LOneSpearmanBrTokens", fmt_dec(corr("better_return", "spearman_r"), 3))
    emit.macro("LOnePearsonBrTokens",  fmt_dec(corr("better_return", "pearson_r"),  3))
    emit.macro("LOneSpearmanMsTokens", fmt_dec(corr("max_sharpe",    "spearman_r"), 3))
    emit.blank()

    emit.comment("% --- 5+ token robustness check ---")
    zd5 = read(gr5 / "zero_distance_counts.csv").set_index("optimisation")
    emit.macro("LOneFivePlusBrLeOnePct", fmt_dec(zd5.at["better_return", "combined_<=1%_pct"], 2))
    emit.macro("LOneFivePlusSrLeOnePct", fmt_dec(zd5.at["safer_risk",    "combined_<=1%_pct"], 2))
    emit.macro("LOneFivePlusMsLeOnePct", fmt_dec(zd5.at["max_sharpe",    "combined_<=1%_pct"], 2))
    dd5 = read(gr5 / "distance_distribution_pct.csv")
    emit.macro("LOneFivePlusMsHighBinPct", fmt_dec(dd5.loc[dd5["bucket"] == "(80,100]", "max_sharpe"].iloc[0], 1))
    rf5 = read(gr5 / "random_forest_importance.csv")
    emit.macro("LOneFivePlusRfSrRsq", fmt_dec(rf5[rf5["target"] == "safer_risk"]["R2_test"].iloc[0], 3))
    emit.macro("LOneFivePlusMsMonthImp", fmt_dec(
        rf5[(rf5["target"] == "max_sharpe") & (rf5["feature"] == "month")]["importance"].iloc[0] * 100, 1))
    cs5 = read(gr5 / "correlation_sample.csv")
    emit.macro("LOneFivePlusSpearmanSrTokens", fmt_dec(
        cs5[(cs5["feature"] == "num_tokens") & (cs5["target"] == "safer_risk")]["spearman_r"].iloc[0], 3))
    ct5 = read(gr5 / "cross_target_correlation.csv", index_col=0)
    emit.macro("LOneFivePlusCorrBrMs", fmt_dec(ct5.at["corr_br_ms", "pearson_r"], 3))
    emit.blank()


def emit_naive_benchmark(emit, capm: Dict[str, float]) -> None:
    emit.comment("% --- L1 distance: baseline portfolio vs naive (equal/mcap) benchmarks ---")
    emit.comment("% L1 distance: baseline vs naive (overall)")
    emit.macro("LOneNaiveEqualMean", fmt_dec(capm["l1_baseline_vs_equal_mean"], 1))
    emit.macro("LOneNaiveEqualMedian", fmt_dec(capm["l1_baseline_vs_equal_median"], 1))
    emit.macro("LOneNaiveMcapMean", fmt_dec(capm["l1_baseline_vs_mcap_mean"], 1))
    emit.macro("LOneNaiveMcapMedian", fmt_dec(capm["l1_baseline_vs_mcap_median"], 1))
    emit.blank()

    # The log's delta is on the raw L1 scale (0-2); * 50 gives pp of the 0-100 distance
    # scale used in the paper. Closer / farther / unchanged use |raw delta| <= 0.001
    # (= 0.05 pp) as "unchanged".
    for naive, label in [("equal_weight", "Equal"), ("mcap_weight", "Mcap")]:
        emit.comment(f"% MPT vs Naive direction: {'Equal-weight' if naive == 'equal_weight' else 'MCap-weight'}"
                     " (mean delta in pp)")
        for mpt, suf in BR_SR_MS.items():
            key = f"naive_{mpt}_{naive}"
            emit.macro(f"LOneDelta{suf}{label}Mean", fmt_signed(capm[f"{key}_mean"] * 50, 2))
            emit.macro(f"LOneDelta{suf}{label}CloserPct", fmt_dec(capm[f"{key}_closer"], 1))
            emit.macro(f"LOneDelta{suf}{label}FartherPct", fmt_dec(capm[f"{key}_farther"], 1))
            emit.macro(f"LOneDelta{suf}{label}UnchangedPct", fmt_dec(capm[f"{key}_unchanged"], 1))
        emit.blank()

    emit.comment("% MinVar closer-to-farther ratio vs equal-weight")
    emit.macro("LOneDeltaSrEqualRatio", fmt_dec(
        capm["naive_safer_risk_equal_weight_closer"] / capm["naive_safer_risk_equal_weight_farther"], 1))
    emit.blank()


def forward_horizon(d: Path) -> Optional[int]:
    """Forward-return horizon in days, as recorded by analyze_pct_beat_market.py
    --prices-dir, which also checks it against the stored market returns."""
    path = d / "return_analysis_results" / "pct_beat_eth_per_block.csv"
    if not path.exists():
        return None
    eth = pd.read_csv(path)
    gap = (eth["market_return_check"] - eth["market_return"]).abs()
    if gap.isna().all() or gap.max() > 1e-9:
        warn(f"horizon not emitted: the recomputed market return does not match ({path})")
        return None
    return int(eth["horizon_days"].iloc[0])


def emit_capm(emit, d: Path, capm: Dict[str, float]) -> None:
    cap, ret_dir = d / "camp_analysis_results_full", d / "return_analysis_results"
    emit.comment("% --- CAPM: alpha/beta of strategies vs market benchmark ---")
    emit.comment("% Beta summary (baseline median + IQR, pooled)")
    emit.macro("CAPMBetaBaselineMedian", fmt_dec(capm["beta_baseline_median"], 3))
    emit.macro("CAPMBetaBaselinePtwentyfive", fmt_dec(capm["beta_baseline_p25"], 2))
    emit.macro("CAPMBetaBaselinePseventyfive", fmt_dec(capm["beta_baseline_p75"], 2))
    emit.blank()

    # Per-snapshot basis: median across snapshots of the within-snapshot statistic.
    rets_pb = read(cap / "returns_per_block.csv")
    alpha_pb = read(cap / "alpha_per_block.csv")

    def pb_med(df: pd.DataFrame, col: str) -> float:
        return float(df[col].median()) * 100.0

    emit.comment("% Alpha medians per strategy (median across snapshots of per-snapshot medians)")
    for s in ["baseline", "mcap_weight", "equal_weight", "better_return", "safer_risk", "max_sharpe"]:
        emit.macro(f"CAPMAlpha{STRAT_TOK[s]}Median", fmt_text_signed(pb_med(alpha_pb, f"median_alpha_{s}"), 2))
    emit.blank()

    emit.comment("% Share with positive alpha (median across snapshots of per-snapshot shares)")
    for s in ["baseline", "mcap_weight", "equal_weight", "better_return", "safer_risk", "max_sharpe"]:
        emit.macro(f"CAPMFracPositive{STRAT_TOK[s]}", fmt_dec(pb_med(alpha_pb, f"frac_positive_alpha_{s}"), 1))
    emit.blank()

    emit.comment("% Median realized returns per strategy + market benchmark (same basis)")
    for s in ["baseline", "better_return", "safer_risk", "max_sharpe", "equal_weight", "mcap_weight"]:
        emit.macro(f"CAPMRet{STRAT_TOK[s]}Median", fmt_text_signed(pb_med(rets_pb, f"median_ret_{s}"), 2))
    emit.macro("CAPMRetMarketMedian", fmt_text_signed(pb_med(rets_pb, "median_market_return"), 2))
    emit.blank()

    emit.comment("% Hit rate vs baseline: share of accounts whose strategy return beats their")
    emit.comment("% own, among accounts whose strategy portfolio lies more than 1 % weight")
    emit.comment("% distance from their own; ties = within 1 %. Medians across snapshots")
    hb = read(ret_dir / "pct_beat_baseline_per_block.csv")
    per = hb.groupby("strategy")[["hit_rate", "pct_same"]].median()
    for s in ["better_return", "safer_risk", "max_sharpe", "equal_weight", "mcap_weight"]:
        emit.macro(f"CAPMRetHit{STRAT_TOK[s]}", fmt_dec(per.at[s, "hit_rate"], 1))
    for s in ["better_return", "safer_risk", "max_sharpe", "equal_weight", "mcap_weight"]:
        emit.macro(f"CAPMRetTie{STRAT_TOK[s]}", fmt_dec(per.at[s, "pct_same"], 1))
    emit.blank()

    emit.comment("% Spearman: L1 gap and beta vs alpha (signed)")
    emit.macro("CAPMSpearmanSrAlpha", fmt_signed(capm["spearman_l1_gap_safer_risk_vs_alpha_baseline"], 3))
    emit.macro("CAPMSpearmanMsAlpha", fmt_signed(capm["spearman_l1_gap_max_sharpe_vs_alpha_baseline"], 3))
    emit.macro("CAPMSpearmanBetaAlpha", fmt_signed(capm["spearman_beta_baseline_vs_alpha_baseline"], 3))
    emit.blank()

    emit.comment("% Beta shift: per-snapshot median beta of the strategy minus that of the baseline,")
    emit.comment("% median across snapshots")
    beta_pb = read(cap / "beta_per_block.csv")
    for s in ["better_return", "safer_risk", "max_sharpe", "equal_weight", "mcap_weight"]:
        shift = (beta_pb[f"median_beta_{s}"] - beta_pb["median_beta_baseline"]).median()
        emit.macro(f"CAPMDeltaBeta{STRAT_TOK[s]}Median", fmt_signed(shift, 3))
    emit.blank()

    emit.comment("% Dataset size: observations, blocks, horizon")
    emit.macro("CAPMObsRaw", str(int(rets_pb["n_wallets"].sum())))
    emit.macro("CAPMBlocksN", str(len(rets_pb)))
    h = forward_horizon(d)
    if h is None:
        warn("\\CAPMHorizonDays/ and \\MptHorizonDays/ skipped: run analyze_pct_beat_market.py --prices-dir")
    else:
        emit.macro("CAPMHorizonDays", str(h))
    emit.blank()


def emit_pct_beat_block_aggregates(emit, d: Path) -> None:
    """Per-block aggregates of pct-beat-market and pct-positive-alpha: median /
    mean across blocks, baseline min / max, the spread of strategy medians
    (PctBeat without baseline, PctPosAlpha with), the lowest pairwise Pearson
    correlation of the six pct-beat columns, and the share of blocks where the
    baseline beats the market for fewer than half of the accounts."""
    emit.comment("% --- Per-block aggregates: pct beating market / pct positive alpha ---")
    df = read(d / "return_analysis_results" / "pct_beat_market_per_block.csv")
    for s, suf in STRAT_TOK.items():
        beat, alph = df[f"pct_beat_market_ret_{s}"], df[f"pct_positive_alpha_{s}"]
        emit.macro(f"PctBeatRet{suf}BlockMedian",  fmt_dec(beat.median(), 2))
        emit.macro(f"PctBeatRet{suf}BlockMean",    fmt_dec(beat.mean(), 2))
        emit.macro(f"PctPosAlpha{suf}BlockMedian", fmt_dec(alph.median(), 2))
        emit.macro(f"PctPosAlpha{suf}BlockMean",   fmt_dec(alph.mean(), 2))
    emit.blank()

    base = df["pct_beat_market_ret_baseline"]
    emit.macro("PctBeatRetBaselineBlockMin", fmt_dec(base.min(), 2))
    emit.macro("PctBeatRetBaselineBlockMax", fmt_dec(base.max(), 2))
    emit.blank()

    # Range and spread of the median across snapshots over the actual, MPT and
    # equal-weight allocations. Mcap-weight is left out: its per-snapshot share is
    # near 0 % or near 100 % in almost every month (the cap-weighted portfolio is
    # close to ETH for nearly every account), counted separately below.
    emit.comment("% Median across snapshots: range and spread over all allocations but mcap-weight")
    no_mcap = [s for s in STRAT_TOK if s != "mcap_weight"]
    beat_med = [df[f"pct_beat_market_ret_{s}"].median() for s in no_mcap]
    alpha_med = [df[f"pct_positive_alpha_{s}"].median() for s in no_mcap]
    emit.macro("PctBeatRetBlockMedianLo", fmt_dec(min(beat_med), 1))
    emit.macro("PctBeatRetBlockMedianHi", fmt_dec(max(beat_med), 1))
    emit.macro("PctPosAlphaBlockMedianLo", fmt_dec(min(alpha_med), 1))
    emit.macro("PctPosAlphaBlockMedianHi", fmt_dec(max(alpha_med), 1))
    emit.macro("PctBeatRetMedianStrategySpread", fmt_dec(max(beat_med) - min(beat_med), 1))
    emit.macro("PctPosAlphaMedianStrategySpread", fmt_dec(max(alpha_med) - min(alpha_med), 1))
    emit.comment("% Mcap-weight: snapshots in which fewer than 10 % / more than 90 % of accounts")
    emit.comment("% beat the market or have a positive alpha")
    for col, tok in [("pct_beat_market_ret_mcap_weight", "PctBeatRetMcap"),
                     ("pct_positive_alpha_mcap_weight", "PctPosAlphaMcap")]:
        emit.macro(f"{tok}BlocksBelowTen", str(int((df[col] < 10).sum())))
        emit.macro(f"{tok}BlocksAboveNinety", str(int((df[col] > 90).sum())))
    emit.blank()

    corr = df[[f"pct_beat_market_ret_{s}" for s in STRAT_TOK]].corr().to_numpy()
    off = [corr[i, j] for i in range(len(corr)) for j in range(i + 1, len(corr))]
    emit.macro("PctBeatCrossStratCorrLo", fmt_dec(min(off), 3))
    emit.macro("PctBeatCrossStratCorrMedian", fmt_dec(pd.Series(off).median(), 3))
    emit.blank()

    emit.macro("PctBlocksBelowFifty", fmt_dec((base < 50).sum() / len(df) * 100, 1))
    eth_path = d / "return_analysis_results" / "pct_beat_eth_per_block.csv"
    if eth_path.exists():
        eth = pd.read_csv(eth_path)[["block_number", "eth_return", "market_return"]]
        m = df.merge(eth, on="block_number")
        up = m["eth_return"] > m["market_return"]
        emit.comment("% Median share of baseline accounts beating the market, in snapshots where")
        emit.comment("% ETH beat the WETH/WBTC market and in the others")
        emit.macro("PctBeatRetBaselineEthUpMedian", fmt_dec(m.loc[up, "pct_beat_market_ret_baseline"].median(), 1))
        emit.macro("PctBeatRetBaselineEthDownMedian", fmt_dec(m.loc[~up, "pct_beat_market_ret_baseline"].median(), 1))
    emit.blank()


def emit_return_paths(emit, d: Path) -> None:
    """Return paths over the 72 snapshots (returns_per_block.csv, per-snapshot
    medians): cumulative excess over the market, the spread between the six
    allocations, the market's swing, and how closely the accounts follow ETH."""
    emit.comment("% --- Return paths: per-snapshot medians over the 72 snapshots ---")
    r = read(d / "camp_analysis_results_full" / "returns_per_block.csv")
    med = pd.DataFrame({s: r[f"median_ret_{s}"] * 100 for s in STRATEGIES})
    mkt = r["median_market_return"] * 100
    cum = med.sub(mkt, axis=0).cumsum()
    emit.comment("% Cumulative sum of monthly median excess returns over the market, pp")
    for s in STRATEGIES:
        emit.macro(f"CumExcess{STRAT_TOK[s]}Pp", fmt_text_pm(cum[s].iloc[-1], 1))
    peak = int(cum["baseline"].idxmax())
    emit.macro("CumExcessBaselinePeakPp", fmt_text_pm(cum["baseline"].iloc[peak], 1))
    emit.macro("CumExcessBaselinePeakDate", pd.Timestamp(r["block_date"].iloc[peak]).strftime("%B %Y"))
    spread = med.max(axis=1) - med.min(axis=1)
    emit.comment("% Spread between the six allocations' medians in a snapshot (median over snapshots)")
    emit.comment("% and the range of the market return over the snapshots, pp")
    emit.macro("RetStratSpreadMedianPp", fmt_dec(spread.median(), 1))
    emit.macro("RetMarketSwingPp", fmt_dec(mkt.max() - mkt.min(), 1))
    prev = read(d / "return_analysis_results" / "eth_btc_prevalence.csv").set_index("metric")
    emit.macro("RetHoldsEthPct", fmt_dec(prev.at["holds_eth", "pct"], 1))
    eth_path = d / "return_analysis_results" / "pct_beat_eth_per_block.csv"
    if eth_path.exists():
        m = r.merge(pd.read_csv(eth_path)[["block_number", "eth_return"]], on="block_number")
        emit.comment("% Correlation over snapshots of the median mcap-weight / baseline return with ETH's")
        emit.macro("RetMcapEthCorr", fmt_dec(m["median_ret_mcap_weight"].corr(m["eth_return"]), 3))
        emit.macro("RetBaselineEthCorr", fmt_dec(m["median_ret_baseline"].corr(m["eth_return"]), 3))
    emit.blank()


def emit_quantile_macros(emit, df: pd.DataFrame, metric_tok: Dict[str, str]) -> None:
    """Median across blocks of per-block quantiles, in pp: {prefix}{Strategy}{BotOne,Med,TopOne}
    and {prefix}{Strategy}Skew = Q99 + Q01."""
    g = df.groupby(["strategy", "metric", "quantile"])["value"].median().reset_index()
    g["value_pct"] = g["value"] * 100.0
    for metric, m_tok in metric_tok.items():
        for s, s_tok in STRAT_TOK.items():
            sub = g[(g["strategy"] == s) & (g["metric"] == metric)]
            for q, q_tok in QUANTILE_TOK.items():
                emit.macro(f"{m_tok}{s_tok}{q_tok}", fmt_math_minus(sub.loc[sub["quantile"] == q, "value_pct"].iloc[0]))
            q01 = sub.loc[sub["quantile"] == 0.01, "value_pct"].iloc[0]
            q99 = sub.loc[sub["quantile"] == 0.99, "value_pct"].iloc[0]
            emit.macro(f"{m_tok}{s_tok}Skew", fmt_math_minus(q99 + q01))
        emit.blank()


def emit_beat_market_quantiles(emit, d: Path) -> None:
    emit.comment("% --- Beat-market margin: cross-sectional quantiles per strategy ---")
    df = read(d / "return_analysis_results" / "excess_return_quantiles_per_block.csv")
    emit_quantile_macros(emit, df, {"excess_return": "BeatExcRet", "alpha": "BeatAlpha"})
    q01 = df[(df["strategy"] == "baseline") & (df["metric"] == "excess_return") & (df["quantile"] == 0.01)]
    emit.macro("BeatExcRetBaselineBotOneAbs", f"{abs(q01['value'].median() * 100):.2f}")
    emit.blank()


def emit_eth_benchmark(emit, d: Path) -> None:
    """Beat-ETH shares and margins (analyze_pct_beat_market.py --prices-dir)."""
    ret_dir = d / "return_analysis_results"
    pct_path, q_path = ret_dir / "pct_beat_eth_per_block.csv", ret_dir / "excess_vs_eth_quantiles_per_block.csv"
    if not (pct_path.exists() and q_path.exists()):
        warn("ETH-benchmark macros skipped: run analyze_pct_beat_market.py --prices-dir")
        emit.comment("% --- ETH benchmark: skipped (pct_beat_eth_per_block.csv not present) ---")
        emit.blank()
        return
    emit.comment("% --- ETH benchmark: accounts beating ETH's forward return ---")
    eth = pd.read_csv(pct_path)
    valid = eth["eth_return"].notna()
    emit.comment("% Share of snapshots where ETH beat the 50/50 WETH/WBTC market")
    emit.macro("EthBeatMarketBlocksPct",
               fmt_dec((eth.loc[valid, "eth_return"] > eth.loc[valid, "market_return"]).mean() * 100, 1))
    for s, suf in STRAT_TOK.items():
        col = eth[f"pct_beat_eth_ret_{s}"]
        emit.macro(f"PctBeatEth{suf}BlockMedian", fmt_dec(col.median(), 2))
        emit.macro(f"PctBeatEth{suf}BlockMean",   fmt_dec(col.mean(), 2))
    emit.blank()
    emit_quantile_macros(emit, pd.read_csv(q_path), {"excess_vs_eth": "BeatExcEth"})


def emit_return_analysis(emit, d: Path) -> None:
    ret_dir = d / "return_analysis_results"
    emit.comment("% --- Return analysis: realized 20-day returns ---")
    emit.comment("% Median baseline return by MinVar distance bucket (pooled, %)")
    rb = read(ret_dir / "returns_by_distance_bucket_safer_risk.csv").set_index("dist_bucket")
    emit.macro("RetDistZeroMedianBaseline", fmt_dec(rb.at["0_exact", "median_baseline"] * 100, 2))
    emit.macro("RetDistHighMedianBaseline", fmt_dec(rb.at["(80,100]", "median_baseline"] * 100, 2))
    emit.blank()

    emit.comment("% Share of account-snapshots with a negative return, range over the six portfolios")
    rd = read(ret_dir / "return_descriptive_stats.csv").set_index("portfolio")
    neg = rd.loc[STRATEGIES, "negative_pct"]
    emit.macro("RetNegativePctLo", str(round(neg.min())))
    emit.macro("RetNegativePctHi", str(round(neg.max())))
    emit.blank()


def emit_l1_fitting(emit, d: Path) -> None:
    emit.comment("% --- L1 fitting: power-decay fit d(n) = c - a n^{-b} on the mean, psi = a / c ---")
    df = read(d / "l1_formula_fits.csv")
    df = df[df["statistic"] == "mean"]
    for opt, suf in MRV_MVR_MSR.items():
        row = df[df["strategy"] == opt].iloc[0]
        emit.macro(f"FitPsi{suf}",   f"{row['a'] / row['c']:.4f}")
        emit.macro(f"FitGamma{suf}", f"{row['b']:.4f}")
        emit.macro(f"FitDinf{suf}",  f"{row['c']:.2f}")
        emit.macro(f"FitRsq{suf}",   f"{row['r2']:.4f}")
        emit.macro(f"FitMae{suf}",   f"{row['mae']:.2f}")
    emit.blank()


def emit_rf_shap_extended(emit, d: Path) -> None:
    rs = d / "rf_shap_results"
    emit.comment("% --- Random Forest + SHAP (extended feature set, random 80/20 split) ---")
    emit.comment("% Ranges over baseline + MPT + equal-weight targets; mcap-weight separately")
    ext = read(rs / "rf_shap_importance.csv", index_col=0)
    base = read(rs / "rf_3feat_importance.csv", index_col=0)
    excluded = {f"{p}_{s}" for p in ("ret", "alpha") for s in RF_RANGE_EXCLUDE}
    ext_in, base_in = ext[~ext["target"].isin(excluded)], base[~base["target"].isin(excluded)]

    def imp_range(df, prefix: str, feat: str, col: str = "importance") -> Tuple[float, float]:
        v = df[df["target"].str.startswith(prefix) & (df["feature"] == feat)][col] * 100
        return float(v.min()), float(v.max())

    for prefix, tok in [("ret_", "Ret"), ("alpha_", "Alpha")]:
        r2 = ext_in[ext_in["target"].str.startswith(prefix)]["R2_test"]
        emit.macro(f"RfExt{tok}RsqLo", fmt_dec(r2.min(), 2))
        emit.macro(f"RfExt{tok}RsqHi", fmt_dec(r2.max(), 2))
    for feat, tok in [("month", "Month"), ("beta_baseline", "Beta"), ("log_value_usd", "Value")]:
        lo, hi = imp_range(ext_in, "ret_", feat)
        emit.macro(f"RfExtRet{tok}ImpLo", fmt_dec(lo, 0))
        emit.macro(f"RfExtRet{tok}ImpHi", fmt_dec(hi, 0))
    for feat, tok in [("month", "Month"), ("beta_baseline", "Beta")]:
        lo, hi = imp_range(ext_in, "alpha_", feat)
        emit.macro(f"RfExtAlpha{tok}ImpLo", fmt_dec(lo, 0))
        emit.macro(f"RfExtAlpha{tok}ImpHi", fmt_dec(hi, 0))
    emit.comment("% Permutation importance shares (same models): month, beta, value; the largest")
    emit.comment("% share of any other feature (size N, ETH / BTC indicators) in any model")
    for prefix, tok in [("ret_", "Ret"), ("alpha_", "Alpha")]:
        for feat, ftok in [("month", "Month"), ("beta_baseline", "Beta"), ("log_value_usd", "Value")]:
            lo, hi = imp_range(ext_in, prefix, feat, "perm_share")
            emit.macro(f"RfExt{tok}{ftok}PermLo", fmt_dec(lo, 0))
            emit.macro(f"RfExt{tok}{ftok}PermHi", fmt_dec(hi, 0))
    other = ext_in[ext_in["feature"].isin(["num_tokens", "holds_eth", "holds_btc"])]["perm_share"] * 100
    emit.macro("RfExtOtherPermHi", fmt_dec(other.max(), 0))
    ind = ext_in[ext_in["feature"].isin(["holds_eth", "holds_btc"])]["perm_share"] * 100
    emit.macro("RfExtAssetIndPermHi", fmt_dec(ind.max(), 0))

    emit.comment("% Mcap-weight targets (cap-weighted portfolio ~ ETH for nearly every account)")
    for prefix, tok in [("ret", "Ret"), ("alpha", "Alpha")]:
        sub = ext[ext["target"] == f"{prefix}_mcap_weight"]
        emit.macro(f"RfExtMcap{tok}Rsq", fmt_dec(sub["R2_test"].iloc[0], 3))
        emit.macro(f"RfExtMcap{tok}MonthImp", fmt_dec(sub.loc[sub["feature"] == "month", "importance"].iloc[0] * 100, 0))

    delta = ext_in.groupby("target")["R2_test"].first() - base_in.groupby("target")["R2_test"].first()
    ret_d, alpha_d = delta[delta.index.str.startswith("ret_")], delta[delta.index.str.startswith("alpha_")]
    emit.macro("RfDeltaRsqRetLo",   fmt_signed(ret_d.min(), 2))
    emit.macro("RfDeltaRsqRetHi",   fmt_signed(ret_d.max(), 2))
    emit.macro("RfDeltaRsqAlphaLo", fmt_signed(alpha_d.min(), 2))
    emit.macro("RfDeltaRsqAlphaHi", fmt_signed(alpha_d.max(), 2))
    emit.blank()

    shap = read(rs / "rf_shap_values_baseline.csv")
    emit.comment("% Mean |SHAP| of the baseline return model, pp; SHAP sample size")
    emit.macro("ShapMonthPp", fmt_dec(shap["month"].abs().mean(), 1))
    emit.macro("ShapBetaPp",  fmt_dec(shap["beta_baseline"].abs().mean(), 1))
    emit.macro("ShapValuePp", fmt_dec(shap["log_value_usd"].abs().mean(), 1))
    others = ["num_tokens", "holds_eth", "holds_btc"]
    emit.macro("ShapOtherPpHi", fmt_dec(shap[others].abs().mean().max(), 1))
    emit.macro("ShapSampleK", fmt_dec(len(shap) / 1000, 0))
    emit.blank()

    emit.comment("% --- Month-grouped validation: test months never seen in training ---")
    vs = read(rs / "variance_shares.csv", index_col=0)
    emit.comment("% Share of variance between months (all rows weighted by rows / by months), %")
    for target, tok in [("ret_baseline", "Ret"), ("alpha_baseline", "Alpha")]:
        emit.macro(f"VarShare{tok}BetweenRowsPct",   fmt_dec(vs.at[target, "between_month_share_all_rows"] * 100, 1))
        emit.macro(f"VarShare{tok}BetweenMonthsPct", fmt_dec(vs.at[target, "between_month_share_all_months"] * 100, 1))
    v = read(rs / "rf_variants_summary.csv")
    a6 = v[v["variant"] == "a_6feat"].set_index("target")
    emit.comment("% Sample (accounts per snapshot, total in millions) and the (a) R^2 of the baseline")
    n_months = len(read(rs / "month_outcomes.csv"))
    emit.macro("RfSamplePerMonth", fmt_thousands_tex(a6.at["ret_baseline", "n"] / n_months))
    emit.macro("RfSampleM", fmt_dec(a6.at["ret_baseline", "n"] / 1e6, 1))
    emit.macro("RfRandomRetRsq", fmt_dec(a6.at["ret_baseline", "R2"], 2))
    emit.macro("RfRandomAlphaRsq", fmt_dec(a6.at["alpha_baseline", "R2"], 2))
    emit.comment("% Out-of-fold R^2: (b) account features + month, (c) market state, (d) within-month")
    for variant, tok in [("b", "Grouped"), ("c", "State"), ("d", "Within")]:
        for target, ttok in [("ret_baseline", "Ret"), ("alpha_baseline", "Alpha")]:
            row = v[(v["variant"] == variant) & v["target"].str.startswith(target)]
            emit.macro(f"Rf{tok}{ttok}Rsq", fmt_dec(row["R2"].iloc[0], 2))
    fc = read(rs / "rf_forward_chaining.csv")
    emit.comment("% Forward chaining (market state), out-of-sample R^2 per test year: range")
    for target, ttok in [("ret_baseline", "Ret"), ("alpha_baseline", "Alpha")]:
        r = fc[fc["target"] == target]["R2_os"]
        emit.macro(f"RfForward{ttok}RsqLo", fmt_dec(r.min(), 2))
        emit.macro(f"RfForward{ttok}RsqHi", fmt_dec(r.max(), 2))
    emit.blank()


def emit_rf_and_robustness(emit, d: Path) -> None:
    emit.comment("% --- On-chain stablecoin yield evidence (DefiLlama, Ethereum-only) ---")
    by_asset = read(d / "stablecoin_overall_avg_apy_by_asset_ethereum.csv").set_index("asset")
    combined = read(d / "stablecoin_overall_avg_apy_combined_ethereum.csv").iloc[0]
    for asset in ("USDC", "USDT"):
        row = by_asset.loc[asset]
        emit.macro(f"Stablecoin{asset}AvgAPY",    fmt_dec(row["overall_avg_apy"], 2))
        emit.macro(f"Stablecoin{asset}MedianAPY", fmt_dec(row["overall_median_monthly_apy"], 2))
        emit.macro(f"Stablecoin{asset}Months",    str(int(row["months_observed"])))
    emit.macro("StablecoinPooledAvgAPY",    fmt_dec(combined["overall_avg_apy"], 2))
    emit.macro("StablecoinPooledMedianAPY", fmt_dec(combined["overall_median_monthly_apy"], 2))
    emit.macro("StablecoinPooledMonths",    str(int(combined["months_observed"])))
    emit.blank()

    path = d / "rf_test_eth" / "comparison" / "overall_summary.csv"
    if not path.exists():
        warn(f"\\RfDefault/ and \\RfRobust*/ skipped: copy rf_zero_test.py output to {path.parent}")
        emit.comment("% --- Risk-free rate robustness: skipped (rf_zero_test.py output not present) ---")
        emit.blank()
        return
    o = pd.read_csv(path).iloc[0]
    emit.comment("% --- Risk-free rate: the rf_zero_test.py re-run of the MSR optimisation ---")
    emit.comment(f"% RfDefault = the non-zero calibration the re-run tests (production MSR uses"
                 f" r_f = {o['rf_production'] * 100:g} %)")
    emit.macro("RfDefault", fmt_dec(o["rf_test"] * 100, 0))
    emit.macro("RfRobustBlocks", str(int(o["n_blocks"])))
    emit.macro("RfRobustWalletsPerBlock", fmt_thousands_tex(o["n_total"] / o["n_blocks"]))
    emit.macro("RfRobustTotalWallets", fmt_thousands_tex(o["n_total"]))
    emit.comment("% MSR weight L1 distance between the two rates, 0--100 scale (l1 / 2 * 100)")
    for stat, tok in [("median", "Median"), ("mean", "Mean"), ("p95", "TopFive"), ("p99", "TopOne")]:
        emit.macro(f"RfRobustWeightLOne{tok}Pct", fmt_dec(o[f"weight_l1_max_sharpe_{stat}"] / 2 * 100, 1))
    emit.comment("% Change in the L1-style gaps (re-run minus production), pp on the 0--100 scale")
    for col, tok in [("l1_gap_max_sharpe", "Gap"), ("l1_max_sharpe_vs_equal", "VsEqual"),
                     ("l1_max_sharpe_vs_mcap", "VsMcap")]:
        for stat, stok in [("median", "Median"), ("mean", "Mean"), ("p95_abs", "TopFive")]:
            emit.macro(f"RfRobust{tok}Diff{stok}PP", fmt_text_signed(o[f"diff_{col}_{stat}"] / 2 * 100, 1))
    emit.blank()


def emit_mpt_config(emit, d: Path) -> None:
    emit.comment("% --- MPT backtest: observations, blocks, strategies, horizon, wallets per block ---")
    pb = read(d / "distance_per_block_results" / "per_block_distance_stats.csv")
    bg = read(d / "distance_per_block_results" / "per_block_distance_by_token_group.csv")
    nw = bg[bg["optimisation"] == "better_return"].groupby("block_number")["count"].sum()
    emit.macro("MptObservations", f"{int(nw.sum()):,}")
    emit.macro("MptObservationsM", fmt_dec(nw.sum() / 1e6, 1))
    emit.macro("MptBlocks", str(pb["block_number"].nunique()))
    rets_pb = read(d / "camp_analysis_results_full" / "returns_per_block.csv")
    emit.macro("MptStrategies", str(sum(f"median_ret_{s}" in rets_pb for s in STRATEGIES)))
    h = forward_horizon(d)
    if h is not None:
        emit.macro("MptHorizonDays", str(h))
    emit.macro("MptMedianWalletsPerBlock", f"{int(round(nw.median())):,}")
    emit.macro("MptMinWalletsPerBlock", f"{int(nw.min()):,}")
    emit.macro("MptMaxWalletsPerBlock", f"{int(nw.max()):,}")
    emit.blank()


def emit_l1_token_group(emit, d: Path) -> None:
    emit.comment("% --- L1 distance by token-group threshold (N = 5) ---")
    df = read(d / "distance_per_block_results" / "per_block_distance_by_token_group.csv")
    emit.macro("TotalBlocks", str(df["block_number"].nunique()))
    emit.blank()
    emit.comment("% --- Per-group means (time-series averages) ---")
    for opt, suf in MRV_MVR_MSR.items():
        for grp, label in [("lt5", "Small"), ("gte5", "Large")]:
            sub = df[(df["optimisation"] == opt) & (df["token_group"] == grp)]
            emit.macro(f"LOne{label}{suf}Mean", f"{sub['mean_dist'].mean():.1f}")
        emit.blank()
    emit.comment("% --- Hypothesis test results (paired Wilcoxon, N=5 threshold) ---")
    tests = read(d / "distance_per_block_results" / "hypothesis_tests_l1_by_token_group.csv")
    for opt, suf in MRV_MVR_MSR.items():
        row = tests[tests["optimisation"] == opt].iloc[0]
        emit.macro(f"LOneDelta{suf}", f"{row['mean_diff_gte5_minus_lt5']:.1f}")
        emit.macro(f"LOneCohen{suf}", f"{row['cohens_d_paired']:.2f}")
    emit.blank()


# -----------------------------------------------------------------------------
# entry point
# -----------------------------------------------------------------------------
def emit_distance_return_macros(emit, data_dir: Optional[Path] = None) -> None:
    """Emit every distance / return macro. `emit` needs .comment(text),
    .macro(name, value), .macro_raw(line) and .blank() (MacroEmitter in
    generate_all_macros.py)."""
    d = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    capm = parse_capm_log(d / "analyze_extended_capm_output.txt")
    emit_l1_distance_main(emit, d)
    emit_naive_benchmark(emit, capm)
    emit_capm(emit, d, capm)
    emit_pct_beat_block_aggregates(emit, d)
    emit_return_paths(emit, d)
    emit_beat_market_quantiles(emit, d)
    emit_eth_benchmark(emit, d)
    emit_return_analysis(emit, d)
    emit_l1_fitting(emit, d)
    emit_rf_shap_extended(emit, d)
    emit_rf_and_robustness(emit, d)
    emit_mpt_config(emit, d)
    emit_l1_token_group(emit, d)


# -----------------------------------------------------------------------------
# appendix tables (printed, pasted into the LaTeX source)
# -----------------------------------------------------------------------------
def table_rows(data_dir: Optional[Path] = None) -> str:
    """Rows of the threshold-sweep table (paper App. C, tab:threshold_sweep).
    The N = 5 rows use the \\LOne{Small,Large}*Mean/, \\LOneDelta*/ and
    \\LOneCohen*/ macros of the main text."""
    d = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    sw = read(d / "distance_per_block_results" / "hypothesis_tests_threshold_sweep.csv")
    p = sw["wilcoxon_p"]
    out = [f"% Threshold sweep: Wilcoxon p from {p.min():.3g} to {p.max():.3g}, "
           f"{int(sw['n_blocks'].min())}-{int(sw['n_blocks'].max())} paired blocks"]
    for i, (opt, suf) in enumerate(MRV_MVR_MSR.items()):
        if i:
            out.append(r"\midrule")
        for j, (_, r) in enumerate(sw[sw["optimisation"] == opt].sort_values("threshold").iterrows()):
            lead = rf"\gls{{{suf}}}" if j == 0 else ""
            n = int(r["threshold"])
            if n == SPLIT_N:
                cells = [rf"\LOneSmall{suf}Mean/", rf"\LOneLarge{suf}Mean/", rf"\LOneDelta{suf}/", rf"\LOneCohen{suf}/"]
            else:
                cells = [f"{r['mean_below']:.1f}", f"{r['mean_above']:.1f}", f"{r['mean_diff']:.1f}",
                         f"{r['cohens_d']:.2f}"]
            out.append(f"{lead} & {n} & " + " & ".join(cells) + r" \\")

    # Random Forest tables (paper App. D): permutation importance shares of the
    # baseline models, and test R^2 of the 3- vs 6-feature models per strategy.
    rs = d / "rf_shap_results"
    ext = read(rs / "rf_shap_importance.csv", index_col=0)
    b3 = read(rs / "rf_3feat_importance.csv", index_col=0)

    def share(df, target, feat):
        v = df[(df["target"] == target) & (df["feature"] == feat)]["perm_share"]
        if v.empty:
            return "---"
        x = float(v.iloc[0]) * 100
        return r"$<$1\%" if x < 0.5 else f"{x:.0f}\\%"

    out.append("% Random Forest permutation importance shares, baseline strategy (tab:rf_importance)")
    for feat, label in [("month", "Snapshot month"), ("beta_baseline", r"Portfolio $\beta$"),
                        ("log_value_usd", "Log portfolio value"), ("num_tokens", r"Portfolio size $N$"),
                        ("holds_eth", "Holds ETH"), ("holds_btc", "Holds BTC wrapper")]:
        cells = [share(b3, "ret_baseline", feat), share(ext, "ret_baseline", feat),
                 share(b3, "alpha_baseline", feat), share(ext, "alpha_baseline", feat)]
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    v = read(rs / "rf_variants_summary.csv")
    r2 = v[v["variant"].isin(["a_3feat", "a_6feat"])].set_index(["variant", "target"])["R2"]
    out.append(r"\midrule")
    out.append(r"$R^{2}$ (test) & " + " & ".join(
        f"{r2[(var, tgt)]:.2f}" for tgt in ("ret_baseline", "alpha_baseline") for var in ("a_3feat", "a_6feat")) + r" \\")

    out.append("% Test R^2 of the 3- vs 6-feature models per strategy (tab:rf_3v6)")
    labels = {"baseline": "Baseline", "better_return": "MaxRet", "safer_risk": "MinVar",
              "max_sharpe": "MaxSR", "equal_weight": "Equal-weight", "mcap_weight": "MCap-weight"}
    for s, label in labels.items():
        cells = []
        for pre in ("ret", "alpha"):
            a, b = r2[("a_3feat", f"{pre}_{s}")], r2[("a_6feat", f"{pre}_{s}")]
            cells += [f"{a:.2f}", f"{b:.2f}", f"{b - a:+.2f}"]
        out.append(f"{label:<13} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


class _PrintEmitter:
    """Minimal stand-in for generate_all_macros.MacroEmitter."""

    def __init__(self):
        self.lines = []

    def comment(self, text: str = "") -> None:
        self.lines.extend(text.split("\n"))

    def blank(self) -> None:
        self.lines.append("")

    def macro(self, name: str, value: str) -> None:
        self.lines.append(rf"\def\{name}/{{{value}}}")

    def macro_raw(self, line: str) -> None:
        self.lines.append(line)


def main() -> None:
    p = argparse.ArgumentParser(description="Print the distance / return LaTeX macros (or the appendix table rows).")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--tables", action="store_true", help="Print the appendix table rows instead of the macros.")
    args = p.parse_args()
    if args.tables:
        print(table_rows(args.data_dir))
        return
    e = _PrintEmitter()
    emit_distance_return_macros(e, args.data_dir)
    print("\n".join(e.lines))


if __name__ == "__main__":
    main()
