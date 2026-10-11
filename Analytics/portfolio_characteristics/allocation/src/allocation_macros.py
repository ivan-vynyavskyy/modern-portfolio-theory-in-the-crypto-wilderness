#!/usr/bin/env python3
"""
allocation_macros.py

Description:
    Single owner of every LaTeX macro derived from the allocation CSVs in
    ../data: weight concentration (weight_concentration.py, with and without
    positions below 1 USD) and trading versus drift (trading_vs_drift.py).
    Called by Analytics/generate_all_macros.py; can also be run standalone to
    print the macros, and with --tables to print the rows of the appendix
    tables (paper App. B), which are pasted into the LaTeX source.

    Aggregation basis, as for the portfolio-size macros: each quantity is
    computed per snapshot (per consecutive snapshot pair for trading) and
    then averaged over the snapshots. Shares are in percent. "Multi-asset"
    means at least two priced assets, ETH and WETH counted as one.

Macros:
    Effective diversification (weight_concentration_summary.csv,
    effective_n_hist.csv):
        \\EffSingleAssetPct/              accounts with >= 99 % of value in one asset
        \\EffSingleAssetFirstPct/, \\EffSingleAssetLastPct/   same, first / last snapshot
        \\EffGtNinetyPct/                 accounts with > 90 % in one asset
        \\EffNTwoPlusPct/                 accounts with effective N >= 2
        \\EffMultiMaxWeightMedianPct/     median largest weight of multi-asset accounts
        \\EffMultiEffNMedian/             median effective N of multi-asset accounts
        \\EffTenPlusMeanN/, \\EffTenPlusEffNMedian/  accounts with 10+ assets:
                                         mean N and median effective N
        \\EffRichMultiMeanN/              multi-asset accounts above 100K USD: mean N
        \\EffRichMultiMaxWeightMedianPct/ ... and their median largest weight
        \\EffMidMultiMaxWeightMedianPct/  median largest weight at 100-1K USD
    Dust (the same with positions below 1 USD ignored,
    weight_concentration_summary_min1usd.csv):
        \\EffDustMultiDropPct/, \\EffDustMultiDropLastPct/
            reduction in the number of multi-asset accounts (mean, last snapshot)
        \\EffDustFivePlusDropPct/, \\EffDustFivePlusDropLastPct/
            reduction in the number of accounts with 5+ assets
    Trading (trading_drift_summary.csv, trading_drift_hist.csv, _entries.csv):
        \\TradePairsN/                    consecutive snapshot pairs
        \\TradeNoChangePct/               accounts keeping a balance whose holdings
                                         do not change at all
        \\TradeNoChangeMultiPct/          same, multi-asset accounts
        \\TradeNoChangeMultiLowPct/, \\TradeNoChangeMultiHighPct/
                                         same, multi-asset accounts below 1 USD / above 100K USD
        \\TradeQuarterMultiPct/           multi-asset accounts with turnover > 25 %
        \\TradeQuarterTurnoverSharePct/   lower bound (rounded down) of their share
                                         of all multi-asset turnover
        \\TradeTurnoverMultiPct/, \\TradeDriftMultiPct/  mean turnover / drift,
                                         multi-asset accounts
        \\TradeQuarter{OneK,TenK}{TwentyOne,TwentyFive}Pct/
            turnover > 25 % among multi-asset accounts at 1K-10K / 10K-100K USD,
            mean over the pairs starting in 2021 / 2025
        \\TradeExitPct/, \\TradeExitEOAPct/, \\TradeExitCAPct/  accounts emptying
                                         their balance per month
        \\TradeEntriesM/                  accounts gaining a balance per month, millions
        \\TradeEntriesSmallPct/           share of them starting below 100 USD

Input (in --data-dir): the CSVs named above.

Usage:
    python allocation_macros.py                 # prints the macros
    python allocation_macros.py --tables        # prints the appendix table rows
    python allocation_macros.py --data-dir ../data
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CONC = "weight_concentration_summary.csv"
CONC_DUST = "weight_concentration_summary_min1usd.csv"
EFFN_HIST = "effective_n_hist.csv"
EFFN_HIST_DUST = "effective_n_hist_min1usd.csv"
DRIFT = "trading_drift_summary.csv"
DRIFT_HIST = "trading_drift_hist.csv"
ENTRIES = "trading_drift_entries.csv"

ALL = "ALL"
MULTI = "2+"
N_MULTI = ["2", "3–4", "5–9", "10+"]
WEALTH = ["0–1", "1–100", "100–1K", "1K–10K", "10K–100K", ">100K"]
WEALTH_TEX = {"0–1": r"\$0--\$1", "1–100": r"\$1--\$100", "100–1K": r"\$100--\$1{,}000",
              "1K–10K": r"\$1{,}000--\$10{,}000", "10K–100K": r"\$10{,}000--\$100{,}000",
              ">100K": r"$>$\$100{,}000"}
ABOVE_100 = ["100–1K", "1K–10K", "10K–100K", ">100K"]
EFFN_GE2 = ["[2,3)", "[3,5)", "[5,10)", "10+"]
QUARTER_EDGES = {"(25,50%]": 0.25, "(50,75%]": 0.50, "(75,100%]": 0.75}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def fmt_dec(v, dp: int) -> str:
    return f"{round(float(v), dp):.{dp}f}"


def read(data_dir: Path, name: str) -> pd.DataFrame:
    p = data_dir / name
    if not p.exists():
        raise SystemExit(f"[allocation_macros] missing input {p}")
    return pd.read_csv(p)


def rows(df: pd.DataFrame, wt: str = ALL, wb: str = ALL, nb: str = ALL, key: str = "block") -> pd.DataFrame:
    """Rows of one group (wallet type x wealth bucket x size bucket), one per
    snapshot, indexed by `key`. Use sum_rows to add several groups."""
    x = df[(df["wallet_type"] == wt) & (df["wealth_bucket"] == wb) & (df["n_bucket"] == nb)]
    return x.set_index(key).sort_index()


def sum_rows(df: pd.DataFrame, cols, wt: str = ALL, wbs=(ALL,), nbs=(ALL,), key: str = "block") -> pd.DataFrame:
    x = df[(df["wallet_type"] == wt) & df["wealth_bucket"].isin(wbs) & df["n_bucket"].isin(nbs)]
    return x.groupby(key)[list(cols)].sum().sort_index()


def effn_ge2_count(hist: pd.DataFrame, wbs=(ALL,)) -> pd.Series:
    x = hist[(hist["wallet_type"] == ALL) & hist["wealth_bucket"].isin(wbs) & (hist["n_bucket"] == MULTI)
             & hist["effective_n_bin"].isin(EFFN_GE2)]
    return x.groupby("block")["n_accounts"].sum()


# -----------------------------------------------------------------------------
# quantities
# -----------------------------------------------------------------------------
def concentration_values(conc: pd.DataFrame, effn: pd.DataFrame) -> dict:
    a = rows(conc)
    one = rows(conc, nb="1")
    m = rows(conc, nb=MULTI)
    ten = rows(conc, nb="10+")
    rich = rows(conc, wb=">100K", nb=MULTI)
    mid = rows(conc, wb="100–1K", nb=MULTI)
    ge99 = a["pct_max_weight_ge_99"]
    return {
        "single": ge99.mean(),
        "single_first": ge99.iloc[0],
        "single_last": ge99.iloc[-1],
        "gt90": (100.0 * (one["n_accounts"] + a["n_cap_binding"]) / a["n_accounts"]).mean(),
        "effn2": (100.0 * effn_ge2_count(effn).reindex(a.index).fillna(0) / a["n_accounts"]).mean(),
        "multi_maxw": 100.0 * m["median_max_weight"].mean(),
        "multi_effn": m["median_effective_n"].mean(),
        "ten_n": ten["mean_n"].mean(),
        "ten_effn": ten["median_effective_n"].mean(),
        "rich_n": rich["mean_n"].mean(),
        "rich_maxw": 100.0 * rich["median_max_weight"].mean(),
        "mid_maxw": 100.0 * mid["median_max_weight"].mean(),
    }


def dust_values(conc: pd.DataFrame, dust: pd.DataFrame) -> dict:
    m0, m1 = rows(conc, nb=MULTI)["n_accounts"], rows(dust, nb=MULTI)["n_accounts"]
    f0 = sum_rows(conc, ["n_accounts"], nbs=("5–9", "10+"))["n_accounts"]
    f1 = sum_rows(dust, ["n_accounts"], nbs=("5–9", "10+"))["n_accounts"]
    drop_m = 100.0 * (1.0 - m1.reindex(m0.index) / m0)
    drop_f = 100.0 * (1.0 - f1.reindex(f0.index) / f0)
    return {"multi": drop_m.mean(), "multi_last": drop_m.iloc[-1],
            "five": drop_f.mean(), "five_last": drop_f.iloc[-1]}


def drift_frame(drift: pd.DataFrame, wbs=(ALL,), nbs=tuple(N_MULTI), wt: str = ALL) -> pd.DataFrame:
    """Per pair: counts summed over the groups, turnover and drift sums rebuilt
    from the group means."""
    x = drift[(drift["wallet_type"] == wt) & drift["wealth_bucket"].isin(wbs) & drift["n_bucket"].isin(nbs)].copy()
    x["sum_turnover"] = x["mean_turnover"].fillna(0.0) * x["n_measured"]
    x["sum_drift"] = x["mean_drift"].fillna(0.0) * x["n_measured"]
    cols = ["n_accounts_t", "n_exit", "n_stay", "n_measured", "n_units_unchanged",
            "n_turnover_le_25pct", "sum_turnover", "sum_drift"]
    return x.groupby("block_t")[cols].sum().sort_index()


def drift_shares(g: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "no_change": 100.0 * g["n_units_unchanged"] / g["n_stay"],
        "quarter": 100.0 - 100.0 * g["n_turnover_le_25pct"] / g["n_measured"],
        "turnover": 100.0 * g["sum_turnover"] / g["n_measured"],
        "drift": 100.0 * g["sum_drift"] / g["n_measured"],
    })


def pair_years(drift: pd.DataFrame) -> pd.Series:
    """Calendar year of the first snapshot of each pair (monthly snapshots from
    January 2020, one per month)."""
    blocks = sorted(drift["block_t"].unique())
    return pd.Series({b: 2020 + i // 12 for i, b in enumerate(blocks)})


def trading_values(drift: pd.DataFrame, hist: pd.DataFrame, entries: pd.DataFrame) -> dict:
    allacc = rows(drift, key="block_t")
    multi = drift_shares(drift_frame(drift))
    low = drift_shares(drift_frame(drift, wbs=("0–1",)))
    high = drift_shares(drift_frame(drift, wbs=(">100K",)))

    # lower bound of the turnover share of accounts above 25 %: bin count x lower edge
    h = hist[(hist["wallet_type"] == ALL) & (hist["wealth_bucket"] == ALL) & hist["n_bucket"].isin(N_MULTI)
             & (hist["metric"] == "turnover") & hist["bin"].isin(QUARTER_EDGES)]
    low_bound = (h["n_accounts"] * h["bin"].map(QUARTER_EDGES)).groupby(h["block_t"]).sum()
    g = drift_frame(drift)
    share_lb = 100.0 * low_bound.reindex(g.index).fillna(0.0) / g["sum_turnover"]

    years = pair_years(drift)
    by_year = {}
    for tag, wb in (("OneK", "1K–10K"), ("TenK", "10K–100K")):
        q = drift_shares(drift_frame(drift, wbs=(wb,)))["quarter"]
        for ytag, year in (("TwentyOne", 2021), ("TwentyFive", 2025)):
            by_year[f"{tag}{ytag}"] = q[years.reindex(q.index) == year].mean()

    e = entries[entries["wallet_type"] == ALL]
    e_all = e[e["wealth_bucket"] == ALL].set_index("block_t")["n_entries"].sort_index()
    e_small = e[e["wealth_bucket"].isin(["0–1", "1–100"])].groupby("block_t")["n_entries"].sum()
    return {
        "pairs": len(allacc),
        "no_change": allacc["pct_units_unchanged"].mean(),
        "no_change_multi": multi["no_change"].mean(),
        "no_change_low": low["no_change"].mean(),
        "no_change_high": high["no_change"].mean(),
        "quarter": multi["quarter"].mean(),
        "quarter_share_lb": math.floor(share_lb.mean()),
        "turnover": multi["turnover"].mean(),
        "drift": multi["drift"].mean(),
        "by_year": by_year,
        "exit": allacc["pct_exit"].mean(),
        "exit_eoa": rows(drift, wt="EOA", key="block_t")["pct_exit"].mean(),
        "exit_ca": rows(drift, wt="CA", key="block_t")["pct_exit"].mean(),
        "entries_m": e_all.mean() / 1e6,
        "entries_small": (100.0 * e_small.reindex(e_all.index).fillna(0) / e_all).mean(),
    }


# -----------------------------------------------------------------------------
# emitter
# -----------------------------------------------------------------------------
def emit_allocation_macros(emit, data_dir: Optional[Path] = None) -> None:
    """Write all allocation macros through `emit`, which must provide
    .comment(text), .macro(name, value) and .blank() (MacroEmitter in
    generate_all_macros.py, or _PrintEmitter below)."""
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    conc, dust = read(data_dir, CONC), read(data_dir, CONC_DUST)
    effn = read(data_dir, EFFN_HIST)
    drift, dhist, ent = read(data_dir, DRIFT), read(data_dir, DRIFT_HIST), read(data_dir, ENTRIES)

    c = concentration_values(conc, effn)
    emit.comment(f"% --- Effective diversification (avg across {conc['block'].nunique()} monthly snapshots; "
                 "allocation/src/allocation_macros.py) ---")
    emit.macro("EffSingleAssetPct", fmt_dec(c["single"], 1))
    emit.macro("EffSingleAssetFirstPct", fmt_dec(c["single_first"], 1))
    emit.macro("EffSingleAssetLastPct", fmt_dec(c["single_last"], 1))
    emit.macro("EffGtNinetyPct", fmt_dec(c["gt90"], 1))
    emit.macro("EffNTwoPlusPct", fmt_dec(c["effn2"], 1))
    emit.macro("EffMultiMaxWeightMedianPct", fmt_dec(c["multi_maxw"], 1))
    emit.macro("EffMultiEffNMedian", fmt_dec(c["multi_effn"], 2))
    emit.macro("EffTenPlusMeanN", fmt_dec(c["ten_n"], 1))
    emit.macro("EffTenPlusEffNMedian", fmt_dec(c["ten_effn"], 2))
    emit.macro("EffRichMultiMeanN", fmt_dec(c["rich_n"], 1))
    emit.macro("EffRichMultiMaxWeightMedianPct", fmt_dec(c["rich_maxw"], 1))
    emit.macro("EffMidMultiMaxWeightMedianPct", fmt_dec(c["mid_maxw"], 1))

    dv = dust_values(conc, dust)
    emit.macro("EffDustMultiDropPct", fmt_dec(dv["multi"], 1))
    emit.macro("EffDustMultiDropLastPct", fmt_dec(dv["multi_last"], 1))
    emit.macro("EffDustFivePlusDropPct", fmt_dec(dv["five"], 1))
    emit.macro("EffDustFivePlusDropLastPct", fmt_dec(dv["five_last"], 1))
    emit.blank()

    t = trading_values(drift, dhist, ent)
    emit.comment(f"% --- Trading activity (avg across {t['pairs']} consecutive snapshot pairs; "
                 "allocation/src/allocation_macros.py) ---")
    emit.macro("TradePairsN", str(t["pairs"]))
    emit.macro("TradeNoChangePct", fmt_dec(t["no_change"], 1))
    emit.macro("TradeNoChangeMultiPct", fmt_dec(t["no_change_multi"], 1))
    emit.macro("TradeNoChangeMultiLowPct", fmt_dec(t["no_change_low"], 1))
    emit.macro("TradeNoChangeMultiHighPct", fmt_dec(t["no_change_high"], 1))
    emit.macro("TradeQuarterMultiPct", fmt_dec(t["quarter"], 1))
    emit.macro("TradeQuarterTurnoverSharePct", str(t["quarter_share_lb"]))
    emit.macro("TradeTurnoverMultiPct", fmt_dec(t["turnover"], 1))
    emit.macro("TradeDriftMultiPct", fmt_dec(t["drift"], 1))
    for tag, v in t["by_year"].items():
        emit.macro(f"TradeQuarter{tag}Pct", fmt_dec(v, 1))
    emit.macro("TradeExitPct", fmt_dec(t["exit"], 2))
    emit.macro("TradeExitEOAPct", fmt_dec(t["exit_eoa"], 2))
    emit.macro("TradeExitCAPct", fmt_dec(t["exit_ca"], 2))
    emit.macro("TradeEntriesM", fmt_dec(t["entries_m"], 2))
    emit.macro("TradeEntriesSmallPct", fmt_dec(t["entries_small"], 1))
    emit.blank()


# -----------------------------------------------------------------------------
# appendix tables (printed, pasted into the LaTeX source)
# -----------------------------------------------------------------------------
def table_rows(data_dir: Path) -> str:
    conc, dust = read(data_dir, CONC), read(data_dir, CONC_DUST)
    effn, effn_d = read(data_dir, EFFN_HIST), read(data_dir, EFFN_HIST_DUST)
    drift = read(data_dir, DRIFT)
    out = []

    # Table 1: nominal versus effective size of multi-asset portfolios
    out.append("% Nominal vs effective size (multi-asset accounts, mean over snapshots)")
    m = rows(conc, nb=MULTI)["n_accounts"]
    for nb, label in (("2", "2"), ("3–4", "3--4"), ("5–9", "5--9"), ("10+", "10+"), (MULTI, "All multi-asset")):
        r = rows(conc, nb=nb)
        share = (100.0 * r["n_accounts"] / m).mean()
        out.append(f"{label} & {share:.1f} & {r['mean_n'].mean():.1f} & {r['median_max_weight'].mean():.2f} & "
                   f"{r['median_effective_n'].mean():.2f} & {r['pct_max_weight_ge_99'].mean():.1f} \\\\")

    # Table 2: with and without positions below 1 USD
    out.append("% Dust sensitivity (mean over snapshots)")

    def above100(df: pd.DataFrame, eh: pd.DataFrame) -> dict:
        a = sum_rows(df, ["n_accounts", "n_cap_binding", "n_max_weight_ge_99"], wbs=ABOVE_100)
        cnt = {nb: sum_rows(df, ["n_accounts"], wbs=ABOVE_100, nbs=(nb,))["n_accounts"] for nb in ("1", "2", "3–4")}
        five = sum_rows(df, ["n_accounts"], wbs=ABOVE_100, nbs=("5–9", "10+"))["n_accounts"]
        pct = lambda s: (100.0 * s.reindex(a.index).fillna(0) / a["n_accounts"]).mean()  # noqa: E731
        return {"1": pct(cnt["1"]), "2": pct(cnt["2"]), "3–4": pct(cnt["3–4"]), "5+": pct(five),
                "ge99": pct(a["n_max_weight_ge_99"]), "gt90": pct(cnt["1"] + a["n_cap_binding"]),
                "effn2": pct(effn_ge2_count(eh, wbs=ABOVE_100))}

    x, y = above100(conc, effn), above100(dust, effn_d)
    for key, label in (("1", "One asset"), ("2", "Two assets"), ("3–4", "Three to four assets"),
                       ("5+", "Five or more assets"), ("ge99", r"At least 99\,\% in one asset"),
                       ("gt90", r"More than 90\,\% in one asset"), ("effn2", r"$N_{\mathrm{eff}} \geq 2$")):
        out.append(f"{label} & {x[key]:.1f} & {y[key]:.1f} \\\\")
    mm0, mm1 = rows(conc, nb=MULTI)["n_accounts"], rows(dust, nb=MULTI)["n_accounts"]
    f0 = sum_rows(conc, ["n_accounts"], nbs=("5–9", "10+"))["n_accounts"]
    f1 = sum_rows(dust, ["n_accounts"], nbs=("5–9", "10+"))["n_accounts"]
    out.append(f"Multi-asset accounts (million) & {mm0.mean() / 1e6:.2f} & {mm1.mean() / 1e6:.2f} \\\\")
    out.append(f"Accounts with five or more assets (thousand) & {f0.mean() / 1e3:.0f} & {f1.mean() / 1e3:.0f} \\\\")

    # Table 3: trading activity of multi-asset accounts by wealth
    out.append("% Trading activity of multi-asset accounts by wealth at t (mean over pairs)")
    for wb in WEALTH + [ALL]:
        s = drift_shares(drift_frame(drift, wbs=(wb,))).mean()
        label = "All multi-asset" if wb == ALL else WEALTH_TEX[wb]
        out.append(f"{label} & {s['no_change']:.1f} & {s['quarter']:.1f} & {s['turnover']:.1f} & {s['drift']:.1f} \\\\")
    return "\n".join(out)


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
    ap = argparse.ArgumentParser(description="Print the allocation LaTeX macros (or the appendix table rows).")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--tables", action="store_true", help="Print the appendix table rows instead of the macros.")
    args = ap.parse_args()
    if args.tables:
        print(table_rows(Path(args.data_dir)))
        return
    e = _PrintEmitter()
    emit_allocation_macros(e, Path(args.data_dir))
    print("\n".join(e.lines))


if __name__ == "__main__":
    main()
