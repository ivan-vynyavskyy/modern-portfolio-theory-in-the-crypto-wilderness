#!/usr/bin/env python3
"""
distribution_of_wealth_macros.py

Description:
    Single owner of every LaTeX macro derived from the wealth-distribution
    CSVs in ../data (produced by compute_value_buckets_per_block.py,
    compute_top_value_concentration_per_block.py and
    top_pct_ca_eoa_breakdown.py). Called by Analytics/generate_all_macros.py;
    can also be run standalone to print the macros or the yearly top-share
    table (formerly compute_topshare_yearly_table.py).

    All statistics are per monthly snapshot and aggregated over the 72
    snapshots (2020-01 .. 2025-12). Native ETH is included; the beacon
    deposit contract is excluded upstream (see the producing scripts).
    Snapshot dates come from token_inequality_per_block.csv in the same
    folder; DeFiLlama's Ethereum TVL from defillama_eth_tvl.csv (cached by
    plot/load_defillama_tvl.R, one row per snapshot date).

Macros (names and formatting unchanged from the original macro.tex):
    From value_buckets_minimal.csv
        \\WealthBucketCA<B>P/, \\WealthBucketEOA<B>P/
                                  mean % of CA / EOA accounts per USD bucket
                                  (B = ZeroOne, OneHundred, HundredOneK,
                                  OneKTenK, TenKHundredK, OverHundredK)
        \\WealthShareCAMeanP/, \\WealthShareEOAMeanP/
                                  mean share of total value held by CA / EOA
        \\CAShareHoldersMeanPct/  mean % of the accounts holding assets at a
                                  snapshot that are CAs (CA / ALL wallet_count;
                                  the population the top-holder breakdowns rank,
                                  unlike the cumulative \\CAShareEndPct/)
        \\WealthPeakTwentyone/, \\WealthPeakTwentyfiveB/
                                  highest snapshot total value in 2021 / 2025
        \\OverallWealthPeak/, \\OverallWealthPeakDate/, \\OverallWealthEnd/
        \\CaHoldingsPeak/         highest snapshot value held by CAs
    With defillama_eth_tvl.csv
        \\EthTvlPeak/, \\EthTvlPeakDate/, \\EthTvlEnd/
        \\CaOverTvlPeakRatio/     CA-held value / TVL at the TVL peak
        \\WealthOverTvlMedianRatio/, \\WealthOverTvlMinRatio/, \\WealthOverTvlMaxRatio/
                                  total value / TVL over the snapshots
    From top_held_percent_gr_1.csv (accounts above 1 USD, wallet_type ALL)
        \\TopOneWealthTwenty/, \\TopOneWealthTwentyfive/
                                  mean top-1 % value share over the 2020 /
                                  2025 snapshots
        \\TopFiveTenWealthMin/    lowest top-5 % value share over all snapshots
        \\TopTenWealthMin/        lowest top-10 % value share over all snapshots
    From top_1_pct_ca_eoa_breakdown.csv, top_0_1_pct_ca_eoa_breakdown.csv
        \\TopOneCAPctWallets/, \\TopOneEOAPctWallets/
                                  mean % of the top-1 % accounts that are CA / EOA
        \\TopOneCAPctValue/, \\TopOneEOAPctValue/
                                  mean % of the top-1 % value held by CA / EOA
        \\TopPointOneCAPctWallets/, \\TopPointOneCAPctValue/
                                  the same for the top 0.1 %

    Comparison with the ERC-20-only reconstruction (emitted only when the
    *_erc20_only.csv files are present): every macro above with suffix
    ErcOnly (e.g. \\WealthShareCAMeanPErcOnly/, \\TopOneWealthTwentyfiveErcOnly/),
    plus the changes ETH-inclusive minus ERC-20-only:
        \\WealthShareCAMeanDeltaPP/       CA share of value, percentage points
        \\OverallWealthEndDeltaPct/       total value at the last snapshot, %
        \\TopOneWealthTwentyfiveDeltaPP/  top-1 % share in 2025, percentage points
        \\TopOneCAPctValueDeltaPP/        CA share of the top-1 % value, pp

Input (in --data-dir):
    value_buckets_minimal.csv        (block, wallet_type, bucket, wallet_count,
                                      wallet_pct, value_sum_usd, value_pct)
    top_held_percent_gr_1.csv        (block_number, wallet_type, wallets_considered,
                                      total_value_usd, top_{1,5,10}_share_pct, ...)
    top_1_pct_ca_eoa_breakdown.csv, top_0_1_pct_ca_eoa_breakdown.csv
    token_inequality_per_block.csv   (date, block, ...) for the block -> date map
    defillama_eth_tvl.csv            (date, tvl_usd)
    optional: the first four with suffix _erc20_only

Usage:
    python distribution_of_wealth_macros.py                    # prints the macros
    python distribution_of_wealth_macros.py --data-dir ../data
    python distribution_of_wealth_macros.py --yearly-table     # yearly top-share table (ALL)
    python distribution_of_wealth_macros.py --yearly-table --wallet-type CA --latex
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ERC_ONLY_SUFFIX = "_erc20_only"

F_BUCKETS = "value_buckets_minimal"
F_TOPSHARE = "top_held_percent_gr_1"
F_TOP1 = "top_1_pct_ca_eoa_breakdown"
F_TOP01 = "top_0_1_pct_ca_eoa_breakdown"
F_DATES = "token_inequality_per_block"
F_TVL = "defillama_eth_tvl"

# macro suffix -> bucket label in value_buckets_minimal.csv
BUCKET_MAP = [
    ("ZeroOneP", "0–1"),
    ("OneHundredP", "1–100"),
    ("HundredOneKP", "100–1K"),
    ("OneKTenKP", "1K–10K"),
    ("TenKHundredKP", "10K–100K"),
    ("OverHundredKP", ">100K"),
]


# -----------------------------------------------------------------------------
# formatting (same output as generate_all_macros)
# -----------------------------------------------------------------------------
def fmt_dec(v, dp: int) -> str:
    return f"{round(float(v), dp):.{dp}f}"


def fmt_num2(v) -> str:
    return rf"\num{{{float(v):.2f}}}"


def usd_b(v) -> str:
    return rf"\${fmt_dec(float(v) / 1e9, 1)}\,B"


def usd_t(v) -> str:
    return rf"\${fmt_dec(float(v) / 1e12, 2)}\,T"


# -----------------------------------------------------------------------------
# loading
# -----------------------------------------------------------------------------
def _read(path: Path, required: set) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"[distribution_of_wealth_macros] {path.name} lacks columns {sorted(missing)}")
    return df


def load_block_dates(data_dir: Path, suffix: str = "") -> pd.Series:
    """block -> snapshot date (Timestamp), from the inequality CSV of the same run."""
    df = pd.read_csv(data_dir / f"{F_DATES}{suffix}.csv", usecols=["date", "block"], dtype=str)
    df["block"] = pd.to_numeric(df["block"], errors="coerce")  # tolerate a stray header row
    df = df.dropna(subset=["block"]).drop_duplicates("block")
    return pd.Series(pd.to_datetime(df["date"]).values, index=df["block"].astype("int64").values).sort_index()


def load_buckets(data_dir: Path, suffix: str = "") -> pd.DataFrame:
    return _read(data_dir / f"{F_BUCKETS}{suffix}.csv",
                 {"block", "wallet_type", "bucket", "wallet_count", "wallet_pct", "value_sum_usd"})


def load_topshare(data_dir: Path, suffix: str = "") -> pd.DataFrame:
    return _read(data_dir / f"{F_TOPSHARE}{suffix}.csv",
                 {"block_number", "wallet_type", "wallets_considered", "top_1_share_pct",
                  "top_5_share_pct", "top_10_share_pct"})


def load_top_breakdowns(data_dir: Path, suffix: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    need = {"block", "wallet_type", "n_wallets_pct_of_top1", "value_pct_of_top1"}
    return (_read(data_dir / f"{F_TOP1}{suffix}.csv", need),
            _read(data_dir / f"{F_TOP01}{suffix}.csv", need))


def load_tvl(data_dir: Path) -> pd.DataFrame:
    tvl = _read(data_dir / f"{F_TVL}.csv", {"date", "tvl_usd"})
    tvl["date"] = pd.to_datetime(tvl["date"])
    return tvl.sort_values("date").reset_index(drop=True)


def has_erc_only(data_dir: Path) -> bool:
    return all((data_dir / f"{f}{ERC_ONLY_SUFFIX}.csv").exists()
               for f in (F_BUCKETS, F_TOPSHARE, F_TOP1, F_TOP01, F_DATES))


# -----------------------------------------------------------------------------
# quantities
# -----------------------------------------------------------------------------
def wealth_buckets(vb: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """{wallet_type: {bucket: mean over blocks of wallet_pct}} for CA and EOA."""
    out = {}
    for wt in ("CA", "EOA"):
        out[wt] = vb[vb["wallet_type"] == wt].groupby("bucket")["wallet_pct"].mean().to_dict()
    return out


def wealth_series(vb: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """Per snapshot: date, overall_wealth_usd (ALL), ca_holdings_usd (CA), sorted by date."""
    overall = vb[vb["wallet_type"] == "ALL"].groupby("block")["value_sum_usd"].sum().rename("overall_wealth_usd")
    ca = vb[vb["wallet_type"] == "CA"].groupby("block")["value_sum_usd"].sum().rename("ca_holdings_usd")
    ws = pd.concat([overall, ca], axis=1).reset_index().rename(columns={"index": "block"})
    ws["date"] = ws["block"].map(dates)
    if ws["date"].isna().any():
        raise SystemExit(f"[distribution_of_wealth_macros] no snapshot date for blocks "
                         f"{ws.loc[ws['date'].isna(), 'block'].tolist()}")
    return ws.sort_values("date").reset_index(drop=True)


def with_tvl(ws: pd.DataFrame, tvl: pd.DataFrame) -> pd.DataFrame:
    """Attach DeFiLlama TVL on the snapshot date (or the last earlier day)."""
    out = pd.merge_asof(ws, tvl.rename(columns={"tvl_usd": "defillama_eth_tvl_usd"}), on="date", direction="backward")
    if out["defillama_eth_tvl_usd"].isna().any():
        raise SystemExit("[distribution_of_wealth_macros] DeFiLlama TVL missing for some snapshot dates")
    return out


def wealth_share(ws: pd.DataFrame) -> Dict[str, float]:
    ca_share = 100.0 * ws["ca_holdings_usd"] / ws["overall_wealth_usd"]
    return {"ca_mean": ca_share.mean(), "eoa_mean": 100.0 - ca_share.mean()}


def ca_holder_share(vb: pd.DataFrame) -> float:
    """Mean over snapshots of the CA share of the accounts holding assets
    (CA / ALL wallet_count summed over all USD buckets)."""
    n = vb.groupby(["block", "wallet_type"])["wallet_count"].sum().unstack()
    return float((100.0 * n["CA"] / n["ALL"]).mean())


def wealth_composition(wt: pd.DataFrame) -> Dict[str, object]:
    """Raw values behind the wealth-composition macros (wt = with_tvl(wealth_series(...)))."""
    yr = wt["date"].dt.year
    tvl_peak = wt.loc[wt["defillama_eth_tvl_usd"].idxmax()]
    ov_peak = wt.loc[wt["overall_wealth_usd"].idxmax()]
    ca_peak = wt.loc[wt["ca_holdings_usd"].idxmax()]
    last = wt.iloc[-1]
    ratio = wt["overall_wealth_usd"] / wt["defillama_eth_tvl_usd"]
    return {
        "peak_2021": wt.loc[yr == 2021, "overall_wealth_usd"].max(),
        "peak_2025": wt.loc[yr == 2025, "overall_wealth_usd"].max(),
        "tvl_peak": tvl_peak["defillama_eth_tvl_usd"], "tvl_peak_date": tvl_peak["date"],
        "tvl_end": last["defillama_eth_tvl_usd"],
        "overall_peak": ov_peak["overall_wealth_usd"], "overall_peak_date": ov_peak["date"],
        "overall_end": last["overall_wealth_usd"],
        "ca_peak": ca_peak["ca_holdings_usd"],
        "ca_over_tvl_at_tvl_peak": tvl_peak["ca_holdings_usd"] / tvl_peak["defillama_eth_tvl_usd"],
        "wealth_over_tvl_median": ratio.median(), "wealth_over_tvl_min": ratio.min(), "wealth_over_tvl_max": ratio.max(),
    }


def topshare_yearly(th: pd.DataFrame, dates: pd.Series, wallet_type: str = "ALL") -> pd.DataFrame:
    """Yearly top-holder concentration table (accounts above 1 USD): one row per
    calendar year with the number of snapshots, mean accounts considered, and
    mean / min / max of the top-1 % share plus mean top-5 % and top-10 % shares."""
    d = th[th["wallet_type"].astype(str).str.strip().str.upper() == wallet_type.upper()].copy()
    d["date"] = d["block_number"].map(dates)
    d = d.dropna(subset=["date"])
    if d.empty:
        raise SystemExit(f"[distribution_of_wealth_macros] no {wallet_type} rows with a snapshot date")
    d["year"] = d["date"].dt.year
    y = (d.groupby("year")
         .agg(n_snapshots=("block_number", "count"), avg_wallets=("wallets_considered", "mean"),
              top1_avg=("top_1_share_pct", "mean"), top1_min=("top_1_share_pct", "min"),
              top1_max=("top_1_share_pct", "max"), top5_avg=("top_5_share_pct", "mean"),
              top10_avg=("top_10_share_pct", "mean"))
         .reset_index().sort_values("year"))
    y["avg_wallets_m"] = y["avg_wallets"] / 1e6
    return y.reset_index(drop=True)


def yearly_table_text(y: pd.DataFrame, wallet_type: str, latex: bool = False) -> str:
    show = pd.DataFrame({
        "Year": y["year"].astype(int),
        "Avg accounts (M)": y["avg_wallets_m"].map(lambda x: f"{x:.3f}"),
        "Top 1% (avg)": y["top1_avg"].map(lambda x: f"{x:.2f}"),
        "Top 1% (min--max)": [f"{a:.2f}--{b:.2f}" for a, b in zip(y["top1_min"], y["top1_max"])],
        "Top 5% (avg)": y["top5_avg"].map(lambda x: f"{x:.2f}"),
        "Top 10% (avg)": y["top10_avg"].map(lambda x: f"{x:.2f}"),
        "N snapshots": y["n_snapshots"].astype(int),
    })
    lines = [f"Yearly top-holder concentration ({wallet_type} accounts above 1 USD):", show.to_string(index=False)]
    if latex:
        lines.append("\nLaTeX rows (Year & accounts M & top1 avg & top1 min--max & top5 & top10):")
        for _, r in y.iterrows():
            lines.append(f"{int(r['year'])} & {r['avg_wallets_m']:.3f} & {r['top1_avg']:.2f} & "
                         f"{r['top1_min']:.2f}--{r['top1_max']:.2f} & {r['top5_avg']:.2f} & {r['top10_avg']:.2f} \\\\")
    return "\n".join(lines)


def top_concentration(th: pd.DataFrame, dates: pd.Series) -> Dict[str, float]:
    y = topshare_yearly(th, dates, "ALL").set_index("year")
    d = th[th["wallet_type"] == "ALL"]
    for year in (2020, 2025):
        if year not in y.index:
            raise SystemExit(f"[distribution_of_wealth_macros] no {year} snapshots in {F_TOPSHARE}.csv")
    return {"top1_2020": y.loc[2020, "top1_avg"], "top1_2025": y.loc[2025, "top1_avg"],
            "top5_min": d["top_5_share_pct"].min(), "top10_min": d["top_10_share_pct"].min()}


def top1_breakdown(br1: pd.DataFrame, br01: pd.DataFrame) -> Dict[str, float]:
    ca, eoa = br1[br1["wallet_type"] == "CA"], br1[br1["wallet_type"] == "EOA"]
    ca01 = br01[br01["wallet_type"] == "CA"]
    return {
        "top1_ca_wallets": ca["n_wallets_pct_of_top1"].mean(), "top1_eoa_wallets": eoa["n_wallets_pct_of_top1"].mean(),
        "top1_ca_value": ca["value_pct_of_top1"].mean(), "top1_eoa_value": eoa["value_pct_of_top1"].mean(),
        "top01_ca_wallets": ca01["n_wallets_pct_of_top1"].mean(), "top01_ca_value": ca01["value_pct_of_top1"].mean(),
    }


# -----------------------------------------------------------------------------
# emitters (one block per data family; `suffix` gives the ErcOnly variants)
# -----------------------------------------------------------------------------
def emit_wealth_buckets(emit, wb: Dict[str, Dict[str, float]], suffix: str = "") -> None:
    for wt in ("CA", "EOA"):
        emit.comment(f"% --- {wt} ---")
        for macro_suffix, bucket in BUCKET_MAP:
            emit.macro(f"WealthBucket{wt}{macro_suffix}{suffix}", fmt_num2(wb[wt][bucket]))
        emit.blank()


def emit_wealth_share(emit, ws: Dict[str, float], suffix: str = "") -> None:
    emit.comment("% --- CA share stats ---")
    emit.macro(f"WealthShareCAMeanP{suffix}", fmt_num2(ws["ca_mean"]))
    emit.macro(f"CAShareHoldersMeanPct{suffix}", fmt_dec(ws["ca_holders_mean"], 2))
    emit.blank()
    emit.comment("% --- EOA share stats ---")
    emit.macro(f"WealthShareEOAMeanP{suffix}", fmt_num2(ws["eoa_mean"]))
    emit.blank()


def emit_composition(emit, c: Dict[str, object], suffix: str = "") -> None:
    emit.comment("% --- Wealth composition: peak total-value snapshots (per calendar year) ---")
    emit.macro(f"WealthPeakTwentyone{suffix}", usd_t(c["peak_2021"]))
    emit.macro(f"WealthPeakTwentyfiveB{suffix}", usd_t(c["peak_2025"]))
    emit.blank()
    emit.comment("% --- Wealth vs DeFiLlama TVL (Appendix B figure): peaks, endpoints, ratios ---")
    emit.macro(f"EthTvlPeak{suffix}", usd_b(c["tvl_peak"]))
    emit.macro(f"EthTvlPeakDate{suffix}", c["tvl_peak_date"].strftime("%B %Y"))
    emit.macro(f"EthTvlEnd{suffix}", usd_b(c["tvl_end"]))
    emit.macro(f"OverallWealthPeak{suffix}", usd_t(c["overall_peak"]))
    emit.macro(f"OverallWealthPeakDate{suffix}", c["overall_peak_date"].strftime("%B %Y"))
    emit.macro(f"OverallWealthEnd{suffix}", usd_t(c["overall_end"]))
    emit.macro(f"CaHoldingsPeak{suffix}", usd_b(c["ca_peak"]))
    emit.macro(f"CaOverTvlPeakRatio{suffix}", fmt_dec(c["ca_over_tvl_at_tvl_peak"], 1))
    emit.macro(f"WealthOverTvlMedianRatio{suffix}", fmt_dec(c["wealth_over_tvl_median"], 0))
    emit.macro(f"WealthOverTvlMinRatio{suffix}", fmt_dec(c["wealth_over_tvl_min"], 1))
    emit.macro(f"WealthOverTvlMaxRatio{suffix}", fmt_dec(c["wealth_over_tvl_max"], 1))
    emit.blank()


def emit_top_concentration(emit, t: Dict[str, float], suffix: str = "") -> None:
    emit.comment("% --- Top wallet wealth concentration (top_held_percent_gr_1.csv, ALL accounts > 1 USD) ---")
    emit.macro(f"TopOneWealthTwenty{suffix}", fmt_dec(t["top1_2020"], 2))
    emit.macro(f"TopOneWealthTwentyfive{suffix}", fmt_dec(t["top1_2025"], 2))
    emit.macro(f"TopFiveTenWealthMin{suffix}", fmt_dec(t["top5_min"], 1))
    emit.macro(f"TopTenWealthMin{suffix}", fmt_dec(t["top10_min"], 1))
    emit.blank()


def emit_top1_breakdown(emit, b: Dict[str, float], suffix: str = "") -> None:
    emit.comment("% --- Top-1% CA/EOA wealth concentration breakdown ---")
    emit.comment("% Top 1%")
    emit.macro(f"TopOneCAPctWallets{suffix}", fmt_dec(b["top1_ca_wallets"], 2))
    emit.macro(f"TopOneEOAPctWallets{suffix}", fmt_dec(b["top1_eoa_wallets"], 2))
    emit.macro(f"TopOneCAPctValue{suffix}", fmt_dec(b["top1_ca_value"], 2))
    emit.macro(f"TopOneEOAPctValue{suffix}", fmt_dec(b["top1_eoa_value"], 2))
    emit.blank()
    emit.comment("% Top 0.1%")
    emit.macro(f"TopPointOneCAPctWallets{suffix}", fmt_dec(b["top01_ca_wallets"], 2))
    emit.macro(f"TopPointOneCAPctValue{suffix}", fmt_dec(b["top01_ca_value"], 2))
    emit.blank()


def _all_quantities(data_dir: Path, tvl: pd.DataFrame, suffix: str = "") -> Dict[str, object]:
    dates = load_block_dates(data_dir, suffix)
    vb = load_buckets(data_dir, suffix)
    th = load_topshare(data_dir, suffix)
    br1, br01 = load_top_breakdowns(data_dir, suffix)
    wt = with_tvl(wealth_series(vb, dates), tvl)
    share = {**wealth_share(wt), "ca_holders_mean": ca_holder_share(vb)}
    return {"buckets": wealth_buckets(vb), "share": share, "composition": wealth_composition(wt),
            "top": top_concentration(th, dates), "breakdown": top1_breakdown(br1, br01)}


def emit_distribution_of_wealth_macros(emit, data_dir: Optional[Path] = None) -> None:
    """Write all wealth-distribution macros through `emit`, which must provide
    .comment(text), .macro(name, value) and .blank() (MacroEmitter in
    generate_all_macros.py, or _PrintEmitter below)."""
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    tvl = load_tvl(data_dir)
    new = _all_quantities(data_dir, tvl)

    emit.comment("% --- Wealth buckets: average $-bucket allocation by wallet type "
                 "(distribution_of_wealth/src/distribution_of_wealth_macros.py) ---")
    emit_wealth_buckets(emit, new["buckets"])
    emit.comment("% --- Wealth share: CA vs EOA fraction of total value over time ---")
    emit_wealth_share(emit, new["share"])
    emit_composition(emit, new["composition"])
    emit_top_concentration(emit, new["top"])
    emit_top1_breakdown(emit, new["breakdown"])

    if not has_erc_only(data_dir):
        return
    old = _all_quantities(data_dir, tvl, ERC_ONLY_SUFFIX)
    emit.comment("% --- Same quantities on the ERC-20-only reconstruction (April 2026) ---")
    emit_wealth_buckets(emit, old["buckets"], "ErcOnly")
    emit_wealth_share(emit, old["share"], "ErcOnly")
    emit_composition(emit, old["composition"], "ErcOnly")
    emit_top_concentration(emit, old["top"], "ErcOnly")
    emit_top1_breakdown(emit, old["breakdown"], "ErcOnly")
    emit.comment("% --- Changes ETH-inclusive minus ERC-20-only ---")
    emit.macro("WealthShareCAMeanDeltaPP", fmt_dec(new["share"]["ca_mean"] - old["share"]["ca_mean"], 2))
    emit.macro("OverallWealthEndDeltaPct",
               fmt_dec(100.0 * (new["composition"]["overall_end"] / old["composition"]["overall_end"] - 1.0), 1))
    emit.macro("TopOneWealthTwentyfiveDeltaPP", fmt_dec(new["top"]["top1_2025"] - old["top"]["top1_2025"], 2))
    emit.macro("TopOneCAPctValueDeltaPP", fmt_dec(new["breakdown"]["top1_ca_value"] - old["breakdown"]["top1_ca_value"], 2))
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
    ap = argparse.ArgumentParser(description="Print the wealth-distribution LaTeX macros or the yearly top-share table.")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--yearly-table", action="store_true", help="Print the yearly top-share table instead of the macros.")
    ap.add_argument("--wallet-type", default="ALL", help="Wallet type for --yearly-table (ALL, EOA, CA).")
    ap.add_argument("--latex", action="store_true", help="With --yearly-table: also print LaTeX tabular rows.")
    ap.add_argument("--erc20-only", action="store_true", help="With --yearly-table: use the *_erc20_only files.")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    if args.yearly_table:
        suffix = ERC_ONLY_SUFFIX if args.erc20_only else ""
        y = topshare_yearly(load_topshare(data_dir, suffix), load_block_dates(data_dir, suffix), args.wallet_type)
        print(yearly_table_text(y, args.wallet_type.upper(), args.latex))
        return
    e = _PrintEmitter()
    emit_distribution_of_wealth_macros(e, data_dir)
    print("\n".join(e.lines))


if __name__ == "__main__":
    main()
