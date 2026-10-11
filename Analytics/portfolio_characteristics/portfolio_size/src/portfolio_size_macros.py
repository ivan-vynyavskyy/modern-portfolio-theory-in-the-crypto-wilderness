#!/usr/bin/env python3
"""
portfolio_size_macros.py

Description:
    Single owner of every LaTeX macro derived from the portfolio-size CSVs
    in ../data (produced by portfolio_size_by_wealth_bucket.py). Called by
    Analytics/generate_all_macros.py; can also be run standalone to print
    the macros.

    All values are shares of accounts in percent, averaged over the monthly
    snapshots. "Token" in the macro names is historical: since the
    ETH-inclusive reconstruction the counts are numbers of distinct assets,
    native ETH included (ETH and WETH folded into one asset).

Macros:
    Referenced in the paper (definition unchanged: per-block share with the
    size-0 accounts in the denominator, then mean across blocks):
        \\PortSizeOneTokenPct/        accounts holding exactly 1 asset
        \\PortSizeTwoTokenPct/        exactly 2
        \\PortSizeThreeFourTokenPct/  3 or 4
        \\PortSizeFivePlusPct/        5 or more
        \\PortSizeTwoOfMultiPct/      2-asset accounts as % of multi-asset accounts

    Added with the ETH-inclusive data, same coarse definition:
        \\PortSizeOneTokenEOAPct/, \\PortSizeOneTokenCAPct/
            single-asset share within EOAs / within contract accounts

    Added, per wealth bucket, size-0 accounts dropped (what the stacked
    figure shows):
        \\PortSizeOneToken<Bucket>Pct/  with <Bucket> one of
            ZeroOne        $0-1
            OneHundred     $1-100
            HundredOneK    $100-1,000
            OneKTenK       $1,000-10,000
            TenKHundredK   $10,000-100,000
            AboveHundredK  >$100,000

    Comparison with the ERC-20-only reconstruction (emitted only when the
    *_erc20_only.csv files are present; used in the thesis):
        \\PortSize...ErcOnly            the macros above recomputed on the
                                       token-only data (same definitions)
        \\PortSizeAccountsPerSnapshotM/, \\PortSizeAccountsPerSnapshotMErcOnly/
            accounts per monthly snapshot, millions, mean over blocks
        \\PortSizeAccountsRatioEthVsErcOnly/  ratio of the two
        \\PortSizeOneTokenPctDeltaPP/, \\PortSizeTwoTokenPctDeltaPP/
            change in the single- and two-asset share, ETH-inclusive minus
            ERC-20-only, in percentage points

Input (in --data-dir):
    portfolio_size_hist_by_wealth.csv
    portfolio_size_hist_by_wealth_wallet_type.csv
    optional: the same two files with suffix _erc20_only, and
    portfolio_size_stats_by_wealth[_erc20_only].csv for the account counts

Usage:
    python portfolio_size_macros.py                 # prints the macros
    python portfolio_size_macros.py --data-dir ../data
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

HIST_ALL = "portfolio_size_hist_by_wealth.csv"
HIST_BY_TYPE = "portfolio_size_hist_by_wealth_wallet_type.csv"
STATS_ALL = "portfolio_size_stats_by_wealth.csv"
ERC_ONLY_SUFFIX = "_erc20_only"

COARSE_CATS = ["0", "1", "2", "3-4", "5+"]

# wealth bucket label in the CSV -> macro name fragment
WEALTH_TAGS: "OrderedDict[str, str]" = OrderedDict([
    ("0–1", "ZeroOne"),
    ("1–100", "OneHundred"),
    ("100–1,000", "HundredOneK"),
    ("1,000–10,000", "OneKTenK"),
    ("10,000–100,000", "TenKHundredK"),
    (">100,000", "AboveHundredK"),
])


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def fmt_dec(v, dp: int) -> str:
    return f"{round(float(v), dp):.{dp}f}"


def coarse_cat(b: str) -> str:
    """Map a size bucket label to the coarse categories used in the paper."""
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
    raise ValueError(f"unknown size bucket {b!r}")


def read_hist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"size_bucket": str, "wealth_bucket": str})
    missing = {"block", "size_bucket", "wallet_count"} - set(df.columns)
    if missing:
        raise SystemExit(f"[portfolio_size_macros] {path} lacks columns {sorted(missing)}")
    return df


def coarse_counts(hist: pd.DataFrame, keys: Iterable[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Account counts per (keys x coarse category) and the row totals.
    Size-0 accounts stay in the total, matching the original macro
    definition (the ETH-inclusive data simply has none)."""
    h = hist.copy()
    h["cat"] = h["size_bucket"].map(coarse_cat)
    pb = (h.groupby([*keys, "cat"])["wallet_count"].sum()
            .unstack(fill_value=0)
            .reindex(columns=COARSE_CATS, fill_value=0))
    return pb, pb.sum(axis=1)


def single_share_by_wealth(hist: pd.DataFrame) -> pd.Series:
    """Single-asset share within each wealth bucket, size-0 dropped, per
    block, then mean across blocks (the figure's definition)."""
    h = hist[hist["size_bucket"] != "0"].copy()
    denom = h.groupby(["block", "wealth_bucket"])["wallet_count"].transform("sum")
    h["pct"] = 100.0 * h["wallet_count"] / denom
    one = h[h["size_bucket"] == "1"].groupby(["block", "wealth_bucket"])["pct"].sum()
    return one.groupby(level="wealth_bucket").mean()


# -----------------------------------------------------------------------------
# emitter
# -----------------------------------------------------------------------------
def compute_shares(hist: pd.DataFrame, hist_wt: pd.DataFrame) -> dict:
    """All portfolio-size quantities for one data set, as plain floats."""
    out: dict = {}
    pb, total = coarse_counts(hist, ["block"])
    for c in ("1", "2", "3-4", "5+"):
        out[c] = (pb[c] / total * 100.0).mean()
    multi = pb["2"] + pb["3-4"] + pb["5+"]
    out["two_of_multi"] = (pb["2"] / multi * 100.0).mean()

    pbt, total_t = coarse_counts(hist_wt, ["block", "wallet_type"])
    one_by_type = (pbt["1"] / total_t * 100.0).groupby(level="wallet_type").mean()
    for wt in ("EOA", "CA"):
        if wt not in one_by_type.index:
            raise SystemExit(f"[portfolio_size_macros] wallet_type {wt!r} missing in the by-type histogram")
        out[f"one_{wt}"] = one_by_type[wt]

    one_by_wealth = single_share_by_wealth(hist)
    for bucket, tag in WEALTH_TAGS.items():
        if bucket not in one_by_wealth.index:
            raise SystemExit(f"[portfolio_size_macros] wealth bucket {bucket!r} missing in the histogram")
        out[f"one_{tag}"] = one_by_wealth[bucket]
    return out


def emit_shares(emit, v: dict, suffix: str = "") -> None:
    emit.macro(f"PortSizeOneTokenPct{suffix}", fmt_dec(v["1"], 2))
    emit.macro(f"PortSizeTwoTokenPct{suffix}", fmt_dec(v["2"], 2))
    emit.macro(f"PortSizeThreeFourTokenPct{suffix}", fmt_dec(v["3-4"], 2))
    emit.macro(f"PortSizeFivePlusPct{suffix}", fmt_dec(v["5+"], 2))
    emit.macro(f"PortSizeTwoOfMultiPct{suffix}", fmt_dec(v["two_of_multi"], 1))
    for wt in ("EOA", "CA"):
        emit.macro(f"PortSizeOneToken{wt}Pct{suffix}", fmt_dec(v[f"one_{wt}"], 2))
    for tag in WEALTH_TAGS.values():
        emit.macro(f"PortSizeOneToken{tag}Pct{suffix}", fmt_dec(v[f"one_{tag}"], 1))


def accounts_per_snapshot(stats_path: Path) -> float:
    st = pd.read_csv(stats_path)
    return float(st.groupby("block")["n_wallets"].sum().mean())


def emit_portfolio_size_macros(emit, data_dir: Optional[Path] = None) -> None:
    """Write all portfolio-size macros through `emit`, which must provide
    .comment(text), .macro(name, value) and .blank() (MacroEmitter in
    generate_all_macros.py, or _PrintEmitter below)."""
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    hist = read_hist(data_dir / HIST_ALL)
    hist_wt = read_hist(data_dir / HIST_BY_TYPE)
    n_blocks = hist["block"].nunique()

    emit.comment(f"% --- Portfolio size distribution (avg across {n_blocks} monthly snapshots; "
                 "portfolio_size/src/portfolio_size_macros.py) ---")
    new = compute_shares(hist, hist_wt)
    emit_shares(emit, new)
    emit.blank()

    # optional comparison with the ERC-20-only reconstruction
    old_hist_p = data_dir / f"portfolio_size_hist_by_wealth{ERC_ONLY_SUFFIX}.csv"
    old_wt_p = data_dir / f"portfolio_size_hist_by_wealth_wallet_type{ERC_ONLY_SUFFIX}.csv"
    if not (old_hist_p.exists() and old_wt_p.exists()):
        return
    emit.comment("% --- Same quantities on the ERC-20-only reconstruction (April 2026), "
                 "plus the change when native ETH is included ---")
    old = compute_shares(read_hist(old_hist_p), read_hist(old_wt_p))
    emit_shares(emit, old, "ErcOnly")
    emit.macro("PortSizeOneTokenPctDeltaPP", fmt_dec(new["1"] - old["1"], 1))
    emit.macro("PortSizeTwoTokenPctDeltaPP", fmt_dec(new["2"] - old["2"], 1))

    stats_new_p = data_dir / STATS_ALL
    stats_old_p = data_dir / f"portfolio_size_stats_by_wealth{ERC_ONLY_SUFFIX}.csv"
    if stats_new_p.exists() and stats_old_p.exists():
        acc_new = accounts_per_snapshot(stats_new_p)
        acc_old = accounts_per_snapshot(stats_old_p)
        emit.macro("PortSizeAccountsPerSnapshotM", fmt_dec(acc_new / 1e6, 1))
        emit.macro("PortSizeAccountsPerSnapshotMErcOnly", fmt_dec(acc_old / 1e6, 1))
        emit.macro("PortSizeAccountsRatioEthVsErcOnly", fmt_dec(acc_new / acc_old, 2))
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
    ap = argparse.ArgumentParser(description="Print the portfolio-size LaTeX macros.")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = ap.parse_args()
    e = _PrintEmitter()
    emit_portfolio_size_macros(e, Path(args.data_dir))
    print("\n".join(e.lines))


if __name__ == "__main__":
    main()
