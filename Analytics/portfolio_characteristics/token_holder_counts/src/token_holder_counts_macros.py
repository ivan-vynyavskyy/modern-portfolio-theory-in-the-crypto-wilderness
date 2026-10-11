#!/usr/bin/env python3
"""
token_holder_counts_macros.py

Description:
    Single owner of every LaTeX macro derived from the yearly token-holder
    CSVs in ../data (produced by analyze_holders_distribution_by_year.py).
    Called by Analytics/generate_all_macros.py; can also be run standalone
    to print the macros.

    The statistics are per snapshot row (one row per token and monthly
    snapshot, native ETH excluded), aggregated by calendar year.

Macros (names and formatting unchanged from the original macro.tex; note
the legacy form WITHOUT a trailing slash):
    \\HoldersAvgYtwenty, \\HoldersMedYtwenty          mean / median holders per
    \\HoldersAvgYtwentyfive, \\HoldersMedYtwentyfive  token in 2020 and 2025
    \\HoldersPctGtTenKYtwenty, \\HoldersPctGtTenKYtwentyfive
                                                    % of token snapshots with
                                                    more than 10,000 holders
    \\HoldersPctOneKTenKYtwenty, \\HoldersPctOneKTenKYtwentyfour,
    \\HoldersPctOneKTenKYtwentyfive                  1,000-10,000 holders
    \\HoldersPctHundredOneKYtwenty, \\HoldersPctHundredOneKYtwentyfive
                                                    100-1,000 holders

    Comparison with the ERC-20-only reconstruction (emitted only when the
    *_erc20_only.csv files are present): the same macros with suffix
    ErcOnly, e.g. \\HoldersAvgYtwentyErcOnly.

Input (in --data-dir):
    holders_overall_yearly_summary.csv   (year, n_obs, n_tokens, avg_holders, median_holders)
    holders_overall_yearly_buckets.csv   (year, pct_0–100, pct_100–1K, pct_1K–10K, pct_>10K, ...)
    optional: the same two files with suffix _erc20_only

Usage:
    python token_holder_counts_macros.py                 # prints the macros
    python token_holder_counts_macros.py --data-dir ../data
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
F_SUMMARY = "holders_overall_yearly_summary"
F_BUCKETS = "holders_overall_yearly_buckets"
ERC_ONLY_SUFFIX = "_erc20_only"

YEARS = {2020: "Ytwenty", 2024: "Ytwentyfour", 2025: "Ytwentyfive"}


# -----------------------------------------------------------------------------
# helpers (same output as generate_all_macros.fmt_num_int / fmt_pct)
# -----------------------------------------------------------------------------
def fmt_num_int(v) -> str:
    return rf"\num{{{int(round(float(v)))}}}"


def fmt_pct(v, dp: int = 1) -> str:
    return rf"\num{{{round(float(v), dp):.{dp}f}}}\%"


def load(data_dir: Path, suffix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(data_dir / f"{F_SUMMARY}{suffix}.csv")
    buckets = pd.read_csv(data_dir / f"{F_BUCKETS}{suffix}.csv")
    for df, need, name in ((summary, {"year", "avg_holders", "median_holders"}, F_SUMMARY),
                           (buckets, {"year", "pct_100–1K", "pct_1K–10K", "pct_>10K"}, F_BUCKETS)):
        missing = need - set(df.columns)
        if missing:
            raise SystemExit(f"[token_holder_counts_macros] {name}{suffix}.csv lacks columns {sorted(missing)}")
    for y in YEARS:
        if y not in set(summary["year"]) or y not in set(buckets["year"]):
            raise SystemExit(f"[token_holder_counts_macros] year {y} missing in {F_SUMMARY}{suffix}.csv / {F_BUCKETS}{suffix}.csv")
    return summary.set_index("year"), buckets.set_index("year")


# -----------------------------------------------------------------------------
# emitter
# -----------------------------------------------------------------------------
def emit_holders_block(emit, summary: pd.DataFrame, buckets: pd.DataFrame, suffix: str = "") -> None:
    s20, s25 = summary.loc[2020], summary.loc[2025]
    b20, b24, b25 = buckets.loc[2020], buckets.loc[2024], buckets.loc[2025]

    emit.macro(f"HoldersAvgYtwenty{suffix}", fmt_num_int(s20["avg_holders"]), slash=False)
    emit.macro(f"HoldersMedYtwenty{suffix}", fmt_num_int(s20["median_holders"]), slash=False)
    emit.blank()
    emit.macro(f"HoldersAvgYtwentyfive{suffix}", fmt_num_int(s25["avg_holders"]), slash=False)
    emit.macro(f"HoldersMedYtwentyfive{suffix}", fmt_num_int(s25["median_holders"]), slash=False)
    emit.blank()
    emit.macro(f"HoldersPctGtTenKYtwenty{suffix}", fmt_pct(b20["pct_>10K"]), slash=False)
    emit.macro(f"HoldersPctGtTenKYtwentyfive{suffix}", fmt_pct(b25["pct_>10K"]), slash=False)
    emit.blank()
    emit.macro(f"HoldersPctOneKTenKYtwenty{suffix}", fmt_pct(b20["pct_1K–10K"]), slash=False)
    emit.macro(f"HoldersPctOneKTenKYtwentyfour{suffix}", fmt_pct(b24["pct_1K–10K"]), slash=False)
    emit.macro(f"HoldersPctOneKTenKYtwentyfive{suffix}", fmt_pct(b25["pct_1K–10K"]), slash=False)
    emit.blank()
    emit.macro(f"HoldersPctHundredOneKYtwenty{suffix}", fmt_pct(b20["pct_100–1K"]), slash=False)
    emit.macro(f"HoldersPctHundredOneKYtwentyfive{suffix}", fmt_pct(b25["pct_100–1K"]), slash=False)
    emit.blank()


def emit_token_holder_counts_macros(emit, data_dir: Optional[Path] = None) -> None:
    """Write all token-holder-count macros through `emit`, which must provide
    .comment(text), .macro(name, value, slash) and .blank() (MacroEmitter in
    generate_all_macros.py, or _PrintEmitter below)."""
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    summary, buckets = load(data_dir)
    emit.comment("% --- Token holder counts: distribution per year (2020 vs 2025; "
                 "token_holder_counts/src/token_holder_counts_macros.py) ---")
    emit_holders_block(emit, summary, buckets)

    old_s = data_dir / f"{F_SUMMARY}{ERC_ONLY_SUFFIX}.csv"
    old_b = data_dir / f"{F_BUCKETS}{ERC_ONLY_SUFFIX}.csv"
    if old_s.exists() and old_b.exists():
        emit.comment("% --- Same quantities on the ERC-20-only reconstruction (April 2026) ---")
        emit_holders_block(emit, *load(data_dir, ERC_ONLY_SUFFIX), suffix="ErcOnly")


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
    ap = argparse.ArgumentParser(description="Print the token-holder-count LaTeX macros.")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = ap.parse_args()
    e = _PrintEmitter()
    emit_token_holder_counts_macros(e, Path(args.data_dir))
    print("\n".join(e.lines))


if __name__ == "__main__":
    main()
