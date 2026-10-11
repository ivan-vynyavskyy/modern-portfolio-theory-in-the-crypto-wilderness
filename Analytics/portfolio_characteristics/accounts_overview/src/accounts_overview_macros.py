#!/usr/bin/env python3
"""
accounts_overview_macros.py

Description:
    Single owner of every LaTeX macro derived from the account-count series
    in ../data (output_accounts_eoa_vs_ca.csv, produced by
    count_accounts_eoa_vs_ca.py). Called by Analytics/generate_all_macros.py;
    can also be run standalone to print the macros.

    The series is the cumulative number of accounts of the study's universe
    (addresses that ever moved a whitelisted ERC-20 token; the zero address
    excluded) observed up to each monthly snapshot, split into externally
    owned accounts (EOA) and contract accounts (CA).

Macros (names and formatting unchanged from the original macro.tex):
    \\TotalAccountsRaw/        accounts at the last snapshot, plain integer
    \\TotalAccountsRounded/    the same in millions (TeX arithmetic on the raw value)
    \\TotalAccountsEnd/        the same as \\num{}
    \\EOAAccountsEnd/, \\CAAccountsEnd/   EOA / CA accounts at the last snapshot
    \\EOACAGR/, \\CACAGR/      compound annual growth rate over the whole series
    \\EOAAvgIncreaseYear/, \\CAAvgIncreaseYear/
                              mean absolute increase per year (total increase /
                              elapsed years)
    \\CAShareEndPct/          CA share of all accounts at the last snapshot

Input (in --data-dir):
    output_accounts_eoa_vs_ca.csv   (block, date, eoa_accounts, ca_accounts)

Usage:
    python accounts_overview_macros.py                 # prints the macros
    python accounts_overview_macros.py --data-dir ../data
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
F_ACCOUNTS = "output_accounts_eoa_vs_ca.csv"


# -----------------------------------------------------------------------------
# helpers (same output as generate_all_macros.fmt_num_int / fmt_si_pct)
# -----------------------------------------------------------------------------
def fmt_num_int(v) -> str:
    return rf"\num{{{int(round(float(v)))}}}"


def fmt_si_pct(v, dp: int = 2) -> str:
    return rf"\SI{{{round(float(v), dp):.{dp}f}}}{{\percent}}"


def load(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / F_ACCOUNTS)
    missing = {"block", "date", "eoa_accounts", "ca_accounts"} - set(df.columns)
    if missing:
        raise SystemExit(f"[accounts_overview_macros] {F_ACCOUNTS} lacks columns {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("block").reset_index(drop=True)
    if len(df) < 2:
        raise SystemExit(f"[accounts_overview_macros] {F_ACCOUNTS} needs at least two snapshots")
    return df


# -----------------------------------------------------------------------------
# quantities
# -----------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> dict:
    first, last = df.iloc[0], df.iloc[-1]
    years = (last["date"] - first["date"]).days / 365.25
    eoa_end, ca_end = int(last["eoa_accounts"]), int(last["ca_accounts"])
    return {
        "total_end": eoa_end + ca_end,
        "eoa_end": eoa_end,
        "ca_end": ca_end,
        "years": years,
        "eoa_cagr": ((last["eoa_accounts"] / first["eoa_accounts"]) ** (1 / years) - 1) * 100,
        "ca_cagr": ((last["ca_accounts"] / first["ca_accounts"]) ** (1 / years) - 1) * 100,
        "eoa_avg_inc": (last["eoa_accounts"] - first["eoa_accounts"]) / years,
        "ca_avg_inc": (last["ca_accounts"] - first["ca_accounts"]) / years,
        "ca_share_end": 100.0 * ca_end / (eoa_end + ca_end),
        "n_snapshots": len(df),
        "first_date": first["date"].date().isoformat(),
        "last_date": last["date"].date().isoformat(),
    }


# -----------------------------------------------------------------------------
# emitter
# -----------------------------------------------------------------------------
def emit_accounts_overview_macros(emit, data_dir: Optional[Path] = None) -> None:
    """Write all account-count macros through `emit`, which must provide
    .comment(text), .macro(name, value), .macro_raw(line) and .blank()
    (MacroEmitter in generate_all_macros.py, or _PrintEmitter below)."""
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    v = compute(load(data_dir))

    emit.comment("% --- Total accounts at end-of-study (millions; "
                 "accounts_overview/src/accounts_overview_macros.py) ---")
    emit.macro("TotalAccountsRaw", str(v["total_end"]))
    emit.macro_raw(r"\def\TotalAccountsRounded/{\the\numexpr\TotalAccountsRaw//1000000\relax\,M}")
    emit.blank()

    emit.comment(f"% --- Account counts: EOA vs CA growth (CAGR + ratios; {v['n_snapshots']} snapshots, "
                 f"{v['first_date']} to {v['last_date']}) ---")
    emit.macro("EOAAccountsEnd", fmt_num_int(v["eoa_end"]))
    emit.macro("CAAccountsEnd", fmt_num_int(v["ca_end"]))
    emit.blank()
    emit.macro("EOACAGR", fmt_si_pct(v["eoa_cagr"], 2))
    emit.macro("EOAAvgIncreaseYear", fmt_num_int(v["eoa_avg_inc"]))
    emit.blank()
    emit.macro("CACAGR", fmt_si_pct(v["ca_cagr"], 2))
    emit.macro("CAAvgIncreaseYear", fmt_num_int(v["ca_avg_inc"]))
    emit.blank()
    emit.macro_raw(r"\def\TotalAccountsEnd/{\num{\TotalAccountsRaw/}} % EOA + CA")
    emit.macro("CAShareEndPct", fmt_si_pct(v["ca_share_end"], 2))
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

    def macro_raw(self, line: str) -> None:
        self.lines.append(line.split(" %")[0].rstrip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Print the account-count LaTeX macros.")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = ap.parse_args()
    e = _PrintEmitter()
    emit_accounts_overview_macros(e, Path(args.data_dir))
    print("\n".join(e.lines))


if __name__ == "__main__":
    main()
