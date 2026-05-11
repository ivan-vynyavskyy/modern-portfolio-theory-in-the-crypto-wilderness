#!/usr/bin/env python3
"""
0_5_update_status_with_erc20_verification.py

Purpose
-------
Update an ETH token status CSV by flipping is_erc_20 to true only when both
static and runtime ERC-20 verification succeed.

What it does
------------
- Loads the status CSV and the proxy resolver TSV/CSV.
- Computes a per-address verdict requiring verified_static == 'Y' and verified_runtime == 'Y'.
- Updates is_erc_20 only for rows originally marked false, then writes a CSV report.

Inputs
------
- --status-csv (CSV; columns: symbol, name, contract_address, is_erc_20, creation_block)
- --resolver-tsv (TSV/CSV; columns: address, verified_static, verified_runtime, ...)

Outputs
-------
- --output (CSV; same columns as status CSV; updated is_erc_20 values)

CLI
---
Examples:
  $ python 0_5_update_status_with_erc20_verification.py --status-csv ../data/status_eth_tokens.csv \
      --resolver-tsv ../data/erc20_tokens_proxy_resolved.tsv --output ../data/status_eth_tokens.csv
  $ python 0_5_update_status_with_erc20_verification.py --status-csv status.csv --resolver-tsv results.tsv

Notes
-----
- Only rows with is_erc_20 == "false" are eligible for flipping to true.
- Resolver duplicates keep the last occurrence per address.
- Address matching is done on normalized lowercase 0x-addresses.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# Ensure helpers/ is importable when running from src/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.ethereum_codec_utils import normalize_hex_address
from utils.log_utils import log

# ----------------------------- Constants -----------------------------
DEFAULT_OUTPUT_PATH = "../data/status_eth_tokens.csv"
STATUS_REQUIRED_COLUMNS = {
    "symbol",
    "name",
    "contract_address",
    "is_erc_20",
    "creation_block",
}
RESOLVER_REQUIRED_COLUMNS = {"address", "verified_static", "verified_runtime"}

# ----------------------------- Helpers -----------------------------

def _load_resolver_verdict(resolver_path: Path) -> Dict[str, bool]:
    """Return a map of address -> (verified_static and verified_runtime)."""
    # Auto-detect delimiter (tab or comma); python engine required for sep=None
    df = pd.read_csv(
        resolver_path,
        dtype=str,
        sep=None,
        engine="python",
    )

    df.columns = [str(c).strip() for c in df.columns]
    missing = RESOLVER_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{resolver_path} is missing required columns: {', '.join(sorted(missing))}"
        )

    df["address_lc"] = df["address"].map(normalize_hex_address)
    df = df[df["address_lc"] != ""].copy()

    df["verified_static"] = df["verified_static"].astype(str).str.upper().str.strip()
    df["verified_runtime"] = df["verified_runtime"].astype(str).str.upper().str.strip()
    df["both_ok"] = (df["verified_static"] == "Y") & (df["verified_runtime"] == "Y")

    # Keep the last occurrence per address
    df = df.drop_duplicates(subset=["address_lc"], keep="last")

    return dict(zip(df["address_lc"], df["both_ok"]))


def _read_status_csv(path: Path) -> pd.DataFrame:
    """Read the status CSV with expected parsing options."""
    return pd.read_csv(
        path,
        dtype=str,
        low_memory=False,
        keep_default_na=False,
        na_values=[],
    )


def _validate_status_columns(status: pd.DataFrame, status_path: Path) -> None:
    """Raise if the status CSV is missing required columns."""
    missing = STATUS_REQUIRED_COLUMNS - set(status.columns)
    if missing:
        raise ValueError(
            f"{status_path} is missing required columns: {', '.join(sorted(missing))}"
        )


def _update_is_erc20_flags(status: pd.DataFrame, verdict_map: Dict[str, bool]) -> pd.Series:
    """Return updated is_erc_20 flags following resolver verdicts."""
    addr_lc = status["contract_address"].map(normalize_hex_address)
    is_false = status["is_erc_20"].astype(str).str.strip().str.lower() == "false"

    both_ok_series = addr_lc.map(verdict_map).fillna(False)
    new_flags = status["is_erc_20"].astype(str).str.strip().str.lower()
    new_flags = new_flags.where(
        ~is_false,
        other=both_ok_series.map(lambda ok: "true" if ok else "false"),
    )
    return new_flags


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Flip is_erc_20 to true only if static AND runtime verification both pass."
    )
    parser.add_argument(
        "--status-csv",
        required=True,
        type=Path,
        help="Input eth token status CSV (symbol,name,contract_address,is_erc_20,creation_block)",
    )
    parser.add_argument(
        "--resolver-tsv",
        required=True,
        type=Path,
        help="Proxy resolver TSV/CSV (address, verified_static, verified_runtime, ...)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        type=Path,
        help="Output CSV path (default: ../data/status_eth_tokens.csv)",
    )
    return parser


def main() -> int:
    """Run the CLI entrypoint and return an exit code."""
    args = _build_arg_parser().parse_args()

    status = _read_status_csv(args.status_csv)
    _validate_status_columns(status, args.status_csv)

    verdict_map = _load_resolver_verdict(args.resolver_tsv)

    out = status.copy()
    new_flags = _update_is_erc20_flags(out, verdict_map)
    out["is_erc_20"] = new_flags

    out.to_csv(args.output, index=False)

    is_false = status["is_erc_20"].astype(str).str.strip().str.lower() == "false"
    total = len(out)
    flipped_true = int((is_false & (new_flags == "true")).sum())
    still_false = int((is_false & (new_flags == "false")).sum())
    log(
        f"Wrote {args.output} ({total} rows). "
        f"Updated {flipped_true} tokens to true; {still_false} remained false."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
