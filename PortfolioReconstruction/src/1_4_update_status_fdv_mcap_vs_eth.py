#!/usr/bin/env python3
"""
1_4_update_status_fdv_mcap_vs_eth.py

Purpose
-------
Augment the ETH token status table with a flag indicating whether each token's FDV
and market cap are strictly below Ethereum's market cap.

What it does
------------
- Loads CoinGecko market metrics and identifies the Ethereum reference row.
- Indexes metrics by contract address for fast lookup.
- Writes a status CSV with a new `fdv_mcap_lt_eth` column set to TRUE/FALSE/UNKNOWN.

Inputs
------
- status CSV (CSV; must include `contract_address`)
- metrics CSV (CSV from `1_3_fetch_supply_and_mcap_from_cg.py`; includes `cg_id`,
  `symbol`, `contract_address`, `market_cap_usd`, `fdv_usd`)

Outputs
-------
- <status_in>.with_eth_caps.csv (CSV; all original columns plus `fdv_mcap_lt_eth`)

CLI
---
Examples:
  $ python 1_4_update_status_fdv_mcap_vs_eth.py --status-in data/eth_tokens_status.csv --metrics tokens_cg_metrics.csv
  $ python 1_4_update_status_fdv_mcap_vs_eth.py --status-in data/eth_tokens_status.csv --metrics tokens_cg_metrics.csv \
      --status-out data/eth_tokens_status.with_eth_caps.csv

Notes
-----
- Ethereum is resolved by `cg_id=ethereum`, or by `symbol=ETH` with an empty
  `contract_address`.
- `fdv_mcap_lt_eth` is TRUE only when both `fdv_usd` and `market_cap_usd` are
  valid and strictly below Ethereum's market cap; otherwise FALSE or UNKNOWN.
"""

import argparse
import csv
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure helpers/ is importable when running from src/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.ethereum_codec_utils import normalize_hex_address
from utils.log_utils import log, log_exc

# ----------------------------- Constants -----------------------------
COLUMN_NAME = "fdv_mcap_lt_eth"
ETH_CG_ID = "ethereum"
ETH_SYMBOL = "ETH"
FLAG_TRUE = "TRUE"
FLAG_FALSE = "FALSE"
FLAG_UNKNOWN = "UNKNOWN"

CSVRow = Dict[str, str]


def parse_decimal_or_none(value: str) -> Optional[Decimal]:
    """Parse a Decimal from a string or return None if invalid/empty."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def read_csv_rows(path: Path) -> Tuple[List[CSVRow], List[str]]:
    """Read a CSV file into rows and fieldnames."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    return rows, fields


def find_eth_row(rows: List[CSVRow]) -> CSVRow:
    """Return the Ethereum metrics row or raise if it cannot be found."""
    for row in rows:
        cg_id = (row.get("cg_id") or "").strip().lower()
        if cg_id == ETH_CG_ID:
            return row
    for row in rows:
        symbol = (row.get("symbol") or "").strip().upper()
        contract_address = (row.get("contract_address") or "").strip()
        if symbol == ETH_SYMBOL and contract_address == "":
            return row
    raise ValueError("Could not locate ETH row in metrics CSV.")


def load_metrics(path: Path) -> Tuple[List[CSVRow], Dict[str, CSVRow], Decimal]:
    """Load metrics, build address index, and return the ETH market cap."""
    rows, _fields = read_csv_rows(path)
    if not rows:
        raise ValueError("Metrics CSV is empty.")

    eth_row = find_eth_row(rows)
    eth_mcap = parse_decimal_or_none(eth_row.get("market_cap_usd", ""))
    if eth_mcap is None:
        raise ValueError("ETH row has no valid market_cap_usd value.")

    index: Dict[str, CSVRow] = {}
    for row in rows:
        addr = normalize_hex_address(row.get("contract_address", ""))
        if not addr:
            continue
        if addr not in index:
            index[addr] = row

    return rows, index, eth_mcap


def derive_out_path(status_in: Path) -> Path:
    """Derive the default output path by inserting .with_eth_caps before suffix."""
    base_no_suffix = status_in.with_suffix("")
    ext = status_in.suffix or ".csv"
    filename = f"{base_no_suffix.name}.with_eth_caps{ext}"
    return base_no_suffix.with_name(filename)


def update_status_rows(
    status_rows: List[CSVRow],
    metrics_index: Dict[str, CSVRow],
    eth_mcap: Decimal,
) -> Tuple[int, int, int, int]:
    """Update status rows with ETH comparison flags and return counters."""
    updated_true = 0
    updated_false = 0
    missing_metrics = 0
    missing_numbers = 0

    for row in status_rows:
        addr = normalize_hex_address(row.get("contract_address", ""))
        metric_row = metrics_index.get(addr)
        flag_value = FLAG_UNKNOWN

        if metric_row is None:
            missing_metrics += 1
        else:
            mcap = parse_decimal_or_none(metric_row.get("market_cap_usd", ""))
            fdv = parse_decimal_or_none(metric_row.get("fdv_usd", ""))
            if mcap is None or fdv is None:
                missing_numbers += 1
            else:
                if mcap < eth_mcap and fdv < eth_mcap:
                    flag_value = FLAG_TRUE
                    updated_true += 1
                else:
                    flag_value = FLAG_FALSE
                    updated_false += 1

        row[COLUMN_NAME] = flag_value

    return updated_true, updated_false, missing_metrics, missing_numbers


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the script."""
    ap = argparse.ArgumentParser(
        description="Append ETH-market-cap comparison flags onto the token status CSV."
    )
    ap.add_argument("--status-in", dest="status_in_csv", required=True, help="Existing status CSV.")
    ap.add_argument(
        "--metrics",
        dest="metrics_csv",
        required=True,
        help="Output CSV from 1_3_fetch_supply_and_mcap_from_cg.py (contains FDV/mcap data).",
    )
    ap.add_argument(
        "--status-out",
        dest="status_out_csv",
        default=None,
        help="Where to write the augmented status CSV (default: <status_in>.with_eth_caps.csv).",
    )
    return ap.parse_args()


def main() -> int:
    """Run the CLI entrypoint and return an exit code."""
    args = parse_args()

    status_in = Path(args.status_in_csv)
    metrics_path = Path(args.metrics_csv)
    status_out = Path(args.status_out_csv) if args.status_out_csv else derive_out_path(status_in)

    metrics_rows, metrics_index, eth_mcap = load_metrics(metrics_path)
    log(f"Loaded {len(metrics_rows)} metric rows, ETH market_cap_usd={eth_mcap}")

    status_rows, status_fields = read_csv_rows(status_in)
    if not status_rows:
        log_exc("status CSV has no rows.")
        return 1

    out_fields = list(status_fields)
    if COLUMN_NAME not in out_fields:
        out_fields.append(COLUMN_NAME)

    updated_true, updated_false, missing_metrics, missing_numbers = update_status_rows(
        status_rows,
        metrics_index,
        eth_mcap,
    )

    with status_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(status_rows)

    log(
        f"wrote {len(status_rows)} rows to {status_out} | TRUE={updated_true}, "
        f"FALSE={updated_false}, UNKNOWN(missing metrics)={missing_metrics}, "
        f"UNKNOWN(missing numbers)={missing_numbers}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
