#!/usr/bin/env python3
"""
3_1_add_local_status.py

Purpose
-------
Add a local validation status column to a token status CSV using a TSV lookup.

What it does
------------
- Reads a status CSV and a validation-status TSV.
- Normalizes addresses and maps each token to a local status.
- Writes the updated CSV and prints a summary of status counts.

Inputs
------
- status CSV (CSV; must include a contract address column such as `contract_address`)
- validation TSV (TSV; `address<TAB>validation_status`)

Outputs
-------
- output CSV (CSV; original columns plus `local_status`)

CLI
---
Examples:
  $ python 3_1_add_local_status.py --status-csv status_eth_tokens.csv --validation-tsv validation_status.tsv --out status_eth_tokens.updated.csv
  $ python 3_1_add_local_status.py --status-csv status_eth_tokens.csv --validation-tsv validation_status.tsv --out status_eth_tokens.local.csv

Notes
-----
- Addresses are normalized to lowercase `0x...` with length 42.
- Validation statuses like PASS/NOT_PASS are normalized; other values are preserved.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional

# Ensure helpers/ is importable when running from src/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.ethereum_codec_utils import normalize_hex_address
from utils.log_utils import log, log_exc

# ----------------------------- Constants -----------------------------
STATUS_COL = "local_status"
PREFERRED_STATUS_ORDER = ("PASS", "NOT_PASS", "NO_DATA")


def parse_validation_line(line: str) -> Optional[tuple[str, str]]:
    """Return (raw_address, raw_status) from a TSV line or None if unparseable."""
    parts = line.split("\t")
    if len(parts) < 2:
        parts = line.split()
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def normalize_validation_status(raw_status: str) -> str:
    """Return a normalized validation status string."""
    status = (raw_status or "").strip().upper()
    if status in {"PASS", "PASSED", "OK", "TRUE", "1"}:
        return "PASS"
    if status in {"NOT_PASS", "NOT-PASS", "FAIL", "FAILED", "FALSE", "0"}:
        return "NOT_PASS"
    return status


def load_validation_map(tsv_path: Path) -> Dict[str, str]:
    """Return a {address -> normalized_status} map from a validation TSV."""
    mapping: Dict[str, str] = {}
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parsed = parse_validation_line(line)
            if not parsed:
                log_exc(f"[warn] {tsv_path}:{line_no}: cannot parse line: {line!r}")
                continue

            addr_raw, status_raw = parsed
            addr = normalize_hex_address(addr_raw)
            if not addr:
                log_exc(f"[warn] {tsv_path}:{line_no}: invalid address: {addr_raw!r}")
                continue

            mapping[addr] = normalize_validation_status(status_raw)
    return mapping


def detect_address_column(fieldnames: Iterable[str]) -> str:
    """Return the column name that contains token addresses."""
    candidates = ("contract_address", "token_address", "address", "token", "contract")
    fieldnames_list = list(fieldnames)
    cleaned = [fn.strip() for fn in fieldnames_list]
    for want in candidates:
        for fn in cleaned:
            if fn.lower() == want:
                return fn
    for fn in cleaned:
        if "address" in fn.lower():
            return fn
    raise SystemExit(
        f"Could not detect address column. Columns are: {fieldnames_list}. "
        "Rename your address column to e.g. 'contract_address'."
    )


def _pct(n: int, total: int) -> str:
    """Return n/total as a percentage string."""
    if total <= 0:
        return "0.00%"
    return f"{(100.0 * n / total):.2f}%"


def process(status_csv: Path, validation_tsv: Path, out_csv: Path) -> None:
    """Read inputs, add local_status, and write the updated CSV."""
    val_map = load_validation_map(validation_tsv)

    total = 0
    matched = 0
    no_data = 0
    invalid_addr = 0
    status_counts: Counter[str] = Counter()

    with status_csv.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise SystemExit("Status CSV has no header row.")
        addr_col = detect_address_column(reader.fieldnames)

        out_fieldnames = list(reader.fieldnames)
        if STATUS_COL not in out_fieldnames:
            out_fieldnames.append(STATUS_COL)

        with out_csv.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
            writer.writeheader()

            for row in reader:
                total += 1
                addr = normalize_hex_address(row.get(addr_col, ""))
                if not addr:
                    invalid_addr += 1
                    row[STATUS_COL] = "NO_DATA"
                    status_counts["NO_DATA"] += 1
                    writer.writerow(row)
                    continue

                v = val_map.get(addr)
                if v is None:
                    row[STATUS_COL] = "NO_DATA"
                    no_data += 1
                    status_counts["NO_DATA"] += 1
                else:
                    # Enforce PASS / NOT_PASS if matched; otherwise preserve TSV status.
                    if v == "PASS":
                        row[STATUS_COL] = "PASS"
                    elif v == "NOT_PASS":
                        row[STATUS_COL] = "NOT_PASS"
                    else:
                        row[STATUS_COL] = v
                    matched += 1
                    status_counts[row[STATUS_COL]] += 1

                writer.writerow(row)

    log(f"wrote: {out_csv}")
    log(f"  rows processed: {total:,}")
    log(f"  matched in validation: {matched:,} ({_pct(matched, total)})")
    log(f"  NO_DATA: {no_data:,} ({_pct(no_data, total)})")
    log(f"  invalid/empty address in status CSV: {invalid_addr:,} ({_pct(invalid_addr, total)})")

    # --- local_status distribution ---
    log("\n========== local_status summary ==========")
    if total == 0:
        log("No rows.")
        return

    # show PASS / NOT_PASS / NO_DATA first (if present), then any other statuses
    shown = set()
    for k in PREFERRED_STATUS_ORDER:
        if k in status_counts:
            log(f"{k:10s}: {status_counts[k]:,}  ({_pct(status_counts[k], total)})")
            shown.add(k)

    for k in sorted(status_counts.keys()):
        if k in shown:
            continue
        log(f"{k:10s}: {status_counts[k]:,}  ({_pct(status_counts[k], total)})")

    log("=========================================")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Return parsed command-line arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--status-csv", required=True, type=Path, help="Input status_eth CSV")
    ap.add_argument("--validation-tsv", required=True, type=Path, help="Input validation_status TSV")
    ap.add_argument("--out", required=True, type=Path, help="Output CSV with added local_status column")
    return ap.parse_args(argv)


def main() -> int:
    """Parse CLI args and run the status update pipeline."""
    args = parse_args()

    if not args.status_csv.exists():
        raise SystemExit(f"Missing file: {args.status_csv}")
    if not args.validation_tsv.exists():
        raise SystemExit(f"Missing file: {args.validation_tsv}")

    process(args.status_csv, args.validation_tsv, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
