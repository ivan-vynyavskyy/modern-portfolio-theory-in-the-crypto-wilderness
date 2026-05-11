#!/usr/bin/env python3
"""
1_1_update_token_status_with_price.py

Purpose
-------
Merge local token verification status with CoinGecko price download metadata.

What it does
------------
- Loads the local token status CSV and price status CSV with expected schemas.
- Derives `price_data_status` and error messages based on download results.
- Merges price fields into the local table and writes a sorted output CSV.

Inputs
------
- Local status CSV (UTF-8; columns: symbol, name, contract_address, is_erc_20, ...)
- Price status CSV (UTF-8; columns: symbol, name, contract_address, status, first_date, last_date, num_points, error_message)

Outputs
-------
- Merged status CSV (UTF-8; original local columns plus price columns)

CLI
---
Examples:
  $ python 1_1_update_token_status_with_price.py --local ../data/status_eth_tokens.csv --price ../data/prices_download_status.csv --output ../data/status_eth_tokens.csv
  $ python 1_1_update_token_status_with_price.py --local ../data/status_eth_tokens.csv --price ../data/prices_download_status.csv --output ../data/status_eth_tokens.csv --include-unmatched

Notes
-----
- `price_data_status` uses a 15-day threshold for success and marks 2–14 points as not_enough_data.
- Matching uses lowercased, stripped `contract_address`.
- If price columns already exist in the local CSV, they are renamed with a `price_` prefix.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure helpers/ is importable when running from src/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.log_utils import log

# ----------------------------- Constants -----------------------------
PRICE_RESULT_COLUMNS = [
    "price_data_status",
    "first_date",
    "last_date",
    "num_points",
    "error_message",
]
MIN_POINTS_FOR_SUCCESS = 15


def _load_local_table(path: Path) -> pd.DataFrame:
    """Load the local status CSV using the exact, known schema."""
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    required = ("symbol", "name", "contract_address", "is_erc_20")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def _load_price_table(path: Path) -> pd.DataFrame:
    """Load the price status CSV using the expected schema."""
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    required = (
        "symbol",
        "name",
        "contract_address",
        "status",
        "first_date",
        "last_date",
        "num_points",
        "error_message",
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_addr(x: Optional[str]) -> str:
    """Normalize contract addresses for matching."""
    if x is None:
        return ""
    return str(x).strip().lower()


def _norm_local_status(x: Optional[str]) -> str:
    """Normalize local status values to PASS/NOT_PASS/no_data."""
    if x is None:
        return "no_data"
    s = str(x).strip().lower()
    if s in {"pass", "ok", "yes", "true"}:
        return "PASS"
    if s in {"not_pass", "fail", "failed", "no", "false", "error", "bad"}:
        return "NOT_PASS"
    if s in {"no_data", "na", "n/a", ""}:
        return "no_data"
    return "no_data"


def _safe_int(x: object) -> Optional[int]:
    """Parse an integer or return None when invalid."""
    try:
        return int(x)
    except Exception:
        return None


def _price_success_flag(x: Optional[str]) -> bool:
    """Return True if the status string indicates success."""
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {
        "success",
        "succeeded",
        "ok",
        "pass",
        "passed",
        "complete",
        "completed",
        "done",
        "true",
    }


def _price_failure_flag(x: Optional[str]) -> bool:
    """Return True if the status string indicates failure."""
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"failure", "failed", "error", "err", "false", "timeout", "bad"}


def _resolve_price_column(df: pd.DataFrame, base: str) -> str:
    """Resolve the actual output column name for a price field."""
    if base in df.columns:
        return base
    if base.startswith("price_"):
        cand = f"{base}_price"
        if cand in df.columns:
            return cand
    else:
        cand = f"price_{base}"
        if cand in df.columns:
            return cand
    raise KeyError(f"Could not resolve column for {base}")


def _derive_price_status(row: pd.Series) -> str:
    """Derive normalized price_data_status from raw status/num_points."""
    n = row["__num_points_int"]
    succ = row["__succ"]
    fail = row["__fail"]

    # Single point → effectively "not yet available"
    if n is not None and n == 1:
        return "not_available"

    # If status says success and we have some points, check threshold
    if succ and n is not None:
        if n >= MIN_POINTS_FOR_SUCCESS:
            return "success"
        if n > 1:
            return "not_enough_data"

    # Explicit failure or clearly bad num_points
    if fail or (n is not None and n <= 0):
        return "failure"

    # Fallback purely on num_points if flags are weird/missing
    if n is not None and n >= MIN_POINTS_FOR_SUCCESS:
        return "success"
    if n is not None and n > 1:
        return "not_enough_data"

    # Default → failure
    return "failure"


def _derive_price_error_message(row: pd.Series, respect_price_error: bool) -> str:
    """Return the standardized error_message for a price row."""
    st = row["price_data_status"]
    msg = row["error_message"] if isinstance(row["error_message"], str) else ""
    if st == "not_available":
        if respect_price_error and msg.strip():
            return msg
        return "price not yet available"
    if st == "not_enough_data":
        if respect_price_error and msg.strip():
            return msg
        return "price history has fewer than 15 days of data"
    if st == "failure":
        return msg.strip() if msg.strip() else "price download failed"
    return ""


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_tables(
    local_path: Path,
    price_path: Path,
    respect_price_error: bool = False,
    include_unmatched_price: bool = False,
) -> pd.DataFrame:
    """Merge local status and price status tables into a unified DataFrame."""
    # Load normalized LOCAL
    local_df = _load_local_table(local_path)
    local_value_cols = list(local_df.columns)
    if "local_status" in local_df.columns:
        local_df["local_status"] = local_df["local_status"].map(_norm_local_status)
    elif "status" in local_df.columns:
        local_df["local_status"] = local_df["status"].map(_norm_local_status)
        if "local_status" not in local_value_cols:
            local_value_cols.append("local_status")
    local_df["__addr_key"] = local_df["contract_address"].map(_clean_addr)

    # Load PRICE table
    price_df = _load_price_table(price_path)
    price_df["__addr_key"] = price_df["contract_address"].map(_clean_addr)
    price_df["__num_points_int"] = price_df["num_points"].map(_safe_int)
    price_df["__succ"] = price_df["status"].map(_price_success_flag)
    price_df["__fail"] = price_df["status"].map(_price_failure_flag)

    price_df["price_data_status"] = price_df.apply(_derive_price_status, axis=1)
    price_df["__err2"] = price_df.apply(
        _derive_price_error_message, axis=1, respect_price_error=respect_price_error
    )
    price_df["error_message"] = price_df["__err2"]

    # Keep first/last/num for success AND not_enough_data, blank for others
    mask_clear = ~price_df["price_data_status"].isin(["success", "not_enough_data"])
    price_df.loc[mask_clear, ["first_date", "last_date", "num_points"]] = ""

    price_column_renames = {}
    price_output_columns = []
    for col in PRICE_RESULT_COLUMNS:
        target = col
        if target in local_value_cols:
            target = f"price_{target}" if not target.startswith("price_") else f"{target}_price"
        price_output_columns.append(target)
        if target != col:
            price_column_renames[col] = target

    price_keep = price_df[["__addr_key"] + PRICE_RESULT_COLUMNS].copy()
    if price_column_renames:
        price_keep = price_keep.rename(columns=price_column_renames)

    merged = pd.merge(
        local_df,
        price_keep,
        how="left",
        on="__addr_key",
        suffixes=("", "_price"),
    )

    price_status_col = price_column_renames.get("price_data_status", "price_data_status")
    error_col = price_column_renames.get("error_message", "error_message")
    first_col = price_column_renames.get("first_date", "first_date")
    last_col = price_column_renames.get("last_date", "last_date")
    points_col = price_column_renames.get("num_points", "num_points")

    # No matching price row -> failure
    merged[price_status_col] = merged[price_status_col].fillna("failure")
    merged[error_col] = merged[error_col].fillna("no matching price status row")

    # Keep first/last/num for success AND not_enough_data, blank out for the rest
    mask_keep = merged[price_status_col].isin(["success", "not_enough_data"])
    merged.loc[~mask_keep, [first_col, last_col, points_col]] = ""

    # Sorting helpers
    local_rank = {"PASS": 0, "NOT_PASS": 1, "no_data": 2}
    price_rank = {
        "success": 0,
        "not_enough_data": 1,
        "failure": 2,
        "not_available": 3,
    }

    if "local_status" in merged.columns:
        merged["_lsort"] = merged["local_status"].map(lambda x: local_rank.get(x, 3))
    else:
        merged["_lsort"] = 0
    merged["_psort"] = merged[price_status_col].map(lambda x: price_rank.get(x, 3))
    merged["_ssort"] = merged["symbol"].fillna("").str.lower()

    sort_cols = ["_lsort", "_psort", "_ssort"]
    merged = merged.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    drop_helper = [c for c in merged.columns if c.startswith("__") or c in sort_cols]
    merged = merged.drop(columns=drop_helper)

    out_cols = [c for c in local_value_cols if c in merged.columns]
    out_cols.extend([c for c in price_output_columns if c in merged.columns])
    out = merged[out_cols].copy()

    # Sort by requested buckets
    if include_unmatched_price:
        local_keys = set(local_df["__addr_key"])
        extra = price_df[~price_df["__addr_key"].isin(local_keys)].copy()
        if not extra.empty:
            extra = extra.rename(columns=price_column_renames)
            extra_rows = pd.DataFrame("", index=extra.index, columns=out_cols)
            for base_col in ("symbol", "name", "contract_address"):
                if base_col in extra_rows.columns and base_col in extra.columns:
                    extra_rows[base_col] = extra[base_col]
            if "local_status" in extra_rows.columns:
                extra_rows["local_status"] = "no_data"
            for pcol in price_output_columns:
                if pcol in extra_rows.columns and pcol in extra.columns:
                    extra_rows[pcol] = extra[pcol]
            out = pd.concat([out, extra_rows], ignore_index=True)

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Merge local token status with price download status."
    )
    p.add_argument("--local", required=True, type=Path, help="Local status CSV.")
    p.add_argument("--price", required=True, type=Path, help="Price status CSV.")
    p.add_argument("--output", required=True, type=Path, help="Merged output CSV.")
    p.add_argument(
        "--respect-price-error",
        action="store_true",
        help="Keep PRICE file error_message for not_available/not_enough_data rows.",
    )
    p.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Append PRICE rows that don't appear in LOCAL.",
    )
    return p.parse_args()


def main() -> int:
    """Run the CLI entrypoint and return an exit code."""
    args = _parse_args()
    df = merge_tables(
        local_path=args.local,
        price_path=args.price,
        respect_price_error=args.respect_price_error,
        include_unmatched_price=args.include_unmatched,
    )
    df.to_csv(args.output, index=False)
    log(f"Wrote merged table: {args.output} ({len(df)} rows)")

    price_status_col = _resolve_price_column(df, "price_data_status")
    total_tokens = len(df)
    is_erc20 = df["is_erc_20"].fillna("").str.strip().str.lower()
    price_status = df[price_status_col].fillna("").str.strip().str.lower()

    num_non_erc20 = (is_erc20 == "false").sum()
    num_price_not_success = (price_status != "success").sum()
    num_erc20_with_price = ((is_erc20 == "true") & (price_status == "success")).sum()
    num_not_enough = (price_status == "not_enough_data").sum()

    log("Summary:")
    log(f"  Total tokens: {total_tokens}")
    log(f"  Tokens with is_erc_20 == false: {num_non_erc20}")
    log(f"  Tokens with price_data_status != success: {num_price_not_success}")
    log(
        "  Tokens with price_data_status == not_enough_data "
        f"(num_points < 15 and > 1): {num_not_enough}"
    )
    log(
        "  Tokens with is_erc_20 == true and price_data_status == success: "
        f"{num_erc20_with_price}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
