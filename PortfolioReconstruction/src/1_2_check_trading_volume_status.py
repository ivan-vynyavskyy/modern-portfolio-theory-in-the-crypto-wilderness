#!/usr/bin/env python3
"""
1_2_check_trading_volume_status.py

Purpose
-------
Annotate ERC-20 tokens with a trading-volume status using CoinGecko volume data.

What it does
------------
- Filters ERC-20 tokens that have successful price data.
- Checks recent volume over a short lookback window.
- Falls back to all-time volume when recent volume is zero.

Inputs
------
- Status CSV (CSV; required columns: contract_address, is_erc_20, price_data_status; optional: symbol, name)
- CoinGecko API key (string; via --api-key, optional)

Outputs
-------
- Output CSV (CSV; input columns plus trading_volume_status or the provided status column)

CLI
---
Examples:
  $ python 1_2_check_trading_volume_status.py --csv data/status_eth_tokens.csv
  $ python 1_2_check_trading_volume_status.py --csv data/status_eth_tokens.csv --out data/status_eth_tokens_with_vol.csv --days 30 --api-key $COINGECKO_API_KEY

Notes
-----
- Only ERC-20 tokens with price_data_status == "success" are evaluated.
- Tokens with zero recent volume are checked over the full historical range.
- Tokens that are not evaluated remain "unchecked".
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests

# Ensure helpers/ is importable when running from src/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


from utils.http_utils import build_retry_session
from utils.log_utils import log

# ----------------------------- Constants -----------------------------
USER_AGENT = "volume-status-script/1.0"

DEFAULT_STATUS_COLUMN = "trading_volume_status"
DEFAULT_DAYS = 10
DEFAULT_VS = "usd"
DEFAULT_PAUSE_SEC = 0.8
DEFAULT_RANGE_FROM = 1451606400  # 2016-01-01

REQUIRED_COLUMNS = ("contract_address", "is_erc_20", "price_data_status")


def _http_error_from_status(status_code: int) -> Optional[str]:
    """Map HTTP status codes to error labels, or None if OK."""
    if status_code == 404:
        return "not_found_404"
    if status_code == 429:
        return "rate_limited_429"
    if status_code >= 400:
        return f"http_{status_code}"
    return None


def fetch_volume_series(
    session: requests.Session,
    base_url: str,
    address: str,
    vs_currency: str,
    days: int,
    pause_sec: float,
) -> Tuple[Optional[List[Tuple[int, float]]], Optional[str]]:
    """Fetch recent volume series from /market_chart and return (series, error)."""
    url = f"{base_url}/coins/ethereum/contract/{address}/market_chart"
    params = {"vs_currency": vs_currency, "days": str(days), "interval": "daily"}
    try:
        resp = session.get(url, params=params, timeout=30)
    except Exception as e:  # pragma: no cover - network errors
        return None, f"request_error:{e!r}"

    time.sleep(pause_sec)

    status_err = _http_error_from_status(resp.status_code)
    if status_err:
        return None, status_err

    try:
        data = resp.json()
    except Exception as e:  # pragma: no cover - unexpected body
        return None, f"json_error:{e!r}"

    vols = data.get("total_volumes")
    if not isinstance(vols, list) or not vols:
        return None, "no_total_volumes"

    series: List[Tuple[int, float]] = []
    for item in vols:
        try:
            ts_ms, vol = item
            series.append((int(ts_ms), float(vol)))
        except Exception:
            continue

    if not series:
        return None, "empty_series"

    return series, None


def fetch_volume_range_sum(
    session: requests.Session,
    base_url: str,
    address: str,
    vs_currency: str,
    start_ts: int,
    end_ts: int,
    pause_sec: float,
) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """Fetch all-time volume sum from /market_chart/range."""
    url = f"{base_url}/coins/ethereum/contract/{address}/market_chart/range"
    params = {"vs_currency": vs_currency, "from": start_ts, "to": end_ts}
    try:
        resp = session.get(url, params=params, timeout=60)
    except Exception as e:  # pragma: no cover - network errors
        return None, None, f"request_error:{e!r}"

    time.sleep(pause_sec)

    status_err = _http_error_from_status(resp.status_code)
    if status_err:
        return None, None, status_err

    try:
        data = resp.json()
    except Exception as e:  # pragma: no cover - unexpected body
        return None, None, f"json_error:{e!r}"

    vols = data.get("total_volumes")
    if not isinstance(vols, list):
        return None, None, "no_total_volumes"

    total = 0.0
    n = 0
    for item in vols:
        try:
            _, val = item
            total += float(val)
            n += 1
        except Exception:
            continue

    if n == 0:
        return 0.0, 0, None

    return total, n, None


def _safe_zero(value: Optional[float]) -> float:
    """Convert None/NaN to 0.0, otherwise cast to float."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return float(value)


def _ensure_required_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise SystemExit if any required columns are missing."""
    for col in columns:
        if col not in df.columns:
            raise SystemExit(f"Input CSV is missing required column '{col}'.")


def _prepare_work_dataframe(df: pd.DataFrame, status_column: str) -> pd.DataFrame:
    """Return the filtered, normalized, de-duplicated work set for evaluation."""
    df[status_column] = "unchecked"

    mask = (
        df["is_erc_20"].astype(str).str.strip().str.lower().eq("true")
        & df["price_data_status"].astype(str).str.lower().eq("success")
        & df["contract_address"].notna()
    )
    work = df[mask].copy()
    work["contract_address"] = work["contract_address"].astype(str).str.strip().str.lower()
    work = work[work["contract_address"] != ""]

    if work.empty:
        return work

    return work.drop_duplicates(subset=["contract_address"])

def _evaluate_token(
    session: requests.Session,
    base_url: str,
    addr: str,
    vs_currency: str,
    days: int,
    pause_sec: float,
    range_from: int,
    range_to: int,
) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[str]]:
    """Return (status, volume, error, stage) for a single token."""
    series, err = fetch_volume_series(
        session,
        base_url,
        addr,
        vs_currency=vs_currency,
        days=days,
        pause_sec=pause_sec,
    )
    if err:
        return None, None, f"10d volume ERROR: {err}", None

    volume_10d = _safe_zero(sum(val for _, val in series))
    if volume_10d > 0:
        return "good", volume_10d, None, "10d"

    total_all, _, err2 = fetch_volume_range_sum(
        session,
        base_url,
        addr,
        vs_currency=vs_currency,
        start_ts=range_from,
        end_ts=range_to,
        pause_sec=pause_sec,
    )
    if err2:
        return None, None, f"all-time volume ERROR: {err2}", None

    total_all = _safe_zero(total_all)
    status = "good" if total_all > 0 else "bad"
    return status, total_all, None, "all-time"

def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    ap = argparse.ArgumentParser(description="Tag ERC-20 tokens by trading-volume sufficiency.")
    ap.add_argument("--csv", required=True, help="Input status CSV (e.g., status_eth_tokens.csv).")
    ap.add_argument(
        "--out",
        default=None,
        help="Optional path to write updated CSV. Defaults to in-place overwrite of --csv.",
    )
    ap.add_argument(
        "--status-column",
        default=DEFAULT_STATUS_COLUMN,
        help=f"Column name to write (default: {DEFAULT_STATUS_COLUMN}).",
    )
    ap.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Lookback window (days) for the first pass (default: {DEFAULT_DAYS}).",
    )
    ap.add_argument(
        "--vs",
        default=DEFAULT_VS,
        help=f"CoinGecko vs_currency (default: {DEFAULT_VS}).",
    )
    ap.add_argument(
        "--pause",
        type=float,
        default=DEFAULT_PAUSE_SEC,
        help=f"Pause between API calls in seconds (default: {DEFAULT_PAUSE_SEC}).",
    )
    ap.add_argument(
        "--range-from",
        type=int,
        default=DEFAULT_RANGE_FROM,
        help="UNIX timestamp for all-time range start (default: 2016-01-01).",
    )
    ap.add_argument(
        "--range-to",
        type=int,
        default=None,
        help="UNIX timestamp for all-time range end (default: now).",
    )
    ap.add_argument("--api-key", default=None, help="CoinGecko API key (Demo/Pro).")
    return ap

def main() -> int:
    """Run the CLI workflow and return a process exit code."""
    args = _build_arg_parser().parse_args()

    out_path = Path(args.out or args.csv)
    end_ts = args.range_to or int(time.time())

    df = pd.read_csv(args.csv)
    _ensure_required_columns(df, REQUIRED_COLUMNS)

    work = _prepare_work_dataframe(df, args.status_column)
    if work.empty:
        log("No ERC-20 tokens with successful price data to evaluate. Nothing to do.")
        df.to_csv(out_path, index=False)
        return 0

    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if args.api_key:
        headers["x-cg-pro-api-key"] = args.api_key
    session = build_retry_session(headers=headers)
    base_url = "https://pro-api.coingecko.com/api/v3"

    log(
        f"Evaluating {len(work)} token(s); vs={args.vs}, days={args.days}, "
        f"endpoint=PRO"
    )

    for idx, (row_idx, row) in enumerate(work.iterrows(), start=1):
        addr = row["contract_address"]
        symbol = str(row.get("symbol", "") or "")
        name = str(row.get("name", "") or "")

        status, volume, err, stage = _evaluate_token(
            session,
            base_url,
            addr,
            vs_currency=args.vs,
            days=args.days,
            pause_sec=args.pause,
            range_from=args.range_from,
            range_to=end_ts,
        )

        if err:
            log(f"[{idx}/{len(work)}] {symbol or '-'} | {name or '-'} | {addr} -> {err}")
            continue

        df.at[row_idx, args.status_column] = status
        if stage == "10d":
            log(
                f"[{idx}/{len(work)}] {symbol or '-'} | {name or '-'} | {addr} -> "
                f"10d volume {volume:,.2f} -> GOOD"
            )
            continue

        state_msg = "GOOD" if status == "good" else "BAD"
        log(
            f"[{idx}/{len(work)}] {symbol or '-'} | {name or '-'} | {addr} -> "
            f"all-time volume {volume:,.2f} -> {state_msg}"
        )

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    log(f"Wrote updated CSV with column '{args.status_column}' to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
