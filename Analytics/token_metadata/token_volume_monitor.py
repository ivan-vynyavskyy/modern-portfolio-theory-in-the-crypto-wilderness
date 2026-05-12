#!/usr/bin/env python3
"""
token_volume_monitor.py

Workflow:
  1. Read an ETH token status CSV (status_eth_tokens.csv by default).
  2. Fetch recent-N-day trading volumes via CoinGecko's /market_chart.
  3. Optionally, select low-volume tokens and fetch their all-time volume
     via /market_chart/range for a "second pass".

Both stages are configurable and can write independent CSV outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

COINGECKO_BASE_PUBLIC = "https://api.coingecko.com/api/v3"
COINGECKO_BASE_PRO = "https://pro-api.coingecko.com/api/v3"


def _mk_session(api_key: Optional[str]) -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    sess.mount("https://", adapter)
    sess.headers.update({"Accept": "application/json", "User-Agent": "token-volume-monitor/1.0"})
    if api_key:
        sess.headers.update({"x-cg-pro-api-key": api_key})
    return sess


def fetch_volume_series(
    session: requests.Session,
    base_url: str,
    address: str,
    vs_currency: str,
    days: int,
    pause_sec: float,
) -> Tuple[Optional[List[Tuple[int, float]]], Optional[str]]:
    """
    Call /market_chart for the last `days` to obtain total_volumes.
    Returns (series, error) where series is [(ts_ms, volume), ...].
    """
    url = f"{base_url}/coins/ethereum/contract/{address}/market_chart"
    params = {"vs_currency": vs_currency, "days": str(days), "interval": "daily"}
    try:
        resp = session.get(url, params=params, timeout=30)
    except Exception as exc:  # pragma: no cover - network-specific
        return None, f"request_error:{exc!r}"

    time.sleep(pause_sec)

    if resp.status_code == 404:
        return None, "not_found_404"
    if resp.status_code == 429:
        return None, "rate_limited_429"
    if resp.status_code >= 400:
        return None, f"http_{resp.status_code}"

    try:
        payload = resp.json()
    except Exception as exc:  # pragma: no cover - malformed body
        return None, f"json_error:{exc!r}"

    vols = payload.get("total_volumes")
    if not isinstance(vols, list) or not vols:
        return None, "no_total_volumes"

    series: List[Tuple[int, float]] = []
    for entry in vols:
        try:
            ts, val = entry
            series.append((int(ts), float(val)))
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
    """
    Call /market_chart/range and sum total_volumes over [start_ts, end_ts].
    Returns (sum_volume, num_points, error).
    """
    url = f"{base_url}/coins/ethereum/contract/{address}/market_chart/range"
    params = {"vs_currency": vs_currency, "from": start_ts, "to": end_ts}
    try:
        resp = session.get(url, params=params, timeout=60)
    except Exception as exc:  # pragma: no cover
        return None, None, f"request_error:{exc!r}"

    time.sleep(pause_sec)

    if resp.status_code == 404:
        return None, None, "not_found_404"
    if resp.status_code == 429:
        return None, None, "rate_limited_429"
    if resp.status_code >= 400:
        return None, None, f"http_{resp.status_code}"

    try:
        payload = resp.json()
    except Exception as exc:  # pragma: no cover
        return None, None, f"json_error:{exc!r}"

    vols = payload.get("total_volumes")
    if not isinstance(vols, list):
        return None, None, "no_total_volumes"

    total = 0.0
    points = 0
    for entry in vols:
        try:
            _, val = entry
            total += float(val)
            points += 1
        except Exception:
            continue

    if points == 0:
        return 0.0, 0, None

    return total, points, None


def _to_float(val) -> float:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return float("nan")
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return float("nan")


def _load_status_dataframe(path: str, include_non_erc20: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = {
        "symbol",
        "name",
        "contract_address",
        "is_erc_20",
        "creation_block",
        "price_data_status",
        "first_date",
        "last_date",
        "num_points",
        "error_message",
    }
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Input CSV missing columns: {sorted(missing)}")

    work = df.copy()
    if not include_non_erc20:
        work = work[work["is_erc_20"].astype(str).str.lower().isin({"true", "1", "yes"})]

    work = work.dropna(subset=["contract_address"]).copy()
    work["contract_address"] = work["contract_address"].astype(str).str.strip().str.lower()
    work = work[work["contract_address"] != ""]
    work = work.drop_duplicates(subset=["contract_address"]).reset_index(drop=True)
    return work


def _init_writer(path: Optional[str], fieldnames: List[str]):
    if not path:
        return None, None
    fobj = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fobj, fieldnames=fieldnames)
    writer.writeheader()
    fobj.flush()
    return fobj, writer


def run_recent_volume_pass(
    df_tokens: pd.DataFrame,
    session: requests.Session,
    base_url: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    vol_col = f"volume_{args.days}d_{args.vs}"
    fieldnames = ["contract_address", "symbol", "name", vol_col, "points", "status"]
    out_path = args.out_10d or f"volumes_{args.days}d_{args.vs}.csv"
    csv_file, csv_writer = _init_writer(out_path, fieldnames)

    results: List[Dict[str, Any]] = []
    total = len(df_tokens)
    print(
        f"[10d] Processing {total} token(s); vs={args.vs}, days={args.days}, "
        f"endpoint={'PRO' if args.api_key else 'PUBLIC'}"
    )

    for idx, row in df_tokens.iterrows():
        addr = row["contract_address"]
        symbol = str(row.get("symbol", "") or "")
        name = str(row.get("name", "") or "")

        series, err = fetch_volume_series(
            session,
            base_url,
            addr,
            vs_currency=args.vs,
            days=args.days,
            pause_sec=args.pause,
        )

        if err:
            print(f"[10d {idx+1}/{total}] {symbol or '-'} | {name or '-'} | {addr} -> ERROR: {err}")
            out_row = {
                "contract_address": addr,
                "symbol": symbol,
                "name": name,
                vol_col: None,
                "points": 0,
                "status": err,
            }
        else:
            total_vol = float(sum(val for _, val in series)) if series else 0.0
            print(
                f"[10d {idx+1}/{total}] {symbol or '-'} | {name or '-'} | {addr} -> "
                f"{vol_col} = {total_vol:,.2f} (points={len(series)})"
            )
            out_row = {
                "contract_address": addr,
                "symbol": symbol,
                "name": name,
                vol_col: total_vol,
                "points": len(series),
                "status": "ok",
            }

        results.append(out_row)
        if csv_writer:
            csv_writer.writerow(out_row)
            csv_file.flush()

    if csv_file:
        csv_file.close()
        print(f"[10d] Results written to {out_path}")
    else:
        print("[10d] No CSV output path provided; results kept in memory only.")

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by=vol_col, ascending=False, na_position="last").reset_index(drop=True)

    print("\n[10d] Top results (up to 50 rows):")
    with pd.option_context("display.max_rows", 50, "display.width", 200, "display.max_colwidth", 80):
        print(res_df[["symbol", "name", "contract_address", vol_col, "points", "status"]].to_string(index=False))

    return res_df


def run_alltime_pass(
    ten_day_df: pd.DataFrame,
    session: requests.Session,
    base_url: str,
    args: argparse.Namespace,
) -> None:
    vol_col = f"volume_{args.days}d_{args.vs}"
    if vol_col not in ten_day_df.columns:
        raise SystemExit(f"Column '{vol_col}' not found in 10d results; cannot run all-time stage.")

    work = ten_day_df.copy()
    if not args.alltime_include_errors_as_zero:
        work = work[work["status"].astype(str).str.lower().eq("ok")].copy()
    else:
        work.loc[~work["status"].astype(str).str.lower().eq("ok"), vol_col] = 0

    work[vol_col] = work[vol_col].apply(_to_float)
    sel = work[work[vol_col] < args.alltime_threshold].copy()

    if sel.empty:
        print(
            "[all-time] No tokens meet the selection criteria "
            f"(threshold={args.alltime_threshold}). Nothing to do."
        )
        return

    out_path = args.alltime_out or "volumes_alltime.csv"
    fieldnames = [
        "contract_address",
        "symbol",
        "name",
        vol_col,
        "alltime_volume_usd",
        "alltime_points",
        "status_alltime",
    ]
    csv_file, csv_writer = _init_writer(out_path, fieldnames)

    print(
        f"[all-time] Selected {len(sel)} token(s) with {vol_col} < {args.alltime_threshold}. "
        f"Range start={args.alltime_from_ts}, end={args.alltime_to_ts or 'now'}"
    )
    end_ts = args.alltime_to_ts or int(time.time())

    for idx, row in sel.reset_index(drop=True).iterrows():
        addr = str(row["contract_address"]).strip().lower()
        symbol = str(row.get("symbol", "") or "")
        name = str(row.get("name", "") or "")
        recent_val = row.get(vol_col)

        total, points, err = fetch_volume_range_sum(
            session,
            base_url,
            addr,
            vs_currency=args.vs,
            start_ts=args.alltime_from_ts,
            end_ts=end_ts,
            pause_sec=args.alltime_pause,
        )

        if err:
            print(f"[all-time {idx+1}/{len(sel)}] {symbol or '-'} | {name or '-'} | {addr} -> ERROR: {err}")
            out_row = {
                "contract_address": addr,
                "symbol": symbol,
                "name": name,
                vol_col: recent_val,
                "alltime_volume_usd": None,
                "alltime_points": None,
                "status_alltime": err,
            }
        else:
            print(
                f"[all-time {idx+1}/{len(sel)}] {symbol or '-'} | {name or '-'} | {addr} -> "
                f"all-time volume = {total:,.2f} (points={points})"
            )
            out_row = {
                "contract_address": addr,
                "symbol": symbol,
                "name": name,
                vol_col: recent_val,
                "alltime_volume_usd": float(total),
                "alltime_points": int(points),
                "status_alltime": "ok",
            }

        if csv_writer:
            csv_writer.writerow(out_row)
            csv_file.flush()

    if csv_file:
        csv_file.close()
        print(f"[all-time] Results written to {out_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Combined recent/all-time trading volume workflow.")
    ap.add_argument("--status-csv", required=True, help="Path to status_eth_tokens.csv (or similar).")
    ap.add_argument("--out-10d", default=None, help="CSV to write recent-volume results (default: volumes_{days}d_{vs}.csv).")
    ap.add_argument("--days", type=int, default=10, help="Window (days) for the first pass (default: 10).")
    ap.add_argument("--vs", default="usd", help="CoinGecko vs_currency (default: usd).")
    ap.add_argument("--include-non-erc20", action="store_true", help="Also process non-ERC20 rows (default: False).")
    ap.add_argument("--pause", type=float, default=0.8, help="Pause (seconds) between /market_chart requests.")
    ap.add_argument("--api-key", default=None, help="CoinGecko API key (Pro/Demo).")

    ap.add_argument("--alltime", action="store_true", help="Enable the all-time fallback pass.")
    ap.add_argument("--alltime-threshold", type=float, default=1.0, help="Recent-volume threshold that triggers all-time queries.")
    ap.add_argument("--alltime-out", default=None, help="CSV to write all-time results (default: volumes_alltime.csv).")
    ap.add_argument(
        "--alltime-include-errors-as-zero",
        action="store_true",
        help="Select errored 10d rows as if their volume were zero.",
    )
    ap.add_argument("--alltime-from-ts", dest="alltime_from_ts", type=int, default=1451606400, help="Range start (UNIX seconds).")
    ap.add_argument("--alltime-to-ts", dest="alltime_to_ts", type=int, default=None, help="Range end (UNIX seconds, default: now).")
    ap.add_argument("--alltime-pause", type=float, default=0.7, help="Pause between /market_chart/range requests.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    tokens_df = _load_status_dataframe(args.status_csv, args.include_non_erc20)
    if tokens_df.empty:
        print("No tokens to process after filtering; exiting.")
        return

    session = _mk_session(args.api_key)
    base_url = COINGECKO_BASE_PRO if args.api_key else COINGECKO_BASE_PUBLIC

    ten_day_df = run_recent_volume_pass(tokens_df, session, base_url, args)

    if args.alltime:
        run_alltime_pass(ten_day_df, session, base_url, args)
    else:
        print("[all-time] Skipped (enable with --alltime).")


if __name__ == "__main__":
    main()
