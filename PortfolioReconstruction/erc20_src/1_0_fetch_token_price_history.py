#!/usr/bin/env python3
"""
1_0_fetch_token_price_history.py

Purpose
-------
Batch-download full-history daily close prices from CoinGecko for token contract addresses.

What it does
------------
- Reads a token metadata table and filters to ERC-20 rows (if provided).
- Fetches `days=max` market_chart data per contract and computes daily close in UTC.
- Writes per-token price CSVs and appends a status CSV for success/failure.

Inputs
------
- --input (CSV/TSV; columns: symbol, name, contract_address; optional is_erc_20)
- COINGECKO_API_KEY (env var; optional if --api-key provided)

Outputs
-------
- --output-dir/<contract_address>_price.csv (CSV; columns: date, daily_close_<vs>)
- --status-file (CSV; columns: symbol,name,contract_address,status,first_date,last_date,num_days,num_points,error_message)

CLI
---
Examples:
  $ python 1_0_fetch_token_price_history.py --input tokens.csv --output-dir ./prices --status-file ./prices/status.csv --platform-id ethereum --vs-currency usd
  $ python 1_0_fetch_token_price_history.py --input tokens.tsv --output-dir ./prices --status-file ./prices/status.csv --api-key $COINGECKO_API_KEY --overwrite

Notes
-----
- Input delimiter is auto-detected (comma, tab, semicolon, pipe).
- Daily close is the last observation in UTC for each calendar date from CoinGecko.
- Non-success tokens are recorded in the status file and processing continues.
"""

import argparse
import csv
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests

# ----------------------------- Constants -----------------------------
COINGECKO_PRO_BASE = "https://pro-api.coingecko.com/api/v3"
DEFAULT_PLATFORM_ID = "ethereum"
DEFAULT_VS_CURRENCY = "usd"
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_WAIT = 5.0  # seconds base wait (exponential backoff)
DEFAULT_TIMEOUT = 60  # seconds
DEFAULT_SLEEP_BETWEEN = 0.5  # polite pause between tokens
TRUE_STRINGS = {"true", "1", "yes", "y", "t"}

# ----------------------------- Helpers -----------------------------

def get_api_key(explicit: Optional[str]) -> str:
    """Return the API key from CLI or environment, or exit if missing."""
    if explicit:
        return explicit
    env = os.getenv("COINGECKO_API_KEY")
    if env:
        return env
    raise SystemExit("CoinGecko API key required: pass --api-key or set COINGECKO_API_KEY env var.")


def autodetect_delimiter(sample_path: Path, fallback: str = ",") -> str:
    """Return the detected delimiter for a delimited text file."""
    with sample_path.open("r", newline="") as f:
        sample = f.read(4096)
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample)
        return dialect.delimiter
    except Exception:
        return fallback


def read_token_table(path: Path) -> Tuple[pd.DataFrame, int, int]:
    """Read the token metadata table and apply required normalization."""
    delim = autodetect_delimiter(path)
    df = pd.read_csv(path, sep=delim, dtype=str).rename(columns=str.lower)
    required = {"symbol", "name", "contract_address"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input file missing required columns: {sorted(missing)}")
    # Normalize whitespace and lowercase contract addresses.
    df["contract_address"] = df["contract_address"].str.strip().str.lower()
    df["symbol"] = df["symbol"].str.strip()
    df["name"] = df["name"].str.strip()
    # Drop rows with blank contract address.
    df = df[df["contract_address"].astype(bool)].copy()
    total_tokens = len(df)
    if "is_erc_20" in df.columns:
        normalized = (
            df["is_erc_20"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        erc_mask = normalized.isin(TRUE_STRINGS)
    else:
        erc_mask = pd.Series([True] * len(df), index=df.index)
    erc20_tokens = int(erc_mask.sum())
    df = df[erc_mask].copy()
    return df.reset_index(drop=True), total_tokens, erc20_tokens


def build_market_chart_url(platform_id: str, contract_address: str) -> str:
    """Return the CoinGecko market_chart URL for a token contract."""
    return f"{COINGECKO_PRO_BASE}/coins/{platform_id}/contract/{contract_address}/market_chart"


def fetch_market_chart(
    session: requests.Session,
    platform_id: str,
    contract_address: str,
    vs_currency: str,
    api_key: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_wait: float = DEFAULT_RETRY_WAIT,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Fetch the raw market_chart JSON with retries and backoff."""
    url = build_market_chart_url(platform_id, contract_address)
    params = {"vs_currency": vs_currency, "days": "max"}
    headers = {"X-CG-Pro-API-Key": api_key}

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = session.get(url, headers=headers, params=params, timeout=timeout)
        except requests.RequestException as exc:  # network / timeout
            if attempt > max_retries:
                raise RuntimeError(f"Network error after {max_retries} retries: {exc}") from exc
            sleep_for = retry_wait * (2 ** (attempt - 1))
            logging.warning("Network error %s (attempt %s/%s) -- retrying in %.1fs", exc, attempt, max_retries, sleep_for)
            time.sleep(sleep_for)
            continue

        # Respect status codes.
        if resp.status_code == 429:  # rate limited
            if attempt > max_retries:
                raise RuntimeError("HTTP 429 Too Many Requests -- retries exhausted")
            # Try honor Retry-After header if present
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    sleep_for = float(retry_after)
                except ValueError:
                    sleep_for = retry_wait * (2 ** (attempt - 1))
            else:
                sleep_for = retry_wait * (2 ** (attempt - 1))
            logging.warning("Rate limited (429) for %s -- sleeping %.1fs", contract_address, sleep_for)
            time.sleep(sleep_for)
            continue

        if resp.status_code >= 400:
            # Non-retryable (404, 401, etc.)
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            data = resp.json()
        except ValueError as exc:  # json decode error
            raise RuntimeError(f"Invalid JSON in response: {exc}") from exc

        return data


def prices_json_to_daily_close(
    data: Dict[str, Any],
    vs_currency: str,
) -> pd.DataFrame:
    """Convert market_chart JSON into a daily close DataFrame."""
    if "prices" not in data:
        raise RuntimeError("Response JSON missing 'prices' field")

    prices = data["prices"]
    if not prices:
        raise RuntimeError("Empty 'prices' array")

    df = pd.DataFrame(prices, columns=["timestamp_ms", f"price_{vs_currency}"])
    # Ensure numeric & sort
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df = df.dropna(subset=["timestamp_ms"]).sort_values("timestamp_ms")

    # Convert to date in UTC
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.date

    # Daily close = last observation in that date (already sorted)
    daily = (
        df.groupby("date")[f"price_{vs_currency}"]
        .last()
        .reset_index()
        .rename(columns={f"price_{vs_currency}": f"daily_close_{vs_currency}"})
    )

    return daily


def safe_filename(contract_address: str, suffix: str = "_price.csv") -> str:
    """Return a filesystem-safe filename for a contract address."""
    # Contract addresses are hex; just lower + ensure no path separators.
    safe = contract_address.lower().replace("/", "_")
    return f"{safe}{suffix}"


def append_status(
    status_path: Path,
    symbol: str,
    name: str,
    contract_address: str,
    status: str,
    first_date: Optional[str] = None,
    last_date: Optional[str] = None,
    num_days: Optional[int] = None,
    num_points: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """Append a status row to the status CSV, creating it if needed."""
    header_needed = not status_path.exists()
    with status_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow([
                "symbol",
                "name",
                "contract_address",
                "status",
                "first_date",
                "last_date",
                "num_days",
                "num_points",
                "error_message",
            ])
        writer.writerow([
            symbol,
            name,
            contract_address,
            status,
            first_date or "",
            last_date or "",
            num_days if num_days is not None else "",
            num_points if num_points is not None else "",
            error_message or "",
        ])


def process_one_token(
    session: requests.Session,
    row: pd.Series,
    vs_currency: str,
    platform_id: str,
    api_key: str,
    out_dir: Path,
    overwrite: bool,
    max_retries: int,
    retry_wait: float,
    timeout: int,
    status_path: Path,
) -> bool:
    """Return True if a fetch/write was attempted; False if skipped."""
    symbol = row["symbol"]
    name = row["name"]
    ca = row["contract_address"]

    out_file = out_dir / safe_filename(ca)
    if out_file.exists() and not overwrite:
        logging.info("Skipping %s (%s) -- output exists", symbol, ca)
        try:
            df_existing = pd.read_csv(out_file)
            if not df_existing.empty:
                first_date = df_existing["date"].iloc[0]
                last_date = df_existing["date"].iloc[-1]
                num_days = len(df_existing)
            else:
                first_date = last_date = None
                num_days = 0
            append_status(status_path, symbol, name, ca, "exists", first_date, last_date, num_days, None, None)
        except Exception:
            append_status(status_path, symbol, name, ca, "exists", None, None, None, None, None)
        return False

    try:
        raw = fetch_market_chart(
            session=session,
            platform_id=platform_id,
            contract_address=ca,
            vs_currency=vs_currency,
            api_key=api_key,
            max_retries=max_retries,
            retry_wait=retry_wait,
            timeout=timeout,
        )
        daily = prices_json_to_daily_close(raw, vs_currency=vs_currency)
        daily.to_csv(out_file, index=False)
        first_date = daily["date"].iloc[0].isoformat() if hasattr(daily["date"].iloc[0], "isoformat") else str(daily["date"].iloc[0])
        last_date = daily["date"].iloc[-1].isoformat() if hasattr(daily["date"].iloc[-1], "isoformat") else str(daily["date"].iloc[-1])
        num_days = len(daily)
        num_points = len(raw.get("prices", []))
        append_status(status_path, symbol, name, ca, "success", first_date, last_date, num_days, num_points, None)
        logging.info("Fetched %s rows for %s (%s)", num_days, symbol, ca)
    except Exception as exc:
        logging.error("Failed %s (%s): %s", symbol, ca, exc)
        append_status(status_path, symbol, name, ca, "failure", None, None, None, None, str(exc))
        return True
    return True


def parse_args(argv: Optional[Any] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Batch download CoinGecko daily closes for token contracts.")
    p.add_argument("--input", required=True, type=Path, help="Input CSV/TSV with symbol,name,contract_address columns.")
    p.add_argument("--output-dir", required=True, type=Path, help="Directory to write per-token price CSVs.")
    p.add_argument("--status-file", required=True, type=Path, help="CSV file logging per-token status.")
    p.add_argument("--platform-id", default=DEFAULT_PLATFORM_ID, help="CoinGecko platform id (default: ethereum).")
    p.add_argument("--vs-currency", default=DEFAULT_VS_CURRENCY, help="Quote currency, e.g. usd, eur (default: usd).")
    p.add_argument("--api-key", default=None, help="CoinGecko Pro API key (fallback env COINGECKO_API_KEY).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing per-token CSVs.")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max retries for API calls.")
    p.add_argument("--retry-wait", type=float, default=DEFAULT_RETRY_WAIT, help="Base wait seconds for exponential backoff.")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP request timeout seconds.")
    p.add_argument("--sleep-between", type=float, default=DEFAULT_SLEEP_BETWEEN, help="Seconds to sleep between token calls (politeness / rate limit).")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args(argv)


def main(argv: Optional[Any] = None) -> int:
    """Run the CLI entrypoint and return an exit code."""
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    api_key = get_api_key(args.api_key)

    # Ensure output locations
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Ensure status path parent exists
    args.status_file.parent.mkdir(parents=True, exist_ok=True)

    # Read token table
    tokens, total_tokens, erc20_tokens = read_token_table(args.input)
    logging.info("Loaded %s tokens from %s (ERC-20 flagged: %s)", total_tokens, args.input, erc20_tokens)

    session = requests.Session()

    for _, row in tokens.iterrows():
        attempted = process_one_token(
            session=session,
            row=row,
            vs_currency=args.vs_currency,
            platform_id=args.platform_id,
            api_key=api_key,
            out_dir=args.output_dir,
            overwrite=args.overwrite,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
            timeout=args.timeout,
            status_path=args.status_file,
        )
        if attempted and args.sleep_between > 0:
            time.sleep(args.sleep_between)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
