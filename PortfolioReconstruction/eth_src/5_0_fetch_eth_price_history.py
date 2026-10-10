#!/usr/bin/env python3
"""
5_0_fetch_eth_price_history.py

Purpose
-------
ETH twin of 1_0_fetch_token_price_history.py: download the full-history daily
close price of native ETH from CoinGecko and write it as a per-token price CSV
under the ETH pseudo-token address, so the portfolio pipeline (4_0) consumes
it exactly like any ERC-20 price file.

What it does
------------
- Fetches `days=max` market_chart data for the `ethereum` COIN id (native ETH
  is not a contract, so the coin endpoint replaces 1_0's contract endpoint —
  the only functional difference from 1_0).
- Computes daily close in UTC identically to 1_0.
- Writes <output-dir>/<pseudo_token>_price.csv and appends the same status CSV
  format 1_0 uses.
- Also writes <output-dir>/<pseudo_token>_supply.csv (date, market_cap_usd,
  circulating_supply = market cap / price, daily) from the same response.
  5_0_portfolio_backtest_pipeline.py takes it as --eth-supply-csv so the
  market-cap-weight strategy uses ETH's circulating supply instead of the
  ledger's balance sum, which double counts beacon-chain withdrawals
  (the deposit contract keeps every ETH ever deposited).

Inputs
------
- COINGECKO_API_KEY (env var; optional if --api-key provided)

Outputs
-------
- --output-dir/0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_price.csv
  (CSV; columns: date, daily_close_<vs>)
- --output-dir/0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_supply.csv
  (CSV; columns: date, market_cap_usd, circulating_supply)
- --status-file (CSV; columns: symbol,name,contract_address,status,first_date,last_date,num_days,num_points,error_message)

CLI
---
Examples:
  $ python 5_0_fetch_eth_price_history.py --output-dir ./prices --status-file ./prices/status.csv
  $ python 5_0_fetch_eth_price_history.py --output-dir ./prices --status-file ./prices/status.csv --api-key $COINGECKO_API_KEY --overwrite

Notes
-----
- Daily close is the last observation in UTC for each calendar date from CoinGecko.
- The pseudo address 0xee...ee (20 bytes of 0xee) is the marker 4_0 and the
  ETH aggregator use for native ETH.
"""

import argparse
import csv
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

# ----------------------------- Constants -----------------------------
COINGECKO_PRO_BASE = "https://pro-api.coingecko.com/api/v3"
DEFAULT_COIN_ID = "ethereum"
DEFAULT_VS_CURRENCY = "usd"
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_WAIT = 5.0  # seconds base wait (exponential backoff)
DEFAULT_TIMEOUT = 60  # seconds

ETH_PSEUDO_TOKEN = "0x" + "ee" * 20
ETH_SYMBOL = "ETH"
ETH_NAME = "Ethereum"

# ----------------------------- Helpers -----------------------------

def get_api_key(explicit: Optional[str]) -> str:
    """Return the API key from CLI or environment, or exit if missing."""
    if explicit:
        return explicit
    env = os.getenv("COINGECKO_API_KEY")
    if env:
        return env
    raise SystemExit("CoinGecko API key required: pass --api-key or set COINGECKO_API_KEY env var.")


def build_market_chart_url(coin_id: str) -> str:
    """Return the CoinGecko market_chart URL for a native coin id."""
    return f"{COINGECKO_PRO_BASE}/coins/{coin_id}/market_chart"


def fetch_market_chart(
    session: requests.Session,
    coin_id: str,
    vs_currency: str,
    api_key: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_wait: float = DEFAULT_RETRY_WAIT,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Fetch the raw market_chart JSON with retries and backoff."""
    url = build_market_chart_url(coin_id)
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
            logging.warning("Rate limited (429) for %s -- sleeping %.1fs", coin_id, sleep_for)
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


def market_chart_to_daily_supply(data: Dict[str, Any], vs_currency: str) -> pd.DataFrame:
    """Daily circulating supply = daily market cap / daily close, both taken as
    the last observation of each UTC calendar date."""
    if "market_caps" not in data or not data["market_caps"]:
        raise RuntimeError("Response JSON missing or empty 'market_caps' field")
    caps = pd.DataFrame(data["market_caps"], columns=["timestamp_ms", "market_cap_usd"])
    prices = pd.DataFrame(data["prices"], columns=["timestamp_ms", "price"])
    frames = []
    for df, col in ((caps, "market_cap_usd"), (prices, "price")):
        df = df.copy()
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().sort_values("timestamp_ms")
        df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.date
        frames.append(df.groupby("date")[col].last())
    daily = pd.concat(frames, axis=1).dropna()
    daily = daily[daily["price"] > 0]
    daily["circulating_supply"] = daily["market_cap_usd"] / daily["price"]
    return daily.reset_index()[["date", "market_cap_usd", "circulating_supply"]]


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


def process_eth(
    session: requests.Session,
    coin_id: str,
    token_address: str,
    vs_currency: str,
    api_key: str,
    out_dir: Path,
    overwrite: bool,
    max_retries: int,
    retry_wait: float,
    timeout: int,
    status_path: Path,
) -> bool:
    """Return True if a fetch/write was attempted; False if skipped."""
    out_file = out_dir / safe_filename(token_address)
    supply_file = out_dir / safe_filename(token_address, "_supply.csv")
    if out_file.exists() and supply_file.exists() and not overwrite:
        logging.info("Skipping %s (%s) -- price and supply outputs exist", ETH_SYMBOL, token_address)
        try:
            df_existing = pd.read_csv(out_file)
            if not df_existing.empty:
                first_date = df_existing["date"].iloc[0]
                last_date = df_existing["date"].iloc[-1]
                num_days = len(df_existing)
            else:
                first_date = last_date = None
                num_days = 0
            append_status(status_path, ETH_SYMBOL, ETH_NAME, token_address, "exists", first_date, last_date, num_days, None, None)
        except Exception:
            append_status(status_path, ETH_SYMBOL, ETH_NAME, token_address, "exists", None, None, None, None, None)
        return False

    try:
        raw = fetch_market_chart(
            session=session,
            coin_id=coin_id,
            vs_currency=vs_currency,
            api_key=api_key,
            max_retries=max_retries,
            retry_wait=retry_wait,
            timeout=timeout,
        )
        daily = prices_json_to_daily_close(raw, vs_currency=vs_currency)
        if overwrite or not out_file.exists():
            daily.to_csv(out_file, index=False)
        else:
            logging.info("Price CSV %s kept (exists); writing the supply CSV only", out_file)
        supply = market_chart_to_daily_supply(raw, vs_currency=vs_currency)
        supply.to_csv(supply_file, index=False)
        logging.info("Wrote %s rows of circulating supply to %s", len(supply), supply_file)
        first_date = daily["date"].iloc[0].isoformat() if hasattr(daily["date"].iloc[0], "isoformat") else str(daily["date"].iloc[0])
        last_date = daily["date"].iloc[-1].isoformat() if hasattr(daily["date"].iloc[-1], "isoformat") else str(daily["date"].iloc[-1])
        num_days = len(daily)
        num_points = len(raw.get("prices", []))
        append_status(status_path, ETH_SYMBOL, ETH_NAME, token_address, "success", first_date, last_date, num_days, num_points, None)
        logging.info("Fetched %s rows for %s (%s)", num_days, ETH_SYMBOL, token_address)
    except Exception as exc:
        logging.error("Failed %s (%s): %s", ETH_SYMBOL, token_address, exc)
        append_status(status_path, ETH_SYMBOL, ETH_NAME, token_address, "failure", None, None, None, None, str(exc))
        return True
    return True


def parse_args(argv: Optional[Any] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Download CoinGecko daily closes for native ETH under the pseudo-token address.")
    p.add_argument("--output-dir", required=True, type=Path, help="Directory to write the ETH price CSV.")
    p.add_argument("--status-file", required=True, type=Path, help="CSV file logging fetch status.")
    p.add_argument("--coin-id", default=DEFAULT_COIN_ID, help="CoinGecko coin id (default: ethereum).")
    p.add_argument("--token-address", default=ETH_PSEUDO_TOKEN, help=f"Pseudo token address for the output filename (default: {ETH_PSEUDO_TOKEN}).")
    p.add_argument("--vs-currency", default=DEFAULT_VS_CURRENCY, help="Quote currency, e.g. usd, eur (default: usd).")
    p.add_argument("--api-key", default=None, help="CoinGecko Pro API key (fallback env COINGECKO_API_KEY).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite the existing ETH price CSV.")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max retries for API calls.")
    p.add_argument("--retry-wait", type=float, default=DEFAULT_RETRY_WAIT, help="Base wait seconds for exponential backoff.")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP request timeout seconds.")
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
    args.status_file.parent.mkdir(parents=True, exist_ok=True)

    token_address = args.token_address.lower()
    logging.info("Fetching %s (%s) as pseudo token %s", ETH_NAME, args.coin_id, token_address)

    session = requests.Session()
    process_eth(
        session=session,
        coin_id=args.coin_id,
        token_address=token_address,
        vs_currency=args.vs_currency,
        api_key=api_key,
        out_dir=args.output_dir,
        overwrite=args.overwrite,
        max_retries=args.max_retries,
        retry_wait=args.retry_wait,
        timeout=args.timeout,
        status_path=args.status_file,
    )

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
