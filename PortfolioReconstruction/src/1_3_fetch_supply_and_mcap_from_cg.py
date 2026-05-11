#!/usr/bin/env python3
"""
1_3_fetch_supply_and_mcap_from_cg.py

Purpose
-------
Fetch circulating supply, FDV, and market cap metrics from CoinGecko Pro for ERC-20 tokens in a CSV plus native ETH.

What it does
------------
- Reads and filters input rows to eligible ERC-20 contracts.
- Queries CoinGecko Pro for supply and market data per contract and for ETH.
- Writes metrics row-by-row to an output CSV while printing progress.

Inputs
------
- --in CSV (columns: symbol,name,contract_address,is_erc_20,creation_block,price_data_status,trading_volume_status)
- COINGECKO_API_KEY (env var; optional if --api-key provided)

Outputs
-------
- --out CSV (columns: contract_address,symbol,name,circulating_supply,total_supply,max_supply,price_usd,market_cap_usd,fdv_usd,cg_id,cg_data_updated_at,error)

CLI
---
Examples:
  $ python 1_3_fetch_supply_and_mcap_from_cg.py --in tokens.csv --out tokens_cg_metrics.csv
  $ python 1_3_fetch_supply_and_mcap_from_cg.py --in tokens.csv --out tokens_cg_metrics.csv --api-key YOUR_KEY

Notes
-----
- Filters require is_erc_20 == true, price_data_status == success, and trading_volume_status == good.
- Contract addresses are normalized to lowercase 0x... and must be 42 chars to be queried.
- ETH is always queried and written as a separate row with an empty contract_address.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

# Ensure helpers/ is importable when running from src/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.ethereum_codec_utils import normalize_hex_address
from utils.log_utils import log, log_exc

# ----------------------------- Constants -----------------------------
COINGECKO_BASE_PRO = "https://pro-api.coingecko.com/api/v3"
CONTRACT_URL = COINGECKO_BASE_PRO + "/coins/ethereum/contract/{addr}"
ETH_URL = COINGECKO_BASE_PRO + "/coins/ethereum"

REQUIRED_INPUT_COLUMNS = {
    "symbol",
    "name",
    "contract_address",
    "is_erc_20",
    "creation_block",
    "price_data_status",
    "trading_volume_status",
}

OUTPUT_FIELDS = [
    "contract_address",
    "symbol",
    "name",
    "circulating_supply",
    "total_supply",
    "max_supply",
    "price_usd",
    "market_cap_usd",
    "fdv_usd",
    "cg_id",
    "cg_data_updated_at",
    "error",
]


# ------------------------------ Helpers ------------------------------
def is_success(value: str) -> bool:
    """Return True if a status string equals 'success'."""
    return str(value).strip().lower() == "success"


def is_good_volume(value: str) -> bool:
    """Return True if a volume status string equals 'good'."""
    return str(value).strip().lower() == "good"


def cg_get_contract(
    session: requests.Session, api_key: str, address: str, timeout: int = 30
) -> Dict[str, Any]:
    """Fetch CoinGecko data for an ERC-20 contract address."""
    url = CONTRACT_URL.format(addr=address)
    headers = {"x-cg-pro-api-key": api_key}
    response = session.get(url, headers=headers, timeout=timeout)
    if response.status_code == 404:
        return {"_not_found": True}
    response.raise_for_status()
    return response.json()


def cg_get_eth(session: requests.Session, api_key: str, timeout: int = 30) -> Dict[str, Any]:
    """Fetch CoinGecko data for native Ethereum."""
    headers = {"x-cg-pro-api-key": api_key}
    response = session.get(ETH_URL, headers=headers, timeout=timeout)
    if response.status_code == 404:
        return {"_not_found": True}
    response.raise_for_status()
    return response.json()


def extract_market_fields(data: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Extract market fields from a CoinGecko /coins/... response."""
    market_data = data.get("market_data") or {}

    circulating = market_data.get("circulating_supply")
    total_supply = market_data.get("total_supply")
    max_supply = market_data.get("max_supply")

    price = (market_data.get("current_price") or {}).get("usd")
    market_cap = (market_data.get("market_cap") or {}).get("usd")

    fdv_map = market_data.get("fully_diluted_valuation") or {}
    fdv_usd = fdv_map.get("usd") if isinstance(fdv_map, dict) else None

    updated_at = data.get("last_updated") or market_data.get("last_updated_at")

    # fallback compute mcap if missing but circ & price exist
    if market_cap is None and circulating is not None and price is not None:
        try:
            market_cap = float(circulating) * float(price)
        except Exception:
            pass

    # fallback compute fdv if missing but max_supply & price exist
    if fdv_usd is None and max_supply is not None and price is not None:
        try:
            fdv_usd = float(max_supply) * float(price)
        except Exception:
            pass

    return circulating, total_supply, max_supply, price, market_cap, fdv_usd, updated_at


def format_float(value: Any, fmt: str) -> str:
    """Format a numeric value using a format spec or return empty string."""
    if value is None:
        return ""
    return f"{float(value):{fmt}}"


def read_filtered_rows(
    in_path: Path,
) -> Tuple[List[Dict[str, str]], int, int, int]:
    """Read and filter input rows, returning rows and summary counts."""
    rows: List[Dict[str, str]] = []
    total_tokens = 0
    matches_all_filters = 0

    with in_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_INPUT_COLUMNS - set(reader.fieldnames or [])
        if missing:
            log_exc(
                "Input CSV missing required columns: "
                + ", ".join(sorted(missing))
            )
            return [], 0, 0, 1

        for row in reader:
            total_tokens += 1

            erc_ok = (row.get("is_erc_20", "") or "").strip().lower() == "true"
            price_ok = is_success(row.get("price_data_status", ""))
            volume_ok = is_good_volume(row.get("trading_volume_status", ""))

            meets_filters = erc_ok and price_ok and volume_ok
            if meets_filters:
                matches_all_filters += 1

                contract_address = normalize_hex_address(row.get("contract_address", ""))
                if not contract_address or len(contract_address) != 42:
                    # Invalid or missing address -> can't query CoinGecko, skip from requests.
                    continue

                rows.append(
                    {
                        "symbol_in": row.get("symbol", ""),
                        "name_in": row.get("name", ""),
                        "contract_address": contract_address,
                    }
                )

    return rows, total_tokens, matches_all_filters, 0


def fetch_eth_row(session: requests.Session, api_key: str) -> Dict[str, str]:
    """Fetch and format the native ETH output row."""
    eth_err = ""
    eth_circ = None
    eth_total = None
    eth_max = None
    eth_price = None
    eth_mcap = None
    eth_fdv = None
    eth_cg_id = None
    eth_updated = None

    try:
        eth_data = cg_get_eth(session, api_key)
        if eth_data.get("_not_found"):
            eth_err = "not_found_on_coingecko"
        else:
            eth_cg_id = eth_data.get("id")
            (
                eth_circ,
                eth_total,
                eth_max,
                eth_price,
                eth_mcap,
                eth_fdv,
                eth_updated,
            ) = extract_market_fields(eth_data)
    except requests.HTTPError as exc:
        eth_err = f"http_error_{exc.response.status_code}"
    except requests.RequestException as exc:
        eth_err = f"request_error_{type(exc).__name__}"
    except Exception as exc:
        eth_err = f"unexpected_{type(exc).__name__}"

    return {
        "contract_address": "",  # native
        "symbol": "ETH",
        "name": "Ethereum",
        "circulating_supply": format_float(eth_circ, ".8f"),
        "total_supply": format_float(eth_total, ".8f"),
        "max_supply": format_float(eth_max, ".8f"),
        "price_usd": format_float(eth_price, ".8f"),
        "market_cap_usd": format_float(eth_mcap, ".2f"),
        "fdv_usd": format_float(eth_fdv, ".2f"),
        "cg_id": eth_cg_id or "",
        "cg_data_updated_at": eth_updated or "",
        "error": eth_err,
    }


def fetch_contract_row(
    session: requests.Session, api_key: str, address: str, symbol_in: str, name_in: str
) -> Dict[str, str]:
    """Fetch and format a token output row for a contract address."""
    symbol = symbol_in
    name = name_in
    circ = None
    total_supply = None
    max_supply = None
    price = None
    market_cap = None
    fdv = None
    cg_id = None
    updated_at = None
    err = ""

    try:
        data = cg_get_contract(session, api_key, address)
        if data.get("_not_found"):
            err = "not_found_on_coingecko"
        else:
            if not symbol:
                symbol = (data.get("symbol") or "").upper()
            if not name:
                name = data.get("name") or ""
            cg_id = data.get("id")

            (
                circ,
                total_supply,
                max_supply,
                price,
                market_cap,
                fdv,
                updated_at,
            ) = extract_market_fields(data)
    except requests.HTTPError as exc:
        err = f"http_error_{exc.response.status_code}"
    except requests.RequestException as exc:
        err = f"request_error_{type(exc).__name__}"
    except Exception as exc:
        err = f"unexpected_{type(exc).__name__}"

    return {
        "contract_address": address,
        "symbol": symbol or "",
        "name": name or "",
        "circulating_supply": format_float(circ, ".8f"),
        "total_supply": format_float(total_supply, ".8f"),
        "max_supply": format_float(max_supply, ".8f"),
        "price_usd": format_float(price, ".8f"),
        "market_cap_usd": format_float(market_cap, ".2f"),
        "fdv_usd": format_float(fdv, ".2f"),
        "cg_id": cg_id or "",
        "cg_data_updated_at": updated_at or "",
        "error": err,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fetch supply, FDV, and market cap (USD) from CoinGecko Pro "
            "for ERC-20s listed in a CSV + native ETH."
        )
    )
    parser.add_argument("--in", dest="in_csv", required=True, help="Input CSV with required columns.")
    parser.add_argument("--out", dest="out_csv", default="tokens_cg_metrics.csv", help="Output CSV path.")
    parser.add_argument("--sleep", type=float, default=0.8, help="Seconds to sleep between requests.")
    parser.add_argument("--max", type=int, default=None, help="Optional max number of tokens to process.")
    parser.add_argument("--api-key", default=None, help="CoinGecko API key (Pro/Demo).")
    return parser.parse_args(argv)


def main() -> int:
    """Run the CLI entrypoint."""
    args = parse_args()

    api_key = args.api_key or os.getenv("COINGECKO_API_KEY")
    if not api_key:
        log_exc("Provide --api-key or set COINGECKO_API_KEY.")
        return 1

    rows, total_tokens, matches_all_filters, exit_code = read_filtered_rows(Path(args.in_csv))
    if exit_code != 0:
        return exit_code

    not_matching_filters = total_tokens - matches_all_filters
    log("Input summary:")
    log(f"  Total tokens (rows) in input: {total_tokens}")
    log(
        "  Tokens matching filters "
        "(is_erc_20 == true, price_data_status == success, trading_volume_status == good): "
        f"{matches_all_filters}"
    )
    log(f"  Tokens NOT matching all filters: {not_matching_filters}")
    log(f"  Tokens that will be queried on CoinGecko (valid ERC-20 contract_address): {len(rows)}")

    if args.max is not None:
        rows = rows[: args.max]

    if not rows:
        log("No rows with valid contract_address matched the filters. Nothing to do (besides ETH).")

    session = requests.Session()

    written_rows = 0
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        f_out.flush()

        # ---- First: ETH row ----
        eth_row = fetch_eth_row(session, api_key)
        writer.writerow(eth_row)
        f_out.flush()
        written_rows += 1

        log(
            f"[ETH] native | ETH | Ethereum | "
            f"circ={eth_row['circulating_supply'] or '-'} | "
            f"total={eth_row['total_supply'] or '-'} | "
            f"max={eth_row['max_supply'] or '-'} | "
            f"px_usd={eth_row['price_usd'] or '-'} | "
            f"mcap_usd={eth_row['market_cap_usd'] or '-'} | "
            f"fdv_usd={eth_row['fdv_usd'] or '-'} | "
            f"{'OK' if not eth_row['error'] else 'ERR: ' + eth_row['error']}"
        )

        if rows and args.sleep > 0:
            time.sleep(args.sleep)

        # ---- Then: token rows ----
        total_tokens_to_query = len(rows)
        for i, row in enumerate(rows, 1):
            addr = row["contract_address"]
            sym_in = row["symbol_in"]
            name_in = row["name_in"]

            out_row = fetch_contract_row(session, api_key, addr, sym_in, name_in)

            writer.writerow(out_row)
            f_out.flush()
            written_rows += 1

            log(
                f"[{i}/{total_tokens_to_query}] {addr} "
                f"{(out_row['symbol'] or '').upper()} | {out_row['name'] or ''} | "
                f"circ={out_row['circulating_supply'] or '-'} | "
                f"total={out_row['total_supply'] or '-'} | "
                f"max={out_row['max_supply'] or '-'} | "
                f"px_usd={out_row['price_usd'] or '-'} | "
                f"mcap_usd={out_row['market_cap_usd'] or '-'} | "
                f"fdv_usd={out_row['fdv_usd'] or '-'} | "
                f"{'OK' if not out_row['error'] else 'ERR: ' + out_row['error']}"
            )

            if i < total_tokens_to_query and args.sleep > 0:
                time.sleep(args.sleep)

    log(f"Done. Wrote {written_rows} rows to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
