"""
token_filters.py

Purpose
-------
Load and filter ERC-20 token allow-lists from the pipeline's status CSV.

What it does
------------
- Reads a status CSV and applies the standard inclusion filters:
  is_erc_20 == "true", price_data_status == "success",
  trading_volume_status == "good", fdv_mcap_lt_eth != "false".
- Returns a set of normalized lowercase 0x-prefixed contract addresses.

Inputs
------
- Status CSV path (columns: contract_address, is_erc_20, price_data_status,
  trading_volume_status, fdv_mcap_lt_eth)

Outputs
-------
- Set of lowercase 0x-prefixed address strings

Notes
-----
- Column values are matched exactly against the values produced by
  earlier pipeline steps (0_3, 1_1, 1_2, 1_4).
"""

import csv
from typing import Callable, Optional, Set

from utils.ethereum_codec_utils import normalize_hex_address


def load_token_allow_list(
    csv_path: str,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Set[str]:
    """Load ERC-20 allow-list addresses from a status CSV.

    Returns lowercase 0x-prefixed address strings that pass all filters.
    """
    allow: Set[str] = set()

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if (row.get("is_erc_20", "") or "").strip().lower() != "true":
                continue
            if (row.get("price_data_status", "") or "").strip().lower() != "success":
                continue
            if (row.get("trading_volume_status", "") or "").strip().lower() != "good":
                continue
            if (row.get("fdv_mcap_lt_eth", "") or "").strip().upper() == "FALSE":
                continue
            addr = normalize_hex_address(row.get("contract_address", ""))
            if addr and len(addr) == 42:
                allow.add(addr)

    if log_fn:
        log_fn(
            f"allow-list: {len(allow):,} ERC-20 tokens with price, "
            "volume and FDV/MCAP filters loaded"
        )
    return allow
