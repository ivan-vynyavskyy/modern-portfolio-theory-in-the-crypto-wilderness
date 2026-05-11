#!/usr/bin/env python3
"""
ethereum_codec_utils.py

Purpose
-------
Shared helpers for decoding Ethereum trace payloads, normalizing addresses,
and checking ERC-20 bytecode selectors.

What it does
------------
- Converts raw parquet/CSV representations (hex strings, bytes, ints) into
  normalized 0x-prefixed addresses and hex bytecode strings.
- Checks deployed bytecode for the presence of ERC-20 method selectors and
  event topic prefixes.
- Loads existing address sets from TSV files for deduplication.

Inputs
------
- Raw address/bytecode values from Parquet or CSV columns

Outputs
-------
- Normalized hex addresses and boolean selector-presence checks

Notes
-----
- Functions are intentionally low-level to avoid web3 as a hard dependency;
  an optional checksum_func can be passed where needed.
"""

import ast
import math
import os
from typing import Iterable, Optional, Set

# ERC-20 selectors/events used for static bytecode checks
REQUIRED_METHOD_SELECTORS = {
    # totalSupply()
    "18160ddd",
    # balanceOf(address)
    "70a08231",
    # transfer(address,uint256)
    "a9059cbb",
    # transferFrom(address,address,uint256)
    "23b872dd",
    # approve(address,uint256)
    "095ea7b3",
    # allowance(address,address)
    "dd62ed3e",
}

REQUIRED_EVENT_PREFIXES = {
    # Transfer(address,address,uint256)
    "ddf252ad",
    # Approval(address,address,uint256)
    "8c5be1e5",
}


def is_nan_like(value) -> bool:
    """Return True when value behaves like NaN for our parquet inputs."""
    return isinstance(value, float) and math.isnan(value)


def to_bytes_any(value) -> bytes:
    """Convert parquet / CSV representations (hex strings, repr bytes, ints) into raw bytes."""
    if value is None or is_nan_like(value):
        return b""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        s = value.strip()
        if s.startswith(("0x", "0X")):
            try:
                return bytes.fromhex(s[2:])
            except ValueError:
                return b""
        try:
            evaluated = ast.literal_eval(s)
            if isinstance(evaluated, (bytes, bytearray)):
                return bytes(evaluated)
        except Exception:
            pass
        try:
            no_space = "".join(s.split())
            if all(c in "0123456789abcdefABCDEF" for c in no_space) and len(no_space) % 2 == 0:
                return bytes.fromhex(no_space)
        except Exception:
            pass
        return b""
    if isinstance(value, int):
        try:
            return value.to_bytes((value.bit_length() + 7) // 8, "big")
        except Exception:
            return b""
    return b""


def to_hex_address(value, *, checksum_func=None) -> Optional[str]:
    """
    Convert supported representations to a 0x-prefixed hex address.
    Optionally accepts a checksum_func (e.g., Web3.to_checksum_address).
    """
    raw = to_bytes_any(value)
    if len(raw) == 0:
        return None
    if len(raw) < 20:
        return None
    if len(raw) > 20:
        raw = raw[-20:]
    addr = "0x" + raw.hex()
    if checksum_func is not None:
        try:
            return checksum_func(addr)
        except Exception:
            return None
    return addr


def to_hex_bytes(value) -> str:
    """Convert supported representations to a lowercase hex string (without 0x)."""
    return to_bytes_any(value).hex()


def normalize_hex_address(value) -> str:
    """Return a lowercase 0x-address when possible; otherwise return an empty string."""
    addr = to_hex_address(value)
    return addr.lower() if addr else ""


def hex_contains_all(haystack_hex: str, needles: Iterable[str]) -> bool:
    """Return True if every selector/event prefix is present in haystack_hex."""
    hay = haystack_hex.lower()
    return all(n.lower() in hay for n in needles)


def erc20_selectors_present(code_hex: str, require_events: bool = True) -> bool:
    """Return True if ERC-20 selectors (and optionally events) are present in code."""
    if not code_hex or code_hex in ("0x", ""):
        return False
    if not hex_contains_all(code_hex, REQUIRED_METHOD_SELECTORS):
        return False
    if require_events and not hex_contains_all(code_hex, REQUIRED_EVENT_PREFIXES):
        return False
    return True


def load_existing_addresses(tsv_path: str) -> Set[str]:
    """Read first-column addresses from a TSV/CSV file and return them lowercase."""
    seen: Set[str] = set()
    if os.path.exists(tsv_path):
        try:
            with open(tsv_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    parts = line.split("\t", 1)
                    addr = parts[0].strip()
                    if addr.startswith("0x") and len(addr) == 42:
                        seen.add(addr.lower())
        except Exception:
            pass
    return seen


__all__ = [
    "erc20_selectors_present",
    "hex_contains_all",
    "is_nan_like",
    "load_existing_addresses",
    "normalize_hex_address",
    "REQUIRED_EVENT_PREFIXES",
    "REQUIRED_METHOD_SELECTORS",
    "to_bytes_any",
    "to_hex_address",
    "to_hex_bytes",
]
