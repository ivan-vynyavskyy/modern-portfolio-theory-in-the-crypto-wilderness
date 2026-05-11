#!/usr/bin/env python3
"""
0_4_resolve_proxies_and_verify_erc20.py

Purpose
-------
Discover ERC-20 token contracts and resolve proxy implementations via RPC,
then verify them and write a TSV summary.

What it does
------------
- Builds candidate contract addresses from a status CSV, parquet traces, or a TSV list.
- Detects common proxy patterns and resolves implementation/beacon targets.
- Verifies ERC-20 presence with static selectors and runtime calls, then records metadata.

Inputs
------
- --status-csv (CSV; columns: contract_address, is_erc_20, creation_block)
- --traces-dir (parquet; columns: block_id, trace_type, status, error, to_address, output)
- --addresses-tsv (TSV; first column is address)
- WEB3_RPC_URL (env var) or --rpc (HTTP RPC endpoint)

Outputs
-------
- --out-tokens (TSV; columns: address, block_id, proxy_type, implementation, beacon,
  verified_static, verified_runtime, symbol, decimals, name)

CLI
---
Examples:
  $ python 0_4_resolve_proxies_and_verify_erc20.py --traces-dir /data/traces --rpc https://mainnet.infura.io/v3/<key>
  $ python 0_4_resolve_proxies_and_verify_erc20.py --addresses-tsv seeds.tsv --rpc $WEB3_RPC_URL
  $ python 0_4_resolve_proxies_and_verify_erc20.py --traces-dir traces --engine fastparquet --block-mode deploy

Notes
-----
- When --status-csv is provided, only rows with is_erc_20 == 'false' are processed.
- For duplicate addresses, the earliest non-zero creation_block is kept.
- --block-mode=deploy uses the contract's deploy block when available; otherwise latest.
- Results are appended to --out-tokens, deduped by first-column address.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
from web3 import Web3

# Ensure helpers/ is importable when running from src/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.ethereum_codec_utils import (
    erc20_selectors_present,
    load_existing_addresses,
    to_hex_address as codec_to_hex_address,
)
from utils.log_utils import log, log_exc

# ----------------------------- Constants -----------------------------

PARQUET_COLUMNS = [
    "block_id",
    "tx_hash",
    "transaction_index",
    "trace_index",
    "trace_type",
    "error",
    "status",
    "to_address",
    "output",
]

# EIP-1967 storage slots (keccak256("eip1967.proxy.implementation") - 1, etc.)
EIP1967_IMPL_SLOT = Web3.to_int(
    hexstr="0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
)
EIP1967_BEACON_SLOT = Web3.to_int(
    hexstr="0xa3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50"
)
# Old ZeppelinOS / OpenZeppelin "unstructured storage" slot (used by FiatTokenProxy / USDC-style)
ZEPPELINOS_IMPL_SLOT = Web3.to_int(Web3.keccak(text="org.zeppelinos.proxy.implementation"))

# EIP-1167 minimal proxy pattern
EIP1167_RUNTIME_PREFIX = "363d3d373d3d3d363d73"
EIP1167_RUNTIME_SUFFIX = "5af43d82803e903d91602b57fd5bf3"

ZERO_ADDRESS = "0x" + "00" * 20
RPC_TIMEOUT_SECONDS = 30

# ----------------------------- General helpers -----------------------------

def to_hex_address(value) -> Optional[str]:
    """Decode an address-like value and return a checksum address."""
    return codec_to_hex_address(value, checksum_func=Web3.to_checksum_address)


def append_result(tsv_path: Path, row: Sequence[object]) -> None:
    """Append a TSV row, creating parent folders if needed."""
    tsv_path = Path(tsv_path)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join("" if x is None else str(x) for x in row) + "\n")
        handle.flush()


def walk_parquet_files(root: Path) -> Iterator[Path]:
    """Yield parquet files under a directory in a stable per-folder order."""
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename.lower().endswith(".parquet"):
                yield Path(dirpath) / filename


# ----------------------------- Web3 helpers -----------------------------

def function_selector(signature: str) -> bytes:
    """Return the 4-byte selector for a function signature."""
    return Web3.keccak(text=signature)[:4]


SEL_TOTAL_SUPPLY = function_selector("totalSupply()")
SEL_BALANCE_OF = function_selector("balanceOf(address)")
SEL_ALLOWANCE = function_selector("allowance(address,address)")
SEL_NAME = function_selector("name()")
SEL_SYMBOL = function_selector("symbol()")
SEL_DECIMALS = function_selector("decimals()")
SEL_BEACON_IMPL = function_selector("implementation()")  # for Beacon


def pad32(value: bytes) -> bytes:
    """Left-pad a byte string to 32 bytes."""
    return value.rjust(32, b"\x00")


def enc_address(addr: str) -> bytes:
    """ABI-encode a single address as 32 bytes."""
    return pad32(Web3.to_bytes(hexstr=addr)[-20:])


def enc_two_addresses(first: str, second: str) -> bytes:
    """ABI-encode two addresses as 64 bytes."""
    return enc_address(first) + enc_address(second)


def eth_call_bytes(w3: Web3, to: str, data: bytes, block_ref) -> Optional[bytes]:
    """Call eth_call and return raw bytes or None on failure."""
    try:
        res = w3.eth.call({"to": to, "data": data}, block_identifier=block_ref)
        if res is None:
            return None
        return bytes(res)
    except Exception:
        return None


def decode_dynamic_string_or_bytes32(ret: bytes) -> Optional[str]:
    """Decode a dynamic ABI string or a bytes32 fallback to UTF-8."""
    # Try dynamic string (ABI): offset(32) | length(32) | bytes(length)
    try:
        if ret and len(ret) >= 64:
            off = int.from_bytes(ret[:32], "big")
            if off == 32 and len(ret) >= 64:
                length = int.from_bytes(ret[32:64], "big")
                start = 64
                end = start + length
                if end <= len(ret):
                    return ret[start:end].decode("utf-8", errors="ignore")
    except Exception:
        pass
    # Try bytes32-as-string fallback
    try:
        if ret and len(ret) >= 32:
            raw = ret[:32].rstrip(b"\x00")
            if raw:
                return raw.decode("utf-8", errors="ignore")
    except Exception:
        pass
    return None


def is_eip1167_minimal_proxy(bytecode_hex: str) -> Tuple[bool, Optional[str]]:
    """Detect an EIP-1167 minimal proxy and return (is_proxy, implementation)."""
    hex_str = bytecode_hex.lower()
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]
    # Common exact pattern
    if hex_str.startswith(EIP1167_RUNTIME_PREFIX) and hex_str.endswith(EIP1167_RUNTIME_SUFFIX):
        mid = hex_str[len(EIP1167_RUNTIME_PREFIX) : -len(EIP1167_RUNTIME_SUFFIX)]
        if len(mid) >= 40:
            return True, Web3.to_checksum_address("0x" + mid[:40])
        return True, None
    # Tolerant regex (compiler metadata appended)
    match = re.search(
        r"(?:^|)363d3d373d3d3d363d73([0-9a-f]{40})5af43d82803e903d91602b57fd5bf3(?:[0-9a-f]*)$",
        hex_str,
    )
    if match:
        return True, Web3.to_checksum_address("0x" + match.group(1))
    return False, None


def _read_storage_address(w3: Web3, proxy: str, slot: int, block_ref) -> Optional[str]:
    """Read an address from a storage slot, returning a checksum address or None."""
    try:
        raw = w3.eth.get_storage_at(proxy, slot, block_identifier=block_ref)
        if raw and len(raw) == 32 and any(raw):
            addr = "0x" + raw[-20:].hex()
            if addr != ZERO_ADDRESS:
                return Web3.to_checksum_address(addr)
    except Exception:
        pass
    return None


def read_eip1967_impl(w3: Web3, proxy: str, block_ref) -> Optional[str]:
    """Read the EIP-1967 implementation slot."""
    return _read_storage_address(w3, proxy, EIP1967_IMPL_SLOT, block_ref)


def read_eip1967_beacon(w3: Web3, proxy: str, block_ref) -> Optional[str]:
    """Read the EIP-1967 beacon slot."""
    return _read_storage_address(w3, proxy, EIP1967_BEACON_SLOT, block_ref)


def read_zeppelinos_impl(w3: Web3, proxy: str, block_ref) -> Optional[str]:
    """Read the ZeppelinOS unstructured storage implementation slot."""
    return _read_storage_address(w3, proxy, ZEPPELINOS_IMPL_SLOT, block_ref)


def call_beacon_implementation(w3: Web3, beacon: str, block_ref) -> Optional[str]:
    """Call implementation() on a beacon and return the implementation address."""
    out = eth_call_bytes(w3, beacon, SEL_BEACON_IMPL, block_ref)
    if out and len(out) >= 32:
        addr = "0x" + out[-20:].hex()
        if addr != ZERO_ADDRESS:
            return Web3.to_checksum_address(addr)
    return None


def code_at(w3: Web3, addr: str, block_ref) -> str:
    """Return contract code at a block reference as a hex string."""
    try:
        return w3.eth.get_code(addr, block_identifier=block_ref).hex()
    except Exception:
        return "0x"


def call_exposed_implementation(w3: Web3, proxy: str, block_ref) -> Optional[str]:
    """Call implementation() directly on a proxy that exposes it."""
    out = eth_call_bytes(w3, proxy, SEL_BEACON_IMPL, block_ref)
    if out and len(out) >= 32:
        addr = "0x" + out[-20:].hex()
        if addr != ZERO_ADDRESS:
            return Web3.to_checksum_address(addr)
    return None


# ----------------------------- ERC-20 verification -----------------------------

def erc20_static_selectors_present(code_hex: str, require_events: bool) -> bool:
    """Return True if ERC-20 selectors (and optionally events) are present in code."""
    return erc20_selectors_present(code_hex, require_events=require_events)


def erc20_runtime_probe(w3: Web3, target: str, block_ref) -> Tuple[bool, Dict[str, Optional[str]]]:
    """Probe runtime ERC-20 views and return (ok, metadata)."""
    meta: Dict[str, Optional[str]] = {"symbol": None, "name": None, "decimals": None}

    # totalSupply()
    total_supply = eth_call_bytes(w3, target, SEL_TOTAL_SUPPLY, block_ref)
    if not total_supply or len(total_supply) < 32:
        return False, meta

    # balanceOf(0x...01)
    dummy = Web3.to_checksum_address("0x" + "00" * 19 + "01")
    balance = eth_call_bytes(w3, target, SEL_BALANCE_OF + enc_address(dummy), block_ref)
    if not balance or len(balance) < 32:
        return False, meta

    # allowance(0x...01, 0x...02)
    dummy2 = Web3.to_checksum_address("0x" + "00" * 19 + "02")
    allowance = eth_call_bytes(w3, target, SEL_ALLOWANCE + enc_two_addresses(dummy, dummy2), block_ref)
    if not allowance or len(allowance) < 32:
        return False, meta

    # Optional metadata
    symbol_raw = eth_call_bytes(w3, target, SEL_SYMBOL, block_ref)
    if symbol_raw:
        meta["symbol"] = decode_dynamic_string_or_bytes32(symbol_raw)
    name_raw = eth_call_bytes(w3, target, SEL_NAME, block_ref)
    if name_raw:
        meta["name"] = decode_dynamic_string_or_bytes32(name_raw)
    decimals_raw = eth_call_bytes(w3, target, SEL_DECIMALS, block_ref)
    if decimals_raw and len(decimals_raw) >= 32:
        meta["decimals"] = str(int.from_bytes(decimals_raw[-32:], "big"))

    return True, meta


def _apply_runtime_verification(
    w3: Web3, result: Dict[str, Optional[str]], target: str, block_ref
) -> None:
    """Run runtime verification and update result dict in-place."""
    ok, meta = erc20_runtime_probe(w3, target, block_ref)
    if ok:
        result["verified_runtime"] = "Y"
        result["symbol"], result["decimals"], result["name"] = (
            meta["symbol"],
            meta["decimals"],
            meta["name"],
        )


def _apply_static_verification(result: Dict[str, Optional[str]], code_hex: str) -> None:
    """Run static verification and update result dict in-place."""
    # Legacy behavior: events are not required in static checks here.
    if erc20_static_selectors_present(code_hex, require_events=False):
        result["verified_static"] = "Y"


# ----------------------------- Main scanning logic -----------------------------

def collect_created_contracts_from_parquet(parquet_path: Path, engine: str) -> pd.DataFrame:
    """Load parquet traces and return successful CREATE/CREATE2 contracts."""
    try:
        df = pd.read_parquet(parquet_path, columns=PARQUET_COLUMNS, engine=engine)
    except Exception as exc:
        log_exc(f"[warn] Cannot read {parquet_path}: {exc}")
        return pd.DataFrame(columns=["block_id", "to_address", "output", "trace_type", "status", "error"])

    # Accept create + create2
    trace_type = df["trace_type"].astype(str).str.lower()
    mask_ok = trace_type.isin(["create", "create2"])
    if "status" in df.columns:
        mask_ok &= (df["status"] == 1) | (df["status"] == True)
    if "error" in df.columns:
        mask_ok &= (df["error"].isna()) | (df["error"] == "") | (df["error"].isnull())

    sub = df.loc[mask_ok, ["block_id", "to_address", "output"]].copy()
    sub["to_address"] = sub["to_address"].apply(to_hex_address)
    return sub[sub["to_address"].notna()]


def load_addresses_tsv(path: Path) -> List[str]:
    """Load a TSV file and return checksum addresses from the first column."""
    out: List[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            addr = line.strip().split("\t")[0]
            if addr.startswith("0x") and len(addr) == 42:
                out.append(Web3.to_checksum_address(addr))
    return out


def classify_contract(
    w3: Web3,
    addr: str,
    block_id: Optional[int],
    block_mode: str,
    require_events: bool,
) -> Dict[str, Optional[str]]:
    """Classify a contract address, resolving proxies and ERC-20 verification."""
    block_ref = "latest" if block_mode == "latest" else block_id
    result: Dict[str, Optional[str]] = {
        "address": addr,
        "block_id": block_id,
        "proxy_type": "undetermined",
        "implementation": None,
        "beacon": None,
        "verified_static": "N",
        "verified_runtime": "N",
        "symbol": None,
        "decimals": None,
        "name": None,
    }

    code = code_at(w3, addr, block_ref)
    if code in ("0x", ""):
        return result

    # 1) EIP-1167 minimal proxy
    is_min_proxy, impl_addr = is_eip1167_minimal_proxy(code)
    if is_min_proxy:
        result["proxy_type"] = "minimal_proxy"
        if impl_addr:
            result["implementation"] = impl_addr
            _apply_static_verification(result, code_at(w3, impl_addr, block_ref))
        _apply_runtime_verification(w3, result, addr, block_ref)
        return result

    # 2) EIP-1967 proxy (Transparent/UUPS)
    impl1967 = read_eip1967_impl(w3, addr, block_ref)
    if impl1967:
        result["proxy_type"] = "eip1967_proxy"
        result["implementation"] = impl1967
        _apply_static_verification(result, code_at(w3, impl1967, block_ref))
        _apply_runtime_verification(w3, result, addr, block_ref)
        return result

    # 2.5) Old unstructured-storage proxy (USDC / FiatTokenProxy style)
    impl_zep = read_zeppelinos_impl(w3, addr, block_ref)
    if impl_zep:
        result["proxy_type"] = "zeppelinos_unstructured_proxy"
        result["implementation"] = impl_zep
        _apply_static_verification(result, code_at(w3, impl_zep, block_ref))
        _apply_runtime_verification(w3, result, addr, block_ref)
        return result

    # 3) Beacon proxy
    beacon = read_eip1967_beacon(w3, addr, block_ref)
    if beacon:
        result["proxy_type"] = "beacon_proxy"
        result["beacon"] = beacon
        impl_from_beacon = call_beacon_implementation(w3, beacon, block_ref)
        if impl_from_beacon:
            result["implementation"] = impl_from_beacon
            _apply_static_verification(result, code_at(w3, impl_from_beacon, block_ref))
        _apply_runtime_verification(w3, result, addr, block_ref)
        return result

    # 3.5) Proxy exposes implementation() directly (USDC / AdminUpgradeabilityProxy style)
    impl_exposed = call_exposed_implementation(w3, addr, block_ref)
    if impl_exposed:
        result["proxy_type"] = "exposes_implementation"
        result["implementation"] = impl_exposed
        _apply_static_verification(result, code_at(w3, impl_exposed, block_ref))
        _apply_runtime_verification(w3, result, addr, block_ref)
        return result

    # 4) Fallback (proxy pattern undetermined or direct implementation)
    _apply_static_verification(result, code)
    _apply_runtime_verification(w3, result, addr, block_ref)
    return result


def _read_status_csv(path: Path) -> pd.DataFrame:
    """Read the status CSV with optional creation_block column."""
    try:
        return pd.read_csv(
            path,
            dtype=str,
            usecols=["contract_address", "is_erc_20", "creation_block"],
            keep_default_na=False,
            na_values=[],
            low_memory=False,
        )
    except ValueError:
        df = pd.read_csv(
            path,
            dtype=str,
            usecols=["contract_address", "is_erc_20"],
            low_memory=False,
        )
        df["creation_block"] = ""
        return df


def _build_candidates_from_status(df: pd.DataFrame) -> Dict[str, int]:
    """Build candidate addresses from a status CSV dataframe."""
    candidates: Dict[str, int] = {}
    mask = df["is_erc_20"].astype(str).str.strip().str.lower() == "false"
    pending = df.loc[mask].copy()

    for _, row in pending.iterrows():
        addr = (row.get("contract_address") or "").strip()
        if not Web3.is_address(addr):
            continue
        blk_str = (row.get("creation_block") or "").strip()
        block_id = int(blk_str) if blk_str.isdigit() else 0
        key = addr.lower()
        if (key not in candidates) or (block_id and (candidates[key] == 0 or block_id < candidates[key])):
            candidates[key] = block_id

    return candidates


def _add_candidates_from_traces(candidates: Dict[str, int], traces_dir: Path, engine: str) -> None:
    """Update candidates in-place by scanning parquet traces."""
    for parquet_path in walk_parquet_files(traces_dir):
        dfp = collect_created_contracts_from_parquet(parquet_path, engine)
        for row in dfp.itertuples(index=False):
            addr = getattr(row, "to_address")
            block_id = int(getattr(row, "block_id"))
            if addr:
                key = addr.lower()
                if (key not in candidates) or (block_id < candidates[key]):
                    candidates[key] = block_id


def _add_candidates_from_tsv(candidates: Dict[str, int], addresses_tsv: Path) -> None:
    """Update candidates in-place from a TSV address list."""
    for address in load_addresses_tsv(addresses_tsv):
        candidates.setdefault(address.lower(), 0)


def _ensure_out_header(out_path: Path) -> None:
    """Ensure the output TSV has a header row."""
    if out_path.exists():
        return
    append_result(
        out_path,
        [
            "address",
            "block_id",
            "proxy_type",
            "implementation",
            "beacon",
            "verified_static",
            "verified_runtime",
            "symbol",
            "decimals",
            "name",
        ],
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Classify ERC-20 tokens (proxy detection + verification) via RPC. "
            "If --status-csv is provided, only check rows with is_erc_20 == 'false'."
        )
    )
    parser.add_argument(
        "--status-csv",
        default=None,
        help=(
            "Path to eth_tokens_status.csv (columns: symbol,name,contract_address,is_erc_20,creation_block)."
        ),
    )
    parser.add_argument(
        "--traces-dir",
        default=None,
        help="Root folder with partitioned parquet files. (Ignored if --status-csv is used.)",
    )
    parser.add_argument(
        "--addresses-tsv",
        default=None,
        help="TSV with addresses to verify (first column). (Ignored if --status-csv is used.)",
    )
    parser.add_argument("--engine", default="pyarrow", choices=["pyarrow", "fastparquet"], help="Pandas parquet engine.")
    parser.add_argument(
        "--out-tokens",
        default="erc20_tokens_resolver_results.tsv",
        help="Final token list with proxy resolution and metadata.",
    )
    parser.add_argument(
        "--rpc",
        default=os.environ.get("WEB3_RPC_URL", ""),
        help="HTTP RPC endpoint (or set WEB3_RPC_URL).",
    )
    parser.add_argument(
        "--block-mode",
        default="latest",
        choices=["latest", "deploy"],
        help="Use 'latest' or contract's deploy block for classification.",
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="(Legacy switch) if set, do not require events in static check.",
    )
    return parser


def main() -> int:
    """Run the CLI entrypoint and return an exit code."""
    args = _build_arg_parser().parse_args()

    if not args.rpc:
        log_exc("RPC endpoint required (pass --rpc or set WEB3_RPC_URL).")
        return 1

    w3 = Web3(Web3.HTTPProvider(args.rpc, request_kwargs={"timeout": RPC_TIMEOUT_SECONDS}))
    if not w3.is_connected():
        log_exc("cannot connect to RPC.")
        return 1

    require_events = not args.no_events
    _ = require_events  # retained for API compatibility (legacy behavior)

    candidates: Dict[str, int] = {}

    if args.status_csv:
        log(f"loading status CSV: {args.status_csv}")
        status_df = _read_status_csv(Path(args.status_csv))
        candidates = _build_candidates_from_status(status_df)

        if not candidates:
            log("nothing to do: no rows with is_erc_20 == 'false' found.")
            return 0
    else:
        # Fallback to original discovery flows
        if args.traces_dir:
            log(f"scanning parquet under: {args.traces_dir}")
            _add_candidates_from_traces(candidates, Path(args.traces_dir), args.engine)

        if args.addresses_tsv:
            log(f"loading addresses from: {args.addresses_tsv}")
            _add_candidates_from_tsv(candidates, Path(args.addresses_tsv))

        if not candidates:
            log_exc("No candidates found. Provide --status-csv OR --traces-dir / --addresses-tsv.")
            return 0

    log(f"total candidates: {len(candidates)}")

    out_path = Path(args.out_tokens)
    _ensure_out_header(out_path)

    already_tokens = load_existing_addresses(str(out_path))
    new_tokens = 0

    for addr_lc, block_id in candidates.items():
        addr = Web3.to_checksum_address(addr_lc)

        if addr.lower() in already_tokens:
            continue

        res = classify_contract(
            w3=w3,
            addr=addr,
            block_id=block_id if block_id != 0 else None,
            block_mode=args.block_mode,
            require_events=require_events,
        )

        append_result(
            out_path,
            [
                res["address"],
                res["block_id"] or 0,
                res["proxy_type"],
                res["implementation"] or "",
                res["beacon"] or "",
                res["verified_static"],
                res["verified_runtime"],
                res["symbol"] or "",
                res["decimals"] or "",
                res["name"] or "",
            ],
        )
        already_tokens.add(addr.lower())
        new_tokens += 1

    log(f"wrote +{new_tokens} tokens to {args.out_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
