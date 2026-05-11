#!/usr/bin/env python3
"""
4_0_portfolio_reconstruction_pipeline.py

Purpose
-------
Reconstruct per-wallet ERC-20 portfolios across multiple snapshot blocks and write
normalized holdings snapshots for downstream analytics.

What it does
------------
- Discovers whitelisted token Parquet shards and resolves snapshot blocks from a curated date list.
- Aggregates token balances per wallet with DuckDB and Python fallbacks, then enriches with decimals and prices.
- Applies optional overflow cleaning and writes holdings snapshots to a single Parquet output.

Inputs
------
- --data-root (dir; token_address=*/**/*.parquet)
- --block-to-date-csv (CSV; columns include block_number/block and date/bound_date/block_time_utc)
- --prices-dir (dir; {token}_price.csv with columns date,daily_close_usd)
- --decimals-cache (JSON; token->decimals cache)
- --token-status-csv (CSV; columns include contract_address,name,symbol,local_status)
- --rpc-url (string; Ethereum RPC endpoint)
- --input-wallets-parquet (Parquet file or directory; wallet address column)
- --clean-drop-file (optional TXT; one wallet address per line)
- --clean-suspects-json (optional JSON list; suspect token addresses)

Outputs
-------
- --output (Parquet; columns: wallet_address, block_number, total_value_usd, num_tokens, holdings)

CLI
---
Examples:
  $ python 4_0_portfolio_reconstruction_pipeline.py --data-root /data/transfers \
    --block-to-date-csv /data/block_dates.csv --prices-dir /data/prices \
    --decimals-cache /data/decimals.json --token-status-csv /data/token_status.csv \
    --rpc-url https://rpc.example --input-wallets-parquet /data/wallets.parquet \
    --output /data/portfolio_snapshots.parquet
  $ python 4_0_portfolio_reconstruction_pipeline.py --data-root /data/transfers \
    --block-to-date-csv /data/block_dates.csv --prices-dir /data/prices \
    --decimals-cache /data/decimals.json --token-status-csv /data/token_status.csv \
    --rpc-url https://rpc.example --input-wallets-parquet /data/wallets.parquet \
    --output /data/portfolio_snapshots.parquet --no-forward-fill

Notes
-----
- Snapshot dates are a fixed policy (monthly starts plus curated event neighbors).
- Balances are aggregated up to each snapshot block and stored as strings in output JSON.
- Optional overflow cleaning can drop suspect tokens and burn wallets.
"""

from __future__ import annotations

# ----------------------------- Standard library -----------------------------
import argparse
import json
import math
import os
import pathlib
import random
import re
import sys
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN, getcontext
from functools import lru_cache
from multiprocessing import Event, Process, Queue, cpu_count
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import multiprocessing as mp

# ----------------------------- Third-party -----------------------------
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from web3 import HTTPProvider, Web3

# Ensure utils/ is importable when running from src/.
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.log_utils import log, log_exc

# ----------------------------- Constants -----------------------------
ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "decimals",
     "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
]

UINT256_PREFIX = "115792089237316195423570985"  # substring probe for 2^256-1
UINT255_MAX_INT = (1 << 255) - 1                # 77 digits
MIN_IMPOSSIBLE_DIGITS = 77

DEFAULT_DROP_WALLETS = {
    "0x000000000000000000000000000000000000dead",
    "0xdead000000000000000042069420694206942069",
}

DECIMAL_CONTEXT_PRECISION = 200
getcontext().prec = DECIMAL_CONTEXT_PRECISION

WORKER_TIMEOUT_SEC = 3600
MIN_BATCH_TO_SPLIT = 4_000
SLEEP_POLL_SEC = 0.25

OUTPUT_SCHEMA = pa.schema([
    pa.field("wallet_address", pa.string()),
    pa.field("block_number", pa.int64()),
    pa.field("total_value_usd", pa.string()),
    pa.field("num_tokens", pa.int32()),
    pa.field("holdings", pa.string()),
])


# ======================= CLI / Config =======================

@dataclass
class OptimizationConfig:
    """Runtime tuning knobs for parallelism and DuckDB settings."""

    processes: int
    batch_size: int
    token_chunk_size: int = 64
    duckdb_threads: int = 2
    duckdb_memory: str = "2GB"
    forward_fill_price: bool = True
    lru_cache_size: int = 4096


# ======================= Utility =======================

def is_hex_address(s: str) -> bool:
    """Return True if s is a 0x-prefixed 40-hex address string."""
    return isinstance(s, str) and bool(ADDRESS_RE.match(s))


def normalize_eth_address(s: str) -> Optional[str]:
    """Normalize a value to lower-case 0x-address or return None."""
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    if not s.startswith("0x"):
        s = "0x" + s
    return s if is_hex_address(s) else None


def require_file(path: str | os.PathLike[str], desc: str = "file") -> None:
    """Raise FileNotFoundError if a required file path does not exist."""
    if not path:
        raise FileNotFoundError(f"{desc} not found: {path}")
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")


def dec_to_str(x: Optional[Decimal]) -> Optional[str]:
    """Convert a Decimal to a plain fixed-point string."""
    if x is None:
        return None
    return format(x, "f")


def _to_py_int(v: Any) -> int:
    """Convert a value to int with robust handling for Parquet types."""
    if v is None:
        return 0
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, float):
        if np.isnan(v):
            return 0
        return int(Decimal(str(v)))
    if isinstance(v, Decimal):
        try:
            return int(v)
        except Exception:
            return int(v.to_integral_value(rounding=ROUND_DOWN))
    if isinstance(v, (bytes, bytearray)):
        v = v.decode(errors="ignore")
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return 0
    try:
        return int(Decimal(s))
    except (InvalidOperation, ValueError):
        try:
            return int(float(s))
        except Exception:
            return 0


def load_token_status_data(csv_path: str) -> Tuple[List[str], Dict[str, str]]:
    """Load whitelist tokens and token name map from the status CSV."""
    require_file(csv_path, "Token status CSV")
    df = pd.read_csv(csv_path, dtype=str)

    required_cols = [
        "contract_address",
        "name",
        "symbol",
        "is_erc_20",
        "price_data_status",
        "local_status",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in token status CSV: {missing}")

    df["contract_address"] = df["contract_address"].apply(lambda x: str(x).strip().lower())

    def _norm_str(val: Any) -> str:
        return str(val or "").strip().lower()

    mask = (df["local_status"].apply(_norm_str) == "pass")

    whitelist_series = df.loc[mask, "contract_address"].dropna().astype(str).str.strip().str.lower()
    whitelist_tokens = list(dict.fromkeys(a for a in whitelist_series if a.startswith("0x") and len(a) == 42))

    token_name_map: Dict[str, str] = {}
    for _, row in df[["contract_address", "name", "symbol"]].iterrows():
        addr = str(row["contract_address"]).strip().lower()
        if not addr:
            continue
        name_val = row["name"]
        symbol_val = row["symbol"]
        name = "" if pd.isna(name_val) else str(name_val).strip()
        symbol = "" if pd.isna(symbol_val) else str(symbol_val).strip()
        candidate = name or symbol or "UNKNOWN"
        existing = token_name_map.get(addr)
        if existing is None:
            token_name_map[addr] = candidate
        elif existing in ("", "UNKNOWN") and candidate not in ("", "UNKNOWN"):
            token_name_map[addr] = candidate

    return whitelist_tokens, token_name_map


# ======================= Decimals cache & token names =======================

def load_decimals_cache(path: str | os.PathLike[str]) -> Dict[str, int]:
    """Load a token decimals cache from JSON or return an empty dict."""
    if not path:
        raise ValueError("DECIMALS_CACHE path is empty.")
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_decimals_cache(path: str | os.PathLike[str], data: Dict[str, int]) -> None:
    """Persist the token decimals cache to JSON."""
    if not path:
        raise ValueError("DECIMALS_CACHE path is empty.")
    p = pathlib.Path(path)
    if p.parent and p.parent != pathlib.Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def fetch_token_decimals(tokens: List[str], w3: Web3, cache_file: str) -> Dict[str, int]:
    """Fetch missing token decimals from RPC, update cache, and return map."""
    cache_data = load_decimals_cache(cache_file)
    missing = [t.lower() for t in tokens if t.lower() not in cache_data]
    if missing:
        log(f"[decimals] Fetching decimals for {len(missing)} missing tokens...")
    for token in missing:
        try:
            checksum = Web3.to_checksum_address(token)
            contract = w3.eth.contract(address=checksum, abi=ERC20_ABI)
            dec = contract.functions.decimals().call()
            if not isinstance(dec, int) or dec < 0 or dec > 255:
                log(f"[decimals] Invalid decimals {dec} for {token} — fallback 18")
                dec = 18
        except Exception as e:
            log_exc(f"[decimals] Failed {token}: {e} — fallback 18")
            dec = 18
        cache_data[token] = dec
    if missing:
        save_decimals_cache(cache_file, cache_data)
    return {t: cache_data.get(t.lower(), 18) for t in tokens}


# ======================= Overflow cleaning heuristics =======================

def _is_pure_int(s: str) -> bool:
    """Return True if s contains only digits."""
    return bool(re.fullmatch(r"\d+", s or ""))


def _digits_no_dot(s: str) -> int:
    """Count digits in a string while ignoring non-digits."""
    return len(re.sub(r"[^\d]", "", s or ""))


def _dec_gt_uint255(s: str) -> bool:
    """Return True if s is numerically greater than 2^255-1."""
    if not s:
        return False
    try:
        if _is_pure_int(s):
            return int(s) > UINT255_MAX_INT
        if _digits_no_dot(s) >= MIN_IMPOSSIBLE_DIGITS:
            return True
        return Decimal(s) > Decimal(UINT255_MAX_INT)
    except (InvalidOperation, ValueError, OverflowError):
        return False


def _is_impossible_balance_str(bal_s: str) -> bool:
    """Return True if a balance string looks like an overflow artifact."""
    return (UINT256_PREFIX in bal_s) or (_digits_no_dot(bal_s) >= MIN_IMPOSSIBLE_DIGITS) or _dec_gt_uint255(bal_s)


def _normalize_addr_lower(s: Any) -> str:
    """Normalize a value to a lower-case string for address matching."""
    return str(s or "").strip().lower()


def _to_dec_safe(x: Any) -> Optional[Decimal]:
    """Convert a value to Decimal or return None on failure."""
    if x is None:
        return None
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError, TypeError):
        return None


# ======================= Prices =======================

@lru_cache(maxsize=4096)
def _load_price_file_cached(token_lower: str, prices_dir: str) -> Optional[pd.DataFrame]:
    """Load a token price CSV and return a date-sorted DataFrame."""
    price_file = pathlib.Path(prices_dir) / f"{token_lower}_price.csv"
    if not price_file.exists():
        return None
    df = pd.read_csv(price_file, usecols=["date", "daily_close_usd"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date")


def load_price_file(token: str, prices_dir: str) -> Optional[pd.DataFrame]:
    """Return cached price data for a token or None if missing."""
    # IMPORTANT: treat returned df as read-only (do not mutate) since it is cached.
    return _load_price_file_cached(token.lower(), prices_dir)


def get_price_for_date(
    token: str,
    target_date: pd.Timestamp,
    prices_dir: str,
    forward_fill: bool = True,
) -> Optional[Decimal]:
    """Return token price for a date, optionally forward-filling earlier prices."""
    df = load_price_file(token, prices_dir)
    if df is None or df.empty:
        return None
    d = target_date.date()
    exact = df[df.date == d]
    if not exact.empty:
        return Decimal(str(exact.daily_close_usd.iloc[0]))
    if forward_fill:
        prior = df[df.date < d]
        if not prior.empty:
            return Decimal(str(prior.daily_close_usd.iloc[-1]))
    return None


def load_prices(
    tokens: List[str],
    date: pd.Timestamp,
    prices_dir: str,
    forward_fill: bool,
) -> Dict[str, Optional[Decimal]]:
    """Return a token->price map for a specific snapshot date."""
    return {t: get_price_for_date(t, date, prices_dir, forward_fill) for t in tokens}


# ======================= Token Parquet discovery =======================

def discover_token_files_parallel(
    data_root: str,
    token_whitelist: Optional[Set[str]] = None,
    max_workers: int = 4,
) -> Dict[str, List[str]]:
    """Discover token parquet files under data_root in parallel."""
    def _scan_token_dir(token_dir: pathlib.Path) -> Tuple[Optional[str], List[str]]:
        if not token_dir.is_dir():
            return None, []
        name = token_dir.name
        if "=" not in name:
            return None, []
        token = name.split("=", 1)[1].lower().strip()
        if token_whitelist and token not in token_whitelist:
            return None, []
        if not is_hex_address(token):
            return None, []
        parquet_files = list(token_dir.rglob("*.parquet"))
        return token, [str(f) for f in parquet_files]

    base_path = pathlib.Path(data_root)
    token_dirs = list(base_path.glob("token_address=*"))

    result: Dict[str, List[str]] = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_scan_token_dir, td) for td in token_dirs]
        for fut in as_completed(futs):
            token, files = fut.result()
            if token and files:
                result[token] = files

    log(f"[discover] {len(result)} tokens | {sum(len(f) for f in result.values())} files")
    return result


# ======================= Block <-> Date =======================

def _load_block_date_arrays(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load block and date arrays from a block-to-date CSV."""
    require_file(csv_path, "Block-to-date CSV")
    df = pd.read_csv(csv_path).drop_duplicates()
    block_col = next((c for c in ["block_number", "block"] if c in df.columns), None)
    date_col = next((c for c in ["date", "bound_date", "block_time_utc"] if c in df.columns), None)
    if not block_col or not date_col:
        raise ValueError(f"[blocks] CSV must include block and date columns; got: {list(df.columns)}")
    df = df[[block_col, date_col]].copy()
    df = df.rename(columns={block_col: "block_number", date_col: "bound_date"})
    df = df.sort_values("block_number").reset_index(drop=True)
    blocks = df["block_number"].to_numpy(dtype=np.int64)
    dates = pd.to_datetime(df["bound_date"]).dt.normalize().to_numpy()
    return blocks, dates


def resolve_blocks_for_dates(
    dates: List[str] | List[pd.Timestamp],
    block_to_date_csv: str,
) -> List[int]:
    """Resolve block numbers for a list of dates using a block mapping CSV."""
    if not dates:
        return []
    m = pd.read_csv(block_to_date_csv)
    block_col = next((c for c in ["block_number", "block"] if c in m.columns), None)
    if block_col is None:
        raise ValueError(
            f"Could not find a block column in {block_to_date_csv}. "
            f"Expected one of ['block_number','block']; got {list(m.columns)}"
        )
    date_col = next((c for c in ["date", "bound_date", "block_time_utc"] if c in m.columns), None)
    if date_col is None:
        raise ValueError(
            "Expected a date-like column ('date' or 'bound_date' or 'block_time_utc') "
            f"in {block_to_date_csv}. Got: {list(m.columns)}"
        )
    m[date_col] = pd.to_datetime(m[date_col], errors="coerce").dt.normalize()
    m = m.dropna(subset=[date_col, block_col])

    out_blocks: List[int] = []
    for d in dates:
        d_norm = pd.to_datetime(d, errors="coerce")
        if pd.isna(d_norm):
            log(f"[blocks] WARN could not parse date '{d}' (skip)")
            continue
        d_norm = d_norm.normalize()
        row = m.loc[m[date_col] == d_norm]
        if row.empty:
            log(f"[blocks] WARN no block for date {d_norm.date()} (skip)")
            continue
        block_num = int(row.iloc[0][block_col])
        out_blocks.append(block_num)
    if not out_blocks:
        log("[blocks] WARN no valid dates/blocks resolved.")
    return out_blocks


# ======================= Core aggregation per token =======================

def _sql_quote_path(p: str) -> str:
    """Quote a path for DuckDB SQL string literals."""
    return "'" + p.replace("'", "''") + "'"


def query_token_balances_optimized(
    token: str,
    parquet_files: List[str],
    wallets: List[str],
    snapshot_blocks: List[int],
    con: duckdb.DuckDBPyConnection,
) -> Optional[pd.DataFrame]:
    """Aggregate balances for a token using DuckDB with a Python fallback."""
    if not parquet_files:
        return None

    wallets_lc = [w.lower() for w in wallets]
    if not wallets_lc:
        return None

    snapshot_blocks_sorted = sorted(set(int(b) for b in snapshot_blocks))
    if not snapshot_blocks_sorted:
        return None

    max_block = int(snapshot_blocks_sorted[-1])
    files_str = ", ".join(_sql_quote_path(p) for p in parquet_files)

    wallet_rel = f"wallet_filter_{uuid.uuid4().hex}"
    wallets_df = pd.DataFrame({"wallet": wallets_lc})

    con.register(wallet_rel, wallets_df)
    try:
        try:
            probe_sql = f"""
            WITH t AS (
              SELECT
                lower(r.address) AS address,
                r.block,
                TRY_CAST(r.value AS DECIMAL(38,0)) AS val_dec
              FROM read_parquet([{files_str}]) AS r
              JOIN {wallet_rel} AS w
                ON lower(r.address) = w.wallet
              WHERE r.block <= {max_block}
            )
            SELECT COUNT(*) AS n_all, COUNT(val_dec) AS n_cast FROM t
            """
            n_all, n_cast = con.execute(probe_sql).fetchone()
            if n_all == 0:
                return None
            if n_cast < n_all:
                raise ValueError("Values exceed DECIMAL(38,0) – switching to Python fallback")

            cond_sums = ", ".join(
                f"SUM(CASE WHEN block <= {b} THEN val_dec ELSE 0 END) AS bal_{b}"
                for b in snapshot_blocks_sorted
            )

            sql_fast = f"""
            WITH t AS (
              SELECT
                lower(r.address) AS address,
                r.block,
                TRY_CAST(r.value AS DECIMAL(38,0)) AS val_dec
              FROM read_parquet([{files_str}]) AS r
              JOIN {wallet_rel} AS w
                ON lower(r.address) = w.wallet
              WHERE r.block <= {max_block}
            )
            SELECT
              '{token}' AS token_address,
              address AS wallet,
              {cond_sums}
            FROM t
            GROUP BY address
            """
            df = con.execute(sql_fast).fetchdf()
            if df.empty:
                return None

            for c in df.columns:
                if c.startswith("bal_"):
                    df[c] = df[c].apply(lambda x: int(x) if x is not None else 0)

            return df

        except Exception:
            sql_fallback = f"""
            SELECT
              lower(r.address) AS address,
              r.block,
              r.value
            FROM read_parquet([{files_str}]) AS r
            JOIN {wallet_rel} AS w
              ON lower(r.address) = w.wallet
            WHERE r.block <= {max_block}
            """
            raw = con.execute(sql_fallback).fetchdf()
            if raw.empty:
                return None

            raw["address"] = raw["address"].astype(str)
            raw = raw.sort_values(["address", "block"], kind="mergesort").reset_index(drop=True)
            raw["value_int"] = raw["value"].apply(_to_py_int)

            results: List[Dict[str, Any]] = []
            for wallet in wallets_lc:
                sub = raw[raw["address"] == wallet]
                if sub.empty:
                    continue
                sub = sub[["block", "value_int"]].sort_values("block", kind="mergesort")
                blocks_arr = sub["block"].to_numpy()
                vals = sub["value_int"].to_numpy()

                acc = 0
                cum = []
                for v in vals:
                    acc += int(v)
                    cum.append(acc)

                import bisect
                out = {"token_address": token, "wallet": wallet}
                for b in snapshot_blocks_sorted:
                    i = bisect.bisect_right(blocks_arr, b) - 1
                    out[f"bal_{b}"] = int(cum[i]) if i >= 0 else 0
                results.append(out)

            if not results:
                return None
            return pd.DataFrame(results)

    finally:
        try:
            con.unregister(wallet_rel)
        except Exception:
            pass


# ======================= Per-batch aggregation & enrichment =======================

def build_portfolio_dataframe(
    block: int,
    wallet_balances: Dict[str, Dict[str, int]],
    decimals_map: Dict[str, int],
    token_names: Dict[str, str],
    price_data: Dict[str, Optional[Decimal]],
    snapshot_date: pd.Timestamp,
    *,
    exclude_negative: bool = True,
) -> pd.DataFrame:
    """Build a per-position DataFrame for a snapshot block."""
    records: List[Dict[str, Any]] = []

    for wallet, token_balances in wallet_balances.items():
        if not token_balances:
            continue
        for token, raw_balance in token_balances.items():
            if raw_balance == 0:
                continue
            decimals = int(decimals_map.get(token.lower(), 18))
            human_balance = (Decimal(raw_balance) / (Decimal(10) ** Decimal(decimals)))
            if exclude_negative and human_balance <= 0:
                continue
            price = price_data.get(token)
            usd_value = (human_balance * price) if (price is not None) else None

            records.append({
                "wallet": wallet,
                "token_address": token,
                "token_name": token_names.get(token.lower(), token),
                "decimals": decimals,
                "balance": human_balance,
                "price_usd": price,
                "usd_value": usd_value,
                "block": block,
                "date": snapshot_date,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    df["pct"] = None
    for w in df.wallet.unique():
        mask = (df.wallet == w)
        total = sum((v for v in df.loc[mask, "usd_value"] if v is not None), Decimal(0))
        if total > 0:
            df.loc[mask, "pct"] = df.loc[mask, "usd_value"].apply(
                lambda v: (v / total * Decimal(100)) if v is not None else None
            )

    df = df.sort_values(["wallet", "usd_value"], ascending=[True, False], na_position="last").reset_index(drop=True)
    return df


def parallel_token_balance_aggregation(
    data_root: str,
    wallets: List[str],
    snapshot_blocks: List[int],
    token_files: Dict[str, List[str]],
    cfg: OptimizationConfig,
) -> Dict[str, Any]:
    """Aggregate balances for a wallet batch across all tokens."""
    t_start = time.perf_counter()

    token_items = list(token_files.items())
    if not token_items:
        log("[batch] WARN no token files discovered")
    chunks = [token_items[i:i + cfg.token_chunk_size] for i in range(0, len(token_items), cfg.token_chunk_size)]
    log(f"[batch] {len(token_items)} tokens in {len(chunks)} chunks")

    def process_token_chunk(chunk: List[Tuple[str, List[str]]]) -> List[pd.DataFrame]:
        out: List[pd.DataFrame] = []
        con = duckdb.connect()
        try:
            try:
                con.execute(f"SET threads={int(max(1, min(cfg.duckdb_threads, 8)))}")
                con.execute(f"SET memory_limit='{cfg.duckdb_memory}'")
            except Exception:
                pass
            for token, files in chunk:
                try:
                    df = query_token_balances_optimized(token, files, wallets, snapshot_blocks, con)
                    if df is not None and not df.empty:
                        out.append(df)
                except Exception as e:
                    log_exc(f"[batch] token {token} failed: {e}")
        finally:
            con.close()
        return out

    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_results: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as ex:
        futs = [ex.submit(process_token_chunk, ch) for ch in chunks]
        for fut in as_completed(futs):
            try:
                all_results.extend(fut.result())
            except Exception as e:
                log_exc(f"[batch] chunk failed: {e}")

    if not all_results:
        elapsed = time.perf_counter() - t_start
        return {
            "wallets": wallets,
            "snapshots": {b: {w: {} for w in wallets} for b in snapshot_blocks},
            "stats": {
                "processing_time_s": elapsed,
                "tokens_processed": 0,
                "chunks_used": len(chunks),
                "total_balances": 0,
            },
        }

    combined_df = pd.concat(all_results, ignore_index=True)

    nested_results: Dict[int, Dict[str, Dict[str, int]]] = {b: {w: {} for w in wallets} for b in snapshot_blocks}
    for _, row in combined_df.iterrows():
        token = row["token_address"]
        wallet = row["wallet"]
        for b in snapshot_blocks:
            bal = row.get(f"bal_{b}", 0)
            if bal and int(bal) != 0:
                nested_results[b][wallet][token] = int(bal)

    elapsed = time.perf_counter() - t_start
    total_balances = sum(len(wd) for bd in nested_results.values() for wd in bd.values())
    log(f"[batch] aggregation done in {elapsed:.2f}s; tokens={len(token_items)} balances={total_balances}")

    return {
        "wallets": wallets,
        "snapshots": nested_results,
        "stats": {
            "processing_time_s": elapsed,
            "tokens_processed": len(token_items),
            "chunks_used": len(chunks),
            "total_balances": total_balances,
        },
    }


# ==== Timeout + split-aware scheduler =========================================

@dataclass
class Task:
    """Container for a wallet batch task."""

    task_id: int
    wallets: List[str]


def append_problem_wallets(filepath: str, worker_id: int, wallets: List[str], reason: str = "timeout") -> None:
    """Append problematic wallets to a TSV file with a reason."""
    path = pathlib.Path(filepath)
    if path.parent and path.parent != pathlib.Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for w in wallets:
            f.write(f"{worker_id}\t{w}\t{reason}\n")


def _spawn_worker(
    task: Task,
    *,
    active: Dict[int, Tuple[mp.Process, float, Task]],
    max_procs: int,
    snapshot_blocks: List[int],
    data_root: str,
    prices_dir: str,
    token_files: Dict[str, List[str]],
    cfg: OptimizationConfig,
    out_queue: Queue,
    token_names: Dict[str, str],
    decimals_map: Dict[str, int],
    block_to_date_map: Dict[int, pd.Timestamp],
    cleaning_enabled: bool,
    clean_drop_wallets: Set[str],
    clean_suspect_tokens: Set[str],
) -> None:
    """Start a worker process for a Task and register it as active."""
    p = mp.Process(
        target=process_wallet_batch,
        args=(
            task.task_id,
            task.wallets,
            snapshot_blocks,
            data_root,
            prices_dir,
            token_files,
            cfg,
            out_queue,
            token_names,
            decimals_map,
            block_to_date_map,
            cleaning_enabled,
            clean_drop_wallets,
            clean_suspect_tokens,
        ),
        daemon=False,
    )
    p.start()
    active[task.task_id] = (p, time.perf_counter(), task)
    log(f"[main] started worker {task.task_id} (batch_size={len(task.wallets)}) | active={len(active)}/{max_procs}")


def _split_task(task: Task, next_id: int) -> Tuple[Task, Task, int]:
    """Split a Task into two halves and return updated task ids."""
    mid = max(1, len(task.wallets) // 2)
    left = Task(next_id, task.wallets[:mid])
    next_id += 1
    right = Task(next_id, task.wallets[mid:])
    next_id += 1
    return left, right, next_id


def run_with_timeouts_and_splits(
    initial_batches: List[List[str]],
    max_procs: int,
    snapshot_blocks: List[int],
    data_root: str,
    prices_dir: str,
    token_files: Dict[str, List[str]],
    cfg: OptimizationConfig,
    out_queue: Queue,
    token_names: Dict[str, str],
    decimals_map: Dict[str, int],
    block_to_date_map: Dict[int, pd.Timestamp],
    problem_wallets_path: str,
    cleaning_enabled: bool,
    clean_drop_wallets: Set[str],
    clean_suspect_tokens: Set[str],
    *,
    worker_timeout_sec: int = WORKER_TIMEOUT_SEC,
    min_batch_to_split: int = MIN_BATCH_TO_SPLIT,
    sleep_poll_sec: float = SLEEP_POLL_SEC,
) -> None:
    """Supervise worker processes with timeouts and adaptive splitting."""
    next_task_id = 1
    pending: Deque[Task] = deque(Task(next_task_id + i, b) for i, b in enumerate(initial_batches))
    next_task_id += len(initial_batches)

    active: Dict[int, Tuple[mp.Process, float, Task]] = {}

    def backfill() -> None:
        while pending and len(active) < max_procs:
            _spawn_worker(
                pending.popleft(),
                active=active,
                max_procs=max_procs,
                snapshot_blocks=snapshot_blocks,
                data_root=data_root,
                prices_dir=prices_dir,
                token_files=token_files,
                cfg=cfg,
                out_queue=out_queue,
                token_names=token_names,
                decimals_map=decimals_map,
                block_to_date_map=block_to_date_map,
                cleaning_enabled=cleaning_enabled,
                clean_drop_wallets=clean_drop_wallets,
                clean_suspect_tokens=clean_suspect_tokens,
            )

    backfill()

    while active or pending:
        finished_any = False

        for tid, (proc, _start_ts, _task) in list(active.items()):
            if not proc.is_alive():
                try:
                    proc.join()
                except Exception:
                    pass
                active.pop(tid, None)
                log(f"[main] worker finished | id={tid} | active={len(active)}/{max_procs}")
                finished_any = True

        now = time.perf_counter()
        for tid, (proc, start_ts, task) in list(active.items()):
            if now - start_ts > worker_timeout_sec:
                log(
                    f"[main] TIMEOUT | id={tid} | size={len(task.wallets)} | limit={worker_timeout_sec}s — terminating"
                )
                try:
                    proc.terminate()
                    proc.join(timeout=5)
                except Exception:
                    pass
                active.pop(tid, None)

                if len(task.wallets) <= min_batch_to_split:
                    append_problem_wallets(problem_wallets_path, tid, task.wallets, reason="timeout_min_batch")
                    log(f"[main] recorded {len(task.wallets)} problem wallets from worker {tid} → {problem_wallets_path}")
                else:
                    left, right, next_task_id = _split_task(task, next_task_id)
                    pending.appendleft(right)
                    pending.appendleft(left)
                    log(
                        f"[main] split worker {tid} → new tasks {left.task_id} ({len(left.wallets)}) "
                        f"+ {right.task_id} ({len(right.wallets)})"
                    )

                finished_any = True

        if finished_any or (pending and len(active) < max_procs):
            backfill()

        if not finished_any:
            time.sleep(sleep_poll_sec)


# ======================= Writer Process (single) =======================

def writer_process(out_path: str, q: Queue, stop_event: Event) -> None:
    """Write final output rows to a single Parquet file."""
    out_file = pathlib.Path(out_path)
    if out_file.parent and out_file.parent != pathlib.Path("."):
        out_file.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[pq.ParquetWriter] = None
    total_rows = 0
    t0 = time.perf_counter()
    try:
        while not (stop_event.is_set() and q.empty()):
            try:
                df = q.get(timeout=0.5)
            except Exception:
                continue
            if df is None:
                continue
            df = df[["wallet_address", "block_number", "total_value_usd", "num_tokens", "holdings"]].copy()
            table = pa.Table.from_pandas(df, schema=OUTPUT_SCHEMA, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_file), OUTPUT_SCHEMA, compression="snappy")
            writer.write_table(table)
            total_rows += len(df)
    except Exception as e:
        log_exc(f"[writer] ERROR: {e}")
        traceback.print_exc()
    finally:
        if writer is not None:
            writer.close()
        elapsed = time.perf_counter() - t0
        size_mb = out_file.stat().st_size / (1024 * 1024) if out_file.exists() else 0
        log(f"[writer] closed | rows={total_rows} | {size_mb:.2f} MB | {elapsed:.2f}s")


# ======================= Batch worker =======================

def _filter_positions_for_wallet(
    wallet_df: pd.DataFrame,
    *,
    cleaning_enabled: bool,
    clean_suspect_tokens: Set[str],
) -> List[Dict[str, Any]]:
    """Filter and normalize token positions for a single wallet snapshot."""
    positions: List[Dict[str, Any]] = []
    for _, r in wallet_df.iterrows():
        bal_dec = r["balance"]
        bal_str = dec_to_str(bal_dec)
        tok = r["token_address"].lower()
        if cleaning_enabled:
            if tok in clean_suspect_tokens or _is_impossible_balance_str(bal_str):
                continue
        positions.append({
            "token_address": r["token_address"],
            "balance": bal_str,
            "value_usd": r["usd_value"],
        })
    return positions


def process_wallet_batch(
    batch_id: int,
    wallets: List[str],
    snapshot_blocks: List[int],
    data_root: str,
    prices_dir: str,
    token_files: Dict[str, List[str]],
    cfg: OptimizationConfig,
    out_queue: Queue,
    token_names: Dict[str, str],
    decimals_map: Dict[str, int],
    block_to_date_map: Dict[int, pd.Timestamp],
    cleaning_enabled: bool,
    clean_drop_wallets: Set[str],
    clean_suspect_tokens: Set[str],
) -> None:
    """Process one wallet batch and enqueue output rows for writing."""
    t0 = time.perf_counter()
    try:
        if not wallets:
            log(f"[worker {batch_id}] WARN empty wallet batch")
            return

        agg_res = parallel_token_balance_aggregation(
            data_root=data_root,
            wallets=wallets,
            snapshot_blocks=snapshot_blocks,
            token_files=token_files,
            cfg=cfg,
        )

        toks: Set[str] = set()
        for bd in agg_res["snapshots"].values():
            for wd in bd.values():
                toks.update(wd.keys())
        log(f"[worker {batch_id}] tokens_in_batch={len(toks)}")

        final_rows: List[Dict[str, Any]] = []
        emitted = 0

        for block in snapshot_blocks:
            snapshot_date = block_to_date_map.get(int(block))
            if snapshot_date is None:
                continue
            block_balances = agg_res["snapshots"].get(block, {})
            if not any(block_balances.values()):
                continue
            block_tokens = sorted(set(t for w in block_balances.values() for t in w.keys()))
            prices = load_prices(block_tokens, snapshot_date, prices_dir, cfg.forward_fill_price)

            enriched_df = build_portfolio_dataframe(
                block, block_balances, decimals_map, token_names, prices, snapshot_date, exclude_negative=True
            )
            if enriched_df.empty:
                continue

            for wallet, wallet_df in enriched_df.groupby("wallet"):
                if cleaning_enabled and _normalize_addr_lower(wallet) in clean_drop_wallets:
                    continue

                positions = _filter_positions_for_wallet(
                    wallet_df,
                    cleaning_enabled=cleaning_enabled,
                    clean_suspect_tokens=clean_suspect_tokens,
                )
                if cleaning_enabled and not positions:
                    continue

                total_dec = sum((p["value_usd"] for p in positions if p["value_usd"] is not None), Decimal(0))
                num_tokens = int(
                    sum(1 for p in positions if _to_dec_safe(p.get("balance")) and _to_dec_safe(p.get("balance")) > 0)
                )

                if total_dec > 0:
                    for p in positions:
                        v = p["value_usd"]
                        pct = (v / total_dec * Decimal(100)) if (v is not None and v > 0) else Decimal(0)
                        p["percentage"] = dec_to_str(pct)
                else:
                    for p in positions:
                        p["percentage"] = dec_to_str(Decimal(0))

                holdings: List[Dict[str, Any]] = []
                for p in positions:
                    holdings.append({
                        "token_address": p["token_address"],
                        "balance": p["balance"],
                        "value_usd": dec_to_str(p["value_usd"]),
                        "percentage": p["percentage"],
                    })

                if cleaning_enabled and not holdings:
                    continue

                final_rows.append({
                    "wallet_address": wallet,
                    "block_number": int(block),
                    "total_value_usd": dec_to_str(total_dec),
                    "num_tokens": num_tokens,
                    "holdings": json.dumps(holdings, ensure_ascii=False),
                })

            if len(final_rows) >= 50_000:
                df_out = pd.DataFrame(final_rows)
                out_queue.put(df_out)
                emitted += len(final_rows)
                final_rows.clear()

        if final_rows:
            df_out = pd.DataFrame(final_rows)
            out_queue.put(df_out)
            emitted += len(final_rows)
            final_rows.clear()

        elapsed = time.perf_counter() - t0
        log(f"[worker {batch_id}] done | wallets={len(wallets)} | rows emitted≈{emitted} | {elapsed:.2f}s")

    except Exception as e:
        log_exc(f"[worker {batch_id}] ERROR: {e}")
        traceback.print_exc()


# ======================= Wallet batches (streaming reader) =======================

def make_batch_size_generator(
    base: int,
    jitter: float = 0.25,
    min_size: int = 25_000,
    max_size: int = 150_000,
) -> Iterator[int]:
    """Yield an endless stream of target batch sizes with jitter."""
    lo = int(base * (1 - jitter))
    hi = int(base * (1 + jitter))
    lo = max(lo, min_size)
    hi = min(hi, max_size)
    while True:
        yield max(min(int(random.uniform(lo, hi)), max_size), min_size)


def stream_wallet_batches_dynamic(
    parquet_path: str,
    column: str,
    base_batch_size: int,
    size_iter: Iterable[int],
    arrow_batch_rows: int = 1_000_000,
    progress_every: int = 5_000_000,
) -> Iterator[List[str]]:
    """Stream wallet addresses from Parquet and yield dynamic-sized batches."""
    dataset = ds.dataset(parquet_path, format="parquet")
    schema = dataset.schema
    if column not in schema.names:
        raise ValueError(f"[wallets] column '{column}' not found. Available: {schema.names}")

    scanner = ds.Scanner.from_dataset(dataset, columns=[column], batch_size=arrow_batch_rows)

    batch: List[str] = []
    seen = 0
    yielded = 0
    size_iter = iter(size_iter)
    target = next(size_iter)

    log(f"[wallets] streaming column='{column}' base_batch={base_batch_size} (dynamic sizes)")

    reader = scanner.to_reader()
    for rb in reader:
        values = rb.column(0).to_pylist()
        for v in values:
            addr = normalize_eth_address(str(v))
            if not addr:
                continue
            batch.append(addr)
            if len(batch) >= target:
                yield batch
                yielded += 1
                batch = []
                target = next(size_iter)
        seen += len(values)
        if progress_every and seen // progress_every > (seen - len(values)) // progress_every:
            log(f"[wallets] scanned ~{seen:,} rows | yielded={yielded} batches")

    if batch:
        yield batch
        yielded += 1
    log(f"[wallets] done streaming | total_rows≈{seen:,} | total_batches={yielded}")


def estimate_parquet_rows(parquet_path: str) -> int:
    """Estimate total rows in a Parquet file or directory."""
    p = pathlib.Path(parquet_path)
    if p.is_file():
        try:
            pf = pq.ParquetFile(str(p))
            return pf.metadata.num_rows if pf.metadata is not None else pf.read().num_rows
        except Exception:
            pass

    try:
        dataset = ds.dataset(parquet_path, format="parquet")
        if hasattr(dataset, "count_rows"):
            return int(dataset.count_rows())
        total = 0
        files = getattr(dataset, "files", None)
        if files:
            for f in files:
                try:
                    pf = pq.ParquetFile(f)
                    if pf.metadata is not None:
                        total += pf.metadata.num_rows
                    else:
                        total += pf.read().num_rows
                except Exception:
                    pass
            if total > 0:
                return total
        tbl = dataset.to_table(columns=[dataset.schema.names[0]])
        return tbl.num_rows
    except Exception:
        return 0


# ======================= Dates list (hardcoded policy) =======================

def build_dates_list() -> List[str]:
    """Return snapshot dates (monthly starts plus curated event neighbors)."""
    dates: List[str] = []

    start = pd.Timestamp("2023-03-01")
    end = pd.Timestamp("2025-03-01")
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur = (cur + pd.offsets.MonthBegin(1))

    key_dates = [
        ("2023-03-11", "2023-03-12", "2023-03-13"),
        ("2023-04-11", "2023-04-12", "2023-04-13"),
        ("2023-06-14", "2023-06-15", "2023-06-16"),
        ("2023-10-15", "2023-10-16", "2023-10-17"),
        ("2024-01-09", "2024-01-10", "2024-01-11"),
        ("2024-03-12", "2024-03-13", "2024-03-14"),
        ("2024-03-13", "2024-03-14", "2024-03-15"),
        ("2024-04-19", "2024-04-20", "2024-04-21"),
        ("2024-05-22", "2024-05-23", "2024-05-24"),
        ("2024-07-22", "2024-07-23", "2024-07-24"),
    ]

    neighbors = [d for (d_minus, _d, d_plus) in key_dates for d in (d_minus, d_plus)]
    seen = set()
    merged = []
    for d in dates + neighbors:
        if d not in seen:
            seen.add(d)
            merged.append(d)
    return merged

def _validate_inputs(args: argparse.Namespace) -> None:
    """Validate required input paths and directories."""
    for pth, desc in [
        (args.data_root, "data_root"),
        (args.block_to_date_csv, "block_to_date_csv"),
        (args.prices_dir, "prices_dir"),
        (args.decimals_cache, "decimals_cache"),
        (args.token_status_csv, "token_status_csv"),
        (args.input_wallets_parquet, "input_wallets_parquet"),
    ]:
        if desc in ("data_root", "prices_dir"):
            if not pathlib.Path(pth).is_dir():
                raise FileNotFoundError(f"{desc} must be a directory: {pth}")
        else:
            require_file(pth, desc)


def _load_clean_drop_wallets(args: argparse.Namespace, cleaning_enabled: bool) -> Set[str]:
    """Load wallet drop list for cleaning if enabled."""
    clean_drop_wallets: Set[str] = set()
    if not cleaning_enabled:
        return clean_drop_wallets
    if args.clean_drop_dead_defaults:
        clean_drop_wallets.update(DEFAULT_DROP_WALLETS)
    if args.clean_drop_file:
        if not os.path.exists(args.clean_drop_file):
            log_exc(f"[clean] drop-file not found: {args.clean_drop_file}")
        else:
            with open(args.clean_drop_file, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if w:
                        clean_drop_wallets.add(_normalize_addr_lower(w))
    return clean_drop_wallets


def _load_clean_suspects(args: argparse.Namespace, cleaning_enabled: bool) -> Set[str]:
    """Load suspect token list for cleaning if enabled."""
    if not (cleaning_enabled and args.clean_suspects_json):
        return set()
    try:
        with open(args.clean_suspects_json, "r", encoding="utf-8") as f:
            provided = json.load(f) or []
        suspects = {_normalize_addr_lower(t) for t in provided}
        log(f"[clean] loaded {len(suspects)} suspect tokens from {args.clean_suspects_json}")
        return suspects
    except Exception as e:
        log_exc(f"[clean] failed to load suspects JSON: {e}")
        return set()


def _log_wallets_parquet_info(path: str) -> None:
    """Log basic Parquet metadata for the wallets input."""
    try:
        if os.path.isfile(path):
            pf = pq.ParquetFile(path)
            log(f"[wallets] file | row_groups={pf.num_row_groups}")
            if pf.num_row_groups:
                md = pf.metadata
                rg0 = md.row_group(0)
                log(f"[wallets] first RG rows={rg0.num_rows:,}")
        else:
            log("[wallets] directory input (using dataset scanner)")
    except Exception:
        pass


def _build_block_to_date_map(block_to_date_csv: str, snapshot_blocks: List[int]) -> Dict[int, pd.Timestamp]:
    """Build a block->date map for the resolved snapshot blocks."""
    blocks_arr, dates_arr = _load_block_date_arrays(block_to_date_csv)
    all_block_to_date = {int(b): pd.Timestamp(d) for b, d in zip(blocks_arr, dates_arr)}
    return {int(b): all_block_to_date[int(b)] for b in snapshot_blocks if int(b) in all_block_to_date}


# ======================= CLI & Main =======================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="ERC-20 multi-snapshot portfolio reconstruction (cluster-ready).")
    p.add_argument("--data-root", required=True, help="Root dir with token_address=*/**/*.parquet")
    p.add_argument("--block-to-date-csv", required=True, help="CSV mapping blocks to dates")
    p.add_argument("--prices-dir", required=True, help="Dir with {token}_price.csv files")
    p.add_argument("--decimals-cache", required=True, help="JSON cache for token decimals")
    p.add_argument(
        "--token-status-csv",
        required=True,
        help=(
            "Token status CSV with columns: symbol,name,contract_address,is_erc_20,creation_block,"
            "price_data_status,first_date,last_date,num_points,error_message,trading_volume_status,"
            "fdv_mcap_lt_eth,local_status"
        ),
    )
    p.add_argument("--rpc-url", required=True, help="ETH RPC endpoint (Infura/Alchemy/etc.)")
    p.add_argument("--input-wallets-parquet", required=True, help="Input Parquet with wallet addresses")
    p.add_argument("--wallets-column", default="address", help="Column name with addresses (default: address)")
    p.add_argument("--output", required=True, help="Output Parquet file path")
    p.add_argument("--batch-size", type=int, default=50_000, help="Wallets per process batch (default: 50k)")
    p.add_argument("--processes", type=int, default=max(1, (cpu_count() or 2) - 1), help="Worker processes")
    p.add_argument("--duckdb-threads", type=int, default=2, help="DuckDB threads per worker (default: 2)")
    p.add_argument("--duckdb-mem", default="2GB", help="DuckDB memory per worker (default: 2GB)")
    p.add_argument("--token-chunk-size", type=int, default=64, help="Tokens per chunk inside a worker (default: 64)")
    p.add_argument("--max-tokens", type=int, default=None, help="Limit number of tokens (debug)")
    p.add_argument("--max-batches", type=int, default=None, help="Limit number of wallet batches (debug)")
    p.add_argument("--no-forward-fill", action="store_true", help="Disable price forward fill")
    p.add_argument(
        "--problem-wallets",
        default=None,
        help=(
            "File to append problematic wallet addresses that still time out after max splits. "
            "Default: <output>.problem_wallets.txt"
        ),
    )
    p.add_argument(
        "--skip-clean-overflows",
        action="store_true",
        help="Skip inline overflow cleaning heuristics (default: run cleaning).",
    )
    p.add_argument(
        "--clean-drop-file",
        default=None,
        help="Optional text file with wallet addresses (one per line) to drop during cleaning.",
    )
    p.add_argument(
        "--clean-drop-dead-defaults",
        action="store_true",
        help="Drop built-in burn/DEAD wallets during cleaning.",
    )
    p.add_argument(
        "--clean-suspects-json",
        default=None,
        help="Optional JSON file with suspect token list to bypass scanning stage.",
    )

    return p.parse_args()

def main() -> int:
    """Run the portfolio reconstruction pipeline."""
    args = parse_args()

    cleaning_enabled = not args.skip_clean_overflows
    raw_output_path = args.output

    log("=== Portfolio Reconstruction ===")
    log(f"data_root={args.data_root}")
    log(f"wallets_parquet={args.input_wallets_parquet}")
    log(f"output_final={args.output}")
    log(f"output_raw={raw_output_path} (cleaning={'on' if cleaning_enabled else 'off'})")
    log(
        "processes={p} batch_size={b} duckdb_threads={t} duckdb_mem={m}".format(
            p=args.processes,
            b=args.batch_size,
            t=args.duckdb_threads,
            m=args.duckdb_mem,
        )
    )
    log("------------------------------------------")

    _validate_inputs(args)

    problem_wallets_path = args.problem_wallets or (args.output + ".problem_wallets.txt")

    clean_drop_wallets = _load_clean_drop_wallets(args, cleaning_enabled)
    clean_suspect_tokens = _load_clean_suspects(args, cleaning_enabled)

    if cleaning_enabled:
        log(
            f"[clean] inline cleaning enabled | drop_wallets={len(clean_drop_wallets)} | "
            f"preset_suspects={len(clean_suspect_tokens)}"
        )

    _log_wallets_parquet_info(args.input_wallets_parquet)

    total_rows = estimate_parquet_rows(args.input_wallets_parquet)
    if total_rows > 0:
        expected_batches = math.ceil(total_rows / args.batch_size)
        log(
            f"[plan] wallets≈{total_rows:,} | batch_size={args.batch_size} "
            f"| expected_batches≈{expected_batches:,} | max_concurrent_processes={args.processes}"
        )
    else:
        log(
            f"[plan] wallets≈unknown (couldn’t estimate quickly) "
            f"| batch_size={args.batch_size} | max_concurrent_processes={args.processes}"
        )

    date_list = build_dates_list()
    log(f"[dates] total={len(date_list)} (first few: {date_list[:6]})")
    snapshot_blocks = resolve_blocks_for_dates(date_list, args.block_to_date_csv)
    if not snapshot_blocks:
        log_exc("[fatal] no snapshot blocks resolved; exiting.")
        return 2
    log(f"[blocks] resolved={len(snapshot_blocks)} (sample: {snapshot_blocks[:40]})")

    whitelist_tokens, token_names = load_token_status_data(args.token_status_csv)
    log(f"[status] whitelist_tokens={len(whitelist_tokens)} | names_loaded={len(token_names)}")

    token_files = discover_token_files_parallel(args.data_root, token_whitelist=set(whitelist_tokens), max_workers=4)
    if args.max_tokens:
        token_files = dict(list(token_files.items())[: args.max_tokens])
        log(f"[discover] limited to first {args.max_tokens} tokens")

    block_to_date_map = _build_block_to_date_map(args.block_to_date_csv, snapshot_blocks)

    all_tokens = list(token_files.keys())
    w3 = Web3(HTTPProvider(args.rpc_url))
    decimals_map = fetch_token_decimals(all_tokens, w3, args.decimals_cache)

    cfg = OptimizationConfig(
        processes=args.processes,
        batch_size=args.batch_size,
        token_chunk_size=args.token_chunk_size,
        duckdb_threads=max(1, args.duckdb_threads),
        duckdb_memory=args.duckdb_mem,
        forward_fill_price=not args.no_forward_fill,
    )

    from multiprocessing import Manager

    manager = Manager()
    q: Queue = manager.Queue(maxsize=8)
    stop_event = manager.Event()
    writer = Process(target=writer_process, args=(raw_output_path, q, stop_event), daemon=False)
    writer.start()

    t_all = time.perf_counter()
    batch_idx = 0
    try:
        delta = 10_000
        size_iter = make_batch_size_generator(
            args.batch_size,
            jitter=0.25,
            min_size=(args.batch_size - delta),
            max_size=(args.batch_size + delta),
        )

        initial_batches: List[List[str]] = []
        for wallets in stream_wallet_batches_dynamic(
            args.input_wallets_parquet,
            args.wallets_column,
            args.batch_size,
            size_iter,
            arrow_batch_rows=250_000,
            progress_every=2_000_000,
        ):
            batch_idx += 1
            initial_batches.append(wallets)

            if args.max_batches and batch_idx >= args.max_batches:
                log(f"[main] reached max_batches={args.max_batches} (debug mode)")
                break

        log(f"[main] prepared {len(initial_batches)} initial batches for scheduling")

        worker_timeout_sec = getattr(args, "worker_timeout_sec", WORKER_TIMEOUT_SEC)
        min_batch_to_split = getattr(args, "min_batch_to_split", MIN_BATCH_TO_SPLIT)

        try:
            global WORKER_TIMEOUT_SEC, MIN_BATCH_TO_SPLIT
            WORKER_TIMEOUT_SEC = worker_timeout_sec
            MIN_BATCH_TO_SPLIT = min_batch_to_split
        except NameError:
            pass

        run_with_timeouts_and_splits(
            initial_batches=initial_batches,
            max_procs=cfg.processes,
            snapshot_blocks=snapshot_blocks,
            data_root=args.data_root,
            prices_dir=args.prices_dir,
            token_files=token_files,
            cfg=cfg,
            out_queue=q,
            token_names=token_names,
            decimals_map=decimals_map,
            block_to_date_map=block_to_date_map,
            cleaning_enabled=cleaning_enabled,
            clean_drop_wallets=clean_drop_wallets,
            clean_suspect_tokens=clean_suspect_tokens,
            worker_timeout_sec=worker_timeout_sec,
            min_batch_to_split=min_batch_to_split,
            problem_wallets_path=problem_wallets_path,
        )

    except KeyboardInterrupt:
        log_exc("[main] KeyboardInterrupt — terminating workers...")
    finally:
        stop_event.set()
        writer.join()
        elapsed_all = time.perf_counter() - t_all
        log(f"[main] all done in {elapsed_all:.2f}s | batches_prepared={batch_idx}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
