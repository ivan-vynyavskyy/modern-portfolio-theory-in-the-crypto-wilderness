#!/usr/bin/env python3
"""
5_0_portfolio_backtest_pipeline.py

Purpose
-------
Run a block-level, wallet-by-wallet backtest of mean-variance portfolio strategies,
compute forward returns and CAPM metrics, and summarize performance across horizons.

What it does
------------
- Streams holdings by block from a parquet dataset and filters eligible wallets.
- Optimizes three MVO strategies per wallet (max-Sharpe, min-variance, max-return-same-vol)
  and evaluates forward returns by horizon.
- Computes CAPM beta/alpha, classifies market regime, and aggregates per-block and
  overall summaries into CSV/parquet outputs.

Inputs
------
- holdings parquet (parquet; columns: wallet_address, block_number, holdings,
  total_value_usd, num_tokens)
- block-to-date map (CSV; columns: block_number or block; date/bound_date/block_time_utc)
- prices directory (per-token CSVs named <token>_price.csv with date,daily_close_usd)

Outputs
-------
- summary CSV (per-block and overall summaries; columns as printed in summary tables)
- wallet-level parquet dataset (partitioned by block_number)

CLI
---
Examples:
  $ python 5_0_portfolio_backtest_pipeline.py --block-to-date-csv data/block_dates.csv \
      --holdings-parquet data/holdings.parquet --prices-dir data/prices
  $ python 5_0_portfolio_backtest_pipeline.py --block-to-date-csv data/block_dates.csv \
      --blocks blocks.txt --out-csv out/custom_backtest_summary.csv

Notes
-----
- PyArrow streaming is used for batching and summaries.
- Market regime is computed from the first batch in each block and drives the dynamic pick.
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import datetime
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from helpers.capm_analytics import (
    build_market_return_series_from_arrays,
    compute_forward_market_return,
    compute_portfolio_beta,
    compute_token_betas,
)
from utils.log_utils import log, log_exc
from helpers.portfolio_analytics import PortfolioAnalytics
from helpers.risk_config import RiskConfig


# ----------------------------- Constants -----------------------------
DEFAULT_HOLDINGS_PARQUET = "./data/portfolio_holdings.parquet"
HORIZON_DAYS = 20
DEFAULT_BATCH_SIZE = 50_000
DEFAULT_BATCH_TIMEOUT_MIN = 15
DEFAULT_WORKERS = 16
DEFAULT_OUT_CSV = "custom_backtest_summary.csv"
DEFAULT_OUT_WALLET_PARQUET = "custom_backtest_wallet_level.parquet"
DEFAULT_REGIME_LOOKBACK_DAYS = 60
DEFAULT_REGIME_TOP_K_ASSETS = 30


# --------------------------- JSON Holdings --------------------------

def parse_holdings_json(raw: Any) -> List[Dict[str, Any]]:
    """Parse a holdings JSON field into a list of token objects."""
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except Exception as exc:
        raise ValueError(f"Invalid holdings JSON: {exc}")


def extract_holdings_map(holdings_json_field: Any) -> Dict[str, float]:
    """Convert holdings JSON into a token->balance map."""
    arr = parse_holdings_json(holdings_json_field)
    if not isinstance(arr, list):
        raise ValueError("Holdings JSON must be a list of token objects")

    out: Dict[str, float] = {}
    for item in arr:
        token = str(item.get("token_address", "")).lower().strip()
        bal = float(item.get("balance", 0.0) or 0.0)
        if token and bal > 0:
            out[token] = out.get(token, 0.0) + bal
    return out


# ---------------------------- Block Dates ---------------------------

@lru_cache(maxsize=1)
def load_block_to_date_map(block_to_date_csv: str) -> pd.DataFrame:
    """Load the block->date mapping file into a normalized DataFrame."""
    mapping = pd.read_csv(block_to_date_csv)

    block_col = next((c for c in ["block_number", "block"] if c in mapping.columns), None)
    if block_col is None:
        raise ValueError(
            f"Could not find a block column in {block_to_date_csv}. "
            f"Expected one of ['block_number','block']; got {list(mapping.columns)}"
        )

    date_col = next(
        (c for c in ["date", "bound_date", "block_time_utc"] if c in mapping.columns),
        None,
    )
    if date_col is None:
        raise ValueError(
            f"Expected a date-like column in {block_to_date_csv}. "
            f"Got: {list(mapping.columns)}"
        )

    mapping[date_col] = (
        pd.to_datetime(mapping[date_col], errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    mapping = mapping.dropna(subset=[date_col, block_col]).copy()

    mapping.rename(columns={block_col: "block_number", date_col: "date"}, inplace=True)
    mapping["block_number"] = mapping["block_number"].astype(int)
    return mapping[["block_number", "date"]]


# --------------------------- Price Utilities ------------------------

@lru_cache(maxsize=None)
def load_price_file(token: str, prices_dir: str) -> Optional[pd.DataFrame]:
    """Load a token price CSV into a DataFrame, if present."""
    price_file = Path(prices_dir) / f"{token.lower()}_price.csv"
    if not price_file.exists():
        return None
    df = pd.read_csv(price_file, usecols=["date", "daily_close_usd"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")
    return df


def get_price_for_date(
    token: str,
    target_date: pd.Timestamp,
    prices_dir: str,
    *,
    forward_fill: bool,
    max_fill_days: Optional[int],
) -> Optional[float]:
    """Return the price for a token on a date, with optional forward-fill."""
    df = load_price_file(token, prices_dir)
    if df is None or df.empty:
        return None

    d = pd.to_datetime(target_date).date()

    exact = df[df["date"] == d]
    if not exact.empty:
        return float(exact["daily_close_usd"].iloc[0])

    if forward_fill:
        prior = df[df["date"] < d]
        if not prior.empty:
            last_row = prior.iloc[-1]
            last_date = last_row["date"]
            delta = d - last_date
            if max_fill_days is not None and delta.days > max_fill_days:
                return None
            return float(last_row["daily_close_usd"])

    return None


def compute_portfolio_value_from_units(
    units: Dict[str, float],
    snapshot_date: pd.Timestamp,
    *,
    prices_dir: str,
    config: RiskConfig,
) -> Optional[float]:
    """Compute the USD value of holdings units at a snapshot date."""
    if not units:
        return 0.0

    total = 0.0
    for token, amount in units.items():
        price = get_price_for_date(
            token,
            snapshot_date,
            prices_dir=prices_dir,
            forward_fill=config.forward_fill_prices,
            max_fill_days=config.max_price_staleness_days,
        )
        if price is None:
            return None
        total += float(amount) * float(price)
    return total


def load_price_window_data(
    tokens: List[str],
    snapshot_date: pd.Timestamp,
    config: RiskConfig,
    prices_dir: str,
    required_cols: Tuple[str, str] = ("date", "daily_close_usd"),
) -> pd.DataFrame:
    """Load a historical price window for a set of tokens."""
    snapshot_date = pd.to_datetime(snapshot_date).tz_localize(None)
    prices_dir_path = Path(prices_dir)

    end_date = min(snapshot_date.normalize(), pd.Timestamp.now().normalize()) - pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=config.window_days)
    date_index = pd.date_range(start=start_date, end=end_date, freq="D")

    series_map: Dict[str, pd.Series] = {}
    for token in tokens:
        token_file = prices_dir_path / f"{token.lower()}_price.csv"
        if not token_file.exists():
            continue

        try:
            df = pd.read_csv(token_file, usecols=list(required_cols))
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        except Exception:
            continue

        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if df.empty:
            continue

        series = pd.Series(df["daily_close_usd"].astype(float).values, index=df["date"].values)
        series = series.reindex(date_index)
        series_map[token] = series

    if not series_map:
        return pd.DataFrame(index=date_index)

    return pd.DataFrame(series_map, index=date_index)


def build_price_window_arrays(
    tokens: List[str],
    snapshot_date: pd.Timestamp,
    config: RiskConfig,
    prices_dir: str,
) -> Tuple[pd.DatetimeIndex, Dict[str, np.ndarray]]:
    """Load a price window and return aligned arrays by token."""
    snapshot_date = pd.to_datetime(snapshot_date).tz_localize(None)
    prices_dir_path = Path(prices_dir)

    end_date = min(snapshot_date.normalize(), pd.Timestamp.now().normalize()) - pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=config.window_days)
    date_index = pd.date_range(start=start_date, end=end_date, freq="D")

    arrays: Dict[str, np.ndarray] = {}
    for token in tokens:
        token_file = prices_dir_path / f"{token.lower()}_price.csv"
        if not token_file.exists():
            continue
        try:
            df = pd.read_csv(token_file, usecols=["date", "daily_close_usd"])
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            if df.empty:
                continue
            series = pd.Series(df["daily_close_usd"].astype(float).values, index=df["date"].values)
            series = series.reindex(date_index)
            arrays[token] = series.values
        except Exception:
            continue

    return date_index, arrays


# ----------------------- Supply / Market Cap ------------------------

def load_supply_data(csv_path: str) -> Dict[str, List[Tuple[int, float]]]:
    """Load token supply CSV into {token_address: [(block, supply), ...]} sorted by block."""
    df = pd.read_csv(csv_path, usecols=["block", "token_address", "supply"])
    df["token_address"] = df["token_address"].str.lower().str.strip()
    df["block"] = pd.to_numeric(df["block"], errors="coerce")
    df["supply"] = pd.to_numeric(df["supply"], errors="coerce")
    df = df.dropna(subset=["block", "token_address", "supply"])
    df["block"] = df["block"].astype(int)
    df = df.sort_values(["token_address", "block"])

    result: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for _, row in df.iterrows():
        result[row["token_address"]].append((int(row["block"]), float(row["supply"])))
    return dict(result)


def get_supply_at_block(
    supply_data: Dict[str, List[Tuple[int, float]]],
    token: str,
    block_number: int,
) -> Optional[float]:
    """Get supply for token at or before block_number using binary search."""
    entries = supply_data.get(token.lower())
    if not entries:
        return None
    blocks = [e[0] for e in entries]
    idx = bisect.bisect_right(blocks, block_number) - 1
    if idx < 0:
        return None
    return entries[idx][1]


def get_market_cap(
    token: str,
    block_number: int,
    snapshot_date: pd.Timestamp,
    prices_dir: str,
    supply_data: Dict[str, List[Tuple[int, float]]],
) -> Optional[float]:
    """Compute market_cap = supply * price. Returns None if either is unavailable."""
    supply = get_supply_at_block(supply_data, token, block_number)
    if supply is None or supply <= 0:
        return None
    price = get_price_for_date(
        token, snapshot_date, prices_dir,
        forward_fill=True, max_fill_days=7,
    )
    if price is None or price <= 0:
        return None
    return supply * price


# ------------------------ Market Regime Logic -----------------------

def classify_market_regime_from_prices(
    prices_df: pd.DataFrame,
    *,
    lookback_days: int = 60,
    t_threshold: float = 1.0,
    benchmark_tokens: Optional[Sequence[str]] = (
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
        "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
    ),
    benchmark_weight: float = 0.5,
    min_assets_pool: int = 2,
    min_assets_bench: int = 1,
) -> Dict[str, Any]:
    """Classify the market regime from a price panel."""
    out: Dict[str, Any] = {
        "regime": "SIDEWAYS",
        "n_assets_pool": 0,
        "n_assets_bench": 0,
        "n_days": 0,
        "mu": np.nan,
        "sigma": np.nan,
        "t_stat": np.nan,
        "cum_return": np.nan,
        "used_benchmark": False,
        "benchmark_tokens_used": [],
        "benchmark_weight": float(benchmark_weight),
    }

    if prices_df is None or prices_df.empty:
        out["regime"] = "UNKNOWN"
        return out

    px = prices_df.copy()
    px.columns = [str(c).lower() for c in px.columns]
    px = px.loc[:, px.notna().sum(axis=0) >= 5]

    out["n_assets_pool"] = int(px.shape[1])
    if px.shape[1] < int(min_assets_pool):
        out["regime"] = "UNKNOWN"
        return out

    rets = np.log(px / px.shift(1))
    r_pool = rets.mean(axis=1, skipna=True)

    r_bench = None
    bench_used: List[str] = []
    if benchmark_tokens:
        bench_cols: List[str] = []
        for token in benchmark_tokens:
            token_norm = str(token).lower()
            if token_norm in rets.columns:
                bench_cols.append(token_norm)
        if len(bench_cols) >= int(min_assets_bench):
            r_bench = rets[bench_cols].mean(axis=1, skipna=True)
            bench_used = bench_cols

    r_m = r_pool.copy()
    bw = float(np.clip(float(benchmark_weight), 0.0, 1.0))

    if r_bench is not None:
        r_m = bw * r_bench + (1.0 - bw) * r_pool
        out["used_benchmark"] = True
        out["benchmark_tokens_used"] = bench_used
        out["n_assets_bench"] = int(len(bench_used))
    else:
        out["used_benchmark"] = False
        out["benchmark_tokens_used"] = []
        out["n_assets_bench"] = 0

    r_m = r_m.dropna()
    if r_m.empty:
        out["regime"] = "UNKNOWN"
        return out

    r_m = r_m.iloc[-int(lookback_days):]
    n = int(len(r_m))
    out["n_days"] = n
    if n < 20:
        out["regime"] = "UNKNOWN"
        return out

    mu = float(r_m.mean())
    sigma = float(r_m.std(ddof=1)) if n > 1 else 0.0
    cum_r = float(r_m.sum())

    if sigma <= 0 or not np.isfinite(sigma):
        t_stat = np.nan
    else:
        t_stat = float(mu / (sigma / np.sqrt(n)))

    out["mu"] = mu
    out["sigma"] = sigma
    out["t_stat"] = t_stat
    out["cum_return"] = cum_r

    thr = float(t_threshold)
    if np.isfinite(t_stat) and (t_stat >= thr) and (cum_r > 0):
        out["regime"] = "BULLISH"
    elif np.isfinite(t_stat) and (t_stat <= -thr) and (cum_r < 0):
        out["regime"] = "BEARISH"
    else:
        out["regime"] = "SIDEWAYS"

    return out


def pick_block_index_tokens(df_block: pd.DataFrame, top_k: int = 30) -> List[str]:
    """Select frequent tokens across wallets in a block for index construction."""
    counts: Dict[str, int] = {}
    for raw in df_block["holdings"].tolist():
        try:
            holdings = extract_holdings_map(raw)
        except Exception:
            continue
        for token in holdings.keys():
            counts[token] = counts.get(token, 0) + 1

    if not counts:
        return []

    tokens_sorted = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [token for token, _ in tokens_sorted[: int(top_k)]]


# ------------------------- Optimization Prep ------------------------

def prepare_portfolio_data_for_optimization(
    holdings_snapshot: Dict[str, float],
    snapshot_date: pd.Timestamp,
    config: RiskConfig,
    prices_dir: str,
    *,
    zero_threshold: Optional[float] = None,
    preloaded_prices_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Prepare prices and returns needed for optimization."""
    if zero_threshold is None:
        zero_threshold = config.zero_threshold

    analytics = PortfolioAnalytics(config)

    if preloaded_prices_df is not None:
        prices_df = preloaded_prices_df
    else:
        prices_df = load_price_window_data(
            tokens=list(holdings_snapshot.keys()),
            snapshot_date=snapshot_date,
            config=config,
            prices_dir=prices_dir,
        )
    if prices_df.empty:
        return {"error": "No price data available in the chosen window."}

    return analytics.prepare_portfolio_data(
        holdings_snapshot=holdings_snapshot,
        snapshot_date=snapshot_date,
        prices_df=prices_df,
        zero_threshold=zero_threshold,
    )


def optimize_at_snapshot_custom_only(
    holdings_json_field: Any,
    snapshot_date: pd.Timestamp,
    config: RiskConfig,
    prices_dir: str,
    mean_reversion_strength: float,
    *,
    mu_trim_frac: float = 0.05,
    preloaded_prices_df: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, Any]]:
    """Optimize portfolio weights for a single snapshot."""
    holdings_map = extract_holdings_map(holdings_json_field)
    if not holdings_map:
        return None

    snapshot_date = pd.to_datetime(snapshot_date)

    prep = prepare_portfolio_data_for_optimization(
        holdings_snapshot=holdings_map,
        snapshot_date=snapshot_date,
        config=config,
        prices_dir=prices_dir,
        zero_threshold=config.zero_threshold,
        preloaded_prices_df=preloaded_prices_df,
    )
    if "error" in prep:
        return None

    tokens = prep["final_tokens"]
    returns_df = prep["returns_df"]
    w_cur_raw = np.array(prep["weights"], dtype=float)
    if len(tokens) < 2:
        return None

    analytics = PortfolioAnalytics(config)
    cov = analytics.ensure_positive_definite(analytics.estimate_covariance_matrix(returns_df))

    mu_raw = returns_df.apply(lambda s: analytics.trimmed_mean(s, frac=float(mu_trim_frac)), axis=0)
    mr_stats = analytics.compute_mean_reversion_metrics(returns_df)
    mu_adj = analytics.adjust_expected_returns_for_mean_reversion(
        mu_raw,
        mr_stats,
        strength=mean_reversion_strength,
    )
    mu = mu_adj.loc[returns_df.columns].values

    w_cur = analytics.project_long_only_cap(w_cur_raw, cap=float(config.max_weight_single_asset))

    initial_variance_daily = np.nan
    initial_expected_return_daily = np.nan
    try:
        metrics_payload = analytics.compute_portfolio_metrics(returns_df, w_cur)
        metrics = metrics_payload.get("metrics", {}) if isinstance(metrics_payload, dict) else {}
        initial_variance_daily = float(metrics.get("variance_daily", np.nan))
        initial_expected_return_daily = float(metrics.get("expected_return_daily", np.nan))
    except Exception:
        initial_variance_daily = np.nan
        initial_expected_return_daily = np.nan
    if not np.isfinite(initial_variance_daily):
        initial_variance_daily = np.nan
    if not np.isfinite(initial_expected_return_daily):
        initial_expected_return_daily = np.nan

    current_variance = float(w_cur @ cov @ w_cur)
    current_return = float(w_cur @ mu)

    w_better_return = analytics.max_return_same_vol(
        cov=cov,
        mu=mu,
        target_var=current_variance,
        long_only=config.long_only,
        cap=config.max_weight_single_asset,
    )

    w_safer_risk = analytics.min_variance_target_return(
        mu=mu,
        cov=cov,
        target_return=current_return,
        long_only=config.long_only,
        cap=config.max_weight_single_asset,
    )

    w_max_sharpe = analytics.max_sharpe_weights(
        mu=mu,
        cov=cov,
        long_only=config.long_only,
        cap=config.max_weight_single_asset,
    )

    snapshot_prices: Dict[str, float] = prep["snapshot_prices"]
    total_value0 = float(prep["total_value"])
    filtered_holdings0: Dict[str, float] = prep["filtered_holdings"]
    token_to_idx = {token: i for i, token in enumerate(tokens)}

    def units_from_weights(weights: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for token, idx in token_to_idx.items():
            price0 = snapshot_prices.get(token)
            if price0 is None or price0 <= 0:
                continue
            alloc_value = float(weights[idx]) * total_value0
            if alloc_value <= 0:
                continue
            out[token] = alloc_value / float(price0)
        return out

    units_initial = dict(filtered_holdings0)
    units_better_return = units_from_weights(w_better_return)
    units_safer_risk = units_from_weights(w_safer_risk)
    units_max_sharpe = units_from_weights(w_max_sharpe)

    l1_gap_better = float(np.sum(np.abs(w_cur - w_better_return)))
    l1_gap_safer = float(np.sum(np.abs(w_cur - w_safer_risk)))
    l1_gap_sharpe = float(np.sum(np.abs(w_cur - w_max_sharpe)))

    return {
        "total_value0": total_value0,
        "snapshot_prices": snapshot_prices,
        "units_initial": units_initial,
        "units_better_return": units_better_return,
        "units_safer_risk": units_safer_risk,
        "units_max_sharpe": units_max_sharpe,
        "l1_gap_better_return": l1_gap_better,
        "l1_gap_safer_risk": l1_gap_safer,
        "l1_gap_max_sharpe": l1_gap_sharpe,
        "initial_variance_daily": initial_variance_daily,
        "initial_expected_return_daily": initial_expected_return_daily,
        "tokens": tokens,
        "w_cur": w_cur,
        "w_better_return": w_better_return,
        "w_safer_risk": w_safer_risk,
        "w_max_sharpe": w_max_sharpe,
    }


# --------------------- Holdings Streaming (PyArrow) -----------------

def parse_blocks_arg(value: Optional[str]) -> Optional[List[int]]:
    """Parse a --blocks argument into a de-duplicated list or None."""
    if value is None:
        return None

    value = str(value).strip()
    if not value:
        return None

    path = Path(value)
    blocks: List[int] = []
    seen: set[int] = set()

    def add_block(raw_val: str) -> None:
        v = int(raw_val)
        if v <= 0:
            raise ValueError("Block numbers must be positive integers.")
        if v not in seen:
            seen.add(v)
            blocks.append(v)

    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            add_block(line)
    else:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            add_block(part)

    if not blocks:
        raise ValueError(f"No valid block numbers parsed from --blocks: {value!r}")

    return blocks


def _normalize_wallet_address(raw: Any) -> Optional[str]:
    """Normalize wallet addresses to lowercase 0x-prefixed hex strings."""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return "0x" + bytes(raw).hex()
    s = str(raw).strip()
    if not s:
        return None
    s = s.lower()
    if s.startswith("0x"):
        s = "0x" + s[2:]
    elif len(s) == 40:
        s = "0x" + s
    else:
        s = "0x" + s.lstrip("0x")
    return s


def _safe_hm(value: Any) -> Dict[str, float]:
    """Parse holdings JSON into a map; return {} on failure."""
    try:
        return extract_holdings_map(value)
    except Exception:
        return {}


def iter_holdings_wallet_batches_for_block(
    holdings_parquet: str,
    block_number: int,
    batch_size: int,
    columns: List[str],
    max_wallets: Optional[int],
) -> Iterator[pd.DataFrame]:
    """Stream holdings rows for a block in batch-sized DataFrames."""
    holdings_ds = ds.dataset(holdings_parquet, format="parquet")
    for required_col in ["wallet_address", "block_number", "holdings"]:
        if required_col not in holdings_ds.schema.names:
            raise ValueError(f"Column {required_col!r} not found in holdings parquet.")

    cols = [c for c in columns if c in holdings_ds.schema.names]
    if "wallet_address" not in cols:
        cols.append("wallet_address")
    if "block_number" not in cols:
        cols.append("block_number")

    filter_expr = ds.field("block_number") == int(block_number)
    batch_size = max(int(batch_size), 1)
    buffer: List[pd.DataFrame] = []
    buffer_count = 0

    for record_batch in holdings_ds.to_batches(columns=cols, filter=filter_expr, batch_size=100_000):
        df = record_batch.to_pandas()
        if df.empty:
            continue

        df["wallet_address"] = df["wallet_address"].apply(_normalize_wallet_address)
        df = df.dropna(subset=["wallet_address"])
        if df.empty:
            continue

        start = 0
        n = len(df)
        while start < n:
            remaining = batch_size - buffer_count
            take = min(remaining, n - start)
            slice_df = df.iloc[start : start + take].copy()
            buffer.append(slice_df)
            buffer_count += len(slice_df)
            start += take

            if buffer_count >= batch_size:
                batch_df = pd.concat(buffer, ignore_index=True)
                yield batch_df
                buffer.clear()
                buffer_count = 0

    if buffer:
        batch_df = pd.concat(buffer, ignore_index=True)
        yield batch_df


def discover_blocks_in_holdings(holdings_parquet: str) -> List[int]:
    """Discover unique block numbers present in holdings parquet."""
    holdings_ds = ds.dataset(holdings_parquet, format="parquet")
    if "block_number" not in holdings_ds.schema.names:
        raise ValueError("holdings_parquet must contain a 'block_number' column.")

    blocks: set[int] = set()
    for record_batch in holdings_ds.to_batches(columns=["block_number"], batch_size=200_000):
        arr = record_batch.column("block_number")
        for b in arr.to_pylist():
            if b is None:
                continue
            blocks.add(int(b))

    blocks_sorted = sorted(blocks)
    log(f"[blocks][auto] Using {len(blocks_sorted)} blocks from holdings: {blocks_sorted}")
    if len(blocks_sorted) > 50:
        log("[warn] Large number of blocks detected; consider --blocks to restrict.")
    return blocks_sorted


# --------------------------- Timing Stats ---------------------------

class OptTimingStats:
    """Thread-safe collector for optimization timing stats."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.opt_calls_total = 0
        self.opt_calls_success = 0
        self.opt_time_total_s = 0.0
        self.opt_time_success_s = 0.0

    def record(self, duration_s: float, success: bool) -> None:
        """Record a single timing sample."""
        with self.lock:
            self.opt_calls_total += 1
            self.opt_time_total_s += float(duration_s)
            if success:
                self.opt_calls_success += 1
                self.opt_time_success_s += float(duration_s)

    def snapshot(self) -> Dict[str, float]:
        """Return a snapshot of aggregated timing statistics."""
        with self.lock:
            return {
                "opt_calls_total": self.opt_calls_total,
                "opt_calls_success": self.opt_calls_success,
                "opt_time_total_s": self.opt_time_total_s,
                "opt_time_success_s": self.opt_time_success_s,
            }


def print_opt_timing(prefix: str, stats: OptTimingStats) -> None:
    """Log timing stats with a standard prefix."""
    snap = stats.snapshot()
    calls = int(snap["opt_calls_total"])
    succ = int(snap["opt_calls_success"])
    t_total = float(snap["opt_time_total_s"])
    t_succ = float(snap["opt_time_success_s"])
    avg_call_ms = (t_total / calls * 1000.0) if calls else float("nan")
    avg_succ_ms = (t_succ / succ * 1000.0) if succ else float("nan")
    log(
        f"{prefix} opt_calls={calls} success={succ} "
        f"avg_call_ms={avg_call_ms:.2f} avg_success_ms={avg_succ_ms:.2f}"
    )


def wealth_bucket_from_value(value: Any) -> str:
    """Map a portfolio value to a coarse wealth bucket."""
    try:
        x = float(value)
    except Exception:
        return "UNKNOWN"

    if not np.isfinite(x) or x < 0:
        return "UNKNOWN"
    if x < 1:
        return "0-1"
    if x < 100:
        return "1-100"
    if x < 1_000:
        return "100-1K"
    if x < 10_000:
        return "1K-10K"
    return "10K+"


# -------------------------- Backtest Globals ------------------------

GLOBAL_TOKEN_PRICES: Dict[str, np.ndarray] = {}
GLOBAL_DATE_INDEX: Optional[pd.DatetimeIndex] = None
GLOBAL_CONFIG: Optional[RiskConfig] = None
GLOBAL_PRICES_DIR: Optional[str] = None
GLOBAL_MEAN_REV: float = 0.0
GLOBAL_MARKET_REGIME: str = "UNKNOWN"
GLOBAL_TOKEN_BETAS: Dict[str, Dict[str, float]] = {}
GLOBAL_MARKET_RETURN_20D: float = float("nan")
GLOBAL_SUPPLY_LOOKUP: Dict[str, List[Tuple[int, float]]] = {}
GLOBAL_BLOCK_NUMBER: int = 0

BENCHMARK_TOKENS: Tuple[str, ...] = (
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",  # WBTC
)


def _build_prices_df_for_tokens(tokens: List[str]) -> pd.DataFrame:
    """Build a price DataFrame for a list of tokens from global arrays."""
    if GLOBAL_DATE_INDEX is None:
        return pd.DataFrame()
    data: Dict[str, np.ndarray] = {}
    for token in tokens:
        arr = GLOBAL_TOKEN_PRICES.get(token)
        if arr is not None and len(arr) == len(GLOBAL_DATE_INDEX):
            data[token] = arr
    if not data:
        return pd.DataFrame(index=GLOBAL_DATE_INDEX)
    return pd.DataFrame(data, index=GLOBAL_DATE_INDEX)


def _worker_init(
    token_prices: Dict[str, np.ndarray],
    date_index: pd.DatetimeIndex,
    config: RiskConfig,
    prices_dir: str,
    mean_reversion_strength: float,
    market_regime: str,
    token_betas: Dict[str, Dict[str, float]],
    market_return_20d: float,
    supply_lookup: Dict[str, List[Tuple[int, float]]],
    block_number: int,
) -> None:
    """Initialize global state for worker processes."""
    global GLOBAL_TOKEN_PRICES, GLOBAL_DATE_INDEX, GLOBAL_CONFIG, GLOBAL_PRICES_DIR
    global GLOBAL_MEAN_REV, GLOBAL_MARKET_REGIME
    global GLOBAL_TOKEN_BETAS, GLOBAL_MARKET_RETURN_20D
    global GLOBAL_SUPPLY_LOOKUP, GLOBAL_BLOCK_NUMBER
    GLOBAL_TOKEN_PRICES = token_prices
    GLOBAL_DATE_INDEX = date_index
    GLOBAL_CONFIG = config
    GLOBAL_PRICES_DIR = prices_dir
    GLOBAL_MEAN_REV = float(mean_reversion_strength)
    GLOBAL_MARKET_REGIME = str(market_regime).upper()
    GLOBAL_TOKEN_BETAS = token_betas
    GLOBAL_MARKET_RETURN_20D = float(market_return_20d)
    GLOBAL_SUPPLY_LOOKUP = supply_lookup
    GLOBAL_BLOCK_NUMBER = int(block_number)


def _cast_num_tokens(val: Any) -> Optional[int]:
    """Cast a numeric value to int if it is a clean integer-like value."""
    try:
        x = float(val)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    xr = int(round(x))
    if abs(x - xr) > 1e-6:
        return None
    return xr


def _units_to_short_holdings_json(
    units: Dict[str, float],
    prices: Dict[str, float],
    total_value: float,
) -> str:
    """Serialize units to a compact holdings JSON string."""
    rows: List[Dict[str, Any]] = []
    if not units or total_value <= 0:
        return "[]"
    for token, balance in units.items():
        price = prices.get(token)
        if price is None or price <= 0:
            continue
        value_usd = float(balance) * float(price)
        if value_usd <= 0:
            continue
        pct = 100.0 * value_usd / float(total_value)
        rows.append(
            {
                "token_address": token,
                "balance": float(balance),
                "percentage": float(round(pct, 8)),
            }
        )
    rows.sort(key=lambda r: r["token_address"])
    return json.dumps(rows, separators=(",", ":"), ensure_ascii=False)


def _process_wallet_task(task: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], float, bool]:
    """Process one wallet task using global price arrays. Returns a single flat record."""
    config = GLOBAL_CONFIG
    if config is None:
        return None, 0.0, False

    snapshot_date = pd.to_datetime(task["date"]).tz_localize(None)
    block_number = int(task["block_number"])
    wallet = str(task.get("wallet_address", "")).lower()
    holdings_map: Dict[str, float] = task["holdings_map"]
    total_value_usd_raw = task.get("total_value_usd", np.nan)
    num_tokens_raw = task.get("num_tokens", np.nan)
    try:
        total_value_usd = float(total_value_usd_raw)
    except Exception:
        total_value_usd = np.nan

    tokens = list(holdings_map.keys())
    price_df = _build_prices_df_for_tokens(tokens)

    success_opt = False
    t_opt = time.perf_counter()
    try:
        opt = optimize_at_snapshot_custom_only(
            holdings_json_field=json.dumps(
                [{"token_address": k, "balance": v} for k, v in holdings_map.items()]
            ),
            snapshot_date=snapshot_date,
            config=config,
            prices_dir=GLOBAL_PRICES_DIR or "",
            mean_reversion_strength=GLOBAL_MEAN_REV,
            mu_trim_frac=0.05,
            preloaded_prices_df=price_df,
        )
    except Exception:
        opt = None
    duration = time.perf_counter() - t_opt

    if opt is not None:
        try:
            success_opt = float(opt.get("total_value0", 0.0)) > 0
        except Exception:
            success_opt = False

    if opt is None:
        return None, duration, success_opt

    total_value0 = float(opt["total_value0"])
    if total_value0 <= 0:
        return None, duration, success_opt

    u0 = opt["units_initial"]
    u1 = opt["units_better_return"]
    u2 = opt["units_safer_risk"]
    u3 = opt["units_max_sharpe"]

    snapshot_prices = opt.get("snapshot_prices", {}) or {}

    l1_better = float(opt["l1_gap_better_return"])
    l1_safer = float(opt["l1_gap_safer_risk"])
    l1_sharpe = float(opt["l1_gap_max_sharpe"])

    init_var = opt.get("initial_variance_daily", np.nan)
    init_ret = opt.get("initial_expected_return_daily", np.nan)

    # --- CAPM: compute portfolio betas ---
    opt_tokens = opt.get("tokens", [])
    w_cur = opt.get("w_cur")
    w_better_return = opt.get("w_better_return")
    w_safer_risk = opt.get("w_safer_risk")
    w_max_sharpe = opt.get("w_max_sharpe")

    beta_baseline = np.nan
    beta_better_return = np.nan
    beta_safer_risk = np.nan
    beta_max_sharpe = np.nan

    if opt_tokens and GLOBAL_TOKEN_BETAS:
        if w_cur is not None:
            beta_baseline = compute_portfolio_beta(GLOBAL_TOKEN_BETAS, w_cur, opt_tokens)
        if w_better_return is not None:
            beta_better_return = compute_portfolio_beta(GLOBAL_TOKEN_BETAS, w_better_return, opt_tokens)
        if w_safer_risk is not None:
            beta_safer_risk = compute_portfolio_beta(GLOBAL_TOKEN_BETAS, w_safer_risk, opt_tokens)
        if w_max_sharpe is not None:
            beta_max_sharpe = compute_portfolio_beta(GLOBAL_TOKEN_BETAS, w_max_sharpe, opt_tokens)

    # --- Naive portfolio strategies ---
    n_tokens = len(opt_tokens)

    # Equal-weight
    w_equal = np.ones(n_tokens) / n_tokens

    # Market-cap-weight
    mcap_valid = False
    w_mcap = None
    if GLOBAL_SUPPLY_LOOKUP:
        market_caps = []
        for token in opt_tokens:
            mcap = get_market_cap(
                token, GLOBAL_BLOCK_NUMBER, snapshot_date,
                GLOBAL_PRICES_DIR or "", GLOBAL_SUPPLY_LOOKUP,
            )
            market_caps.append(mcap if (mcap is not None and mcap > 0) else None)

        n_missing = sum(1 for mc in market_caps if mc is None)
        if n_missing > 0:
            missing_mcap_tokens = [t for t, mc in zip(opt_tokens, market_caps) if mc is None]
            # log_exc(
            #     f"[WARN] Missing market cap for {n_missing}/{n_tokens} tokens "
            #     f"in wallet {wallet} block {block_number}: {missing_mcap_tokens}. "
            #     f"Setting all mcap fields to null for this wallet."
            # )
        else:
            total_known_mcap = sum(mc for mc in market_caps if mc is not None)
            if total_known_mcap > 0:
                mcap_array = np.array([mc / total_known_mcap for mc in market_caps])
                w_mcap = mcap_array / mcap_array.sum()
                mcap_valid = True

    # Reuse the units_from_weights closure from optimize_at_snapshot_custom_only
    token_to_idx = {token: i for i, token in enumerate(opt_tokens)}

    def _units_from_w(weights: np.ndarray) -> Dict[str, float]:
        out_u: Dict[str, float] = {}
        for tk, ix in token_to_idx.items():
            p0 = snapshot_prices.get(tk)
            if p0 is None or p0 <= 0:
                continue
            alloc = float(weights[ix]) * total_value0
            if alloc <= 0:
                continue
            out_u[tk] = alloc / float(p0)
        return out_u

    units_equal = _units_from_w(w_equal)

    # Betas for naive strategies
    beta_equal = np.nan
    if opt_tokens and GLOBAL_TOKEN_BETAS:
        beta_equal = compute_portfolio_beta(GLOBAL_TOKEN_BETAS, w_equal, opt_tokens)

    # L1 distances
    def l1_dist(w_a: np.ndarray, w_b: np.ndarray) -> float:
        return float(np.sum(np.abs(w_a - w_b)))

    l1_baseline_vs_equal = l1_dist(w_cur, w_equal)
    l1_better_return_vs_equal = l1_dist(w_better_return, w_equal)
    l1_safer_risk_vs_equal = l1_dist(w_safer_risk, w_equal)
    l1_max_sharpe_vs_equal = l1_dist(w_max_sharpe, w_equal)

    # Mcap-weight: compute derived fields only if all tokens have market cap
    if mcap_valid:
        units_mcap = _units_from_w(w_mcap)
        beta_mcap = compute_portfolio_beta(GLOBAL_TOKEN_BETAS, w_mcap, opt_tokens) if (opt_tokens and GLOBAL_TOKEN_BETAS) else np.nan
        l1_baseline_vs_mcap = l1_dist(w_cur, w_mcap)
        l1_better_return_vs_mcap = l1_dist(w_better_return, w_mcap)
        l1_safer_risk_vs_mcap = l1_dist(w_safer_risk, w_mcap)
        l1_max_sharpe_vs_mcap = l1_dist(w_max_sharpe, w_mcap)
    else:
        units_mcap = None
        beta_mcap = np.nan
        l1_baseline_vs_mcap = np.nan
        l1_better_return_vs_mcap = np.nan
        l1_safer_risk_vs_mcap = np.nan
        l1_max_sharpe_vs_mcap = np.nan

    # --- Fixed 20-day forward evaluation ---
    horizon_date = snapshot_date + pd.Timedelta(days=HORIZON_DAYS)

    v0 = compute_portfolio_value_from_units(u0, horizon_date, prices_dir=GLOBAL_PRICES_DIR or "", config=config)
    v1 = compute_portfolio_value_from_units(u1, horizon_date, prices_dir=GLOBAL_PRICES_DIR or "", config=config)
    v2 = compute_portfolio_value_from_units(u2, horizon_date, prices_dir=GLOBAL_PRICES_DIR or "", config=config)
    v3 = compute_portfolio_value_from_units(u3, horizon_date, prices_dir=GLOBAL_PRICES_DIR or "", config=config)
    v_equal = compute_portfolio_value_from_units(units_equal, horizon_date, prices_dir=GLOBAL_PRICES_DIR or "", config=config)

    r0 = (v0 / total_value0) - 1.0 if (v0 is not None and v0 > 0) else float("nan")
    r1 = (v1 / total_value0) - 1.0 if (v1 is not None and v1 > 0) else float("nan")
    r2 = (v2 / total_value0) - 1.0 if (v2 is not None and v2 > 0) else float("nan")
    r3 = (v3 / total_value0) - 1.0 if (v3 is not None and v3 > 0) else float("nan")
    r_equal = (v_equal / total_value0) - 1.0 if (v_equal is not None and v_equal > 0) else float("nan")

    if mcap_valid:
        v_mcap = compute_portfolio_value_from_units(units_mcap, horizon_date, prices_dir=GLOBAL_PRICES_DIR or "", config=config)
        r_mcap = (v_mcap / total_value0) - 1.0 if (v_mcap is not None and v_mcap > 0) else float("nan")
    else:
        r_mcap = np.nan

    record = {
        "wallet_address": wallet,
        "block_number": block_number,
        "date": snapshot_date,
        "market_regime": GLOBAL_MARKET_REGIME,
        "total_value_usd": float(total_value_usd) if np.isfinite(total_value_usd) else np.nan,
        "num_tokens": _cast_num_tokens(num_tokens_raw),
        "holdings_short": _units_to_short_holdings_json(u0, snapshot_prices, total_value0),
        "holdings_better_return": _units_to_short_holdings_json(u1, snapshot_prices, total_value0),
        "holdings_safer_risk": _units_to_short_holdings_json(u2, snapshot_prices, total_value0),
        "holdings_max_sharpe": _units_to_short_holdings_json(u3, snapshot_prices, total_value0),
        "holdings_equal_weight": _units_to_short_holdings_json(units_equal, snapshot_prices, total_value0),
        "holdings_mcap_weight": _units_to_short_holdings_json(units_mcap, snapshot_prices, total_value0) if mcap_valid else None,
        "initial_variance_daily": float(init_var) if np.isfinite(init_var) else np.nan,
        "initial_expected_return_daily": float(init_ret) if np.isfinite(init_ret) else np.nan,
        "l1_gap_better_return": l1_better,
        "l1_gap_safer_risk": l1_safer,
        "l1_gap_max_sharpe": l1_sharpe,
        "l1_baseline_vs_equal": l1_baseline_vs_equal,
        "l1_baseline_vs_mcap": l1_baseline_vs_mcap,
        "l1_better_return_vs_equal": l1_better_return_vs_equal,
        "l1_safer_risk_vs_equal": l1_safer_risk_vs_equal,
        "l1_max_sharpe_vs_equal": l1_max_sharpe_vs_equal,
        "l1_better_return_vs_mcap": l1_better_return_vs_mcap,
        "l1_safer_risk_vs_mcap": l1_safer_risk_vs_mcap,
        "l1_max_sharpe_vs_mcap": l1_max_sharpe_vs_mcap,
        "beta_baseline": float(beta_baseline) if np.isfinite(beta_baseline) else np.nan,
        "beta_better_return": float(beta_better_return) if np.isfinite(beta_better_return) else np.nan,
        "beta_safer_risk": float(beta_safer_risk) if np.isfinite(beta_safer_risk) else np.nan,
        "beta_max_sharpe": float(beta_max_sharpe) if np.isfinite(beta_max_sharpe) else np.nan,
        "beta_equal_weight": float(beta_equal) if np.isfinite(beta_equal) else np.nan,
        "beta_mcap_weight": float(beta_mcap) if np.isfinite(beta_mcap) else np.nan,
        "ret_baseline": float(r0),
        "ret_better_return": float(r1) if np.isfinite(r1) else np.nan,
        "ret_safer_risk": float(r2) if np.isfinite(r2) else np.nan,
        "ret_max_sharpe": float(r3) if np.isfinite(r3) else np.nan,
        "ret_equal_weight": float(r_equal) if np.isfinite(r_equal) else np.nan,
        "ret_mcap_weight": float(r_mcap) if np.isfinite(r_mcap) else np.nan,
        "market_return": float(GLOBAL_MARKET_RETURN_20D) if np.isfinite(GLOBAL_MARKET_RETURN_20D) else np.nan,
    }

    return record, duration, success_opt


# ----------------------- Summary/Diagnostics ------------------------

def _capm_agg(group: pd.DataFrame, col: str) -> Tuple[float, float]:
    """Compute mean and median for a CAPM column, returning (mean, median)."""
    s = group[col].dropna() if col in group.columns else pd.Series(dtype=float)
    if len(s) == 0:
        return np.nan, np.nan
    return float(s.mean()), float(s.median())


def summarize_block_results(df_block: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-block results into strategy-level aggregates."""
    if df_block.empty:
        return pd.DataFrame()

    block_num = int(df_block["block_number"].iloc[0])
    block_date = pd.to_datetime(df_block["date"]).min().normalize()

    rows: List[Dict[str, Any]] = []

    group = df_block
    base = group["ret_baseline"].dropna()
    base_mean = float(base.mean()) if len(base) else np.nan
    base_median = float(base.median()) if len(base) else np.nan
    base_count = int(len(base))

    beta_bl_mean, beta_bl_median = _capm_agg(group, "beta_baseline")
    mkt_ret_mean, mkt_ret_median = _capm_agg(group, "market_return")

    def add_strategy(
        strategy_name: str,
        ret_col: str,
        l1_col: Optional[str],
        beta_col: Optional[str] = None,
    ) -> None:
        s = group[ret_col].dropna()
        count = int(len(s))

        mean_ret = float(s.mean()) if count else np.nan
        median_ret = float(s.median()) if count else np.nan

        gg = group[["ret_baseline", ret_col]].dropna()
        hit_rate = float((gg[ret_col] > gg["ret_baseline"]).mean()) if len(gg) else np.nan

        if l1_col is None:
            l1_mean = np.nan
            l1_median = np.nan
        else:
            l1s = group[l1_col].dropna()
            l1_mean = float(l1s.mean()) if len(l1s) else np.nan
            l1_median = float(l1s.median()) if len(l1s) else np.nan

        b_mean, b_median = _capm_agg(group, beta_col) if beta_col else (np.nan, np.nan)

        rows.append(
            {
                "date": block_date.date().isoformat(),
                "block": block_num,
                "strategy": strategy_name,
                "baseline_mean": base_mean,
                "baseline_median": base_median,
                "count_optimized": count,
                "hit_rate": hit_rate,
                "l1_mean": l1_mean,
                "l1_median": l1_median,
                "strategy_mean": mean_ret,
                "strategy_median": median_ret,
                "beta_mean": b_mean,
                "beta_median": b_median,
                "market_return_mean": mkt_ret_mean,
                "market_return_median": mkt_ret_median,
            }
        )

    rows.append(
        {
            "date": block_date.date().isoformat(),
            "block": block_num,
            "strategy": "Baseline",
            "baseline_mean": base_mean,
            "baseline_median": base_median,
            "count_optimized": base_count,
            "hit_rate": np.nan,
            "l1_mean": np.nan,
            "l1_median": np.nan,
            "strategy_mean": base_mean,
            "strategy_median": base_median,
            "beta_mean": beta_bl_mean,
            "beta_median": beta_bl_median,
            "market_return_mean": mkt_ret_mean,
            "market_return_median": mkt_ret_median,
        }
    )

    add_strategy(
        "Max Return (Same Vol)", "ret_better_return", "l1_gap_better_return",
        beta_col="beta_better_return",
    )
    add_strategy(
        "Min Variance (Same Ret)", "ret_safer_risk", "l1_gap_safer_risk",
        beta_col="beta_safer_risk",
    )
    add_strategy(
        "Max Sharpe", "ret_max_sharpe", "l1_gap_max_sharpe",
        beta_col="beta_max_sharpe",
    )
    add_strategy(
        "Equal Weight", "ret_equal_weight", "l1_baseline_vs_equal",
        beta_col="beta_equal_weight",
    )
    add_strategy(
        "Market-Cap Weight", "ret_mcap_weight", "l1_baseline_vs_mcap",
        beta_col="beta_mcap_weight",
    )

    return pd.DataFrame(rows)


def summarize_block_results_by_wealth_bucket(df_block: pd.DataFrame) -> pd.DataFrame:
    """Summarize results by wealth bucket."""
    if df_block.empty:
        return pd.DataFrame()

    # Derive wealth_bucket on-the-fly if not present
    if "wealth_bucket" not in df_block.columns:
        if "total_value_usd" in df_block.columns:
            df_block = df_block.copy()
            df_block["wealth_bucket"] = df_block["total_value_usd"].apply(wealth_bucket_from_value)
        else:
            return pd.DataFrame()

    block_num = int(df_block["block_number"].iloc[0])
    block_date = pd.to_datetime(df_block["date"]).min().normalize()

    rows: List[Dict[str, Any]] = []

    for bucket, group in df_block.groupby("wealth_bucket", sort=False):
        bucket_label = str(bucket)
        base = group["ret_baseline"].dropna()
        base_mean = float(base.mean()) if len(base) else np.nan
        base_median = float(base.median()) if len(base) else np.nan
        base_count = int(len(base))

        beta_bl_mean, beta_bl_median = _capm_agg(group, "beta_baseline")
        mkt_ret_mean, mkt_ret_median = _capm_agg(group, "market_return")

        def add_strategy(
            strategy_name: str,
            ret_col: str,
            l1_col: Optional[str],
            beta_col: Optional[str] = None,
        ) -> None:
            s = group[ret_col].dropna()
            count = int(len(s))

            mean_ret = float(s.mean()) if count else np.nan
            median_ret = float(s.median()) if count else np.nan

            gg = group[["ret_baseline", ret_col]].dropna()
            hit_rate = float((gg[ret_col] > gg["ret_baseline"]).mean()) if len(gg) else np.nan

            if l1_col is None:
                l1_mean = np.nan
                l1_median = np.nan
            else:
                l1s = group[l1_col].dropna()
                l1_mean = float(l1s.mean()) if len(l1s) else np.nan
                l1_median = float(l1s.median()) if len(l1s) else np.nan

            b_mean, b_median = _capm_agg(group, beta_col) if beta_col else (np.nan, np.nan)

            rows.append(
                {
                    "date": block_date.date().isoformat(),
                    "block": block_num,
                    "wealth_bucket": bucket_label,
                    "strategy": strategy_name,
                    "baseline_mean": base_mean,
                    "baseline_median": base_median,
                    "count_optimized": count,
                    "hit_rate": hit_rate,
                    "l1_mean": l1_mean,
                    "l1_median": l1_median,
                    "strategy_mean": mean_ret,
                    "strategy_median": median_ret,
                    "beta_mean": b_mean,
                    "beta_median": b_median,
                    "market_return_mean": mkt_ret_mean,
                    "market_return_median": mkt_ret_median,
                }
            )

        rows.append(
            {
                "date": block_date.date().isoformat(),
                "block": block_num,
                "wealth_bucket": bucket_label,
                "strategy": "Baseline",
                "baseline_mean": base_mean,
                "baseline_median": base_median,
                "count_optimized": base_count,
                "hit_rate": np.nan,
                "l1_mean": np.nan,
                "l1_median": np.nan,
                "strategy_mean": base_mean,
                "strategy_median": base_median,
                "beta_mean": beta_bl_mean,
                "beta_median": beta_bl_median,
                "market_return_mean": mkt_ret_mean,
                "market_return_median": mkt_ret_median,
            }
        )

        add_strategy(
            "Max Return (Same Vol)", "ret_better_return", "l1_gap_better_return",
            beta_col="beta_better_return",
        )
        add_strategy(
            "Min Variance (Same Ret)", "ret_safer_risk", "l1_gap_safer_risk",
            beta_col="beta_safer_risk",
        )
        add_strategy(
            "Max Sharpe", "ret_max_sharpe", "l1_gap_max_sharpe",
            beta_col="beta_max_sharpe",
        )
        add_strategy(
            "Equal Weight", "ret_equal_weight", "l1_baseline_vs_equal",
            beta_col="beta_equal_weight",
        )
        add_strategy(
            "Market-Cap Weight", "ret_mcap_weight", "l1_baseline_vs_mcap",
            beta_col="beta_mcap_weight",
        )

    return pd.DataFrame(rows)


def add_diff_status_and_score_pp(
    df: pd.DataFrame,
    *,
    w_mean: float = 0.5,
    w_median: float = 0.5,
) -> pd.DataFrame:
    """Add status labels and a combined score in percentage points."""
    df = df.copy()

    df["diff_mean_pp"] = (df["strategy_mean"] - df["baseline_mean"]) * 100.0
    df["diff_median_pp"] = (df["strategy_median"] - df["baseline_median"]) * 100.0

    both = (df["diff_mean_pp"] > 0) & (df["diff_median_pp"] > 0)
    one = ((df["diff_mean_pp"] > 0) ^ (df["diff_median_pp"] > 0))
    df["strategy_status"] = np.select(
        [both, one],
        ["BETTER", "MIXED"],
        default="WORSE",
    )

    w_mean = float(w_mean)
    w_median = float(w_median)
    s = w_mean + w_median
    if s <= 0:
        raise ValueError("w_mean + w_median must be > 0")
    w_mean /= s
    w_median /= s
    df["score_pp"] = w_mean * df["diff_mean_pp"] + w_median * df["diff_median_pp"]

    is_base = df["strategy"].astype(str).str.lower().eq("baseline")
    df.loc[is_base, ["diff_mean_pp", "diff_median_pp", "score_pp"]] = np.nan
    df.loc[is_base, "strategy_status"] = ""

    return df


def upsert_summary_csv(out_csv: str, df_new: pd.DataFrame) -> None:
    """Upsert summary rows into a CSV based on key columns."""
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not out_path.exists():
        df_new.to_csv(out_path, index=False)
        log(f"[wrote] Summary CSV: {out_csv}")
        return

    try:
        existing = pd.read_csv(
            out_path,
            dtype={
                "date": str,
                "block": str,
                "wealth_bucket": str,
                "strategy": str,
            },
        )
    except Exception:
        existing = pd.DataFrame()

    key_cols = ["date", "block", "wealth_bucket", "strategy"]

    existing = existing.copy()
    for col in key_cols:
        if col not in existing.columns:
            if col == "wealth_bucket":
                existing[col] = "OVERALL"
            else:
                existing[col] = np.nan
    for col in ["date", "block", "wealth_bucket", "strategy"]:
        if col in existing.columns:
            existing[col] = existing[col].astype(str)

    df_new = df_new.copy()
    for col in key_cols:
        if col not in df_new.columns:
            df_new[col] = np.nan
    for col in ["date", "block", "wealth_bucket", "strategy"]:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype(str)

    combined = pd.concat([existing, df_new], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=key_cols, keep="last")

    columns_order = list(df_new.columns) + [c for c in combined.columns if c not in df_new.columns]
    combined = combined[columns_order]

    combined.to_csv(out_path, index=False)
    log(f"[wrote] Summary CSV (upserted): {out_csv}")


def print_l1_zero_diagnostics(
    df_res_block: Optional[pd.DataFrame],
    blk: int,
    date_str: str,
    *,
    counts: Optional[Dict[str, Tuple[int, int]]] = None,
) -> None:
    """Print L1 distance zero-count diagnostics for a block."""
    prefix = f"[diag][block {blk} {date_str}]"
    if counts is not None:
        zb, tb = counts.get("better", (0, 0))
        zs, ts = counts.get("safer", (0, 0))
        zsh, tsh = counts.get("sharpe", (0, 0))

        def pct(z: int, t: int) -> float:
            return float(z / t * 100.0) if t else np.nan

        log(
            f"{prefix} rows l1=0 "
            f"better {zb}/{tb} ({pct(zb, tb):.2f}%) "
            f"safer {zs}/{ts} ({pct(zs, ts):.2f}%) "
            f"sharpe {zsh}/{tsh} ({pct(zsh, tsh):.2f}%)"
        )
        return

    if df_res_block is None or df_res_block.empty:
        log(f"{prefix} no rows for diagnostics.")
        return

    def zero_stats(series: pd.Series) -> Tuple[int, int, float, pd.Series]:
        non_null = series.dropna()
        zeros = int((non_null == 0).sum())
        total = int(len(non_null))
        pct = float(zeros / total * 100.0) if total else np.nan
        return zeros, total, pct, non_null

    zb, tb, pb, _ = zero_stats(df_res_block["l1_gap_better_return"])
    zs, ts, ps, _ = zero_stats(df_res_block["l1_gap_safer_risk"])
    zsh, tsh, psh, _ = zero_stats(df_res_block["l1_gap_max_sharpe"])

    log(
        f"{prefix} rows l1=0 "
        f"better {zb}/{tb} ({pb:.2f}%) "
        f"safer {zs}/{ts} ({ps:.2f}%) "
        f"sharpe {zsh}/{tsh} ({psh:.2f}%)"
    )


def print_block_summary_table(df_summary: pd.DataFrame, title: str) -> None:
    """Print a formatted summary table for a block or overall view."""
    if df_summary.empty:
        log(f"{title}: (no data)")
        return

    df = df_summary.copy()

    for c in ["baseline_mean", "baseline_median", "strategy_mean", "strategy_median"]:
        df[c] = (df[c] * 100.0).round(4)

    df["hit_rate"] = (df["hit_rate"] * 100.0).round(2)

    df["l1_mean"] = df["l1_mean"].round(4)
    df["l1_median"] = df["l1_median"].round(4)

    if "diff_mean_pp" in df.columns:
        df["diff_mean_pp"] = df["diff_mean_pp"].round(4)
    if "diff_median_pp" in df.columns:
        df["diff_median_pp"] = df["diff_median_pp"].round(4)
    if "score_pp" in df.columns:
        df["score_pp"] = df["score_pp"].round(4)

    for capm_col in ["beta_mean", "beta_median", "market_return_mean", "market_return_median"]:
        if capm_col in df.columns:
            df[capm_col] = df[capm_col].round(4)

    cols = [
        "date",
        "block",
        "strategy",
        "baseline_mean",
        "baseline_median",
        "strategy_mean",
        "strategy_median",
        "diff_mean_pp",
        "diff_median_pp",
        "score_pp",
        "count_optimized",
        "hit_rate",
        "l1_mean",
        "l1_median",
        "strategy_status",
        "beta_mean",
        "beta_median",
        "market_return_mean",
        "market_return_median",
    ]
    cols = [c for c in cols if c in df.columns]

    log("")
    log("=" * 130)
    log(title)
    log("=" * 130)
    log(df[cols].to_string(index=False))


def compute_l1_zero_counts_df(df: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    """Compute counts of zero L1 distances for each strategy column."""
    if df is None or df.empty:
        return {}

    out: Dict[str, Tuple[int, int]] = {}
    cols = {
        "better": "l1_gap_better_return",
        "safer": "l1_gap_safer_risk",
        "sharpe": "l1_gap_max_sharpe",
    }
    for key, col in cols.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        out[key] = (int((s == 0).sum()), int(len(s)))
    return out


def _log_batch_progress(
    blk: int,
    batch_idx: int,
    scanned_rows: int,
    eligible_in_batch: int,
    eligible_selected: int,
    processed_wallets: int,
    max_wallets: Optional[int],
) -> None:
    """Log the standardized batch progress line."""
    max_label = f"{max_wallets:,}" if max_wallets is not None else "ALL"
    log(
        f"[block {blk}] batch {batch_idx} scanned={scanned_rows:,} eligible={eligible_in_batch:,} "
        f"selected_total={eligible_selected:,}/{max_label} processed={processed_wallets:,}/{eligible_selected:,}"
    )


def _log_batch_timeout(
    blk: int,
    batch_idx: int,
    batch_timeout_minutes: int,
    completed_in_batch: int,
    total_tasks: int,
    inflight: int,
    submitted_count: int,
) -> None:
    """Log a standardized batch-timeout warning."""
    log(
        f"[warn][block {blk}] batch {batch_idx} TIMEOUT after {batch_timeout_minutes} min: "
        f"completed={completed_in_batch}/{total_tasks}, inflight={inflight}, "
        f"submitted={submitted_count}, will cancel+skip remainder"
    )


def _log_batch_timeout_details(
    blk: int,
    batch_idx: int,
    skipped_count: int,
    cancelled_count: int,
    wrote_records: int,
) -> None:
    """Log standardized timeout details after batch processing."""
    log(
        f"[warn][block {blk}] batch {batch_idx} timed out; "
        f"skipped_wallets={skipped_count}, cancelled_futures={cancelled_count}, "
        f"wrote_records={wrote_records}"
    )


# ------------------------ Backtest Orchestration ---------------------

def run_backtest_all_blocks(
    holdings_parquet: str,
    block_to_date_csv: str,
    prices_dir: str,
    config: RiskConfig,
    mean_reversion_strength: float,
    workers: int,
    out_csv: Optional[str],
    out_wallet_parquet: str,
    *,
    blocks: Optional[List[int]] = None,
    regime_lookback_days: int = 60,
    regime_top_k_assets: int = 30,
    batch_size: int = 50_000,
    batch_timeout_minutes: int = 15,
    max_wallets: Optional[int] = None,
    supply_lookup: Optional[Dict[str, List[Tuple[int, float]]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the backtest over all blocks and return summary DataFrames."""
    if supply_lookup is None:
        supply_lookup = {}
    log("[info] Using PyArrow streaming for batching and summaries.")
    try:
        block_map = load_block_to_date_map(block_to_date_csv)
        block_dates = dict(zip(block_map["block_number"], block_map["date"]))

        batch_timeout_s = float(batch_timeout_minutes) * 60.0
        overall_stats = OptTimingStats()
        out_wallet_root = Path(out_wallet_parquet)
        out_wallet_root.mkdir(parents=True, exist_ok=True)
        if blocks is not None:
            blocks_to_process = list(blocks)
            if not blocks_to_process:
                log("[warn] No blocks provided after parsing --blocks; nothing to do.")
                return pd.DataFrame(), pd.DataFrame()
            log(f"[blocks] Using user-provided blocks (count={len(blocks_to_process)}): {blocks_to_process}")
        else:
            blocks_to_process = discover_blocks_in_holdings(holdings_parquet)

        if not blocks_to_process:
            log("[warn] No blocks to process.")
            return pd.DataFrame(), pd.DataFrame()

        per_block_overall_rows: List[pd.DataFrame] = []
        per_block_bucket_rows: List[pd.DataFrame] = []

        for bi, blk in enumerate(blocks_to_process, 1):
            log(f"\n[block {bi}/{len(blocks_to_process)}] block={blk}")
            block_stats = OptTimingStats()
            market_regime = "UNKNOWN"
            regime_info = None
            block_date_val = block_dates.get(int(blk))
            if block_date_val is None:
                log(f"[warn][block {blk}] Missing date in block_to_date_map; skipping block.")
                print_opt_timing(prefix=f"[timing][block {blk}]", stats=block_stats)
                continue

            block_date = pd.to_datetime(block_date_val).tz_localize(None)
            block_date_str = block_date.date().isoformat()
            log(f"[block {blk}] date={block_date_str}")

            # Compute forward market return once per block
            forward_market_return = compute_forward_market_return(
                snapshot_date=block_date,
                horizon_days=HORIZON_DAYS,
                prices_dir=prices_dir,
            )
            market_return_20d = forward_market_return if forward_market_return is not None else float("nan")
            log(f"[capm] forward_market_return_20d={market_return_20d:.6g}")

            processed_wallets = 0
            eligible_selected = 0
            columns_needed = ["wallet_address", "block_number", "holdings", "total_value_usd", "num_tokens"]
            block_wallet_dir = out_wallet_root / f"block_number={blk}"
            block_wallet_dir.mkdir(parents=True, exist_ok=True)
            wallet_file_path = block_wallet_dir / "data.parquet"
            wallet_writer: Optional[pq.ParquetWriter] = None
            all_block_results: List[Dict[str, Any]] = []

            try:
                for batch_idx, df_batch in enumerate(
                    iter_holdings_wallet_batches_for_block(
                        holdings_parquet=holdings_parquet,
                        block_number=int(blk),
                        batch_size=int(batch_size),
                        columns=columns_needed,
                        max_wallets=max_wallets,
                    ),
                    start=1,
                ):
                    scanned_rows = len(df_batch)
                    if scanned_rows == 0:
                        continue

                    if "block_number" in df_batch.columns:
                        df_batch["block_number"] = (
                            pd.to_numeric(df_batch["block_number"], errors="coerce")
                            .fillna(int(blk))
                            .astype(int)
                        )
                    else:
                        df_batch["block_number"] = int(blk)
                    df_batch["date"] = block_date

                    if "num_tokens" in df_batch.columns:
                        df_batch["num_tokens"] = pd.to_numeric(df_batch["num_tokens"], errors="coerce")
                    else:
                        df_batch["num_tokens"] = np.nan
                    if "total_value_usd" in df_batch.columns:
                        df_batch["total_value_usd"] = pd.to_numeric(df_batch["total_value_usd"], errors="coerce")
                    else:
                        df_batch["total_value_usd"] = np.nan

                    eligible = df_batch[df_batch["num_tokens"].fillna(0) >= 2].copy()
                    eligible["holdings_map"] = eligible["holdings"].apply(_safe_hm)
                    eligible = eligible[eligible["holdings_map"].map(bool)]

                    if max_wallets is not None:
                        remaining = int(max_wallets) - eligible_selected
                        if remaining <= 0:
                            log(f"[block {blk}] max_wallets cap reached; stopping further batches.")
                            break
                        if len(eligible) > remaining:
                            eligible = eligible.iloc[:remaining].copy()

                    eligible_in_batch = len(eligible)
                    eligible_selected += eligible_in_batch

                    if eligible.empty:
                        _log_batch_progress(
                            blk,
                            batch_idx,
                            scanned_rows,
                            eligible_in_batch,
                            eligible_selected,
                            processed_wallets,
                            max_wallets,
                        )
                        if max_wallets is not None and eligible_selected >= int(max_wallets):
                            log(f"[block {blk}] max_wallets cap reached; stopping further batches.")
                            break
                        continue

                    if regime_info is None:
                        idx_tokens = pick_block_index_tokens(eligible, top_k=int(regime_top_k_assets))
                        cfg_reg = dataclasses.replace(config, window_days=int(regime_lookback_days))
                        prices_idx = load_price_window_data(
                            tokens=idx_tokens,
                            snapshot_date=block_date,
                            config=cfg_reg,
                            prices_dir=prices_dir,
                        )
                        regime_info = classify_market_regime_from_prices(
                            prices_idx,
                            lookback_days=int(regime_lookback_days),
                            t_threshold=1.0,
                        )
                        market_regime = str(regime_info["regime"]).upper()
                        log(
                            "[regime] "
                            f"regime={market_regime} "
                            f"N_pool={regime_info.get('n_assets_pool', np.nan)} "
                            f"N_bench={regime_info.get('n_assets_bench', 0)} "
                            f"bench_used={regime_info.get('used_benchmark', False)} "
                            f"bench_tokens={regime_info.get('benchmark_tokens_used', [])} "
                            f"n_days={regime_info.get('n_days', np.nan)} "
                            f"mu={regime_info.get('mu', np.nan):.6g} "
                            f"sigma={regime_info.get('sigma', np.nan):.6g} "
                            f"T={regime_info.get('t_stat', np.nan):.3f} "
                            f"R={regime_info.get('cum_return', np.nan):.6g}"
                        )

                    unique_tokens: set[str] = set()
                    for hm in eligible["holdings_map"]:
                        unique_tokens.update(hm.keys())

                    date_index, token_prices = build_price_window_arrays(
                        tokens=list(unique_tokens),
                        snapshot_date=block_date,
                        config=config,
                        prices_dir=prices_dir,
                    )
                    if not token_prices:
                        log(f"[block {blk}] batch {batch_idx} has no price data for tokens; skipping.")
                        _log_batch_progress(
                            blk,
                            batch_idx,
                            scanned_rows,
                            eligible_in_batch,
                            eligible_selected,
                            processed_wallets,
                            max_wallets,
                        )
                        if max_wallets is not None and eligible_selected >= int(max_wallets):
                            log(f"[block {blk}] max_wallets cap reached; stopping further batches.")
                            break
                        continue

                    # --- CAPM: compute market returns and token betas ---
                    # Build market return series from preloaded arrays (same window)
                    market_ret_series = build_market_return_series_from_arrays(
                        date_index, token_prices, benchmark_tokens=BENCHMARK_TOKENS,
                    )
                    # Build returns DataFrame for all tokens (for beta regression)
                    all_token_prices_df = pd.DataFrame(
                        {t: arr for t, arr in token_prices.items() if len(arr) == len(date_index)},
                        index=date_index,
                    )
                    if not all_token_prices_df.empty:
                        all_token_prices_df = all_token_prices_df.ffill()
                        all_token_returns_df = np.log(all_token_prices_df / all_token_prices_df.shift(1)).iloc[1:]
                    else:
                        all_token_returns_df = pd.DataFrame()

                    if not market_ret_series.empty and not all_token_returns_df.empty:
                        batch_token_betas = compute_token_betas(
                            all_token_returns_df, market_ret_series,
                            min_observations=config.min_periods,
                        )
                    else:
                        batch_token_betas = {}

                    tasks: List[Dict[str, Any]] = []
                    for row in eligible.itertuples(index=False):
                        tasks.append(
                            {
                                "wallet_address": row.wallet_address,
                                "block_number": int(blk),
                                "date": row.date,
                                "total_value_usd": row.total_value_usd,
                                "num_tokens": row.num_tokens,
                                "holdings_map": row.holdings_map,
                            }
                        )

                    total_tasks = len(tasks)
                    if total_tasks == 0:
                        _log_batch_progress(
                            blk,
                            batch_idx,
                            scanned_rows,
                            eligible_in_batch,
                            eligible_selected,
                            processed_wallets,
                            max_wallets,
                        )
                        if max_wallets is not None and eligible_selected >= int(max_wallets):
                            log(f"[block {blk}] max_wallets cap reached; stopping further batches.")
                            break
                        continue

                    batch_results: List[Dict[str, Any]] = []
                    max_inflight = max(int(workers) * 4, 1)
                    task_iter = iter(tasks)
                    futures = []
                    completed_in_batch = 0
                    submitted_count = 0
                    cancelled_count = 0
                    timed_out = False
                    batch_start = time.perf_counter()
                    deadline = batch_start + batch_timeout_s

                    ex = ProcessPoolExecutor(
                        max_workers=int(workers),
                        initializer=_worker_init,
                        initargs=(
                            token_prices,
                            date_index,
                            config,
                            prices_dir,
                            mean_reversion_strength,
                            market_regime,
                            batch_token_betas,
                            market_return_20d,
                            supply_lookup,
                            int(blk),
                        ),
                    )
                    try:
                        for _ in range(min(max_inflight, total_tasks)):
                            try:
                                futures.append(ex.submit(_process_wallet_task, next(task_iter)))
                                submitted_count += 1
                            except StopIteration:
                                break

                        while futures:
                            remaining = deadline - time.perf_counter()
                            if remaining <= 0:
                                timed_out = True
                                break

                            wait_timeout = min(remaining, 30.0)
                            done, _ = wait(futures, timeout=wait_timeout, return_when=FIRST_COMPLETED)
                            if not done:
                                if time.perf_counter() >= deadline:
                                    timed_out = True
                                    break
                                continue

                            for fut in done:
                                futures.remove(fut)
                                try:
                                    record, duration, success_opt = fut.result()
                                except Exception:
                                    continue
                                block_stats.record(duration, success_opt)
                                overall_stats.record(duration, success_opt)
                                processed_wallets += 1
                                completed_in_batch += 1
                                if record is not None:
                                    batch_results.append(record)
                                if processed_wallets % 10_000 == 0:
                                    log(
                                        f"[block {blk}] processed={processed_wallets:,}/{eligible_selected:,} (selected)"
                                    )

                                try:
                                    nxt = next(task_iter)
                                except StopIteration:
                                    nxt = None

                                if nxt is not None:
                                    futures.append(ex.submit(_process_wallet_task, nxt))
                                    submitted_count += 1

                        if timed_out:
                            _log_batch_timeout(
                                blk,
                                batch_idx,
                                batch_timeout_minutes,
                                completed_in_batch,
                                total_tasks,
                                len(futures),
                                submitted_count,
                            )
                            for f in futures:
                                if f.cancel():
                                    cancelled_count += 1
                    finally:
                        if timed_out:
                            ex.shutdown(wait=False, cancel_futures=True)
                        else:
                            ex.shutdown(wait=True, cancel_futures=False)

                    skipped_count = total_tasks - completed_in_batch

                    if batch_results:
                        wallet_df = pd.DataFrame(batch_results)
                        wallet_df["block_number"] = (
                            pd.to_numeric(wallet_df["block_number"], errors="coerce").astype("int64")
                        )
                        wallet_df["date"] = (
                            pd.to_datetime(wallet_df["date"], errors="coerce").dt.tz_localize(None)
                        )
                        wallet_df["total_value_usd"] = pd.to_numeric(
                            wallet_df["total_value_usd"], errors="coerce"
                        )
                        if "initial_variance_daily" in wallet_df.columns:
                            wallet_df["initial_variance_daily"] = pd.to_numeric(
                                wallet_df["initial_variance_daily"], errors="coerce"
                            )
                        if "initial_expected_return_daily" in wallet_df.columns:
                            wallet_df["initial_expected_return_daily"] = pd.to_numeric(
                                wallet_df["initial_expected_return_daily"], errors="coerce"
                            )
                        if "num_tokens" in wallet_df.columns:
                            wallet_df["num_tokens"] = (
                                pd.to_numeric(wallet_df["num_tokens"], errors="coerce").astype("Int64")
                            )
                        wallet_table = pa.Table.from_pandas(wallet_df, preserve_index=False)
                        if wallet_writer is None:
                            wallet_writer = pq.ParquetWriter(
                                where=str(wallet_file_path),
                                schema=wallet_table.schema,
                                compression='zstd'
                            )
                        wallet_writer.write_table(wallet_table)
                        all_block_results.extend(batch_results)

                    _log_batch_progress(
                        blk,
                        batch_idx,
                        scanned_rows,
                        eligible_in_batch,
                        eligible_selected,
                        processed_wallets,
                        max_wallets,
                    )
                    if timed_out:
                        _log_batch_timeout_details(
                            blk,
                            batch_idx,
                            skipped_count,
                            cancelled_count,
                            len(batch_results),
                        )
            finally:
                if wallet_writer is not None:
                    wallet_writer.close()

                if max_wallets is not None and eligible_selected >= int(max_wallets):
                    log(f"[block {blk}] max_wallets cap reached; stopping further batches.")
                    break

            if processed_wallets == 0:
                log(f"[warn][block {blk}] No valid snapshot results, skipping.")
                print_opt_timing(prefix=f"[timing][block {blk}]", stats=block_stats)
                continue

            df_res_block = pd.DataFrame(all_block_results) if all_block_results else pd.DataFrame()
            if df_res_block.empty:
                log(f"[warn][block {blk}] No results after processing.")
                print_opt_timing(prefix=f"[timing][block {blk}]", stats=block_stats)
                continue

            zero_counts = compute_l1_zero_counts_df(df_res_block)
            df_summary_block = summarize_block_results(df_res_block)
            df_summary_bucket = summarize_block_results_by_wealth_bucket(df_res_block)

            df_summary_block = add_diff_status_and_score_pp(df_summary_block, w_mean=0.5, w_median=0.5)
            if not df_summary_block.empty:
                df_summary_block.loc[:, "wealth_bucket"] = "OVERALL"

            if not df_summary_bucket.empty:
                df_summary_bucket = add_diff_status_and_score_pp(df_summary_bucket, w_mean=0.5, w_median=0.5)

            df_block_out = pd.concat([df_summary_block, df_summary_bucket], ignore_index=True)
            if out_csv and not df_block_out.empty:
                upsert_summary_csv(out_csv, df_block_out)
                log(f"Wrote/Upserted {len(df_block_out)} rows to {out_csv} for block {blk}")

            print_l1_zero_diagnostics(df_res_block, blk=int(blk), date_str=block_date_str, counts=zero_counts)
            print_block_summary_table(
                df_summary_block,
                title=f"BLOCK SUMMARY: block={blk} date={block_date_str} (regime={market_regime})",
            )
            print_opt_timing(prefix=f"[timing][block {blk}]", stats=block_stats)

            per_block_overall_rows.append(df_summary_block)
            per_block_bucket_rows.append(df_summary_bucket)

        print_opt_timing(prefix="[timing][overall]", stats=overall_stats)

        df_per_block_overall = (
            pd.concat(per_block_overall_rows, ignore_index=True) if per_block_overall_rows else pd.DataFrame()
        )
        df_per_block_bucket = (
            pd.concat(per_block_bucket_rows, ignore_index=True) if per_block_bucket_rows else pd.DataFrame()
        )

        # Read all wallet-level parquet back for overall summary
        if out_wallet_root.exists() and any(out_wallet_root.rglob("*.parquet")):
            try:
                wallet_ds = ds.dataset(str(out_wallet_root), format="parquet")
                df_all = wallet_ds.to_table().to_pandas()
            except Exception:
                df_all = pd.DataFrame()
        else:
            df_all = pd.DataFrame()

        df_overall_summary = summarize_block_results(df_all)
        df_overall_summary = add_diff_status_and_score_pp(df_overall_summary, w_mean=0.5, w_median=0.5)
        if not df_overall_summary.empty:
            df_overall_summary.loc[:, "date"] = "OVERALL"
            df_overall_summary.loc[:, "block"] = "OVERALL"
            df_overall_summary.loc[:, "wealth_bucket"] = "OVERALL"

        df_overall_bucket_summary = summarize_block_results_by_wealth_bucket(df_all)
        if not df_overall_bucket_summary.empty:
            df_overall_bucket_summary = add_diff_status_and_score_pp(
                df_overall_bucket_summary,
                w_mean=0.5,
                w_median=0.5,
            )
            df_overall_bucket_summary.loc[:, "date"] = "OVERALL"
            df_overall_bucket_summary.loc[:, "block"] = "OVERALL"

        print_block_summary_table(df_overall_summary, title="OVERALL SUMMARY (ALL BLOCKS COMBINED)")

        df_final_summary = pd.concat(
            [df_per_block_overall, df_per_block_bucket, df_overall_summary, df_overall_bucket_summary],
            ignore_index=True,
        )

        preferred_cols = [
            "date",
            "block",
            "wealth_bucket",
            "strategy",
            "baseline_mean",
            "baseline_median",
            "count_optimized",
            "hit_rate",
            "l1_mean",
            "l1_median",
            "strategy_mean",
            "strategy_median",
            "diff_mean_pp",
            "diff_median_pp",
            "score_pp",
            "strategy_status",
            "beta_mean",
            "beta_median",
            "market_return_mean",
            "market_return_median",
        ]
        ordered_cols = [c for c in preferred_cols if c in df_final_summary.columns]
        remaining_cols = [c for c in df_final_summary.columns if c not in ordered_cols]
        df_final_summary = df_final_summary[ordered_cols + remaining_cols]

        if out_csv:
            upsert_summary_csv(out_csv, df_final_summary)

        return pd.DataFrame(), df_final_summary
    finally:
        GLOBAL_TOKEN_PRICES.clear()
        GLOBAL_TOKEN_BETAS.clear()
        GLOBAL_SUPPLY_LOOKUP.clear()


# ------------------------------- CLI ---------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Block-level wallet-by-wallet MPT backtest with CAPM betas."
    )
    p.add_argument("--holdings-parquet", type=str, default=str(DEFAULT_HOLDINGS_PARQUET))
    p.add_argument("--block-to-date-csv", type=str, required=True)
    p.add_argument("--prices-dir", type=str, default="./data/prices")
    p.add_argument(
        "--horizon-days",
        type=int,
        default=20,
        help="Forward evaluation horizon in days (default: 20).",
    )
    p.add_argument(
        "--blocks",
        type=str,
        default=None,
        help="Comma-separated blocks or file path with one block per line.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Wallet batch size when streaming holdings.",
    )
    p.add_argument(
        "--batch-timeout-minutes",
        type=int,
        default=DEFAULT_BATCH_TIMEOUT_MIN,
        help="Per-batch timeout in minutes.",
    )
    p.add_argument(
        "--max-wallets",
        type=int,
        default=None,
        help="Optional per-block cap on eligible wallets processed (after filtering).",
    )

    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Workers for optimization/backtest.")
    p.add_argument("--mean-reversion-strength", type=float, default=0.5)
    p.add_argument("--out-csv", type=str, default=DEFAULT_OUT_CSV)
    p.add_argument("--out-wallet-parquet", type=str, default=DEFAULT_OUT_WALLET_PARQUET)

    p.add_argument("--regime-lookback-days", type=int, default=DEFAULT_REGIME_LOOKBACK_DAYS)
    p.add_argument("--regime-top-k-assets", type=int, default=DEFAULT_REGIME_TOP_K_ASSETS)
    p.add_argument(
        "--supply-csv",
        type=str,
        default="./data/token_stats.csv",
        help="Path to token supply CSV (columns: block, token_address, supply).",
    )
    return p.parse_args()


def main() -> int:
    """Run the custom backtest CLI entrypoint."""
    global HORIZON_DAYS
    t0 = time.perf_counter()
    args = parse_args()

    HORIZON_DAYS = int(args.horizon_days)

    blocks_list = parse_blocks_arg(args.blocks)

    cfg = RiskConfig()
    cfg = dataclasses.replace(
        cfg,
        window_days=60,
        min_periods=45,
        clip_outliers=None,
        return_type="log",
        shrinkage_method="ledoitwolf",
        max_weight_single_asset=0.9,
    )

    log("CONFIG")
    log("------")
    log(f"holdings_parquet : {args.holdings_parquet}")
    log(f"batch_size       : {args.batch_size}")
    log(f"batch_timeout_min: {args.batch_timeout_minutes}")
    max_wallets_str = "ALL" if args.max_wallets is None else f"{args.max_wallets:,}"
    log(f"max_wallets      : {max_wallets_str} (per block)")
    log(f"prices_dir       : {args.prices_dir}")
    log(f"block_to_date    : {args.block_to_date_csv}")
    log(f"blocks           : {blocks_list if blocks_list is not None else 'AUTO'}")
    log(f"horizon_days     : {HORIZON_DAYS}")
    log(f"workers          : {args.workers}")
    log(f"mean_rev_strength: {args.mean_reversion_strength}")
    log(f"regime_lookback_days: {args.regime_lookback_days}")
    log(f"regime_top_k_assets : {args.regime_top_k_assets}")
    log(f"out_csv          : {args.out_csv}")
    log(f"out_wallet_parquet: {args.out_wallet_parquet}")
    log(f"supply_csv       : {args.supply_csv}")

    # Load supply data for market-cap-weight strategy
    supply_lookup: Dict[str, List[Tuple[int, float]]] = {}
    supply_csv_path = Path(args.supply_csv)
    if supply_csv_path.exists():
        log(f"[supply] Loading supply data from {args.supply_csv} ...")
        supply_lookup = load_supply_data(args.supply_csv)
        log(f"[supply] Loaded supply data for {len(supply_lookup)} tokens.")
    else:
        log(f"[WARN] Supply CSV not found at {args.supply_csv}. "
            f"Market-cap-weight will fall back to equal weight.")

    _, df_summary = run_backtest_all_blocks(
        holdings_parquet=args.holdings_parquet,
        block_to_date_csv=args.block_to_date_csv,
        prices_dir=args.prices_dir,
        config=cfg,
        mean_reversion_strength=float(args.mean_reversion_strength),
        blocks=blocks_list,
        workers=int(args.workers),
        out_csv=args.out_csv,
        out_wallet_parquet=args.out_wallet_parquet,
        regime_lookback_days=int(args.regime_lookback_days),
        regime_top_k_assets=int(args.regime_top_k_assets),
        batch_size=int(args.batch_size),
        batch_timeout_minutes=int(args.batch_timeout_minutes),
        max_wallets=args.max_wallets,
        supply_lookup=supply_lookup,
    )

    if df_summary.empty:
        log("No summary produced.")
    else:
        log("\nDone.")

    elapsed = time.perf_counter() - t0
    log(f"[timing] total_runtime_s={elapsed:.2f} ({elapsed/60.0:.2f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
