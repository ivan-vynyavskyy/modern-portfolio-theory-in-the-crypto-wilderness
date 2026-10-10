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
- Computes CAPM beta/alpha, classifies market regime, and aggregates per-block
  summaries into CSV/parquet outputs. (Whole-dataset aggregates are computed by
  the downstream Analytics scripts from the wallet-level parquet, not here.)

Inputs
------
- holdings parquet (parquet; columns: wallet_address, block_number, holdings,
  total_value_usd, num_tokens)
- block-to-date map (CSV; columns: block_number or block; date/bound_date/block_time_utc)
- prices directory (per-token CSVs named <token>_price.csv with date,daily_close_usd)

Outputs
-------
- summary CSV (per-block summaries; columns as printed in summary tables)
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
import json
import sys
import threading
import time
from collections import OrderedDict, defaultdict
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
# Imports are bare (capm_analytics, portfolio_analytics, risk_config) so the
# script runs from a flat ./src/ directory on the cluster; in the repo layout
# those modules live in helpers/, so put that directory on the path as well.
for _p in (ROOT_DIR, ROOT_DIR / "helpers"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))

from capm_analytics import (
    build_market_return_series_from_arrays,
    compute_forward_market_return,
    compute_portfolio_beta,
    compute_token_betas,
)

import sys
from datetime import datetime


def log(msg: str, *, end: str = "\n") -> None:
    """Write a timestamped line to stdout."""
    print(f"{datetime.now():%H:%M:%S} | {msg}", end=end, flush=True)


def log_exc(msg: str) -> None:
    """Write a timestamped error line to stderr."""
    print(f"{datetime.now():%H:%M:%S} | ERROR: {msg}", file=sys.stderr, flush=True)

from portfolio_analytics import PortfolioAnalytics
from risk_config import RiskConfig


# ----------------------------- Constants -----------------------------
# For PORTFOLIO OPTIMIZATION, WETH is folded into the native-ETH pseudo-token
# (decision 2026-08-25): they are 1:1
# redeemable with the same price process, so treating them as two assets
# created fictitious 2-asset portfolios (ETH+WETH-only pairs, ~2.4% of
# eligible rows) and near-singular covariances. Balances add directly (both
# 18 decimals); the merged position is priced with the ETH price series.
# The MARKET BENCHMARK (WETH+WBTC price series, read from CSVs) is
# deliberately NOT affected. Merging happens parent-side when holdings are
# parsed (extract_holdings_map), so workers only ever see merged maps.
ETH_PSEUDO_TOKEN = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
WETH_TOKEN = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
MERGE_ETH_WETH = True  # --no-merge-eth-weth restores the two-asset view

DEFAULT_HOLDINGS_PARQUET = "./data/portfolio_holdings.parquet"
HORIZON_DAYS = 20
DEFAULT_BATCH_SIZE = 50_000
DEFAULT_BATCH_TIMEOUT_MIN = 15
DEFAULT_WORKERS = 16
DEFAULT_OUT_CSV = "custom_backtest_summary.csv"
DEFAULT_OUT_WALLET_PARQUET = "custom_backtest_wallet_level.parquet"
DEFAULT_REGIME_LOOKBACK_DAYS = 60
DEFAULT_REGIME_TOP_K_ASSETS = 30


# ------------------------------ Holdings ------------------------------

def parse_holdings(raw: Any) -> List[Dict[str, Any]]:
    """Parse the nested holdings column of a 4_0 snapshot (list<struct>,
    arrives as a list/tuple/ndarray of dicts) into a list of token objects."""
    if isinstance(raw, (list, tuple, np.ndarray)):
        return list(raw)
    raise ValueError(f"Unexpected holdings value of type {type(raw).__name__}")


def extract_holdings_map(holdings_field: Any) -> Dict[str, float]:
    """Convert a holdings field into a token->balance map."""
    arr = parse_holdings(holdings_field)

    out: Dict[str, float] = {}
    for item in arr:
        token = str(item.get("token_address", "")).lower().strip()
        if MERGE_ETH_WETH and token == WETH_TOKEN:
            token = ETH_PSEUDO_TOKEN
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

# Each token's full price CSV is parsed at most ONCE per run and kept as a
# sorted Series; every window/date lookup slices this cache. Before this
# cache the same CSVs were re-read from disk for every batch of every block.
# None entries cache missing/unreadable files so they aren't re-probed.
_FULL_PRICE_SERIES: Dict[Tuple[str, str], Optional[pd.Series]] = {}


def _full_price_series(token: str, prices_dir: str) -> Optional[pd.Series]:
    """Full daily-close history for a token (normalized, sorted, unique
    DatetimeIndex — duplicate dates keep the first row, matching the old
    exact-match lookup). Loaded from CSV once per run."""
    key = (token.lower(), prices_dir)
    if key in _FULL_PRICE_SERIES:
        return _FULL_PRICE_SERIES[key]
    price_file = Path(prices_dir) / f"{token.lower()}_price.csv"
    ser: Optional[pd.Series] = None
    if price_file.exists():
        try:
            df = pd.read_csv(price_file, usecols=["date", "daily_close_usd"])
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            df = df.sort_values("date").drop_duplicates("date", keep="first")
            ser = pd.Series(
                df["daily_close_usd"].astype(float).values, index=df["date"].values
            )
        except Exception:
            ser = None
    _FULL_PRICE_SERIES[key] = ser
    return ser


def get_price_for_date(
    token: str,
    target_date: pd.Timestamp,
    prices_dir: str,
    *,
    forward_fill: bool,
    max_fill_days: Optional[int],
) -> Optional[float]:
    """Return the price for a token on a date, with optional forward-fill."""
    ser = _full_price_series(token, prices_dir)
    if ser is None or ser.empty:
        return None

    d = pd.to_datetime(target_date).normalize()
    idx = int(ser.index.searchsorted(d, side="right")) - 1
    if idx < 0:
        return None
    last_date = ser.index[idx]
    if last_date == d:
        return float(ser.iloc[idx])

    if forward_fill:
        delta_days = (d - last_date).days
        if max_fill_days is not None and delta_days > max_fill_days:
            return None
        return float(ser.iloc[idx])

    return None


def _window_date_index(snapshot_date: pd.Timestamp, window_days: int) -> pd.DatetimeIndex:
    """The lookback window index: window_days ending the day BEFORE snapshot."""
    snapshot_date = pd.to_datetime(snapshot_date).tz_localize(None)
    end_date = min(snapshot_date.normalize(), pd.Timestamp.now().normalize()) - pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=window_days)
    return pd.date_range(start=start_date, end=end_date, freq="D")


def load_price_window_data(
    tokens: List[str],
    snapshot_date: pd.Timestamp,
    config: RiskConfig,
    prices_dir: str,
    required_cols: Tuple[str, str] = ("date", "daily_close_usd"),
) -> pd.DataFrame:
    """Build a historical price window for a set of tokens (cached CSVs).

    A token is included iff it has at least one price inside the window —
    the same membership rule as the old per-call CSV filtering.
    """
    date_index = _window_date_index(snapshot_date, config.window_days)

    series_map: Dict[str, pd.Series] = {}
    for token in tokens:
        ser = _full_price_series(token, prices_dir)
        if ser is None or ser.empty:
            continue
        window = ser.reindex(date_index)
        if int(window.notna().sum()) == 0:
            continue
        series_map[token] = window

    if not series_map:
        return pd.DataFrame(index=date_index)

    return pd.DataFrame(series_map, index=date_index)


def build_price_window_arrays(
    tokens: List[str],
    snapshot_date: pd.Timestamp,
    config: RiskConfig,
    prices_dir: str,
) -> Tuple[pd.DatetimeIndex, Dict[str, np.ndarray]]:
    """Build a price window as aligned arrays by token (cached CSVs)."""
    date_index = _window_date_index(snapshot_date, config.window_days)

    arrays: Dict[str, np.ndarray] = {}
    for token in tokens:
        ser = _full_price_series(token, prices_dir)
        if ser is None or ser.empty:
            continue
        window = ser.reindex(date_index)
        if int(window.notna().sum()) == 0:
            continue
        arrays[token] = window.values

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


def load_eth_supply_override(
    csv_path: str,
    block_map: pd.DataFrame,
    max_fill_days: int = 7,
) -> Tuple[List[Tuple[int, float]], List[int]]:
    """Per-snapshot ETH supply from a DAILY circulating-supply CSV.

    The CSV needs a `date` column and a `circulating_supply` (or `supply`)
    column, e.g. the <pseudo>_supply.csv of eth_src/5_0_fetch_eth_price_history.py.
    Every block of `block_map` (block_number, date) gets the value on its date,
    or the latest value up to `max_fill_days` earlier. Returns the sorted
    (block, supply) list plus the blocks that could not be resolved.

    Why an override: the inequality CSV's ETH supply is the sum of all
    execution-layer balances, which after Shapella double counts withdrawn ETH
    (the beacon deposit contract keeps every ETH ever deposited while the
    withdrawals are credited again), about +31% by December 2025. Market
    capitalisation is price times circulating supply."""
    df = pd.read_csv(csv_path)
    col = next((c for c in ("circulating_supply", "supply") if c in df.columns), None)
    if col is None or "date" not in df.columns:
        raise ValueError(f"{csv_path}: need columns 'date' and 'circulating_supply' (or 'supply'); got {list(df.columns)}")
    ser = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(),
                    index=pd.to_datetime(df["date"], errors="coerce").dt.normalize())
    ser = ser[~ser.index.isna()].dropna().sort_index()
    ser = ser[~ser.index.duplicated(keep="last")]
    entries: List[Tuple[int, float]] = []
    missing: List[int] = []
    for blk, date in zip(block_map["block_number"], block_map["date"]):
        d = pd.Timestamp(date).normalize()
        window = ser.loc[d - pd.Timedelta(days=int(max_fill_days)): d]
        if window.empty or float(window.iloc[-1]) <= 0:
            missing.append(int(blk))
            continue
        entries.append((int(blk), float(window.iloc[-1])))
    entries.sort()
    return entries, sorted(missing)


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

def _prepare_set_payload(
    analytics: PortfolioAnalytics,
    config: RiskConfig,
    candidate_tokens: List[str],
    snapshot_date: pd.Timestamp,
    prices_df: pd.DataFrame,
    mu_shrink_lambda: float,
    mu_trim_frac: float,
) -> Optional[Dict[str, Any]]:
    """Token-SET-level math, identical for every wallet holding the same
    ordered set of priced tokens: aligned returns, shrunk covariance,
    cross-mean-shrunk expected returns, snapshot prices, and the
    wallet-independent max-Sharpe weights. None when the set can't be
    optimized (insufficient prices/periods/assets)."""
    # Dummy unit balances: prepare_portfolio_data's set-level outputs
    # (alignment, returns, snapshot prices) don't depend on balances, and its
    # only wallet-level outputs (weights/total) are recomputed in
    # _solve_wallet. zero_threshold=0 because callers pre-filter holdings.
    prep = analytics.prepare_portfolio_data(
        holdings_snapshot={t: 1.0 for t in candidate_tokens},
        snapshot_date=snapshot_date,
        prices_df=prices_df,
        zero_threshold=0.0,
    )
    if "error" in prep:
        return None

    tokens = prep["final_tokens"]
    returns_df = prep["returns_df"]
    if len(tokens) < 2:
        return None

    cov = analytics.ensure_positive_definite(analytics.estimate_covariance_matrix(returns_df))

    mu_raw = returns_df.apply(lambda s: analytics.trimmed_mean(s, frac=float(mu_trim_frac)), axis=0)
    # Bayes-Stein-style shrinkage toward this asset set's cross-sectional
    # mean (the paper's return regularisation, lambda default 0.5). Replaced the AR(1)
    # damping on 2026-08-25.
    mu_adj = analytics.shrink_expected_returns_to_cross_mean(
        mu_raw, lam=mu_shrink_lambda,
    )
    mu = mu_adj.loc[returns_df.columns].values

    w_max_sharpe = analytics.max_sharpe_weights(
        mu=mu,
        cov=cov,
        long_only=config.long_only,
        cap=config.max_weight_single_asset,
    )

    return {
        "final_tokens": tokens,
        "returns_df": returns_df,
        "snapshot_prices": prep["snapshot_prices"],
        "cov": cov,
        "mu": mu,
        # plain (untrimmed) means, as compute_portfolio_metrics uses for the
        # initial-portfolio expected return
        "metrics_mu": returns_df.mean().values,
        "w_max_sharpe": w_max_sharpe,
    }


def _solve_wallet(
    payload: Dict[str, Any],
    holdings_map: Dict[str, float],
    analytics: PortfolioAnalytics,
    config: RiskConfig,
) -> Optional[Dict[str, Any]]:
    """Wallet-level part of the optimization: weights from this wallet's
    balances, the two target-constrained solves, units and L1 gaps."""
    tokens: List[str] = payload["final_tokens"]
    returns_df: pd.DataFrame = payload["returns_df"]
    snapshot_prices: Dict[str, float] = payload["snapshot_prices"]
    cov = payload["cov"]
    mu = payload["mu"]

    filtered_holdings0 = {t: float(holdings_map[t]) for t in tokens}
    position_values: Dict[str, float] = {}
    total_value0 = 0.0
    for t in tokens:
        val = filtered_holdings0[t] * float(snapshot_prices[t])
        position_values[t] = val
        total_value0 += val
    if total_value0 <= 0:
        return None

    w_cur_raw = np.array([position_values[t] / total_value0 for t in tokens], dtype=float)
    w_cur = analytics.project_long_only_cap(w_cur_raw, cap=float(config.max_weight_single_asset))

    # Initial-portfolio metrics: same math as compute_portfolio_metrics, but
    # reusing the payload's covariance instead of re-shrinking it per wallet.
    initial_variance_daily = np.nan
    initial_expected_return_daily = np.nan
    try:
        if not returns_df.empty and w_cur.size:
            initial_variance_daily = float(w_cur @ cov @ w_cur)
            initial_expected_return_daily = float(w_cur @ payload["metrics_mu"])
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
    # True when the better-return solve silently substituted max-Sharpe
    # weights (2-asset target infeasible or failed SLSQP); surfaced as the
    # wallet-level better_return_fallback_used column so the substitution
    # rate is measurable per block.
    better_fallback_used = bool(getattr(analytics, "last_mrv_fallback", False))

    w_safer_risk = analytics.min_variance_target_return(
        mu=mu,
        cov=cov,
        target_return=current_return,
        long_only=config.long_only,
        cap=config.max_weight_single_asset,
    )
    # True when BOTH the equality and the relaxed >= solve failed and the
    # global min-variance portfolio was substituted (return constraint
    # dropped entirely).
    safer_fallback_used = bool(getattr(analytics, "last_minvar_fallback", False))

    w_max_sharpe = payload["w_max_sharpe"]

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
        "better_return_fallback_used": better_fallback_used,
        "safer_risk_fallback_used": safer_fallback_used,
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
    """Parse a holdings field into a map; return {} on failure."""
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
    # partitioning="hive" lets this accept the block-partitioned dataset built by
    # 4_2 (block_number lives in the directory name); harmless on a single file.
    holdings_ds = ds.dataset(holdings_parquet, format="parquet", partitioning="hive")
    for required_col in ["wallet_address", "block_number", "holdings"]:
        if required_col not in holdings_ds.schema.names:
            raise ValueError(f"Column {required_col!r} not found in holdings parquet.")

    cols = [c for c in columns if c in holdings_ds.schema.names]
    if "wallet_address" not in cols:
        cols.append("wallet_address")
    if "block_number" not in cols:
        cols.append("block_number")

    filter_expr = ds.field("block_number") == int(block_number)
    if "num_tokens" in holdings_ds.schema.names:
        # Predicate pushdown: single-token rows (~74% of the ETH-era dataset)
        # are skipped at scan time — before the fat `holdings` column is
        # decoded and converted to pandas — instead of being materialized
        # just to be dropped by the num_tokens >= 2 eligibility filter in
        # the caller (which stays, as belt and braces). NULL num_tokens rows
        # are dropped by both. Note: `scanned=` in batch log lines therefore
        # counts post-pushdown rows, not raw dataset rows.
        filter_expr = filter_expr & (ds.field("num_tokens") >= 2)
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
    holdings_ds = ds.dataset(holdings_parquet, format="parquet", partitioning="hive")
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
GLOBAL_MU_SHRINK: float = 0.0
GLOBAL_MARKET_REGIME: str = "UNKNOWN"
GLOBAL_TOKEN_BETAS: Dict[str, Dict[str, float]] = {}
GLOBAL_MARKET_RETURN_20D: float = float("nan")
# Precomputed once per batch in the parent (all wallets in a block share the
# same snapshot/horizon dates): horizon-date price and market cap per token.
# Workers do dict lookups instead of per-wallet CSV filtering/bisecting.
GLOBAL_HORIZON_PRICES: Dict[str, float] = {}
GLOBAL_MCAPS: Dict[str, Optional[float]] = {}
GLOBAL_ANALYTICS: Optional[PortfolioAnalytics] = None

# Per-worker LRU of _prepare_set_payload results, keyed by the ORDERED tuple
# of a wallet's priced tokens. Ordered (not sorted) so the covariance/solver
# variable order — and therefore every result bit — matches the uncached
# path. Wallets holding the same token set (dominant once native ETH joins:
# millions of {ETH, tokenX} pairs) skip the alignment + Ledoit-Wolf +
# trimmed-mean + mu-shrinkage + max-Sharpe work entirely. The size cap bounds
# worker memory on batches full of distinct exotic sets.
_SET_CACHE: "OrderedDict[Tuple[str, ...], Optional[Dict[str, Any]]]" = OrderedDict()
_SET_CACHE_MAX = 20_000

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
    mu_shrink_lambda: float,
    market_regime: str,
    token_betas: Dict[str, Dict[str, float]],
    market_return_20d: float,
    horizon_prices: Dict[str, float],
    mcap_by_token: Dict[str, Optional[float]],
) -> None:
    """Initialize global state for worker processes."""
    global GLOBAL_TOKEN_PRICES, GLOBAL_DATE_INDEX, GLOBAL_CONFIG, GLOBAL_PRICES_DIR
    global GLOBAL_MU_SHRINK, GLOBAL_MARKET_REGIME
    global GLOBAL_TOKEN_BETAS, GLOBAL_MARKET_RETURN_20D
    global GLOBAL_HORIZON_PRICES, GLOBAL_MCAPS, GLOBAL_ANALYTICS
    GLOBAL_TOKEN_PRICES = token_prices
    GLOBAL_DATE_INDEX = date_index
    GLOBAL_CONFIG = config
    GLOBAL_PRICES_DIR = prices_dir
    GLOBAL_MU_SHRINK = float(mu_shrink_lambda)
    GLOBAL_MARKET_REGIME = str(market_regime).upper()
    GLOBAL_TOKEN_BETAS = token_betas
    GLOBAL_MARKET_RETURN_20D = float(market_return_20d)
    GLOBAL_HORIZON_PRICES = horizon_prices
    GLOBAL_MCAPS = mcap_by_token
    GLOBAL_ANALYTICS = PortfolioAnalytics(config)
    # the cached payloads are only valid for this batch's price arrays
    _SET_CACHE.clear()


def _value_from_price_map(
    units: Dict[str, float],
    price_map: Dict[str, float],
) -> Optional[float]:
    """USD value of holdings units from a precomputed {token: price} map.

    None as soon as any position's price is missing, 0.0 for empty units.
    A NaN price propagates into the total exactly as it did through the
    per-CSV lookup path.
    """
    if not units:
        return 0.0
    total = 0.0
    for token, amount in units.items():
        price = price_map.get(token)
        if price is None:
            return None
        total += float(amount) * float(price)
    return total


def _optimize_wallet_cached(
    holdings_map: Dict[str, float],
    snapshot_date: pd.Timestamp,
) -> Optional[Dict[str, Any]]:
    """Worker-side optimize: the token-set payload is computed once per
    distinct set and reused for every other wallet holding the same set."""
    config = GLOBAL_CONFIG
    analytics = GLOBAL_ANALYTICS
    if config is None or analytics is None or not holdings_map:
        return None

    held = holdings_map
    if config.zero_threshold > 0:
        held = {t: u for t, u in holdings_map.items()
                if float(u) > config.zero_threshold}
        if not held:
            return None

    key = tuple(t for t in held if t in GLOBAL_TOKEN_PRICES)
    if key in _SET_CACHE:
        _SET_CACHE.move_to_end(key)
        payload = _SET_CACHE[key]
    else:
        prices_df = _build_prices_df_for_tokens(list(key))
        payload = None
        if not prices_df.empty:
            payload = _prepare_set_payload(
                analytics, config, list(key), snapshot_date, prices_df,
                GLOBAL_MU_SHRINK, 0.05,
            )
        _SET_CACHE[key] = payload
        if len(_SET_CACHE) > _SET_CACHE_MAX:
            _SET_CACHE.popitem(last=False)

    if payload is None:
        return None
    return _solve_wallet(payload, held, analytics, config)


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

    success_opt = False
    t_opt = time.perf_counter()
    try:
        opt = _optimize_wallet_cached(holdings_map, snapshot_date)
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

    # Market-cap-weight (mcaps precomputed per block in the parent)
    mcap_valid = False
    w_mcap = None
    if GLOBAL_MCAPS:
        market_caps = []
        for token in opt_tokens:
            mcap = GLOBAL_MCAPS.get(token)
            market_caps.append(mcap if (mcap is not None and mcap > 0) else None)

        # Any token without a market cap keeps mcap_valid False -> all mcap
        # fields stay null for this wallet.
        n_missing = sum(1 for mc in market_caps if mc is None)
        if n_missing == 0:
            total_known_mcap = sum(mc for mc in market_caps if mc is not None)
            if total_known_mcap > 0:
                mcap_array = np.array([mc / total_known_mcap for mc in market_caps])
                w_mcap = mcap_array / mcap_array.sum()
                mcap_valid = True

    # Same units-from-weights math as _solve_wallet's closure
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

    # --- Fixed 20-day forward evaluation (horizon prices precomputed) ---
    v0 = _value_from_price_map(u0, GLOBAL_HORIZON_PRICES)
    v1 = _value_from_price_map(u1, GLOBAL_HORIZON_PRICES)
    v2 = _value_from_price_map(u2, GLOBAL_HORIZON_PRICES)
    v3 = _value_from_price_map(u3, GLOBAL_HORIZON_PRICES)
    v_equal = _value_from_price_map(units_equal, GLOBAL_HORIZON_PRICES)

    r0 = (v0 / total_value0) - 1.0 if (v0 is not None and v0 > 0) else float("nan")
    r1 = (v1 / total_value0) - 1.0 if (v1 is not None and v1 > 0) else float("nan")
    r2 = (v2 / total_value0) - 1.0 if (v2 is not None and v2 > 0) else float("nan")
    r3 = (v3 / total_value0) - 1.0 if (v3 is not None and v3 > 0) else float("nan")
    r_equal = (v_equal / total_value0) - 1.0 if (v_equal is not None and v_equal > 0) else float("nan")

    if mcap_valid:
        v_mcap = _value_from_price_map(units_mcap, GLOBAL_HORIZON_PRICES)
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
        "better_return_fallback_used": bool(opt.get("better_return_fallback_used", False)),
        "safer_risk_fallback_used": bool(opt.get("safer_risk_fallback_used", False)),
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
        fallback_col: Optional[str] = None,
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

        fallback_rate = np.nan
        if fallback_col and fallback_col in group.columns and len(group):
            fallback_rate = float(group[fallback_col].fillna(False).astype(bool).mean())

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
                "fallback_rate": fallback_rate,
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
            "fallback_rate": np.nan,
        }
    )

    add_strategy(
        "Max Return (Same Vol)", "ret_better_return", "l1_gap_better_return",
        beta_col="beta_better_return", fallback_col="better_return_fallback_used",
    )
    add_strategy(
        "Min Variance (Same Ret)", "ret_safer_risk", "l1_gap_safer_risk",
        beta_col="beta_safer_risk", fallback_col="safer_risk_fallback_used",
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
            fallback_col: Optional[str] = None,
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

            fallback_rate = np.nan
            if fallback_col and fallback_col in group.columns and len(group):
                fallback_rate = float(group[fallback_col].fillna(False).astype(bool).mean())

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
                    "fallback_rate": fallback_rate,
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
                "fallback_rate": np.nan,
            }
        )

        add_strategy(
            "Max Return (Same Vol)", "ret_better_return", "l1_gap_better_return",
            beta_col="beta_better_return", fallback_col="better_return_fallback_used",
        )
        add_strategy(
            "Min Variance (Same Ret)", "ret_safer_risk", "l1_gap_safer_risk",
            beta_col="beta_safer_risk", fallback_col="safer_risk_fallback_used",
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


HOLDINGS_JSON_COLUMNS = (
    "holdings_short", "holdings_better_return", "holdings_safer_risk",
    "holdings_max_sharpe", "holdings_equal_weight", "holdings_mcap_weight",
)


def _read_block_results_for_summary(path: Path) -> pd.DataFrame:
    """Re-read a just-written block parquet without the six JSON holdings
    columns (no summary input, but most of the row size), so the per-block
    summary never needs the block's records kept in memory during the run."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    # Read the one file directly. pq.read_table() goes through dataset
    # discovery with hive partitioning, and from pyarrow 17 on that also parses
    # the parent directory name block_number=<blk> into a dictionary<int32>
    # partition field, which then collides with the int64 block_number column
    # stored in the file ("Unable to merge: Field block_number has
    # incompatible types"). ParquetFile never looks at the path.
    pf = pq.ParquetFile(path)
    cols = [c for c in pf.schema_arrow.names if c not in HOLDINGS_JSON_COLUMNS]
    return pf.read(columns=cols).to_pandas()


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
    if "fallback_rate" in df.columns:
        df["fallback_rate"] = df["fallback_rate"].round(4)

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
        "fallback_rate",
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

def _wallet_records_to_table(batch_results: List[Dict[str, Any]]) -> pa.Table:
    """Normalize a batch of wallet records into an Arrow table whose schema
    is stable across batches (the per-block ParquetWriter fixes its schema
    from the first batch and raises on any later mismatch)."""
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
    table = pa.Table.from_pandas(wallet_df, preserve_index=False)
    # holdings_mcap_weight is None for every wallet holding an mcap-less
    # token; a batch where that is ALL wallets infers as pa.null and clashes
    # with a string column from any other batch — pin the type.
    if "holdings_mcap_weight" in wallet_df.columns:
        i = table.schema.get_field_index("holdings_mcap_weight")
        table = table.set_column(
            i,
            pa.field("holdings_mcap_weight", pa.string()),
            table.column(i).cast(pa.string()),
        )
    return table


def run_backtest_all_blocks(
    holdings_parquet: str,
    block_to_date_csv: str,
    prices_dir: str,
    config: RiskConfig,
    mu_shrink_lambda: float,
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
        run_t0 = time.perf_counter()

        for bi, blk in enumerate(blocks_to_process, 1):
            block_stats = OptTimingStats()
            market_regime = "UNKNOWN"
            regime_info = None
            block_date_val = block_dates.get(int(blk))
            if block_date_val is None:
                log(f"\n[warn][block {bi}/{len(blocks_to_process)}] block={blk} "
                    f"missing date in block_to_date_map; skipping block.")
                continue

            block_date = pd.to_datetime(block_date_val).tz_localize(None)
            block_date_str = block_date.date().isoformat()
            log(f"\n[block {bi}/{len(blocks_to_process)}] block={blk} date={block_date_str}")
            block_t0 = time.perf_counter()

            # Compute forward market return once per block
            forward_market_return = compute_forward_market_return(
                snapshot_date=block_date,
                horizon_days=HORIZON_DAYS,
                prices_dir=prices_dir,
            )
            if forward_market_return is None:
                log(f"[warn][block {blk}] forward market return unavailable "
                    f"(benchmark price CSVs missing or horizon window uncovered); "
                    f"market_return and alphas will be NaN for this block.")
            market_return_20d = forward_market_return if forward_market_return is not None else float("nan")
            log(f"[capm] forward_market_return_20d={market_return_20d:.6g}")

            processed_wallets = 0
            eligible_selected = 0
            critical_prices_checked = False
            columns_needed = ["wallet_address", "block_number", "holdings", "total_value_usd", "num_tokens"]
            block_wallet_dir = out_wallet_root / f"block_number={blk}"
            block_wallet_dir.mkdir(parents=True, exist_ok=True)
            wallet_file_path = block_wallet_dir / "data.parquet"
            wallet_writer: Optional[pq.ParquetWriter] = None
            records_written = 0

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
                    # len >= 2: a wallet needs at least two distinct held
                    # tokens to ever be optimized (the payload requires two
                    # PRICED assets). ETH+WETH-only pairs collapse to one
                    # asset under the merge and are dropped here instead of
                    # round-tripping through a worker just to fail.
                    eligible = eligible[eligible["holdings_map"].map(len) >= 2]

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
                        # benchmarks appended: the regime classifier anchors on
                        # WETH+WBTC, which the merged holdings no longer surface
                        prices_idx = load_price_window_data(
                            tokens=list(dict.fromkeys(list(idx_tokens) + list(BENCHMARK_TOKENS))),
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

                    # Benchmarks are loaded unconditionally: the market-return
                    # series and token betas need WETH+WBTC arrays, and with
                    # the ETH/WETH merge no holdings map contains WETH anymore.
                    date_index, token_prices = build_price_window_arrays(
                        tokens=list(unique_tokens | set(BENCHMARK_TOKENS)),
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

                    # Horizon prices + market caps are identical for every
                    # wallet in the block — resolve them once per batch here
                    # (cached CSV series) instead of per wallet in workers.
                    horizon_date = block_date + pd.Timedelta(days=HORIZON_DAYS)
                    horizon_prices: Dict[str, float] = {}
                    for token in unique_tokens:
                        p = get_price_for_date(
                            token, horizon_date, prices_dir,
                            forward_fill=config.forward_fill_prices,
                            max_fill_days=config.max_price_staleness_days,
                        )
                        if p is not None:
                            horizon_prices[token] = float(p)
                    mcap_by_token: Dict[str, Optional[float]] = {}
                    if supply_lookup:
                        for token in unique_tokens:
                            mcap_by_token[token] = get_market_cap(
                                token, int(blk), block_date, prices_dir, supply_lookup,
                            )

                    # Once per block: the assets that MUST be priced (ETH
                    # pseudo-token + benchmarks). A long tail of unpriced
                    # ERC-20s is normal; these missing is a broken run.
                    if not critical_prices_checked:
                        critical_prices_checked = True
                        critical = (ETH_PSEUDO_TOKEN, *BENCHMARK_TOKENS)
                        miss_window = [t for t in critical if t not in token_prices]
                        if miss_window:
                            log(f"[warn][block {blk}] CRITICAL price series missing from "
                                f"lookback window: {miss_window} (ETH pseudo-token / benchmarks)")
                        if ETH_PSEUDO_TOKEN in unique_tokens and ETH_PSEUDO_TOKEN not in horizon_prices:
                            log(f"[warn][block {blk}] no ETH price at horizon date "
                                f"{horizon_date.date().isoformat()}; forward returns for "
                                f"ETH-holding wallets will be NaN.")

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
                            mu_shrink_lambda,
                            market_regime,
                            batch_token_betas,
                            market_return_20d,
                            horizon_prices,
                            mcap_by_token,
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
                                    rate = processed_wallets / max(time.perf_counter() - block_t0, 1e-9)
                                    log(
                                        f"[block {blk}] processed={processed_wallets:,}/{eligible_selected:,} "
                                        f"(selected) rate={rate:,.0f} wallets/s"
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
                        wallet_table = _wallet_records_to_table(batch_results)
                        if wallet_writer is None:
                            wallet_writer = pq.ParquetWriter(
                                where=str(wallet_file_path),
                                schema=wallet_table.schema,
                                compression='zstd'
                            )
                        wallet_writer.write_table(wallet_table)
                        records_written += len(batch_results)

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
                # NOTE: no max_wallets break here — the batch loop above stops
                # itself once the cap is reached; a break inside this finally
                # exits the BLOCK loop, skipping this block's summary and every
                # remaining block.
                if wallet_writer is not None:
                    wallet_writer.close()

            if processed_wallets == 0:
                log(f"[warn][block {blk}] No valid snapshot results, skipping.")
                print_opt_timing(prefix=f"[timing][block {blk}]", stats=block_stats)
                continue

            # The records were streamed to the block's parquet batch by batch;
            # re-read them (without the JSON holdings columns) for the summary
            # instead of holding every record of the block in the parent for
            # the whole block -- that list grew to ~50 GB per 4M wallets and
            # made every per-batch fork of the worker pool slower and slower.
            df_res_block = _read_block_results_for_summary(wallet_file_path) if records_written else pd.DataFrame()
            if df_res_block.empty:
                log(f"[warn][block {blk}] No results after processing.")
                print_opt_timing(prefix=f"[timing][block {blk}]", stats=block_stats)
                continue

            n_res = max(len(df_res_block), 1)
            if "better_return_fallback_used" in df_res_block.columns:
                fb_n = int(df_res_block["better_return_fallback_used"].fillna(False).astype(bool).sum())
                log(f"[block {blk}] better-return solves substituted by max-Sharpe fallback: "
                    f"{fb_n}/{len(df_res_block)} ({fb_n / n_res:.2%})")
            if "safer_risk_fallback_used" in df_res_block.columns:
                fb_n = int(df_res_block["safer_risk_fallback_used"].fillna(False).astype(bool).sum())
                log(f"[block {blk}] safer-risk solves substituted by global min-variance fallback: "
                    f"{fb_n}/{len(df_res_block)} ({fb_n / n_res:.2%})")

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

            block_elapsed = time.perf_counter() - block_t0
            run_elapsed = time.perf_counter() - run_t0
            n_blocks = len(blocks_to_process)
            est_remaining_min = (run_elapsed / bi) * (n_blocks - bi) / 60.0
            log(f"[block {blk}] done: processed={processed_wallets:,} "
                f"records_written={records_written:,} "
                f"elapsed={block_elapsed / 60.0:.1f} min -> {wallet_file_path}")
            log(f"[progress] {bi}/{n_blocks} blocks, run_elapsed={run_elapsed / 60.0:.1f} min, "
                f"est_remaining~{est_remaining_min:.0f} min (rough: blocks differ in size)")

        print_opt_timing(prefix="[timing][overall]", stats=overall_stats)

        # Per-block summaries were already upserted into out_csv as each block
        # finished. The whole-dataset "OVERALL" summary that used to be built
        # here (by reading EVERY wallet-level parquet back into memory) was
        # removed: downstream Analytics recomputes global statistics from the
        # wallet-level dataset itself, and the read-back could not scale to
        # post-ETH row counts.
        df_per_block_overall = (
            pd.concat(per_block_overall_rows, ignore_index=True) if per_block_overall_rows else pd.DataFrame()
        )
        df_per_block_bucket = (
            pd.concat(per_block_bucket_rows, ignore_index=True) if per_block_bucket_rows else pd.DataFrame()
        )
        df_final_summary = pd.concat(
            [df_per_block_overall, df_per_block_bucket], ignore_index=True
        )

        return pd.DataFrame(), df_final_summary
    finally:
        GLOBAL_TOKEN_PRICES.clear()
        GLOBAL_TOKEN_BETAS.clear()
        GLOBAL_HORIZON_PRICES.clear()
        GLOBAL_MCAPS.clear()
        _FULL_PRICE_SERIES.clear()


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
    p.add_argument(
        "--mu-shrinkage-lambda",
        type=float,
        default=0.5,
        help="Linear shrinkage of expected returns toward the wallet set's "
             "cross-asset mean: mu <- lam*mean(mu) + (1-lam)*mu "
             "(default 0.5; 0 = no shrinkage).",
    )
    p.add_argument(
        "--no-merge-eth-weth",
        action="store_true",
        help="Treat WETH as an asset distinct from native ETH (default: WETH "
             "balances are folded into the ETH pseudo-token for optimization).",
    )
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
    p.add_argument(
        "--eth-supply-csv",
        type=str,
        default=None,
        help="Optional DAILY circulating-supply CSV for native ETH (columns: date, "
             "circulating_supply), e.g. <pseudo>_supply.csv from "
             "eth_src/5_0_fetch_eth_price_history.py. Replaces the ETH rows of "
             "--supply-csv for the market-cap-weight strategy.",
    )
    return p.parse_args()


def main() -> int:
    """Run the custom backtest CLI entrypoint."""
    global HORIZON_DAYS, MERGE_ETH_WETH
    t0 = time.perf_counter()
    args = parse_args()

    HORIZON_DAYS = int(args.horizon_days)
    if args.no_merge_eth_weth:
        MERGE_ETH_WETH = False

    mu_lam = float(args.mu_shrinkage_lambda)

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
    log(f"mu_shrink_lambda : {mu_lam}")
    log(f"merge_eth_weth   : {MERGE_ETH_WETH}")
    log(f"risk_free_annual : {cfg.risk_free_rate_annual}")
    log(f"regime_lookback_days: {args.regime_lookback_days}")
    log(f"regime_top_k_assets : {args.regime_top_k_assets}")
    log(f"out_csv          : {args.out_csv}")
    log(f"out_wallet_parquet: {args.out_wallet_parquet}")
    log(f"supply_csv       : {args.supply_csv}")
    log(f"eth_supply_csv   : {args.eth_supply_csv or '-'}")

    # Load supply data for market-cap-weight strategy
    supply_lookup: Dict[str, List[Tuple[int, float]]] = {}
    supply_csv_path = Path(args.supply_csv)
    if supply_csv_path.exists():
        log(f"[supply] Loading supply data from {args.supply_csv} ...")
        supply_lookup = load_supply_data(args.supply_csv)
        log(f"[supply] Loaded supply data for {len(supply_lookup)} tokens.")
    else:
        log(f"[WARN] Supply CSV not found at {args.supply_csv}. "
            f"Market-Cap Weight fields will be NULL for all wallets.")
    if args.eth_supply_csv:
        if not Path(args.eth_supply_csv).exists():
            raise SystemExit(f"[supply] --eth-supply-csv not found: {args.eth_supply_csv}")
        eth_entries, eth_missing = load_eth_supply_override(
            args.eth_supply_csv, load_block_to_date_map(args.block_to_date_csv))
        if eth_entries:
            had = ETH_PSEUDO_TOKEN in supply_lookup
            supply_lookup[ETH_PSEUDO_TOKEN] = eth_entries
            log(f"[supply] ETH supply for {len(eth_entries)} snapshot blocks taken from "
                f"{args.eth_supply_csv} (circulating supply"
                f"{'; replaces the ledger balance sum of --supply-csv' if had else ''}); "
                f"e.g. block {eth_entries[0][0]}: {eth_entries[0][1]:,.0f} ETH, "
                f"block {eth_entries[-1][0]}: {eth_entries[-1][1]:,.0f} ETH")
        if eth_missing:
            log(f"[WARN] --eth-supply-csv has no value within 7 days before {len(eth_missing)} "
                f"snapshot date(s) (blocks {eth_missing[:5]}{'...' if len(eth_missing) > 5 else ''}); "
                f"those keep the --supply-csv value if any.")
    if supply_lookup and ETH_PSEUDO_TOKEN not in supply_lookup:
        log(f"[WARN] No ETH supply available (neither in --supply-csv nor via --eth-supply-csv); "
            f"Market-Cap Weight will be NULL for every ETH-holding wallet (~most of the eligible set).")

    _, df_summary = run_backtest_all_blocks(
        holdings_parquet=args.holdings_parquet,
        block_to_date_csv=args.block_to_date_csv,
        prices_dir=args.prices_dir,
        config=cfg,
        mu_shrink_lambda=mu_lam,
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
