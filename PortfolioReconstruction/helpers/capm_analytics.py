"""
capm_analytics.py

Purpose
-------
Compute CAPM metrics (betas, expected returns, single-period CAPM alpha) for ERC-20
portfolio analysis at block-level snapshots.

What it does
------------
- Regresses individual token returns against an equal-weighted benchmark to
  obtain per-token betas via OLS.
- Aggregates token betas into portfolio beta using portfolio weights.
- Computes CAPM expected return and single-period CAPM alpha for forward evaluation.
- Builds market return series from on-disk price CSVs or preloaded arrays.

Inputs
------
- Token return DataFrames and market return Series
- Per-token price CSVs in a prices directory
- RiskConfig for window parameters

Outputs
-------
- Token beta dicts, portfolio beta floats, CAPM expected returns, alpha values
- Market return pd.Series for a lookback or forward window

Notes
-----
- Benchmark defaults to WETH + WBTC equal-weighted log returns.
- Forward market return uses forward-fill with a configurable staleness cap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from helpers.risk_config import RiskConfig


# ----------------------------- Token Betas -----------------------------

def compute_token_betas(
    returns_df: pd.DataFrame,
    market_returns: pd.Series,
    min_observations: int = 30,
) -> Dict[str, Dict[str, float]]:
    """
    For each token, regress its returns against market returns using OLS.

    r_token = alpha + beta * r_market + epsilon

    Args:
        returns_df: columns = token addresses, rows = dates, values = log returns
        market_returns: same index as returns_df
        min_observations: minimum overlapping observations required

    Returns:
        dict: {token_address: {"beta": float, "alpha": float, "r_squared": float}}
    """
    result: Dict[str, Dict[str, float]] = {}

    # Align indices
    common_idx = returns_df.index.intersection(market_returns.index)
    if len(common_idx) < min_observations:
        return result

    r_m = market_returns.reindex(common_idx).values
    mask_m = np.isfinite(r_m)

    for token in returns_df.columns:
        r_t = returns_df[token].reindex(common_idx).values
        valid = mask_m & np.isfinite(r_t)
        n_valid = int(valid.sum())

        if n_valid < min_observations:
            continue

        x = r_m[valid]
        y = r_t[valid]

        # OLS: y = alpha + beta * x
        x_mean = x.mean()
        y_mean = y.mean()
        x_centered = x - x_mean

        ss_xx = float(np.dot(x_centered, x_centered))
        if ss_xx < 1e-18:
            continue

        ss_xy = float(np.dot(x_centered, y - y_mean))
        beta = ss_xy / ss_xx
        alpha = y_mean - beta * x_mean

        # R-squared
        y_pred = alpha + beta * x
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 0.0

        if not (np.isfinite(beta) and np.isfinite(alpha)):
            continue

        result[str(token)] = {
            "beta": float(beta),
            "alpha": float(alpha),
            "r_squared": float(r_squared),
        }

    return result


# ----------------------------- Portfolio Beta --------------------------

def compute_portfolio_beta(
    token_betas: Dict[str, Dict[str, float]],
    weights: np.ndarray,
    tokens: List[str],
) -> float:
    """
    Compute portfolio beta as weighted sum of individual token betas.

    Portfolio beta = sum(w_i * beta_i)
    If a token has no beta (insufficient data), use beta=1.0 as fallback.

    Args:
        token_betas: dict from compute_token_betas
        weights: portfolio weight vector
        tokens: list of token addresses (same order as weights)

    Returns:
        Portfolio beta (float)
    """
    port_beta = 0.0
    for i, token in enumerate(tokens):
        w_i = float(weights[i])
        beta_i = token_betas.get(token, {}).get("beta", 1.0)
        port_beta += w_i * beta_i
    return float(port_beta)


# ----------------------------- CAPM Expected Return --------------------

def compute_capm_expected_return(
    portfolio_beta: float,
    market_return: float,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute CAPM expected return.

    E[R] = Rf + beta * (Rm - Rf)

    Args:
        portfolio_beta: portfolio beta
        market_return: realized market return over the horizon
        risk_free_rate: risk-free rate over the same horizon (default 0.0)

    Returns:
        CAPM expected return (float)
    """
    return risk_free_rate + portfolio_beta * (market_return - risk_free_rate)


# ----------------------------- CAPM Alpha ------------------------------

def compute_capm_alpha(
    actual_return: float,
    capm_expected_return: float,
) -> float:
    """
    Compute Jensen's alpha.

    alpha = R_actual - R_expected

    Args:
        actual_return: realized portfolio return
        capm_expected_return: CAPM expected return

    Returns:
        Jensen's alpha (float)
    """
    return actual_return - capm_expected_return


# ----------------------------- Market Returns --------------------------

def build_market_return_series(
    snapshot_date: pd.Timestamp,
    prices_dir: str,
    config: RiskConfig,
    benchmark_tokens: Tuple[str, ...] = (
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
        "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",  # WBTC
    ),
) -> pd.Series:
    """
    Build daily log return series for the market benchmark.

    Loads price data for benchmark tokens, computes equal-weighted log returns.

    Args:
        snapshot_date: the snapshot date for the lookback window
        prices_dir: directory containing per-token price CSVs
        config: RiskConfig with window_days
        benchmark_tokens: tuple of benchmark token addresses

    Returns:
        pd.Series indexed by date with daily log returns
    """
    snapshot_date = pd.to_datetime(snapshot_date).tz_localize(None)
    prices_dir_path = Path(prices_dir)

    end_date = min(
        snapshot_date.normalize(),
        pd.Timestamp.now().normalize(),
    ) - pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=config.window_days)
    date_index = pd.date_range(start=start_date, end=end_date, freq="D")

    bench_prices: Dict[str, pd.Series] = {}
    for token in benchmark_tokens:
        token_file = prices_dir_path / f"{token.lower()}_price.csv"
        if not token_file.exists():
            continue
        try:
            df = pd.read_csv(token_file, usecols=["date", "daily_close_usd"])
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            if df.empty:
                continue
            series = pd.Series(
                df["daily_close_usd"].astype(float).values,
                index=df["date"].values,
            )
            series = series.reindex(date_index).ffill()
            bench_prices[token] = series
        except Exception:
            continue

    if not bench_prices:
        return pd.Series(dtype=float)

    bench_df = pd.DataFrame(bench_prices, index=date_index)
    # Compute log returns
    log_returns = np.log(bench_df / bench_df.shift(1))
    # Equal-weighted mean across benchmark tokens
    market_returns = log_returns.mean(axis=1, skipna=True).dropna()

    return market_returns


def build_market_return_series_from_arrays(
    date_index: pd.DatetimeIndex,
    token_prices: Dict[str, np.ndarray],
    benchmark_tokens: Tuple[str, ...] = (
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
        "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",  # WBTC
    ),
) -> pd.Series:
    """
    Build market return series from preloaded price arrays (avoids re-reading files).

    Args:
        date_index: DatetimeIndex from the price window
        token_prices: dict of token -> price array (same length as date_index)
        benchmark_tokens: tuple of benchmark token addresses

    Returns:
        pd.Series indexed by date with daily log returns
    """
    bench_series: Dict[str, pd.Series] = {}
    for token in benchmark_tokens:
        arr = token_prices.get(token)
        if arr is not None and len(arr) == len(date_index):
            series = pd.Series(arr, index=date_index).ffill()
            bench_series[token] = series

    if not bench_series:
        return pd.Series(dtype=float)

    bench_df = pd.DataFrame(bench_series, index=date_index)
    log_returns = np.log(bench_df / bench_df.shift(1))
    market_returns = log_returns.mean(axis=1, skipna=True).dropna()

    return market_returns


def compute_forward_market_return(
    snapshot_date: pd.Timestamp,
    horizon_days: int,
    prices_dir: str,
    benchmark_tokens: Tuple[str, ...] = (
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
        "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",  # WBTC
    ),
    max_staleness_days: int = 3,
) -> Optional[float]:
    """
    Compute the realized market return from snapshot_date to snapshot_date + horizon_days.

    Uses log return: ln(P_{t+h} / P_t) averaged across benchmark tokens.

    Args:
        snapshot_date: start date
        horizon_days: number of days forward
        prices_dir: directory containing per-token price CSVs
        benchmark_tokens: tuple of benchmark token addresses
        max_staleness_days: maximum days to forward-fill a missing price

    Returns:
        Realized market log return, or None if data unavailable
    """
    snapshot_date = pd.to_datetime(snapshot_date).tz_localize(None)
    horizon_date = snapshot_date + pd.Timedelta(days=int(horizon_days))
    prices_dir_path = Path(prices_dir)

    log_returns: List[float] = []

    for token in benchmark_tokens:
        token_file = prices_dir_path / f"{token.lower()}_price.csv"
        if not token_file.exists():
            continue
        try:
            df = pd.read_csv(token_file, usecols=["date", "daily_close_usd"])
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            df = df.sort_values("date")

            # Get price at snapshot_date (exact or forward-fill)
            p0 = _get_price_at_date(df, snapshot_date, max_staleness_days)
            # Get price at horizon_date
            p1 = _get_price_at_date(df, horizon_date, max_staleness_days)

            if p0 is not None and p1 is not None and p0 > 0 and p1 > 0:
                log_returns.append(float(np.log(p1 / p0)))
        except Exception:
            continue

    if not log_returns:
        return None

    return float(np.mean(log_returns))


def _get_price_at_date(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    max_staleness_days: int,
) -> Optional[float]:
    """Get price at target_date with optional forward-fill from prior data."""
    d = target_date.normalize()

    exact = df[df["date"] == d]
    if not exact.empty:
        return float(exact["daily_close_usd"].iloc[0])

    # Forward-fill: use most recent prior price
    prior = df[df["date"] < d]
    if not prior.empty:
        last_row = prior.iloc[-1]
        delta = (d - last_row["date"]).days
        if delta <= max_staleness_days:
            return float(last_row["daily_close_usd"])

    return None
