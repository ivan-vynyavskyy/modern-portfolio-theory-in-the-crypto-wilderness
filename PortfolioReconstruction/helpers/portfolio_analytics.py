"""
portfolio_analytics.py

Purpose
-------
Provide mean-variance portfolio optimization and risk analytics utilities.

What it does
------------
- Prepares and validates portfolio data (holdings, prices, returns).
- Estimates covariance matrices with optional Ledoit-Wolf or diagonal shrinkage.
- Solves max-Sharpe, min-variance, max-return-same-vol, and target-return problems
  with long-only and per-asset cap constraints via SLSQP.
- Computes portfolio risk/return metrics and mean-reversion adjustments.

Inputs
------
- Holdings snapshots (dict of token -> units)
- Price DataFrames and RiskConfig parameters

Outputs
-------
- Optimized weight vectors (numpy arrays)
- Portfolio metrics dictionaries

Notes
-----
- All optimizers fall back to analytical or heuristic solutions when SLSQP fails.
- Covariance matrices are forced positive-definite before optimization.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from helpers.risk_config import RiskConfig


class PortfolioAnalytics:
    """
    A helper class for portfolio analysis operations.
    
    Public Methods:
        compute_portfolio_metrics: Calculate portfolio risk and return metrics
        prepare_portfolio_data: Prepare and validate portfolio data for analysis
        trimmed_mean: Compute a trimmed mean for a return series
        compute_mean_reversion_metrics: Compute AR(1) mean-reversion stats per token
        adjust_expected_returns_for_mean_reversion: Adjust expected returns using mean reversion
        project_long_only_cap: Project weights to long-only with per-asset cap
        ensure_positive_definite: Ensure covariance matrix is positive definite
        estimate_covariance_matrix: Estimate covariance matrix with optional shrinkage
        max_sharpe_weights: Optimize weights to maximize Sharpe ratio
        max_return_same_vol: Maximize return subject to a target variance
        min_variance_weights: Minimize variance (global minimum variance portfolio)
        min_variance_target_return: Minimize variance subject to a target return
        
    Private Methods:
        _calculate_returns: Compute returns from price data
        _apply_missing_data_policy: Handle missing data according to policy
        _safe_inv/_normalize_simplex/_project_to_capped_simplex: Internal math utilities
    """
    
    def __init__(self, config: RiskConfig):
        """
        Initialize PortfolioAnalytics with configuration.
        
        Args:
            config: RiskConfig object containing portfolio analysis parameters
        """
        self.config = config
    
    def compute_portfolio_metrics(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute portfolio risk and return metrics.
        
        Args:
            returns_df: DataFrame of asset returns
            weights: Array of portfolio weights
            
        Returns:
            Dictionary containing metrics and mean returns, or error message
        """
        if returns_df.empty or weights.size == 0:
            return {"error": "Empty inputs for metric computation"}

        w = np.asarray(weights, dtype=float)
        mu = returns_df.mean().values
        cov = self.estimate_covariance_matrix(returns_df)
        cov = self.ensure_positive_definite(cov)

        # handle N=1 safely
        if cov.shape == (1, 1):
            var_d = float(cov[0, 0]) * float(w[0] ** 2)
        else:
            var_d = float(w @ cov @ w)

        ret_d = float(w @ mu)
        vol_d = float(np.sqrt(max(var_d, 0.0)))

        K = float(self.config.annualization_factor)
        ret_a = ret_d * K
        vol_a = vol_d * np.sqrt(K)

        rf_d = float(self.config.risk_free_rate_annual) / K
        sharpe_d = (ret_d - rf_d) / vol_d if vol_d > 0 else 0.0
        sharpe_a = (ret_a - float(self.config.risk_free_rate_annual)) / vol_a if vol_a > 0 else 0.0

        series = returns_df.values @ w
        volatility_path = float(np.std(series, ddof=1)) if len(series) > 1 else 0.0
        return_path = float(np.mean(series)) if len(series) > 0 else 0.0

        metrics = {
            "volatility_daily": vol_d,
            "volatility_annual": vol_a,
            "variance_daily": var_d,
            "expected_return_daily": ret_d,
            "expected_return_annual": ret_a,
            "sharpe_ratio_daily": float(sharpe_d),
            "sharpe_ratio_annual": float(sharpe_a),
            "volatility_path_method": volatility_path,
            "return_path_method": return_path,
        }
        return {"metrics": metrics, "mean_returns": mu.tolist()}
    
    def prepare_portfolio_data(
        self,
        holdings_snapshot: Dict[str, float],
        snapshot_date: pd.Timestamp,
        prices_df: pd.DataFrame,
        *,
        zero_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Prepare and validate portfolio data for analysis.
        
        Args:
            holdings_snapshot: Dictionary mapping token addresses to units held
            snapshot_date: Date of the portfolio snapshot
            prices_df: DataFrame of price data for tokens
            zero_threshold: Minimum holdings to include (optional)
            
        Returns:
            Dictionary containing prepared portfolio data or error message
        """
        if not holdings_snapshot:
            return {"error": "Empty holdings snapshot"}

        if zero_threshold > 0:
            holdings_snapshot = {
                t: float(u) for t, u in holdings_snapshot.items()
                if float(u) > zero_threshold
            }

        tokens = list(holdings_snapshot.keys())
        if len(tokens) < self.config.min_assets:
            return {"error": f"Insufficient assets ({len(tokens)} < {self.config.min_assets})"}

        if prices_df.empty:
            return {"error": "No price data available"}

        available_tokens = [t for t in tokens if t in prices_df.columns]
        if len(available_tokens) < self.config.min_assets:
            return {"error": f"Insufficient price data ({len(available_tokens)} assets)"}

        prices_df = prices_df[available_tokens]
        prices_aligned = self._apply_missing_data_policy(prices_df)
        if len(prices_aligned) < self.config.min_periods:
            return {"error": f"Insufficient time periods ({len(prices_aligned)} < {self.config.min_periods})"}

        final_tokens = list(prices_aligned.columns)
        if len(final_tokens) < self.config.min_assets:
            return {"error": f"Insufficient assets after missing data handling ({len(final_tokens)} < {self.config.min_assets})"}

        filtered_holdings = {t: float(holdings_snapshot[t]) for t in final_tokens if t in holdings_snapshot}
        if len(filtered_holdings) < self.config.min_assets:
            return {"error": f"Insufficient holdings after data alignment ({len(filtered_holdings)} < {self.config.min_assets})"}

        snapshot_prices = prices_aligned.iloc[-1]
        position_values: Dict[str, float] = {}
        total_value = 0.0
        for t in final_tokens:
            if t not in filtered_holdings:
                continue
            price = float(snapshot_prices.get(t, np.nan))
            units = filtered_holdings[t]
            if np.isnan(price):
                continue
            val = units * price
            position_values[t] = val
            total_value += val

        if total_value <= 0:
            return {"error": "Non-positive total portfolio value"}

        valid_tokens = [t for t in final_tokens if t in position_values]
        if len(valid_tokens) < self.config.min_assets:
            return {"error": f"Insufficient valid positions ({len(valid_tokens)} < {self.config.min_assets})"}

        prices_aligned = prices_aligned[valid_tokens]
        returns_df = self._calculate_returns(prices_aligned)
        if returns_df.empty or len(returns_df) < 2:
            return {"error": "Insufficient return data"}

        weights = np.array([position_values[t] / total_value for t in valid_tokens], dtype=float)

        return {
            "tokens": tokens,
            "final_tokens": valid_tokens,
            "prices_aligned": prices_aligned,
            "returns_df": returns_df,
            "weights": weights,
            "position_values": position_values,
            "total_value": float(total_value),
            "periods_used": int(len(returns_df)),
            "window_start": returns_df.index[0].isoformat(),
            "window_end": returns_df.index[-1].isoformat(),
            "filtered_holdings": {t: float(filtered_holdings.get(t, 0.0)) for t in valid_tokens},
            "snapshot_prices": {t: float(snapshot_prices[t]) for t in valid_tokens},
        }

    # -------------------------------------------------------------------------
    # Statistical helpers
    # -------------------------------------------------------------------------
    def trimmed_mean(self, series: pd.Series, frac: float = 0.05) -> float:
        """
        Compute a trimmed mean for a 1D return series.

        Args:
            series: Return series
            frac: Fraction to trim from each tail

        Returns:
            Trimmed mean value
        """
        s = series.dropna()
        n = len(s)
        if n == 0:
            return np.nan

        k = int(np.floor(frac * n))
        if 2 * k >= n:
            return float(s.mean())

        s_sorted = s.sort_values()
        return float(s_sorted.iloc[k : n - k].mean())

    def compute_mean_reversion_metrics(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean-reversion metrics per token using AR(1) stats.

        Args:
            returns_df: DataFrame of asset returns

        Returns:
            DataFrame indexed by token with mean-reversion metrics
        """
        stats: List[Dict[str, Any]] = []
        for token in returns_df.columns:
            series = returns_df[token].dropna()
            if len(series) < 30:
                continue

            x = series.values
            if len(x) < 2:
                continue

            x0 = x[:-1]
            x1 = x[1:]

            std0 = np.std(x0)
            std1 = np.std(x1)
            if std0 == 0.0 or std1 == 0.0:
                ac1 = 0.0
            else:
                corr = np.corrcoef(x0, x1)
                ac1 = float(corr[0, 1]) if np.isfinite(corr[0, 1]) else 0.0

            var_x = np.var(x0)
            if var_x == 0.0:
                beta = np.nan
            else:
                cov_xy = np.cov(x0, x1, bias=True)[0, 1]
                beta_val = cov_xy / var_x
                beta = float(beta_val) if np.isfinite(beta_val) else np.nan

            if isinstance(beta, float) and 0 < beta < 1:
                halflife = -np.log(2.0) / np.log(beta)
            else:
                halflife = np.nan

            stats.append(
                {"token": token, "ac1": ac1, "beta_ar1": beta, "halflife_days": halflife}
            )

        if not stats:
            return pd.DataFrame(columns=["ac1", "beta_ar1", "halflife_days"]).set_index(
                pd.Index([], name="token")
            )

        return pd.DataFrame(stats).set_index("token")

    def adjust_expected_returns_for_mean_reversion(
        self,
        mu: pd.Series,
        mr_stats: pd.DataFrame,
        strength: float,
    ) -> pd.Series:
        """
        Adjust expected returns using mean-reversion strength.

        Args:
            mu: Expected returns series
            mr_stats: Mean-reversion stats DataFrame
            strength: Adjustment strength

        Returns:
            Adjusted expected returns series
        """
        mu = mu.copy()
        if mr_stats is None or mr_stats.empty:
            return mu
        strength = float(strength)
        if strength == 0.0:
            return mu

        common = mu.index.intersection(mr_stats.index)
        if common.empty:
            return mu

        for token in common:
            ac1 = mr_stats.loc[token, "ac1"]
            if pd.isna(ac1):
                continue
            factor = 1.0 + strength * float(ac1)
            factor = float(np.clip(factor, 0.0, 2.0))
            mu[token] = mu[token] * factor

        return mu

    def project_long_only_cap(self, w: np.ndarray, cap: float) -> np.ndarray:
        """
        Project weights into long-only and per-asset cap constraints.

        Args:
            w: Raw weights
            cap: Maximum allowed weight per asset

        Returns:
            Projected weights
        """
        w = np.asarray(w, dtype=float)
        w[~np.isfinite(w)] = 0.0
        w = np.maximum(w, 0.0)

        s = float(w.sum())
        if s <= 0:
            n = len(w)
            w = np.full(n, 1.0 / n)
        else:
            w = w / s

        cap = float(cap)
        for _ in range(200):
            over = w > cap
            if not over.any():
                break

            excess = float((w[over] - cap).sum())
            w[over] = cap

            under = ~over
            under_sum = float(w[under].sum())
            if under_sum <= 1e-18:
                n = len(w)
                w = np.full(n, 1.0 / n)
                w = np.minimum(w, cap)
                w = w / w.sum()
                break

            w[under] += w[under] / under_sum * excess

        w = np.minimum(np.maximum(w, 0.0), cap)
        w = w / w.sum()
        return w
    
    def ensure_positive_definite(
        self, 
        cov_matrix: np.ndarray, 
        min_eigenvalue: float = 1e-8
    ) -> np.ndarray:
        """
        Ensure covariance matrix is positive definite.
        
        Args:
            cov_matrix: Covariance matrix
            min_eigenvalue: Minimum eigenvalue threshold
            
        Returns:
            Positive definite covariance matrix
        """
        if cov_matrix.size == 0:
            return cov_matrix
        
        A = np.atleast_2d(cov_matrix)
        if A.shape == (1, 1):
            return A
        
        A = 0.5 * (A + A.T)
        w, V = np.linalg.eigh(A)
        w = np.maximum(w, min_eigenvalue)
        A_pd = (V @ np.diag(w) @ V.T)
        A_pd = 0.5 * (A_pd + A_pd.T)
        
        return A_pd
    
    def estimate_covariance_matrix(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        Estimate covariance matrix with optional shrinkage.
        
        Args:
            returns_df: DataFrame of returns
            
        Returns:
            Covariance matrix
        """
        if returns_df.empty:
            return np.array([])
        
        X = returns_df.values
        if X.shape[1] == 1:
            # single asset: variance scalar
            var = np.var(X[:, 0], ddof=1)
            return np.array([[var]], dtype=float)

        if self.config.shrinkage_method is None:
            cov = np.cov(X, rowvar=False, ddof=1)
        elif self.config.shrinkage_method == "diag":
            sample = np.cov(X, rowvar=False, ddof=1)
            diag = np.diag(np.diag(sample))
            a = float(self.config.shrinkage_intensity)
            cov = (1 - a) * sample + a * diag
        elif self.config.shrinkage_method == "ledoitwolf":
            cov = LedoitWolf().fit(X).covariance_
        else:
            raise ValueError(f"Unknown shrinkage_method: {self.config.shrinkage_method}")
        
        cov = 0.5 * (cov + cov.T)
        return cov

    # -------------------------------------------------------------------------
    # Optimization helpers
    # -------------------------------------------------------------------------
    def _safe_inv(self, A: np.ndarray) -> np.ndarray:
        """Return inverse or pseudo-inverse if singular."""
        A = np.atleast_2d(A)
        try:
            return np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A)

    def _normalize_simplex(self, w: np.ndarray) -> np.ndarray:
        """Project weights to the probability simplex (sum to 1, non-negative)."""
        w = np.asarray(w, dtype=float)
        s = np.sum(w)
        if s == 0:
            return np.full_like(w, 1.0 / w.size)
        return w / s

    def _project_to_capped_simplex(self, w: np.ndarray, cap: Optional[float]) -> np.ndarray:
        """Project weights to long-only simplex with optional individual cap."""
        w = np.asarray(w, dtype=float)
        w = np.maximum(w, 0.0)
        if cap is not None:
            cap = float(cap)
            w = np.minimum(w, cap)
        s = w.sum()
        if s == 0:
            return np.full_like(w, 1.0 / w.size)
        return w / s

    def max_sharpe_weights(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        *,
        w0: Optional[np.ndarray] = None,
        long_only: Optional[bool] = None,
        cap: Optional[float] = None,
    ) -> np.ndarray:
        """
        Constrained max-Sharpe using SLSQP:
            maximize   (w^T mu - rf_d) / sqrt(w^T Σ w)
            subject to sum(w)=1, (if long_only) w>=0, (if cap) w_i <= cap
        """
        if long_only is None:
            long_only = self.config.long_only
        if cap is None:
            cap = self.config.max_weight_single_asset

        mu = np.asarray(mu, dtype=float)
        cov = np.asarray(cov, dtype=float)
        n = mu.size

        rf_d = float(self.config.risk_free_rate_annual) / float(self.config.annualization_factor)

        def neg_sharpe(w: np.ndarray) -> float:
            w = np.asarray(w, dtype=float)
            ret = float(w @ mu) - rf_d
            var = float(w @ cov @ w)
            if var <= 0:
                return 1e6
            return -(ret / np.sqrt(var))

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        lb = 0.0 if long_only else -1.0
        ub = 1.0 if cap is None else float(cap)
        bounds = [(lb, ub) for _ in range(n)]
        x0 = np.full(n, 1.0 / n, dtype=float) if w0 is None else np.asarray(w0, dtype=float)

        res = minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 200, "ftol": 1e-9, "disp": False},
        )

        if not res.success:
            inv_cov = self._safe_inv(cov)
            v = inv_cov @ (mu - rf_d * np.ones(n))
            w = v
            if long_only:
                w = self._project_to_capped_simplex(w, cap=cap)
            return self._normalize_simplex(w)

        w = np.array(res.x, dtype=float)
        if long_only:
            w = np.clip(w, 0.0, ub)
        return self._normalize_simplex(w)

    def max_return_same_vol(
        self,
        cov: np.ndarray,
        mu: np.ndarray,
        target_var: float,
        *,
        long_only: Optional[bool] = None,
        cap: Optional[float] = None,
    ) -> np.ndarray:
        """
        Maximize expected return given a target variance constraint.
        """
        if long_only is None:
            long_only = self.config.long_only
        if cap is None:
            cap = self.config.max_weight_single_asset

        cov = np.atleast_2d(cov)
        mu = np.asarray(mu, dtype=float)
        n = mu.size

        def neg_return(w: np.ndarray) -> float:
            return -float(w @ mu)

        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, cov=cov, target_var=target_var: float(w @ cov @ w) - target_var},
        ]

        bounds = None
        if long_only:
            ub = cap if cap is not None else 1.0
            bounds = [(0.0, ub)] * n

        x0 = np.full(n, 1.0 / n)

        res = minimize(
            neg_return,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-9, "disp": False},
        )

        if not res.success:
            w_fallback = self.max_sharpe_weights(
                mu=mu,
                cov=cov,
                w0=None,
                long_only=long_only,
                cap=cap,
            )
            return self._normalize_simplex(w_fallback)

        w_opt = np.array(res.x, dtype=float)
        if long_only:
            ub = cap if cap is not None else 1.0
            w_opt = np.clip(w_opt, 0.0, ub)
        return self._normalize_simplex(w_opt)

    def min_variance_weights(
        self,
        cov: np.ndarray,
        *,
        long_only: Optional[bool] = None,
        cap: Optional[float] = None,
    ) -> np.ndarray:
        """
        Find weights that minimize portfolio variance, ignoring expected returns.
        """
        if long_only is None:
            long_only = self.config.long_only
        if cap is None:
            cap = self.config.max_weight_single_asset

        cov = np.asarray(cov, dtype=float)
        n = cov.shape[0]

        def objective(w: np.ndarray) -> float:
            return 0.5 * float(w @ cov @ w)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        lb = 0.0 if long_only else -1.0
        ub = 1.0 if cap is None else float(cap)
        bounds = [(lb, ub) for _ in range(n)]

        x0 = np.full(n, 1.0 / n)

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 200, "ftol": 1e-9, "disp": False},
        )

        if not res.success:
            diags = np.diag(cov)
            inv_vars = 1.0 / np.maximum(diags, 1e-6)
            w = inv_vars / np.sum(inv_vars)
            if cap is not None:
                w = self._project_to_capped_simplex(w, cap)
            return self._normalize_simplex(w)

        w_opt = np.array(res.x, dtype=float)
        w_opt[w_opt < 1e-4] = 0.0
        return self._normalize_simplex(w_opt)

    def min_variance_target_return(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        target_return: float,
        *,
        long_only: Optional[bool] = None,
        cap: Optional[float] = None,
    ) -> np.ndarray:
        """
        Minimize variance subject to achieving a target expected return.
        """
        if long_only is None:
            long_only = self.config.long_only
        if cap is None:
            cap = self.config.max_weight_single_asset

        mu = np.asarray(mu, dtype=float)
        cov = np.asarray(cov, dtype=float)
        n = mu.size

        def objective(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: float(w @ mu) - target_return},
        ]

        lb = 0.0 if long_only else -1.0
        ub = 1.0 if cap is None else float(cap)
        bounds = [(lb, ub) for _ in range(n)]

        x0 = np.full(n, 1.0 / n)

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 200, "ftol": 1e-9, "disp": False},
        )

        if not res.success:
            cons[1] = {"type": "ineq", "fun": lambda w: float(w @ mu) - target_return}
            res_relaxed = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
            )

            if res_relaxed.success:
                w = np.array(res_relaxed.x, dtype=float)
                if long_only and cap is not None:
                    w = np.clip(w, 0.0, float(cap))
                return self._normalize_simplex(w)

            return self.min_variance_weights(cov, long_only=long_only, cap=cap)

        w_opt = np.array(res.x, dtype=float)
        w_opt[w_opt < 1e-4] = 0.0
        return self._normalize_simplex(w_opt)
    
    def _calculate_returns(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            prices_df: DataFrame of price data
            
        Returns:
            DataFrame of returns
        """
        if prices_df.empty or len(prices_df) < 2:
            return pd.DataFrame()
        
        if self.config.return_type == "simple":
            ret = prices_df.pct_change()
        elif self.config.return_type == "log":
            ret = np.log(prices_df / prices_df.shift(1))
        else:
            raise ValueError(f"Unknown return_type: {self.config.return_type}")
        
        ret = ret.dropna()
        
        if self.config.clip_outliers:
            q_low = ret.quantile(self.config.clip_outliers, interpolation="linear")
            q_high = ret.quantile(1 - self.config.clip_outliers, interpolation="linear")
            for col in ret.columns:
                ret[col] = ret[col].clip(q_low[col], q_high[col])
        
        return ret
    
    def _apply_missing_data_policy(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply missing data handling policy.
        
        Args:
            prices_df: DataFrame of price data
            
        Returns:
            DataFrame with missing data handled
        """
        if prices_df.empty:
            return prices_df
        
        if self.config.missing_policy == "ffill_then_inner":
            out = prices_df.ffill()
            out = out.dropna(axis=0, how="any")
            return out
        elif self.config.missing_policy == "inner_only":
            return prices_df.dropna(axis=0, how="any")
        else:
            raise ValueError(f"Unknown missing_policy: {self.config.missing_policy}")
