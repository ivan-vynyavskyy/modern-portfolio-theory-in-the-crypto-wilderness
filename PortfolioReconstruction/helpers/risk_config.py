"""
risk_config.py

Purpose
-------
Define shared risk and portfolio configuration used across analytics modules.

What it does
------------
- Provides a frozen RiskConfig dataclass with knobs for lookback window, return
  type, covariance shrinkage, missing-data policy, annualization, and weight caps.

Notes
-----
- All fields have sensible defaults; override by constructing with keyword args.
- shrinkage_method accepts None, "diag", or "ledoitwolf".
"""

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class RiskConfig:
    """Risk-related configuration knobs shared across analytics workflows."""

    # lookback window
    window_days: int = 90
    min_periods: int = 30

    # returns
    return_type: str = "log"  # "simple" or "log"
    clip_outliers: Optional[float] = 0.01

    # covariance / shrinkage
    shrinkage_method: Optional[str] = "ledoitwolf"  # None | "diag" | "ledoitwolf"
    shrinkage_intensity: float = 0.2

    # missing data policy
    missing_policy: str = "ffill_then_inner"  # "ffill_then_inner" | "inner_only"

    # annualization
    annualization_factor: int = 365
    risk_free_rate_annual: float = 0.05

    # portfolio rules
    min_assets: int = 1
    max_weight_single_asset: float = 0.95  # used as weight_cap

    # portfolio / pricing knobs that were globals before
    long_only: bool = True                
    zero_threshold: float = 0.0            
    forward_fill_prices: bool = True      
    max_price_staleness_days: int = 3  
