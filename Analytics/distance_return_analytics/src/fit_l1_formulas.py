#!/usr/bin/env python3
"""
fit_l1_formulas.py

Description:
    Fit closed-form formula for L1 distance as a function of token
    count. Model: d(n) = c - a * n^{-b}, where c is the asymptotic
    ceiling (constrained to c <= 100%). Fits on both mean and median,
    weighted by sqrt(sample_count), via Levenberg-Marquardt with bound
    constraints.

Input:
    tokens_vs_dist_agg.csv (strategy, num_tokens, n, mean_dist, median_dist).

Output:
    l1_formula_fits.csv (fitted parameters per strategy per metric).
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------------
# Config
CSV_IN  = "../data/tokens_vs_dist_agg.csv"
CSV_OUT = "../data/l1_formula_fits.csv"

MIN_OBS    = 30
MAX_TOKENS = 50

STRATEGIES = ["better_return", "safer_risk", "max_sharpe"]
LABELS = {
    "better_return": "MRV (Max Return)",
    "safer_risk":    "MVR (Min Variance)",
    "max_sharpe":    "MSR (Max Sharpe)",
}
KEY_TOKENS = [2, 3, 5, 10, 20, 50]

# -----------------------------------------------------------------------------
# Model
def power_decay(n, a, b, c):
    """d(n) = c - a * n^{-b}"""
    return c - a * np.power(n, -b)

# Bounds: a >= 0, b > 0, 0 <= c <= 100
BOUNDS_LO = [0,      0.001, 0]
BOUNDS_HI = [np.inf, np.inf, 100]

# -----------------------------------------------------------------------------
# Load & filter
df = pd.read_csv(CSV_IN)
df = df[(df["n"] >= MIN_OBS) & (df["num_tokens"] >= 2) & (df["num_tokens"] <= MAX_TOKENS)]
print(f"Loaded {len(df)} rows (tokens 2-{MAX_TOKENS}, n >= {MIN_OBS})")

# -----------------------------------------------------------------------------
# Fit
results = []

for stat_label, col in [("mean", "mean_dist"), ("median", "median_dist")]:
    print(f"\n{'='*70}")
    print(f"FITTING ON {stat_label.upper()}:  d(n) = c - a * n^{{-b}}  (c <= 100)")
    print(f"{'='*70}")

    for strategy in STRATEGIES:
        sub = df[df["strategy"] == strategy].copy()
        x = sub["num_tokens"].values.astype(float)
        y = sub[col].values
        w = np.sqrt(sub["n"].values.astype(float))

        try:
            popt, pcov = curve_fit(
                power_decay, x, y,
                p0=[100, 0.5, 90],
                bounds=(BOUNDS_LO, BOUNDS_HI),
                sigma=1.0 / w,
                maxfev=10000,
            )

            y_pred = power_decay(x, *popt)

            y_bar = np.average(y, weights=w**2)
            ss_res = np.sum(w**2 * (y - y_pred) ** 2)
            ss_tot = np.sum(w**2 * (y - y_bar) ** 2)
            r2 = 1 - ss_res / ss_tot
            mae = np.mean(np.abs(y - y_pred))

            a, b, c = popt
            bound_note = " [bound]" if c >= 99.99 else ""
            print(f"\n  {LABELS[strategy]}:")
            print(f"    d(n) = {c:.2f} - {a:.2f} * n^{{-{b:.4f}}}")
            print(f"    c = {c:.2f}%{bound_note},  R2 = {r2:.4f},  MAE = {mae:.2f}%")

            print(f"    {'n':>5s}  {'Pred':>8s}  {'Actual':>8s}  {'Error':>7s}")
            for n in KEY_TOKENS:
                pred = power_decay(n, *popt)
                row = sub[sub["num_tokens"] == n]
                actual = row[col].values[0] if len(row) > 0 else float("nan")
                err = pred - actual
                print(f"    {n:5d}  {pred:7.2f}%  {actual:7.2f}%  {err:+6.2f}%")

            results.append({
                "strategy": strategy,
                "statistic": stat_label,
                "a": round(float(a), 4),
                "b": round(float(b), 4),
                "c": round(float(c), 4),
                "r2": round(r2, 4),
                "mae": round(mae, 2),
            })

        except Exception as e:
            print(f"\n  {LABELS[strategy]}: FAILED - {e}")

# -----------------------------------------------------------------------------
# Summary
res_df = pd.DataFrame(results)
print(f"\n\n{'='*70}")
print("COMPARISON TABLE")
print(f"{'='*70}")
print(res_df.to_string(index=False))

# -----------------------------------------------------------------------------
# Save
res_df.to_csv(CSV_OUT, index=False)
print(f"\nSaved -> {CSV_OUT}")