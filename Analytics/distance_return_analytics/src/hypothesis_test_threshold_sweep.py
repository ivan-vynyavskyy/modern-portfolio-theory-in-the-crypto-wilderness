#!/usr/bin/env python3
"""
hypothesis_test_threshold_sweep.py

Description:
    Sweep token-count thresholds (N = 3..7) and test which threshold
    maximises the separation between small (<N) and large (>=N)
    portfolios in terms of L1 distance. For each threshold x strategy:
    paired Wilcoxon signed-rank test on block-level means, Cohen's d,
    and mean difference.

Output:
    hypothesis_tests_threshold_sweep.csv
"""

import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from locations import Location

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = str(Location.MPT_DATA_EXTENDED)
OUT_DIR = str(Location.MPT_DISTANCE_RESULTS)
OUT_CSV = os.path.join(OUT_DIR, "hypothesis_tests_threshold_sweep.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DuckDB setup
# ---------------------------------------------------------------------------
con = duckdb.connect(database=":memory:")
con.execute("SET threads = 32")
con.execute("SET memory_limit = '100GB'")

con.execute(f"""
    CREATE VIEW all_data AS
    SELECT * FROM read_parquet('{ROOT}/*/*.parquet', hive_partitioning=true)
""")

STRATEGIES = [
    ("l1_gap_better_return", "better_return"),
    ("l1_gap_safer_risk",    "safer_risk"),
    ("l1_gap_max_sharpe",    "max_sharpe"),
]

THRESHOLDS = [3, 4, 5, 6, 7]

# ---------------------------------------------------------------------------
# Compute per-block means for each threshold × strategy × group
# ---------------------------------------------------------------------------
t0 = time.time()
all_results = []

for N in THRESHOLDS:
    print(f"Threshold N = {N} ...")

    parts = []
    for col, label in STRATEGIES:
        for grp_expr, grp_name in [(f"num_tokens < {N}", "below"),
                                    (f"num_tokens >= {N}", "above")]:
            parts.append(f"""
                SELECT
                    block_number,
                    '{label}'    AS optimisation,
                    '{grp_name}' AS grp,
                    AVG({col} / 2.0 * 100.0) AS mean_dist
                FROM all_data
                WHERE {grp_expr}
                GROUP BY block_number
            """)

    query = "\nUNION ALL\n".join(parts)
    df = con.execute(query).fetchdf()

    # Run tests per strategy
    for opt_name in ["better_return", "safer_risk", "max_sharpe"]:
        sub = df[df["optimisation"] == opt_name]
        below = sub[sub["grp"] == "below"].sort_values("block_number")
        above = sub[sub["grp"] == "above"].sort_values("block_number")

        merged = pd.merge(below, above, on="block_number", suffixes=("_below", "_above"))

        x = merged["mean_dist_below"].values
        y = merged["mean_dist_above"].values
        diff = y - x

        n_blocks = len(diff)
        mean_diff = np.mean(diff)
        sd_diff = np.std(diff, ddof=1)
        cohens_d = mean_diff / sd_diff if sd_diff > 0 else np.nan

        stat_w, p_wilcoxon = sp_stats.wilcoxon(x, y, alternative="two-sided")

        all_results.append({
            "threshold": N,
            "optimisation": opt_name,
            "n_blocks": n_blocks,
            "mean_below": np.mean(x),
            "mean_above": np.mean(y),
            "mean_diff": mean_diff,
            "sd_diff": sd_diff,
            "cohens_d": cohens_d,
            "wilcoxon_stat": stat_w,
            "wilcoxon_p": p_wilcoxon,
        })

con.close()
print(f"Queries done in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Save and display
# ---------------------------------------------------------------------------
res_df = pd.DataFrame(all_results)
res_df.to_csv(OUT_CSV, index=False)
print(f"Saved → {OUT_CSV}\n")

# Pretty-print summary table
print("=" * 90)
print(f"{'N':>3s}  {'Strategy':>15s}  {'mean<N':>7s}  {'mean≥N':>7s}  "
      f"{'Δ':>7s}  {'Cohen d':>8s}  {'Wilcoxon p':>12s}")
print("-" * 90)
for _, r in res_df.iterrows():
    print(f"{r['threshold']:3.0f}  {r['optimisation']:>15s}  "
          f"{r['mean_below']:6.1f}%  {r['mean_above']:6.1f}%  "
          f"{r['mean_diff']:+6.1f}%  {r['cohens_d']:8.2f}  "
          f"{r['wilcoxon_p']:12.2e}")
print()

# Find best threshold per strategy by Cohen's d
print("Best threshold (max Cohen's d) per strategy:")
for opt in ["better_return", "safer_risk", "max_sharpe"]:
    sub = res_df[res_df["optimisation"] == opt]
    best = sub.loc[sub["cohens_d"].idxmax()]
    print(f"  {opt:15s} → N = {best['threshold']:.0f}  "
          f"(d = {best['cohens_d']:.2f}, Δ = {best['mean_diff']:+.1f}%)")
