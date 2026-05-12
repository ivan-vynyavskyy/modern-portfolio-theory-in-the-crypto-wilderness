#!/usr/bin/env python3
"""
distance_per_block_by_token_group.py

Description:
    Per-block L1 distance statistics stratified by token-count group.

Output:
    1. per_block_distance_by_token_group.csv
       token_group in {lt5, gte5} (binary split for main figure).
    2. per_block_distance_by_token_bucket.csv
       token_group in {1, 2, 3, 4, 5, 6-10, 11-20, 21+} (appendix).
    3. hypothesis_tests_l1_by_token_group.csv
       Paired Wilcoxon signed-rank test on block-level means.
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
import time, os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from locations import Location

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = str(Location.MPT_DATA)
OUT_DIR = str(Location.MPT_DISTANCE_RESULTS)

OUT_BINARY  = os.path.join(OUT_DIR, "per_block_distance_by_token_group.csv")
OUT_GRANULAR = os.path.join(OUT_DIR, "per_block_distance_by_token_bucket.csv")
OUT_TESTS   = os.path.join(OUT_DIR, "hypothesis_tests_l1_by_token_group.csv")

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

# ---------------------------------------------------------------------------
# 1. Binary split: lt5 / gte5
# ---------------------------------------------------------------------------
t0 = time.time()
print("Computing per-block stats (binary split) ...")

parts = []
for col, label in STRATEGIES:
    for grp_expr, grp_name in [("token_count < 5", "lt5"),
                                ("token_count >= 5", "gte5")]:
        parts.append(f"""
            SELECT
                block_number,
                MIN(date)                           AS date,
                '{label}'                           AS optimisation,
                '{grp_name}'                        AS token_group,
                AVG({col} / 2.0 * 100.0)            AS mean_dist,
                MEDIAN({col} / 2.0 * 100.0)         AS median_dist,
                STDDEV({col} / 2.0 * 100.0)         AS std_dist,
                COUNT(*)                             AS count
            FROM all_data
            WHERE {grp_expr}
            GROUP BY block_number
        """)

query_binary = "\nUNION ALL\n".join(parts) + "\nORDER BY block_number, optimisation, token_group"
con.execute(f"COPY ({query_binary}) TO '{OUT_BINARY}' (HEADER, DELIMITER ',')")
print(f"  Binary CSV done in {time.time()-t0:.1f}s → {OUT_BINARY}")

# ---------------------------------------------------------------------------
# 2. Granular buckets: 1, 2, 3, 4, 5, 6-10, 11-20, 21+
# ---------------------------------------------------------------------------
t1 = time.time()
print("Computing per-block stats (granular buckets) ...")

BUCKETS = [
    ("token_count = 1",                  "1"),
    ("token_count = 2",                  "2"),
    ("token_count = 3",                  "3"),
    ("token_count = 4",                  "4"),
    ("token_count = 5",                  "5"),
    ("token_count BETWEEN 6 AND 10",     "6-10"),
    ("token_count BETWEEN 11 AND 20",    "11-20"),
    ("token_count >= 21",                "21+"),
]

parts_g = []
for col, label in STRATEGIES:
    for bucket_expr, bucket_name in BUCKETS:
        parts_g.append(f"""
            SELECT
                block_number,
                MIN(date)                           AS date,
                '{label}'                           AS optimisation,
                '{bucket_name}'                     AS token_group,
                AVG({col} / 2.0 * 100.0)            AS mean_dist,
                MEDIAN({col} / 2.0 * 100.0)         AS median_dist,
                STDDEV({col} / 2.0 * 100.0)         AS std_dist,
                COUNT(*)                             AS count
            FROM all_data
            WHERE {bucket_expr}
            GROUP BY block_number
        """)

query_gran = "\nUNION ALL\n".join(parts_g) + "\nORDER BY block_number, optimisation, token_group"
con.execute(f"COPY ({query_gran}) TO '{OUT_GRANULAR}' (HEADER, DELIMITER ',')")
print(f"  Granular CSV done in {time.time()-t1:.1f}s → {OUT_GRANULAR}")

con.close()

# ---------------------------------------------------------------------------
# 3. Hypothesis testing: paired Wilcoxon signed-rank on block-level means
# ---------------------------------------------------------------------------
print("Running hypothesis tests ...")

df = pd.read_csv(OUT_BINARY)

results = []
for opt_name in ["better_return", "safer_risk", "max_sharpe"]:
    sub = df[df["optimisation"] == opt_name]
    lt5  = sub[sub["token_group"] == "lt5"].sort_values("block_number")
    gte5 = sub[sub["token_group"] == "gte5"].sort_values("block_number")

    # Align on block_number (should already match 1-to-1)
    merged = pd.merge(lt5, gte5, on="block_number", suffixes=("_lt5", "_gte5"))

    x = merged["mean_dist_lt5"].values
    y = merged["mean_dist_gte5"].values
    diff = y - x  # gte5 minus lt5

    n_blocks = len(diff)
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)

    # Paired Wilcoxon signed-rank test (two-sided)
    stat_w, p_wilcoxon = sp_stats.wilcoxon(x, y, alternative="two-sided")

    # Paired t-test for reference
    stat_t, p_ttest = sp_stats.ttest_rel(x, y)

    # Effect size: Cohen's d for paired samples
    cohens_d = mean_diff / sd_diff if sd_diff > 0 else np.nan

    results.append({
        "optimisation": opt_name,
        "n_blocks": n_blocks,
        "mean_lt5": np.mean(x),
        "mean_gte5": np.mean(y),
        "mean_diff_gte5_minus_lt5": mean_diff,
        "sd_diff": sd_diff,
        "cohens_d_paired": cohens_d,
        "wilcoxon_stat": stat_w,
        "wilcoxon_p": p_wilcoxon,
        "paired_t_stat": stat_t,
        "paired_t_p": p_ttest,
    })

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_TESTS, index=False)
print(f"  Hypothesis tests → {OUT_TESTS}")

# Pretty-print summary
print("\n" + "=" * 72)
print("HYPOTHESIS TESTS: L1 distance lt5 vs gte5 (paired by block)")
print("=" * 72)
for _, r in res_df.iterrows():
    print(f"\n  {r['optimisation']:15s}  "
          f"mean_lt5={r['mean_lt5']:.1f}%  mean_gte5={r['mean_gte5']:.1f}%  "
          f"Δ={r['mean_diff_gte5_minus_lt5']:+.1f}%")
    print(f"  {'':15s}  "
          f"Wilcoxon p={r['wilcoxon_p']:.2e}  "
          f"paired-t p={r['paired_t_p']:.2e}  "
          f"Cohen's d={r['cohens_d_paired']:.2f}")
print()
