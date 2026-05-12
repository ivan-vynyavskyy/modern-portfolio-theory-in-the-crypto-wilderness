#!/usr/bin/env python3
"""
distance_per_wealth_group.py

Description:
    Compute per-block L1 distance statistics stratified by wealth group
    and token-count group. Each (block, strategy, wealth, token_group)
    combination gets a row with mean, median and std of the L1 distance.

Output:
    CSV with columns: block_number, date, optimisation, wealth_group,
    token_group, mean_dist, median_dist, std_dist.
"""

import duckdb, time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from locations import Location

ROOT = str(Location.MPT_DATA)
OUTPUT = "./per_block_wealth_token_stats.csv"

STRATEGIES = ["better_return", "safer_risk", "max_sharpe"]

WEALTH_GROUPS = [
    ("OVERALL",  "1=1"),
    ("$0-1",     "total_value_usd <= 1"),
    ("$1-100",   "total_value_usd > 1   AND total_value_usd <= 100"),
    ("$100-1K",  "total_value_usd > 100 AND total_value_usd <= 1000"),
    ("$1K-10K",  "total_value_usd > 1000 AND total_value_usd <= 10000"),
    ("$10K+",    "total_value_usd > 10000"),
]

TOKEN_GROUPS = [
    ("all",  "1=1"),
    ("lt5",  "num_tokens < 5"),
    ("gte5", "num_tokens >= 5"),
]

con = duckdb.connect(database=":memory:")
con.execute("SET threads = 32")
con.execute("SET memory_limit = '100GB'")

con.execute(f"""
    CREATE VIEW all_data AS
    SELECT * FROM read_parquet('{ROOT}/*/*.parquet', hive_partitioning=true)
""")

t0 = time.time()
print("Computing per-block stats by wealth × token group ...")

parts = []
for strategy in STRATEGIES:
    col = f"l1_gap_{strategy}"
    pct = f"({col} / 2.0 * 100.0)"

    for wlabel, wwhere in WEALTH_GROUPS:
        for tlabel, twhere in TOKEN_GROUPS:
            combined_where = f"{wwhere} AND {twhere}"
            parts.append(f"""
            SELECT
                block_number,
                MIN(date)          AS date,
                '{strategy}'       AS optimisation,
                '{wlabel}'         AS wealth_group,
                '{tlabel}'         AS token_group,
                AVG({pct})         AS mean_dist,
                MEDIAN({pct})      AS median_dist,
                STDDEV({pct})      AS std_dist
            FROM all_data
            WHERE {combined_where}
            GROUP BY block_number
            """)

full_sql = f"""
    COPY (
        {" UNION ALL ".join(parts)}
        ORDER BY block_number, optimisation, wealth_group, token_group
    ) TO '{OUTPUT}' (HEADER, DELIMITER ',')
"""

con.execute(full_sql)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s -> {OUTPUT}")
print(f"  ({len(parts)} groups = {len(STRATEGIES)} strategies x {len(WEALTH_GROUPS)} wealth x {len(TOKEN_GROUPS)} token)")
con.close()