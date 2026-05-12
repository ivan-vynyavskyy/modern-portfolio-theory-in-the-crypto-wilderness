#!/usr/bin/env python3
"""
distance_per_block_analysis.py

Description:
    Compute per-block aggregate L1 distance statistics (mean, median,
    std) for each optimisation strategy. Produces a single CSV used by
    the time-series plot scripts.

Output:
    CSV with columns: block_number, date, optimisation, mean, median, std.
"""

import duckdb, time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from locations import Location

ROOT = str(Location.MPT_DATA)
OUTPUT = str(Location.MPT_DISTANCE_RESULTS / "per_block_distance_stats.csv")

con = duckdb.connect(database=":memory:")
con.execute("SET threads = 32")
con.execute("SET memory_limit = '100GB'")

con.execute(f"""
    CREATE VIEW all_data AS
    SELECT * FROM read_parquet('{ROOT}/*/*.parquet', hive_partitioning=true)
""")

t0 = time.time()
print("Computing per-block stats ...")

con.execute(f"""
    COPY (
        SELECT
            block_number,
            MIN(date) AS date,
            'better_return' AS optimisation,
            AVG(l1_gap_better_return / 2.0 * 100.0)    AS mean_dist,
            MEDIAN(l1_gap_better_return / 2.0 * 100.0)  AS median_dist,
            STDDEV(l1_gap_better_return / 2.0 * 100.0)  AS std_dist
        FROM all_data
        GROUP BY block_number

        UNION ALL

        SELECT
            block_number,
            MIN(date) AS date,
            'safer_risk' AS optimisation,
            AVG(l1_gap_safer_risk / 2.0 * 100.0)    AS mean_dist,
            MEDIAN(l1_gap_safer_risk / 2.0 * 100.0)  AS median_dist,
            STDDEV(l1_gap_safer_risk / 2.0 * 100.0)  AS std_dist
        FROM all_data
        GROUP BY block_number

        UNION ALL

        SELECT
            block_number,
            MIN(date) AS date,
            'max_sharpe' AS optimisation,
            AVG(l1_gap_max_sharpe / 2.0 * 100.0)    AS mean_dist,
            MEDIAN(l1_gap_max_sharpe / 2.0 * 100.0)  AS median_dist,
            STDDEV(l1_gap_max_sharpe / 2.0 * 100.0)  AS std_dist
        FROM all_data
        GROUP BY block_number

        ORDER BY block_number, optimisation
    ) TO '{OUTPUT}' (HEADER, DELIMITER ',')
""")

print(f"Done in {time.time()-t0:.1f}s → {OUTPUT}")
con.close()