#!/usr/bin/env python3
"""
distance_per_block_analysis.py

Description:
    L1 distance to the three MPT optima per snapshot and by portfolio
    size N, plus the paired tests of the small-vs-large gap. Distances
    are in percent (l1_gap / 2 * 100). N is the number of assets in
    holdings_short, i.e. the optimised set, as in mpt_distance_analysis.py.
    One pass per block for the per-block tables, then one pooled pass
    per strategy for the by-N curve (exact quantiles hold every value
    in memory, so each pooled query covers a single column).

Output (in --out-dir):
    per_block_distance_stats.csv            token_group all / lt5 / gte5
    per_block_distance_by_token_group.csv   token_group lt5 / gte5
    per_block_distance_by_token_bucket.csv  token_group 2, 3, 4, 5, 6-10, 11-20, 21+
        columns: block_number, date, optimisation, token_group,
                 mean_dist, median_dist, std_dist, count
    hypothesis_tests_threshold_sweep.csv    block-level means below vs at or
        above N, N = 3..7: paired Wilcoxon signed-rank, paired t, Cohen's d
    hypothesis_tests_l1_by_token_group.csv  the N = 5 rows of the sweep
    tokens_vs_dist_agg.csv                  pooled over all blocks per exact N:
        strategy, num_tokens, n, mean_dist, median_dist, p25_dist, p75_dist

Usage:
    python distance_per_block_analysis.py \
        --root     <mpt-optimization-results-eth>/wallet_level \
        --out-dir  <mpt-optimization-results-eth>/distance_per_block_results \
        --temp-dir <duckdb_tmp> --threads 32 --mem 100GB
"""

import argparse, os, sys, time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# -----------------------------------------------------------------------------
# Config (all paths come from the command line, no defaults)
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Per-block L1 distance statistics by portfolio size, with threshold tests."
    )
    p.add_argument("--root", type=str, required=True,
                   help="Hive-partitioned wallet-level parquet directory (block_number=*/).")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Output directory for the CSVs.")
    p.add_argument("--temp-dir", type=str, required=True,
                   help="DuckDB spill directory (keep it off the home fileset).")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--mem", type=str, default="100GB",
                   help="DuckDB memory_limit.")
    return p.parse_args()


args = parse_args()
if not os.path.isdir(args.root):
    sys.exit(f"--root is not a directory: {args.root}")

ROOT = args.root
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(args.temp_dir, exist_ok=True)

STRATEGIES = ["better_return", "safer_risk", "max_sharpe"]
THRESHOLDS = [3, 4, 5, 6, 7]
SPLIT = 5   # the threshold of the main text (lt5 / gte5)

# token_group labels per table; the R plots and generate_all_macros.py read
# these labels. A NULL N stays NULL instead of falling into the last group.
GROUPINGS = {
    "all":    "'all'",
    "binary": "CASE WHEN n_assets < 5 THEN 'lt5' WHEN n_assets >= 5 THEN 'gte5' END",
    "bucket": ("CASE WHEN n_assets <= 5 THEN CAST(n_assets AS VARCHAR) "
               "WHEN n_assets <= 10 THEN '6-10' WHEN n_assets <= 20 THEN '11-20' "
               "WHEN n_assets > 20 THEN '21+' END"),
}

DIST_COLS = ", ".join(f"l1_gap_{s} / 2.0 * 100.0 AS {s}" for s in STRATEGIES)

con = duckdb.connect(database=":memory:")
con.execute(f"SET threads = {args.threads}")
con.execute(f"SET memory_limit = '{args.mem}'")
con.execute(f"SET temp_directory = '{args.temp_dir}'")

blocks = sorted(int(p.name.split("=", 1)[1]) for p in Path(ROOT).glob("block_number=*"))
if not blocks:
    sys.exit(f"no block_number=* partitions under {ROOT}")

print(f"Input:    {ROOT}  ({len(blocks)} blocks)")
print(f"Output:   {OUT_DIR}")
print(f"Temp dir: {args.temp_dir}  (threads={args.threads}, memory_limit={args.mem})")


def save(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  → {path}  ({len(df):,} rows)")


# -----------------------------------------------------------------------------
# 1. Per-block statistics (one block in memory at a time)
# -----------------------------------------------------------------------------
STAT_AGGS = ", ".join(
    f"COUNT({s}) AS count_{s}, AVG({s}) AS mean_{s}, "
    f"MEDIAN({s}) AS median_{s}, STDDEV({s}) AS std_{s}"
    for s in STRATEGIES
)
SUM_AGGS = ", ".join(f"COUNT({s}) AS n_{s}, SUM({s}) AS sum_{s}" for s in STRATEGIES)

group_frames = {name: [] for name in GROUPINGS}
size_frames = []   # per block and exact N: counts and sums, for the threshold sweep

t0 = time.time()
print("\nPer-block statistics ...")
for i, blk in enumerate(blocks, 1):
    path = f"{ROOT}/block_number={blk}/*.parquet"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE blk AS
        SELECT CAST(json_array_length(holdings_short) AS INTEGER) AS n_assets, {DIST_COLS}
        FROM read_parquet('{path}')
    """)
    date = pd.Timestamp(con.execute(f"SELECT MIN(date) FROM read_parquet('{path}')").fetchone()[0]).date()

    for name, expr in GROUPINGS.items():
        df = con.execute(f"SELECT {expr} AS token_group, {STAT_AGGS} FROM blk GROUP BY ALL").fetchdf()
        df.insert(0, "date", date)
        df.insert(0, "block_number", blk)
        group_frames[name].append(df)

    df = con.execute(f"SELECT n_assets, {SUM_AGGS} FROM blk GROUP BY n_assets").fetchdf()
    df.insert(0, "block_number", blk)
    size_frames.append(df)

    n_rows = int(group_frames["all"][-1][f"count_{STRATEGIES[0]}"].iloc[0])
    print(f"  [{i:2d}/{len(blocks)}] block {blk} ({date}): {n_rows:,} wallets  "
          f"({time.time() - t0:.0f}s)")
con.execute("DROP TABLE blk")


def to_long(frames):
    """One row per block × optimisation × token_group."""
    wide = pd.concat(frames, ignore_index=True)
    parts = []
    for s in STRATEGIES:
        part = wide[["block_number", "date", "token_group"]].copy()
        part.insert(2, "optimisation", s)
        part["mean_dist"] = wide[f"mean_{s}"]
        part["median_dist"] = wide[f"median_{s}"]
        part["std_dist"] = wide[f"std_{s}"]
        part["count"] = wide[f"count_{s}"]
        parts.append(part)
    return (pd.concat(parts)
              .sort_values(["block_number", "optimisation", "token_group"])
              .reset_index(drop=True))


per_all = to_long(group_frames["all"])
per_binary = to_long(group_frames["binary"])
per_bucket = to_long(group_frames["bucket"])

save(pd.concat([per_all, per_binary])
       .sort_values(["block_number", "optimisation", "token_group"]),
     "per_block_distance_stats.csv")
save(per_binary, "per_block_distance_by_token_group.csv")
save(per_bucket, "per_block_distance_by_token_bucket.csv")

# -----------------------------------------------------------------------------
# 2. Paired tests on block-level means: N below vs at or above the threshold
# -----------------------------------------------------------------------------
print("\nThreshold sweep ...")
sizes = pd.concat(size_frames, ignore_index=True)

rows = []
for thr in THRESHOLDS:
    below = sizes[sizes["n_assets"] < thr].groupby("block_number").sum()
    above = sizes[sizes["n_assets"] >= thr].groupby("block_number").sum()
    for s in STRATEGIES:
        paired = pd.DataFrame({
            "below": below[f"sum_{s}"] / below[f"n_{s}"],
            "above": above[f"sum_{s}"] / above[f"n_{s}"],
        }).dropna()
        x, y = paired["below"].values, paired["above"].values
        diff = y - x
        sd_diff = np.std(diff, ddof=1)
        stat_w, p_w = sp_stats.wilcoxon(x, y, alternative="two-sided")
        stat_t, p_t = sp_stats.ttest_rel(x, y)
        rows.append({
            "threshold": thr,
            "optimisation": s,
            "n_blocks": len(diff),
            "mean_below": x.mean(),
            "mean_above": y.mean(),
            "mean_diff": diff.mean(),
            "sd_diff": sd_diff,
            "cohens_d": diff.mean() / sd_diff if sd_diff > 0 else np.nan,
            "wilcoxon_stat": stat_w,
            "wilcoxon_p": p_w,
            "paired_t_stat": stat_t,
            "paired_t_p": p_t,
        })

sweep = pd.DataFrame(rows)
save(sweep, "hypothesis_tests_threshold_sweep.csv")
save(sweep[sweep["threshold"] == SPLIT]
         .drop(columns="threshold")
         .rename(columns={"mean_below": "mean_lt5", "mean_above": "mean_gte5",
                          "mean_diff": "mean_diff_gte5_minus_lt5",
                          "cohens_d": "cohens_d_paired"}),
     "hypothesis_tests_l1_by_token_group.csv")

# -----------------------------------------------------------------------------
# 3. Pooled distance per exact N (all blocks)
# -----------------------------------------------------------------------------
print("\nPooled distance per N ...")
con.execute(f"""
    CREATE VIEW all_data AS
    SELECT CAST(json_array_length(holdings_short) AS INTEGER) AS n_assets, {DIST_COLS}
    FROM read_parquet('{ROOT}/*/*.parquet')
""")

parts = []
for s in STRATEGIES:
    t1 = time.time()
    df = con.execute(f"""
        SELECT n_assets AS num_tokens, COUNT({s}) AS n, AVG({s}) AS mean_dist,
               QUANTILE_CONT({s}, [0.25, 0.5, 0.75]) AS q
        FROM all_data
        WHERE n_assets IS NOT NULL
        GROUP BY n_assets
    """).fetchdf()
    df[["p25_dist", "median_dist", "p75_dist"]] = pd.DataFrame(df.pop("q").tolist(), index=df.index)
    df.insert(0, "strategy", s)
    parts.append(df)
    print(f"  {s}: {time.time() - t1:.0f}s")

save(pd.concat(parts)[["strategy", "num_tokens", "n", "mean_dist",
                       "median_dist", "p25_dist", "p75_dist"]]
       .sort_values(["strategy", "num_tokens"]),
     "tokens_vs_dist_agg.csv")
con.close()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Time-averaged per-block mean / median distance (%), all wallets")
for s in STRATEGIES:
    sub = per_all[per_all["optimisation"] == s]
    print(f"  {s:15s} mean {sub['mean_dist'].mean():6.2f}   median {sub['median_dist'].mean():6.2f}")

print(f"\nPaired tests by block, below vs at or above N ({sweep['n_blocks'].max()} blocks)")
print(f"{'N':>3s}  {'strategy':15s}  {'mean<N':>7s}  {'mean≥N':>7s}  {'Δ':>6s}  "
      f"{'Cohen d':>7s}  {'Wilcoxon p':>10s}")
for _, r in sweep.iterrows():
    print(f"{r['threshold']:3d}  {r['optimisation']:15s}  {r['mean_below']:6.1f}%  "
          f"{r['mean_above']:6.1f}%  {r['mean_diff']:+6.1f}  {r['cohens_d']:7.2f}  "
          f"{r['wilcoxon_p']:10.2e}")
print(f"\nDone in {time.time() - t0:.0f}s")
