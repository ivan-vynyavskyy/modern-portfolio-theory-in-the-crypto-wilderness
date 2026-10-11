#!/usr/bin/env python3
"""
mpt_distance_analysis.py

Description:
    Full-scale L1 distance analysis pipeline. Covers correlations,
    distribution analysis, bucket breakdowns, and regression models
    (OLS, quantile, RF). Engine: DuckDB + scikit-learn.
    Exact quantiles hold every value in memory, outside DuckDB's
    memory_limit, so each quantile query covers a single column.

Input:
    Hive-partitioned parquet directory (--root) with block_number,
    wallet_address, L1 distances, holdings_short, total_value_usd.
    num_tokens (portfolio size N) is the number of assets in
    holdings_short, i.e. the optimised set, not 5_0's raw on-chain count.

Output:
    CSV exports + analysis_log.txt in --out-dir.

Usage:
    python mpt_distance_analysis.py \
        --root     <mpt-optimization-results-eth>/wallet_level \
        --out-dir  <mpt-optimization-results-eth>/distance_analysis_results \
        --temp-dir <duckdb_tmp> --threads 32 --mem 100GB
    5+ robustness check: add --min-assets 5 and write to
    <mpt-optimization-results-eth>/distance_analysis_results_gr_5.
"""

import argparse, os, sys, time, warnings
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Config (all paths come from the command line, no defaults)
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Full-scale L1 distance analysis of the MPT optimisation results."
    )
    p.add_argument("--root", type=str, required=True,
                   help="Hive-partitioned wallet-level parquet directory (block_number=*/).")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Output directory for CSVs and analysis_log.txt.")
    p.add_argument("--temp-dir", type=str, required=True,
                   help="DuckDB spill directory (keep it off the home fileset).")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--mem", type=str, default="100GB",
                   help="DuckDB memory_limit.")
    p.add_argument("--min-assets", type=int, default=2,
                   help="Keep wallets with at least this many assets (5 = robustness check).")
    return p.parse_args()


args = parse_args()
if not os.path.isdir(args.root):
    sys.exit(f"--root is not a directory: {args.root}")

ROOT = args.root
OUTPUT_DIR = args.out_dir
TEMP_DIR = args.temp_dir
SAMPLE_SIZE = 50_000_000          # 50M for correlations / regressions
REGRESSION_SAMPLE = 10_000_000    # 10M for heavier models (RF, quantile)
N_CORES = args.threads
MEM_LIMIT = args.mem
MIN_ASSETS = args.min_assets
SEED = 42

TARGETS_RAW = ["l1_gap_better_return", "l1_gap_safer_risk", "l1_gap_max_sharpe"]
TARGETS_PCT = ["pct_better_return", "pct_safer_risk", "pct_max_sharpe"]
TARGET_LABELS = ["better_return", "safer_risk", "max_sharpe"]
FEATURES = ["month", "log_value_usd", "num_tokens"]

DISTANCE_BUCKETS = [
    ("0 (exact)", 0, 0),
    ("(0, 1]",  0, 1),        # near-zero but not exactly 0
    ("(1, 20]",  1, 20),
    ("(20, 40]", 20, 40),
    ("(40, 60]", 40, 60),
    ("(60, 80]", 60, 80),
    ("(80, 100]", 80, 100),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Logging helper – tee to console + file
# -----------------------------------------------------------------------------
LOG_PATH = os.path.join(OUTPUT_DIR, "analysis_log.txt")
log_file = open(LOG_PATH, "w", buffering=1)

def log(msg=""):
    """Print to console and append to log file."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if msg else ""
    print(line, flush=True)
    log_file.write(line + "\n")

def log_df(df, title=""):
    """Pretty-print a DataFrame to console + log."""
    if title:
        log(title)
    buf = df.to_string()
    print(buf, flush=True)
    log_file.write(buf + "\n")
    log("")

def save_csv(df, name):
    """Save DataFrame to CSV in output dir."""
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=True)
    log(f"  → saved {path}")

# -----------------------------------------------------------------------------
# 1. Setup DuckDB
# -----------------------------------------------------------------------------
log("=" * 70)
if MIN_ASSETS > 2:
    log(f"MPT DISTANCE ANALYSIS — FILTERED: num_tokens >= {MIN_ASSETS}")
else:
    log("MPT DISTANCE ANALYSIS — FULL SCALE")
log("=" * 70)
log(f"Input:    {ROOT}")
log(f"Output:   {OUTPUT_DIR}")
log(f"Temp dir: {TEMP_DIR}  (threads={N_CORES}, memory_limit={MEM_LIMIT})")
log()

t0 = time.time()
con = duckdb.connect(database=":memory:")
con.execute(f"SET threads = {N_CORES}")
con.execute(f"SET memory_limit = '{MEM_LIMIT}'")
con.execute(f"SET temp_directory = '{TEMP_DIR}'")

# num_tokens = assets in the optimised set (holdings_short). The stored
# column is the raw on-chain count: it counts ETH and WETH separately and
# includes assets dropped for too short a price history. Every wallet has at
# least two, so the filter is only added for --min-assets above 2.
min_assets_filter = (f"WHERE json_array_length(holdings_short) >= {MIN_ASSETS}"
                     if MIN_ASSETS > 2 else "")
con.execute(f"""
    CREATE VIEW all_data AS
    SELECT * REPLACE (CAST(json_array_length(holdings_short) AS INTEGER) AS num_tokens)
    FROM read_parquet('{ROOT}/*/*.parquet', hive_partitioning=true)
    {min_assets_filter}
""")

total_rows = con.sql("SELECT COUNT(*) AS n FROM all_data").fetchone()[0]
log(f"Total rows: {total_rows:,}")
log()

# -----------------------------------------------------------------------------
# 2. FULL-DATA PEARSON CORRELATION  (via DuckDB)
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 2: Full-data Pearson Correlation (all rows)")
log("=" * 70)

# We compute Pearson in DuckDB using CORR() — runs on all 239M rows without
# pulling data into Python.  We need feature engineering inside SQL.
pearson_sql = f"""
SELECT
    CORR(month_feat, l1_gap_better_return / 2 * 100) AS pearson_month_better_return,
    CORR(month_feat, l1_gap_safer_risk    / 2 * 100) AS pearson_month_safer_risk,
    CORR(month_feat, l1_gap_max_sharpe    / 2 * 100) AS pearson_month_max_sharpe,

    CORR(log_val,    l1_gap_better_return / 2 * 100) AS pearson_logval_better_return,
    CORR(log_val,    l1_gap_safer_risk    / 2 * 100) AS pearson_logval_safer_risk,
    CORR(log_val,    l1_gap_max_sharpe    / 2 * 100) AS pearson_logval_max_sharpe,

    CORR(num_tokens, l1_gap_better_return / 2 * 100) AS pearson_numtok_better_return,
    CORR(num_tokens, l1_gap_safer_risk    / 2 * 100) AS pearson_numtok_safer_risk,
    CORR(num_tokens, l1_gap_max_sharpe    / 2 * 100) AS pearson_numtok_max_sharpe
FROM (
    SELECT
        *,
        (EXTRACT(YEAR FROM date) - 2020) * 12
            + EXTRACT(MONTH FROM date)            AS month_feat,
        LN(1 + total_value_usd)                             AS log_val
    FROM all_data
) sub
"""

log("  Computing Pearson on full data via DuckDB ...")
t1 = time.time()
pearson_row = con.sql(pearson_sql).fetchdf().iloc[0]
log(f"  Done in {time.time()-t1:.1f}s")

pearson_data = {}
for feat, feat_key in [("month", "month"), ("log_value_usd", "logval"), ("num_tokens", "numtok")]:
    for tgt in TARGET_LABELS:
        col = f"pearson_{feat_key}_{tgt}"
        pearson_data.setdefault(feat, {})[tgt] = round(pearson_row[col], 4)

pearson_df = pd.DataFrame(pearson_data).T
pearson_df.index.name = "feature"
log_df(pearson_df, "Pearson r (full data, 0-100% scale):")
save_csv(pearson_df, "pearson_full_data.csv")

# -----------------------------------------------------------------------------
# 3. LARGE-SAMPLE SPEARMAN + PEARSON (50M rows)
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 3: Large-sample correlation (50M rows) — Pearson + Spearman")
log("=" * 70)

log(f"  Drawing {SAMPLE_SIZE/1e6:.0f}M sample ...")
t1 = time.time()

sample = con.sql(f"""
    SELECT
        date,
        total_value_usd,
        num_tokens,
        l1_gap_better_return,
        l1_gap_safer_risk,
        l1_gap_max_sharpe
    FROM all_data
    USING SAMPLE {SAMPLE_SIZE}
""").fetchdf()

sample["date_dt"] = pd.to_datetime(sample["date"], unit="ms") if sample["date"].dtype in ["int64", "float64"] else pd.to_datetime(sample["date"])
sample["month"] = (sample["date_dt"].dt.year - 2020) * 12 + sample["date_dt"].dt.month
sample["log_value_usd"] = np.log1p(sample["total_value_usd"])

for raw, pct in zip(TARGETS_RAW, TARGETS_PCT):
    sample[pct] = sample[raw] / 2.0 * 100.0

log(f"  Sample loaded: {len(sample):,} rows in {time.time()-t1:.1f}s")

corr_rows = []
for feat in FEATURES:
    for tgt, label in zip(TARGETS_PCT, TARGET_LABELS):
        mask = sample[[feat, tgt]].dropna().index
        x, y = sample.loc[mask, feat].values, sample.loc[mask, tgt].values
        pr, _ = sp_stats.pearsonr(x, y)
        sr, _ = sp_stats.spearmanr(x, y)
        corr_rows.append({
            "feature": feat, "target": label,
            "pearson_r": round(pr, 4), "spearman_r": round(sr, 4),
        })

corr_df = pd.DataFrame(corr_rows)

piv_p = corr_df.pivot(index="feature", columns="target", values="pearson_r")
piv_s = corr_df.pivot(index="feature", columns="target", values="spearman_r")
log_df(piv_p, f"Pearson r (sample {SAMPLE_SIZE/1e6:.0f}M):")
log_df(piv_s, f"Spearman ρ (sample {SAMPLE_SIZE/1e6:.0f}M):")
save_csv(corr_df, "correlation_sample.csv")

# -----------------------------------------------------------------------------
# 4. DISTANCE DISTRIBUTION PER OPTIMISATION
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 4: Distance Distribution (full data)")
log("=" * 70)

# 4a. Zero-distance counts (exact 0, near-zero <0.001 in raw = <0.05%)
log("  4a. Zero-distance & near-zero counts ...")
zero_sql = """
SELECT
    COUNT(l1_gap_better_return)                                        AS n_better_return,
    COUNT(l1_gap_safer_risk)                                           AS n_safer_risk,
    COUNT(l1_gap_max_sharpe)                                           AS n_max_sharpe,

    SUM(CASE WHEN l1_gap_better_return  = 0 THEN 1 ELSE 0 END)        AS exact0_better_return,
    SUM(CASE WHEN l1_gap_safer_risk     = 0 THEN 1 ELSE 0 END)        AS exact0_safer_risk,
    SUM(CASE WHEN l1_gap_max_sharpe     = 0 THEN 1 ELSE 0 END)        AS exact0_max_sharpe,

    SUM(CASE WHEN l1_gap_better_return  > 0 AND l1_gap_better_return  <= 0.02 THEN 1 ELSE 0 END) AS near0_better_return,
    SUM(CASE WHEN l1_gap_safer_risk     > 0 AND l1_gap_safer_risk     <= 0.02 THEN 1 ELSE 0 END) AS near0_safer_risk,
    SUM(CASE WHEN l1_gap_max_sharpe     > 0 AND l1_gap_max_sharpe     <= 0.02 THEN 1 ELSE 0 END) AS near0_max_sharpe
FROM all_data
"""
zdf = con.sql(zero_sql).fetchdf().iloc[0]

# Shares of the rows that have a distance for that optimisation.
zero_results = []
for label in TARGET_LABELS:
    n = int(zdf[f"n_{label}"])
    e0 = int(zdf[f"exact0_{label}"])
    n0 = int(zdf[f"near0_{label}"])
    zero_results.append({
        "optimisation": label,
        "exact_0_count": e0, "exact_0_pct": round(e0/n*100, 4),
        "near_0_count_(0,1%]": n0, "near_0_pct_(0,1%]": round(n0/n*100, 4),
        "combined_<=1%_count": e0+n0, "combined_<=1%_pct": round((e0+n0)/n*100, 4),
    })

zero_df = pd.DataFrame(zero_results).set_index("optimisation")
log_df(zero_df, "Zero-distance counts (exact 0  +  near-zero ≤1%):")
save_csv(zero_df, "zero_distance_counts.csv")

# 4b. General distribution — histogram in SQL via CASE buckets
log("  4b. General distance distribution per bucket ...")

bucket_cases = []
for label in TARGET_LABELS:
    raw = f"l1_gap_{label}"
    pct_expr = f"({raw} / 2.0 * 100.0)"
    bucket_cases.append(f"""
        COUNT({raw})                                                                             AS "{label}_n",
        SUM(CASE WHEN {raw} = 0                                               THEN 1 ELSE 0 END) AS "{label}_0",
        SUM(CASE WHEN {pct_expr} >  0 AND {pct_expr} <=  1                   THEN 1 ELSE 0 END) AS "{label}_0_1",
        SUM(CASE WHEN {pct_expr} >  1 AND {pct_expr} <= 20                   THEN 1 ELSE 0 END) AS "{label}_1_20",
        SUM(CASE WHEN {pct_expr} > 20 AND {pct_expr} <= 40                   THEN 1 ELSE 0 END) AS "{label}_20_40",
        SUM(CASE WHEN {pct_expr} > 40 AND {pct_expr} <= 60                   THEN 1 ELSE 0 END) AS "{label}_40_60",
        SUM(CASE WHEN {pct_expr} > 60 AND {pct_expr} <= 80                   THEN 1 ELSE 0 END) AS "{label}_60_80",
        SUM(CASE WHEN {pct_expr} > 80 AND {pct_expr} <= 100                  THEN 1 ELSE 0 END) AS "{label}_80_100"
    """)

dist_sql = f"SELECT {','.join(bucket_cases)} FROM all_data"
dist_row = con.sql(dist_sql).fetchdf().iloc[0]

bucket_labels = ["0", "(0,1]", "(1,20]", "(20,40]", "(40,60]", "(60,80]", "(80,100]"]
dist_records = []
for label in TARGET_LABELS:
    n = int(dist_row[f"{label}_n"])
    keys = [f"{label}_0", f"{label}_0_1", f"{label}_1_20",
            f"{label}_20_40", f"{label}_40_60", f"{label}_60_80", f"{label}_80_100"]
    for bkt, k in zip(bucket_labels, keys):
        cnt = int(dist_row[k])
        dist_records.append({
            "optimisation": label, "bucket": bkt,
            "count": cnt, "pct": round(cnt/n*100, 2),
        })

dist_df = pd.DataFrame(dist_records)
dist_pivot = dist_df.pivot(index="bucket", columns="optimisation", values="pct")
dist_pivot = dist_pivot.reindex(bucket_labels)  # preserve order
dist_pivot_count = dist_df.pivot(index="bucket", columns="optimisation", values="count")
dist_pivot_count = dist_pivot_count.reindex(bucket_labels)

log_df(dist_pivot, "Distance distribution (% of wallets per bucket):")
log_df(dist_pivot_count, "Distance distribution (count per bucket):")
save_csv(dist_pivot, "distance_distribution_pct.csv")
save_csv(dist_pivot_count, "distance_distribution_count.csv")

# -----------------------------------------------------------------------------
# 5. LOW-DISTANCE WALLET PROFILES (≤1%)
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 5: Profile of wallets with distance ≤1% (per optimisation)")
log("=" * 70)

for label in TARGET_LABELS:
    raw_col = f"l1_gap_{label}"
    pct_expr = f"({raw_col} / 2.0 * 100.0)"
    where_low = f"{pct_expr} <= 1"
    log(f"\n--- {label.upper()} (distance ≤1%) ---")

    # 5a. Basic stats; the medians one per query (exact quantiles hold every
    # value in memory, outside DuckDB's memory_limit).
    stats_sql = f"""
    SELECT
        COUNT(*)                        AS n,
        AVG(num_tokens)                 AS mean_tokens,
        MIN(num_tokens)                 AS min_tokens,
        MAX(num_tokens)                 AS max_tokens,
        STDDEV(num_tokens)              AS std_tokens,
        AVG(total_value_usd)            AS mean_value,
        MIN(total_value_usd)            AS min_value,
        MAX(total_value_usd)            AS max_value,
        STDDEV(total_value_usd)         AS std_value,
        AVG(LN(1+total_value_usd))      AS mean_log_value
    FROM all_data
    WHERE {where_low}
    """
    sdf = con.sql(stats_sql).fetchdf()
    for name, expr in [("median_tokens", "num_tokens"),
                       ("median_value", "total_value_usd"),
                       ("median_log_value", "LN(1+total_value_usd)")]:
        sdf[name] = con.sql(f"SELECT MEDIAN({expr}) FROM all_data WHERE {where_low}").fetchone()[0]
    sdf = sdf[["n", "mean_tokens", "median_tokens", "min_tokens", "max_tokens", "std_tokens",
               "mean_value", "median_value", "min_value", "max_value", "std_value",
               "mean_log_value", "median_log_value"]]
    log_df(sdf.T.rename(columns={0: "value"}), f"  Stats for {label} (dist ≤1%):")
    save_csv(sdf.T.rename(columns={0: "value"}), f"low_dist_stats_{label}.csv")

    # 5b. num_tokens distribution within low-distance wallets
    tok_sql = f"""
    SELECT
        num_tokens,
        COUNT(*) AS cnt
    FROM all_data
    WHERE {where_low}
    GROUP BY num_tokens
    ORDER BY num_tokens
    """
    tok_df = con.sql(tok_sql).fetchdf()
    total_low = tok_df["cnt"].sum()
    tok_df["pct"] = (tok_df["cnt"] / total_low * 100).round(2)
    tok_df["cum_pct"] = tok_df["pct"].cumsum().round(2)
    log_df(tok_df.head(20), f"  num_tokens distribution (top 20) for {label} dist ≤1%:")
    save_csv(tok_df, f"low_dist_tokens_dist_{label}.csv")

    # 5c. Value (USD) quantile breakdown within low-distance wallets
    val_sql = f"""
    SELECT
        CASE
            WHEN total_value_usd <= 1       THEN '$0-1'
            WHEN total_value_usd <= 10      THEN '$1-10'
            WHEN total_value_usd <= 100     THEN '$10-100'
            WHEN total_value_usd <= 1000    THEN '$100-1K'
            WHEN total_value_usd <= 10000   THEN '$1K-10K'
            WHEN total_value_usd <= 100000  THEN '$10K-100K'
            WHEN total_value_usd <= 1000000 THEN '$100K-1M'
            ELSE '$1M+'
        END AS value_bucket,
        COUNT(*) AS cnt
    FROM all_data
    WHERE {where_low}
    GROUP BY value_bucket
    ORDER BY MIN(total_value_usd)
    """
    val_df = con.sql(val_sql).fetchdf()
    val_df["pct"] = (val_df["cnt"] / total_low * 100).round(2)
    log_df(val_df, f"  Value distribution for {label} dist ≤1%:")
    save_csv(val_df.set_index("value_bucket"), f"low_dist_value_dist_{label}.csv")

# -----------------------------------------------------------------------------
# 6. BUCKET ANALYSIS — characteristics per distance bucket
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 6: Wallet characteristics per distance bucket")
log("=" * 70)

for label in TARGET_LABELS:
    raw_col = f"l1_gap_{label}"
    pct_expr = f"({raw_col} / 2.0 * 100.0)"

    log(f"\n--- {label.upper()} ---")

    # Distances outside [0, 100] % keep their own 'NaN' row, as before.
    bucket_sql = f"""
        COALESCE(CASE
            WHEN {raw_col} = 0                                    THEN '0_exact'
            WHEN {pct_expr} >  0 AND {pct_expr} <=  1            THEN '(0,1]'
            WHEN {pct_expr} >  1 AND {pct_expr} <= 20            THEN '(1,20]'
            WHEN {pct_expr} > 20 AND {pct_expr} <= 40            THEN '(20,40]'
            WHEN {pct_expr} > 40 AND {pct_expr} <= 60            THEN '(40,60]'
            WHEN {pct_expr} > 60 AND {pct_expr} <= 80            THEN '(60,80]'
            WHEN {pct_expr} > 80 AND {pct_expr} <= 100           THEN '(80,100]'
        END, 'NaN') AS dist_bucket
    """
    bkt_df = con.sql(f"""
        SELECT {bucket_sql},
               COUNT(*)                       AS n,
               AVG(num_tokens)                AS mean_tokens,
               AVG(total_value_usd)           AS mean_value_usd,
               AVG(LN(1+total_value_usd))     AS mean_log_value
        FROM all_data
        WHERE {raw_col} IS NOT NULL
        GROUP BY dist_bucket
    """).fetchdf().set_index("dist_bucket")
    # Quantiles one expression per query (see 5a), all of it in one list call.
    for expr, qcols in [("num_tokens",            {0.25: "p25_tokens", 0.5: "median_tokens", 0.75: "p75_tokens"}),
                        ("total_value_usd",       {0.25: "p25_value", 0.5: "median_value_usd", 0.75: "p75_value"}),
                        ("LN(1+total_value_usd)", {0.5: "median_log_value"})]:
        q = con.sql(f"""
            SELECT {bucket_sql}, QUANTILE_CONT({expr}, {list(qcols)}) AS q
            FROM all_data
            WHERE {raw_col} IS NOT NULL
            GROUP BY dist_bucket
        """).fetchdf().set_index("dist_bucket")["q"]
        for i, col in enumerate(qcols.values()):
            bkt_df[col] = q.map(lambda v, i=i: v[i])
    bkt_df = bkt_df[["n", "mean_tokens", "median_tokens", "mean_value_usd", "median_value_usd",
                     "mean_log_value", "median_log_value",
                     "p25_tokens", "p75_tokens", "p25_value", "p75_value"]].reset_index()

    # reorder
    order = ["0_exact", "(0,1]", "(1,20]", "(20,40]", "(40,60]", "(60,80]", "(80,100]"]
    bkt_df["dist_bucket"] = pd.Categorical(bkt_df["dist_bucket"], categories=order, ordered=True)
    bkt_df = bkt_df.sort_values("dist_bucket").set_index("dist_bucket")

    log_df(bkt_df.round(2), f"  Bucket characteristics for {label}:")
    save_csv(bkt_df.round(4), f"bucket_stats_{label}.csv")

    # Also: token-count distribution within each bucket
    tok_bucket_sql = f"""
    SELECT
        CASE
            WHEN {raw_col} = 0                                    THEN '0_exact'
            WHEN {pct_expr} >  0 AND {pct_expr} <=  1            THEN '(0,1]'
            WHEN {pct_expr} >  1 AND {pct_expr} <= 20            THEN '(1,20]'
            WHEN {pct_expr} > 20 AND {pct_expr} <= 40            THEN '(20,40]'
            WHEN {pct_expr} > 40 AND {pct_expr} <= 60            THEN '(40,60]'
            WHEN {pct_expr} > 60 AND {pct_expr} <= 80            THEN '(60,80]'
            WHEN {pct_expr} > 80 AND {pct_expr} <= 100           THEN '(80,100]'
        END AS dist_bucket,
        num_tokens,
        COUNT(*) AS cnt
    FROM all_data
    WHERE {raw_col} IS NOT NULL
    GROUP BY dist_bucket, num_tokens
    ORDER BY dist_bucket, num_tokens
    """
    tok_bkt_df = con.sql(tok_bucket_sql).fetchdf()
    save_csv(tok_bkt_df, f"bucket_token_dist_{label}.csv")


# -----------------------------------------------------------------------------
# 7. REGRESSION MODELS
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 7: Regression models")
log("=" * 70)

# Draw regression sample
log(f"  Drawing {REGRESSION_SAMPLE/1e6:.0f}M sample for regression ...")
t1 = time.time()
reg_sample = con.sql(f"""
    SELECT
        total_value_usd,
        num_tokens,
        date,
        l1_gap_better_return,
        l1_gap_safer_risk,
        l1_gap_max_sharpe
    FROM all_data
    USING SAMPLE {REGRESSION_SAMPLE}
""").fetchdf()

_dates = pd.to_datetime(reg_sample["date"], unit="ms") if reg_sample["date"].dtype in ["int64", "float64"] else pd.to_datetime(reg_sample["date"])
reg_sample["month"] = (
    (_dates.dt.year - 2020) * 12
    + _dates.dt.month
)
reg_sample["log_value_usd"] = np.log1p(reg_sample["total_value_usd"])
for raw, pct in zip(TARGETS_RAW, TARGETS_PCT):
    reg_sample[pct] = reg_sample[raw] / 2.0 * 100.0

log(f"  Regression sample: {len(reg_sample):,} rows in {time.time()-t1:.1f}s")

# ---------- 7a. OLS ----------
log("\n  --- 7a. OLS Regression ---")
import statsmodels.api as sm

X = reg_sample[FEATURES].copy()
X = sm.add_constant(X)

ols_results = {}
for tgt, label in zip(TARGETS_PCT, TARGET_LABELS):
    y = reg_sample[tgt].values
    mask = ~(np.isnan(y) | np.isinf(y))
    model = sm.OLS(y[mask], X.values[mask]).fit()

    log(f"\n  OLS: {label}")
    log(f"  R² = {model.rsquared:.4f},  Adj-R² = {model.rsquared_adj:.4f},  N = {mask.sum():,}")

    coef_df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t_stat": model.tvalues,
        "p_value": model.pvalues,
    }, index=["const"] + FEATURES)
    coef_df["significant"] = coef_df["p_value"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    )

    log_df(coef_df.round(6), f"  Coefficients ({label}):")
    save_csv(coef_df, f"ols_coefs_{label}.csv")
    ols_results[label] = {
        "R2": round(model.rsquared, 4),
        "Adj_R2": round(model.rsquared_adj, 4),
        "F_stat": round(model.fvalue, 2),
        "F_pvalue": model.f_pvalue,
    }

ols_summary = pd.DataFrame(ols_results).T
ols_summary.index.name = "target"
log_df(ols_summary, "\n  OLS Summary:")
save_csv(ols_summary, "ols_summary.csv")

# ---------- 7b. Quantile Regression (median, 25th, 75th) ----------
log("\n  --- 7b. Quantile Regression ---")
import statsmodels.formula.api as smf

# Use a smaller subset for quantile (it's slower)
QR_SIZE = min(2_000_000, len(reg_sample))
qr_sub = reg_sample.sample(n=QR_SIZE, random_state=SEED)
log(f"  Quantile regression on {QR_SIZE/1e6:.0f}M rows")

quantile_results = []
for tgt, label in zip(TARGETS_PCT, TARGET_LABELS):
    qr_sub_clean = qr_sub[FEATURES + [tgt]].dropna()

    for q in [0.25, 0.50, 0.75]:
        log(f"  QuantReg: {label}, q={q:.2f} ...")
        formula = f"{tgt} ~ month + log_value_usd + num_tokens"
        model = smf.quantreg(formula, data=qr_sub_clean).fit(q=q, max_iter=500)

        coef_df = pd.DataFrame({
            "coef": model.params,
            "std_err": model.bse,
            "t_stat": model.tvalues,
            "p_value": model.pvalues,
        })
        coef_df["quantile"] = q
        coef_df["target"] = label
        coef_df["significant"] = coef_df["p_value"].apply(
            lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        )

        log_df(coef_df.round(6), f"  QuantReg coefficients ({label}, q={q}):")
        quantile_results.append(coef_df)

qr_all = pd.concat(quantile_results)
save_csv(qr_all, "quantile_regression_all.csv")

# ---------- 7c. Random Forest Feature Importance ----------
log("\n  --- 7c. Random Forest (feature importance) ---")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

RF_SIZE = min(2_000_000, len(reg_sample))
rf_sub = reg_sample.sample(n=RF_SIZE, random_state=SEED)
log(f"  Random Forest on {RF_SIZE/1e6:.0f}M rows, {N_CORES} cores")

rf_results = []
for tgt, label in zip(TARGETS_PCT, TARGET_LABELS):
    log(f"\n  Training RF for {label} ...")
    t1 = time.time()

    Xr = rf_sub[FEATURES].values
    yr = rf_sub[tgt].values
    mask = ~(np.isnan(yr) | np.isinf(yr))
    Xr, yr = Xr[mask], yr[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=SEED
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=100,
        n_jobs=N_CORES,
        random_state=SEED,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp_df["target"] = label
    imp_df["R2_test"] = round(r2, 4)
    imp_df["MAE_test"] = round(mae, 2)

    log(f"  RF {label}: R²={r2:.4f}, MAE={mae:.2f}  ({time.time()-t1:.1f}s)")
    log_df(imp_df.round(4), f"  Feature importances ({label}):")
    rf_results.append(imp_df)

rf_all = pd.concat(rf_results)
save_csv(rf_all, "random_forest_importance.csv")


# -----------------------------------------------------------------------------
# 8. ADDITIONAL ANALYTICS
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 8: Additional Analytics")
log("=" * 70)

# 8a. Cross-distance correlation: are wallets that are optimal for one
#     objective also optimal for another?
log("\n  8a. Cross-target correlation (how correlated are the 3 distances?)")
cross_sql = """
SELECT
    CORR(l1_gap_better_return, l1_gap_safer_risk) AS corr_br_sr,
    CORR(l1_gap_better_return, l1_gap_max_sharpe) AS corr_br_ms,
    CORR(l1_gap_safer_risk,    l1_gap_max_sharpe) AS corr_sr_ms
FROM all_data
"""
cross_df = con.sql(cross_sql).fetchdf()
log_df(cross_df.T.rename(columns={0: "pearson_r"}).round(4),
       "  Cross-target Pearson (full data):")
save_csv(cross_df.T.rename(columns={0: "pearson_r"}), "cross_target_correlation.csv")

# 8b. Wallets that are simultaneously optimal across all 3 objectives
log("\n  8b. Multi-optimal wallets (distance=0 for all 3 objectives)")
multi_sql = """
SELECT
    COUNT(l1_gap_better_return + l1_gap_safer_risk + l1_gap_max_sharpe) AS total,  -- rows with all three distances
    SUM(CASE WHEN l1_gap_better_return = 0 AND l1_gap_safer_risk = 0 THEN 1 ELSE 0 END) AS opt_br_and_sr,
    SUM(CASE WHEN l1_gap_better_return = 0 AND l1_gap_max_sharpe = 0 THEN 1 ELSE 0 END) AS opt_br_and_ms,
    SUM(CASE WHEN l1_gap_safer_risk = 0    AND l1_gap_max_sharpe = 0 THEN 1 ELSE 0 END) AS opt_sr_and_ms,
    SUM(CASE WHEN l1_gap_better_return = 0 AND l1_gap_safer_risk = 0 AND l1_gap_max_sharpe = 0 THEN 1 ELSE 0 END) AS opt_all_3
FROM all_data
"""
multi_df = con.sql(multi_sql).fetchdf().iloc[0]
total = multi_df["total"]
log(f"  Total wallets: {total:,}")
for k in ["opt_br_and_sr", "opt_br_and_ms", "opt_sr_and_ms", "opt_all_3"]:
    cnt = int(multi_df[k])
    log(f"  {k}: {cnt:,} ({cnt/total*100:.2f}%)")

multi_out = pd.DataFrame({
    "metric": ["opt_br_and_sr", "opt_br_and_ms", "opt_sr_and_ms", "opt_all_3"],
    "count": [int(multi_df[k]) for k in ["opt_br_and_sr", "opt_br_and_ms", "opt_sr_and_ms", "opt_all_3"]],
    "pct": [round(int(multi_df[k])/total*100, 4) for k in ["opt_br_and_sr", "opt_br_and_ms", "opt_sr_and_ms", "opt_all_3"]],
}).set_index("metric")
save_csv(multi_out, "multi_optimal_wallets.csv")

# 8c. Non-linear features: add num_tokens² and interaction terms to OLS
log("\n  8c. Extended OLS with polynomial + interaction features")

reg_ext = reg_sample[FEATURES + TARGETS_PCT].dropna().copy()
reg_ext["num_tokens_sq"] = reg_ext["num_tokens"] ** 2
reg_ext["log_value_x_tokens"] = reg_ext["log_value_usd"] * reg_ext["num_tokens"]
reg_ext["log_num_tokens"] = np.log1p(reg_ext["num_tokens"])

EXT_FEATURES = FEATURES + ["num_tokens_sq", "log_value_x_tokens", "log_num_tokens"]
X_ext = sm.add_constant(reg_ext[EXT_FEATURES])

for tgt, label in zip(TARGETS_PCT, TARGET_LABELS):
    y = reg_ext[tgt].values
    model = sm.OLS(y, X_ext.values).fit()
    log(f"\n  Extended OLS: {label}  R² = {model.rsquared:.4f} (vs base OLS R² above)")

    coef_df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t_stat": model.tvalues,
        "p_value": model.pvalues,
    }, index=["const"] + EXT_FEATURES)
    coef_df["significant"] = coef_df["p_value"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    )
    log_df(coef_df.round(6), f"  Extended OLS coefficients ({label}):")
    save_csv(coef_df, f"ols_extended_coefs_{label}.csv")

# 8d. Single-token wallets — what fraction are at distance 0?
log("\n  8d. Single-token wallets (num_tokens = 1)")
single_sql = """
SELECT
    COUNT(*) AS total_single_token,
    SUM(CASE WHEN l1_gap_better_return = 0 THEN 1 ELSE 0 END) AS opt_better_return,
    SUM(CASE WHEN l1_gap_safer_risk = 0 THEN 1 ELSE 0 END)    AS opt_safer_risk,
    SUM(CASE WHEN l1_gap_max_sharpe = 0 THEN 1 ELSE 0 END)    AS opt_max_sharpe
FROM all_data
WHERE num_tokens = 1
"""
single_df = con.sql(single_sql).fetchdf().iloc[0]
st = int(single_df["total_single_token"])
log(f"  Single-token wallets: {st:,} ({st/total*100:.2f}% of all)")
if st > 0:
    for k in ["opt_better_return", "opt_safer_risk", "opt_max_sharpe"]:
        v = int(single_df[k])
        log(f"  {k} at dist=0: {v:,} ({v/st*100:.2f}% of single-token)")
else:
    log("  No single-token wallets found — checking minimum token count ...")
    min_tok = con.sql("SELECT MIN(num_tokens) AS mn, MODE(num_tokens) AS mode_tok FROM all_data").fetchdf().iloc[0]
    log(f"  Min num_tokens = {int(min_tok['mn'])}, Mode = {int(min_tok['mode_tok'])}")
    # Re-run for the minimum token count
    mn = int(min_tok["mn"])
    log(f"\n  Smallest-portfolio wallets (num_tokens = {mn}):")
    small_sql = f"""
    SELECT
        COUNT(*) AS total_smallest,
        SUM(CASE WHEN l1_gap_better_return = 0 THEN 1 ELSE 0 END) AS opt_better_return,
        SUM(CASE WHEN l1_gap_safer_risk = 0 THEN 1 ELSE 0 END)    AS opt_safer_risk,
        SUM(CASE WHEN l1_gap_max_sharpe = 0 THEN 1 ELSE 0 END)    AS opt_max_sharpe
    FROM all_data
    WHERE num_tokens = {mn}
    """
    small_df = con.sql(small_sql).fetchdf().iloc[0]
    sm_total = int(small_df["total_smallest"])
    log(f"  Wallets with {mn} tokens: {sm_total:,} ({sm_total/total*100:.2f}% of all)")
    for k in ["opt_better_return", "opt_safer_risk", "opt_max_sharpe"]:
        v = int(small_df[k]) if not pd.isna(small_df[k]) else 0
        log(f"  {k} at dist=0: {v:,} ({v/sm_total*100:.2f}% of {mn}-token wallets)")


# -----------------------------------------------------------------------------
# DONE
# -----------------------------------------------------------------------------
elapsed = time.time() - t0
log()
log("=" * 70)
log(f"ANALYSIS COMPLETE — {elapsed/60:.1f} minutes")
log(f"All results saved to: {OUTPUT_DIR}")
log("=" * 70)

# List all output files
log("\nOutput files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    log(f"  {f:50s} {size:>12,} bytes")

log_file.close()
con.close()
print("\nDone.")
