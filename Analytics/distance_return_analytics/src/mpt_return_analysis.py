#!/usr/bin/env python3
"""
mpt_return_analysis.py

Description:
    Full-scale return analysis pipeline. Companion to
    mpt_distance_analysis.py -- same infrastructure but targets are
    20-day forward returns (ret_*) instead of L1 distances. Covers
    correlations, return distributions, baseline vs optimised
    comparisons, regressions, ETH/BTC holder analysis, and CAPM alpha.
    Designed for ~239M rows, 128 GB RAM, 32 cores.

Input:
    Hive-partitioned parquet directory (ROOT) with block_number,
    wallet_address, returns, betas, market_return, num_tokens.

Output:
    CSV exports + return_analysis_log.txt in OUTPUT_DIR.
9.  Alpha analysis (ret - beta × market_return)
10. Additional analytics
"""

import os, sys, time, json, warnings
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from locations import Location

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
ROOT = str(Location.MPT_DATA_EXTENDED)
OUTPUT_DIR = str(Location.MPT_RETURN_RESULTS)
SAMPLE_SIZE = 50_000_000          # 50M for correlations
REGRESSION_SAMPLE = 10_000_000    # 10M for heavier models
N_CORES = 32
SEED = 42

# Return columns — baseline + 5 strategies
RETURN_COLS = [
    "ret_baseline", "ret_better_return", "ret_safer_risk",
    "ret_max_sharpe", "ret_equal_weight", "ret_mcap_weight",
]
RETURN_LABELS = [
    "baseline", "better_return", "safer_risk",
    "max_sharpe", "equal_weight", "mcap_weight",
]

# Optimised-only (for comparison vs baseline)
OPT_RETURN_COLS = [
    "ret_better_return", "ret_safer_risk", "ret_max_sharpe",
    "ret_equal_weight", "ret_mcap_weight",
]
OPT_RETURN_LABELS = [
    "better_return", "safer_risk", "max_sharpe",
    "equal_weight", "mcap_weight",
]

FEATURES = ["month", "log_value_usd", "num_tokens"]

# Beta columns (for alpha derivation)
BETA_COLS = [
    "beta_baseline", "beta_better_return", "beta_safer_risk",
    "beta_max_sharpe", "beta_equal_weight", "beta_mcap_weight",
]

# Return buckets (percentage points)
RETURN_BUCKETS_LABELS = [
    "< -50%", "[-50%, -20%)", "[-20%, -5%)", "[-5%, 0%)",
    "[0%, 5%)", "[5%, 20%)", "[20%, 50%)", "≥ 50%",
]

# -----------------------------------------------------------------------------
# ETH and BTC wrapper token addresses (lowercase)
# -----------------------------------------------------------------------------
ETH_WRAPPER_ADDRESSES = [
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
    "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0",  # wstETH
    "0xbe9895146f7af43049ca1c1ae358b0541ea49704",  # cbETH
    "0xae78736cd615f374d3085123a210448e74fc6393",  # rETH
    "0xf951e335afb289353dc249e82926178eac7ded78",  # swETH
    "0xd5f7838f5c461feff7fe49ea5ebaf7728bb0adfa",  # mETH
    "0xa35b1b31ce002fbf2058d22f30f95d405200a15b",  # ETHx
    "0x5e8422345238f34275888049021821e8e08caa1f",  # frxETH
    "0xac3e018457b222d93114458476f3e3416abbe38f",  # sfrxETH
    "0xf1c9acdc66974dfb6decb12aa385b9cd01190e38",  # osETH
    "0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee",  # weETH
    "0xbf5495efe5db9ce00f80364c8b423567e58d2110",  # ezETH
    "0xa1290d69c65a6fe4df752f95823fae25cb99e5a7",  # rsETH
    "0xd9a442856c234a39a81a089c06451ebaa4306a72",  # pufETH
    "0xe95a203b1a91a908f9b9ce46459d101078c2c3cb",  # ankrETH
    "0xa2e3356610840701bdf5611a53974510ae27e2e1",  # wBETH
    "0xfe2e637202056d30016725477c5da089ab0a043a",  # sETH2
]

BTC_WRAPPER_ADDRESSES = [
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",  # WBTC
    "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf",  # cbBTC
    "0x18084fba666a33d37592fa2633fd49a74dd93a88",  # tBTC
    "0xeb4c2781e4eba804ce9a9803c67d0893436bb27d",  # renBTC
    "0x0316eb71485b0ab14103307bf65a021042c6d380",  # HBTC
    "0x9be89d2a4cd102d8fecc6bf9da793be995c22541",  # bBTC
    "0x8236a87084f8b84306f72007f36f2618a5634494",  # LBTC
    "0xc96de26018a54d51c097160568752c4e3bd6c364",  # FBTC
    "0x7a56e1c57c7475ccf742a1832b028f0456652f97",  # SolvBTC
    "0xf469fbd2abcd6b9de8e169d128226c0fc90a012e",  # pumpBTC
    "0x004e9c3ef86bc1ca1f0bb5c7662861ee93350568",  # uniBTC
    "0x3f67093dffd4f0af4f2918703c92b60acb7ad78b",  # 21BTC
    "0x73e0c0d45e048d25fc26fa3159b0aa04bfa4db98",  # kBTC
]


def _sql_holds_any(column, addresses):
    """Build a SQL CASE expression that returns 1 if `column` contains any
    of the given token addresses (simple string-contains check — fast on
    239M rows, avoids full JSON parsing)."""
    conditions = " OR ".join(
        f"LOWER({column}) LIKE '%{addr}%'" for addr in addresses
    )
    return f"CASE WHEN ({conditions}) THEN 1 ELSE 0 END"


os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Logging helper – tee to console + file
# -----------------------------------------------------------------------------
LOG_PATH = os.path.join(OUTPUT_DIR, "return_analysis_log.txt")
log_file = open(LOG_PATH, "w", buffering=1)


def log(msg=""):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if msg else ""
    print(line, flush=True)
    log_file.write(line + "\n")


def log_df(df, title=""):
    if title:
        log(title)
    buf = df.to_string()
    print(buf, flush=True)
    log_file.write(buf + "\n")
    log("")


def save_csv(df, name):
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=True)
    log(f"  → saved {path}")


# -----------------------------------------------------------------------------
# 1. Setup DuckDB
# -----------------------------------------------------------------------------
log("=" * 70)
log("MPT RETURN ANALYSIS — FULL SCALE")
log("=" * 70)
log()

t0 = time.time()
con = duckdb.connect(database=":memory:")
con.execute(f"SET threads = {N_CORES}")
con.execute("SET memory_limit = '100GB'")

con.execute(f"""
    CREATE VIEW all_data AS
    SELECT *
    FROM read_parquet('{ROOT}/*/*.parquet', hive_partitioning=true)
""")

total_rows = con.sql("SELECT COUNT(*) AS n FROM all_data").fetchone()[0]
log(f"Total rows: {total_rows:,}")
log()


# -----------------------------------------------------------------------------
# 2. FULL-DATA PEARSON CORRELATION (returns vs features)
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 2: Full-data Pearson Correlation — returns vs features")
log("=" * 70)

# Build CORR() expressions for every (feature, return_col) pair
corr_exprs = []
for feat_sql, feat_key in [("month_feat", "month"), ("log_val", "logval"), ("num_tokens", "numtok")]:
    for ret_col, ret_label in zip(RETURN_COLS, RETURN_LABELS):
        alias = f"pearson_{feat_key}_{ret_label}"
        corr_exprs.append(f"CORR({feat_sql}, {ret_col}) AS {alias}")

# Also correlate market_return
for feat_sql, feat_key in [("month_feat", "month"), ("log_val", "logval"), ("num_tokens", "numtok")]:
    corr_exprs.append(f"CORR({feat_sql}, market_return) AS pearson_{feat_key}_market")

corr_exprs_str = ",\n    ".join(corr_exprs)
pearson_sql = f"""
SELECT
    {corr_exprs_str}
FROM (
    SELECT
        *,
        (EXTRACT(YEAR FROM date) - 2020) * 12
            + EXTRACT(MONTH FROM date)      AS month_feat,
        LN(1 + total_value_usd)             AS log_val
    FROM all_data
) sub
"""

log("  Computing Pearson on full data via DuckDB ...")
t1 = time.time()
pearson_row = con.sql(pearson_sql).fetchdf().iloc[0]
log(f"  Done in {time.time()-t1:.1f}s")

all_ret_labels = RETURN_LABELS + ["market"]
pearson_data = {}
for feat, feat_key in [("month", "month"), ("log_value_usd", "logval"), ("num_tokens", "numtok")]:
    for rl in all_ret_labels:
        col = f"pearson_{feat_key}_{rl}"
        pearson_data.setdefault(feat, {})[rl] = round(pearson_row[col], 4)

pearson_df = pd.DataFrame(pearson_data).T
pearson_df.index.name = "feature"
log_df(pearson_df, "Pearson r — features vs returns (full data):")
save_csv(pearson_df, "pearson_returns_full_data.csv")

# 2b. Binary outcome: win (ret > 0) vs lose (ret <= 0)
log("\n  2b. Pearson correlation — features vs binary win/lose (full data) ...")
bin_corr_exprs = []
for feat_sql, feat_key in [("month_feat", "month"), ("log_val", "logval"), ("num_tokens", "numtok")]:
    for ret_col, ret_label in zip(RETURN_COLS, RETURN_LABELS):
        win_expr = f"CASE WHEN {ret_col} > 0 THEN 1 ELSE 0 END"
        alias = f"pearson_{feat_key}_win_{ret_label}"
        bin_corr_exprs.append(f"CORR({feat_sql}, {win_expr}) AS {alias}")

bin_corr_exprs_str = ",\n    ".join(bin_corr_exprs)
bin_pearson_sql = f"""
SELECT
    {bin_corr_exprs_str}
FROM (
    SELECT
        *,
        (EXTRACT(YEAR FROM date) - 2020) * 12
            + EXTRACT(MONTH FROM date)      AS month_feat,
        LN(1 + total_value_usd)             AS log_val
    FROM all_data
) sub
"""

t1 = time.time()
bin_pearson_row = con.sql(bin_pearson_sql).fetchdf().iloc[0]
log(f"  Done in {time.time()-t1:.1f}s")

bin_pearson_data = {}
for feat, feat_key in [("month", "month"), ("log_value_usd", "logval"), ("num_tokens", "numtok")]:
    for rl in RETURN_LABELS:
        col = f"pearson_{feat_key}_win_{rl}"
        bin_pearson_data.setdefault(feat, {})[rl] = round(bin_pearson_row[col], 4)

bin_pearson_df = pd.DataFrame(bin_pearson_data).T
bin_pearson_df.index.name = "feature"
log_df(bin_pearson_df, "Pearson r — features vs P(win) where win = ret > 0 (full data):")
save_csv(bin_pearson_df, "pearson_win_lose_full_data.csv")


# -----------------------------------------------------------------------------
# 3. LARGE-SAMPLE SPEARMAN + PEARSON (50M rows)
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 3: Large-sample correlation (50M rows) — Pearson + Spearman")
log("=" * 70)

log(f"  Drawing {SAMPLE_SIZE/1e6:.0f}M sample ...")
t1 = time.time()

sample_cols = ["date", "total_value_usd", "num_tokens", "market_return"] + RETURN_COLS
sample = con.sql(f"""
    SELECT {', '.join(sample_cols)}
    FROM all_data
    USING SAMPLE {SAMPLE_SIZE}
""").fetchdf()

sample["date_dt"] = (
    pd.to_datetime(sample["date"], unit="ms")
    if sample["date"].dtype in ["int64", "float64"]
    else pd.to_datetime(sample["date"])
)
sample["month"] = (sample["date_dt"].dt.year - 2020) * 12 + sample["date_dt"].dt.month
sample["log_value_usd"] = np.log1p(sample["total_value_usd"])
log(f"  Sample loaded: {len(sample):,} rows in {time.time()-t1:.1f}s")

corr_rows = []
all_targets = RETURN_COLS + ["market_return"]
all_target_labels = RETURN_LABELS + ["market"]
for feat in FEATURES:
    for tgt, label in zip(all_targets, all_target_labels):
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
save_csv(corr_df, "correlation_returns_sample.csv")

# 3b. Binary outcome: win (ret > 0) vs lose — Pearson + Spearman on sample
log(f"\n  3b. Binary win/lose correlation (sample {SAMPLE_SIZE/1e6:.0f}M) ...")

# Create binary win columns
for ret_col, label in zip(RETURN_COLS, RETURN_LABELS):
    sample[f"win_{label}"] = (sample[ret_col] > 0).astype(int)

bin_corr_rows = []
for feat in FEATURES:
    for label in RETURN_LABELS:
        win_col = f"win_{label}"
        mask = sample[[feat, win_col]].dropna().index
        x, y = sample.loc[mask, feat].values, sample.loc[mask, win_col].values
        pr, _ = sp_stats.pearsonr(x, y)
        sr, _ = sp_stats.spearmanr(x, y)
        bin_corr_rows.append({
            "feature": feat, "target": label,
            "pearson_r": round(pr, 4), "spearman_r": round(sr, 4),
        })

bin_corr_df = pd.DataFrame(bin_corr_rows)
bin_piv_p = bin_corr_df.pivot(index="feature", columns="target", values="pearson_r")
bin_piv_s = bin_corr_df.pivot(index="feature", columns="target", values="spearman_r")
log_df(bin_piv_p, f"Pearson r — features vs P(win) (sample {SAMPLE_SIZE/1e6:.0f}M):")
log_df(bin_piv_s, f"Spearman ρ — features vs P(win) (sample {SAMPLE_SIZE/1e6:.0f}M):")
save_csv(bin_corr_df, "correlation_win_lose_sample.csv")


# -----------------------------------------------------------------------------
# 4. RETURN DISTRIBUTION & DESCRIPTIVE STATISTICS (full data)
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 4: Return Distribution (full data)")
log("=" * 70)

# 4a. Descriptive statistics for every return column + market_return
log("  4a. Descriptive statistics ...")
desc_parts = []
for col in RETURN_COLS + ["market_return"]:
    desc_parts.append(f"""
        AVG({col})                                                 AS mean_{col},
        STDDEV({col})                                              AS std_{col},
        MIN({col})                                                 AS min_{col},
        PERCENTILE_CONT(0.05)  WITHIN GROUP (ORDER BY {col})      AS p05_{col},
        PERCENTILE_CONT(0.25)  WITHIN GROUP (ORDER BY {col})      AS p25_{col},
        MEDIAN({col})                                              AS median_{col},
        PERCENTILE_CONT(0.75)  WITHIN GROUP (ORDER BY {col})      AS p75_{col},
        PERCENTILE_CONT(0.95)  WITHIN GROUP (ORDER BY {col})      AS p95_{col},
        MAX({col})                                                 AS max_{col},
        SUM(CASE WHEN {col} > 0 THEN 1 ELSE 0 END)               AS positive_{col},
        SUM(CASE WHEN {col} = 0 THEN 1 ELSE 0 END)               AS zero_{col},
        SUM(CASE WHEN {col} < 0 THEN 1 ELSE 0 END)               AS negative_{col}
    """)

desc_sql = f"SELECT COUNT(*) AS total, {','.join(desc_parts)} FROM all_data"
t1 = time.time()
desc_row = con.sql(desc_sql).fetchdf().iloc[0]
total = desc_row["total"]
log(f"  Done in {time.time()-t1:.1f}s")

desc_records = []
for col, label in zip(RETURN_COLS + ["market_return"], RETURN_LABELS + ["market"]):
    desc_records.append({
        "portfolio": label,
        "mean": round(desc_row[f"mean_{col}"], 6),
        "std": round(desc_row[f"std_{col}"], 6),
        "min": round(desc_row[f"min_{col}"], 4),
        "p05": round(desc_row[f"p05_{col}"], 4),
        "p25": round(desc_row[f"p25_{col}"], 4),
        "median": round(desc_row[f"median_{col}"], 4),
        "p75": round(desc_row[f"p75_{col}"], 4),
        "p95": round(desc_row[f"p95_{col}"], 4),
        "max": round(desc_row[f"max_{col}"], 4),
        "positive_pct": round(int(desc_row[f"positive_{col}"]) / total * 100, 2),
        "zero_pct": round(int(desc_row[f"zero_{col}"]) / total * 100, 2),
        "negative_pct": round(int(desc_row[f"negative_{col}"]) / total * 100, 2),
    })

desc_df = pd.DataFrame(desc_records).set_index("portfolio")
log_df(desc_df, "Return descriptive statistics (full data):")
save_csv(desc_df, "return_descriptive_stats.csv")


# 4b. Return bucket distribution
log("  4b. Return bucket distribution ...")
bucket_cases = []
for col, label in zip(RETURN_COLS + ["market_return"], RETURN_LABELS + ["market"]):
    # Returns are in decimal form (e.g., -0.31 = -31%)
    pct = f"({col} * 100.0)"
    bucket_cases.append(f"""
        SUM(CASE WHEN {pct} <  -50                       THEN 1 ELSE 0 END) AS "{label}_lt_neg50",
        SUM(CASE WHEN {pct} >= -50 AND {pct} < -20       THEN 1 ELSE 0 END) AS "{label}_neg50_neg20",
        SUM(CASE WHEN {pct} >= -20 AND {pct} <  -5       THEN 1 ELSE 0 END) AS "{label}_neg20_neg5",
        SUM(CASE WHEN {pct} >=  -5 AND {pct} <   0       THEN 1 ELSE 0 END) AS "{label}_neg5_0",
        SUM(CASE WHEN {pct} >=   0 AND {pct} <   5       THEN 1 ELSE 0 END) AS "{label}_0_5",
        SUM(CASE WHEN {pct} >=   5 AND {pct} <  20       THEN 1 ELSE 0 END) AS "{label}_5_20",
        SUM(CASE WHEN {pct} >=  20 AND {pct} <  50       THEN 1 ELSE 0 END) AS "{label}_20_50",
        SUM(CASE WHEN {pct} >=  50                        THEN 1 ELSE 0 END) AS "{label}_gte50"
    """)

dist_sql = f"SELECT COUNT(*) AS total, {','.join(bucket_cases)} FROM all_data"
t1 = time.time()
dist_row = con.sql(dist_sql).fetchdf().iloc[0]
log(f"  Bucket query done in {time.time()-t1:.1f}s")

suffix_keys = [
    "lt_neg50", "neg50_neg20", "neg20_neg5", "neg5_0",
    "0_5", "5_20", "20_50", "gte50",
]
dist_records = []
for label in RETURN_LABELS + ["market"]:
    for bkt_label, sfx in zip(RETURN_BUCKETS_LABELS, suffix_keys):
        cnt = int(dist_row[f"{label}_{sfx}"])
        dist_records.append({
            "portfolio": label, "bucket": bkt_label,
            "count": cnt, "pct": round(cnt / total * 100, 2),
        })

dist_df = pd.DataFrame(dist_records)
dist_pivot_pct = dist_df.pivot(index="bucket", columns="portfolio", values="pct")
dist_pivot_pct = dist_pivot_pct.reindex(RETURN_BUCKETS_LABELS)
dist_pivot_cnt = dist_df.pivot(index="bucket", columns="portfolio", values="count")
dist_pivot_cnt = dist_pivot_cnt.reindex(RETURN_BUCKETS_LABELS)

log_df(dist_pivot_pct, "Return distribution (% of wallets per bucket):")
log_df(dist_pivot_cnt, "Return distribution (count per bucket):")
save_csv(dist_pivot_pct, "return_distribution_pct.csv")
save_csv(dist_pivot_cnt, "return_distribution_count.csv")


# -----------------------------------------------------------------------------
# 5. BASELINE vs OPTIMISED RETURN COMPARISON
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 5: Baseline vs Optimised Return Comparison (full data)")
log("=" * 70)

# 5a. How often does each strategy beat baseline?
log("  5a. Win rates — strategy beats baseline ...")
win_parts = []
for col, label in zip(OPT_RETURN_COLS, OPT_RETURN_LABELS):
    win_parts.append(f"""
        SUM(CASE WHEN {col} > ret_baseline THEN 1 ELSE 0 END) AS wins_{label},
        SUM(CASE WHEN {col} = ret_baseline THEN 1 ELSE 0 END) AS ties_{label},
        SUM(CASE WHEN {col} < ret_baseline THEN 1 ELSE 0 END) AS losses_{label},
        AVG({col} - ret_baseline)                              AS mean_diff_{label},
        MEDIAN({col} - ret_baseline)                           AS median_diff_{label}
    """)

win_sql = f"SELECT COUNT(*) AS total, {','.join(win_parts)} FROM all_data"
t1 = time.time()
win_row = con.sql(win_sql).fetchdf().iloc[0]
log(f"  Done in {time.time()-t1:.1f}s")

win_records = []
for label in OPT_RETURN_LABELS:
    wins = int(win_row[f"wins_{label}"])
    ties = int(win_row[f"ties_{label}"])
    losses = int(win_row[f"losses_{label}"])
    win_records.append({
        "strategy": label,
        "win_pct": round(wins / total * 100, 2),
        "tie_pct": round(ties / total * 100, 2),
        "loss_pct": round(losses / total * 100, 2),
        "mean_diff_vs_baseline": round(win_row[f"mean_diff_{label}"], 6),
        "median_diff_vs_baseline": round(win_row[f"median_diff_{label}"], 6),
    })

win_df = pd.DataFrame(win_records).set_index("strategy")
log_df(win_df, "Strategy vs Baseline — win/tie/loss rates + return differences:")
save_csv(win_df, "strategy_vs_baseline_win_rates.csv")


# -----------------------------------------------------------------------------
# 6. RETURN PROFILES PER DISTANCE BUCKET
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 6: Median returns per L1-distance bucket")
log("=" * 70)

# For each optimisation (better_return, safer_risk, max_sharpe),
# bucket wallets by their L1 distance and show mean returns.
DIST_TARGETS_RAW = ["l1_gap_better_return", "l1_gap_safer_risk", "l1_gap_max_sharpe"]
DIST_LABELS = ["better_return", "safer_risk", "max_sharpe"]

for dist_col, dist_label in zip(DIST_TARGETS_RAW, DIST_LABELS):
    pct_expr = f"({dist_col} / 2.0 * 100.0)"
    log(f"\n--- {dist_label.upper()} distance buckets ---")

    ret_avgs = ", ".join(
        f"MEDIAN({rc}) AS median_{rl}" for rc, rl in zip(RETURN_COLS, RETURN_LABELS)
    )
    ret_avgs += ", MEDIAN(market_return) AS median_market"

    bucket_sql = f"""
    SELECT
        CASE
            WHEN {dist_col} = 0                                    THEN '0_exact'
            WHEN {pct_expr} >  0 AND {pct_expr} <=  1             THEN '(0,1]'
            WHEN {pct_expr} >  1 AND {pct_expr} <= 20             THEN '(1,20]'
            WHEN {pct_expr} > 20 AND {pct_expr} <= 40             THEN '(20,40]'
            WHEN {pct_expr} > 40 AND {pct_expr} <= 60             THEN '(40,60]'
            WHEN {pct_expr} > 60 AND {pct_expr} <= 80             THEN '(60,80]'
            WHEN {pct_expr} > 80 AND {pct_expr} <= 100            THEN '(80,100]'
        END AS dist_bucket,
        COUNT(*) AS n,
        {ret_avgs}
    FROM all_data
    WHERE {dist_col} IS NOT NULL
    GROUP BY dist_bucket
    ORDER BY dist_bucket
    """

    bkt_df = con.sql(bucket_sql).fetchdf()
    order = ["0_exact", "(0,1]", "(1,20]", "(20,40]", "(40,60]", "(60,80]", "(80,100]"]
    bkt_df["dist_bucket"] = pd.Categorical(bkt_df["dist_bucket"], categories=order, ordered=True)
    bkt_df = bkt_df.sort_values("dist_bucket").set_index("dist_bucket")

    log_df(bkt_df.round(6), f"  Median returns per distance bucket ({dist_label}):")
    save_csv(bkt_df.round(6), f"returns_by_distance_bucket_{dist_label}.csv")


# -----------------------------------------------------------------------------
# 7. REGRESSION MODELS (returns as targets)
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 7: Regression models — returns as targets")
log("=" * 70)

log(f"  Drawing {REGRESSION_SAMPLE/1e6:.0f}M sample for regression ...")
t1 = time.time()

reg_cols = (
    ["total_value_usd", "num_tokens", "date", "market_return"]
    + RETURN_COLS
    + BETA_COLS
)
reg_sample = con.sql(f"""
    SELECT {', '.join(reg_cols)}
    FROM all_data
    USING SAMPLE {REGRESSION_SAMPLE}
""").fetchdf()

_dates = (
    pd.to_datetime(reg_sample["date"], unit="ms")
    if reg_sample["date"].dtype in ["int64", "float64"]
    else pd.to_datetime(reg_sample["date"])
)
reg_sample["month"] = (_dates.dt.year - 2020) * 12 + _dates.dt.month
reg_sample["log_value_usd"] = np.log1p(reg_sample["total_value_usd"])

# Derive alphas: alpha_X = ret_X - beta_X × market_return
for ret_col, beta_col, label in zip(RETURN_COLS, BETA_COLS, RETURN_LABELS):
    reg_sample[f"alpha_{label}"] = (
        reg_sample[ret_col] - reg_sample[beta_col] * reg_sample["market_return"]
    )

log(f"  Regression sample: {len(reg_sample):,} rows in {time.time()-t1:.1f}s")

# Convert returns to percentage for readability in regression
for col, label in zip(RETURN_COLS, RETURN_LABELS):
    reg_sample[f"ret_pct_{label}"] = reg_sample[col] * 100.0

RET_PCT_COLS = [f"ret_pct_{l}" for l in RETURN_LABELS]

# Winsorise returns + alphas at 5th / 95th percentile to remove extreme outliers
WINSOR_LO, WINSOR_HI = 0.05, 0.95
log(f"  Winsorising return & alpha columns at p{int(WINSOR_LO*100)}/p{int(WINSOR_HI*100)} ...")

cols_to_winsorise = RET_PCT_COLS + [f"alpha_{l}" for l in RETURN_LABELS]
pre_clip_rows = len(reg_sample)
for col in cols_to_winsorise:
    vals = reg_sample[col].dropna()
    lo, hi = vals.quantile(WINSOR_LO), vals.quantile(WINSOR_HI)
    reg_sample[col] = reg_sample[col].clip(lower=lo, upper=hi)
    log(f"    {col}: clipped to [{lo:.4f}, {hi:.4f}]")

log(f"  Winsorisation done (rows unchanged: {len(reg_sample):,})")

# ---------- 7a. OLS ----------
log("\n  --- 7a. OLS Regression ---")
import statsmodels.api as sm

X = reg_sample[FEATURES].copy()
X = sm.add_constant(X)

ols_results = {}
for tgt, label in zip(RET_PCT_COLS, RETURN_LABELS):
    y = reg_sample[tgt].values
    mask = ~(np.isnan(y) | np.isinf(y))
    model = sm.OLS(y[mask], X.values[mask]).fit()

    log(f"\n  OLS: ret_{label} (%)")
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

    log_df(coef_df.round(6), f"  Coefficients (ret_{label} %):")
    save_csv(coef_df, f"ols_return_coefs_{label}.csv")
    ols_results[label] = {
        "R2": round(model.rsquared, 4),
        "Adj_R2": round(model.rsquared_adj, 4),
        "F_stat": round(model.fvalue, 2),
        "F_pvalue": model.f_pvalue,
    }

ols_summary = pd.DataFrame(ols_results).T
ols_summary.index.name = "target"
log_df(ols_summary, "\n  OLS Summary (returns):")
save_csv(ols_summary, "ols_return_summary.csv")

# ---------- 7b. Quantile Regression ----------
log("\n  --- 7b. Quantile Regression ---")
import statsmodels.formula.api as smf

QR_SIZE = min(2_000_000, len(reg_sample))
qr_sub = reg_sample.sample(n=QR_SIZE, random_state=SEED)
log(f"  Quantile regression on {QR_SIZE/1e6:.0f}M rows")

quantile_results = []
for tgt, label in zip(RET_PCT_COLS, RETURN_LABELS):
    qr_sub_clean = qr_sub[FEATURES + [tgt]].dropna().copy()
    qr_sub_clean.rename(columns={tgt: "y"}, inplace=True)

    for q in [0.25, 0.50, 0.75]:
        log(f"  QuantReg: ret_{label}, q={q:.2f} ...")
        formula = "y ~ month + log_value_usd + num_tokens"
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

        log_df(coef_df.round(6), f"  QuantReg coefficients (ret_{label}, q={q}):")
        quantile_results.append(coef_df)

qr_all = pd.concat(quantile_results)
save_csv(qr_all, "quantile_regression_returns_all.csv")

# ---------- 7c. Random Forest ----------
log("\n  --- 7c. Random Forest (feature importance) ---")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

RF_SIZE = min(2_000_000, len(reg_sample))
rf_sub = reg_sample.sample(n=RF_SIZE, random_state=SEED)
log(f"  Random Forest on {RF_SIZE/1e6:.0f}M rows, {N_CORES} cores")

rf_results = []
for tgt, label in zip(RET_PCT_COLS, RETURN_LABELS):
    log(f"\n  Training RF for ret_{label} ...")
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

    log(f"  RF ret_{label}: R²={r2:.4f}, MAE={mae:.2f}  ({time.time()-t1:.1f}s)")
    log_df(imp_df.round(4), f"  Feature importances (ret_{label}):")
    rf_results.append(imp_df)

rf_all = pd.concat(rf_results)
save_csv(rf_all, "random_forest_return_importance.csv")


# -----------------------------------------------------------------------------
# 8. ETH/BTC HOLDER ANALYSIS + REGRESSIONS
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 8: ETH/BTC Holder Detection & Regressions")
log("=" * 70)

# 8a. Full-data prevalence counts (DuckDB, using string LIKE on holdings_short)
log("  8a. Prevalence of ETH/BTC wrappers in baseline holdings ...")

holds_eth_expr = _sql_holds_any("holdings_short", ETH_WRAPPER_ADDRESSES)
holds_btc_expr = _sql_holds_any("holdings_short", BTC_WRAPPER_ADDRESSES)

prevalence_sql = f"""
SELECT
    COUNT(*)                                                 AS total,
    SUM({holds_eth_expr})                                    AS holds_eth,
    SUM({holds_btc_expr})                                    AS holds_btc,
    SUM(CASE WHEN ({holds_eth_expr}) = 1
              AND ({holds_btc_expr}) = 1 THEN 1 ELSE 0 END) AS holds_both,
    SUM(CASE WHEN ({holds_eth_expr}) = 0
              AND ({holds_btc_expr}) = 0 THEN 1 ELSE 0 END) AS holds_neither
FROM all_data
"""

t1 = time.time()
prev_row = con.sql(prevalence_sql).fetchdf().iloc[0]
log(f"  Done in {time.time()-t1:.1f}s")

for k in ["holds_eth", "holds_btc", "holds_both", "holds_neither"]:
    cnt = int(prev_row[k])
    log(f"  {k}: {cnt:,} ({cnt/total*100:.2f}%)")

prev_out = pd.DataFrame({
    "metric": ["holds_eth", "holds_btc", "holds_both", "holds_neither"],
    "count": [int(prev_row[k]) for k in ["holds_eth", "holds_btc", "holds_both", "holds_neither"]],
    "pct": [round(int(prev_row[k])/total*100, 4) for k in ["holds_eth", "holds_btc", "holds_both", "holds_neither"]],
}).set_index("metric")
save_csv(prev_out, "eth_btc_prevalence.csv")

# 8b. Median returns by ETH/BTC holder status (full data, DuckDB)
log("\n  8b. Median returns by ETH/BTC holder status ...")

ret_medians_str = ", ".join(
    f"MEDIAN({rc}) AS median_{rl}" for rc, rl in zip(RETURN_COLS, RETURN_LABELS)
)
ret_medians_str += ", MEDIAN(market_return) AS median_market"

holder_group_sql = f"""
SELECT
    {holds_eth_expr} AS holds_eth,
    {holds_btc_expr} AS holds_btc,
    COUNT(*) AS n,
    {ret_medians_str}
FROM all_data
GROUP BY holds_eth, holds_btc
ORDER BY holds_eth, holds_btc
"""

t1 = time.time()
holder_df = con.sql(holder_group_sql).fetchdf()
log(f"  Done in {time.time()-t1:.1f}s")
log_df(holder_df.round(6), "  Median returns by (holds_eth, holds_btc):")
save_csv(holder_df.round(6), "returns_by_eth_btc_holder.csv")

# 8c. Regression sample with ETH/BTC flags
log("\n  8c. Drawing regression sample with ETH/BTC flags ...")
t1 = time.time()

eth_btc_reg_cols = (
    ["total_value_usd", "num_tokens", "date", "market_return", "holdings_short"]
    + RETURN_COLS
    + BETA_COLS
)
eth_btc_sample = con.sql(f"""
    SELECT {', '.join(eth_btc_reg_cols)}
    FROM all_data
    USING SAMPLE {REGRESSION_SAMPLE}
""").fetchdf()

_dates2 = (
    pd.to_datetime(eth_btc_sample["date"], unit="ms")
    if eth_btc_sample["date"].dtype in ["int64", "float64"]
    else pd.to_datetime(eth_btc_sample["date"])
)
eth_btc_sample["month"] = (_dates2.dt.year - 2020) * 12 + _dates2.dt.month
eth_btc_sample["log_value_usd"] = np.log1p(eth_btc_sample["total_value_usd"])

# Parse ETH/BTC flags from holdings_short (Python-side, vectorised string check)
holdings_lower = eth_btc_sample["holdings_short"].fillna("").str.lower()
eth_btc_sample["holds_eth"] = holdings_lower.apply(
    lambda s: int(any(addr in s for addr in ETH_WRAPPER_ADDRESSES))
)
eth_btc_sample["holds_btc"] = holdings_lower.apply(
    lambda s: int(any(addr in s for addr in BTC_WRAPPER_ADDRESSES))
)

# Convert returns to %
for col, label in zip(RETURN_COLS, RETURN_LABELS):
    eth_btc_sample[f"ret_pct_{label}"] = eth_btc_sample[col] * 100.0

# Derive alphas
for ret_col, beta_col, label in zip(RETURN_COLS, BETA_COLS, RETURN_LABELS):
    eth_btc_sample[f"alpha_{label}"] = (
        eth_btc_sample[ret_col] - eth_btc_sample[beta_col] * eth_btc_sample["market_return"]
    )
    eth_btc_sample[f"alpha_pct_{label}"] = eth_btc_sample[f"alpha_{label}"] * 100.0

# Winsorise ETH/BTC sample at same thresholds
log(f"  Winsorising ETH/BTC sample at p{int(WINSOR_LO*100)}/p{int(WINSOR_HI*100)} ...")
eb_cols_to_winsorise = (
    [f"ret_pct_{l}" for l in RETURN_LABELS]
    + [f"alpha_{l}" for l in RETURN_LABELS]
    + [f"alpha_pct_{l}" for l in RETURN_LABELS]
)
for col in eb_cols_to_winsorise:
    vals = eth_btc_sample[col].dropna()
    lo, hi = vals.quantile(WINSOR_LO), vals.quantile(WINSOR_HI)
    eth_btc_sample[col] = eth_btc_sample[col].clip(lower=lo, upper=hi)

log(f"  Sample: {len(eth_btc_sample):,} rows in {time.time()-t1:.1f}s")
log(f"  holds_eth=1: {eth_btc_sample['holds_eth'].sum():,}  "
    f"holds_btc=1: {eth_btc_sample['holds_btc'].sum():,}")

# Drop holdings text column to save memory
eth_btc_sample.drop(columns=["holdings_short"], inplace=True)

# 8d. OLS Regressions: features + holds_eth + holds_btc → returns
log("\n  8d. OLS with ETH/BTC holder dummies ...")

FEATURES_ETH_BTC = FEATURES + ["holds_eth", "holds_btc"]
X_eb = eth_btc_sample[FEATURES_ETH_BTC].copy()
X_eb = sm.add_constant(X_eb)

ols_eb_results = {}
for tgt, label in zip(RET_PCT_COLS, RETURN_LABELS):
    y = eth_btc_sample[tgt].values
    mask = ~(np.isnan(y) | np.isinf(y))
    model = sm.OLS(y[mask], X_eb.values[mask]).fit()

    log(f"\n  OLS (ETH/BTC): ret_{label} (%)")
    log(f"  R² = {model.rsquared:.4f},  Adj-R² = {model.rsquared_adj:.4f},  N = {mask.sum():,}")

    coef_df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t_stat": model.tvalues,
        "p_value": model.pvalues,
    }, index=["const"] + FEATURES_ETH_BTC)
    coef_df["significant"] = coef_df["p_value"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    )

    log_df(coef_df.round(6), f"  Coefficients (ret_{label} %, with ETH/BTC):")
    save_csv(coef_df, f"ols_return_ethbtc_coefs_{label}.csv")
    ols_eb_results[label] = {
        "R2": round(model.rsquared, 4),
        "Adj_R2": round(model.rsquared_adj, 4),
        "F_stat": round(model.fvalue, 2),
        "F_pvalue": model.f_pvalue,
    }

ols_eb_summary = pd.DataFrame(ols_eb_results).T
ols_eb_summary.index.name = "target"
log_df(ols_eb_summary, "\n  OLS Summary (returns, with ETH/BTC dummies):")
save_csv(ols_eb_summary, "ols_return_ethbtc_summary.csv")

# 8e. OLS: ETH/BTC dummies → alpha (risk-adjusted return)
log("\n  8e. OLS with ETH/BTC dummies → alpha (risk-adjusted return) ...")

ALPHA_PCT_COLS = [f"alpha_pct_{l}" for l in RETURN_LABELS]

ols_alpha_results = {}
for tgt, label in zip(ALPHA_PCT_COLS, RETURN_LABELS):
    y = eth_btc_sample[tgt].values
    mask = ~(np.isnan(y) | np.isinf(y))
    model = sm.OLS(y[mask], X_eb.values[mask]).fit()

    log(f"\n  OLS (ETH/BTC): alpha_{label} (%)")
    log(f"  R² = {model.rsquared:.4f},  Adj-R² = {model.rsquared_adj:.4f},  N = {mask.sum():,}")

    coef_df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t_stat": model.tvalues,
        "p_value": model.pvalues,
    }, index=["const"] + FEATURES_ETH_BTC)
    coef_df["significant"] = coef_df["p_value"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    )

    log_df(coef_df.round(6), f"  Coefficients (alpha_{label} %, with ETH/BTC):")
    save_csv(coef_df, f"ols_alpha_ethbtc_coefs_{label}.csv")
    ols_alpha_results[label] = {
        "R2": round(model.rsquared, 4),
        "Adj_R2": round(model.rsquared_adj, 4),
        "F_stat": round(model.fvalue, 2),
        "F_pvalue": model.f_pvalue,
    }

ols_alpha_summary = pd.DataFrame(ols_alpha_results).T
ols_alpha_summary.index.name = "target"
log_df(ols_alpha_summary, "\n  OLS Summary (alpha, with ETH/BTC dummies):")
save_csv(ols_alpha_summary, "ols_alpha_ethbtc_summary.csv")

# 8f. Random Forest with ETH/BTC features
log("\n  8f. Random Forest with ETH/BTC features ...")

RF_EB_SIZE = min(2_000_000, len(eth_btc_sample))
rf_eb_sub = eth_btc_sample.sample(n=RF_EB_SIZE, random_state=SEED)
log(f"  RF on {RF_EB_SIZE/1e6:.0f}M rows with features: {FEATURES_ETH_BTC}")

rf_eb_results = []
for tgt, label in zip(RET_PCT_COLS, RETURN_LABELS):
    log(f"\n  Training RF for ret_{label} (with ETH/BTC) ...")
    t1 = time.time()

    Xr = rf_eb_sub[FEATURES_ETH_BTC].values
    yr = rf_eb_sub[tgt].values
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
        "feature": FEATURES_ETH_BTC,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp_df["target"] = label
    imp_df["R2_test"] = round(r2, 4)
    imp_df["MAE_test"] = round(mae, 2)

    log(f"  RF ret_{label} (ETH/BTC): R²={r2:.4f}, MAE={mae:.2f}  ({time.time()-t1:.1f}s)")
    log_df(imp_df.round(4), f"  Feature importances (ret_{label}, with ETH/BTC):")
    rf_eb_results.append(imp_df)

rf_eb_all = pd.concat(rf_eb_results)
save_csv(rf_eb_all, "random_forest_return_ethbtc_importance.csv")


# -----------------------------------------------------------------------------
# 9. ALPHA ANALYSIS
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 9: Alpha Analysis (full data)")
log("=" * 70)

# 9a. Full-data alpha descriptive stats
log("  9a. Alpha descriptive statistics ...")
alpha_parts = []
for ret_col, beta_col, label in zip(RETURN_COLS, BETA_COLS, RETURN_LABELS):
    alpha_expr = f"({ret_col} - {beta_col} * market_return)"
    alpha_parts.append(f"""
        AVG({alpha_expr})                                              AS mean_alpha_{label},
        STDDEV({alpha_expr})                                           AS std_alpha_{label},
        MEDIAN({alpha_expr})                                           AS median_alpha_{label},
        PERCENTILE_CONT(0.05)  WITHIN GROUP (ORDER BY {alpha_expr})   AS p05_alpha_{label},
        PERCENTILE_CONT(0.95)  WITHIN GROUP (ORDER BY {alpha_expr})   AS p95_alpha_{label},
        SUM(CASE WHEN {alpha_expr} > 0 THEN 1 ELSE 0 END)            AS positive_alpha_{label}
    """)

alpha_sql = f"SELECT COUNT(*) AS total, {','.join(alpha_parts)} FROM all_data"
t1 = time.time()
alpha_row = con.sql(alpha_sql).fetchdf().iloc[0]
log(f"  Done in {time.time()-t1:.1f}s")

alpha_records = []
for label in RETURN_LABELS:
    pos = int(alpha_row[f"positive_alpha_{label}"])
    alpha_records.append({
        "portfolio": label,
        "mean_alpha": round(alpha_row[f"mean_alpha_{label}"], 6),
        "std_alpha": round(alpha_row[f"std_alpha_{label}"], 6),
        "median_alpha": round(alpha_row[f"median_alpha_{label}"], 6),
        "p05_alpha": round(alpha_row[f"p05_alpha_{label}"], 4),
        "p95_alpha": round(alpha_row[f"p95_alpha_{label}"], 4),
        "positive_alpha_pct": round(pos / total * 100, 2),
    })

alpha_desc_df = pd.DataFrame(alpha_records).set_index("portfolio")
log_df(alpha_desc_df, "Alpha descriptive statistics (full data):")
save_csv(alpha_desc_df, "alpha_descriptive_stats.csv")

# 9b. Alpha improvement: optimised alpha minus baseline alpha
log("\n  9b. Alpha improvement: optimised − baseline alpha ...")
alpha_imp_parts = []
for ret_col, beta_col, label in zip(OPT_RETURN_COLS,
                                      BETA_COLS[1:],  # skip beta_baseline
                                      OPT_RETURN_LABELS):
    opt_alpha = f"({ret_col} - {beta_col} * market_return)"
    base_alpha = f"(ret_baseline - beta_baseline * market_return)"
    diff = f"({opt_alpha} - {base_alpha})"
    alpha_imp_parts.append(f"""
        AVG({diff})    AS mean_alpha_imp_{label},
        MEDIAN({diff}) AS median_alpha_imp_{label},
        SUM(CASE WHEN {diff} > 0 THEN 1 ELSE 0 END) AS alpha_wins_{label}
    """)

alpha_imp_sql = f"SELECT COUNT(*) AS total, {','.join(alpha_imp_parts)} FROM all_data"
alpha_imp_row = con.sql(alpha_imp_sql).fetchdf().iloc[0]

alpha_imp_records = []
for label in OPT_RETURN_LABELS:
    wins = int(alpha_imp_row[f"alpha_wins_{label}"])
    alpha_imp_records.append({
        "strategy": label,
        "mean_alpha_improvement": round(alpha_imp_row[f"mean_alpha_imp_{label}"], 6),
        "median_alpha_improvement": round(alpha_imp_row[f"median_alpha_imp_{label}"], 6),
        "alpha_win_pct": round(wins / total * 100, 2),
    })

alpha_imp_df = pd.DataFrame(alpha_imp_records).set_index("strategy")
log_df(alpha_imp_df, "Alpha improvement over baseline:")
save_csv(alpha_imp_df, "alpha_improvement_vs_baseline.csv")


# -----------------------------------------------------------------------------
# 10. ADDITIONAL ANALYTICS
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 10: Additional Analytics")
log("=" * 70)

# 10a. Cross-return correlation: how correlated are strategy returns?
log("\n  10a. Cross-return correlation matrix ...")
cross_ret_pairs = []
for i, (c1, l1) in enumerate(zip(RETURN_COLS, RETURN_LABELS)):
    for c2, l2 in zip(RETURN_COLS[i+1:], RETURN_LABELS[i+1:]):
        cross_ret_pairs.append(
            f"CORR({c1}, {c2}) AS corr_{l1}_{l2}"
        )

cross_ret_sql = f"SELECT {', '.join(cross_ret_pairs)} FROM all_data"
t1 = time.time()
cross_ret_row = con.sql(cross_ret_sql).fetchdf().iloc[0]
log(f"  Done in {time.time()-t1:.1f}s")

# Build a full correlation matrix
cross_ret_mat = pd.DataFrame(
    np.eye(len(RETURN_LABELS)),
    index=RETURN_LABELS, columns=RETURN_LABELS,
)
for i, l1 in enumerate(RETURN_LABELS):
    for l2 in RETURN_LABELS[i+1:]:
        val = round(cross_ret_row[f"corr_{l1}_{l2}"], 4)
        cross_ret_mat.loc[l1, l2] = val
        cross_ret_mat.loc[l2, l1] = val

log_df(cross_ret_mat, "Cross-return Pearson correlation matrix:")
save_csv(cross_ret_mat, "cross_return_correlation.csv")

# 10b. Returns by num_tokens buckets
log("\n  10b. Mean returns by num_tokens bucket ...")

tok_ret_avgs = ", ".join(
    f"AVG({rc}) AS mean_{rl}" for rc, rl in zip(RETURN_COLS, RETURN_LABELS)
)
tok_ret_avgs += ", AVG(market_return) AS mean_market"

tok_ret_sql = f"""
SELECT
    CASE
        WHEN num_tokens <= 2   THEN '1-2'
        WHEN num_tokens <= 5   THEN '3-5'
        WHEN num_tokens <= 10  THEN '6-10'
        WHEN num_tokens <= 20  THEN '11-20'
        WHEN num_tokens <= 50  THEN '21-50'
        ELSE '50+'
    END AS token_bucket,
    COUNT(*) AS n,
    {tok_ret_avgs}
FROM all_data
GROUP BY token_bucket
ORDER BY MIN(num_tokens)
"""

tok_ret_df = con.sql(tok_ret_sql).fetchdf()
log_df(tok_ret_df.round(6), "Mean returns by num_tokens bucket:")
save_csv(tok_ret_df.set_index("token_bucket").round(6), "returns_by_token_bucket.csv")

# 10c. Returns by value (USD) bucket
log("\n  10c. Mean returns by portfolio value bucket ...")

val_ret_avgs = ", ".join(
    f"AVG({rc}) AS mean_{rl}" for rc, rl in zip(RETURN_COLS, RETURN_LABELS)
)
val_ret_avgs += ", AVG(market_return) AS mean_market"

val_ret_sql = f"""
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
    COUNT(*) AS n,
    {val_ret_avgs}
FROM all_data
GROUP BY value_bucket
ORDER BY MIN(total_value_usd)
"""

val_ret_df = con.sql(val_ret_sql).fetchdf()
log_df(val_ret_df.round(6), "Mean returns by portfolio value bucket:")
save_csv(val_ret_df.set_index("value_bucket").round(6), "returns_by_value_bucket.csv")

# 10d. Extended OLS: polynomial + interaction + ETH/BTC
log("\n  10d. Extended OLS with polynomial + interaction + ETH/BTC features")

reg_ext = eth_btc_sample[FEATURES_ETH_BTC + RET_PCT_COLS + ALPHA_PCT_COLS].dropna().copy()
reg_ext["num_tokens_sq"] = reg_ext["num_tokens"] ** 2
reg_ext["log_value_x_tokens"] = reg_ext["log_value_usd"] * reg_ext["num_tokens"]
reg_ext["log_num_tokens"] = np.log1p(reg_ext["num_tokens"])
reg_ext["eth_x_btc"] = reg_ext["holds_eth"] * reg_ext["holds_btc"]

EXT_FEATURES = FEATURES_ETH_BTC + [
    "num_tokens_sq", "log_value_x_tokens", "log_num_tokens", "eth_x_btc",
]
X_ext = sm.add_constant(reg_ext[EXT_FEATURES])

for tgt, label in zip(RET_PCT_COLS, RETURN_LABELS):
    y = reg_ext[tgt].values
    model = sm.OLS(y, X_ext.values).fit()
    log(f"\n  Extended OLS: ret_{label}  R² = {model.rsquared:.4f}")

    coef_df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t_stat": model.tvalues,
        "p_value": model.pvalues,
    }, index=["const"] + EXT_FEATURES)
    coef_df["significant"] = coef_df["p_value"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    )
    log_df(coef_df.round(6), f"  Extended OLS coefficients (ret_{label}):")
    save_csv(coef_df, f"ols_return_extended_coefs_{label}.csv")

# Also for alpha targets
for tgt, label in zip(ALPHA_PCT_COLS, RETURN_LABELS):
    y = reg_ext[tgt].values
    mask = ~(np.isnan(y) | np.isinf(y))
    if mask.sum() < 100:
        log(f"  Skipping alpha_{label} — too few valid rows")
        continue
    model = sm.OLS(y[mask], X_ext.values[mask]).fit()
    log(f"\n  Extended OLS: alpha_{label}  R² = {model.rsquared:.4f}")

    coef_df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t_stat": model.tvalues,
        "p_value": model.pvalues,
    }, index=["const"] + EXT_FEATURES)
    coef_df["significant"] = coef_df["p_value"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    )
    log_df(coef_df.round(6), f"  Extended OLS coefficients (alpha_{label}):")
    save_csv(coef_df, f"ols_alpha_extended_coefs_{label}.csv")


# -----------------------------------------------------------------------------
# DONE
# -----------------------------------------------------------------------------
elapsed = time.time() - t0
log()
log("=" * 70)
log(f"RETURN ANALYSIS COMPLETE — {elapsed/60:.1f} minutes")
log(f"All results saved to: {OUTPUT_DIR}")
log("=" * 70)

# List all output files
log("\nOutput files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    log(f"  {f:55s} {size:>12,} bytes")

log_file.close()
con.close()
print("\nDone.")