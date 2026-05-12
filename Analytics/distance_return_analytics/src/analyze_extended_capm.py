#!/usr/bin/env python3
"""
analyze_extended_capm.py

Description:
    Comprehensive analysis of MPT optimisation results with CAPM betas,
    naive strategies (equal-weight, mcap-weight), and cross-distances.
    Engine: DuckDB (handles 200M+ rows without loading into pandas).

Input:
    Hive-partitioned parquet (--root) with block_number, wallet_address,
    L1 distances, returns, betas, and alphas.

Output:
    CSV files for per-block stats in --out-dir, plus console summaries.

Usage:
    python analyze_extended_capm.py \
        --root /path/to/parquet/data --out-dir ./results
"""

import argparse
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from locations import Location


# -----------------------------------------------------------------------------
# CONFIG

DEFAULT_ROOT = str(Location.MPT_DATA_EXTENDED)
DEFAULT_OUT_DIR = "./analysis_results_3"

STRATEGIES = ["baseline", "better_return", "safer_risk", "max_sharpe", "equal_weight", "mcap_weight"]
MPT_STRATEGIES = ["better_return", "safer_risk", "max_sharpe"]
NAIVE_STRATEGIES = ["equal_weight", "mcap_weight"]

L1_BASELINE_VS_MPT = ["l1_gap_better_return", "l1_gap_safer_risk", "l1_gap_max_sharpe"]
L1_BASELINE_VS_NAIVE = ["l1_baseline_vs_equal", "l1_baseline_vs_mcap"]
L1_CROSS = [
    "l1_better_return_vs_equal", "l1_safer_risk_vs_equal", "l1_max_sharpe_vs_equal",
    "l1_better_return_vs_mcap", "l1_safer_risk_vs_mcap", "l1_max_sharpe_vs_mcap",
]
ALL_L1 = L1_BASELINE_VS_MPT + L1_BASELINE_VS_NAIVE + L1_CROSS

RET_COLS = [f"ret_{s}" for s in STRATEGIES] + ["market_return"]
BETA_COLS = [f"beta_{s}" for s in STRATEGIES]


# -----------------------------------------------------------------------------
# HELPERS

def separator(title: str, char: str = "=", width: int = 80) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_df_nice(df: pd.DataFrame, title: str = "", float_fmt: str = ".6f") -> None:
    if title:
        print(f"\n--- {title} ---")
    with pd.option_context("display.float_format", f"{{:{float_fmt}}}".format, "display.max_columns", 50, "display.width", 200):
        print(df.to_string())
    print()


def save_csv(df: pd.DataFrame, path: str, label: str) -> None:
    df.to_csv(path, index=False)
    print(f"  [saved] {label} -> {path}")


# -----------------------------------------------------------------------------
# SETUP

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extended CAPM analysis of MPT optimization results.")
    p.add_argument("--root", type=str, default=DEFAULT_ROOT, help="Root parquet directory.")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for CSV files.")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--memory-limit", type=str, default="100GB")
    return p.parse_args()


def create_view(con: duckdb.DuckDBPyConnection, parquet_glob: str) -> int:
    """Create the main view with all derived columns. Returns row count."""
    con.execute(f"""
        CREATE OR REPLACE VIEW wd AS
        SELECT
            *,
            -- Alphas: ret - beta * market_return
            ret_baseline       - beta_baseline       * market_return AS alpha_baseline,
            ret_better_return  - beta_better_return  * market_return AS alpha_better_return,
            ret_safer_risk     - beta_safer_risk     * market_return AS alpha_safer_risk,
            ret_max_sharpe     - beta_max_sharpe     * market_return AS alpha_max_sharpe,
            ret_equal_weight   - beta_equal_weight   * market_return AS alpha_equal_weight,
            ret_mcap_weight    - beta_mcap_weight    * market_return AS alpha_mcap_weight,

            -- Alpha diffs vs baseline
            (ret_better_return - beta_better_return * market_return) -
                (ret_baseline  - beta_baseline      * market_return) AS alpha_diff_better_return,
            (ret_safer_risk    - beta_safer_risk    * market_return) -
                (ret_baseline  - beta_baseline      * market_return) AS alpha_diff_safer_risk,
            (ret_max_sharpe    - beta_max_sharpe    * market_return) -
                (ret_baseline  - beta_baseline      * market_return) AS alpha_diff_max_sharpe,
            (ret_equal_weight  - beta_equal_weight  * market_return) -
                (ret_baseline  - beta_baseline      * market_return) AS alpha_diff_equal_weight,
            (ret_mcap_weight   - beta_mcap_weight   * market_return) -
                (ret_baseline  - beta_baseline      * market_return) AS alpha_diff_mcap_weight,

            -- L1 as pct
            l1_gap_better_return     / 2.0 * 100.0 AS l1_pct_better_return,
            l1_gap_safer_risk        / 2.0 * 100.0 AS l1_pct_safer_risk,
            l1_gap_max_sharpe        / 2.0 * 100.0 AS l1_pct_max_sharpe,
            l1_baseline_vs_equal     / 2.0 * 100.0 AS l1_pct_baseline_vs_equal,
            l1_baseline_vs_mcap      / 2.0 * 100.0 AS l1_pct_baseline_vs_mcap,

            -- Wealth bucket
            CASE
                WHEN total_value_usd < 1      THEN '0-1'
                WHEN total_value_usd < 100    THEN '1-100'
                WHEN total_value_usd < 1000   THEN '100-1K'
                WHEN total_value_usd < 10000  THEN '1K-10K'
                ELSE '10K+'
            END AS wealth_bucket,

            LN(1 + total_value_usd) AS log_value_usd

        FROM read_parquet({parquet_glob}, hive_partitioning=true)
        WHERE ret_baseline IS NOT NULL
          AND beta_baseline IS NOT NULL
          AND market_return IS NOT NULL
    """)

    n = con.sql("SELECT COUNT(*) FROM wd").fetchone()[0]
    return n


# -----------------------------------------------------------------------------
# 1 & 2: L1 DISTANCE STATS

def analyze_l1_distances(con: duckdb.DuckDBPyConnection, out_dir: str) -> None:
    separator("1 & 2. L1 Distance Statistics")

    cols_sql = ",\n        ".join([
        f"AVG({c}) AS mean_{c}, MEDIAN({c}) AS median_{c}, STDDEV({c}) AS std_{c}"
        for c in ALL_L1
    ])

    # Overall
    overall = con.sql(f"""
        SELECT COUNT(*) AS n, {cols_sql}
        FROM wd
    """).fetchdf()

    # Reshape for nice printing (convert to % scale: raw / 2 * 100)
    rows = []
    for c in ALL_L1:
        rows.append({
            "l1_metric": c,
            "mean (%)": overall[f"mean_{c}"].iloc[0] / 2 * 100,
            "median (%)": overall[f"median_{c}"].iloc[0] / 2 * 100,
            "std (%)": overall[f"std_{c}"].iloc[0] / 2 * 100,
        })
    df_overall = pd.DataFrame(rows)
    print_df_nice(df_overall, "L1 Distances — Overall (all blocks)")

    # Per block
    cols_sql_block = ",\n        ".join([
        f"AVG({c}) AS mean_{c}, MEDIAN({c}) AS median_{c}, STDDEV({c}) AS std_{c}"
        for c in ALL_L1
    ])
    per_block = con.sql(f"""
        SELECT
            block_number,
            MIN(date) AS block_date,
            COUNT(*) AS n_wallets,
            {cols_sql_block}
        FROM wd
        GROUP BY block_number
        ORDER BY block_number
    """).fetchdf()

    save_csv(per_block, os.path.join(out_dir, "l1_distances_per_block.csv"), "L1 distances per block")


# -----------------------------------------------------------------------------
# 3: RETURN STATS

def analyze_returns(con: duckdb.DuckDBPyConnection, out_dir: str) -> None:
    separator("3. Return Statistics (20-day forward)")

    ret_cols = RET_COLS
    cols_sql = ",\n        ".join([
        f"AVG({c}) AS mean_{c}, MEDIAN({c}) AS median_{c}, STDDEV({c}) AS std_{c}"
        for c in ret_cols
    ])

    overall = con.sql(f"SELECT COUNT(*) AS n, {cols_sql} FROM wd").fetchdf()

    rows = []
    for c in ret_cols:
        rows.append({
            "return_metric": c,
            "median (%)": overall[f"median_{c}"].iloc[0] * 100,
        })
    df_overall = pd.DataFrame(rows)
    print_df_nice(df_overall, "Returns — Overall (all blocks)")

    # Per block
    cols_sql_block = ",\n        ".join([
        f"AVG({c}) AS mean_{c}, MEDIAN({c}) AS median_{c}, STDDEV({c}) AS std_{c}"
        for c in ret_cols
    ])
    per_block = con.sql(f"""
        SELECT
            block_number,
            MIN(date) AS block_date,
            COUNT(*) AS n_wallets,
            {cols_sql_block}
        FROM wd
        GROUP BY block_number
        ORDER BY block_number
    """).fetchdf()

    save_csv(per_block, os.path.join(out_dir, "returns_per_block.csv"), "Returns per block")


# -----------------------------------------------------------------------------
# 4: BETA & ALPHA STATS

def analyze_betas_alphas(con: duckdb.DuckDBPyConnection, out_dir: str) -> None:
    separator("4. Beta & Alpha Statistics")

    alpha_cols = [f"alpha_{s}" for s in STRATEGIES]
    beta_cols = BETA_COLS
    all_cols = beta_cols + alpha_cols

    cols_sql = ",\n        ".join([
        f"AVG({c}) AS mean_{c}, MEDIAN({c}) AS median_{c}, STDDEV({c}) AS std_{c}"
        for c in all_cols
    ])

    overall = con.sql(f"SELECT COUNT(*) AS n, {cols_sql} FROM wd").fetchdf()

    print("--- Betas — Overall ---")
    rows = []
    for c in beta_cols:
        rows.append({
            "metric": c,
            "mean": overall[f"mean_{c}"].iloc[0],
            "median": overall[f"median_{c}"].iloc[0],
            "std": overall[f"std_{c}"].iloc[0],
        })
    print_df_nice(pd.DataFrame(rows))

    print("--- Alphas — Overall ---")
    rows = []
    for c in alpha_cols:
        rows.append({
            "metric": c,
            "mean (%)": overall[f"mean_{c}"].iloc[0] * 100,
            "median (%)": overall[f"median_{c}"].iloc[0] * 100,
            "std (%)": overall[f"std_{c}"].iloc[0] * 100,
        })
    print_df_nice(pd.DataFrame(rows))

    # Per block
    cols_sql_block = ",\n        ".join([
        f"AVG({c}) AS mean_{c}, MEDIAN({c}) AS median_{c}, STDDEV({c}) AS std_{c}"
        for c in all_cols
    ])
    per_block = con.sql(f"""
        SELECT
            block_number,
            MIN(date) AS block_date,
            COUNT(*) AS n_wallets,
            {cols_sql_block}
        FROM wd
        GROUP BY block_number
        ORDER BY block_number
    """).fetchdf()

    save_csv(per_block, os.path.join(out_dir, "betas_alphas_per_block.csv"), "Betas & Alphas per block")


# -----------------------------------------------------------------------------
# 5: ALPHA DISTRIBUTION

def analyze_alpha_distribution(con: duckdb.DuckDBPyConnection, out_dir: str) -> None:
    separator("5. Alpha Distribution — Do Wallets Outperform On a Risk-Adjusted Basis?")

    alpha_cols = [f"alpha_{s}" for s in STRATEGIES]

    # Overall stats + fraction positive
    parts = []
    for c in alpha_cols:
        parts.append(f"""
            AVG({c}) AS mean_{c},
            MEDIAN({c}) AS median_{c},
            STDDEV({c}) AS std_{c},
            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY {c}) AS p5_{c},
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {c}) AS p25_{c},
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {c}) AS p75_{c},
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {c}) AS p95_{c},
            AVG(CASE WHEN {c} > 0 THEN 1.0 ELSE 0.0 END) AS frac_positive_{c}
        """)
    cols_sql = ",\n".join(parts)

    overall = con.sql(f"SELECT COUNT(*) AS n, {cols_sql} FROM wd").fetchdf()

    rows = []
    for c in alpha_cols:
        rows.append({
            "strategy": c.replace("alpha_", ""),
            "mean (%)": overall[f"mean_{c}"].iloc[0] * 100,
            "median (%)": overall[f"median_{c}"].iloc[0] * 100,
            "std (%)": overall[f"std_{c}"].iloc[0] * 100,
            "p5 (%)": overall[f"p5_{c}"].iloc[0] * 100,
            "p25 (%)": overall[f"p25_{c}"].iloc[0] * 100,
            "p75 (%)": overall[f"p75_{c}"].iloc[0] * 100,
            "p95 (%)": overall[f"p95_{c}"].iloc[0] * 100,
            "frac_positive": overall[f"frac_positive_{c}"].iloc[0],
        })
    df_summary = pd.DataFrame(rows)
    print_df_nice(df_summary, "Alpha Summary by Strategy")

    # Histogram data: binned alpha distribution per strategy (for plotting)
    hist_rows = []
    for s in STRATEGIES:
        col = f"alpha_{s}"
        binned = con.sql(f"""
            SELECT
                '{s}' AS strategy,
                FLOOR({col} * 100) / 100.0 AS alpha_bin,
                COUNT(*) AS count
            FROM wd
            WHERE {col} IS NOT NULL
              AND {col} BETWEEN -2 AND 2
            GROUP BY alpha_bin
            ORDER BY alpha_bin
        """).fetchdf()
        hist_rows.append(binned)

    df_hist = pd.concat(hist_rows, ignore_index=True)
    save_csv(df_hist, os.path.join(out_dir, "alpha_distribution_binned.csv"), "Alpha distribution (binned)")

    # Per block: fraction positive alpha per strategy
    parts_block = []
    for s in STRATEGIES:
        c = f"alpha_{s}"
        parts_block.append(f"""
            AVG({c}) AS mean_{c},
            MEDIAN({c}) AS median_{c},
            AVG(CASE WHEN {c} > 0 THEN 1.0 ELSE 0.0 END) AS frac_positive_{c}
        """)
    cols_block = ",\n".join(parts_block)

    per_block = con.sql(f"""
        SELECT
            block_number,
            MIN(date) AS block_date,
            COUNT(*) AS n_wallets,
            {cols_block}
        FROM wd
        GROUP BY block_number
        ORDER BY block_number
    """).fetchdf()

    save_csv(per_block, os.path.join(out_dir, "alpha_per_block.csv"), "Alpha stats per block")


# -----------------------------------------------------------------------------
# 6: BETA DISTRIBUTION

def analyze_beta_distribution(con: duckdb.DuckDBPyConnection, out_dir: str) -> None:
    separator("6. Beta Distribution — How Much Market Risk Do Wallets Take?")

    beta_cols = BETA_COLS

    parts = []
    for c in beta_cols:
        parts.append(f"""
            AVG({c}) AS mean_{c},
            MEDIAN({c}) AS median_{c},
            STDDEV({c}) AS std_{c},
            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY {c}) AS p5_{c},
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {c}) AS p25_{c},
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {c}) AS p75_{c},
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {c}) AS p95_{c},
            AVG(CASE WHEN {c} > 1 THEN 1.0 ELSE 0.0 END) AS frac_above_1_{c},
            AVG(CASE WHEN {c} < 0 THEN 1.0 ELSE 0.0 END) AS frac_negative_{c}
        """)
    cols_sql = ",\n".join(parts)
    overall = con.sql(f"SELECT COUNT(*) AS n, {cols_sql} FROM wd").fetchdf()

    rows = []
    for c in beta_cols:
        rows.append({
            "strategy": c.replace("beta_", ""),
            "mean": overall[f"mean_{c}"].iloc[0],
            "median": overall[f"median_{c}"].iloc[0],
            "std": overall[f"std_{c}"].iloc[0],
            "p5": overall[f"p5_{c}"].iloc[0],
            "p25": overall[f"p25_{c}"].iloc[0],
            "p75": overall[f"p75_{c}"].iloc[0],
            "p95": overall[f"p95_{c}"].iloc[0],
            "frac_above_1": overall[f"frac_above_1_{c}"].iloc[0],
            "frac_negative": overall[f"frac_negative_{c}"].iloc[0],
        })
    df_summary = pd.DataFrame(rows)
    print_df_nice(df_summary, "Beta Summary by Strategy")

    # Histogram data for plotting
    hist_rows = []
    for s in STRATEGIES:
        col = f"beta_{s}"
        binned = con.sql(f"""
            SELECT
                '{s}' AS strategy,
                FLOOR({col} * 20) / 20.0 AS beta_bin,
                COUNT(*) AS count
            FROM wd
            WHERE {col} IS NOT NULL
              AND {col} BETWEEN -1 AND 5
            GROUP BY beta_bin
            ORDER BY beta_bin
        """).fetchdf()
        hist_rows.append(binned)

    df_hist = pd.concat(hist_rows, ignore_index=True)
    save_csv(df_hist, os.path.join(out_dir, "beta_distribution_binned.csv"), "Beta distribution (binned)")

    # Per block
    parts_block = []
    for c in beta_cols:
        parts_block.append(f"AVG({c}) AS mean_{c}, MEDIAN({c}) AS median_{c}")
    cols_block = ",\n".join(parts_block)

    per_block = con.sql(f"""
        SELECT
            block_number,
            MIN(date) AS block_date,
            COUNT(*) AS n_wallets,
            {cols_block}
        FROM wd
        GROUP BY block_number
        ORDER BY block_number
    """).fetchdf()

    save_csv(per_block, os.path.join(out_dir, "beta_per_block.csv"), "Beta stats per block")


# -----------------------------------------------------------------------------
# 7: L1 GAP vs ALPHA

def analyze_l1_vs_alpha(con: duckdb.DuckDBPyConnection, out_dir: str) -> None:
    separator("7. Does L1 Gap Predict Alpha?")

    # Pearson correlations (full data)
    corr_pairs = [
        ("l1_gap_better_return",    "alpha_baseline", "L1 better_ret vs α_baseline"),
        ("l1_gap_safer_risk",       "alpha_baseline", "L1 safer_risk vs α_baseline"),
        ("l1_gap_max_sharpe",       "alpha_baseline", "L1 max_sharpe vs α_baseline"),
        ("l1_baseline_vs_equal",    "alpha_baseline", "L1 base_vs_equal vs α_baseline"),
        ("l1_baseline_vs_mcap",     "alpha_baseline", "L1 base_vs_mcap vs α_baseline"),
        ("num_tokens",              "alpha_baseline", "num_tokens vs α_baseline"),
        ("beta_baseline",           "alpha_baseline", "β_baseline vs α_baseline"),
        ("l1_gap_safer_risk",       "alpha_safer_risk", "L1 safer_risk vs α_safer_risk"),
        ("l1_gap_max_sharpe",       "alpha_max_sharpe", "L1 max_sharpe vs α_max_sharpe"),
        ("l1_baseline_vs_equal",    "alpha_equal_weight", "L1 base_vs_equal vs α_equal_weight"),
        ("l1_baseline_vs_mcap",     "alpha_mcap_weight",  "L1 base_vs_mcap vs α_mcap_weight"),
    ]

    corr_sql = ",\n        ".join([
        f"CORR({a}, {b}) AS pearson_{a}_vs_{b}" for a, b, _ in corr_pairs
    ])
    corr_df = con.sql(f"SELECT {corr_sql} FROM wd").fetchdf()

    print("--- Pearson Correlations (full data) ---\n")
    for a, b, label in corr_pairs:
        val = corr_df[f"pearson_{a}_vs_{b}"].iloc[0]
        print(f"  {label:50s}  r = {val:+.4f}")
    print()

    # Spearman on sample (DuckDB doesn't have native Spearman)
    sample_size = min(5_000_000, con.sql("SELECT COUNT(*) FROM wd").fetchone()[0])
    spearman_cols = [
        "l1_gap_better_return", "l1_gap_safer_risk", "l1_gap_max_sharpe",
        "l1_baseline_vs_equal", "l1_baseline_vs_mcap",
        "alpha_baseline", "alpha_better_return", "alpha_safer_risk",
        "alpha_max_sharpe", "alpha_equal_weight", "alpha_mcap_weight",
        "num_tokens", "beta_baseline",
    ]
    sample = con.sql(f"""
        SELECT {', '.join(spearman_cols)}
        FROM wd
        USING SAMPLE {sample_size}
    """).fetchdf()

    print(f"--- Spearman Correlations (sample n={len(sample):,}) ---\n")
    spearman_pairs = [
        ("l1_gap_safer_risk",    "alpha_baseline"),
        ("l1_gap_better_return", "alpha_baseline"),
        ("l1_gap_max_sharpe",    "alpha_baseline"),
        ("l1_baseline_vs_equal", "alpha_baseline"),
        ("l1_baseline_vs_mcap",  "alpha_baseline"),
        ("num_tokens",           "alpha_baseline"),
        ("beta_baseline",        "alpha_baseline"),
        ("l1_baseline_vs_equal", "alpha_equal_weight"),
        ("l1_baseline_vs_mcap",  "alpha_mcap_weight"),
    ]
    spearman_rows = []
    for a, b in spearman_pairs:
        mask = sample[[a, b]].dropna().index
        if len(mask) < 100:
            continue
        rho, p = sp_stats.spearmanr(sample.loc[mask, a], sample.loc[mask, b])
        label = f"{a} vs {b}"
        print(f"  {label:55s}  ρ = {rho:+.4f}  (p = {p:.2e})")
        spearman_rows.append({"col_a": a, "col_b": b, "spearman_rho": rho, "p_value": p})
    print()

    save_csv(pd.DataFrame(spearman_rows), os.path.join(out_dir, "spearman_l1_vs_alpha.csv"), "Spearman L1 vs Alpha")

    # Binned: mean alpha by L1 gap ventiles (for plotting)
    l1_vs_alpha_configs = [
        ("l1_pct_better_return", "alpha_baseline", "l1_better_return_vs_alpha_baseline"),
        ("l1_pct_safer_risk",    "alpha_baseline", "l1_safer_risk_vs_alpha_baseline"),
        ("l1_pct_max_sharpe",    "alpha_baseline", "l1_max_sharpe_vs_alpha_baseline"),
        ("l1_pct_baseline_vs_equal", "alpha_baseline", "l1_equal_vs_alpha_baseline"),
        ("l1_pct_baseline_vs_mcap",  "alpha_baseline", "l1_mcap_vs_alpha_baseline"),
        ("l1_pct_baseline_vs_equal", "alpha_equal_weight", "l1_equal_vs_alpha_equal"),
        ("l1_pct_baseline_vs_mcap",  "alpha_mcap_weight",  "l1_mcap_vs_alpha_mcap"),
    ]

    all_binned = []
    for l1_col, alpha_col, label in l1_vs_alpha_configs:
        binned = con.sql(f"""
            WITH ventiles AS (
                SELECT
                    NTILE(20) OVER (ORDER BY {l1_col}) AS bin,
                    {l1_col} AS l1_val,
                    {alpha_col} AS alpha_val
                FROM wd
                WHERE {l1_col} IS NOT NULL AND {alpha_col} IS NOT NULL
            )
            SELECT
                '{label}' AS pair,
                bin,
                AVG(l1_val) AS mean_l1_pct,
                AVG(alpha_val) AS mean_alpha,
                MEDIAN(alpha_val) AS median_alpha,
                COUNT(*) AS n
            FROM ventiles
            GROUP BY bin
            ORDER BY bin
        """).fetchdf()
        all_binned.append(binned)

    df_binned = pd.concat(all_binned, ignore_index=True)
    save_csv(df_binned, os.path.join(out_dir, "l1_vs_alpha_binned.csv"), "L1 vs Alpha (binned ventiles)")


# -----------------------------------------------------------------------------
# 8: DID OPTIMIZER IMPROVE?

def analyze_optimizer_improvement(con: duckdb.DuckDBPyConnection, out_dir: str) -> None:
    separator("8. Did the Optimizer Improve Risk-Adjusted Returns?")

    strategies_vs_baseline = STRATEGIES[1:]  # all except baseline

    parts = []
    for s in strategies_vs_baseline:
        parts.append(f"""
            -- {s}: raw return diff
            AVG(ret_{s} - ret_baseline) * 100            AS mean_ret_diff_{s}_pp,
            MEDIAN(ret_{s} - ret_baseline) * 100         AS median_ret_diff_{s}_pp,

            -- {s}: alpha diff
            AVG(alpha_diff_{s}) * 100                    AS mean_alpha_diff_{s}_pp,
            MEDIAN(alpha_diff_{s}) * 100                 AS median_alpha_diff_{s}_pp,

            -- {s}: hit rates
            AVG(CASE WHEN ret_{s} > ret_baseline THEN 1.0 ELSE 0.0 END) * 100
                AS ret_hit_rate_{s}_pct,
            AVG(CASE WHEN alpha_diff_{s} > 0 THEN 1.0 ELSE 0.0 END) * 100
                AS alpha_hit_rate_{s}_pct
        """)

    cols_sql = ",\n".join(parts)
    overall = con.sql(f"SELECT COUNT(*) AS n, {cols_sql} FROM wd").fetchdf()

    rows = []
    for s in strategies_vs_baseline:
        rows.append({
            "strategy": s,
            "mean_ret_diff_pp": overall[f"mean_ret_diff_{s}_pp"].iloc[0],
            "median_ret_diff_pp": overall[f"median_ret_diff_{s}_pp"].iloc[0],
            "mean_alpha_diff_pp": overall[f"mean_alpha_diff_{s}_pp"].iloc[0],
            "median_alpha_diff_pp": overall[f"median_alpha_diff_{s}_pp"].iloc[0],
            "ret_hit_rate_pct": overall[f"ret_hit_rate_{s}_pct"].iloc[0],
            "alpha_hit_rate_pct": overall[f"alpha_hit_rate_{s}_pct"].iloc[0],
        })
    df_overall = pd.DataFrame(rows)
    print_df_nice(df_overall, "Strategy Improvement Over Baseline — Overall")

    # Per block for plotting
    parts_block = []
    for s in strategies_vs_baseline:
        parts_block.append(f"""
            AVG(ret_{s} - ret_baseline) * 100 AS mean_ret_diff_{s}_pp,
            AVG(alpha_diff_{s}) * 100 AS mean_alpha_diff_{s}_pp,
            AVG(CASE WHEN alpha_diff_{s} > 0 THEN 1.0 ELSE 0.0 END) * 100
                AS alpha_hit_rate_{s}_pct
        """)
    cols_block = ",\n".join(parts_block)

    per_block = con.sql(f"""
        SELECT
            block_number,
            MIN(date) AS block_date,
            COUNT(*) AS n_wallets,
            {cols_block}
        FROM wd
        GROUP BY block_number
        ORDER BY block_number
    """).fetchdf()

    save_csv(per_block, os.path.join(out_dir, "optimizer_improvement_per_block.csv"),
             "Optimizer improvement per block")

    # Alpha diff distribution (binned, for plotting)
    hist_rows = []
    for s in strategies_vs_baseline:
        binned = con.sql(f"""
            SELECT
                '{s}' AS strategy,
                FLOOR(alpha_diff_{s} * 200) / 200.0 AS alpha_diff_bin,
                COUNT(*) AS count
            FROM wd
            WHERE alpha_diff_{s} IS NOT NULL
              AND alpha_diff_{s} BETWEEN -1 AND 1
            GROUP BY alpha_diff_bin
            ORDER BY alpha_diff_bin
        """).fetchdf()
        hist_rows.append(binned)

    df_hist = pd.concat(hist_rows, ignore_index=True)
    save_csv(df_hist, os.path.join(out_dir, "alpha_diff_distribution_binned.csv"),
             "Alpha diff distribution (binned)")


# -----------------------------------------------------------------------------
# 9: BETA CHANGE BY OPTIMIZER

def analyze_beta_change(con: duckdb.DuckDBPyConnection, out_dir: str) -> None:
    separator("9. Does the Optimizer Increase or Decrease Market Risk?")

    strategies_vs_baseline = STRATEGIES[1:]

    parts = []
    for s in strategies_vs_baseline:
        parts.append(f"""
            AVG(beta_{s} - beta_baseline)    AS mean_dbeta_{s},
            MEDIAN(beta_{s} - beta_baseline) AS median_dbeta_{s},
            STDDEV(beta_{s} - beta_baseline) AS std_dbeta_{s},
            AVG(CASE WHEN beta_{s} > beta_baseline THEN 1.0 ELSE 0.0 END) * 100
                AS pct_increased_{s}
        """)
    cols_sql = ",\n".join(parts)
    overall = con.sql(f"SELECT COUNT(*) AS n, {cols_sql} FROM wd").fetchdf()

    rows = []
    for s in strategies_vs_baseline:
        rows.append({
            "strategy": s,
            "mean_Δβ": overall[f"mean_dbeta_{s}"].iloc[0],
            "median_Δβ": overall[f"median_dbeta_{s}"].iloc[0],
            "std_Δβ": overall[f"std_dbeta_{s}"].iloc[0],
            "pct_increased": overall[f"pct_increased_{s}"].iloc[0],
        })
    df_overall = pd.DataFrame(rows)
    print_df_nice(df_overall, "Beta Change (strategy − baseline) — Overall")

    # By num_tokens (for plotting: does optimizer change beta differently for complex portfolios?)
    parts_nt = []
    for s in strategies_vs_baseline:
        parts_nt.append(f"AVG(beta_{s} - beta_baseline) AS mean_dbeta_{s}")
    cols_nt = ",\n".join(parts_nt)

    by_ntokens = con.sql(f"""
        SELECT
            num_tokens,
            COUNT(*) AS n,
            {cols_nt}
        FROM wd
        WHERE num_tokens <= 30
        GROUP BY num_tokens
        HAVING COUNT(*) >= 200
        ORDER BY num_tokens
    """).fetchdf()

    save_csv(by_ntokens, os.path.join(out_dir, "beta_change_by_num_tokens.csv"),
             "Beta change by num_tokens")

    # Per block
    parts_block = []
    for s in strategies_vs_baseline:
        parts_block.append(f"AVG(beta_{s} - beta_baseline) AS mean_dbeta_{s}")
    cols_block = ",\n".join(parts_block)

    per_block = con.sql(f"""
        SELECT
            block_number,
            MIN(date) AS block_date,
            COUNT(*) AS n_wallets,
            {cols_block}
        FROM wd
        GROUP BY block_number
        ORDER BY block_number
    """).fetchdf()

    save_csv(per_block, os.path.join(out_dir, "beta_change_per_block.csv"), "Beta change per block")


# -----------------------------------------------------------------------------
# EXTRA: CROSS-STRATEGY COMPARISON

def analyze_cross_strategy(con: duckdb.DuckDBPyConnection, out_dir: str) -> None:
    separator("EXTRA: Cross-Strategy Return & Alpha Comparison")

    # Rank strategies by return per wallet
    rank_df = con.sql("""
        SELECT
            -- Which strategy had the best return per wallet?
            AVG(CASE WHEN ret_better_return >= ret_safer_risk
                      AND ret_better_return >= ret_max_sharpe
                      AND ret_better_return >= ret_equal_weight
                      AND ret_better_return >= COALESCE(ret_mcap_weight, -1e18)
                      AND ret_better_return >= ret_baseline
                 THEN 1.0 ELSE 0.0 END) * 100 AS pct_best_better_return,
            AVG(CASE WHEN ret_safer_risk >= ret_better_return
                      AND ret_safer_risk >= ret_max_sharpe
                      AND ret_safer_risk >= ret_equal_weight
                      AND ret_safer_risk >= COALESCE(ret_mcap_weight, -1e18)
                      AND ret_safer_risk >= ret_baseline
                 THEN 1.0 ELSE 0.0 END) * 100 AS pct_best_safer_risk,
            AVG(CASE WHEN ret_max_sharpe >= ret_better_return
                      AND ret_max_sharpe >= ret_safer_risk
                      AND ret_max_sharpe >= ret_equal_weight
                      AND ret_max_sharpe >= COALESCE(ret_mcap_weight, -1e18)
                      AND ret_max_sharpe >= ret_baseline
                 THEN 1.0 ELSE 0.0 END) * 100 AS pct_best_max_sharpe,
            AVG(CASE WHEN ret_equal_weight >= ret_better_return
                      AND ret_equal_weight >= ret_safer_risk
                      AND ret_equal_weight >= ret_max_sharpe
                      AND ret_equal_weight >= COALESCE(ret_mcap_weight, -1e18)
                      AND ret_equal_weight >= ret_baseline
                 THEN 1.0 ELSE 0.0 END) * 100 AS pct_best_equal_weight,
            AVG(CASE WHEN ret_baseline >= ret_better_return
                      AND ret_baseline >= ret_safer_risk
                      AND ret_baseline >= ret_max_sharpe
                      AND ret_baseline >= ret_equal_weight
                      AND ret_baseline >= COALESCE(ret_mcap_weight, -1e18)
                 THEN 1.0 ELSE 0.0 END) * 100 AS pct_best_baseline
        FROM wd
    """).fetchdf()

    print("--- How often is each strategy the BEST (highest return)? ---")
    print_df_nice(rank_df.T.rename(columns={0: "pct_best"}))

    # Naive vs MPT: which is closer to baseline?
    naive_vs_mpt = con.sql("""
        SELECT
            AVG(l1_baseline_vs_equal) AS mean_l1_baseline_vs_equal,
            AVG(l1_baseline_vs_mcap)  AS mean_l1_baseline_vs_mcap,
            AVG(l1_gap_better_return) AS mean_l1_baseline_vs_better_return,
            AVG(l1_gap_safer_risk)    AS mean_l1_baseline_vs_safer_risk,
            AVG(l1_gap_max_sharpe)    AS mean_l1_baseline_vs_max_sharpe,

            -- How often is baseline closer to equal-weight than to any MPT strategy?
            AVG(CASE WHEN l1_baseline_vs_equal < l1_gap_safer_risk
                 THEN 1.0 ELSE 0.0 END) * 100 AS pct_closer_to_equal_than_safer,
            AVG(CASE WHEN l1_baseline_vs_mcap < l1_gap_safer_risk
                 THEN 1.0 ELSE 0.0 END) * 100 AS pct_closer_to_mcap_than_safer
        FROM wd
    """).fetchdf()

    print("--- How far is baseline from each strategy (mean L1 %)? ---")
    # Convert L1 columns to % for printing (raw / 2 * 100)
    naive_print = naive_vs_mpt.copy()
    for col in naive_print.columns:
        if col.startswith("mean_l1_"):
            naive_print[col] = naive_print[col] / 2 * 100
    print_df_nice(naive_print.T.rename(columns={0: "value"}))

    # Per block for time series
    per_block = con.sql("""
        SELECT
            block_number,
            MIN(date) AS block_date,
            COUNT(*) AS n_wallets,
            AVG(ret_baseline) * 100 AS mean_ret_baseline_pct,
            AVG(ret_equal_weight) * 100 AS mean_ret_equal_pct,
            AVG(ret_mcap_weight) * 100 AS mean_ret_mcap_pct,
            AVG(ret_better_return) * 100 AS mean_ret_better_pct,
            AVG(ret_safer_risk) * 100 AS mean_ret_safer_pct,
            AVG(ret_max_sharpe) * 100 AS mean_ret_sharpe_pct,
            AVG(alpha_baseline) * 100 AS mean_alpha_baseline_pct,
            AVG(alpha_equal_weight) * 100 AS mean_alpha_equal_pct,
            AVG(alpha_mcap_weight) * 100 AS mean_alpha_mcap_pct,
            AVG(market_return) * 100 AS mean_market_return_pct
        FROM wd
        GROUP BY block_number
        ORDER BY block_number
    """).fetchdf()

    save_csv(per_block, os.path.join(out_dir, "cross_strategy_per_block.csv"),
             "Cross-strategy returns/alpha per block")


# -----------------------------------------------------------------------------
# EXTRA: MPT vs NAIVE DIRECTION

def analyze_mpt_vs_naive_direction(con: duckdb.DuckDBPyConnection, out_dir: str) -> None:
    separator("EXTRA: Does MPT Optimization Move Wallets Toward or Away From Naive Strategies?")

    # 3 MPT × 2 naive = 6 pairs
    # naive_short maps to the suffix used in L1 column names
    pairs = [
        (mpt, naive, "equal" if naive == "equal_weight" else "mcap")
        for mpt in MPT_STRATEGIES
        for naive in NAIVE_STRATEGIES
    ]

    # Build SQL: delta = l1_{mpt}_vs_{naive_short} - l1_baseline_vs_{naive_short}
    parts = []
    for mpt, naive, ns in pairs:
        tag = f"{mpt}_vs_{ns}"
        parts.append(f"""
            AVG(l1_{mpt}_vs_{ns} - l1_baseline_vs_{ns})    AS mean_delta_{tag},
            MEDIAN(l1_{mpt}_vs_{ns} - l1_baseline_vs_{ns}) AS median_delta_{tag},
            AVG(CASE WHEN (l1_{mpt}_vs_{ns} - l1_baseline_vs_{ns}) < -0.001
                 THEN 1.0 ELSE 0.0 END) * 100 AS pct_closer_{tag},
            AVG(CASE WHEN (l1_{mpt}_vs_{ns} - l1_baseline_vs_{ns}) >  0.001
                 THEN 1.0 ELSE 0.0 END) * 100 AS pct_farther_{tag},
            AVG(CASE WHEN ABS(l1_{mpt}_vs_{ns} - l1_baseline_vs_{ns}) <= 0.001
                 THEN 1.0 ELSE 0.0 END) * 100 AS pct_unchanged_{tag}
        """)
    cols_sql = ",\n".join(parts)

    overall = con.sql(f"SELECT COUNT(*) AS n, {cols_sql} FROM wd").fetchdf()

    rows = []
    for mpt, naive, ns in pairs:
        tag = f"{mpt}_vs_{ns}"
        rows.append({
            "mpt_strategy": mpt,
            "naive_strategy": naive,
            "mean_delta_l1": overall[f"mean_delta_{tag}"].iloc[0],
            "median_delta_l1": overall[f"median_delta_{tag}"].iloc[0],
            "pct_moved_closer": overall[f"pct_closer_{tag}"].iloc[0],
            "pct_moved_farther": overall[f"pct_farther_{tag}"].iloc[0],
            "pct_unchanged": overall[f"pct_unchanged_{tag}"].iloc[0],
        })
    df_overall = pd.DataFrame(rows)
    print_df_nice(df_overall, "MPT vs Naive Direction — Overall")

    # Per block
    parts_block = []
    for mpt, naive, ns in pairs:
        tag = f"{mpt}_vs_{ns}"
        parts_block.append(f"""
            AVG(l1_{mpt}_vs_{ns} - l1_baseline_vs_{ns})    AS mean_delta_{tag},
            MEDIAN(l1_{mpt}_vs_{ns} - l1_baseline_vs_{ns}) AS median_delta_{tag},
            AVG(CASE WHEN (l1_{mpt}_vs_{ns} - l1_baseline_vs_{ns}) < -0.001
                 THEN 1.0 ELSE 0.0 END) * 100 AS pct_closer_{tag},
            AVG(CASE WHEN (l1_{mpt}_vs_{ns} - l1_baseline_vs_{ns}) >  0.001
                 THEN 1.0 ELSE 0.0 END) * 100 AS pct_farther_{tag},
            AVG(CASE WHEN ABS(l1_{mpt}_vs_{ns} - l1_baseline_vs_{ns}) <= 0.001
                 THEN 1.0 ELSE 0.0 END) * 100 AS pct_unchanged_{tag}
        """)
    cols_block = ",\n".join(parts_block)

    per_block = con.sql(f"""
        SELECT
            block_number,
            MIN(date) AS block_date,
            COUNT(*) AS n_wallets,
            {cols_block}
        FROM wd
        GROUP BY block_number
        ORDER BY block_number
    """).fetchdf()

    save_csv(per_block, os.path.join(out_dir, "mpt_vs_naive_direction_per_block.csv"),
             "MPT vs Naive direction per block")


# -----------------------------------------------------------------------------
# EXTRA: DATA COVERAGE

def analyze_coverage(con: duckdb.DuckDBPyConnection) -> None:
    separator("DATA COVERAGE")

    coverage = con.sql("""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT block_number) AS n_blocks,

            -- Mcap coverage (null = missing supply/price)
            SUM(CASE WHEN ret_mcap_weight IS NOT NULL THEN 1 ELSE 0 END) AS n_with_mcap,
            SUM(CASE WHEN ret_mcap_weight IS NULL THEN 1 ELSE 0 END) AS n_without_mcap,
            ROUND(SUM(CASE WHEN ret_mcap_weight IS NOT NULL THEN 1 ELSE 0 END) * 100.0
                  / COUNT(*), 2) AS mcap_coverage_pct,

            AVG(num_tokens) AS mean_num_tokens,
            MEDIAN(num_tokens) AS median_num_tokens,
            AVG(total_value_usd) AS mean_value_usd,
            MEDIAN(total_value_usd) AS median_value_usd
        FROM wd
    """).fetchdf()

    print_df_nice(coverage.T.rename(columns={0: "value"}), "Data Coverage")


# -----------------------------------------------------------------------------
# MAIN

def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute(f"SET threads TO {args.threads}")

    parquet_glob = f"'{args.root}/**/data.parquet'"

    separator("LOADING DATA")
    n = create_view(con, parquet_glob)
    print(f"Total rows: {n:,}")
    print(f"Output directory: {out_dir}")

    analyze_coverage(con)
    analyze_l1_distances(con, out_dir)
    analyze_returns(con, out_dir)
    analyze_betas_alphas(con, out_dir)
    analyze_alpha_distribution(con, out_dir)
    analyze_beta_distribution(con, out_dir)
    analyze_l1_vs_alpha(con, out_dir)
    analyze_optimizer_improvement(con, out_dir)
    analyze_beta_change(con, out_dir)
    analyze_cross_strategy(con, out_dir)
    analyze_mpt_vs_naive_direction(con, out_dir)

    separator("DONE")
    print(f"All CSV files saved to: {out_dir}/")
    print(f"Files:")
    for f in sorted(Path(out_dir).glob("*.csv")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:50s} {size_mb:8.2f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())