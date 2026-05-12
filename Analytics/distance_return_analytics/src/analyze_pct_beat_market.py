#!/usr/bin/env python3
"""
analyze_pct_beat_market.py

Description:
    Per-block beat-market statistics for the 6 portfolio strategies
    (baseline + 5 optimised/naive). Two output families:

      (1) Wide CSV: per-block percentage of wallets that beat the market
          and percentage with positive CAPM alpha (one column per
          strategy x metric).

      (2) Long CSV: per-block quantiles of the beat-margin distribution.
          Distributional metrics per strategy:
              excess_return = ret_s - market_return
              alpha         = ret_s - beta_s * market_return
                              (cross-sectional CAPM alpha)
          Quantiles 1, 5, 10, 25, 50, 75, 90, 95, 99 are stored in a
          ggplot-friendly long format.

Input:
    Hive-partitioned parquet (--root) with per-wallet returns, betas,
    and market_return columns.

Output (in --out-dir):
    pct_beat_market_per_block.csv
    excess_return_quantiles_per_block.csv
"""

import argparse
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from locations import Location


# -----------------------------------------------------------------------------
# CONFIG

DEFAULT_ROOT = str(Location.MPT_DATA_EXTENDED)
DEFAULT_OUT_DIR = str(Location.MPT_RETURN_RESULTS)
DEFAULT_PCT_NAME = "pct_beat_market_per_block.csv"
DEFAULT_QUANTILE_NAME = "excess_return_quantiles_per_block.csv"

STRATEGIES = [
    "baseline",
    "better_return",
    "safer_risk",
    "max_sharpe",
    "equal_weight",
    "mcap_weight",
]

# Quantiles of the beat-margin distribution to materialise.
QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


# -----------------------------------------------------------------------------
# SQL BUILDERS

def build_pct_columns_sql(strategies):
    """% of wallets that beat the market (raw return) and have positive alpha."""
    parts = []
    for s in strategies:
        parts.append(f"""
            AVG(CASE
                WHEN ret_{s} IS NULL OR market_return IS NULL THEN NULL
                WHEN ret_{s} > market_return THEN 1.0
                ELSE 0.0
            END) * 100 AS pct_beat_market_ret_{s}""")
        parts.append(f"""
            AVG(CASE
                WHEN ret_{s} IS NULL
                  OR beta_{s} IS NULL
                  OR market_return IS NULL THEN NULL
                WHEN (ret_{s} - beta_{s} * market_return) > 0 THEN 1.0
                ELSE 0.0
            END) * 100 AS pct_positive_alpha_{s}""")
    return ",\n".join(parts)


def build_quantile_columns_sql(strategies, quantiles):
    """Quantiles of the excess-return and alpha distributions per strategy.

    Each strategy contributes two list-valued columns:
        exc_q_<s>     -- quantiles of (ret_s - market_return)
        alpha_q_<s>   -- quantiles of (ret_s - beta_s * market_return)
    NULL inputs are dropped via the inner CASE; quantile_cont then
    ignores them, so each strategy uses its own valid sub-population.
    """
    q_array = "[" + ", ".join(f"{q}" for q in quantiles) + "]"
    parts = []
    for s in strategies:
        parts.append(f"""
            quantile_cont(
                CASE
                    WHEN ret_{s} IS NOT NULL AND market_return IS NOT NULL
                    THEN ret_{s} - market_return
                END,
                {q_array}
            ) AS exc_q_{s}""")
        parts.append(f"""
            quantile_cont(
                CASE
                    WHEN ret_{s} IS NOT NULL
                     AND beta_{s} IS NOT NULL
                     AND market_return IS NOT NULL
                    THEN ret_{s} - beta_{s} * market_return
                END,
                {q_array}
            ) AS alpha_q_{s}""")
    return ",\n".join(parts)


# -----------------------------------------------------------------------------
# RESHAPING

def explode_quantiles_to_long(df, strategies, quantiles):
    """Turn the list-typed quantile columns into a tidy long frame.

    Output columns: block_number, block_date, n_wallets, strategy,
                    metric ('excess_return' | 'alpha'), quantile, value.
    """
    rows = []
    for _, r in df.iterrows():
        for s in strategies:
            for metric, col in (("excess_return", f"exc_q_{s}"),
                                ("alpha",         f"alpha_q_{s}")):
                vals = r[col]
                if vals is None:
                    continue
                for q, v in zip(quantiles, vals):
                    rows.append({
                        "block_number": r["block_number"],
                        "block_date":   r["block_date"],
                        "n_wallets":    r["n_wallets"],
                        "strategy":     s,
                        "metric":       metric,
                        "quantile":     q,
                        "value":        v,
                    })
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["block_number", "strategy", "metric", "quantile"]
    ).reset_index(drop=True)


# -----------------------------------------------------------------------------
# MAIN

def parse_args():
    p = argparse.ArgumentParser(
        description="Per-block beat-market percentages and excess-return "
                    "quantiles, all 6 strategies."
    )
    p.add_argument("--root", type=str, default=DEFAULT_ROOT,
                   help="Root parquet directory.")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                   help="Output directory for CSVs.")
    p.add_argument("--out-name", type=str, default=DEFAULT_PCT_NAME,
                   help="Filename for the wide pct-beat-market CSV.")
    p.add_argument("--quantile-name", type=str, default=DEFAULT_QUANTILE_NAME,
                   help="Filename for the long excess-return-quantile CSV.")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--memory-limit", type=str, default="100GB")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_pct = os.path.join(args.out_dir, args.out_name)
    out_q   = os.path.join(args.out_dir, args.quantile_name)

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute(f"SET threads TO {args.threads}")

    parquet_glob = f"{args.root}/*/*.parquet"
    print(f"Reading: {parquet_glob}")

    pct_sql = build_pct_columns_sql(STRATEGIES)
    q_sql   = build_quantile_columns_sql(STRATEGIES, QUANTILES)

    sql = f"""
        SELECT
            block_number,
            MIN(date) AS block_date,
            COUNT(*)  AS n_wallets,
            {pct_sql},
            {q_sql}
        FROM read_parquet('{parquet_glob}', hive_partitioning=true)
        WHERE ret_baseline   IS NOT NULL
          AND beta_baseline  IS NOT NULL
          AND market_return  IS NOT NULL
        GROUP BY block_number
        ORDER BY block_number
    """

    print("Running per-block aggregation (pct + quantiles) ...")
    df = con.sql(sql).fetchdf()
    print(f"  -> {len(df)} blocks, "
          f"{df['n_wallets'].sum():,} wallet-block observations")

    # ------------------------------------------------------------------
    # 1) Wide pct CSV (unchanged schema)
    # ------------------------------------------------------------------
    pct_cols = [c for c in df.columns
                if c.startswith("pct_beat_market_ret_")
                or c.startswith("pct_positive_alpha_")]
    df_pct = df[["block_number", "block_date", "n_wallets"] + pct_cols]
    df_pct.to_csv(out_pct, index=False)
    print(f"  saved: {out_pct}")

    # ------------------------------------------------------------------
    # 2) Long quantile CSV (block x strategy x metric x quantile -> value)
    # ------------------------------------------------------------------
    df_long = explode_quantiles_to_long(df, STRATEGIES, QUANTILES)
    df_long.to_csv(out_q, index=False)
    print(f"  saved: {out_q}  ({len(df_long):,} rows)")

    # ------------------------------------------------------------------
    # Console summary: pct-beat-market across blocks
    # ------------------------------------------------------------------
    rows = []
    for s in STRATEGIES:
        rows.append({
            "strategy": s,
            "mean_pct_beat_ret":     df[f"pct_beat_market_ret_{s}"].mean(),
            "median_pct_beat_ret":   df[f"pct_beat_market_ret_{s}"].median(),
            "mean_pct_pos_alpha":    df[f"pct_positive_alpha_{s}"].mean(),
            "median_pct_pos_alpha":  df[f"pct_positive_alpha_{s}"].median(),
        })
    summary = pd.DataFrame(rows).set_index("strategy")

    print()
    print("=" * 80)
    print("  Average across blocks (unweighted, %)")
    print("=" * 80)
    with pd.option_context("display.float_format", "{:.2f}".format,
                           "display.width", 200):
        print(summary.to_string())

    # ------------------------------------------------------------------
    # Console summary: median across blocks of each (strategy, metric, q)
    # ------------------------------------------------------------------
    q_summary = (df_long
                 .groupby(["strategy", "metric", "quantile"])["value"]
                 .median()
                 .unstack("quantile"))

    print()
    print("=" * 80)
    print("  Excess-return / alpha quantiles -- median across blocks")
    print("=" * 80)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200):
        print(q_summary.to_string())
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())