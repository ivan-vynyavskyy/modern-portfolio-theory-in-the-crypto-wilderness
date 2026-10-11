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

      (3) With --prices-dir, the same two families against ETH instead of
          the market: % of wallets whose 20-day return beats ETH's, and
          quantiles of ret_s - eth_return. ETH's forward return comes from
          the WETH price file with the backtest's rule (close on the
          snapshot date and on date + horizon, a missing close filled from
          at most 3 days before). The 50/50 WETH/WBTC market return is
          recomputed the same way and compared with the stored
          market_return, which checks dates and horizon.

      (4) Long CSV: per block and strategy, wins / losses of the strategy's
          return against the wallet's own (baseline) return. A strategy
          portfolio within 1 % weight distance of the wallet's own (raw L1
          <= 0.02, the paper's near-optimal threshold) counts as the same
          portfolio, not as a win or a loss: such pairs (mostly two-asset
          wallets at the MinVar / MaxRet optimum) differ only by solver
          precision, so their exact return comparison is noise. Exact
          comparisons (ties = equal returns) are kept for reference.

    Exact quantile states hold every value in memory, outside DuckDB's
    memory_limit, so the aggregation runs one block at a time.

Input:
    Hive-partitioned parquet (--root) with per-wallet returns, betas,
    and market_return columns.

Output (in --out-dir):
    pct_beat_market_per_block.csv
    excess_return_quantiles_per_block.csv
    pct_beat_eth_per_block.csv              (with --prices-dir)
    excess_vs_eth_quantiles_per_block.csv   (with --prices-dir)
    pct_beat_baseline_per_block.csv

Usage:
    python analyze_pct_beat_market.py \
        --root     <mpt-optimization-results-eth>/wallet_level \
        --out-dir  <mpt-optimization-results-eth>/return_analysis_results \
        --temp-dir <duckdb_tmp> --threads 32 --mem 100GB \
        --prices-dir <prices>
"""

import argparse
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# CONFIG

DEFAULT_PCT_NAME = "pct_beat_market_per_block.csv"
DEFAULT_QUANTILE_NAME = "excess_return_quantiles_per_block.csv"
ETH_PCT_NAME = "pct_beat_eth_per_block.csv"
ETH_QUANTILE_NAME = "excess_vs_eth_quantiles_per_block.csv"
BASELINE_NAME = "pct_beat_baseline_per_block.csv"

# Weight distance (raw L1, 0-2) between each strategy portfolio and the
# wallet's own; up to SAME_L1 (= 1 % of the paper's 0-100 distance) the two
# count as the same portfolio when comparing returns.
DIST_COL = {
    "better_return": "l1_gap_better_return",
    "safer_risk":    "l1_gap_safer_risk",
    "max_sharpe":    "l1_gap_max_sharpe",
    "equal_weight":  "l1_baseline_vs_equal",
    "mcap_weight":   "l1_baseline_vs_mcap",
}
SAME_L1 = 0.02

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
MAX_STALENESS_DAYS = 3   # as compute_forward_market_return in the backtest

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


def build_eth_columns_sql(strategies, quantiles, eth_return):
    """% of wallets beating ETH's forward return and quantiles of the margin."""
    q_array = "[" + ", ".join(f"{q}" for q in quantiles) + "]"
    eth = repr(float(eth_return))
    parts = []
    for s in strategies:
        parts.append(f"""
            AVG(CASE
                WHEN ret_{s} IS NULL THEN NULL
                WHEN ret_{s} > {eth} THEN 1.0
                ELSE 0.0
            END) * 100 AS pct_beat_eth_ret_{s}""")
        parts.append(f"""
            quantile_cont(
                CASE WHEN ret_{s} IS NOT NULL THEN ret_{s} - {eth} END,
                {q_array}
            ) AS eth_q_{s}""")
    return ",\n".join(parts)


def build_baseline_columns_sql(dist_col, same_l1):
    """Counts of the strategy return against the wallet's own (baseline) return:
    pairs within same_l1 weight distance are the same portfolio (n_same), the
    rest are wins or losses; exact comparisons are kept for reference."""
    parts = []
    for s, d in dist_col.items():
        valid = f"ret_{s} IS NOT NULL AND {d} IS NOT NULL"
        parts.append(f"""
            CAST(SUM(CASE WHEN {valid} THEN 1 ELSE 0 END) AS BIGINT) AS n_cmp_{s},
            CAST(SUM(CASE WHEN {valid} AND {d} <= {same_l1} THEN 1 ELSE 0 END) AS BIGINT) AS n_same_{s},
            CAST(SUM(CASE WHEN {valid} AND {d} > {same_l1} AND ret_{s} > ret_baseline THEN 1 ELSE 0 END) AS BIGINT) AS wins_{s},
            CAST(SUM(CASE WHEN {valid} AND {d} > {same_l1} AND ret_{s} < ret_baseline THEN 1 ELSE 0 END) AS BIGINT) AS losses_{s},
            CAST(SUM(CASE WHEN {valid} AND ret_{s} > ret_baseline THEN 1 ELSE 0 END) AS BIGINT) AS wins_exact_{s},
            CAST(SUM(CASE WHEN {valid} AND ret_{s} = ret_baseline THEN 1 ELSE 0 END) AS BIGINT) AS ties_exact_{s},
            CAST(SUM(CASE WHEN {valid} AND ret_{s} < ret_baseline THEN 1 ELSE 0 END) AS BIGINT) AS losses_exact_{s}""")
    return ",\n".join(parts)


# -----------------------------------------------------------------------------
# BENCHMARK PRICES (same rule as compute_forward_market_return in the backtest)

def load_prices(prices_dir, token):
    path = Path(prices_dir) / f"{token}_price.csv"
    if not path.exists():
        sys.exit(f"missing price file {path}")
    df = pd.read_csv(path, usecols=["date", "daily_close_usd"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def price_at(prices, day):
    """Close on `day`, else the latest close at most MAX_STALENESS_DAYS earlier."""
    exact = prices[prices["date"] == day]
    if not exact.empty:
        return float(exact["daily_close_usd"].iloc[0])
    prior = prices[prices["date"] < day]
    if not prior.empty and (day - prior["date"].iloc[-1]).days <= MAX_STALENESS_DAYS:
        return float(prior["daily_close_usd"].iloc[-1])
    return None


def forward_return(prices, snapshot, horizon_days):
    day = pd.Timestamp(snapshot).tz_localize(None).normalize()
    p0 = price_at(prices, day)
    p1 = price_at(prices, day + pd.Timedelta(days=horizon_days))
    if p0 is None or p1 is None or p0 <= 0 or p1 <= 0:
        return None
    return p1 / p0 - 1.0


# -----------------------------------------------------------------------------
# RESHAPING

def explode_quantiles_to_long(df, strategies, quantiles,
                              metrics=(("excess_return", "exc_q_"), ("alpha", "alpha_q_"))):
    """Turn the list-typed quantile columns into a tidy long frame.

    Output columns: block_number, block_date, n_wallets, strategy,
                    metric ('excess_return' | 'alpha' | ...), quantile, value.
    """
    rows = []
    for _, r in df.iterrows():
        for s in strategies:
            for metric, prefix in metrics:
                vals = r.get(prefix + s)
                if vals is None or (isinstance(vals, float) and np.isnan(vals)):
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
    p.add_argument("--root", type=str, required=True,
                   help="Hive-partitioned wallet-level parquet directory (block_number=*/).")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Output directory for CSVs.")
    p.add_argument("--temp-dir", type=str, required=True,
                   help="DuckDB spill directory (keep it off the home fileset).")
    p.add_argument("--out-name", type=str, default=DEFAULT_PCT_NAME,
                   help="Filename for the wide pct-beat-market CSV.")
    p.add_argument("--quantile-name", type=str, default=DEFAULT_QUANTILE_NAME,
                   help="Filename for the long excess-return-quantile CSV.")
    p.add_argument("--prices-dir", type=str, default=None,
                   help="Per-token price CSVs of the backtest (<address>_price.csv); "
                        "adds the ETH-benchmark outputs.")
    p.add_argument("--horizon-days", type=int, default=20,
                   help="Forward-return horizon of the backtest [default: 20].")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--mem", type=str, default="100GB",
                   help="DuckDB memory_limit.")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.root):
        sys.exit(f"--root is not a directory: {args.root}")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    out_pct = os.path.join(args.out_dir, args.out_name)
    out_q   = os.path.join(args.out_dir, args.quantile_name)

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{args.mem}'")
    con.execute(f"SET threads TO {args.threads}")
    con.execute(f"SET temp_directory = '{args.temp_dir}'")

    blocks = sorted(int(p.name.split("=", 1)[1]) for p in Path(args.root).glob("block_number=*"))
    if not blocks:
        sys.exit(f"no block_number=* partitions under {args.root}")
    print(f"Reading: {args.root}  ({len(blocks)} blocks)")

    pct_sql = build_pct_columns_sql(STRATEGIES)
    q_sql   = build_quantile_columns_sql(STRATEGIES, QUANTILES)
    base_sql = build_baseline_columns_sql(DIST_COL, SAME_L1)
    weth = wbtc = None
    if args.prices_dir:
        weth, wbtc = load_prices(args.prices_dir, WETH), load_prices(args.prices_dir, WBTC)

    print("Running per-block aggregation (pct + quantiles) ...")
    frames, eth_rows = [], []
    for i, blk in enumerate(blocks, 1):
        path = f"{args.root}/block_number={blk}/*.parquet"
        eth_sql = ""
        if weth is not None:
            date, m_lo, m_hi = con.sql(
                f"SELECT MIN(date), MIN(market_return), MAX(market_return) FROM read_parquet('{path}')"
            ).fetchone()
            eth_ret = forward_return(weth, date, args.horizon_days)
            btc_ret = forward_return(wbtc, date, args.horizon_days)
            check = (eth_ret + btc_ret) / 2 if eth_ret is not None and btc_ret is not None else None
            eth_rows.append({"block_number": blk, "horizon_days": args.horizon_days, "eth_return": eth_ret,
                             "market_return": m_lo, "market_return_check": check,
                             "market_return_spread": m_hi - m_lo if m_lo is not None else None})
            if eth_ret is not None:
                eth_sql = ",\n" + build_eth_columns_sql(STRATEGIES, QUANTILES, eth_ret)
        frames.append(con.sql(f"""
            SELECT
                {blk}     AS block_number,
                MIN(date) AS block_date,
                COUNT(*)  AS n_wallets,
                {pct_sql},
                {base_sql},
                {q_sql}{eth_sql}
            FROM read_parquet('{path}')
            WHERE ret_baseline   IS NOT NULL
              AND beta_baseline  IS NOT NULL
              AND market_return  IS NOT NULL
        """).fetchdf())
        print(f"  [{i:2d}/{len(blocks)}] block {blk}: {int(frames[-1]['n_wallets'].iloc[0]):,} wallets",
              flush=True)
    df = pd.concat(frames, ignore_index=True)
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

    write_baseline_outputs(df, args)
    if weth is not None:
        write_eth_outputs(df, pd.DataFrame(eth_rows), args)

    return 0


def write_baseline_outputs(df, args):
    """Wins / losses against the wallet's own return, one row per block and strategy."""
    cols = ["n_cmp", "n_same", "wins", "losses", "wins_exact", "ties_exact", "losses_exact"]
    rows = []
    for s in DIST_COL:
        part = df[["block_number", "block_date"] + [f"{c}_{s}" for c in cols]].copy()
        part.columns = ["block_number", "block_date"] + cols
        part.insert(2, "strategy", s)
        rows.append(part)
    long = pd.concat(rows, ignore_index=True).sort_values(["block_number", "strategy"])
    long["pct_same"] = long["n_same"] / long["n_cmp"] * 100
    long["hit_rate"] = long["wins"] / (long["wins"] + long["losses"]) * 100
    long["hit_rate_exact"] = long["wins_exact"] / (long["wins_exact"] + long["losses_exact"]) * 100
    out = os.path.join(args.out_dir, BASELINE_NAME)
    long.to_csv(out, index=False)
    print(f"  saved: {out}")

    tot = long.groupby("strategy")[cols].sum()
    summary = pd.DataFrame({
        "pct_same": tot["n_same"] / tot["n_cmp"] * 100,
        "hit_rate_pooled": tot["wins"] / (tot["wins"] + tot["losses"]) * 100,
        "hit_rate_block_median": long.groupby("strategy")["hit_rate"].median(),
        "pct_ties_exact": tot["ties_exact"] / tot["n_cmp"] * 100,
        "hit_rate_exact_pooled": tot["wins_exact"] / (tot["wins_exact"] + tot["losses_exact"]) * 100,
    }).loc[list(DIST_COL)]
    print()
    print("=" * 80)
    print(f"  Strategy vs the wallet's own return (same portfolio: raw L1 <= {SAME_L1})")
    print("=" * 80)
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200):
        print(summary.to_string())
    print()


def write_eth_outputs(df, eth, args):
    """ETH-benchmark CSVs and the check of the recomputed market return."""
    eth = df[["block_number", "block_date", "n_wallets"]].merge(eth, on="block_number")
    gap = (eth["market_return_check"] - eth["market_return"]).abs()
    print("=" * 80)
    print(f"  ETH benchmark ({args.horizon_days}-day forward return, WETH price file)")
    print("=" * 80)
    print(f"  market return recomputed from WETH/WBTC: {int(gap.notna().sum())}/{len(eth)} blocks, "
          f"max |diff| vs stored market_return = {gap.max():.2e}; "
          f"spread within a block <= {eth['market_return_spread'].max():.1e}")
    print(f"  ETH return missing for {int(eth['eth_return'].isna().sum())} block(s); "
          f"ETH beat the market in {(eth['eth_return'] > eth['market_return']).sum()}/"
          f"{int(eth['eth_return'].notna().sum())} blocks")

    pct_cols = [f"pct_beat_eth_ret_{s}" for s in STRATEGIES]
    for c in pct_cols:
        if c not in df:
            df[c] = np.nan
    out_pct = os.path.join(args.out_dir, ETH_PCT_NAME)
    eth.merge(df[["block_number"] + pct_cols], on="block_number").to_csv(out_pct, index=False)
    print(f"  saved: {out_pct}")

    long = explode_quantiles_to_long(df, STRATEGIES, QUANTILES, metrics=(("excess_vs_eth", "eth_q_"),))
    out_q = os.path.join(args.out_dir, ETH_QUANTILE_NAME)
    long.to_csv(out_q, index=False)
    print(f"  saved: {out_q}  ({len(long):,} rows)")

    summary = pd.DataFrame({
        "median_pct_beat_eth": [df[c].median() for c in pct_cols],
        "median_pct_beat_market": [df[f"pct_beat_market_ret_{s}"].median() for s in STRATEGIES],
    }, index=STRATEGIES)
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200):
        print("\n  % of wallets beating ETH vs beating the market (median across blocks)")
        print(summary.to_string())
    print()


if __name__ == "__main__":
    raise SystemExit(main())