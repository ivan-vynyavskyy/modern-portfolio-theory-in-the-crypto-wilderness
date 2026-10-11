#!/usr/bin/env python3
"""
rf_zero_test.py

Description:
    Sensitivity of the max-Sharpe (MSR) results to the risk-free rate
    (appendix C, rf robustness). Production 5_0 optimises with r_f = 0; this
    re-runs the unchanged pipeline with r_f = --rf (default 5 %, the T-bill
    anchor) on a random sample of wallets per snapshot and compares the two
    account by account. Only the max-Sharpe fields can differ. Every other
    field must come out identical, which checks that the re-run reproduces
    production; with --rf 0 the whole comparison must be zero.

    Stage 1  sample --per-block wallets per snapshot from the production
             output and copy their holdings rows from the holdings root that
             production read (same hive layout, so 5_0 reads them the same way)
    Stage 2  run 5_0 on the sample with the production settings, r_f = --rf
    Stage 3  join with production; per-wallet and per-snapshot differences and
             an overall summary (source of the RfRobust* constants)

    Each stage skips snapshots already on disk; --force redoes them.

Input:
    --wallet-root       production 5_0 wallet-level output (r_f = 0)
    --holdings-root     4_2 holdings root production read (block_number=*/)
    --pipeline-dir      folder with 5_0_portfolio_backtest_pipeline.py and its
                        helpers (capm_analytics, portfolio_analytics, risk_config)
    --block-to-date-csv, --prices-dir, --supply-csv, --eth-supply-csv:
                        the files production used

Output (in --out-dir):
    sample_holdings/block_number=*/data.parquet           stage 1
    results/block_number=*/data.parquet, results_summary.csv  stage 2
    comparison/wallet_diffs.parquet, per_block_summary.csv,
    comparison/overall_summary.csv                         stage 3
    Differences are re-run minus production, i.e. r_f = --rf minus r_f = 0.

Usage:
    python rf_zero_test.py \
        --wallet-root <mpt-optimization-results-eth>/wallet_level \
        --holdings-root <DATA>/recon-eth-erc20-partitioned \
        --pipeline-dir <src> \
        --block-to-date-csv <DATA>/data/blocks_closest_to_midnight.csv \
        --prices-dir <DATA>/prices \
        --supply-csv <DATA>/data/token_inequality_per_block.csv \
        --eth-supply-csv <DATA>/prices/0xee..ee_supply.csv \
        --out-dir <DATA>/rf_test_eth --temp-dir <duckdb_tmp> --workers 32
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ----------------------------------------------------------------------------
# Config (all paths come from the command line, no defaults)
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Max-Sharpe sensitivity to the risk-free rate: r_f = --rf vs production r_f = 0."
    )
    ap.add_argument("--wallet-root", required=True,
                    help="Production 5_0 wallet-level output (block_number=*/).")
    ap.add_argument("--holdings-root", required=True,
                    help="4_2 holdings root that production read (block_number=*/).")
    ap.add_argument("--pipeline-dir", required=True,
                    help="Folder with 5_0_portfolio_backtest_pipeline.py and its helpers.")
    ap.add_argument("--block-to-date-csv", required=True)
    ap.add_argument("--prices-dir", required=True)
    ap.add_argument("--supply-csv", required=True)
    ap.add_argument("--eth-supply-csv", default=None,
                    help="ETH circulating supply override, as in production.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temp-dir", required=True,
                    help="DuckDB spill directory (keep it off the home fileset).")
    ap.add_argument("--rf", type=float, default=0.05,
                    help="Annual risk-free rate of the re-run (production: 0).")
    ap.add_argument("--per-block", type=int, default=5_000,
                    help="Wallets sampled per snapshot.")
    ap.add_argument("--workers", type=int, default=32, help="5_0 worker processes.")
    ap.add_argument("--threads", type=int, default=32, help="DuckDB threads.")
    ap.add_argument("--mem", default="100GB", help="DuckDB memory_limit.")
    ap.add_argument("--blocks", default=None,
                    help="Comma-separated subset of snapshots (default: all in --wallet-root).")
    ap.add_argument("--force", action="store_true",
                    help="Redo stages for snapshots already on disk.")
    return ap.parse_args()


args = parse_args()
SEED = 20250506

# Production settings: 5_0 main() and the production runs (Sept 2026).
PROD_RISK_CONFIG = dict(
    window_days=60, min_periods=45, clip_outliers=None, return_type="log",
    shrinkage_method="ledoitwolf", max_weight_single_asset=0.9,
)
PROD_MU_SHRINK_LAMBDA = 0.5
PROD_HORIZON_DAYS = 20
PROD_REGIME_LOOKBACK_DAYS = 50
PROD_REGIME_TOP_K_ASSETS = 30

OUT_DIR = Path(args.out_dir)
SAMPLE_DIR = OUT_DIR / "sample_holdings"
RESULTS_DIR = OUT_DIR / "results"
RESULTS_CSV = OUT_DIR / "results_summary.csv"
COMPARISON_DIR = OUT_DIR / "comparison"
for d in (SAMPLE_DIR, RESULTS_DIR, COMPARISON_DIR, Path(args.temp_dir)):
    d.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute(f"SET threads = {args.threads}")
con.execute(f"SET memory_limit = '{args.mem}'")
con.execute(f"SET temp_directory = '{args.temp_dir}'")


def block_file(root: Path, blk: int) -> Path:
    return root / f"block_number={blk}" / "data.parquet"


def readable(path: Path) -> bool:
    """A complete parquet file (a crashed write has no footer)."""
    try:
        pq.read_metadata(str(path))
        return True
    except Exception:
        return False


all_blocks = sorted(int(p.name.split("=", 1)[1]) for p in Path(args.wallet_root).glob("block_number=*"))
BLOCKS = [int(b) for b in args.blocks.split(",")] if args.blocks else all_blocks
if not BLOCKS:
    sys.exit(f"no block_number=* folders in {args.wallet_root}")

# ----------------------------------------------------------------------------
# Stage 1: sample wallets and copy their holdings
# ----------------------------------------------------------------------------

def stage_1_sample() -> None:
    print("=" * 80)
    print(f"STAGE 1: sample {args.per_block:,} wallets per snapshot ({len(BLOCKS)} snapshots)")
    print("=" * 80)
    for blk in BLOCKS:
        out = block_file(SAMPLE_DIR, blk)
        if readable(out) and not args.force:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".parquet.tmp")
        t0 = time.perf_counter()
        # Seeded hash per wallet and snapshot: random, reproducible. The rows
        # are copied unchanged (hive_partitioning off, so no block_number
        # column enters the file body, exactly as in the 4_2 layout).
        con.execute(f"""
            COPY (
                SELECT h.*
                FROM read_parquet('{args.holdings_root}/block_number={blk}/*.parquet',
                                  hive_partitioning=false) h
                WHERE lower(h.wallet_address) IN (
                    SELECT wallet_address
                    FROM read_parquet('{args.wallet_root}/block_number={blk}/*.parquet',
                                      hive_partitioning=false)
                    ORDER BY hash(wallet_address || '-{blk}-{SEED}')
                    LIMIT {args.per_block}
                )
            ) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)
        """)
        tmp.replace(out)
        n = pq.read_metadata(str(out)).num_rows
        print(f"[block {blk}] {n:,} wallets sampled ({time.perf_counter() - t0:.1f}s)")

# ----------------------------------------------------------------------------
# Stage 2: re-run 5_0 on the sample with r_f = --rf
# ----------------------------------------------------------------------------

def load_pipeline() -> Any:
    """Import 5_0 as a library. Its helpers use bare imports (flat src/ on
    the cluster), so the pipeline folder goes on sys.path first. The module
    is registered under its name before exec, and loaded at import time
    (below), so worker processes can unpickle its functions under both fork
    and spawn."""
    pipeline_dir = Path(args.pipeline_dir).resolve()
    path = pipeline_dir / "5_0_portfolio_backtest_pipeline.py"
    if not path.exists():
        sys.exit(f"not found: {path}")
    for d in (pipeline_dir, pipeline_dir.parent / "helpers"):
        if d.is_dir() and str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("backtest_pipeline", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backtest_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


bp = load_pipeline()


def stage_2_rerun() -> None:
    print("=" * 80)
    print(f"STAGE 2: re-run 5_0 with risk_free_rate_annual = {args.rf}")
    print("=" * 80)
    todo = [b for b in BLOCKS if args.force or not readable(block_file(RESULTS_DIR, b))]
    if not todo:
        print("[stage 2] all snapshots already re-run")
        return
    from risk_config import RiskConfig  # on sys.path after load_pipeline()

    cfg = dataclasses.replace(RiskConfig(), **PROD_RISK_CONFIG, risk_free_rate_annual=args.rf)
    bp.HORIZON_DAYS = PROD_HORIZON_DAYS
    bp.MERGE_ETH_WETH = True
    print(f"[cfg] {cfg}")

    supply_lookup = bp.load_supply_data(args.supply_csv)
    if args.eth_supply_csv:
        entries, missing = bp.load_eth_supply_override(
            args.eth_supply_csv, bp.load_block_to_date_map(args.block_to_date_csv))
        if entries:
            supply_lookup[bp.ETH_PSEUDO_TOKEN] = entries
        print(f"[supply] ETH override for {len(entries)} snapshots, {len(missing)} without a value")
    print(f"[supply] {len(supply_lookup):,} tokens")

    t0 = time.perf_counter()
    bp.run_backtest_all_blocks(
        holdings_parquet=str(SAMPLE_DIR),
        block_to_date_csv=args.block_to_date_csv,
        prices_dir=args.prices_dir,
        config=cfg,
        mu_shrink_lambda=PROD_MU_SHRINK_LAMBDA,
        workers=args.workers,
        out_csv=str(RESULTS_CSV),
        out_wallet_parquet=str(RESULTS_DIR),
        blocks=todo,
        regime_lookback_days=PROD_REGIME_LOOKBACK_DAYS,
        regime_top_k_assets=PROD_REGIME_TOP_K_ASSETS,
        batch_size=max(args.per_block, 1_000),   # one batch per snapshot
        batch_timeout_minutes=30,
        max_wallets=None,
        supply_lookup=supply_lookup,
    )
    print(f"[stage 2] {len(todo)} snapshots in {(time.perf_counter() - t0) / 60:.1f} min")

# ----------------------------------------------------------------------------
# Stage 3: compare with production
# ----------------------------------------------------------------------------

# Fields that may change with r_f
MSR_HOLDINGS = ["holdings_max_sharpe"]
MSR_SCALARS = ["l1_gap_max_sharpe", "l1_max_sharpe_vs_equal", "l1_max_sharpe_vs_mcap",
               "beta_max_sharpe", "ret_max_sharpe"]
# Fields that must not (the reproduction check)
FIXED_HOLDINGS = ["holdings_short", "holdings_better_return", "holdings_safer_risk",
                  "holdings_equal_weight", "holdings_mcap_weight"]
FIXED_SCALARS = ["l1_gap_better_return", "l1_gap_safer_risk",
                 "beta_baseline", "beta_better_return", "beta_safer_risk",
                 "beta_equal_weight", "beta_mcap_weight",
                 "ret_baseline", "ret_better_return", "ret_safer_risk",
                 "ret_equal_weight", "ret_mcap_weight", "market_return"]
FIXED_FLAGS = ["better_return_fallback_used", "safer_risk_fallback_used"]
COLUMNS = ["wallet_address"] + MSR_HOLDINGS + MSR_SCALARS + FIXED_HOLDINGS + FIXED_SCALARS + FIXED_FLAGS


def weights(holdings_json: Any) -> Dict[str, float]:
    """Stored holdings JSON -> {token: weight}."""
    if not isinstance(holdings_json, str):
        return {}
    out: Dict[str, float] = {}
    for item in json.loads(holdings_json):
        pct = float(item.get("percentage", 0.0))
        if pct > 0:
            tok = str(item.get("token_address", "")).lower()
            out[tok] = out.get(tok, 0.0) + pct / 100.0
    return out


def l1(a: Dict[str, float], b: Dict[str, float]) -> float:
    return float(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)))


def stage_3_compare() -> None:
    print("=" * 80)
    print(f"STAGE 3: compare r_f = {args.rf} (re-run) with r_f = 0 (production)")
    print("=" * 80)
    cols = ", ".join(COLUMNS)
    per_wallet: List[pd.DataFrame] = []
    per_block: List[Dict[str, Any]] = []
    for blk in BLOCKS:
        test_file = block_file(RESULTS_DIR, blk)
        if not readable(test_file):
            print(f"[block {blk}] no re-run results; skipped")
            continue
        test = con.sql(f"SELECT {cols} FROM read_parquet('{test_file}', hive_partitioning=false)").df()
        prod = con.sql(f"""
            SELECT {cols}
            FROM read_parquet('{args.wallet_root}/block_number={blk}/*.parquet', hive_partitioning=false)
            WHERE wallet_address IN (SELECT wallet_address FROM read_parquet('{test_file}', hive_partitioning=false))
        """).df()
        m = test.merge(prod, on="wallet_address", how="inner", suffixes=("_test", "_prod"))
        if m.empty:
            print(f"[block {blk}] no wallets in common; skipped")
            continue

        m["weight_l1_max_sharpe"] = [l1(weights(a), weights(b)) for a, b in
                                     zip(m["holdings_max_sharpe_test"], m["holdings_max_sharpe_prod"])]
        for c in MSR_SCALARS:
            m[f"diff_{c}"] = m[f"{c}_test"] - m[f"{c}_prod"]
        for c in FIXED_HOLDINGS:
            m[f"check_l1_{c}"] = [l1(weights(a), weights(b)) for a, b in zip(m[f"{c}_test"], m[f"{c}_prod"])]
        for c in FIXED_SCALARS:
            m[f"check_diff_{c}"] = (m[f"{c}_test"] - m[f"{c}_prod"]).abs()
        for c in FIXED_FLAGS:
            m[f"check_flag_{c}"] = (m[f"{c}_test"] != m[f"{c}_prod"]).astype(float)
        m["block_number"] = blk

        check_cols = [c for c in m.columns if c.startswith("check_")]
        keep = (["block_number", "wallet_address", "weight_l1_max_sharpe"]
                + [f"diff_{c}" for c in MSR_SCALARS] + check_cols)
        per_wallet.append(m[keep])

        agg: Dict[str, Any] = {"block_number": blk, "n_compared": len(m),
                               "n_sampled": pq.read_metadata(str(block_file(SAMPLE_DIR, blk))).num_rows}
        for c in ["weight_l1_max_sharpe"] + [f"diff_{x}" for x in MSR_SCALARS]:
            s = m[c].dropna()
            if len(s):
                agg[f"{c}_median"] = float(s.median())
                agg[f"{c}_mean"] = float(s.mean())
                agg[f"{c}_p95_abs"] = float(np.quantile(np.abs(s), 0.95))
        agg["max_check_abs"] = float(np.nanmax(m[check_cols].to_numpy(dtype=float)))
        per_block.append(agg)
        print(f"[block {blk}] n={len(m):,}  MSR weight L1 median={agg.get('weight_l1_max_sharpe_median', np.nan):.4g}  "
              f"max check={agg['max_check_abs']:.3g}")

    if not per_wallet:
        print("[stage 3] nothing to compare")
        return
    df = pd.concat(per_wallet, ignore_index=True)
    df.to_parquet(COMPARISON_DIR / "wallet_diffs.parquet", index=False)
    pd.DataFrame(per_block).to_csv(COMPARISON_DIR / "per_block_summary.csv", index=False)

    overall: Dict[str, Any] = {"rf_test": args.rf, "rf_production": 0.0,
                               "n_blocks": df["block_number"].nunique(), "n_total": len(df)}
    s = df["weight_l1_max_sharpe"].dropna()
    overall.update({
        "weight_l1_max_sharpe_median": float(s.median()),
        "weight_l1_max_sharpe_mean": float(s.mean()),
        "weight_l1_max_sharpe_p95": float(np.quantile(s, 0.95)),
        "weight_l1_max_sharpe_p99": float(np.quantile(s, 0.99)),
        "weight_l1_max_sharpe_max": float(s.max()),
    })
    for c in MSR_SCALARS:
        sc = df[f"diff_{c}"].dropna()
        if len(sc):
            overall[f"diff_{c}_median"] = float(sc.median())
            overall[f"diff_{c}_mean"] = float(sc.mean())
            overall[f"diff_{c}_p95_abs"] = float(np.quantile(np.abs(sc), 0.95))
    for thr in (0.01, 0.05, 0.10):
        overall[f"share_weight_l1_gt_{thr:.2f}"] = float((s > thr).mean())
    check_cols = [c for c in df.columns if c.startswith("check_")]
    overall["max_check_abs"] = float(np.nanmax(df[check_cols].to_numpy(dtype=float)))
    pd.DataFrame([overall]).to_csv(COMPARISON_DIR / "overall_summary.csv", index=False)

    print()
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    for k, v in overall.items():
        print(f"  {k:44s}: {v:.6g}" if isinstance(v, float) else f"  {k:44s}: {v}")
    print()
    print("  - max_check_abs must be ~0: every non-MSR field and both fallback flags")
    print("    match production, so the re-run reproduces it and only r_f differs.")
    print("  - weight_l1_max_sharpe is on [0, 2] (sum of |Δw|): 0.05 = 2.5 % of the value moved.")
    print("  - diff_* are re-run minus production; returns in return units (x100 for pp).")


def main() -> int:
    t0 = time.perf_counter()
    stage_1_sample()
    stage_2_rerun()
    stage_3_compare()
    print(f"\n[total] runtime = {(time.perf_counter() - t0) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
