#!/usr/bin/env python3
"""
4_4_eth_wealth_distribution.py

Purpose
-------
Distribution of native-ETH holdings across the ETH + ERC-20 holdings dataset,
to ground the "is small ETH an investment position or gas float?" decision
(gas-reserve threshold for the robustness slice) in actual data.

Reports, for three populations of wallet-block rows —
    ALL      every row holding ETH
    ETH-only rows with num_tokens = 1 and ETH (never eligible)
    ELIGIBLE rows with num_tokens >= 2 holding ETH
— the following:

1. Quantiles of the ETH position in USD and in native ETH units.
2. Histograms (USD buckets and ETH-unit buckets).
3. ETH-share-of-portfolio-value buckets (ELIGIBLE only).
4. Gas-threshold impact table: for candidate thresholds ($5/10/20/50/100),
   how many ELIGIBLE rows have an ETH position below it (rows a robustness
   slice would reclassify/exclude).
5. Optional per-block CSV (counts + p10/p50/p90 of eligible ETH USD),
   labeled with dates when --block-to-date-csv is given.

Single DuckDB scans, nothing big in Python — safe on a ~100 GB node.

CLI
---
  $ DUCKDB_THREADS=16 DUCKDB_MEMORY_LIMIT=60GB \
    python 4_4_eth_wealth_distribution.py \
        --holdings '/path/to/recon-eth-erc20/recon-eth-erc20-*.parquet' \
        --block-to-date-csv <full mapping csv> --out-csv eth_dist_per_block.csv
  # --holdings also accepts the 4_2 partitioned ROOT directory (hive layout).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import duckdb
import pandas as pd

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.log_utils import log  # noqa: E402

ETH = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"

QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

USD_BUCKETS = [("<$1", "eth_usd < 1"), ("$1-5", "eth_usd >= 1 AND eth_usd < 5"),
               ("$5-10", "eth_usd >= 5 AND eth_usd < 10"),
               ("$10-20", "eth_usd >= 10 AND eth_usd < 20"),
               ("$20-50", "eth_usd >= 20 AND eth_usd < 50"),
               ("$50-100", "eth_usd >= 50 AND eth_usd < 100"),
               ("$100-1k", "eth_usd >= 100 AND eth_usd < 1000"),
               ("$1k-10k", "eth_usd >= 1000 AND eth_usd < 10000"),
               (">$10k", "eth_usd >= 10000")]

BAL_BUCKETS = [("<0.001", "eth_bal < 0.001"),
               ("0.001-0.01", "eth_bal >= 0.001 AND eth_bal < 0.01"),
               ("0.01-0.1", "eth_bal >= 0.01 AND eth_bal < 0.1"),
               ("0.1-1", "eth_bal >= 0.1 AND eth_bal < 1"),
               ("1-10", "eth_bal >= 1 AND eth_bal < 10"),
               (">10", "eth_bal >= 10")]

SHARE_BUCKETS = [("<5%", "sh < 0.05"), ("5-10%", "sh >= 0.05 AND sh < 0.10"),
                 ("10-25%", "sh >= 0.10 AND sh < 0.25"),
                 ("25-50%", "sh >= 0.25 AND sh < 0.50"),
                 ("50-75%", "sh >= 0.50 AND sh < 0.75"),
                 ("75-95%", "sh >= 0.75 AND sh < 0.95"),
                 (">=95%", "sh >= 0.95")]

THRESHOLDS = [5, 10, 20, 50, 100]

POPS = {"ALL": "TRUE",
        "ETH_only": "num_tokens = 1",
        "ELIGIBLE": "num_tokens >= 2"}


def holdings_scan_sql(holdings_path: str) -> str:
    p = pathlib.Path(holdings_path)
    if p.is_dir():
        return f"read_parquet('{holdings_path}/*/*.parquet', hive_partitioning=1)"
    return f"read_parquet('{holdings_path}')"


def rows_cte(holdings_path: str) -> str:
    """Rows holding ETH, with the ETH position extracted."""
    return f"""
        SELECT CAST(block_number AS BIGINT) AS block_number,
               num_tokens,
               CAST(total_value_usd AS DOUBLE) AS tv,
               CAST(e[1].balance AS DOUBLE)    AS eth_bal,
               CAST(e[1].value_usd AS DOUBLE)  AS eth_usd
        FROM (
          SELECT block_number, num_tokens, total_value_usd,
                 list_filter(holdings, x -> lower(x.token_address) = '{ETH}') AS e
          FROM {holdings_scan_sql(holdings_path)}
        )
        WHERE len(e) > 0
    """


def q(con, sql: str, label: str):
    t0 = time.perf_counter()
    df = con.execute(sql).fetchdf()
    log(f"[scan] {label}: {time.perf_counter() - t0:.0f}s")
    return df


def fmt(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.5f}"


def main() -> int:
    p = argparse.ArgumentParser(description="ETH holdings distribution")
    p.add_argument("--holdings", required=True,
                   help="holdings parquet glob OR the 4_2 partitioned root dir")
    p.add_argument("--block-to-date-csv", default=None)
    p.add_argument("--out-csv", default=None, help="per-block stats CSV")
    p.add_argument("--threads", type=int, default=int(os.getenv("DUCKDB_THREADS", "16")))
    p.add_argument("--memory", default=os.getenv("DUCKDB_MEMORY_LIMIT", "60GB"))
    p.add_argument("--tmp-dir", default=None)
    args = p.parse_args()

    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory}'")
    tmp = args.tmp_dir or str(pathlib.Path(".").resolve() / "duckdb_tmp_4_4")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order=false")
    log(f"[duckdb] threads={args.threads} memory={args.memory} tmp={tmp}")

    # ---- one scan: quantiles + histograms + thresholds + share buckets ----
    parts = []
    for pop, cond in POPS.items():
        parts.append(f"count(*) FILTER (WHERE {cond}) AS n_{pop}")
        parts.append(f"approx_quantile(eth_usd, {QS}) FILTER (WHERE {cond}) AS q_usd_{pop}")
        parts.append(f"approx_quantile(eth_bal, {QS}) FILTER (WHERE {cond}) AS q_bal_{pop}")
        for i, (_, bcond) in enumerate(USD_BUCKETS):
            parts.append(f"count(*) FILTER (WHERE {cond} AND {bcond}) AS usd{i}_{pop}")
        for i, (_, bcond) in enumerate(BAL_BUCKETS):
            parts.append(f"count(*) FILTER (WHERE {cond} AND {bcond}) AS bal{i}_{pop}")
    for t in THRESHOLDS:
        parts.append(f"count(*) FILTER (WHERE num_tokens >= 2 AND eth_usd < {t}) "
                     f"AS thr_{t}")
    for i, (_, scond) in enumerate(SHARE_BUCKETS):
        parts.append(
            f"count(*) FILTER (WHERE num_tokens >= 2 AND tv > 0 AND "
            f"{scond.replace('sh', '(eth_usd / tv)')}) AS share{i}")

    stats = q(con, f"WITH r AS ({rows_cte(args.holdings)}) "
                   f"SELECT {', '.join(parts)} FROM r",
              "ETH distribution scan").iloc[0]

    log(f"\n{'=' * 74}")
    log("ETH POSITION DISTRIBUTION (wallet-block observations holding ETH)")
    log("=" * 74)
    for pop in POPS:
        log(f"{pop:>9}: {int(stats[f'n_{pop}']):,} rows")

    log(f"\n--- quantiles of ETH position, USD ---")
    header = " | ".join(f"{f'p{int(x * 100)}':>9}" for x in QS)
    log(f"{'pop':>9} | {header}")
    for pop in POPS:
        vals = " | ".join(f"{fmt(v):>9}" for v in stats[f"q_usd_{pop}"])
        log(f"{pop:>9} | {vals}")

    log(f"\n--- quantiles of ETH position, native ETH units ---")
    log(f"{'pop':>9} | {header}")
    for pop in POPS:
        vals = " | ".join(f"{fmt(v):>9}" for v in stats[f"q_bal_{pop}"])
        log(f"{pop:>9} | {vals}")

    log(f"\n--- ETH position histogram, USD (share within population) ---")
    log(f"{'bucket':>12} | " + " | ".join(f"{p:>21}" for p in POPS))
    for i, (name, _) in enumerate(USD_BUCKETS):
        cells = []
        for pop in POPS:
            c, n = int(stats[f"usd{i}_{pop}"]), int(stats[f"n_{pop}"])
            cells.append(f"{c:>13,} {100.0 * c / n if n else 0:>5.1f}%")
        log(f"{name:>12} | " + " | ".join(cells))

    log(f"\n--- ETH position histogram, native units ---")
    log(f"{'bucket':>12} | " + " | ".join(f"{p:>21}" for p in POPS))
    for i, (name, _) in enumerate(BAL_BUCKETS):
        cells = []
        for pop in POPS:
            c, n = int(stats[f"bal{i}_{pop}"]), int(stats[f"n_{pop}"])
            cells.append(f"{c:>13,} {100.0 * c / n if n else 0:>5.1f}%")
        log(f"{name:>12} | " + " | ".join(cells))

    n_el = int(stats["n_ELIGIBLE"])
    log(f"\n--- ETH share of portfolio value (ELIGIBLE rows holding ETH) ---")
    for i, (name, _) in enumerate(SHARE_BUCKETS):
        c = int(stats[f"share{i}"])
        log(f"{name:>8} : {c:>13,}  ({100.0 * c / n_el if n_el else 0:.1f}%)")

    log(f"\n--- gas-threshold impact on ELIGIBLE rows holding ETH ---")
    log("(rows whose ETH position a robustness slice would reclassify as gas)")
    for t in THRESHOLDS:
        c = int(stats[f"thr_{t}"])
        log(f"ETH position < ${t:>3} : {c:>13,}  "
            f"({100.0 * c / n_el if n_el else 0:.1f}% of eligible ETH holders)")

    # ---- per-block CSV ----
    if args.out_csv:
        per_block = q(con, f"""
            WITH r AS ({rows_cte(args.holdings)})
            SELECT block_number,
                   count(*)                                       AS eth_holder_rows,
                   count(*) FILTER (WHERE num_tokens >= 2)        AS eligible_eth_rows,
                   approx_quantile(eth_usd, 0.10) FILTER (WHERE num_tokens >= 2) AS elig_eth_usd_p10,
                   approx_quantile(eth_usd, 0.50) FILTER (WHERE num_tokens >= 2) AS elig_eth_usd_p50,
                   approx_quantile(eth_usd, 0.90) FILTER (WHERE num_tokens >= 2) AS elig_eth_usd_p90,
                   approx_quantile(eth_bal, 0.50) FILTER (WHERE num_tokens >= 2) AS elig_eth_bal_p50,
                   count(*) FILTER (WHERE num_tokens >= 2 AND eth_usd < 20)      AS elig_eth_under_20usd
            FROM r GROUP BY 1 ORDER BY 1
        """, "per-block scan")
        if args.block_to_date_csv and pathlib.Path(args.block_to_date_csv).exists():
            m = pd.read_csv(args.block_to_date_csv)
            bcol = next((c for c in ("block_number", "block") if c in m.columns), None)
            dcol = next((c for c in ("date", "bound_date", "block_time_utc")
                         if c in m.columns), None)
            if bcol and dcol:
                m = m[[bcol, dcol]].rename(columns={bcol: "block_number", dcol: "date"})
                m["date"] = m["date"].astype(str).str[:10]
                per_block = per_block.merge(m, on="block_number", how="left")
        per_block.to_csv(args.out_csv, index=False)
        log(f"\n[wrote] {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
