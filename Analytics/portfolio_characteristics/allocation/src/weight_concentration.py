#!/usr/bin/env python3
"""
weight_concentration.py

Description:
    Weight concentration of actual portfolios.
    Portfolio size N counts assets; this script measures how the value is
    spread over them. Per account and snapshot:

        max_weight     largest position weight (ETH and WETH folded)
        hhi            Herfindahl index of the weights, sum of w_i^2
        effective_n    1 / hhi, the number of equally weighted assets with
                       the same concentration
        cap binding    N >= 2 and max_weight > --cap (0.9 by default). The
                       optimisation (5_0) projects the actual weights onto the
                       capped simplex before computing the frontier anchors and
                       distances, so these are the portfolios the projection
                       changes. The weight moved by the projection (half the
                       l1 change) is exactly max_weight - cap, whatever the
                       projection rule, because only the excess over the cap
                       is redistributed.

    An account with ten tokens and nearly everything in one of them is
    effectively a single-asset portfolio: effective_n close to 1.

    Weights use the positions with value_usd above --min-position-usd
    (default 0: every priced position). With --min-position-usd 1 the script
    gives the dust-free view; wealth buckets
    always use the full portfolio value.

    Per snapshot the holdings are read once; per-account metrics are computed
    with list lambdas on the nested holdings column (no account-level
    GROUP BY) into a DuckDB temp table, and only aggregates reach Python.
    Sized for up to ~90 M accounts per snapshot with 32 threads and 100 GB.

    Note on comparability with 5_0: the optimisation also drops assets with
    fewer than 45 days of price history in the lookback window, so its
    multi-asset set is slightly smaller than N >= 2 here.

Conventions as in the other Obj A scripts: ETH and WETH are one asset, the
    beacon deposit contract is excluded, wallets in --ca-tsv are CAs,
    positions without a price carry no weight, accounts whose positions all
    lack a price are left out.

Outputs (in --out-dir, appended per snapshot; suffix _min<X>usd when
--min-position-usd > 0):
    weight_concentration_summary.csv
        block, wallet_type, wealth_bucket, n_bucket (1, 2, 3–4, 5–9, 10+, 2+ =
        all multi-asset, ALL), n_accounts, value_usd, mean_n,
        mean/p25/median/p75/p90 max_weight, mean/p25/median/p75/p90
        effective_n, mean_hhi, mean_effective_n_over_n, n_max_weight_ge_99,
        pct_max_weight_ge_99, n_cap_binding, pct_cap_binding,
        value_cap_binding, pct_value_cap_binding, mean_cap_shift_binding,
        mean_cap_shift_multi (average over all multi-asset accounts)
    max_weight_hist.csv      (multi-asset accounts only)
        block, wallet_type, wealth_bucket, n_bucket, max_weight_bin,
        n_accounts, pct_accounts, value_usd, pct_value
    effective_n_hist.csv     (multi-asset accounts only)
        block, wallet_type, wealth_bucket, n_bucket, effective_n_bin,
        n_accounts, pct_accounts, value_usd, pct_value

Usage (cluster):
    python weight_concentration.py \\
        --parquet /path/to/recon-eth-erc20-partitioned \\
        --ca-tsv  /path/to/data/contract_accounts_in_dataset.tsv \\
        --out-dir ../data --threads 32 --mem 100GB \\
        --temp-dir /path/to/duckdb_tmp --resume
    Dust-free variant:  add --min-position-usd 1
    Optional: --cap 0.9   --blocks 9193266 ...   --max-blocks 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _alloc_common import (  # noqa: E402
    DEFAULT_DATA_DIR, ETH_PSEUDO_TOKEN, WETH_TOKEN, accounts_sql, append_csv,
    bin_code_sql, get_duck, grouping_sets_sql, label_groups, list_blocks, load_ca_table, log,
    nbucket_code_sql, order_frame, prepare_outputs, sql_str_list, wealth_code_sql,
)

MAXW_BINS = [
    ("[0,0.25)", "{x} < 0.25"),
    ("[0.25,0.5)", "{x} < 0.5"),
    ("[0.5,0.6)", "{x} < 0.6"),
    ("[0.6,0.7)", "{x} < 0.7"),
    ("[0.7,0.8)", "{x} < 0.8"),
    ("[0.8,0.9]", "{x} <= 0.9"),
    ("(0.9,0.95]", "{x} <= 0.95"),
    ("(0.95,0.99]", "{x} <= 0.99"),
    ("(0.99,1]", "TRUE"),
]
EFFN_BINS = [
    ("[1,1.1)", "{x} < 1.1"),
    ("[1.1,1.5)", "{x} < 1.5"),
    ("[1.5,2)", "{x} < 2"),
    ("[2,3)", "{x} < 3"),
    ("[3,5)", "{x} < 5"),
    ("[5,10)", "{x} < 10"),
    ("10+", "TRUE"),
]
QUANTS = [0.25, 0.5, 0.75, 0.9]
MULTI = "2+"


def suffix_for(min_usd: float) -> str:
    if min_usd <= 0:
        return ""
    txt = f"{min_usd:g}".replace(".", "p")
    return f"_min{txt}usd"


def build_account_table(con, src_sql: str, min_usd: float) -> None:
    """Temp table acct: wallet type, wealth bucket (full value), number of
    weighted assets n, weighted value V, max weight mw and HHI."""
    eth = sql_str_list([ETH_PSEUDO_TOKEN, WETH_TOKEN])
    thr = float(min_usd)
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE acct AS
    WITH src AS ({src_sql}),
    m AS (
      SELECT addr,
             COALESCE(list_sum(list_transform(list_filter(holdings, h -> COALESCE(h.value_usd, 0.0) > 0),
                                              h -> h.value_usd::DOUBLE)), 0.0) AS V_all,
             list_transform(list_filter(holdings, h -> COALESCE(h.value_usd, 0.0) > {thr}),
                            h -> struct_pack(t := lower(h.token_address), v := h.value_usd::DOUBLE)) AS pos
      FROM src
    ),
    a AS (
      SELECT addr, V_all,
        COALESCE(list_sum(list_transform(pos, p -> p.v)), 0.0) AS V,
        COALESCE(list_sum(list_transform(list_filter(pos, p -> list_contains({eth}, p.t)), p -> p.v)), 0.0) AS E,
        len(pos) AS n_raw,
        len(list_filter(pos, p -> list_contains({eth}, p.t))) AS n_eth_raw,
        COALESCE(list_sum(list_transform(list_filter(pos, p -> NOT list_contains({eth}, p.t)), p -> p.v * p.v)), 0.0) AS sq_other,
        COALESCE(list_max(list_transform(list_filter(pos, p -> NOT list_contains({eth}, p.t)), p -> p.v)), 0.0) AS max_other
      FROM m
    )
    SELECT addr IN (SELECT address FROM ca) AS is_ca,
           {wealth_code_sql('V_all')} AS wb,
           (n_raw - GREATEST(n_eth_raw - 1, 0))::INTEGER AS n,
           V,
           CASE WHEN V > 0 THEN GREATEST(max_other, E) / V END AS mw,
           CASE WHEN V > 0 THEN (sq_other + E * E) / (V * V) END AS hhi
    FROM a
    """)


def _summary_select(cap: float) -> str:
    return f"""
        COUNT(*) AS n_accounts,
        SUM(V) AS value_usd,
        AVG(n) AS mean_n,
        AVG(mw) AS mean_max_weight,
        quantile_cont(mw, {QUANTS}) AS q_mw,
        AVG(1.0 / hhi) AS mean_effective_n,
        quantile_cont(1.0 / hhi, {QUANTS}) AS q_effn,
        AVG(hhi) AS mean_hhi,
        AVG((1.0 / hhi) / n) AS mean_effective_n_over_n,
        COUNT(*) FILTER (WHERE mw >= 0.99) AS n_max_weight_ge_99,
        COUNT(*) FILTER (WHERE n >= 2 AND mw > {cap}) AS n_cap_binding,
        COALESCE(SUM(V) FILTER (WHERE n >= 2 AND mw > {cap}), 0.0) AS value_cap_binding,
        COALESCE(SUM(mw - {cap}) FILTER (WHERE n >= 2 AND mw > {cap}), 0.0) AS sum_cap_shift,
        COUNT(*) FILTER (WHERE n >= 2) AS n_multi
    """


def summary_frame(con, block: int, cap: float) -> pd.DataFrame:
    base = f"SELECT *, {nbucket_code_sql('n')} AS nb FROM acct WHERE V > 0 AND n >= 1"
    by_n = con.execute(f"""
        SELECT GROUPING(is_ca) AS g_is_ca, GROUPING(wb) AS g_wb, GROUPING(nb) AS g_nb, is_ca, wb, nb,
               {_summary_select(cap)}
        FROM ({base}) GROUP BY CUBE (is_ca, wb, nb)
    """).df()
    by_n = label_groups(by_n, ["is_ca", "wb", "nb"])
    multi = con.execute(f"""
        SELECT GROUPING(is_ca) AS g_is_ca, GROUPING(wb) AS g_wb, is_ca, wb, {_summary_select(cap)}
        FROM ({base}) WHERE n >= 2 GROUP BY CUBE (is_ca, wb)
    """).df()
    multi = label_groups(multi, ["is_ca", "wb"])
    multi["n_bucket"] = MULTI
    df = pd.concat([by_n, multi], ignore_index=True)

    for col, name in (("q_mw", "max_weight"), ("q_effn", "effective_n")):
        q = df.pop(col).map(lambda v: list(v) if isinstance(v, (list, np.ndarray)) else [np.nan] * len(QUANTS))
        for i, lab in enumerate(["p25", "median", "p75", "p90"]):
            df[f"{lab}_{name}"] = q.map(lambda v: v[i])
    n = df["n_accounts"].astype(float)
    df["pct_max_weight_ge_99"] = 100.0 * df["n_max_weight_ge_99"] / n
    df["pct_cap_binding"] = 100.0 * df["n_cap_binding"] / n
    df["pct_value_cap_binding"] = 100.0 * df["value_cap_binding"] / df["value_usd"]
    df["mean_cap_shift_binding"] = df["sum_cap_shift"] / df["n_cap_binding"].replace(0, np.nan)
    df["mean_cap_shift_multi"] = df["sum_cap_shift"] / df["n_multi"].replace(0, np.nan)
    df = df.drop(columns=["sum_cap_shift", "n_multi"])
    df.insert(0, "block", int(block))
    cols = ["block", "wallet_type", "wealth_bucket", "n_bucket", "n_accounts", "value_usd", "mean_n",
            "mean_max_weight", "p25_max_weight", "median_max_weight", "p75_max_weight", "p90_max_weight",
            "mean_effective_n", "p25_effective_n", "median_effective_n", "p75_effective_n", "p90_effective_n",
            "mean_hhi", "mean_effective_n_over_n", "n_max_weight_ge_99", "pct_max_weight_ge_99",
            "n_cap_binding", "pct_cap_binding", "value_cap_binding", "pct_value_cap_binding",
            "mean_cap_shift_binding", "mean_cap_shift_multi"]
    return order_frame(df[cols])


def hist_frame(con, block: int, expr: str, bins, label: str) -> pd.DataFrame:
    df = con.execute(f"""
        SELECT GROUPING(is_ca) AS g_is_ca, GROUPING(wb) AS g_wb, GROUPING(nb) AS g_nb, is_ca, wb, nb, bn,
               COUNT(*) AS n_accounts, SUM(V) AS value_usd
        FROM (SELECT is_ca, wb, V, {nbucket_code_sql('n')} AS nb, {bin_code_sql(expr, bins)} AS bn
              FROM acct WHERE V > 0 AND n >= 2)
        GROUP BY {grouping_sets_sql(['is_ca', 'wb', 'nb'], ['bn'])}
    """).df()
    df = label_groups(df, ["is_ca", "wb", "nb"])
    df.loc[df["n_bucket"] == "ALL", "n_bucket"] = MULTI   # all multi-asset accounts
    df[label] = df.pop("bn").map(lambda c: bins[int(c)][0])
    grp = df.groupby(["wallet_type", "wealth_bucket", "n_bucket"])
    df["pct_accounts"] = 100.0 * df["n_accounts"] / grp["n_accounts"].transform("sum")
    df["pct_value"] = 100.0 * df["value_usd"] / grp["value_usd"].transform("sum")
    df.insert(0, "block", int(block))
    df = order_frame(df, extra=[(label, [b for b, _ in bins])])
    return df[["block", "wallet_type", "wealth_bucket", "n_bucket", label, "n_accounts", "pct_accounts",
               "value_usd", "pct_value"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Weight concentration (max weight, effective N, cap check).")
    ap.add_argument("--parquet", required=True, help="Reconstructed holdings: 4_0 parquet file or 4_2 hive root.")
    ap.add_argument("--ca-tsv", required=True, help="TSV with contract-account addresses (first column).")
    ap.add_argument("--out-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--cap", type=float, default=0.9, help="Single-asset weight cap of the optimisation.")
    ap.add_argument("--min-position-usd", type=float, default=0.0,
                    help="Ignore positions worth this much or less when computing weights (dust filter).")
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--mem", default="100GB")
    ap.add_argument("--temp-dir", default=None, help="DuckDB spill directory.")
    ap.add_argument("--blocks", type=int, nargs="*", default=None)
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="Skip snapshots already present in every output.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = suffix_for(args.min_position_usd)
    paths = {n: out_dir / f"{n}{sfx}.csv" for n in ("weight_concentration_summary", "max_weight_hist", "effective_n_hist")}
    done = prepare_outputs(list(paths.values()), ["block"], args.resume)

    con = get_duck(args.threads, args.mem, args.temp_dir)
    log(f"CA addresses: {load_ca_table(con, args.ca_tsv):,}")
    blocks = [int(b) for b in args.blocks] if args.blocks else list_blocks(con, args.parquet)
    if args.max_blocks is not None:
        blocks = blocks[: args.max_blocks]
    log(f"snapshots: {len(blocks)} | cap {args.cap} | min position {args.min_position_usd} USD | input {args.parquet}")

    for i, block in enumerate(blocks, start=1):
        if (block,) in done:
            log(f"[{i}/{len(blocks)}] block {block}: already done, skipped")
            continue
        t0 = time.time()
        build_account_table(con, accounts_sql(args.parquet, block), args.min_position_usd)
        n_acc, n_multi, n_bind = con.execute(f"""
            SELECT COUNT(*) FILTER (WHERE V > 0 AND n >= 1),
                   COUNT(*) FILTER (WHERE V > 0 AND n >= 2),
                   COUNT(*) FILTER (WHERE V > 0 AND n >= 2 AND mw > {args.cap}) FROM acct""").fetchone()
        if not n_acc:
            log(f"[{i}/{len(blocks)}] block {block}: no valued accounts, skipped")
            continue
        append_csv(summary_frame(con, block, args.cap), paths["weight_concentration_summary"])
        append_csv(hist_frame(con, block, "mw", MAXW_BINS, "max_weight_bin"), paths["max_weight_hist"])
        append_csv(hist_frame(con, block, "1.0 / hhi", EFFN_BINS, "effective_n_bin"), paths["effective_n_hist"])
        share = n_bind / n_multi if n_multi else float("nan")
        log(f"[{i}/{len(blocks)}] block {block}: {int(n_acc):,} accounts, {int(n_multi):,} multi-asset, "
            f"cap binding {share:.1%}, {time.time() - t0:.0f}s")

    con.execute("DROP TABLE IF EXISTS acct")
    log("done:\n" + "\n".join(f"  - {p.resolve()}" for p in paths.values()))


if __name__ == "__main__":
    main()
