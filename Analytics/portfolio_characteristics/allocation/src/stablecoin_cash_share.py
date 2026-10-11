#!/usr/bin/env python3
"""
stablecoin_cash_share.py

Description:
    Stablecoins as the on-chain cash position.
    MPT's two-fund separation says investors should all hold the same risky
    portfolio and differ only in how much cash they hold. With stablecoins as
    cash, this script measures per account and snapshot

      1. the cash share, stablecoin value over portfolio value, and how it
         varies with wealth, account type and time;
      2. the composition of the risky part (everything that is not cash) as a
         function of the cash share: share of native ETH (ETH + WETH), of ETH
         derivatives (liquid-staking tokens), of BTC wrappers, the number of
         risky assets and the Herfindahl index of the risky weights. Under
         two-fund separation these do not depend on the cash share.

    Cash is defined by the curated list ../data/stablecoins.csv (written by
    build_stablecoin_list.py) and --cash-def:
        core         high-confidence USD stablecoins (USDT, USDC, DAI, ...)
        extended     every USD-pegged token, including synthetic dollars,
                     T-bill-backed dollars, yield-bearing wrappers and
                     stablecoin LP tokens
        extended_fx  extended plus non-USD fiat stablecoins (EURC, XSGD, ...)
    Output names carry the definition as suffix, so definitions can be run
    side by side.

    Each snapshot is read twice: once for the token-level aggregates (one
    unnest, grouped by token: a few thousand groups) and once for the
    per-account metrics, computed with list lambdas on the nested holdings
    column, so there is no account-level GROUP BY. Account rows stay in a
    DuckDB temp table (spilled to --temp-dir when needed); only aggregates
    reach Python. Sized for the ETH-inclusive data: up to ~90 M accounts and
    ~125 M positions per snapshot with 32 threads and 100 GB.

    Conventions as in the other Obj A scripts: ETH and WETH are one asset,
    the beacon deposit contract is excluded, wallets in --ca-tsv are CAs,
    weights are value_usd shares, positions without a price carry no weight,
    and accounts whose positions all lack a price are counted
    (n_accounts_unpriced) but not analysed.

Outputs (in --out-dir, appended per snapshot; <def> = cash definition):
    cash_share_summary_<def>.csv
        block, wallet_type, wealth_bucket (ALL rows included), n_accounts,
        n_with_cash, pct_with_cash, n_cash_only, pct_cash_only,
        mean_cash_share, p25/median/p75/p90 of the cash share among cash
        holders, value_usd, cash_value_usd, cash_share_value_weighted,
        eth_value_usd, eth_derivative_value_usd, btc_value_usd
    cash_share_hist_<def>.csv
        block, wallet_type, wealth_bucket, cash_bin, n_accounts,
        pct_accounts, value_usd, pct_value
    risky_mix_by_cash_bin_<def>.csv          (two-fund separation test)
        block, wallet_type, wealth_bucket, cash_bin (0, (0,10%], (10,50%],
        (50,90%], (90,100%), ALL), n_accounts, risky_value_usd,
        mean/median ETH share of the risky part, mean ETH-derivative share,
        mean BTC-wrapper share, mean majors share (ETH + derivatives + BTC),
        risky-value-weighted ETH and majors shares, mean/median number of
        risky assets, share with a single risky asset, mean/median HHI of
        the risky weights
    stablecoin_composition_<def>.csv
        block, token, symbol, asset_class, value_usd, pct_of_cash, positions
    block_aggregates_<def>.csv
        block, n_accounts, n_accounts_unpriced, value_usd, cash_value_usd,
        eth_value_usd, eth_derivative_value_usd, btc_value_usd and their
        shares of value_usd, n_cash_tokens_held

Usage (cluster):
    python stablecoin_cash_share.py \\
        --parquet /path/to/recon-eth-erc20-partitioned \\
        --ca-tsv  /path/to/data/contract_accounts_in_dataset.tsv \\
        --cash-def core --out-dir ../data --threads 32 --mem 100GB \\
        --temp-dir /path/to/duckdb_tmp --resume
    Optional: --blocks 9193266 23914921   --max-blocks 2   --stablecoins-csv PATH
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
    ALL, BTC_WRAPPER_TOKENS, DEFAULT_DATA_DIR, DEFAULT_STABLECOINS_CSV, ETH_DERIVATIVE_TOKENS,
    ETH_PSEUDO_TOKEN, WETH_TOKEN, accounts_sql, append_csv, bin_code_sql, get_duck,
    grouping_sets_sql, label_groups, list_blocks, load_ca_table, load_cash_tokens, log,
    order_frame, prepare_outputs, sql_str_list, wealth_code_sql,
)

CASH_BINS = [
    ("0", "{x} <= 0"),
    ("(0,1%]", "{x} <= 0.01"),
    ("(1,5%]", "{x} <= 0.05"),
    ("(5,10%]", "{x} <= 0.10"),
    ("(10,25%]", "{x} <= 0.25"),
    ("(25,50%]", "{x} <= 0.50"),
    ("(50,75%]", "{x} <= 0.75"),
    ("(75,90%]", "{x} <= 0.90"),
    ("(90,99%]", "{x} <= 0.99"),
    ("(99,100%)", "{x} < 1"),
    ("100%", "TRUE"),
]
# Coarse bins for the risky-mix comparison (accounts with a risky part only)
MIX_BINS = [
    ("0", "{x} <= 0"),
    ("(0,10%]", "{x} <= 0.10"),
    ("(10,50%]", "{x} <= 0.50"),
    ("(50,90%]", "{x} <= 0.90"),
    ("(90,100%)", "TRUE"),
]
QUANTS = [0.25, 0.5, 0.75, 0.9]


def output_paths(out_dir: Path, cash_def: str) -> dict:
    names = ["cash_share_summary", "cash_share_hist", "risky_mix_by_cash_bin",
             "stablecoin_composition", "block_aggregates"]
    return {n: out_dir / f"{n}_{cash_def}.csv" for n in names}


def token_aggregates(con, src_sql: str) -> pd.DataFrame:
    """Value and position count per token (WETH folded into ETH) at one snapshot."""
    return con.execute(f"""
        SELECT token, SUM(v) AS value_usd, COUNT(*) AS positions FROM (
          SELECT CASE WHEN t = '{WETH_TOKEN}' THEN '{ETH_PSEUDO_TOKEN}' ELSE t END AS token, v
          FROM (SELECT lower(h.token_address) AS t, h.value_usd::DOUBLE AS v
                FROM (SELECT unnest(holdings) AS h FROM ({src_sql})))
          WHERE v > 0)
        GROUP BY token
    """).df()


def build_account_table(con, src_sql: str, cash_list: str, eth_list: str, der_list: str, btc_list: str) -> None:
    """Temp table acct: one row per account with value, cash, ETH, ETH-derivative
    and BTC-wrapper value, number of risky assets and HHI of the risky weights."""
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE acct AS
    WITH src AS ({src_sql}),
    m AS (
      SELECT addr,
             list_transform(list_filter(holdings, h -> COALESCE(h.value_usd, 0.0) > 0),
                            h -> struct_pack(t := lower(h.token_address), v := h.value_usd::DOUBLE)) AS pos
      FROM src
    ),
    a AS (
      SELECT addr,
        COALESCE(list_sum(list_transform(pos, p -> p.v)), 0.0) AS V,
        COALESCE(list_sum(list_transform(list_filter(pos, p -> list_contains({cash_list}, p.t)), p -> p.v)), 0.0) AS C,
        COALESCE(list_sum(list_transform(list_filter(pos, p -> list_contains({eth_list}, p.t)), p -> p.v)), 0.0) AS E,
        COALESCE(list_sum(list_transform(list_filter(pos, p -> list_contains({der_list}, p.t)), p -> p.v)), 0.0) AS D,
        COALESCE(list_sum(list_transform(list_filter(pos, p -> list_contains({btc_list}, p.t)), p -> p.v)), 0.0) AS B,
        len(list_filter(pos, p -> NOT list_contains({cash_list}, p.t))) AS n_risky_raw,
        len(list_filter(pos, p -> list_contains({eth_list}, p.t))) AS n_eth_raw,
        COALESCE(list_sum(list_transform(
            list_filter(pos, p -> NOT list_contains({cash_list}, p.t) AND NOT list_contains({eth_list}, p.t)),
            p -> p.v * p.v)), 0.0) AS sq_other
      FROM m
    )
    SELECT addr IN (SELECT address FROM ca) AS is_ca,
           {wealth_code_sql('V')} AS wb,
           V, C, E, D, B,
           (n_risky_raw - GREATEST(n_eth_raw - 1, 0))::INTEGER AS n_risky,
           CASE WHEN V - C > 0 THEN (sq_other + E * E) / ((V - C) * (V - C)) END AS hhi_r
    FROM a
    """)


def summary_frame(con, block: int) -> pd.DataFrame:
    df = con.execute(f"""
        SELECT GROUPING(is_ca) AS g_is_ca, GROUPING(wb) AS g_wb, is_ca, wb,
               COUNT(*) AS n_accounts,
               COUNT(*) FILTER (WHERE C > 0) AS n_with_cash,
               COUNT(*) FILTER (WHERE C > 0 AND V - C <= 0) AS n_cash_only,
               AVG(C / V) AS mean_cash_share,
               quantile_cont(C / V, {QUANTS}) FILTER (WHERE C > 0) AS q,
               SUM(V) AS value_usd, SUM(C) AS cash_value_usd, SUM(E) AS eth_value_usd,
               SUM(D) AS eth_derivative_value_usd, SUM(B) AS btc_value_usd
        FROM acct WHERE V > 0
        GROUP BY CUBE (is_ca, wb)
    """).df()
    df = label_groups(df, ["is_ca", "wb"])
    q = df.pop("q").map(lambda v: list(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else [np.nan] * len(QUANTS))
    for i, name in enumerate(["p25", "median", "p75", "p90"]):
        df[f"{name}_cash_share_holders"] = q.map(lambda v: v[i])
    df["pct_with_cash"] = 100.0 * df["n_with_cash"] / df["n_accounts"]
    df["pct_cash_only"] = 100.0 * df["n_cash_only"] / df["n_accounts"]
    df["cash_share_value_weighted"] = df["cash_value_usd"] / df["value_usd"]
    df.insert(0, "block", int(block))
    cols = ["block", "wallet_type", "wealth_bucket", "n_accounts", "n_with_cash", "pct_with_cash",
            "n_cash_only", "pct_cash_only", "mean_cash_share", "p25_cash_share_holders",
            "median_cash_share_holders", "p75_cash_share_holders", "p90_cash_share_holders",
            "value_usd", "cash_value_usd", "cash_share_value_weighted", "eth_value_usd",
            "eth_derivative_value_usd", "btc_value_usd"]
    return order_frame(df[cols])


def hist_frame(con, block: int) -> pd.DataFrame:
    df = con.execute(f"""
        SELECT GROUPING(is_ca) AS g_is_ca, GROUPING(wb) AS g_wb, is_ca, wb, cb,
               COUNT(*) AS n_accounts, SUM(V) AS value_usd
        FROM (SELECT is_ca, wb, V, {bin_code_sql('C / V', CASH_BINS)} AS cb FROM acct WHERE V > 0)
        GROUP BY {grouping_sets_sql(['is_ca', 'wb'], ['cb'])}
    """).df()
    df = label_groups(df, ["is_ca", "wb"])
    df["cash_bin"] = df.pop("cb").map(lambda c: CASH_BINS[int(c)][0])
    grp = df.groupby(["wallet_type", "wealth_bucket"])
    df["pct_accounts"] = 100.0 * df["n_accounts"] / grp["n_accounts"].transform("sum")
    df["pct_value"] = 100.0 * df["value_usd"] / grp["value_usd"].transform("sum")
    df.insert(0, "block", int(block))
    df = order_frame(df, extra=[("cash_bin", [b for b, _ in CASH_BINS])])
    return df[["block", "wallet_type", "wealth_bucket", "cash_bin", "n_accounts", "pct_accounts",
               "value_usd", "pct_value"]]


def risky_mix_frame(con, block: int) -> pd.DataFrame:
    df = con.execute(f"""
        SELECT GROUPING(is_ca) AS g_is_ca, GROUPING(wb) AS g_wb, GROUPING(mb) AS g_mb, is_ca, wb, mb,
               COUNT(*) AS n_accounts,
               SUM(R) AS risky_value_usd,
               AVG(E / R) AS mean_eth_share,
               quantile_cont(E / R, 0.5) AS median_eth_share,
               AVG(D / R) AS mean_eth_derivative_share,
               AVG(B / R) AS mean_btc_share,
               AVG((E + D + B) / R) AS mean_majors_share,
               SUM(E) / SUM(R) AS eth_share_value_weighted,
               SUM(E + D + B) / SUM(R) AS majors_share_value_weighted,
               AVG(n_risky) AS mean_n_risky,
               quantile_cont(n_risky, 0.5) AS median_n_risky,
               AVG(CASE WHEN n_risky = 1 THEN 1.0 ELSE 0.0 END) AS share_single_risky,
               AVG(hhi_r) AS mean_hhi_risky,
               quantile_cont(hhi_r, 0.5) AS median_hhi_risky
        FROM (SELECT *, V - C AS R, {bin_code_sql('C / V', MIX_BINS)} AS mb
              FROM acct WHERE V > 0 AND V - C > 0)
        GROUP BY CUBE (is_ca, wb, mb)
    """).df()
    flag = df.pop("g_mb").astype(int) == 1
    df["cash_bin"] = df.pop("mb").map(lambda c: MIX_BINS[int(c)][0] if pd.notna(c) else ALL)
    df.loc[flag, "cash_bin"] = ALL
    df = label_groups(df, ["is_ca", "wb"])
    df.insert(0, "block", int(block))
    df = order_frame(df, extra=[("cash_bin", [ALL] + [b for b, _ in MIX_BINS])])
    lead = ["block", "wallet_type", "wealth_bucket", "cash_bin"]
    return df[lead + [c for c in df.columns if c not in lead]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Stablecoin cash share and risky mix (two-fund separation test).")
    ap.add_argument("--parquet", required=True, help="Reconstructed holdings: 4_0 parquet file or 4_2 hive root.")
    ap.add_argument("--ca-tsv", required=True, help="TSV with contract-account addresses (first column).")
    ap.add_argument("--stablecoins-csv", default=str(DEFAULT_STABLECOINS_CSV))
    ap.add_argument("--cash-def", choices=["core", "extended", "extended_fx"], default="core")
    ap.add_argument("--out-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--mem", default="100GB")
    ap.add_argument("--temp-dir", default=None, help="DuckDB spill directory.")
    ap.add_argument("--blocks", type=int, nargs="*", default=None)
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="Skip snapshots already present in every output.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(out_dir, args.cash_def)
    done = prepare_outputs(list(paths.values()), ["block"], args.resume)

    cash = load_cash_tokens(args.stablecoins_csv, args.cash_def)
    cash_meta = cash.set_index("contract_address")[["symbol", "asset_class"]]
    cash_set = set(cash["contract_address"])
    log(f"cash definition '{args.cash_def}': {len(cash_set)} tokens from {args.stablecoins_csv}")
    cash_list = sql_str_list(cash_set)
    eth_list = sql_str_list([ETH_PSEUDO_TOKEN, WETH_TOKEN])
    der_list = sql_str_list(ETH_DERIVATIVE_TOKENS)
    btc_list = sql_str_list(BTC_WRAPPER_TOKENS)

    con = get_duck(args.threads, args.mem, args.temp_dir)
    log(f"CA addresses: {load_ca_table(con, args.ca_tsv):,}")
    blocks = [int(b) for b in args.blocks] if args.blocks else list_blocks(con, args.parquet)
    if args.max_blocks is not None:
        blocks = blocks[: args.max_blocks]
    log(f"snapshots: {len(blocks)} | input {args.parquet}")

    for i, block in enumerate(blocks, start=1):
        if (block,) in done:
            log(f"[{i}/{len(blocks)}] block {block}: already done, skipped")
            continue
        t0 = time.time()
        src = accounts_sql(args.parquet, block)

        tok = token_aggregates(con, src)
        build_account_table(con, src, cash_list, eth_list, der_list, btc_list)
        n_acc, n_unpriced, v_acct = con.execute(
            "SELECT COUNT(*) FILTER (WHERE V > 0), COUNT(*) FILTER (WHERE V <= 0), SUM(V) FROM acct").fetchone()
        if not n_acc:
            log(f"[{i}/{len(blocks)}] block {block}: no valued accounts, skipped")
            continue

        # consistency: token-level and account-level totals must agree
        v_tok = float(tok["value_usd"].sum())
        if abs(v_tok - float(v_acct)) > 1e-6 * max(v_tok, 1.0):
            log(f"[warn] block {block}: token total {v_tok:,.0f} != account total {float(v_acct):,.0f}")

        tok["is_cash"] = tok["token"].isin(cash_set)
        cash_tok = tok[tok["is_cash"]].copy()
        cash_total = float(cash_tok["value_usd"].sum())
        comp = cash_tok.assign(
            block=int(block),
            symbol=cash_tok["token"].map(cash_meta["symbol"]),
            asset_class=cash_tok["token"].map(cash_meta["asset_class"]),
            pct_of_cash=100.0 * cash_tok["value_usd"] / cash_total if cash_total > 0 else np.nan,
        ).sort_values("value_usd", ascending=False)

        v_eth = float(tok.loc[tok["token"] == ETH_PSEUDO_TOKEN, "value_usd"].sum())
        v_der = float(tok.loc[tok["token"].isin(ETH_DERIVATIVE_TOKENS), "value_usd"].sum())
        v_btc = float(tok.loc[tok["token"].isin(BTC_WRAPPER_TOKENS), "value_usd"].sum())
        agg = pd.DataFrame([{
            "block": int(block), "n_accounts": int(n_acc), "n_accounts_unpriced": int(n_unpriced),
            "value_usd": v_tok, "cash_value_usd": cash_total, "cash_share": cash_total / v_tok,
            "eth_value_usd": v_eth, "eth_share": v_eth / v_tok,
            "eth_derivative_value_usd": v_der, "eth_derivative_share": v_der / v_tok,
            "btc_value_usd": v_btc, "btc_share": v_btc / v_tok,
            "n_cash_tokens_held": int(len(cash_tok)),
        }])

        append_csv(summary_frame(con, block), paths["cash_share_summary"])
        append_csv(hist_frame(con, block), paths["cash_share_hist"])
        append_csv(risky_mix_frame(con, block), paths["risky_mix_by_cash_bin"])
        append_csv(comp, paths["stablecoin_composition"],
                   ["block", "token", "symbol", "asset_class", "value_usd", "pct_of_cash", "positions"])
        append_csv(agg, paths["block_aggregates"])
        log(f"[{i}/{len(blocks)}] block {block}: {int(n_acc):,} accounts ({int(n_unpriced):,} unpriced), "
            f"cash share of value {cash_total / v_tok:.2%}, {time.time() - t0:.0f}s")

    con.execute("DROP TABLE IF EXISTS acct")
    log("done:\n" + "\n".join(f"  - {p.resolve()}" for p in paths.values()))


if __name__ == "__main__":
    main()
