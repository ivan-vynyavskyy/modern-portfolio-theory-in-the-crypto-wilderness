#!/usr/bin/env python3
"""
trading_vs_drift.py

Description:
    Trading versus drift. For every pair of
    consecutive snapshots t -> t1 and every account valued at t, the script
    compares the account's actual weights at t1 with the weights its old
    holdings would have after price moves alone:

        passive value   p_i = units_i(t) x price_i(t1)
        passive weights p_i / sum_j p_j
        actual weights  value_i(t1) / sum_j value_j(t1)
        turnover        1/2 sum_i |actual_i - passive_i| over the union of
                        assets: the share of the portfolio that was actively
                        reallocated. 0 = pure buy-and-hold (or pro-rata
                        deposits/withdrawals, which do not change weights),
                        1 = everything swapped into other assets.
        drift           1/2 sum_i |w_i(t) - passive_i|: how far prices alone
                        moved the weights, for comparison.

    If most accounts never trade, the distance to the frontier measures
    inertia rather than choice. Accounts are also classified by whether any
    token balance changed at all (units_unchanged) and whether any token
    other than ETH changed (units_unchanged_ex_eth; ETH moves with every gas
    payment), and by net flows (actual value at t1 against passive value).

    Caveats: a transfer is not always a trade (moving tokens to one's own
    second address or to an exchange looks like selling), airdrops look like
    buying, addresses are not people, and monthly snapshots miss round trips
    within a month. Assets without a price at t1 get no passive value and no
    actual weight (share reported as share_v0_unpriced_t1).
    Relies on the reconstruction invariant that a holdings row lists each
    token once (checked by 4_1_verify_holdings.py); ETH and WETH are merged here.

    Scale: one snapshot holds up to ~125 M positions, so the pair join is done
    out of core. Each snapshot is read once and its positions (ETH and WETH
    folded) are written to --stage-dir as parquet, split into --buckets
    address buckets taken from the last two hex digits of the address
    (deterministic, independent of the DuckDB version). The pair is then
    joined bucket by bucket (FULL OUTER JOIN on address and token, window
    totals per account), and only group-level partial sums reach Python.
    At most two staged snapshots are on disk at a time (a few GB each);
    they are deleted after use unless --keep-stage is given.

Conventions as in the other Obj A scripts: ETH and WETH are one asset, the
    beacon deposit contract is excluded, wallets in --ca-tsv are CAs,
    positions without a price carry no weight. Groups (wealth bucket, number
    of assets) are taken at t.

Outputs (in --out-dir, appended per pair):
    trading_drift_summary.csv
        block_t, block_t1, wallet_type, wealth_bucket, n_bucket (ALL rows
        included), n_accounts_t, n_exit, pct_exit, n_stay, n_measured,
        n_units_unchanged, pct_units_unchanged, n_units_unchanged_ex_eth,
        pct_units_unchanged_ex_eth, n_turnover_zero, pct_turnover_zero,
        n/pct_turnover_le_{0_1pct,1pct,5pct,10pct,25pct,50pct},
        mean_turnover, mean_drift, turnover_value_weighted, n/pct_added_any,
        n/pct_removed_any, n/pct_net_inflow_10pct, n/pct_net_outflow_10pct,
        sum_v0, share_v0_unpriced_t1
        (pct_exit of n_accounts_t; unchanged/added/removed of n_stay;
        turnover and flow shares of n_measured = stayers with passive value)
    trading_drift_hist.csv
        block_t, block_t1, wallet_type, wealth_bucket, n_bucket, metric
        (turnover | drift), bin, n_accounts, pct_accounts
    trading_drift_entries.csv
        block_t, block_t1, wallet_type, wealth_bucket (at t1), n_entries,
        value_entries_usd   (accounts without value at t, valued at t1)
    trading_drift_pairs.csv
        block_t, block_t1, n_positions_t, n_positions_t1, n_accounts_t,
        n_entries, buckets, runtime_s

Usage (cluster):
    python trading_vs_drift.py \\
        --parquet /path/to/recon-eth-erc20-partitioned \\
        --ca-tsv  /path/to/data/contract_accounts_in_dataset.tsv \\
        --out-dir ../data --stage-dir /path/to/duckdb_tmp/alloc_stage \\
        --threads 32 --mem 100GB --temp-dir /path/to/duckdb_tmp \\
        --buckets 16 --resume
    Optional: --blocks 9193266 9393154 ...  (pairs are consecutive blocks of the
              list)   --max-blocks 3   --keep-stage
"""

from __future__ import annotations

import argparse
import itertools
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _alloc_common import (  # noqa: E402
    ALL, DEFAULT_DATA_DIR, ETH_PSEUDO_TOKEN, N_LABELS, WEALTH_LABELS, WETH_TOKEN, accounts_sql,
    append_csv, bin_code_sql, get_duck, list_blocks, load_ca_table, log, nbucket_code_sql,
    order_frame, prepare_outputs, sql_quote, sql_str_list, wealth_code_sql,
)

TURNOVER_BINS = [
    ("0", "{x} <= 1e-12"),
    ("(0,0.1%]", "{x} <= 0.001"),
    ("(0.1,1%]", "{x} <= 0.01"),
    ("(1,5%]", "{x} <= 0.05"),
    ("(5,10%]", "{x} <= 0.10"),
    ("(10,25%]", "{x} <= 0.25"),
    ("(25,50%]", "{x} <= 0.50"),
    ("(50,75%]", "{x} <= 0.75"),
    ("(75,100%]", "TRUE"),
]
THRESHOLDS = [(0.001, "0_1pct"), (0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct"), (0.25, "25pct"), (0.50, "50pct")]
UNITS_TOL = 1e-9   # relative tolerance for "balance unchanged" (balances are float64 of exact decimals)
HEX_BUCKET = ("((strpos('0123456789abcdef', substr(addr, 41, 1)) - 1) * 16 "
              "+ (strpos('0123456789abcdef', substr(addr, 42, 1)) - 1))")
DIMS = ["wallet_type", "wealth_bucket", "n_bucket"]


# -----------------------------------------------------------------------------
# Staging
# -----------------------------------------------------------------------------
def stage_block(con, parquet: str, block: int, stage_root: Path, buckets: int) -> Path:
    """Write the snapshot's positions (addr, token, units u, value v; ETH and
    WETH folded, balance > 0) to stage_root/block_number=<b>/pos/bkt=<k>/ and
    the per-token price (value / units) to .../prices.parquet."""
    d = stage_root / f"block_number={int(block)}"
    marker, tag = d / "_SUCCESS", f"buckets={buckets}"
    if marker.exists() and marker.read_text().strip() == tag:
        return d
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eth = sql_str_list([ETH_PSEUDO_TOKEN, WETH_TOKEN])
    bal = "COALESCE(TRY_CAST(h.balance AS DOUBLE), 0.0)"
    sel = f"""
    WITH src AS ({accounts_sql(parquet, block)}),
    e AS (
      SELECT addr, holdings,
        COALESCE(list_sum(list_transform(
            list_filter(holdings, h -> list_contains({eth}, lower(h.token_address)) AND {bal} > 0),
            h -> {bal})), 0.0) AS eth_u,
        COALESCE(list_sum(list_transform(
            list_filter(holdings, h -> list_contains({eth}, lower(h.token_address)) AND {bal} > 0),
            h -> COALESCE(h.value_usd, 0.0)::DOUBLE)), 0.0) AS eth_v
      FROM src
    ),
    p AS (
      SELECT addr, unnest(list_concat(
          list_transform(
            list_filter(holdings, h -> NOT list_contains({eth}, lower(h.token_address)) AND {bal} > 0),
            h -> struct_pack(token := lower(h.token_address), u := {bal}, v := COALESCE(h.value_usd, 0.0)::DOUBLE)),
          list_filter([struct_pack(token := '{ETH_PSEUDO_TOKEN}', u := eth_u, v := eth_v)], x -> x.u > 0)
        )) AS x
      FROM e
    )
    SELECT (GREATEST({HEX_BUCKET}, 0) % {int(buckets)})::INTEGER AS bkt,
           addr, x.token AS token, x.u AS u, x.v AS v
    FROM p
    """
    con.execute(f"COPY ({sel}) TO {sql_quote(str(d / 'pos'))} "
                f"(FORMAT PARQUET, PARTITION_BY (bkt), COMPRESSION ZSTD)")
    pos_glob = sql_quote(str(d / "pos" / "*" / "*.parquet"))
    con.execute(f"""COPY (SELECT token, SUM(v) / SUM(u) AS price FROM read_parquet({pos_glob})
                          WHERE v > 0 AND u > 0 GROUP BY token)
                    TO {sql_quote(str(d / 'prices.parquet'))} (FORMAT PARQUET)""")
    n_pos = con.execute(f"SELECT COUNT(*) FROM read_parquet({pos_glob})").fetchone()[0]
    (d / "n_positions.txt").write_text(str(int(n_pos)))
    marker.write_text(tag)
    log(f"  staged block {block}: {int(n_pos):,} positions in {time.time() - t0:.0f}s")
    return d


def bucket_relation(d: Path, k: int) -> str:
    part = d / "pos" / f"bkt={k}"
    if part.is_dir() and any(part.glob("*.parquet")):
        return (f"SELECT addr, token, u, v FROM read_parquet("
                f"{sql_quote(str(part / '*.parquet'))}, hive_partitioning=false)")
    return "SELECT NULL::VARCHAR AS addr, NULL::VARCHAR AS token, NULL::DOUBLE AS u, NULL::DOUBLE AS v WHERE FALSE"


# -----------------------------------------------------------------------------
# Pair computation
# -----------------------------------------------------------------------------
def account_table(con, d0: Path, d1: Path, k: int) -> None:
    """Temp table acc: one row per account of bucket k with value at t (V0),
    passive value at t1 (P1), actual value at t1 (A1), turnover, drift and
    change flags."""
    tol = UNITS_TOL
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE acc AS
    WITH a AS ({bucket_relation(d0, k)}),
         b AS ({bucket_relation(d1, k)}),
    j AS (
      SELECT COALESCE(a.addr, b.addr) AS addr, COALESCE(a.token, b.token) AS token,
             COALESCE(a.u, 0.0) AS u0, COALESCE(a.v, 0.0) AS v0,
             COALESCE(b.u, 0.0) AS u1, COALESCE(b.v, 0.0) AS v1
      FROM a FULL OUTER JOIN b ON a.addr = b.addr AND a.token = b.token
    ),
    pos AS (
      SELECT j.addr, j.token, j.u0, j.u1,
             CASE WHEN j.v0 > 0 THEN j.v0 ELSE 0.0 END AS w0v,
             CASE WHEN j.v0 > 0 THEN j.u0 * COALESCE(p1.price, 0.0) ELSE 0.0 END AS pv,
             CASE WHEN j.v1 > 0 THEN j.v1 ELSE 0.0 END AS a1v
      FROM j LEFT JOIN p1 ON j.token = p1.token
    ),
    tot AS (
      SELECT *, SUM(w0v) OVER w AS V0, SUM(pv) OVER w AS P1, SUM(a1v) OVER w AS A1
      FROM pos WINDOW w AS (PARTITION BY addr)
    )
    SELECT addr,
           MAX(V0) AS V0, MAX(P1) AS P1, MAX(A1) AS A1,
           COUNT(*) FILTER (WHERE w0v > 0) AS n0,
           0.5 * SUM(CASE WHEN P1 > 0 AND A1 > 0 THEN abs(a1v / A1 - pv / P1) ELSE 0.0 END) AS turnover,
           0.5 * SUM(CASE WHEN V0 > 0 AND P1 > 0 THEN abs(w0v / V0 - pv / P1) ELSE 0.0 END) AS drift,
           COUNT(*) FILTER (WHERE u0 > 0 AND u1 <= 0) AS n_removed,
           COUNT(*) FILTER (WHERE u0 <= 0 AND u1 > 0) AS n_added,
           BOOL_AND(abs(u1 - u0) <= {tol} * GREATEST(abs(u0), abs(u1))) AS units_unchanged,
           BOOL_AND(token = '{ETH_PSEUDO_TOKEN}' OR abs(u1 - u0) <= {tol} * GREATEST(abs(u0), abs(u1)))
               AS units_unchanged_ex_eth,
           SUM(CASE WHEN w0v > 0 AND pv <= 0 THEN w0v ELSE 0.0 END) AS v0_unpriced_t1
    FROM tot GROUP BY addr
    """)


def partial_frames(con):
    """Group-level partial sums of the current bucket (codes, not labels)."""
    x = f"""
    WITH x AS (
      SELECT acc.*, addr IN (SELECT address FROM ca) AS is_ca,
             {wealth_code_sql('V0')} AS wb, {nbucket_code_sql('n0')} AS nb, {wealth_code_sql('A1')} AS wb1
      FROM acc
    )"""
    measured = "A1 > 0 AND P1 > 0"
    thr = ",\n".join(f"COUNT(*) FILTER (WHERE {measured} AND turnover <= {t}) AS n_turnover_le_{name}"
                     for t, name in THRESHOLDS)
    summ = con.execute(f"""{x}
    SELECT is_ca, wb, nb,
      COUNT(*) AS n_accounts_t,
      COUNT(*) FILTER (WHERE A1 <= 0) AS n_exit,
      COUNT(*) FILTER (WHERE A1 > 0) AS n_stay,
      COUNT(*) FILTER (WHERE {measured}) AS n_measured,
      COUNT(*) FILTER (WHERE A1 > 0 AND units_unchanged) AS n_units_unchanged,
      COUNT(*) FILTER (WHERE A1 > 0 AND units_unchanged_ex_eth) AS n_units_unchanged_ex_eth,
      COUNT(*) FILTER (WHERE {measured} AND turnover <= 1e-12) AS n_turnover_zero,
      {thr},
      COALESCE(SUM(turnover) FILTER (WHERE {measured}), 0.0) AS sum_turnover,
      COALESCE(SUM(drift) FILTER (WHERE {measured}), 0.0) AS sum_drift,
      COALESCE(SUM(V0) FILTER (WHERE {measured}), 0.0) AS sum_v0_measured,
      COALESCE(SUM(V0 * turnover) FILTER (WHERE {measured}), 0.0) AS sum_v0_x_turnover,
      COUNT(*) FILTER (WHERE A1 > 0 AND n_added > 0) AS n_added_any,
      COUNT(*) FILTER (WHERE A1 > 0 AND n_removed > 0) AS n_removed_any,
      COUNT(*) FILTER (WHERE {measured} AND A1 > 1.1 * P1) AS n_net_inflow_10pct,
      COUNT(*) FILTER (WHERE {measured} AND A1 < 0.9 * P1) AS n_net_outflow_10pct,
      SUM(V0) AS sum_v0,
      SUM(v0_unpriced_t1) AS sum_v0_unpriced_t1
    FROM x WHERE V0 > 0 GROUP BY is_ca, wb, nb
    """).df()
    hist = con.execute(f"""{x}
    SELECT is_ca, wb, nb, 'turnover' AS metric, {bin_code_sql('turnover', TURNOVER_BINS)} AS bn, COUNT(*) AS n_accounts
    FROM x WHERE V0 > 0 AND {measured} GROUP BY ALL
    UNION ALL
    SELECT is_ca, wb, nb, 'drift' AS metric, {bin_code_sql('drift', TURNOVER_BINS)} AS bn, COUNT(*) AS n_accounts
    FROM x WHERE V0 > 0 AND {measured} GROUP BY ALL
    """).df()
    ent = con.execute(f"""{x}
    SELECT is_ca, wb1 AS wb, COUNT(*) AS n_entries, SUM(A1) AS value_entries_usd
    FROM x WHERE V0 <= 0 AND A1 > 0 GROUP BY is_ca, wb1
    """).df()
    return summ, hist, ent


def to_labels(df: pd.DataFrame, n_dim: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["wallet_type"] = out.pop("is_ca").map({True: "CA", False: "EOA"})
    out["wealth_bucket"] = out.pop("wb").map(lambda c: WEALTH_LABELS[int(c)])
    if n_dim:
        out["n_bucket"] = out.pop("nb").map(lambda c: N_LABELS[int(c)])
    return out


def cube_sum(df: pd.DataFrame, dims: List[str], keys: List[str], sum_cols: List[str]) -> pd.DataFrame:
    """Sum `sum_cols` over every subset of `dims` (the missing dims become ALL)."""
    parts = []
    for r in range(len(dims), -1, -1):
        for keep in itertools.combinations(dims, r):
            by = list(keep) + keys
            if by:
                g = df.groupby(by, dropna=False)[sum_cols].sum().reset_index()
            else:
                g = df[sum_cols].sum().to_frame().T
            for d in dims:
                if d not in keep:
                    g[d] = ALL
            parts.append(g)
    return pd.concat(parts, ignore_index=True)


def finish_summary(summ: pd.DataFrame, bt: int, bt1: int) -> pd.DataFrame:
    sum_cols = [c for c in summ.columns if c not in DIMS]
    df = cube_sum(summ, DIMS, [], sum_cols)
    pct = lambda a, b: 100.0 * df[a] / df[b].where(df[b] > 0)  # noqa: E731
    df["pct_exit"] = pct("n_exit", "n_accounts_t")
    df["pct_units_unchanged"] = pct("n_units_unchanged", "n_stay")
    df["pct_units_unchanged_ex_eth"] = pct("n_units_unchanged_ex_eth", "n_stay")
    df["pct_turnover_zero"] = pct("n_turnover_zero", "n_measured")
    for _, name in THRESHOLDS:
        df[f"pct_turnover_le_{name}"] = pct(f"n_turnover_le_{name}", "n_measured")
    df["mean_turnover"] = df["sum_turnover"] / df["n_measured"].where(df["n_measured"] > 0)
    df["mean_drift"] = df["sum_drift"] / df["n_measured"].where(df["n_measured"] > 0)
    df["turnover_value_weighted"] = df["sum_v0_x_turnover"] / df["sum_v0_measured"].where(df["sum_v0_measured"] > 0)
    df["pct_added_any"] = pct("n_added_any", "n_stay")
    df["pct_removed_any"] = pct("n_removed_any", "n_stay")
    df["pct_net_inflow_10pct"] = pct("n_net_inflow_10pct", "n_measured")
    df["pct_net_outflow_10pct"] = pct("n_net_outflow_10pct", "n_measured")
    df["share_v0_unpriced_t1"] = df["sum_v0_unpriced_t1"] / df["sum_v0"].where(df["sum_v0"] > 0)
    df.insert(0, "block_t1", int(bt1))
    df.insert(0, "block_t", int(bt))
    thr_cols = [c for _, name in THRESHOLDS for c in (f"n_turnover_le_{name}", f"pct_turnover_le_{name}")]
    cols = (["block_t", "block_t1", *DIMS, "n_accounts_t", "n_exit", "pct_exit", "n_stay", "n_measured",
             "n_units_unchanged", "pct_units_unchanged", "n_units_unchanged_ex_eth", "pct_units_unchanged_ex_eth",
             "n_turnover_zero", "pct_turnover_zero", *thr_cols, "mean_turnover", "mean_drift",
             "turnover_value_weighted", "n_added_any", "pct_added_any", "n_removed_any", "pct_removed_any",
             "n_net_inflow_10pct", "pct_net_inflow_10pct", "n_net_outflow_10pct", "pct_net_outflow_10pct",
             "sum_v0", "share_v0_unpriced_t1"])
    return order_frame(df[cols])


def finish_hist(hist: pd.DataFrame, bt: int, bt1: int) -> pd.DataFrame:
    hist = hist.copy()
    hist["bin"] = hist.pop("bn").map(lambda c: TURNOVER_BINS[int(c)][0])
    df = cube_sum(hist, DIMS, ["metric", "bin"], ["n_accounts"])
    df["pct_accounts"] = 100.0 * df["n_accounts"] / df.groupby(DIMS + ["metric"])["n_accounts"].transform("sum")
    df.insert(0, "block_t1", int(bt1))
    df.insert(0, "block_t", int(bt))
    df = order_frame(df, extra=[("metric", ["turnover", "drift"]), ("bin", [b for b, _ in TURNOVER_BINS])])
    return df[["block_t", "block_t1", *DIMS, "metric", "bin", "n_accounts", "pct_accounts"]]


def finish_entries(ent: pd.DataFrame, bt: int, bt1: int) -> pd.DataFrame:
    df = cube_sum(ent, ["wallet_type", "wealth_bucket"], [], ["n_entries", "value_entries_usd"])
    df.insert(0, "block_t1", int(bt1))
    df.insert(0, "block_t", int(bt))
    return order_frame(df[["block_t", "block_t1", "wallet_type", "wealth_bucket", "n_entries", "value_entries_usd"]])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Trading versus price drift between consecutive snapshots.")
    ap.add_argument("--parquet", required=True, help="Reconstructed holdings: 4_0 parquet file or 4_2 hive root.")
    ap.add_argument("--ca-tsv", required=True, help="TSV with contract-account addresses (first column).")
    ap.add_argument("--out-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--stage-dir", default=None,
                    help="Where staged positions go (default: <temp-dir>/alloc_stage or <out-dir>/_stage).")
    ap.add_argument("--buckets", type=int, default=16, help="Address buckets per pair join (1..256).")
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--mem", default="100GB")
    ap.add_argument("--temp-dir", default=None, help="DuckDB spill directory.")
    ap.add_argument("--blocks", type=int, nargs="*", default=None)
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--keep-stage", action="store_true", help="Do not delete staged snapshots.")
    ap.add_argument("--resume", action="store_true", help="Skip pairs already present in every output.")
    args = ap.parse_args()
    if not 1 <= args.buckets <= 256:
        raise SystemExit("--buckets must be between 1 and 256")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_root = Path(args.stage_dir) if args.stage_dir else (
        Path(args.temp_dir) / "alloc_stage" if args.temp_dir else out_dir / "_stage")
    stage_root.mkdir(parents=True, exist_ok=True)
    names = ("trading_drift_summary", "trading_drift_hist", "trading_drift_entries", "trading_drift_pairs")
    paths = {n: out_dir / f"{n}.csv" for n in names}
    done = prepare_outputs(list(paths.values()), ["block_t", "block_t1"], args.resume)

    con = get_duck(args.threads, args.mem, args.temp_dir)
    log(f"CA addresses: {load_ca_table(con, args.ca_tsv):,}")
    blocks = sorted(int(b) for b in args.blocks) if args.blocks else list_blocks(con, args.parquet)
    if args.max_blocks is not None:
        blocks = blocks[: args.max_blocks]
    pairs = list(zip(blocks[:-1], blocks[1:]))
    log(f"pairs: {len(pairs)} | buckets {args.buckets} | stage {stage_root} | input {args.parquet}")

    for i, (bt, bt1) in enumerate(pairs, start=1):
        if (bt, bt1) in done:
            log(f"[{i}/{len(pairs)}] {bt} -> {bt1}: already done, skipped")
            continue
        t0 = time.time()
        log(f"[{i}/{len(pairs)}] {bt} -> {bt1}")
        d0 = stage_block(con, args.parquet, bt, stage_root, args.buckets)
        d1 = stage_block(con, args.parquet, bt1, stage_root, args.buckets)
        con.execute(f"CREATE OR REPLACE TEMP TABLE p1 AS SELECT * FROM read_parquet({sql_quote(str(d1 / 'prices.parquet'))})")

        parts_s, parts_h, parts_e = [], [], []
        for k in range(args.buckets):
            account_table(con, d0, d1, k)
            s, h, e = partial_frames(con)
            parts_s.append(s), parts_h.append(h), parts_e.append(e)
        con.execute("DROP TABLE IF EXISTS acc")

        summ = pd.concat(parts_s, ignore_index=True)
        if summ.empty:
            log(f"  no accounts valued at {bt}, pair skipped")
            continue
        summ = to_labels(summ).groupby(DIMS, as_index=False).sum(numeric_only=True)
        hist = to_labels(pd.concat(parts_h, ignore_index=True)).groupby(DIMS + ["metric", "bn"], as_index=False)["n_accounts"].sum()
        ent_raw = pd.concat(parts_e, ignore_index=True)
        ent = (to_labels(ent_raw, n_dim=False).groupby(["wallet_type", "wealth_bucket"], as_index=False)
               [["n_entries", "value_entries_usd"]].sum()) if not ent_raw.empty else \
            pd.DataFrame(columns=["wallet_type", "wealth_bucket", "n_entries", "value_entries_usd"])

        summary = finish_summary(summ, bt, bt1)
        append_csv(summary, paths["trading_drift_summary"])
        append_csv(finish_hist(hist, bt, bt1), paths["trading_drift_hist"])
        if not ent.empty:
            append_csv(finish_entries(ent, bt, bt1), paths["trading_drift_entries"])
        tot = summary[(summary["wallet_type"] == ALL) & (summary["wealth_bucket"] == ALL) & (summary["n_bucket"] == ALL)].iloc[0]
        pair_row = pd.DataFrame([{
            "block_t": bt, "block_t1": bt1,
            "n_positions_t": int((d0 / "n_positions.txt").read_text()),
            "n_positions_t1": int((d1 / "n_positions.txt").read_text()),
            "n_accounts_t": int(tot["n_accounts_t"]), "n_entries": int(ent["n_entries"].sum()) if not ent.empty else 0,
            "buckets": args.buckets, "runtime_s": round(time.time() - t0, 1),
        }])
        append_csv(pair_row, paths["trading_drift_pairs"])
        log(f"  {int(tot['n_accounts_t']):,} accounts at t | exit {tot['pct_exit']:.1f}% | "
            f"units unchanged {tot['pct_units_unchanged']:.1f}% (ex ETH {tot['pct_units_unchanged_ex_eth']:.1f}%) | "
            f"turnover 0: {tot['pct_turnover_zero']:.1f}%, mean {tot['mean_turnover']:.3f} | {time.time() - t0:.0f}s")

        if not args.keep_stage:
            shutil.rmtree(d0, ignore_errors=True)   # the next pair starts at bt1

    if not args.keep_stage:
        for d in stage_root.glob("block_number=*"):
            shutil.rmtree(d, ignore_errors=True)
    log("done:\n" + "\n".join(f"  - {p.resolve()}" for p in paths.values()))


if __name__ == "__main__":
    main()
