#!/usr/bin/env python3
"""
top_pct_ca_eoa_breakdown.py

Description:
    For each Ethereum block snapshot, identify the top-X% of wallets by
    total_value_usd and report how that concentration splits between
    contract accounts (CA) and externally-owned accounts (EOA). X is
    set via --top-pct (default 1). Emits a per-block, per-type
    breakdown CSV, a per-block summary CSV, and a parquet with
    (block, address) for every wallet that lands in the top-X% at any
    snapshot. Output CSV names include the percentile, e.g.
    top_1_pct_ca_eoa_breakdown.csv for --top-pct 1.

Input:
    - Parquet with block_number, wallet_address, total_value_usd
      (--parquet).
    - TSV with contract-account addresses as the first whitespace token
      of each line (--ca-tsv).

Output (in --out-dir):
    - top_<pct>_pct_ca_eoa_breakdown.csv  (per block, per wallet_type)
    - top_<pct>_pct_summary.csv           (per block, overall)
    - <--top-addresses>.parquet           (block, address) tuples

Usage:
    python top_pct_ca_eoa_breakdown.py \\
        --parquet /path/recon.parquet \\
        --ca-tsv  /path/contract_addresses.tsv \\
        --out-dir ./analytics_out \\
        --top-addresses top_addresses.parquet \\
        --top-pct 1 \\
        --threads 32 --mem 100GB
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set

import duckdb
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _format_pct_tag(pct: float) -> str:
    """Format a percentile into a filename-safe tag, e.g. 1 -> '1', 0.5 -> '0_5'."""
    if float(pct).is_integer():
        return str(int(pct))
    return str(pct).replace(".", "_")


def normalize_address(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", "ignore")
        except Exception:
            return ""
    return str(x).strip().lower()


def load_contract_addresses(tsv_path: str | Path) -> Set[str]:
    ca: Set[str] = set()
    with Path(tsv_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            addr = normalize_address(line.split("\t")[0])
            if addr:
                ca.add(addr)
    return ca


def get_duck(threads: int, mem: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads = ?", [int(threads)])
    con.execute("PRAGMA memory_limit = ?", [str(mem)])
    return con


def get_blocks(con: duckdb.DuckDBPyConnection, parquet: str) -> List[int]:
    rows = con.execute(
        "SELECT DISTINCT block_number::BIGINT AS b FROM read_parquet(?) ORDER BY b",
        [parquet],
    ).fetchall()
    return [int(r[0]) for r in rows]


def append_csv(df: pd.DataFrame, path: Path, sort_cols: List[str]) -> None:
    if df.empty:
        return
    df = df.sort_values(sort_cols)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def value_stats(s: pd.Series) -> dict:
    if s.empty:
        return dict(mean_value=None, median_value=None, p75_value=None,
                    p90_value=None, p99_value=None, max_value=None)
    s = s.astype("float64")
    q = s.quantile([0.5, 0.75, 0.90, 0.99], interpolation="linear")
    return dict(
        mean_value=float(s.mean()),
        median_value=float(q.loc[0.50]),
        p75_value=float(q.loc[0.75]),
        p90_value=float(q.loc[0.90]),
        p99_value=float(q.loc[0.99]),
        max_value=float(s.max()),
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Top-X%% holder CA/EOA breakdown.")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--ca-tsv", required=True)
    ap.add_argument("--out-dir", default="analytics_out")
    ap.add_argument("--top-addresses", required=True,
                    help="Output parquet path for (block, address) of top-X%% wallets.")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--mem", type=str, default="100GB")
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--blocks", nargs="*", type=int, default=None)
    ap.add_argument("--top-pct", type=float, default=1.0,
                    help="Percentile cutoff (default 1 = top 1%%).")
    args = ap.parse_args()

    parquet = str(Path(args.parquet).resolve())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pct_tag = _format_pct_tag(args.top_pct)
    out_breakdown = out_dir / f"top_{pct_tag}_pct_ca_eoa_breakdown.csv"
    out_summary   = out_dir / f"top_{pct_tag}_pct_summary.csv"
    out_addr      = Path(args.top_addresses)
    out_breakdown.unlink(missing_ok=True)
    out_summary.unlink(missing_ok=True)
    out_addr.unlink(missing_ok=True)

    ca_set = load_contract_addresses(args.ca_tsv)
    print(f"[info] loaded CA addresses: {len(ca_set):,}")

    con = get_duck(args.threads, args.mem)

    if args.blocks:
        blocks = [int(b) for b in args.blocks]
    else:
        blocks = get_blocks(con, parquet)
        if args.max_blocks is not None:
            blocks = blocks[: args.max_blocks]

    print(f"[info] blocks: {len(blocks):,}  top_pct: {args.top_pct}%")

    q = """
    SELECT
      wallet_address,
      COALESCE(TRY_CAST(total_value_usd AS DOUBLE), 0.0) AS total_value_usd
    FROM read_parquet(?)
    WHERE block_number = ?
    """

    cutoff_frac = 1.0 - args.top_pct / 100.0

    addr_chunks: List[pd.DataFrame] = []

    for i, block in enumerate(blocks, start=1):
        print(f"\n==== block {block} ({i}/{len(blocks)}) ====")

        df = con.execute(q, [parquet, int(block)]).df()
        if df.empty:
            print("[warn] no rows")
            continue

        df["wallet_norm"] = df["wallet_address"].map(normalize_address)
        df["wallet_type"] = df["wallet_norm"].apply(
            lambda a: "CA" if a in ca_set else "EOA"
        )

        value_all = float(df["total_value_usd"].sum())
        n_all = len(df)

        threshold = float(np.quantile(df["total_value_usd"].values, cutoff_frac))
        top = df[df["total_value_usd"] >= threshold].copy()
        value_top = float(top["total_value_usd"].sum())
        n_top = len(top)

        addr_chunks.append(
            top[["wallet_norm"]].rename(columns={"wallet_norm": "address"}).assign(block=int(block))
        )

        append_csv(
            pd.DataFrame([{
                "block": int(block),
                "n_wallets_all": n_all,
                "n_wallets_top1": n_top,
                "value_all": round(value_all, 2),
                "value_top1": round(value_top, 2),
                "value_pct_top1": round(100.0 * value_top / value_all, 4) if value_all > 0 else 0.0,
                "threshold_usd": round(threshold, 2),
            }]),
            out_summary,
            sort_cols=["block"],
        )

        rows = []
        for wt, sub in top.groupby("wallet_type", sort=False):
            n_wt = len(sub)
            v_wt = float(sub["total_value_usd"].sum())
            rows.append({
                "block": int(block),
                "wallet_type": wt,
                "n_wallets": n_wt,
                "n_wallets_pct_of_top1": round(100.0 * n_wt / n_top, 4) if n_top else 0.0,
                "total_value_usd": round(v_wt, 2),
                "value_pct_of_top1": round(100.0 * v_wt / value_top, 4) if value_top else 0.0,
                "value_pct_of_all": round(100.0 * v_wt / value_all, 4) if value_all else 0.0,
                **value_stats(sub["total_value_usd"]),
            })

        append_csv(
            pd.DataFrame(rows),
            out_breakdown,
            sort_cols=["block", "wallet_type"],
        )

        print(f"[info] n_all={n_all:,}  n_top1={n_top:,}  threshold=${threshold:,.2f}"
              f"  top1_value_share={100*value_top/value_all:.2f}%")
        for r in rows:
            print(f"       {r['wallet_type']}: n={r['n_wallets']:,}"
                  f"  value_pct_of_top1={r['value_pct_of_top1']:.2f}%"
                  f"  value_pct_of_all={r['value_pct_of_all']:.2f}%")

    if addr_chunks:
        addr_df = pd.concat(addr_chunks, ignore_index=True)[["block", "address"]]
        addr_df["block"] = addr_df["block"].astype("int64")
        addr_df.to_parquet(out_addr, index=False, engine="pyarrow")
        print(f"\n[info] wrote {len(addr_df):,} address rows to {out_addr.resolve()}")
    else:
        print("\n[warn] no address rows to write")

    print(f"\n[done]\n  - {out_breakdown.resolve()}\n  - {out_summary.resolve()}\n  - {out_addr.resolve()}")


if __name__ == "__main__":
    main()