#!/usr/bin/env python3
"""
compute_top_value_concentration_per_block.py

Description:
    For each Ethereum block snapshot, compute the share of total USD
    wealth held by the Top 1%, Top 5% and Top 10% wallets, reported
    separately for wallet_type in {ALL, EOA, CA}. Only wallets with
    total_value_usd above a configurable threshold (--min-value-usd,
    default 1 USD) are counted, and `wallets_considered` records how
    many such wallets existed per snapshot/type.

    The contract-address list is loaded in Python with a permissive
    parser (ADDR_RE) to avoid DuckDB CSV/TSV sniffing issues on large
    or irregular TSVs.

Input:
    - Parquet with block_number, wallet_address, total_value_usd
      (--parquet).
    - TSV with contract-account addresses; the first 0x-hex token of
      each line is extracted (--ca-tsv).

Output:
    CSV with columns:
        block_number, wallet_type, wallets_considered, total_value_usd,
        top_1_value_usd, top_5_value_usd, top_10_value_usd,
        top_1_share_pct, top_5_share_pct, top_10_share_pct
    (--out-csv).

Usage:
    python compute_top_value_concentration_per_block.py \\
        --parquet /path/recon.parquet \\
        --ca-tsv  /path/contract_addresses.tsv \\
        --out-csv top_value_concentration_per_block.csv \\
        --min-value-usd 1 \\
        --threads 8 --mem 12GB
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

import duckdb
import numpy as np
import pandas as pd


ADDR_RE = re.compile(r"0x[a-fA-F0-9]{40}")


def normalize_address(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", "ignore")
        except Exception:
            return ""
    s = str(x).strip().lower()
    if not s:
        return ""
    if not s.startswith("0x"):
        s = "0x" + s
    return s


def load_contract_addresses_robust(path: str | Path) -> Set[str]:
    """Load CA addresses from an arbitrary 'tsv-ish' file.

    For each line: first try to pull a 0x{40 hex} substring out of the
    line; if none is found, fall back to the first whitespace-delimited
    token. Addresses are normalised to lowercase with an 0x prefix.
    """
    path = Path(path)
    ca: Set[str] = set()

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            m = ADDR_RE.search(line)
            if m:
                addr = normalize_address(m.group(0))
                if addr:
                    ca.add(addr)
                continue

            # Fallback: first token split by tab then whitespace.
            token = line.split("\t", 1)[0].strip()
            if not token:
                continue
            token = token.split(None, 1)[0].strip()
            addr = normalize_address(token)
            if len(addr) == 42 and addr.startswith("0x"):
                ca.add(addr)

    return ca


def get_duck(*, threads: int, mem: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads = ?", [int(threads)])
    con.execute("PRAGMA memory_limit = ?", [str(mem)])
    return con


def discover_blocks(con: duckdb.DuckDBPyConnection, parquet_path: str) -> List[int]:
    rows = con.execute(
        "SELECT DISTINCT block_number FROM read_parquet(?) ORDER BY block_number",
        [parquet_path],
    ).fetchall()
    return [int(r[0]) for r in rows]


def top_k_value_sum(sorted_desc_vals: np.ndarray, k: int) -> float:
    """Sum the first k entries of an already-descending-sorted array."""
    if sorted_desc_vals.size == 0 or k <= 0:
        return 0.0
    k = min(k, sorted_desc_vals.size)
    return float(sorted_desc_vals[:k].sum())


def compute_top_sums(vals: np.ndarray, pct: float) -> Dict[str, float]:
    """Return {"k", "top_sum"} for the top `pct` fraction of `vals`.

    `vals` is a 1D array of positive floats (the caller has already
    filtered to above `--min-value-usd`).
    """
    n = int(vals.size)
    if n == 0:
        return {"k": 0, "top_sum": 0.0}

    s = np.sort(vals)[::-1]
    k = max(1, int(np.ceil(pct * n)))
    return {"k": k, "top_sum": top_k_value_sum(s, k)}


def compute_block(
    con: duckdb.DuckDBPyConnection,
    *,
    parquet_path: str,
    block_number: int,
    ca_set: Set[str],
    min_value_usd: float,
) -> pd.DataFrame:
    """Return top-1/5/10% concentration stats for one block.

    For the given block, loads (wallet_address, total_value_usd) above
    `min_value_usd`, classifies each wallet as CA (if in `ca_set`) or
    EOA, and computes absolute top-k sums plus share-of-total for ALL,
    EOA, and CA.
    """
    q = """
    SELECT
      wallet_address,
      TRY_CAST(total_value_usd AS DOUBLE) AS v
    FROM read_parquet(?)
    WHERE block_number = ?
      AND TRY_CAST(total_value_usd AS DOUBLE) > ?
    """
    df = con.execute(q, [parquet_path, int(block_number), float(min_value_usd)]).df()

    if df.empty:
        return pd.DataFrame(
            [
                {
                    "block_number": int(block_number),
                    "wallet_type": wt,
                    "wallets_considered": 0,
                    "total_value_usd": 0.0,
                    "top_1_value_usd": 0.0,
                    "top_5_value_usd": 0.0,
                    "top_10_value_usd": 0.0,
                    "top_1_share_pct": 0.0,
                    "top_5_share_pct": 0.0,
                    "top_10_share_pct": 0.0,
                }
                for wt in ["ALL", "EOA", "CA"]
            ]
        )

    df["wa_norm"] = df["wallet_address"].map(normalize_address)
    df["is_ca"] = df["wa_norm"].isin(ca_set)
    df["v"] = df["v"].astype(float)

    out_rows: List[dict] = []

    def compute_for(mask: np.ndarray, wallet_type: str) -> None:
        sub = df.loc[mask, "v"].to_numpy(dtype=np.float64)
        n = int(sub.size)
        total = float(sub.sum()) if n else 0.0

        t1 = compute_top_sums(sub, 0.01)
        t5 = compute_top_sums(sub, 0.05)
        t10 = compute_top_sums(sub, 0.10)

        top1 = float(t1["top_sum"])
        top5 = float(t5["top_sum"])
        top10 = float(t10["top_sum"])

        denom = total if total > 0 else 1e-12

        out_rows.append(
            {
                "block_number": int(block_number),
                "wallet_type": wallet_type,
                "wallets_considered": n,
                "total_value_usd": total,
                "top_1_value_usd": top1,
                "top_5_value_usd": top5,
                "top_10_value_usd": top10,
                "top_1_share_pct": 100.0 * top1 / denom,
                "top_5_share_pct": 100.0 * top5 / denom,
                "top_10_share_pct": 100.0 * top10 / denom,
            }
        )

    compute_for(np.ones(len(df), dtype=bool), "ALL")
    compute_for((~df["is_ca"]).to_numpy(dtype=bool), "EOA")
    compute_for(df["is_ca"].to_numpy(dtype=bool), "CA")

    res = pd.DataFrame(out_rows)
    order = {"ALL": 0, "EOA": 1, "CA": 2}
    res["__o"] = res["wallet_type"].map(order).astype(int)
    res = res.sort_values("__o").drop(columns="__o")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--ca-tsv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mem", type=str, default="12GB")
    ap.add_argument("--min-value-usd", type=float, default=1.0)
    ap.add_argument("--blocks", type=int, nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    parquet_path = str(Path(args.parquet).resolve())
    ca_path = str(Path(args.ca_tsv).resolve())
    out_csv = Path(args.out_csv).resolve()

    if args.overwrite and out_csv.exists():
        out_csv.unlink()

    print(f"[info] parquet: {parquet_path}")
    print(f"[info] ca-tsv : {ca_path}")
    print(f"[info] filter : total_value_usd > {args.min_value_usd}")
    print(f"[info] out    : {out_csv}")

    ca_set = load_contract_addresses_robust(ca_path)
    print(f"[info] loaded CA addresses: {len(ca_set):,}")

    con = get_duck(threads=args.threads, mem=args.mem)

    if args.blocks and len(args.blocks) > 0:
        blocks = [int(b) for b in args.blocks]
    else:
        blocks = discover_blocks(con, parquet_path)

    print(f"[info] blocks: {len(blocks)}")

    for i, blk in enumerate(blocks, start=1):
        print(f"\n==================== Block {blk} ({i}/{len(blocks)}) ====================")
        df_blk = compute_block(
            con,
            parquet_path=parquet_path,
            block_number=int(blk),
            ca_set=ca_set,
            min_value_usd=float(args.min_value_usd),
        )

        for _, r in df_blk.iterrows():
            print(
                f"  {r['wallet_type']:>3} | wallets>{args.min_value_usd:g}: {int(r['wallets_considered']):,} "
                f"| total=${r['total_value_usd']:,.2f} "
                f"| top1=${r['top_1_value_usd']:,.2f} ({r['top_1_share_pct']:.2f}%) "
                f"| top5=${r['top_5_value_usd']:,.2f} ({r['top_5_share_pct']:.2f}%) "
                f"| top10=${r['top_10_value_usd']:,.2f} ({r['top_10_share_pct']:.2f}%)"
            )

        header = not out_csv.exists()
        df_blk.to_csv(out_csv, mode="a", header=header, index=False)

    con.close()
    print(f"\n[done] wrote: {out_csv}")


if __name__ == "__main__":
    main()
