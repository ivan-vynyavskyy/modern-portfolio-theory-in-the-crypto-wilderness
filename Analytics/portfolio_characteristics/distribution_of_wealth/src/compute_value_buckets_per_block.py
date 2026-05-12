#!/usr/bin/env python3
"""
compute_value_buckets_per_block.py

Description:
    For each Ethereum block snapshot, count wallets and sum USD value held,
    bucketed by total_value_usd ranges (0-1, 1-100, 100-1K, 1K-10K,
    10K-100K, >100K) and split by wallet type (EOA, CA, ALL = EOA + CA).
    Results are appended per block, so the script is restart-safe on very
    large parquet inputs.

Input:
    - Parquet file with columns: block_number, wallet_address,
      total_value_usd (passed via --parquet).
    - TSV with contract-account addresses; first column named 'addr'
      (passed via --ca-tsv).

Output:
    CSV with columns:
        block, wallet_type, bucket, wallet_count, wallet_pct,
        value_sum_usd, value_pct
    (default path: value_buckets_minimal.csv; override with --out).

Usage:
    python compute_value_buckets_per_block.py \\
        --parquet /path/to/recon.parquet \\
        --ca-tsv  /path/to/contract_addresses.tsv \\
        --out     value_buckets_minimal.csv \\
        --threads 16 --mem 24GB --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import duckdb
import pandas as pd


DEFAULT_BLOCKS = [
    16308190, 16530248, 16730072, 16950603, 17162287, 17382266, 17595510, 17816434,
    18037988, 18251965, 18473543, 18687851, 18908895, 19129889, 19336607, 19557289,
    19771560, 19993250, 20207949, 20429973, 20651994, 20866919, 21089069, 21303934,
    21525891, 21747950, 21948292, 22170335, 22385294, 22606143, 22820674, 23042514,
    23264566, 23479244, 23700767, 23914921
]


BUCKET_CASE_SQL = """
CASE
  WHEN v <    1      THEN '0–1'
  WHEN v <  100      THEN '1–100'
  WHEN v < 1000      THEN '100–1K'
  WHEN v < 10000     THEN '1K–10K'
  WHEN v < 100000    THEN '10K–100K'
  ELSE '>100K'
END
"""


def connect_duckdb(threads: int, mem: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads = ?", [threads])
    con.execute("PRAGMA memory_limit = ?", [mem])
    con.execute("PRAGMA enable_progress_bar=true;")
    con.execute("PRAGMA preserve_insertion_order=false;")
    return con


def load_ca_table(con: duckdb.DuckDBPyConnection, ca_tsv_path: str) -> None:
    """Load contract-account addresses from TSV into a temp `ca` table."""
    ca_path_sql = ca_tsv_path.replace("'", "''")

    con.execute("""
    CREATE TEMP TABLE ca_raw (
      addr VARCHAR,
      creation_block BIGINT,
      first_seen_block BIGINT,
      ca_effective_block BIGINT
    );
    """)

    # AUTO_DETECT=false avoids sniffing issues on very large TSVs.
    con.execute(f"""
    COPY ca_raw FROM '{ca_path_sql}'
    (
      FORMAT CSV,
      DELIMITER '\t',
      HEADER true,
      AUTO_DETECT false,
      IGNORE_ERRORS true,
      NULL_PADDING true,
      MAX_LINE_SIZE 10000000
    );
    """)

    con.execute("""
    CREATE TEMP TABLE ca AS
    SELECT DISTINCT lower(trim(addr)) AS address
    FROM ca_raw
    WHERE addr IS NOT NULL AND trim(addr) != '';
    """)

    con.execute("DROP TABLE ca_raw;")


def parse_blocks(s: str) -> List[int]:
    """Parse a comma-separated block list like '9193266,9393154,...'."""
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def analyze_one_block(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
    block: int,
) -> pd.DataFrame:
    """
    Returns a DataFrame with:
    block,wallet_type,bucket,wallet_count,wallet_pct,value_sum_usd,value_pct
    """
    q = f"""
    WITH base AS (
      SELECT
        block_number AS block,
        lower(trim(wallet_address)) AS wallet_address_norm,
        GREATEST(COALESCE(TRY_CAST(total_value_usd AS DOUBLE), 0.0), 0.0) AS v
      FROM read_parquet(?)
      WHERE block_number = ?
        AND wallet_address IS NOT NULL
    ),
    typed AS (
      SELECT
        block,
        v,
        CASE
          WHEN wallet_address_norm IN (SELECT address FROM ca) THEN 'CA'
          ELSE 'EOA'
        END AS wallet_type,
        {BUCKET_CASE_SQL} AS bucket
      FROM base
    ),
    agg_ca_eoa AS (
      SELECT
        block,
        wallet_type,
        bucket,
        COUNT(*) AS wallet_count,
        SUM(v) AS value_sum_usd
      FROM typed
      GROUP BY 1,2,3
    ),
    agg_all AS (
      SELECT
        block,
        'ALL' AS wallet_type,
        bucket,
        COUNT(*) AS wallet_count,
        SUM(v) AS value_sum_usd
      FROM typed
      GROUP BY 1,2,3
    ),
    unioned AS (
      SELECT * FROM agg_ca_eoa
      UNION ALL
      SELECT * FROM agg_all
    )
    SELECT
      block,
      wallet_type,
      bucket,
      wallet_count,
      ROUND(100.0 * wallet_count
            / NULLIF(SUM(wallet_count) OVER (PARTITION BY wallet_type), 0), 4) AS wallet_pct,
      ROUND(value_sum_usd, 6) AS value_sum_usd,
      ROUND(100.0 * value_sum_usd
            / NULLIF(SUM(value_sum_usd) OVER (PARTITION BY wallet_type), 0.0), 4) AS value_pct
    FROM unioned
    ORDER BY wallet_type, bucket;
    """
    return con.execute(q, [parquet_path, block]).df()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Input parquet (huge)")
    ap.add_argument("--ca-tsv", required=True, help="TSV with contract addresses (header + addr column)")
    ap.add_argument("--out", default="value_buckets_minimal.csv", help="Output CSV")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--mem", type=str, default="24GB")
    ap.add_argument(
        "--blocks",
        default="",
        help="Comma-separated block list. If empty, uses built-in default block list.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If output exists, skip blocks already present in output.",
    )
    args = ap.parse_args()

    parquet_path = str(Path(args.parquet).resolve())
    ca_path = str(Path(args.ca_tsv).resolve())
    out_path = Path(args.out).resolve()

    blocks = parse_blocks(args.blocks) if args.blocks.strip() else DEFAULT_BLOCKS

    # Resume logic: read already-done blocks from the existing output file.
    done_blocks = set()
    if args.resume and out_path.exists():
        try:
            prev = pd.read_csv(out_path, usecols=["block"])
            done_blocks = set(prev["block"].unique().tolist())
            print(f"[resume] found {len(done_blocks)} blocks already in {out_path}")
        except Exception as e:
            print(f"[resume] could not read existing output ({e}); recomputing all blocks")

    con = connect_duckdb(args.threads, args.mem)

    print("[info] loading CA table...")
    load_ca_table(con, ca_path)
    print("[info] CA table ready")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not out_path.exists()

    for i, block in enumerate(blocks, start=1):
        if block in done_blocks:
            print(f"[{i}/{len(blocks)}] skip block {block} (already done)")
            continue

        print(f"[{i}/{len(blocks)}] processing block {block} ...")
        df = analyze_one_block(con, parquet_path, block)

        # Append immediately so the run is checkpointed per block.
        df.to_csv(out_path, mode="a", index=False, header=write_header)
        write_header = False

        print(df.head(8).to_string(index=False))
        print(f"[ok] appended {len(df)} rows for block {block} -> {out_path}")

    print(f"\n[done] output: {out_path}")


if __name__ == "__main__":
    main()
