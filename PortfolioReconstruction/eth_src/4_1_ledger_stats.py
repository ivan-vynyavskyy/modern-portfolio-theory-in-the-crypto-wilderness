#!/usr/bin/env python3
"""
4_1_ledger_stats.py

Purpose
-------
Explorative global statistics over the aggregated ETH address-flow ledger
(output of 2_0_eth_transfer_aggregator.py): how many unique users, how active,
who holds what, and how the headline numbers compare to public sources.

Why this is cheap here: shard files are address-hash bucketed AND sorted by
address, so every address lives in exactly ONE file -- exact unique-address
counts, per-address balances and top-K lists come from independent per-shard
aggregations merged in Python (no global dedupe over 12.9B rows).

Report tiers
------------
  A. instant   : shard files, total rows, block range (parquet metadata)
  B. headers   : blocks, transactions, txs/block, date range
                 (needs the transfer ledger's decisions.json; skipped if absent)
  C. per-shard : exact unique addresses, final-balance distribution,
                 top holders, top most-active addresses, per-era row activity
                 (one pass over each shard; ~10-20 min at 16 threads)

CLI
---
  $ DUCKDB_THREADS=16 DUCKDB_MEMORY_LIMIT=64GB \
    python 4_1_ledger_stats.py --data-dir /path/to/ledgers
  $ python 4_1_ledger_stats.py --data-dir ... --no-shard-scan   # tiers A+B only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb

sys.path.append(str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    LONDON_BLOCK,
    MERGE_BLOCK,
    SHAPELLA_BLOCK,
    partition_files,
    setup_logging,
)

ETH_PSEUDO_TOKEN = "0x" + "ee" * 20
WEI = 10**18

# label -> upper bound in wei (checked in order; final bucket is the rest)
BALANCE_BUCKETS = [
    ("0 (emptied)",        1),
    ("<0.001 ETH (dust)",  10**15),
    ("0.001 - 0.1 ETH",    10**17),
    ("0.1 - 1 ETH",        10**18),
    ("1 - 10 ETH",         10**19),
    ("10 - 100 ETH",       10**20),
    ("100 - 10k ETH",      10**22),
    (">= 10k ETH (whale)", None),
]

ERA_CASE = f"""CASE
    WHEN block < {LONDON_BLOCK} THEN 'pre-London'
    WHEN block < {MERGE_BLOCK} THEN 'London-Merge'
    WHEN block < {SHAPELLA_BLOCK} THEN 'Merge-Shapella'
    ELSE 'post-Shapella' END"""
ERA_ORDER = ["pre-London", "London-Merge", "Merge-Shapella", "post-Shapella"]

REFERENCES = [
    ("unique addresses", "etherscan.io/chart/address (~300M+); ours counts "
     "addresses that ever moved native ETH incl. gas payers"),
    ("transactions", "etherscan.io/chart/tx (cumulative)"),
    ("total EL supply", "sum of balances == minted-burned; exceeds published "
     "supply by cumulative principal withdrawals (deposit-contract double "
     "count)"),
    ("top holder", "the Beacon Deposit Contract 0x00000000219ab540... is the "
     "largest ETH holder (etherscan.io/accounts)"),
    ("most active", "expect PoW pools (Ethermine 0xea674f...) and MEV "
     "builders (beaverbuild 0x952222...) -- a fee flow per tx of every "
     "block they produced"),
]


def eth(wei: int) -> str:
    return f"{wei / WEI:,.2f} ETH"


def bucket_sql() -> str:
    branches = []
    for label, hi in BALANCE_BUCKETS:
        if hi is None:
            branches.append(f"ELSE '{label}'")
        else:
            branches.append(f"WHEN bal < {hi} THEN '{label}'")
    return "CASE " + " ".join(branches) + " END"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Global stats over the aggregated ETH address-flow ledger")
    parser.add_argument("--data-dir", required=True,
                        help="ledgers root containing token_address=<token>/")
    parser.add_argument("--token", default=ETH_PSEUDO_TOKEN)
    parser.add_argument("--decisions", default=None,
                        help="transfer ledger decisions.json for the block-"
                             "header tier (default: $OUTPUT_BASE/decisions.json)")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--no-shard-scan", action="store_true",
                        help="skip tier C (per-shard balance/address scan)")
    parser.add_argument("--threads", type=int,
                        default=int(os.getenv("DUCKDB_THREADS", "16")))
    parser.add_argument("--memory",
                        default=os.getenv("DUCKDB_MEMORY_LIMIT", "64GB"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    token = args.token.lower()
    token_dir = data_dir / f"token_address={token}"
    log = setup_logging(Path(__file__).resolve().parent / "logs",
                        "eth_ledger_stats")

    files = sorted(p for p in token_dir.glob("*.parquet") if p.is_file())
    if not files:
        log.error("no shard files under %s", token_dir)
        return 1
    glob = str(token_dir / "*.parquet")

    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory}'")

    # ---------------- tier A: instant (parquet metadata) ----------------
    n_rows, lo_blk, hi_blk = con.execute(
        f"SELECT count(*), MIN(block), MAX(block) FROM read_parquet('{glob}')"
    ).fetchone()
    log.info("=" * 74)
    log.info("ETH ADDRESS-FLOW LEDGER STATS  (%s)", token_dir)
    log.info("=" * 74)
    log.info("shard files          : %d", len(files))
    log.info("flow rows total      : %s", f"{int(n_rows):,}")
    log.info("block range          : %s - %s", f"{int(lo_blk):,}",
             f"{int(hi_blk):,}")

    # ---------------- tier B: block headers (seconds) -------------------
    n_blocks = int(hi_blk) - int(lo_blk) + 1  # fallback if no decisions.json
    dec_path = (Path(args.decisions).expanduser().resolve() if args.decisions
                else Path(os.getenv("OUTPUT_BASE",
                                    "eth_transfers"))
                / "decisions.json")
    if dec_path.exists():
        dec = json.loads(dec_path.read_text())
        b = dec["block"]
        c = b["cols"]
        bfiles = [f for p in b["partitions"] for f in partition_files(p)]
        names = b.get("schema_names", [])
        has_txc = "transaction_count" in names
        has_ts = "timestamp" in names
        t0 = time.perf_counter()
        nb, ntx, ts_min, ts_max = con.execute(f"""
            SELECT count(*),
                   {'SUM(transaction_count)' if has_txc else 'NULL'},
                   {'MIN(timestamp)' if has_ts else 'NULL'},
                   {'MAX(timestamp)' if has_ts else 'NULL'}
            FROM read_parquet({[str(f) for f in bfiles]!r}, union_by_name=true)
            WHERE CAST({c['number']} AS BIGINT) BETWEEN {lo_blk} AND {hi_blk}
        """).fetchone()
        n_blocks = int(nb)
        log.info("-" * 74)
        log.info("chain activity (block headers, %.1fs):",
                 time.perf_counter() - t0)
        log.info("  blocks              : %s", f"{n_blocks:,}")
        if has_ts:
            d0 = datetime.fromtimestamp(int(ts_min), tz=timezone.utc).date()
            d1 = datetime.fromtimestamp(int(ts_max), tz=timezone.utc).date()
            log.info("  date range          : %s -> %s", d0, d1)
        if has_txc:
            log.info("  transactions        : %s", f"{int(ntx):,}")
            log.info("  avg txs / block     : %.1f", int(ntx) / n_blocks)
        log.info("  avg flow rows/block : %.1f", int(n_rows) / n_blocks)
    else:
        log.warning("decisions.json not found (%s) -> skipping header tier",
                    dec_path)

    # ---------------- tier C: per-shard exact scan ----------------------
    if args.no_shard_scan:
        log.info("-" * 74)
        log.info("tier C skipped (--no-shard-scan)")
        return 0

    t0 = time.perf_counter()
    n_addr = 0
    n_addr_nonzero = 0
    n_addr_negative = 0
    sum_bal = 0
    credit_wei = 0
    credit_rows = 0
    buckets: Counter = Counter()
    bucket_wei: Counter = Counter()
    era_rows: Counter = Counter()
    top_bal: list[tuple[str, int, int]] = []
    top_act: list[tuple[str, int, int]] = []
    bsql = bucket_sql()

    for i, f in enumerate(files):
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE pa AS
            SELECT address,
                   SUM(CAST(value AS HUGEINT)) AS bal,
                   count(*) AS n_rows,
                   COALESCE(SUM(CAST(value AS HUGEINT))
                            FILTER (WHERE value > 0), 0) AS credit,
                   count(*) FILTER (WHERE value > 0) AS credit_n
            FROM read_parquet('{f}') GROUP BY address
        """)
        na, nz, neg, sb, cw, cr = con.execute("""
            SELECT count(*),
                   count(*) FILTER (WHERE bal > 0),
                   count(*) FILTER (WHERE bal < 0),
                   COALESCE(SUM(bal), 0),
                   COALESCE(SUM(credit), 0),
                   COALESCE(SUM(credit_n), 0)
            FROM pa
        """).fetchone()
        n_addr += int(na)
        n_addr_nonzero += int(nz)
        n_addr_negative += int(neg)
        sum_bal += int(sb)
        credit_wei += int(cw)
        credit_rows += int(cr)
        for label, cnt, w in con.execute(
            f"SELECT {bsql}, count(*), COALESCE(SUM(bal), 0) "
            f"FROM pa GROUP BY 1"
        ).fetchall():
            buckets[label] += int(cnt)
            bucket_wei[label] += int(w)
        top_bal += [(a, int(v), int(r)) for a, v, r in con.execute(
            f"SELECT address, bal, n_rows FROM pa "
            f"ORDER BY bal DESC LIMIT {args.top}").fetchall()]
        top_act += [(a, int(r), int(v)) for a, r, v in con.execute(
            f"SELECT address, n_rows, bal FROM pa "
            f"ORDER BY n_rows DESC LIMIT {args.top}").fetchall()]
        for era, cnt in con.execute(
            f"SELECT {ERA_CASE}, count(*) FROM read_parquet('{f}') GROUP BY 1"
        ).fetchall():
            era_rows[era] += int(cnt)
        log.info("  shard scan progress: %d/%d (%s)", i + 1, len(files), f.name)

    top_bal = sorted(top_bal, key=lambda t: -t[1])[:args.top]
    top_act = sorted(top_act, key=lambda t: -t[1])[:args.top]

    log.info("-" * 74)
    log.info("addresses (EXACT, %.0fs):", time.perf_counter() - t0)
    log.info("  unique addresses ever active : %s   (ref: etherscan "
             "address chart)", f"{n_addr:,}")
    log.info("  with balance > 0 at block %s : %s (%.1f%%)",
             f"{int(hi_blk):,}", f"{n_addr_nonzero:,}",
             100.0 * n_addr_nonzero / n_addr)
    log.info("  with NEGATIVE balance        : %s   (must be 0 -- a nonzero "
             "count means reconstruction error)", f"{n_addr_negative:,}")
    log.info("  sum of final balances        : %s   (identity: == minted - "
             "burned of the transfer ledger)", eth(sum_bal))
    log.info("  total credited volume        : %s over %s credit rows",
             eth(credit_wei), f"{credit_rows:,}")
    log.info("-" * 74)
    log.info("final balance distribution:")
    log.info("  %-22s %16s %10s %24s", "bucket", "addresses", "share", "held")
    for label, _ in BALANCE_BUCKETS:
        if label in buckets:
            log.info("  %-22s %16s %9.2f%% %24s", label,
                     f"{buckets[label]:,}", 100.0 * buckets[label] / n_addr,
                     eth(bucket_wei[label]))
    log.info("-" * 74)
    log.info("top %d holders (final balance):", args.top)
    for a, v, r in top_bal:
        log.info("  %s  %22s  (%s rows)", a, eth(v), f"{r:,}")
    log.info("-" * 74)
    log.info("top %d most active addresses (ledger rows):", args.top)
    for a, r, v in top_act:
        log.info("  %s  %14s rows  (balance %s)", a, f"{r:,}", eth(v))
    log.info("-" * 74)
    log.info("flow rows per era:")
    for era in ERA_ORDER:
        if era in era_rows:
            log.info("  %-14s: %16s (%.1f%%)", era, f"{era_rows[era]:,}",
                     100.0 * era_rows[era] / int(n_rows))
    log.info("-" * 74)
    log.info("reference points:")
    for name, ref in REFERENCES:
        log.info("  - %-18s %s", name + ":", ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
