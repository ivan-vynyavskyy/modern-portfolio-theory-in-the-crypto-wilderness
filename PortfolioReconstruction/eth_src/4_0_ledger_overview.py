#!/usr/bin/env python3
"""
4_0_ledger_overview.py

Purpose
-------
Fast global sanity picture of the reconstructed ETH transfer ledger, with
reference points from public sources (Etherscan / ultrasound.money /
beaconcha.in) so headline numbers can be eyeballed against "what Ethereum
actually looks like".

Three cost tiers (all enabled by default):
  1. per-chunk _SUCCESS stats            -> instant (no data scan)
  2. block-header scan via decisions.json -> seconds (24M rows)
  3. approx-distinct ledger scan          -> a few minutes (--no-scan to skip)

CLI
---
  $ python 4_0_ledger_overview.py                # full picture
  $ python 4_0_ledger_overview.py --no-scan      # skip the distinct-count scan
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

sys.path.append(str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    Config,
    add_config_args,
    partition_files,
    setup_logging,
)

WEI = 10**18

# Public reference points for eyeballing (approximate, mid-2020s figures —
# verify against the live sources when writing anything down).
REFERENCES = [
    ("genesis allocation", "72,009,990.50 ETH exactly (mainnet genesis.json)"),
    ("supply at the Merge", "~120.52M ETH (etherscan.io/chart/ethersupply); "
     "differs from EL genesis+PoW by pre-Merge burn and pre-Merge beacon issuance"),
    ("total burned since London", "ultrasound.money ('burned' counter)"),
    ("current total supply", "~120.x M ETH (ultrasound.money) — our EL-balance-sum "
     "figure EXCEEDS it by cumulative principal withdrawals (deposit-contract "
     "double count); cross-check instead via the deposit contract balance"),
    ("cumulative transactions", "etherscan.io/chart/tx (cumulative)"),
    ("unique addresses", "etherscan.io/chart/address — ours counts addresses that "
     "ever moved native ETH incl. gas, so same order of magnitude"),
    ("beacon withdrawals", "beaconcha.in (total withdrawn ETH)"),
]


def eth(wei: int) -> str:
    return f"{wei / WEI:,.2f} ETH"


def main() -> int:
    parser = argparse.ArgumentParser(description="Global overview of the ETH ledger")
    parser.add_argument("--no-scan", action="store_true",
                        help="skip the approx-distinct full-ledger scan")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = Config.from_args(args)
    log = setup_logging(cfg.logs_dir, "ledger_overview")

    # ---------- tier 1: per-chunk _SUCCESS stats (instant) ----------
    chunks = []
    for marker in sorted(cfg.output_base.glob("block_range=*/_SUCCESS")):
        chunks.append(json.loads(marker.read_text()))
    if not chunks:
        log.error("no completed chunks under %s", cfg.output_base)
        return 1

    rows_total = sum(c["rows_out"] for c in chunks)
    minted = sum(int(c["minted_wei"]) for c in chunks)
    burned = sum(int(c["burned_wei"]) for c in chunks)
    lo = min(c["lo"] for c in chunks)
    hi = max(c["hi"] for c in chunks)

    by_type: dict[str, dict] = {}
    for c in chunks:
        for t, v in (c.get("by_type") or {}).items():
            agg = by_type.setdefault(t, {"rows": 0, "wei": 0})
            agg["rows"] += v["rows"]
            agg["wei"] += int(v["wei"])

    log.info("=" * 74)
    log.info("LEDGER OVERVIEW  (%d chunks, blocks %s - %s)", len(chunks),
             f"{lo:,}", f"{hi:,}")
    log.info("=" * 74)
    log.info("transfer rows total   : %s", f"{rows_total:,}")
    log.info("%-12s %16s %26s", "type", "rows", "total value")
    for t in sorted(by_type):
        v = by_type[t]
        log.info("%-12s %16s %26s", t, f"{v['rows']:,}", eth(v["wei"]))

    # ---------- derived protocol-level checkpoints ----------
    gen = by_type.get("genesis", {}).get("wei", 0)
    rew = by_type.get("reward", {}).get("wei", 0)
    wdr = by_type.get("withdrawal", {}).get("wei", 0)
    log.info("-" * 74)
    log.info("supply accounting:")
    log.info("  genesis               : %s   (reference: 72,009,990.50 ETH exact)", eth(gen))
    log.info("  PoW issuance (rewards): %s", eth(rew))
    log.info("  genesis + rewards     : %s   (identity: published at-Merge supply "
             "~120.52M = this - pre-Merge burn ~2.65M + pre-Merge beacon issuance ~0.9M)",
             eth(gen + rew))
    log.info("  beacon withdrawals    : %s   (reference: beaconcha.in total)", eth(wdr))
    log.info("  burned (base+blob fee): %s   (reference: ultrasound.money)", eth(burned))
    log.info("  EL supply (mint-burn) : %s   (identity: EXCEEDS published total "
             "supply by cumulative principal withdrawals — withdrawals re-mint "
             "principal while deposits stay in the deposit contract; "
             "ours - published ~= principal withdrawn - unwithdrawn CL rewards)",
             eth(minted - burned))

    # ---------- tier 2: block headers (seconds) ----------
    if cfg.decisions_path.exists():
        dec = json.loads(cfg.decisions_path.read_text())
        b = dec["block"]
        c = b["cols"]
        bfiles = [f for p in b["partitions"]
                  if p["lo"] is not None and p["lo"] <= hi and p["hi"] >= lo
                  for f in partition_files(p)]
        names = b.get("schema_names", [])
        has_txc = "transaction_count" in names
        has_ts = "timestamp" in names
        con = duckdb.connect()
        con.execute(f"SET threads={cfg.duckdb_threads}")
        t0 = time.perf_counter()
        nblocks, ntx, ts_min, ts_max = con.execute(f"""
            SELECT count(*),
                   {'SUM(transaction_count)' if has_txc else 'NULL'},
                   {'MIN(timestamp)' if has_ts else 'NULL'},
                   {'MAX(timestamp)' if has_ts else 'NULL'}
            FROM read_parquet({[str(f) for f in bfiles]!r}, union_by_name=true)
            WHERE CAST({c['number']} AS BIGINT) BETWEEN {lo} AND {hi}
        """).fetchone()
        log.info("-" * 74)
        log.info("chain coverage (block headers, %.1fs):", time.perf_counter() - t0)
        log.info("  blocks                : %s", f"{nblocks:,}")
        if has_ts:
            d0 = datetime.fromtimestamp(int(ts_min), tz=timezone.utc).date()
            d1 = datetime.fromtimestamp(int(ts_max), tz=timezone.utc).date()
            log.info("  date range            : %s -> %s", d0, d1)
        if has_txc:
            log.info("  transactions (headers): %s   (reference: etherscan.io/chart/tx "
                     "cumulative)", f"{int(ntx):,}")
            log.info("  avg txs / block       : %.1f", ntx / nblocks)
    else:
        log.warning("decisions.json not found -> skipping block-header tier")

    # ---------- tier 3: distinct counts over the ledger (minutes) ----------
    if not args.no_scan:
        con = duckdb.connect()
        cfg.scratch_tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET memory_limit='{cfg.duckdb_memory_limit}'")
        con.execute(f"SET threads={cfg.duckdb_threads}")
        con.execute(f"SET temp_directory='{cfg.scratch_tmp}'")
        con.execute("SET preserve_insertion_order=false")
        # EXACT counts, memory-bounded (a single global count(DISTINCT) over
        # billions of rows does NOT spill well and can take the node down):
        # - process chunk by chunk: chunks are block-partitioned, so tx hashes
        #   never span chunks -> global tx count = sum of per-chunk distincts;
        # - per-chunk DISTINCT addresses go to a scratch spill file, deduped
        #   once at the end with a spill-friendly GROUP BY;
        # - GROUP BY (not DISTINCT aggregates) everywhere so DuckDB can go
        #   out-of-core.
        t0 = time.perf_counter()
        chunk_dirs = sorted(cfg.output_base.glob("block_range=*"))
        addr_tmp = cfg.scratch_tmp / "overview_addr_spill"
        addr_tmp.mkdir(parents=True, exist_ok=True)
        for f in addr_tmp.glob("*.parquet"):
            f.unlink()
        n_tx = 0
        n_rows = 0
        for i, d in enumerate(chunk_dirs):
            g = str(d / "*.parquet")
            c_tx, c_rows = con.execute(f"""
                SELECT (SELECT count(*) FROM
                          (SELECT tx_hash FROM read_parquet('{g}')
                           WHERE tx_hash IS NOT NULL GROUP BY tx_hash)),
                       (SELECT count(*) FROM read_parquet('{g}'))
            """).fetchone()
            n_tx += int(c_tx)
            n_rows += int(c_rows)
            con.execute(f"""
                COPY (
                  SELECT addr FROM (
                    SELECT sender AS addr FROM read_parquet('{g}')
                    WHERE sender IS NOT NULL
                    UNION ALL
                    SELECT recipient FROM read_parquet('{g}')
                    WHERE recipient IS NOT NULL
                  ) GROUP BY addr
                ) TO '{addr_tmp}/chunk_{i:03d}.parquet'
                  (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
            log.info("  distinct-count progress: chunk %d/%d (%s)",
                     i + 1, len(chunk_dirs), d.name)
        n_addr, = con.execute(f"""
            SELECT count(*) FROM (
              SELECT addr FROM read_parquet('{addr_tmp}/*.parquet') GROUP BY addr
            )
        """).fetchone()
        for f in addr_tmp.glob("*.parquet"):
            f.unlink()
        log.info("-" * 74)
        log.info("ledger distinct counts (EXACT, %.0fs):", time.perf_counter() - t0)
        log.info("  unique addresses      : %s   (reference: etherscan.io/chart/address)",
                 f"{n_addr:,}")
        log.info("  distinct tx hashes    : %s   (vs header tx count: gap = txs with "
                 "no ledger row, i.e. zero-fee zero-value)", f"{int(n_tx):,}")
        log.info("  ledger rows (scan)    : %s   (matches chunk stats: %s)",
                 f"{n_rows:,}", rows_total == n_rows)

    log.info("-" * 74)
    log.info("reference sources for comparison:")
    for name, ref in REFERENCES:
        log.info("  - %-26s %s", name + ":", ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
