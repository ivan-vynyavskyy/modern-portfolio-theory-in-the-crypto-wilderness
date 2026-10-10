#!/usr/bin/env python3
"""
2_0_eth_transfer_aggregator.py

Purpose
-------
ETH analog of ../erc20_src/2_1_erc20_transfer_aggregator.py: turn the native ETH
transfer ledger (block_range=*/part-*.parquet from 1_0_build_transfers.py) into
the per-token address-flow ledger format the portfolio reconstruction
pipeline (4_0) consumes.

What it does
------------
- Explodes every transfer into a debit row (sender, -value) and a credit row
  (recipient, +value). NULL sides (mint/burn: issuance, withdrawals, genesis,
  fee burn) emit no row — unlike ERC-20 there is no zero-address convention.
- Sorts by (address, block, log_index) and writes the same layout as 2_1:
  <output>/token_address=<token>/<token>.parquet
  with columns: address, block, log_index, value (DECIMAL(38,0), signed).
- Runs fully out-of-core in DuckDB (the exploded ledger is ~2x the transfer
  ledger, i.e. billions of rows — the in-memory pyarrow approach of 2_1
  cannot hold it).

Invariant checked after writing: SUM(value) over the whole address ledger
equals minted - burned of the transfer ledger (each transfer contributes
+v and -v except the mint/burn sides).

Inputs
------
- --output-base (config): the transfer ledger root with block_range=*/
- --output: ledgers dir (e.g. the pipeline's ledgers_dir)
- --token: pseudo token address for native ETH
  (default 0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee, the common ETH marker)

Outputs
-------
- <output>/token_address=<token>/<token>.parquet          (default)
- <output>/token_address=<token>/<token>-shard-<k>.parquet (--shards N:
  N address-hash-bucketed files, each sorted; a wallet's rows live in
  exactly one shard. Use when the single-file external sort is too slow.)

CLI
---
  $ python 2_0_eth_transfer_aggregator.py --output /path/to/ledgers
  $ python 2_0_eth_transfer_aggregator.py --output /path/to/ledgers --shards 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import duckdb

sys.path.append(str(Path(__file__).resolve().parent))
from config import Config, add_config_args, setup_logging  # noqa: E402

ETH_PSEUDO_TOKEN = "0x" + "ee" * 20

_MEM_UNITS = {"B": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def _parse_mem_bytes(s: str) -> int | None:
    m = re.match(r"^\s*([\d.]+)\s*([KMGT]?)I?B?\s*$", s.strip(), re.IGNORECASE)
    if not m:
        return None
    return int(float(m.group(1)) * _MEM_UNITS[m.group(2).upper() or "B"])


def _available_ram_bytes() -> int | None:
    """Physical RAM, further limited by a cgroup cap if we run inside one
    (SLURM job / Jupyter container) — sysconf alone reports the whole host."""
    limits = []
    try:
        limits.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError):
        pass
    for p in ("/sys/fs/cgroup/memory.max",                    # cgroup v2
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):  # cgroup v1
        try:
            txt = Path(p).read_text().strip()
            if txt.isdigit() and int(txt) < 1 << 60:  # skip "max"/unlimited
                limits.append(int(txt))
        except OSError:
            pass
    return min(limits) if limits else None


def safe_memory_limit(configured: str, log) -> str:
    """A DuckDB memory_limit above physical RAM disables spilling (the OS
    swaps before DuckDB ever thinks it is out of memory) — clamp to 60% of
    what this node/container really has."""
    want = _parse_mem_bytes(configured)
    have = _available_ram_bytes()
    if want is None or have is None:
        return configured
    cap = int(have * 0.6)
    if want <= cap:
        return configured
    clamped = f"{cap // (1 << 30)}GB"
    log.warning("memory_limit %s exceeds 60%% of available RAM (%.0fGB) "
                "-> clamped to %s so sorts spill instead of swapping",
                configured, have / (1 << 30), clamped)
    return clamped

EXPLODED_SQL = """
    SELECT sender AS address, block, log_index, -value AS value
    FROM read_parquet('{glob}') WHERE sender IS NOT NULL
    UNION ALL
    SELECT recipient AS address, block, log_index, value
    FROM read_parquet('{glob}') WHERE recipient IS NOT NULL
"""


def open_connection(cfg: Config, mem_limit: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    cfg.scratch_tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit='{mem_limit}'")
    con.execute(f"SET threads={cfg.duckdb_threads}")
    con.execute(f"SET temp_directory='{cfg.scratch_tmp}'")
    # explicit finite cap: the default ("available disk at temp_directory")
    # is mis-detected on GPFS and then spills fail with a bogus
    # "PiB used" OutOfMemoryException
    con.execute("SET max_temp_directory_size='1TiB'")
    con.execute("SET preserve_insertion_order=false")
    return con


def clean_stale_spill(cfg: Config, log) -> None:
    """Killed runs leave DuckDB spill files behind in scratch — they can eat
    hundreds of GB of quota and starve later spills. Safe to delete at
    startup: no other connection of ours is live yet."""
    freed = 0
    for pat in ("duckdb_temp*", "*.tmp"):
        for f in cfg.scratch_tmp.glob(pat):
            if f.is_file():
                freed += f.stat().st_size
                f.unlink(missing_ok=True)
    if freed:
        log.info("removed stale DuckDB spill files: %.1f GB freed",
                 freed / (1 << 30))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Native ETH transfer ledger -> per-address flow ledger (2_1 format)"
    )
    parser.add_argument("--output", required=True,
                        help="ledgers output dir (token_address=<token>/ is created inside)")
    parser.add_argument("--token", default=ETH_PSEUDO_TOKEN,
                        help=f"pseudo token address for ETH (default {ETH_PSEUDO_TOKEN})")
    parser.add_argument("--shards", type=int, default=0,
                        help="0 = one globally sorted file (exact 2_1 contract); "
                             "N>0 = N address-hash shards, each sorted internally")
    parser.add_argument("--force", action="store_true")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = Config.from_args(args)
    log = setup_logging(cfg.logs_dir, "eth_aggregator")

    token = args.token.lower()
    in_glob = str(cfg.output_base / "block_range=*" / "*.parquet")
    token_dir = Path(args.output).expanduser().resolve() / f"token_address={token}"
    success = token_dir / "_SUCCESS"
    if success.exists() and not args.force:
        log.info("output already built (%s) -> nothing to do (use --force)", success)
        return 0
    token_dir.mkdir(parents=True, exist_ok=True)
    success.unlink(missing_ok=True)
    # tmp leftovers are always garbage; finished shard files are kept unless
    # --force (they were atomically renamed into place, so existing = complete)
    for stale in token_dir.glob("*.parquet.tmp-*"):
        stale.unlink()
    if args.force:
        for stale in token_dir.glob("*.parquet"):
            stale.unlink()

    mem_limit = safe_memory_limit(cfg.duckdb_memory_limit, log)
    cfg.scratch_tmp.mkdir(parents=True, exist_ok=True)
    clean_stale_spill(cfg, log)
    con = open_connection(cfg, mem_limit)
    t0 = time.perf_counter()

    # transfer-ledger side of the invariant
    n_in, minted, burned = con.execute(
        f"SELECT count(*), "
        f"COALESCE(SUM(value) FILTER (WHERE sender IS NULL), 0), "
        f"COALESCE(SUM(value) FILTER (WHERE recipient IS NULL), 0) "
        f"FROM read_parquet('{in_glob}')"
    ).fetchone()
    minted, burned = int(minted), int(burned)
    log.info("input: %s transfer rows | minted=%d burned=%d", f"{n_in:,}", minted, burned)

    exploded = EXPLODED_SQL.format(glob=in_glob)
    rows_out = 0
    if args.shards <= 0:
        out_path = token_dir / f"{token}.parquet"
        tmp = token_dir / f"{token}.parquet.tmp-{os.getpid()}"
        log.info("writing single globally sorted ledger (external sort; scratch "
                 "needs roughly 2-3x the output size) ...")
        rows_out = con.execute(
            f"COPY (SELECT address, block, log_index, value FROM ({exploded}) "
            f"ORDER BY address, block, log_index) "
            f"TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        ).fetchone()[0]
        os.replace(tmp, out_path)
        log.info("wrote %s rows -> %s", f"{rows_out:,}", out_path.name)
    else:
        # Two phases so the 7.4B-row ledger is scanned ONCE, not once per
        # shard:
        #   1. explode + bucket by hash(address) into a hive-partitioned
        #      scratch dir (streaming write, no sort -> low memory);
        #   2. per shard, sort only that bucket (1/N of the data) and write
        #      the final file. Completed shards are skipped on resume.
        part_root = cfg.scratch_tmp / f"eth_agg_part_{args.shards}"
        marker = part_root / "_PARTITIONED"
        marker_want = json.dumps({"rows_in": n_in, "shards": args.shards})
        if marker.exists() and marker.read_text().strip() == marker_want:
            log.info("phase 1/2: reusing partitioned scratch %s", part_root)
        else:
            if part_root.exists():
                shutil.rmtree(part_root)
            log.info("phase 1/2: exploding + bucketing into %d shards "
                     "(single ledger scan) -> %s ...", args.shards, part_root)
            tp = time.perf_counter()
            con.execute(
                f"COPY (SELECT address, block, log_index, value, "
                f"hash(address) % {args.shards} AS shard FROM ({exploded})) "
                f"TO '{part_root}' "
                f"(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (shard))"
            )
            marker.write_text(marker_want + "\n")
            log.info("phase 1/2 done in %.1fs", time.perf_counter() - tp)
        # Buckets are uneven: one address's rows can never be split across
        # buckets, and mega-addresses (MEV builders collecting a fee_tip for
        # every tx of every block they built) hold 10^8-10^9 rows on their
        # own. A bucket whose sort exceeds RAM spills to GPFS and crawls, so:
        # - log the bucket size BEFORE sorting (big != hung);
        # - split oversized buckets into sub-files by a secondary address
        #   hash (a wallet still lives in exactly one output file).
        SORT_ROWS_CAP = 250_000_000  # ~25GB sort footprint, comfortably in RAM
        for s in range(args.shards):
            shard_glob = str(part_root / f"shard={s}" / "*.parquet")
            single = token_dir / f"{token}-shard-{s:03d}.parquet"
            if single.exists():  # complete from an earlier run (atomic rename)
                n = con.execute(
                    f"SELECT count(*) FROM read_parquet('{single}')"
                ).fetchone()[0]
                rows_out += n
                log.info("shard %d/%d: exists (%s rows) -> skipped",
                         s + 1, args.shards, f"{n:,}")
                continue
            n_bucket = con.execute(
                f"SELECT count(*) FROM read_parquet('{shard_glob}')"
            ).fetchone()[0]
            k = max(1, math.ceil(n_bucket / SORT_ROWS_CAP))
            if k > 1:
                log.info("shard %d/%d: %s bucket rows -> splitting into %d "
                         "sub-files (one address's rows cannot be split; a "
                         "dominant address keeps its sub-sort large and it "
                         "may spill — slow but progressing)",
                         s + 1, args.shards, f"{n_bucket:,}", k)
            for j in range(k):
                out_path = single if k == 1 else \
                    token_dir / f"{token}-shard-{s:03d}-{j:02d}.parquet"
                if out_path.exists():
                    n = con.execute(
                        f"SELECT count(*) FROM read_parquet('{out_path}')"
                    ).fetchone()[0]
                    rows_out += n
                    log.info("shard %d/%d sub %d/%d: exists (%s rows) -> "
                             "skipped", s + 1, args.shards, j + 1, k, f"{n:,}")
                    continue
                sub_where = "" if k == 1 else \
                    f"WHERE hash(address || '#sub') % {k} = {j} "
                tmp = token_dir / f"{out_path.name}.tmp-{os.getpid()}"
                ts = time.perf_counter()
                # fresh connection per sort: releases the previous sort's
                # buffer pool back to the OS before the next one allocates
                con.close()
                con = open_connection(cfg, mem_limit)
                n = con.execute(
                    f"COPY (SELECT address, block, log_index, value "
                    f"FROM read_parquet('{shard_glob}') {sub_where}"
                    f"ORDER BY address, block, log_index) "
                    f"TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)"
                ).fetchone()[0]
                os.replace(tmp, out_path)
                rows_out += n
                if k == 1:
                    log.info("shard %d/%d: %s rows in %.1fs", s + 1,
                             args.shards, f"{n:,}", time.perf_counter() - ts)
                else:
                    log.info("shard %d/%d sub %d/%d: %s rows in %.1fs",
                             s + 1, args.shards, j + 1, k, f"{n:,}",
                             time.perf_counter() - ts)
        shutil.rmtree(part_root)

    # invariant: SUM over the address ledger == minted - burned
    out_glob = str(token_dir / "*.parquet")
    total = int(con.execute(
        f"SELECT COALESCE(SUM(value), 0) FROM read_parquet('{out_glob}')"
    ).fetchone()[0])
    expected = minted - burned
    if total != expected:
        log.error("INVARIANT VIOLATED: SUM(address ledger)=%d != minted-burned=%d "
                  "(diff %d wei) — _SUCCESS withheld", total, expected, total - expected)
        return 1
    log.info("invariant OK: SUM(value)=%d == minted-burned", total)

    stats = {
        "token": token, "rows_in": n_in, "rows_out": rows_out,
        "shards": max(args.shards, 0),
        "sum_value_wei": str(total),
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    success.write_text(json.dumps(stats) + "\n")
    log.info("done: %s address-flow rows in %.1fs", f"{rows_out:,}",
             time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
