#!/usr/bin/env python3
"""
4_3_check_block_coverage.py

Purpose
-------
Sanity-check block coverage across the per-period holdings parquets produced
by separate 4_0 runs (or across the consolidated 4_2 dataset) BEFORE merging /
analyzing them:

  1. DUPLICATES  — the same snapshot block present in more than one source
                   (would double every wallet at that block after a merge);
  2. MISSING     — expected snapshot blocks not covered by any source;
  3. UNEXPECTED  — blocks present in a source but absent from the expected list.

Exit code is 0 only when all three checks pass, so this can gate 4_2.

Sources may be parquet files, globs, or a 4_2 dataset root directory
(detected by its block_number=*/ subdirs, read with hive partitioning).

CLI
---
  $ python 4_3_check_block_coverage.py \
        --sources /data/recon-eth-erc20-*.parquet

  $ python 4_3_check_block_coverage.py --sources /data/recon-holdings
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys
from typing import Dict, List, Tuple

import duckdb

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.log_utils import log  # noqa: E402

# Expected snapshot blocks: monthly, 2020-01 .. 2025-12.
BLOCKS: List[int] = [
    9193266, 9393154, 9581792, 9782602, 9976964, 10176690, 10370274,
    10570485, 10771925, 10966874, 11167817, 11363270, 11565019, 11766939,
    11948960, 12150245, 12344945, 12545219, 12738509, 12936340, 13136427,
    13330090, 13527859, 13717847, 13916166, 14116761, 14297759, 14497034,
    14688630, 14881677, 15053226, 15253306, 15449618, 15649595, 15871480,
    16086234, 16308190, 16530248, 16730072, 16950603, 17162287, 17382266,
    17595510, 17816434, 18037988, 18251965, 18473543, 18687851, 18908895,
    19129889, 19336607, 19557289, 19771560, 19993250, 20207949, 20429973,
    20651994, 20866919, 21089069, 21303934, 21525891, 21747950, 21948292,
    22170335, 22385294, 22606143, 22820674, 23042514, 23264566, 23479244,
    23700767, 23914921,
]


def sql_quote(p: str) -> str:
    return "'" + str(p).replace("'", "''") + "'"


def resolve_sources(raws: List[str]) -> List[str]:
    out: List[str] = []
    for s in raws:
        hits = sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s]
        if not hits:
            log(f"[sources] WARN no match for {s!r}")
        out.extend(hits)
    missing = [s for s in out if not pathlib.Path(s).exists()]
    if missing:
        raise SystemExit(f"missing source(s): {missing}")
    if not out:
        raise SystemExit("no sources resolved")
    return out


def blocks_in_source(con: duckdb.DuckDBPyConnection, src: str) -> Dict[int, int]:
    """block_number -> row count for one file or 4_2 dataset root."""
    p = pathlib.Path(src)
    if p.is_dir():
        scan = (f"read_parquet({sql_quote(str(p / 'block_number=*' / '*.parquet'))}, "
                f"hive_partitioning=true)")
    else:
        scan = f"read_parquet({sql_quote(src)})"
    rows = con.execute(
        f"SELECT block_number, count(*) FROM {scan} "
        f"GROUP BY block_number ORDER BY block_number"
    ).fetchall()
    return {int(b): int(n) for b, n in rows}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check holdings parquets for duplicated / missing snapshot blocks")
    ap.add_argument("--sources", nargs="+", required=True,
                    help="holdings parquet files, globs, or a 4_2 dataset root dir")
    ap.add_argument("--threads", type=int,
                    default=int(os.getenv("DUCKDB_THREADS", "16")))
    ap.add_argument("--memory", default=os.getenv("DUCKDB_MEMORY_LIMIT", "32GB"))
    args = ap.parse_args()

    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory}'")

    sources = resolve_sources(args.sources)
    log(f"[plan] {len(sources)} source(s) | {len(BLOCKS)} expected blocks "
        f"({BLOCKS[0]}..{BLOCKS[-1]})")

    # block -> [(source, rows), ...]
    owners: Dict[int, List[Tuple[str, int]]] = {}
    for src in sources:
        tag = pathlib.Path(src).name
        found = blocks_in_source(con, src)
        total = sum(found.values())
        log(f"[{tag}] {len(found)} blocks | {total:,} rows | "
            f"{min(found)}..{max(found)}" if found else f"[{tag}] EMPTY")
        for b, n in found.items():
            owners.setdefault(b, []).append((tag, n))

    expected = set(BLOCKS)
    present = set(owners)
    dups = {b: v for b, v in owners.items() if len(v) > 1}
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)

    log("=" * 74)
    if dups:
        log(f"[FAIL] {len(dups)} block(s) in MORE THAN ONE source:")
        for b in sorted(dups):
            who = ", ".join(f"{tag} ({n:,} rows)" for tag, n in dups[b])
            log(f"    block {b}: {who}")
    else:
        log("[ok] no block appears in more than one source")

    if missing:
        log(f"[FAIL] {len(missing)} expected block(s) MISSING: {missing}")
    else:
        log(f"[ok] all {len(BLOCKS)} expected blocks are covered")

    if unexpected:
        log(f"[FAIL] {len(unexpected)} block(s) NOT in the expected list: {unexpected}")
    else:
        log("[ok] no unexpected blocks")

    ok = not dups and not missing and not unexpected
    log("=" * 74)
    log(f"[result] {'PASS' if ok else 'NOT_PASS'} | "
        f"expected={len(BLOCKS)} present={len(present)} "
        f"duplicated={len(dups)} missing={len(missing)} unexpected={len(unexpected)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
