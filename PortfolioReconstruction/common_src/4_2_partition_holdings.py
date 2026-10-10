#!/usr/bin/env python3
"""
4_2_partition_holdings.py

Purpose
-------
Consolidate the per-period Parquet files produced by separate 4_0 runs into a
single block-partitioned dataset, so downstream code sees ONE path regardless of
how many runs it took to build the history.

Layout
------
    <root>/
        block_number=14116761/2022_02-2023_02_0.parquet
        block_number=14297759/2022_02-2023_02_0.parquet
        ...
        _manifest.csv          <- block -> date, rows, source run, written_at

`block_number` lives in the directory name (Hive convention), not in the file
body. Readers must therefore ask for Hive partitioning:

    ds.dataset(root, format="parquet", partitioning="hive")     # pyarrow
    read_parquet('<root>/**/*.parquet', hive_partitioning=true) # duckdb

That argument is harmless on a plain single file, so the same call site works
for both old single-file outputs and this dataset.

Why partition by block_number
-----------------------------
5_0 filters holdings with `ds.field("block_number") == block` and enumerates the
distinct blocks up front. Against one flat file both operations scan the whole
thing; against this layout the filter touches exactly one directory and the
enumeration is a directory listing.

Inputs
------
- --sources (one or more Parquet files from 4_0; globs allowed)
- --root (output dataset directory)
- --block-to-date-csv (optional; adds a human-readable date to the manifest)

Safety
------
Two 4_0 runs covering the same snapshot date would each contribute a full copy
of every wallet for that block, silently doubling it. A block already owned by
another run is therefore refused unless --allow-overlap is passed. Sources
already recorded in the manifest are skipped so interrupted runs can resume.

CLI
---
  $ python 4_2_partition_holdings.py \
        --sources /data/recon-eth-erc20-2020_01-2022_01.parquet \
                  /data/recon-eth-erc20-2022_02-2023_02.parquet \
        --root /data/recon-holdings \
        --block-to-date-csv /data/block_to_date.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import duckdb

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.log_utils import log, log_exc  # noqa: E402

MANIFEST_NAME = "_manifest.csv"
MANIFEST_COLS = ["block_number", "date", "rows", "source_tag", "source_file",
                 "written_at"]


def open_con(threads: int, memory: str, tmp_dir: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{memory}'")
    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp_dir}'")
    con.execute("SET max_temp_directory_size='500GiB'")
    con.execute("SET preserve_insertion_order=false")
    log(f"[duckdb] threads={threads} memory={memory} temp_directory={tmp_dir}")
    return con


def sql_quote(p: str) -> str:
    return "'" + str(p).replace("'", "''") + "'"


def load_manifest(root: pathlib.Path) -> List[Dict[str, str]]:
    path = root / MANIFEST_NAME
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_manifest(root: pathlib.Path, rows: List[Dict[str, str]]) -> None:
    rows = sorted(rows, key=lambda r: int(r["block_number"]))
    tmp = root / (MANIFEST_NAME + ".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(root / MANIFEST_NAME)


def load_block_dates(csv_path: Optional[str]) -> Dict[int, str]:
    """Map block_number -> date string, tolerating the various column names."""
    if not csv_path or not pathlib.Path(csv_path).exists():
        return {}
    import pandas as pd
    df = pd.read_csv(csv_path)
    block_col = next((c for c in ("block_number", "block") if c in df.columns), None)
    date_col = next((c for c in ("date", "bound_date", "block_time_utc")
                     if c in df.columns), None)
    if not block_col or not date_col:
        log(f"[dates] WARN unusable columns in {csv_path}: {list(df.columns)}")
        return {}
    out: Dict[int, str] = {}
    for b, d in zip(df[block_col], df[date_col]):
        try:
            out[int(b)] = str(d)[:10]
        except (TypeError, ValueError):
            continue
    return out


def source_blocks(con, src: str) -> List[Tuple[int, int]]:
    """(block_number, row_count) present in a source file."""
    return [(int(b), int(n)) for b, n in con.execute(f"""
        SELECT block_number, count(*) FROM read_parquet({sql_quote(src)})
        GROUP BY block_number ORDER BY block_number
    """).fetchall()]


def partition_dir(root: pathlib.Path, block: int) -> pathlib.Path:
    return root / f"block_number={block}"


def existing_owner(root: pathlib.Path, block: int) -> Optional[str]:
    """Tag of the run that already populated this block, if any."""
    d = partition_dir(root, block)
    if not d.is_dir():
        return None
    files = sorted(p.name for p in d.glob("*.parquet"))
    if not files:
        return None
    # filenames are "<tag>_<i>.parquet"
    return files[0].rsplit("_", 1)[0]


def ingest_source(con, root: pathlib.Path, src: str, tag: str,
                  block_dates: Dict[int, str], allow_overlap: bool) -> List[Dict[str, str]]:
    t0 = time.perf_counter()
    blocks = source_blocks(con, src)
    if not blocks:
        log(f"[{tag}] no rows — skipped")
        return []
    total = sum(n for _, n in blocks)
    log(f"[{tag}] {len(blocks)} blocks, {total:,} rows "
        f"({blocks[0][0]}..{blocks[-1][0]})")

    clashes = [(b, o) for b, _ in blocks
               if (o := existing_owner(root, b)) is not None and o != tag]
    if clashes:
        msg = ", ".join(f"block {b} already written by '{o}'" for b, o in clashes[:5])
        if not allow_overlap:
            raise SystemExit(
                f"[{tag}] REFUSING: {len(clashes)} block(s) overlap another run "
                f"({msg}). Two runs covering the same snapshot would duplicate "
                f"every wallet at that block. Re-run with --allow-overlap only "
                f"if you intend to merge them.")
        log(f"[{tag}] WARN overlapping blocks accepted: {msg}")

    log(f"[{tag}] writing partitions ...")
    con.execute(f"""
        COPY (SELECT * FROM read_parquet({sql_quote(src)}))
        TO {sql_quote(str(root))}
        (FORMAT PARQUET, PARTITION_BY (block_number), COMPRESSION ZSTD,
         FILENAME_PATTERN '{tag}_{{i}}', OVERWRITE_OR_IGNORE)
    """)
    log(f"[{tag}] written in {time.perf_counter() - t0:.0f}s — verifying ...")

    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows: List[Dict[str, str]] = []
    bad = 0
    for block, n_src in blocks:
        n_out = con.execute(
            f"SELECT count(*) FROM read_parquet("
            f"{sql_quote(str(partition_dir(root, block) / '*.parquet'))})"
        ).fetchone()[0]
        if n_out != n_src:
            bad += 1
            log(f"[{tag}] MISMATCH block {block}: source={n_src:,} written={n_out:,}")
        rows.append({"block_number": str(block),
                     "date": block_dates.get(block, ""),
                     "rows": str(n_out), "source_tag": tag,
                     "source_file": str(src), "written_at": stamp})
    if bad:
        raise SystemExit(f"[{tag}] {bad} partition(s) have wrong row counts — aborting")
    log(f"[{tag}] OK — {len(blocks)} partitions verified, {total:,} rows "
        f"({time.perf_counter() - t0:.0f}s total)")
    return rows


def main() -> int:
    p = argparse.ArgumentParser(
        description="Consolidate 4_0 outputs into a block-partitioned dataset")
    p.add_argument("--sources", nargs="+", required=True,
                   help="4_0 output parquet files (globs allowed)")
    p.add_argument("--root", required=True, help="output dataset directory")
    p.add_argument("--block-to-date-csv", default=None)
    p.add_argument("--allow-overlap", action="store_true",
                   help="permit a source to write blocks another run already owns")
    p.add_argument("--force", action="store_true",
                   help="re-ingest sources already present in the manifest")
    p.add_argument("--threads", type=int, default=int(os.getenv("DUCKDB_THREADS", "16")))
    p.add_argument("--memory", default=os.getenv("DUCKDB_MEMORY_LIMIT", "64GB"))
    p.add_argument("--tmp-dir", default=None)
    args = p.parse_args()

    root = pathlib.Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    sources: List[str] = []
    for s in args.sources:
        hits = sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s]
        if not hits:
            log(f"[sources] WARN no match for {s!r}")
        sources.extend(hits)
    missing = [s for s in sources if not pathlib.Path(s).exists()]
    if missing:
        raise SystemExit(f"missing source file(s): {missing}")
    if not sources:
        raise SystemExit("no source files resolved")

    tmp_dir = args.tmp_dir or os.getenv("DUCKDB_TMPDIR") or str(root / "duckdb_tmp_4_2")
    con = open_con(args.threads, args.memory, tmp_dir)

    manifest = load_manifest(root)
    done_tags = {r["source_tag"] for r in manifest}
    block_dates = load_block_dates(args.block_to_date_csv)
    log(f"[plan] {len(sources)} source(s) -> {root}")
    if block_dates:
        log(f"[dates] {len(block_dates)} block->date entries loaded")

    for src in sources:
        tag = pathlib.Path(src).stem
        if tag in done_tags and not args.force:
            log(f"[{tag}] already in manifest — skipped (use --force to redo)")
            continue
        manifest = [r for r in manifest if r["source_tag"] != tag]
        manifest.extend(ingest_source(con, root, src, tag, block_dates,
                                      args.allow_overlap))
        write_manifest(root, manifest)

    blocks = sorted({int(r["block_number"]) for r in manifest})
    total = sum(int(r["rows"]) for r in manifest)
    log("=" * 74)
    log(f"[done] {len(blocks)} partitions | {total:,} rows | root={root}")
    if blocks:
        log(f"[done] blocks {blocks[0]}..{blocks[-1]} | manifest={root / MANIFEST_NAME}")
    log("[done] read it with: "
        "ds.dataset(root, format='parquet', partitioning='hive')")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        log_exc(f"[fatal] {exc}")
        raise SystemExit(1)
