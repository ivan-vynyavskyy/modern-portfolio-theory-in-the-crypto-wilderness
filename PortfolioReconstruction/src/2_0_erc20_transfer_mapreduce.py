#!/usr/bin/env python3
"""
2_0_erc20_transfer_mapreduce.py

Purpose
-------
Convert raw Ethereum log Parquet shards into per-token ERC-20 transfer Parquet files, including wrapper mint/burn normalization.

What it does
------------
- Reads raw log shards (plain Parquet or Delta Lake snapshot) and filters Transfer/Deposit/Withdrawal events.
- Decodes sender, recipient, value, block, log_index, tx_hash and writes per-token shard Parquet files.
- Merges shards per token into a single Parquet with a tight value type.

Inputs
------
- --src-raw (directory; Parquet shards or Delta Lake; required columns: topic0, topics, data, block_id, log_index, address, tx_hash)
- --token-csv (CSV; columns: contract_address, is_erc_20, price_data_status, trading_volume_status, fdv_mcap_lt_eth)
- --map-root (directory; intermediate output)
- --final-root (directory; final output)
- SLURM_CPUS_PER_TASK (env var; optional, used by available_worker_count if enabled)

Outputs
-------
- <map-root>/token_address=<token>/shard-*.parquet (Parquet; columns: token_address, sender, recipient, block, log_index, value, tx_hash)
- <final-root>/token_address=<token>/<token>.parquet (Parquet; columns: sender, recipient, block, log_index, value, tx_hash)

CLI
---
Examples:
  $ python 2_0_erc20_transfer_mapreduce.py --src-raw data/logs --map-root /tmp/map --final-root data/out --token-csv tokens.csv
  $ python 2_0_erc20_transfer_mapreduce.py --src-raw /data/logs --map-root /tmp/map --final-root /data/erc20 --token-csv tokens.csv --batch 100 --map-timeout 7200

Notes
-----
- Transfer events are kept only if value > 0 and optionally in an allow-list.
- Deposit/Withdrawal are mapped to synthetic mint/burn using the zero address.
- Value is stored as base-10 strings in map shards and cast to a narrow type in reduce.
"""

from __future__ import annotations

import argparse
import decimal
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Iterable, Optional

import deltalake
import numpy as np
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from deltalake import DeltaTable

# Ensure helpers/ is importable when running from src/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from helpers.token_filters import load_token_allow_list
from utils.log_utils import log


# ----------------------------- Constant Helpers -----------------------------

def _sig_to_str_set(sig: bytes) -> set[str]:
    """Return signature hex with and without 0x prefix."""
    h = sig.hex()
    return {h, "0x" + h}


# ----------------------------- Constants -----------------------------

DECIMAL_CONTEXT_PREC = 80

TRANSFER_SIG = bytes.fromhex("ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef")
DEPOSIT_SIG = bytes.fromhex("e1fffcc4923d04b559f4d29a8bfc6cda04eb5b0d3c460751c2402c5c5cc9109c")
WITHDRAW_SIG = bytes.fromhex("7fcf532c15f0a6db0bd6d0e038bea71d30d808c7d98cb3bf7268a95bf5081b65")

TRANSFER_BYTES = {TRANSFER_SIG}
DEPOSIT_BYTES = {DEPOSIT_SIG}
WITHDRAW_BYTES = {WITHDRAW_SIG}

TRANSFER_STRS = _sig_to_str_set(TRANSFER_SIG)
DEPOSIT_STRS = _sig_to_str_set(DEPOSIT_SIG)
WITHDRAW_STRS = _sig_to_str_set(WITHDRAW_SIG)

EVENT_BYTES = TRANSFER_BYTES | DEPOSIT_BYTES | WITHDRAW_BYTES
EVENT_STRS = TRANSFER_STRS | DEPOSIT_STRS | WITHDRAW_STRS

ZERO_ADDR = "0x" + "0" * 40
HEX_RE = re.compile(r"^0x[0-9a-f]+$", re.I)

INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1
DEC128, DEC256 = pa.decimal128(38, 0), pa.decimal256(76, 0)

MIN_COLS = ["topic0", "topics", "data", "block_id", "log_index", "address", "tx_hash"]

BATCH_TAG_RE = re.compile(r"batch(\d{6})")

CANCELLED_BATCH_IDS: set[int] = set()

# Populated per worker (allow-list)
ALLOWED_BYTES: set[bytes] = set()
PID_DIR: Path | None = None

MapBatch = tuple[list[Path], Path, int]
MergeArgs = tuple[Path, Path]


# ----------------------------- Runtime Setup -----------------------------

decimal.getcontext().prec = DECIMAL_CONTEXT_PREC

try:
    mp.set_start_method("forkserver", force=True)
except (RuntimeError, ValueError):
    mp.set_start_method("spawn", force=True)


# ----------------------------- Compatibility Shims -----------------------------

_schema_cls = getattr(deltalake, "_internal", deltalake).Schema
if not hasattr(_schema_cls, "to_pyarrow") and hasattr(_schema_cls, "to_arrow"):

    def _to_pyarrow(self, *_, **__) -> pa.Schema:
        """Return a pyarrow Schema for older deltalake builds."""
        sch = self.to_arrow()
        return sch if isinstance(sch, pa.Schema) else pa.schema(sch)

    _schema_cls.to_pyarrow = _to_pyarrow  # type: ignore


# ----------------------------- IO Helpers -----------------------------

def load_raw_parquet(path: Path) -> pl.DataFrame:
    """Load raw Parquet with streaming when available."""
    try:
        return (
            pl.scan_parquet(path, low_memory=True)
            .select(MIN_COLS)
            .collect(engine="streaming")
        )
    except Exception as exc:
        log(f"streaming disabled -> fallback for {path.name}: {exc}")
        try:
            return pl.read_parquet(path, columns=MIN_COLS)
        except Exception as exc2:
            log(f"read_parquet failed for {path.name}: {exc2}")
            return pl.DataFrame()


def read_parquet_safe(path: Path, **kwargs) -> Optional[pl.DataFrame]:
    """Read Parquet and return None on failure."""
    try:
        return pl.read_parquet(path, **kwargs)
    except Exception:
        return None


# ----------------------------- Multiprocessing Helpers -----------------------------

def _init_worker(allowed: set[bytes], pid_dir: Path) -> None:
    """Initialize worker globals."""
    global ALLOWED_BYTES, PID_DIR
    ALLOWED_BYTES = allowed
    PID_DIR = pid_dir


def available_worker_count() -> int:
    """Return a CPU count adjusted for current system load."""
    total = int(os.getenv("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    try:
        busy = psutil.cpu_percent(interval=0.15) > 75
    except Exception:
        busy = False
    return max(1, total // 2) if busy else total


def _kill_batch_worker(batch_id: int, pid_dir: Path, grace_s: float = 1.0) -> None:
    """Terminate a stuck worker using its PID file."""
    pid_file = pid_dir / f"{batch_id}.pid"
    if not pid_file.exists():
        return

    try:
        pid_str = pid_file.read_text().strip()
        pid = int(pid_str)
    except (ValueError, OSError):
        return

    try:
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(grace_s)
        except psutil.TimeoutExpired:
            proc.kill()
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass
    except Exception as exc:
        log(f"Failed to kill PID {pid}: {exc}")

    try:
        pid_file.unlink(missing_ok=True)
    except OSError:
        pass


def run_map_batches(
    pool: mp.pool.Pool,
    batch_args: list[MapBatch],
    pid_dir: Path,
    timeout_s: int,
    sleep_s: float = 1.0,
    min_split: int = 4,
) -> int:
    """Run map batches with timeout, retries, and optional splitting."""
    next_bid = max(b for *_, b in batch_args) + 1
    pending: dict[AsyncResult, tuple[MapBatch, float]] = {}
    inflight_count: dict[int, int] = {}
    total_rows = 0

    def add_inflight(bid: int) -> None:
        inflight_count[bid] = inflight_count.get(bid, 0) + 1

    def del_inflight(bid: int) -> None:
        cnt = inflight_count.get(bid, 0) - 1
        if cnt <= 0:
            inflight_count.pop(bid, None)
        else:
            inflight_count[bid] = cnt

    def submit(arg: MapBatch) -> None:
        try:
            res = pool.apply_async(_worker_map, (arg,))
            pending[res] = (arg, time.monotonic())
            add_inflight(arg[2])
        except Exception as exc:
            log(f"Critical Pool Error submitting batch {arg[2]}: {exc}")
            raise

    for arg in batch_args:
        submit(arg)

    while inflight_count:
        finished: list[AsyncResult] = []

        for res, (arg, t0) in list(pending.items()):
            bid = arg[2]

            if res.ready():
                try:
                    total_rows += res.get()
                except Exception as exc:
                    log(f"MAP batch {bid} raised {exc!r} - resubmitting")
                    submit(arg)

                del_inflight(bid)
                finished.append(res)

                try:
                    (pid_dir / f"{bid}.pid").unlink(missing_ok=True)
                except OSError:
                    pass
                continue

            if time.monotonic() - t0 > timeout_s:
                log(f"MAP batch {bid} timed out after {timeout_s}s - killing and retrying")

                _kill_batch_worker(bid, pid_dir)

                try:
                    pool._repopulate_pool()
                except Exception:
                    pass

                del_inflight(bid)

                paths, mroot, _ = arg

                if len(paths) >= min_split:
                    mid = len(paths) // 2
                    left = (paths[:mid], mroot, next_bid)
                    next_bid += 1
                    right = (paths[mid:], mroot, next_bid)
                    next_bid += 1
                    log(f"Splitting batch {bid} into {left[2]} and {right[2]}")
                    submit(left)
                    submit(right)
                else:
                    log(f"Retrying batch {bid} (size {len(paths)}) without split")
                    new_arg = (paths, mroot, next_bid)
                    next_bid += 1
                    submit(new_arg)

                finished.append(res)

        for res in finished:
            pending.pop(res, None)

        time.sleep(sleep_s)

    return total_rows


# ----------------------------- Primitive Helpers -----------------------------

def _last20_as_hex(b: bytes | None) -> str:
    """Return the last 20 bytes as a 0x-prefixed hex address."""
    return (
        "0x" + b[-20:].hex()
        if isinstance(b, (bytes, bytearray)) and len(b) >= 20
        else ZERO_ADDR
    )


def bytes_to_int(obj: bytes | str | int | None) -> int:
    """Convert bytes/hex/int into a non-negative integer."""
    if obj in (None, "", "0x", "0X", b"", 0):
        return 0
    if isinstance(obj, (bytes, bytearray)):
        return int.from_bytes(obj, "big", signed=False)
    s = str(obj).lower()
    if s.startswith("0x"):
        s = s[2:]
    return int(s, 16) if s and re.fullmatch(r"[0-9a-f]+", s) else 0


# ----------------------------- Event Filtering -----------------------------

def filter_relevant_logs(df: pl.DataFrame) -> pl.DataFrame:
    """Filter to Transfer/Deposit/Withdrawal and add an event column."""
    if {"topic0", "address"} - set(df.columns):
        return df.slice(0, 0)

    t0 = pl.col("topic0")
    if df["topic0"].dtype == pl.Binary:
        mask = t0.is_in(list(EVENT_BYTES))
        # mask = t0.is_in(list(DEPOSIT_BYTES | WITHDRAW_BYTES))
        ev = (
            pl.when(t0.is_in(list(DEPOSIT_BYTES))).then(pl.lit("deposit"))
            .when(t0.is_in(list(WITHDRAW_BYTES))).then(pl.lit("withdraw"))
            .otherwise(pl.lit("transfer"))
            .alias("event")
        )
    else:
        mask = t0.is_in(list(EVENT_STRS))
        # mask = t0.is_in(list(DEPOSIT_STRS | WITHDRAW_STRS))
        ev = (
            pl.when(t0.is_in(list(DEPOSIT_STRS))).then(pl.lit("deposit"))
            .when(t0.is_in(list(WITHDRAW_STRS))).then(pl.lit("withdraw"))
            .otherwise(pl.lit("transfer"))
            .alias("event")
        )

    return df.filter(mask).with_columns(ev)


def apply_allow_list(df: pl.DataFrame) -> pl.DataFrame:
    """Filter to allowed token addresses when an allow-list is set."""
    return df if not ALLOWED_BYTES else df.filter(pl.col("address").is_in(ALLOWED_BYTES))


def extract_topics(df: pl.DataFrame) -> tuple[list[bytes], list[bytes], pl.DataFrame]:
    """Ensure topic1/topic2 are present and return them along with df."""
    if {"topic1", "topic2"} <= set(df.columns):
        return df["topic1"].to_list(), df["topic2"].to_list(), df
    if "topics" in df.columns:
        raw = df["topics"].to_list()
        t1, t2 = [], []
        for lst in raw:
            t1.append(
                lst[1]
                if isinstance(lst, list)
                and len(lst) > 1
                and isinstance(lst[1], (bytes, bytearray))
                else b"\x00" * 20
            )
            t2.append(
                lst[2]
                if isinstance(lst, list)
                and len(lst) > 2
                and isinstance(lst[2], (bytes, bytearray))
                else b"\x00" * 20
            )
        df = df.with_columns([pl.Series("topic1", t1), pl.Series("topic2", t2)])
        return t1, t2, df
    return [], [], df.slice(0, 0)


def decode_positive_values(df: pl.DataFrame) -> tuple[pl.DataFrame, list[int]]:
    """Decode data to ints and keep only strictly positive values."""
    values = [bytes_to_int(x) for x in df["data"].to_list()]
    mask = [v > 0 for v in values]
    if not any(mask):
        return df.slice(0, 0), []
    return df.filter(pl.Series(mask)), [v for v in values if v > 0]


def build_clean_frame(df: pl.DataFrame, positive_vals: list[int]) -> pl.DataFrame:
    """Build the canonical per-token dataframe with mint/burn handling."""
    topic1_hex = pl.col("topic1").map_elements(_last20_as_hex, return_dtype=pl.Utf8)
    topic2_hex = pl.col("topic2").map_elements(_last20_as_hex, return_dtype=pl.Utf8)
    zero = pl.lit(ZERO_ADDR, dtype=pl.Utf8)

    sender = (
        pl.when(pl.col("event") == "deposit").then(zero)
        .when(pl.col("event") == "withdraw").then(topic1_hex)
        .otherwise(topic1_hex)
        .alias("sender")
    )
    recipient = (
        pl.when(pl.col("event") == "deposit").then(topic1_hex)
        .when(pl.col("event") == "withdraw").then(zero)
        .otherwise(topic2_hex)
        .alias("recipient")
    )

    df = df.with_columns(
        [
            pl.col("address")
            .map_elements(lambda b: f"0x{bytes(b).hex()}", return_dtype=pl.Utf8)
            .alias("token_address"),
            sender,
            recipient,
            pl.col("block_id").cast(pl.Int64).alias("block"),
            pl.col("log_index").cast(pl.Int32),
            pl.Series("value", [str(v) for v in positive_vals]),
        ]
    )

    df = df.with_columns(
        pl.when(pl.col("tx_hash").is_not_null())
        .then(
            pl.col("tx_hash").map_elements(
                lambda b: f"0x{b.hex()}"
                if isinstance(b, (bytes, bytearray)) and len(b) == 32
                else "0x" + "00" * 64,
                return_dtype=pl.Utf8,
            )
        )
        .otherwise(pl.lit("0x" + "00" * 64))
        .alias("tx_hash")
    )

    return df.select(
        ["token_address", "sender", "recipient", "block", "log_index", "value", "tx_hash"]
    )


# ----------------------------- Schema Helpers (Reduce) -----------------------------

def determine_value_type(paths: Iterable[Path]) -> str:
    """Choose the narrowest Parquet type for the value column."""
    paths = list(paths)
    if not paths:
        return "int64"

    max_digits = 0

    for p in paths:
        df = read_parquet_safe(p, columns=["value"])
        if df is None or df.is_empty():
            continue

        col = df["value"].cast(pl.Utf8)
        lengths = col.str.strip_chars().str.len_chars()
        if lengths.is_empty():
            continue

        md = lengths.max()
        if md is not None:
            max_digits = max(max_digits, int(md))

    if max_digits == 0:
        return "int64"
    if max_digits < 19:
        return "int64"
    if max_digits == 19:
        true_max = None
        for p in paths:
            df = read_parquet_safe(p, columns=["value"])
            if df is None or df.is_empty():
                continue
            col = df["value"].cast(pl.Utf8)
            vals = [v for v in col.to_list() if v is not None]
            vals = [s.strip() for s in vals if s.strip()]
            if not vals:
                continue
            try:
                local_max = max(int(s) for s in vals if s.isdigit())
            except ValueError:
                true_max = None
                break
            if true_max is None or local_max > true_max:
                true_max = local_max

        if true_max is not None and true_max <= INT64_MAX:
            return "int64"

    if max_digits <= 38:
        return "decimal128"
    if max_digits <= 76:
        return "decimal256"
    return "utf8"


def build_parquet_schema(value_type: str) -> pa.Schema:
    """Build the output schema for merged token Parquet files."""
    fields: list[pa.Field] = [
        pa.field("sender", pa.utf8()),
        pa.field("recipient", pa.utf8()),
        pa.field("block", pa.int64()),
        pa.field("log_index", pa.int32()),
    ]
    if value_type == "int64":
        fields.append(pa.field("value", pa.int64()))
    elif value_type == "decimal128":
        fields.append(pa.field("value", DEC128))
    elif value_type == "decimal256":
        fields.append(pa.field("value", DEC256))
    else:
        fields.append(pa.field("value", pa.utf8()))
    fields.append(pa.field("tx_hash", pa.utf8()))
    return pa.schema(fields)


# ----------------------------- Map Write Helpers -----------------------------

def write_token_shards(df: pl.DataFrame, map_root: Path, batch_tag: str, worker_id: int) -> int:
    """Write per-token shard Parquet files and return row count."""
    total = 0
    pid = os.getpid()
    try:
        tokens = df["token_address"].unique().to_list()
    except Exception:
        return 0
    for token in tokens:
        subset = df.filter(pl.col("token_address") == token)
        if subset.is_empty():
            continue
        token_dir = map_root / f"token_address={token}"
        token_dir.mkdir(parents=True, exist_ok=True)
        shard_path = token_dir / f"shard-{worker_id:03d}-{pid}-{batch_tag}.parquet"
        try:
            subset.write_parquet(shard_path, compression="zstd", use_pyarrow=True)
            # log(f"[W{worker_id}] {batch_tag} wrote {subset.height:,} rows -> {shard_path.name}")
            total += subset.height
        except Exception as exc:
            log(f"[W{worker_id}] write shard {token}: {exc}")
    return total


# ----------------------------- Map Worker -----------------------------

def _worker_map(args: MapBatch) -> int:
    """Process a batch of raw files and write per-token shards."""
    raw_paths, map_root, batch_id = args
    wid = batch_id % 10_000
    tag = f"batch{batch_id:06d}"

    if PID_DIR:
        try:
            (PID_DIR / f"{batch_id}.pid").write_text(str(os.getpid()))
        except Exception:
            pass

    log(f"[W{wid}] MAP start {tag} (files={len(raw_paths)})")

    try:
        frames: list[pl.DataFrame] = []
        rows_in = 0
        for path in raw_paths:
            df_raw = load_raw_parquet(path)
            if df_raw.is_empty():
                continue
            df_relevant = filter_relevant_logs(df_raw)
            if df_relevant.is_empty():
                continue
            df_allowed = apply_allow_list(df_relevant)
            if df_allowed.is_empty():
                continue
            _t1, _t2, df_topics = extract_topics(df_allowed)
            if df_topics.is_empty():
                continue
            df_vals, positive = decode_positive_values(df_topics)
            if df_vals.is_empty():
                continue
            df_clean = build_clean_frame(df_vals, positive)
            frames.append(df_clean)
            rows_in += df_clean.height

        if not frames:
            log(f"[W{wid}] MAP done {tag} (empty)")
            return 0

        out_rows = write_token_shards(pl.concat(frames), map_root, tag, wid)
        log(f"[W{wid}] MAP done {tag} -> {out_rows:,} rows (from {rows_in:,})")
        return out_rows
    finally:
        if PID_DIR:
            try:
                (PID_DIR / f"{batch_id}.pid").unlink(missing_ok=True)
            except Exception:
                pass


# ----------------------------- Reduce Helpers -----------------------------

def process_shard_to_table(
    shard: Path,
    schema: pa.Schema,
    value_type: str,
) -> Optional[pa.Table]:
    """Convert a shard Parquet file into a pyarrow Table."""
    df = read_parquet_safe(shard)
    if df is None:
        return None
    array = pa.array
    if value_type == "int64":
        value_arr = array(df["value"].cast(int).to_numpy().astype(np.int64), pa.int64())
    elif value_type == "decimal128":
        value_arr = array([int(x) for x in df["value"].to_list()], DEC128)
    elif value_type == "decimal256":
        value_arr = array([int(x) for x in df["value"].to_list()], DEC256)
    else:
        value_arr = array(df["value"].to_list(), pa.utf8())
    return pa.table(
        dict(
            sender=array(df["sender"].to_list(), pa.utf8()),
            recipient=array(df["recipient"].to_list(), pa.utf8()),
            block=array(df["block"].to_numpy().astype(np.int64), pa.int64()),
            log_index=array(df["log_index"].to_numpy().astype(np.int32), pa.int32()),
            value=value_arr,
            tx_hash=array(df["tx_hash"].to_list(), pa.utf8()),
        ),
        schema=schema,
    )


def write_token_parquet(shards: list[Path], final_root: Path, token: str) -> int:
    """Merge shards for a token into a single Parquet file."""
    value_type = determine_value_type(shards)
    schema = build_parquet_schema(value_type)
    out_dir = final_root / f"token_address={token}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{token}.parquet"

    try:
        writer = pq.ParquetWriter(
            out_file,
            schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics="strip",
        )
    except TypeError:
        writer = pq.ParquetWriter(
            out_file,
            schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )

    total = 0
    for sh in shards:
        tbl = process_shard_to_table(sh, schema, value_type)
        if tbl is None:
            continue
        writer.write_table(tbl)
        total += tbl.num_rows
    writer.close()
    return total


def merge_token_shards(token_dir: Path, final_root: Path) -> int:
    """Merge all shards for a token directory into one file."""
    all_shards = sorted(token_dir.glob("shard-*.parquet"))

    shards: list[Path] = []
    for sh in all_shards:
        bid = _shard_batch_id(sh)
        if bid is not None and bid in CANCELLED_BATCH_IDS:
            log(f"skipping cancelled batch {bid} shard {sh.name}")
            continue
        shards.append(sh)

    if not shards:
        token = token_dir.name.split("=", 1)[1]
        log(f"no shards to merge for token={token} after cancelling bad batches")
        return 0

    token = token_dir.name.split("=", 1)[1]
    t0 = time.perf_counter()
    log(f"MERGE token={token} shards={len(shards)} (filtered from {len(all_shards)})")
    rows = write_token_parquet(shards, final_root, token)
    log(f"MERGED token={token} rows={rows:,} t={time.perf_counter() - t0:.1f}s")
    return rows


def _merge_token_shards_entry(args: MergeArgs) -> int:
    """Entry point for pool.imap_unordered."""
    return merge_token_shards(*args)


# ----------------------------- File Discovery -----------------------------

def list_raw_files(src_root: Path) -> list[Path]:
    """List raw Parquet files from a Delta snapshot or a plain directory."""
    delta_log = src_root / "_delta_log"

    if delta_log.exists():
        log(f"Detected Delta Lake at {src_root}")
        table = DeltaTable(str(src_root))
        files = [
            src_root / rel
            for rel in table.files()
            if str(rel).lower().endswith(".parquet")
        ]
        files.sort()
        log(f"Active Parquet files from Delta snapshot: {len(files):,}")
        return files

    log(f"Detected plain Parquet folder at {src_root}")
    files = [p for p in src_root.rglob("*.parquet") if "_delta_log" not in p.parts]
    files.sort()
    log(f"Raw Parquet files found via rglob(): {len(files):,}")
    return files


def _shard_batch_id(path: Path) -> Optional[int]:
    """Extract batch id from shard filename, or None if absent."""
    m = BATCH_TAG_RE.search(path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Raw logs -> per-token Parquet (with wrapper support)")
    p.add_argument("--src-raw", required=True, help="Root dir with raw log/*.parquet")
    p.add_argument("--map-root", required=True, help="TMP dir for intermediate shards")
    p.add_argument("--final-root", required=True, help="Output dir for final per-token Parquet")
    p.add_argument(
        "--token-csv",
        required=True,
        help="Token status CSV (requires contract_address,is_erc_20,price_data_status)",
    )
    p.add_argument("--batch", type=int, default=48, help="Raw files per MAP batch (default 48)")
    p.add_argument(
        "--map-timeout", type=int, default=3600, help="Seconds per batch before retry (default 3600)"
    )
    return p.parse_args()


def main() -> int:
    """Run the map-reduce pipeline."""
    args = parse_args()
    src_root = Path(args.src_raw).resolve()
    map_root = Path(args.map_root).resolve()
    final_root = Path(args.final_root).resolve()
    token_csv = Path(args.token_csv).resolve()

    allow_strs = load_token_allow_list(str(token_csv), log_fn=log)
    allow_list = {bytes.fromhex(a[2:]) for a in allow_strs}

    if map_root.exists():
        shutil.rmtree(map_root, ignore_errors=True)
    map_root.mkdir(parents=True, exist_ok=True)

    pid_dir = map_root / "pids"
    pid_dir.mkdir(exist_ok=True)

    raw_files = list_raw_files(src_root)

    batch_size = max(1, args.batch)
    batches = [raw_files[i : i + batch_size] for i in range(0, len(raw_files), batch_size)]
    map_args: list[MapBatch] = [(batch, map_root, idx) for idx, batch in enumerate(batches)]

    # available_worker_count()
    worker_count = 12
    timeout_s = max(30, args.map_timeout)

    log(f"MAP: {len(batches):,} batches (batch size={batch_size}, timeout={timeout_s}s)")
    t0 = time.perf_counter()

    with mp.Pool(
        worker_count, initializer=_init_worker, initargs=(allow_list, pid_dir), maxtasksperchild=20
    ) as pool:
        total_rows = run_map_batches(pool, map_args, pid_dir, timeout_s, sleep_s=1)

    log(f"MAP complete -> {total_rows:,} rows in {time.perf_counter() - t0:.1f}s\n")

    token_dirs = [d for d in map_root.glob("token_address=*") if d.is_dir()]
    if not token_dirs:
        log("nothing to merge - no shards produced")
        return 0

    # available_worker_count()
    worker_count = 8
    total_tokens = len(token_dirs)
    progress_mod = max(1, total_tokens // 100)

    log("REDUCE phase...")
    t1 = time.perf_counter()
    reduce_args = ((td, final_root) for td in token_dirs)
    total_rows = 0
    done_tokens = 0
    with mp.Pool(worker_count, maxtasksperchild=200) as pool:
        for rows in pool.imap_unordered(_merge_token_shards_entry, reduce_args, chunksize=1):
            total_rows += rows
            done_tokens += 1
            if done_tokens % progress_mod == 0 or done_tokens == total_tokens:
                pct = 100 * done_tokens / total_tokens
                log(f"   progress: {done_tokens:,}/{total_tokens:,} tokens ({pct:5.1f}%)")

    log(
        f"REDUCE complete -> {total_rows:,} rows in {time.perf_counter() - t1:.1f}s "
        f"(out {final_root})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
