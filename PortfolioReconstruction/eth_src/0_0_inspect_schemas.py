#!/usr/bin/env python3
"""
0_0_inspect_schemas.py

Purpose
-------
Step 0 (MANDATORY) of the ETH native transfer reconstruction: inspect the actual Parquet schemas of trace/,
transaction/ and block/ before any logic runs, detect column names,
value storage formats, the partition layout and the available fallbacks,
and write everything to decisions.json for the other scripts to consume.

What it does
------------
- Globs partitions per source, DESCRIBEs first/middle/last partitions,
  logs columns + dtypes + 5 sample rows.
- Detects: partition-key naming (restartable output mirrors it), value
  storage type (decimal string / hex string / numeric / binary; never
  float), receipt fields on transaction/, base_fee/miner/withdrawals on
  block/, status-vs-error revert regime on trace/, genesis/reward traces.
- Records every branch decision and prints the resulting INCLUDE_* states
  with their bias implications (first-run checklist item 2).

Outputs
-------
- <output_base>/decisions.json

CLI
---
  $ python 0_0_inspect_schemas.py --input-base /path/to/ethereum_dump \
        --output-base /path/to/eth_transfers
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    BLOCK_CANDIDATES,
    Config,
    MERGE_BLOCK,
    SHAPELLA_BLOCK,
    TRACE_CANDIDATES,
    TX_CANDIDATES,
    WITHDRAWAL_FIELD_CANDIDATES,
    add_config_args,
    partition_files,
    setup_logging,
)

log = None  # set in main()


# ----------------------------- Partition discovery -----------------------------

def _sort_key_from_name(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[0]) if nums else -1


def _delta_versions(log_dir: Path) -> tuple[set[int], set[int]]:
    """Commit and checkpoint versions present in a _delta_log directory."""
    commits: set[int] = set()
    checkpoints: set[int] = set()
    for f in log_dir.iterdir():
        m = re.match(r"^(\d{20})\.json$", f.name)
        if m:
            commits.add(int(m.group(1)))
        m = re.match(r"^(\d{20})\.checkpoint.*\.parquet$", f.name)
        if m:
            checkpoints.add(int(m.group(1)))
    return commits, checkpoints


def open_delta_pinned(base: Path):
    """Open a Delta table whose latest log segment is non-contiguous by
    pinning to the highest version reconstructable from a checkpoint plus an
    unbroken chain of commit files (cf. traces_delta_version in the repo's
    pipeline config). Returns (DeltaTable, version) or raises."""
    from deltalake import DeltaTable

    commits, checkpoints = _delta_versions(base / "_delta_log")
    last_exc: Exception = RuntimeError(f"no usable checkpoint in {base}/_delta_log")
    for c in sorted(checkpoints, reverse=True):
        v = c
        while v + 1 in commits:
            v += 1
        try:
            return DeltaTable(str(base), version=v), v
        except Exception as exc:
            last_exc = exc
    raise last_exc


def _partitions_from_delta(base: Path, table) -> list[dict]:
    by_dir: dict[str, list[str]] = {}
    for rel in table.files():
        if not str(rel).lower().endswith(".parquet"):
            continue
        abs_p = base / rel
        by_dir.setdefault(str(abs_p.parent), []).append(str(abs_p))
    parts = [
        {"path": d, "files": sorted(fs), "lo": None, "hi": None}
        for d, fs in by_dir.items()
    ]
    parts.sort(key=lambda p: _sort_key_from_name(Path(p["path"]).name))
    return parts


def discover_partitions(base: Path, warn) -> dict:
    """Detect partition layout: Delta Lake table, hive/plain dirs, or flat files.

    For Delta tables the ACTIVE file set comes from the Delta log (raw
    globbing could pick up stale files superseded by compaction/rewrites).
    Block bounds (lo/hi) are NOT derived here — partition dir names like
    partition=1237 don't encode block ranges; annotate_partition_ranges()
    fills them from parquet footer statistics.

    Returns {"style": ..., "partitions": [{"path", "files"?, "lo", "hi"}, ...]}.
    """
    delta_fallback_error = None
    if (base / "_delta_log").exists():
        env_v = os.environ.get(f"DELTA_VERSION_{base.name.upper()}")
        try:
            from deltalake import DeltaTable

            if env_v is not None:
                table, style = DeltaTable(str(base), version=int(env_v)), "delta-pinned"
                log.info("%s: Delta version pinned to %s via DELTA_VERSION_%s",
                         base.name, env_v, base.name.upper())
            else:
                table, style = DeltaTable(str(base)), "delta"
            # Validate log contiguity ourselves: some delta-rs versions raise
            # "non-contiguous log segment" here, but others SILENTLY skip the
            # missing commits and drop their files from the active set.
            v = table.version()
            commits, checkpoints = _delta_versions(base / "_delta_log")
            cp = max((c for c in checkpoints if c <= v), default=-1)
            missing = set(range(cp + 1, v + 1)) - commits
            if missing:
                raise RuntimeError(
                    f"log segment non-contiguous: commits {sorted(missing)[:5]} "
                    f"missing between checkpoint {cp} and HEAD {v}"
                )
            parts = _partitions_from_delta(base, table)
            log.info("%s: Delta Lake table, %d active files in %d partitions",
                     base.name, sum(len(p["files"]) for p in parts), len(parts))
            return {"style": style, "partitions": parts,
                    "delta_version": int(env_v) if env_v is not None else None}
        except Exception as exc:
            # Do NOT auto-pin yet: whether raw globbing (complete) or a pinned
            # older version (dedup'd but possibly missing the newest commits)
            # is the right call depends on whether stale files actually exist.
            # build_decisions() answers that with an empirical duplicate check.
            delta_fallback_error = str(exc)
            warn(f"{base.name}: _delta_log present but Delta read failed ({exc}); "
                 "using raw globbing for now — a duplicate check decides below "
                 "whether stale superseded files are a real risk")

    subdirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name != "_delta_log")
    if subdirs:
        style = "hive" if all("=" in d.name for d in subdirs) else "plain-dirs"
        parts = [{"path": str(d), "lo": None, "hi": None} for d in subdirs]
    else:
        style = "flat-files"
        parts = [{"path": str(f), "lo": None, "hi": None}
                 for f in sorted(base.glob("*.parquet"))]
    parts.sort(key=lambda p: _sort_key_from_name(Path(p["path"]).name))
    return {"style": style, "partitions": parts,
            "delta_fallback_error": delta_fallback_error}


def annotate_partition_ranges(src: dict, block_col: str, warn) -> None:
    """Fill lo/hi per partition from parquet footer min/max statistics of the
    block column (threaded — GPFS metadata reads are latency-bound). Partitions
    whose files lack statistics fall back to a DuckDB MIN/MAX scan."""
    import concurrent.futures

    parts = src["partitions"]

    def bounds(p: dict):
        lo = hi = None
        for f in partition_files(p):
            md = pq.ParquetFile(f).metadata
            names = [md.schema.column(i).name for i in range(md.num_columns)]
            ci = names.index(block_col)
            for rg in range(md.num_row_groups):
                st = md.row_group(rg).column(ci).statistics
                if st is None or not st.has_min_max:
                    raise ValueError(f"no min/max stats in {f}")
                lo = st.min if lo is None else min(lo, st.min)
                hi = st.max if hi is None else max(hi, st.max)
        return lo, hi

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(bounds, p): p for p in parts}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            p = futures[fut]
            try:
                p["lo"], p["hi"] = fut.result()
            except Exception:
                failed.append(p)
            done += 1
            if done % 500 == 0:
                log.info("  block-range stats: %d/%d partitions", done, len(parts))

    if failed:
        log.info("  %d partitions lack footer stats -> DuckDB MIN/MAX scan", len(failed))
        con = duckdb.connect()
        for p in failed:
            try:
                p["lo"], p["hi"] = con.execute(
                    f"SELECT MIN({block_col}), MAX({block_col}) FROM read_parquet(?)",
                    [partition_files(p)],
                ).fetchone()
            except Exception as exc:
                warn(f"could not determine block range of {p['path']}: {exc} — "
                     "partition will be scanned unconditionally (no pruning)")
        con.close()
    parts.sort(key=lambda p: (p["lo"] is None, p["lo"] if p["lo"] is not None else -1))


def first_parquet_file(p: dict) -> Optional[Path]:
    files = partition_files(p)
    return Path(files[0]) if files else None


# ----------------------------- Schema helpers -----------------------------

def resolve_columns(schema_names: list[str], candidates: dict[str, list[str]]) -> dict:
    """Map canonical names -> actual column names (None if absent)."""
    lower = {n.lower(): n for n in schema_names}
    return {
        canon: next((lower[c] for c in cands if c in lower), None)
        for canon, cands in candidates.items()
    }


def arrow_type_of(schema, col: Optional[str]):
    if col is None:
        return None
    return schema.field(col).type


def classify_value_format(atype, sample: object) -> str:
    """Classify how a numeric quantity is stored. Never float."""
    if atype is None:
        return "missing"
    s = str(atype)
    if "decimal" in s or s.startswith(("int", "uint")):
        return "numeric"
    if s in ("float", "double") or s.startswith(("float", "double", "halffloat")):
        return "FLOAT_FORBIDDEN"
    if s in ("binary", "large_binary") or "fixed_size_binary" in s:
        return "binary"
    if s in ("string", "large_string", "utf8", "large_utf8"):
        if isinstance(sample, str) and sample.strip().lower().startswith("0x"):
            return "hex_string"
        return "dec_string"
    return f"unknown({s})"


def classify_addr_format(atype) -> str:
    if atype is None:
        return "missing"
    s = str(atype)
    if "binary" in s:
        return "binary"
    return "string"


def sample_value(con: duckdb.DuckDBPyConnection, f: Path, col: Optional[str]):
    if col is None:
        return None
    try:
        row = con.execute(
            f'SELECT "{col}" FROM read_parquet(?) WHERE "{col}" IS NOT NULL LIMIT 1', [str(f)]
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def describe_partition(con, name: str, f: Path) -> None:
    """Log columns + dtypes + 5 sample rows for one partition file."""
    log.info("--- %s: schema of %s", name, f)
    for cname, ctype in con.execute(
        "SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM read_parquet(?))",
        [str(f)],
    ).fetchall():
        log.info("    %-32s %s", cname, ctype)
    try:
        rows = con.execute("SELECT * FROM read_parquet(?) LIMIT 5", [str(f)]).fetchdf()
        for line in rows.to_string(max_colwidth=40).splitlines():
            log.info("    | %s", line)
    except Exception as exc:
        log.warning("    sample rows failed: %s", exc)


# ----------------------------- Per-source inspection -----------------------------

def inspect_source(con, base: Path, name: str, candidates: dict, warn) -> dict:
    if not base.exists():
        raise FileNotFoundError(f"{name}/ not found under input base: {base}")
    layout = discover_partitions(base, warn)
    parts = layout["partitions"]
    if not parts:
        raise RuntimeError(f"no partitions found in {base}")
    log.info("%s/: %d partitions, style=%s", name, len(parts), layout["style"])

    probe_idx = sorted({0, len(parts) // 2, len(parts) - 1})
    probe_files = []
    for i in probe_idx:
        f = first_parquet_file(parts[i])
        if f:
            probe_files.append((i, f))
    if not probe_files:
        raise RuntimeError(f"no parquet files found in probed partitions of {base}")

    for i, f in probe_files:
        describe_partition(con, f"{name}[partition {i}]", f)

    # Merge schemas across probes (schema drift between old/new partitions).
    schema = pq.read_schema(probe_files[0][1])
    names = list(schema.names)
    for _, f in probe_files[1:]:
        for extra in pq.read_schema(f).names:
            if extra not in names:
                names.append(extra)
    cols = resolve_columns(names, candidates)
    for canon, actual in cols.items():
        log.info("%s: column %-14s -> %s", name, canon, actual or "MISSING")
    return {
        "base": str(base),
        "partition_style": layout["style"],
        "partitions": parts,
        "delta_fallback_error": layout.get("delta_fallback_error"),
        "delta_version": layout.get("delta_version"),
        "cols": cols,
        "schema_names": names,
        "_probe_files": probe_files,  # stripped before writing decisions.json
    }


# ----------------------------- Stale-file / duplicate resolution -----------------------------

def dedup_sample_check(src: dict, name: str, key_col: Optional[str]) -> Optional[dict]:
    """Empirically test for stale superseded files: within sampled multi-file
    partitions, a unique key (trace_id / tx_hash / block number) must not
    repeat. Returns {"partitions_checked", "duplicate_rows"} or None if no
    key/no multi-file partitions."""
    if key_col is None:
        return None
    multi = [p for p in src["partitions"] if len(partition_files(p)) > 1]
    if not multi and src.get("partition_style") == "flat-files" and len(src["partitions"]) > 1:
        # flat layout: stale files would sit next to each other in the base
        # dir as separate single-file "partitions" — check them as one unit
        multi = [{"path": src["base"],
                  "files": [f for p in src["partitions"] for f in partition_files(p)]}]
    if not multi:
        log.info("%s: no multi-file partitions -> stale duplicates structurally "
                 "unlikely", name)
        return {"partitions_checked": 0, "duplicate_rows": 0}
    sample = {id(p): p for p in (multi[0], multi[len(multi) // 2], multi[-1])}.values()
    con = duckdb.connect()
    dups = checked = 0
    for p in sample:
        n, d = con.execute(
            f"SELECT count(*), count(DISTINCT {key_col}) FROM read_parquet(?)",
            [partition_files(p)],
        ).fetchone()
        dups += n - d
        checked += 1
        log.info("%s: dup check %s: %d rows, %d distinct %s",
                 name, Path(p["path"]).name, n, d, key_col)
    con.close()
    return {"partitions_checked": checked, "duplicate_rows": dups}


def resolve_delta_fallback(src: dict, name: str, key_col: Optional[str], warn) -> None:
    """The Delta log could not be read at HEAD (non-contiguous segment) and
    discovery fell back to raw globbing. Decide what is actually safe:

    - duplicate check clean -> keep raw globbing (append-only dump: complete
      AND correct; the log gap is irrelevant).
    - duplicates found -> pin to the highest checkpoint-reconstructable Delta
      version (dedup'd, but commits after it are excluded -> possibly missing
      the newest blocks); manual override: DELTA_VERSION_<SOURCE>.
    """
    if not src.get("delta_fallback_error"):
        return
    check = dedup_sample_check(src, name, key_col)
    src["dedup_check"] = check
    if check is None:
        warn(f"{name}: cannot run duplicate check (no unique key column) — "
             "raw globbing kept; if the Delta table was ever rewritten, rows "
             f"are double-counted. Consider DELTA_VERSION_{name.upper()}.")
        return
    if check["duplicate_rows"] == 0:
        log.info("%s: duplicate check CLEAN (%d partitions sampled) -> raw "
                 "globbing kept: dump looks append-only, so it is complete and "
                 "the unreadable Delta HEAD does not matter", name,
                 check["partitions_checked"])
        return
    warn(f"{name}: {check['duplicate_rows']} duplicate rows in sampled "
         "partitions — stale superseded files ARE present; pinning Delta version")
    try:
        table, v = open_delta_pinned(Path(src["base"]))
    except Exception as exc:
        warn(f"{name}: version pinning failed too ({exc}). Raw globbing WILL "
             f"double-count. Fix the _delta_log or set DELTA_VERSION_{name.upper()}.")
        return
    raw_count = sum(len(partition_files(p)) for p in src["partitions"])
    parts = _partitions_from_delta(Path(src["base"]), table)
    active_count = sum(len(p["files"]) for p in parts)
    src["partitions"] = parts
    src["partition_style"] = "delta-pinned"
    src["delta_version"] = v
    warn(f"{name}: pinned to Delta version {v}: {active_count} active files "
         f"({raw_count} on disk; the difference is stale files and/or commits "
         "newer than the pin — blocks after the pinned commit are EXCLUDED, "
         "check the resulting block coverage below)")


# ----------------------------- Main detection logic -----------------------------

def build_decisions(cfg: Config) -> dict:
    con = duckdb.connect()
    decisions: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_base": str(cfg.input_base),
        "warnings": [],
    }

    def warn(msg: str) -> None:
        log.warning(msg)
        decisions["warnings"].append(msg)

    trace = inspect_source(con, cfg.input_base / "trace", "trace", TRACE_CANDIDATES, warn)
    tx = inspect_source(con, cfg.input_base / "transaction", "transaction", TX_CANDIDATES, warn)
    block = inspect_source(con, cfg.input_base / "block", "block", BLOCK_CANDIDATES, warn)

    # If the Delta HEAD was unreadable, decide raw-glob vs pinned version
    # empirically (unique keys: trace_id / tx hash / block number).
    resolve_delta_fallback(
        trace, "trace",
        "trace_id" if "trace_id" in trace["schema_names"] else None,
        warn,
    )
    resolve_delta_fallback(tx, "transaction", tx["cols"]["hash"], warn)
    resolve_delta_fallback(block, "block", block["cols"]["number"], warn)

    # Per-partition block bounds from parquet footer stats (partition dir
    # names like partition=1237 do not encode block ranges).
    for src, sname in ((trace, "trace"), (tx, "transaction"), (block, "block")):
        bcol = src["cols"]["block" if sname != "block" else "number"]
        if bcol is None:
            raise RuntimeError(f"{sname}: no block-number column found")
        log.info("%s: computing per-partition block ranges from footer stats ...", sname)
        annotate_partition_ranges(src, bcol, warn)
        known = [p for p in src["partitions"] if p["lo"] is not None]
        if known:
            log.info("%s: block coverage %d - %d across %d/%d partitions", sname,
                     min(p["lo"] for p in known), max(p["hi"] for p in known),
                     len(known), len(src["partitions"]))

    # ---------- trace/ ----------
    t_first = trace["_probe_files"][0][1]
    t_mid = trace["_probe_files"][len(trace["_probe_files"]) // 2][1]
    t_schema = pq.read_schema(t_mid)

    def t_type(canon):
        col = trace["cols"].get(canon)
        return arrow_type_of(t_schema, col) if col in t_schema.names else None

    v_fmt = classify_value_format(
        t_type("value"), sample_value(con, t_mid, trace["cols"]["value"])
    )
    if v_fmt == "FLOAT_FORBIDDEN":
        raise RuntimeError(
            "trace.value is stored as float/double — precision loss guaranteed. "
            "Re-extract with an exact type."
        )
    trace["value_format"] = v_fmt
    trace["addr_format"] = classify_addr_format(t_type("from") or t_type("to"))
    trace["hash_format"] = classify_addr_format(t_type("tx_hash"))

    ta_col = trace["cols"]["trace_address"]
    if ta_col is None:
        trace["trace_address_kind"] = "missing"
        warn("trace.trace_address missing: ordering degraded; error-prefix fallback impossible")
    else:
        ta_t = str(arrow_type_of(t_schema, ta_col)) if ta_col in t_schema.names else "string"
        trace["trace_address_kind"] = "list" if "list" in ta_t else "string"

    # Revert regime
    if trace["cols"]["status"]:
        trace["revert_mode"] = "status"
        log.info("revert handling: trace.status present -> filter status=1 "
                 "(ethereum-etl ancestor-revert propagation assumed)")
    elif trace["cols"]["error"] and trace["trace_address_kind"] != "missing":
        trace["revert_mode"] = "error_prefix"
        warn("revert handling: no status column -> using error prefix-propagation fallback")
    else:
        trace["revert_mode"] = "none"
        warn("revert handling: NEITHER status NOR usable error column -> reverted "
             "transfers WILL leak into the ledger. Fix the extraction.")

    # Genesis / reward traces live in the first partition (block 0 / pre-Merge).
    tt_col = trace["cols"]["trace_type"]
    try:
        counts = dict(
            con.execute(
                f"SELECT lower({tt_col}) tt, count(*) FROM read_parquet(?) GROUP BY 1",
                [partition_files(trace["partitions"][0])],
            ).fetchall()
        )
    except Exception as exc:
        warn(f"trace_type scan of first partition failed: {exc}")
        counts = {}
    trace["has_genesis_traces"] = counts.get("genesis", 0) > 0
    trace["has_reward_traces"] = counts.get("reward", 0) > 0
    log.info("first trace partition trace_type counts: %s", counts)

    # ---------- transaction/ (receipts, the reconstruction rules) ----------
    x_mid = tx["_probe_files"][len(tx["_probe_files"]) // 2][1]
    x_schema = pq.read_schema(x_mid)

    def x_type(canon):
        col = tx["cols"].get(canon)
        return arrow_type_of(x_schema, col) if col in x_schema.names else None

    tx["addr_format"] = classify_addr_format(x_type("from"))
    tx["hash_format"] = classify_addr_format(x_type("hash"))
    gf = classify_value_format(x_type("gas_used"), sample_value(con, x_mid, tx["cols"]["gas_used"]))
    pf = classify_value_format(
        x_type("effective_gas_price") or x_type("gas_price"),
        sample_value(
            con, x_mid, tx["cols"]["effective_gas_price"] or tx["cols"]["gas_price"]
        ),
    )
    if "FLOAT" in gf or "FLOAT" in pf:
        raise RuntimeError("gas fields stored as float — forbidden")
    tx["gas_used_format"] = gf
    tx["gas_price_format"] = pf
    tx["receipts_available"] = bool(
        tx["cols"]["gas_used"]
        and (tx["cols"]["effective_gas_price"] or tx["cols"]["gas_price"])
    )

    # ---------- block/ (base fee, miner, withdrawals) ----------
    b_last = block["_probe_files"][-1][1]
    b_schema = pq.read_schema(b_last)

    def b_type(canon):
        col = block["cols"].get(canon)
        return arrow_type_of(b_schema, col) if col in b_schema.names else None

    block["miner_format"] = classify_addr_format(b_type("miner"))
    block["base_fee_format"] = classify_value_format(
        b_type("base_fee"), sample_value(con, b_last, block["cols"]["base_fee"])
    )
    if "FLOAT" in block["base_fee_format"]:
        raise RuntimeError("block.base_fee_per_gas stored as float — forbidden")

    w_col = block["cols"]["withdrawals"]
    block["withdrawals_mode"] = "absent"
    block["withdrawal_fields"] = {}
    block["withdrawal_amount_format"] = "missing"
    if w_col and w_col in b_schema.names:
        w_type = b_schema.field(w_col).type
        w_type_str = str(w_type)
        if "list" in w_type_str and "struct" in w_type_str:
            block["withdrawals_mode"] = "list_struct"
            inner = w_type.value_type
            field_names = [inner.field(i).name for i in range(inner.num_fields)]
            block["withdrawal_fields"] = resolve_columns(
                field_names, WITHDRAWAL_FIELD_CANDIDATES
            )
            amt = block["withdrawal_fields"].get("amount")
            amt_type = inner.field(amt).type if amt else None
            w_addr = block["withdrawal_fields"].get("address")
            block["withdrawal_address_format"] = classify_addr_format(
                inner.field(w_addr).type if w_addr else None
            )
            sample_w = sample_value(con, b_last, w_col)
            amt_sample = None
            if sample_w:
                try:
                    amt_sample = sample_w[0].get(amt)
                except Exception:
                    amt_sample = None
            block["withdrawal_amount_format"] = classify_value_format(amt_type, amt_sample)
        elif "string" in w_type_str or "utf8" in w_type_str:
            block["withdrawals_mode"] = "json_string"
            block["withdrawal_fields"] = {"address": "address", "amount": "amount",
                                          "index": "index"}
            block["withdrawal_amount_format"] = "dec_string"
        else:
            warn(f"block.withdrawals has unhandled type {w_type_str}; treating as absent")

    # ---------- global block range (from footer stats) ----------
    known = [p for p in block["partitions"] if p["lo"] is not None and p["hi"] is not None]
    if not known:
        raise RuntimeError("could not determine any block ranges for block/ partitions")
    block_range = [min(p["lo"] for p in known), max(p["hi"] for p in known)]
    log.info("global block range: %s", block_range)
    decisions["block_range"] = block_range

    # ---------- resulting flags + bias implications ----------
    include_gas = cfg.include_gas
    if include_gas and not tx["receipts_available"]:
        include_gas = False
        warn(
            "RECEIPT FIELDS MISSING in transaction/ (gas_used / effective_gas_price): "
            "INCLUDE_GAS forced False. Sender balances will be OVERSTATED by cumulative "
            "gas spend. Do NOT approximate from trace gas; re-extract receipts "
            "(cryo transactions with receipts, or ethereum-etl export_receipts). Spec 6."
        )
    include_withdrawals = cfg.include_withdrawals
    if include_withdrawals and block["withdrawals_mode"] == "absent":
        include_withdrawals = False
        if block_range[1] is None or block_range[1] >= SHAPELLA_BLOCK:
            warn(
                "WITHDRAWALS NOT IN BLOCK DUMP: INCLUDE_WITHDRAWALS forced False. "
                "This is the single most important gap: validator/exchange/staking "
                "balances will be badly UNDERSTATED from Apr 2023 (block "
                f"{SHAPELLA_BLOCK}) onward. Source withdrawals separately "
                "(cryo blocks with withdrawals, or the execution API). Spec 7."
            )
        else:
            log.info("withdrawals absent but block range ends pre-Shapella: no impact")
    if not trace["has_genesis_traces"]:
        msg = ("no genesis traces in first trace partition: genesis alloc must be "
               "injected from a genesis.json (--genesis-alloc-json) or early whale "
               "balances will be understated (RQ-A). Spec 7.")
        warn(msg)
    if block_range[1] is not None and block_range[1] > MERGE_BLOCK and not trace[
        "has_reward_traces"
    ]:
        warn("no reward traces found in first partition although range is pre-Merge-"
             "inclusive: pre-Merge issuance would be missing — check the dump")

    decisions["flags"] = {
        "include_gas": include_gas,
        "include_withdrawals": include_withdrawals,
        "include_rewards": cfg.include_rewards,
        "include_genesis": cfg.include_genesis,
    }

    for src in (trace, tx, block):
        src.pop("_probe_files", None)
    decisions["trace"] = trace
    decisions["transaction"] = tx
    decisions["block"] = block

    # Is this ethereum-etl format?
    is_etl = bool(trace["cols"]["status"] and trace["cols"]["tx_index"])
    decisions["looks_like_ethereum_etl"] = is_etl
    log.info("ethereum-etl format (status propagation + transaction_index): %s", is_etl)
    return decisions


def main() -> int:
    global log
    parser = argparse.ArgumentParser(description="Step 0: inspect dump schemas -> decisions.json")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = Config.from_args(args)
    log = setup_logging(cfg.logs_dir, "inspect_schemas")
    log.info("config:\n%s", cfg.describe())

    decisions = build_decisions(cfg)

    cfg.decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.decisions_path, "w") as fh:
        json.dump(decisions, fh, indent=2)
    log.info("wrote %s", cfg.decisions_path)

    # ---------- human-readable report (first-run checklist item 2) ----------
    f = decisions["flags"]
    log.info("=" * 70)
    log.info("STEP 0 REPORT")
    log.info("  block range          : %s", decisions["block_range"])
    for sname in ("trace", "transaction", "block"):
        s = decisions[sname]
        extra = ""
        if s.get("delta_version") is not None:
            extra = f" (pinned Delta version {s['delta_version']})"
        elif s.get("dedup_check") is not None:
            dc = s["dedup_check"]
            extra = (f" (dup check: {dc['duplicate_rows']} dups in "
                     f"{dc['partitions_checked']} sampled partitions)")
        log.info("  %-11s layout   : %s%s", sname, s["partition_style"], extra)
    log.info("  ethereum-etl format  : %s", decisions["looks_like_ethereum_etl"])
    log.info("  revert handling      : %s", decisions["trace"]["revert_mode"])
    log.info("  trace value format   : %s", decisions["trace"]["value_format"])
    log.info("  INCLUDE_GAS          : %s%s", f["include_gas"],
             "" if f["include_gas"] else "  (bias: sender balances OVERstated)")
    log.info("  INCLUDE_WITHDRAWALS  : %s%s", f["include_withdrawals"],
             "" if f["include_withdrawals"] else "  (bias: staking balances UNDERstated post-Apr-2023)")
    log.info("  genesis traces       : %s", decisions["trace"]["has_genesis_traces"])
    log.info("  reward traces        : %s", decisions["trace"]["has_reward_traces"])
    log.info("  withdrawals mode     : %s", decisions["block"]["withdrawals_mode"])
    for w in decisions["warnings"]:
        log.info("  WARNING: %s", w)
    log.info("Review this report before scaling out (build one chunk first, spec checklist 3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
