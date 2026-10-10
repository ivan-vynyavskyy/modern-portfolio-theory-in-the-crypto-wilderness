#!/usr/bin/env python3
"""
1_0_build_transfers.py

Purpose
-------
Main builder for the ETH native transfer ledger.

Per block-range chunk it unions, in one DuckDB query:
  - value rows from trace/ (call/create/suicide + reward/genesis/daofork,
    reverted subtrees excluded),
  - gas rows from transaction/ x block/ (fee_burn/fee_tip, EIP-1559 split
    keyed off base_fee_per_gas NULL-ness),
  - withdrawal rows exploded from block bodies,
applies the deterministic ordering + synthetic log_index (ordering.py) and
writes one Hive-partitioned Parquet per chunk, restartable
via _SUCCESS markers and a _manifest.jsonl.

transaction.value is intentionally never read: the root trace
already carries the external transfer and correctly reflects reverts.

Inputs
------
- decisions.json written by 0_0_inspect_schemas.py (run Step 0 first).
- trace/ transaction/ block/ Parquet partitions under --input-base.

Outputs
-------
- <output_base>/block_range=<lo>_<hi>/part-*.parquet  (+ _SUCCESS)
  (one part file per BLOCK_SUBCHUNK_SIZE sub-range — bounds memory and
   allows mid-chunk resume; log_index is per-block, so the split is safe)
- <output_base>/_manifest.jsonl (one line per completed chunk)

CLI
---
  $ python 1_0_build_transfers.py --chunk-index $SLURM_ARRAY_TASK_ID
  $ python 1_0_build_transfers.py --chunks 0,1,2
  $ python 1_0_build_transfers.py --all
  $ python 1_0_build_transfers.py --list-chunks
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import duckdb
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    Config,
    LONDON_BLOCK,
    MERGE_BLOCK,
    SHAPELLA_BLOCK,
    add_config_args,
    partition_files,
    setup_logging,
)
from ordering import (  # noqa: E402
    log_index_window_sql,
    trace_address_debug_sql,
    trace_address_sort_key_sql,
)

CANONICAL_COLS = "sender, recipient, block, log_index, value, tx_hash"
DEBUG_COLS = CANONICAL_COLS + ", transfer_type, trace_address"

# Column set every union branch must emit (same order everywhere).
UNION_COLS = (
    "sender, recipient, block, value, tx_hash, tx_index, ta_key, "
    "trace_address_str, transfer_type, sub_index"
)


# ----------------------------- SQL expression helpers -----------------------------

def sql_path_list(paths: list[str]) -> str:
    quoted = ", ".join("'" + p.replace("'", "''") + "'" for p in paths)
    return f"[{quoted}]"


def blob_num_sql(col: str) -> str:
    """Big-endian BLOB -> HUGEINT, fully native/vectorized (no Python UDF —
    this runs over billions of rows).

    Fast path (<= 16 bytes, the dump's usual minimal-length encoding):
    direct BIT cast. Slow path (> 16 bytes): some export batches store
    values ZERO-PADDED to 32 bytes — strip leading zero bytes via hex and
    re-parse, so a small padded value decodes instead of being dropped.
    Only a value still exceeding 16 significant bytes (>= 2^128 wei —
    impossible for real amounts) maps to NULL and is counted as a parse
    failure; same for a 16-byte value with the high bit set (negative after
    the BIT cast; NULLIF(GREATEST(x,-1),-1) is that guard)."""
    h = f"ltrim(hex({col}), '0')"
    padded_hex = f"lpad({h}, CAST(ceil(length({h}) / 2.0) * 2 AS INTEGER), '0')"
    slow = (
        f"CASE WHEN length({h}) > 32 THEN NULL "
        f"WHEN length({h}) = 0 THEN 0 "
        f"ELSE CAST(CAST(from_hex({padded_hex}) AS BIT) AS HUGEINT) END"
    )
    fast = (
        f"CASE WHEN octet_length({col}) = 0 THEN 0 "
        f"ELSE CAST(CAST({col} AS BIT) AS HUGEINT) END"
    )
    inner = (
        f"CASE WHEN {col} IS NULL THEN NULL "
        f"WHEN octet_length({col}) <= 16 THEN {fast} "
        f"ELSE {slow} END"
    )
    return f"NULLIF(GREATEST({inner}, -1), -1)"


def num_sql(col: str, fmt: str, target: str = "DECIMAL(38,0)") -> str:
    """Exact numeric parse; never routes through float."""
    if fmt == "hex_string":
        return f"TRY_CAST(hex_to_decstr(CAST({col} AS VARCHAR)) AS {target})"
    if fmt == "binary":
        return f"TRY_CAST({blob_num_sql(col)} AS {target})"
    return f"TRY_CAST({col} AS {target})"  # numeric or decimal string


def addr_sql(col: Optional[str], fmt: str) -> str:
    """Normalize an address/hash column to lowercase 0x string."""
    if col is None:
        return "CAST(NULL AS VARCHAR)"
    if fmt == "binary":
        return f"CASE WHEN {col} IS NULL THEN NULL ELSE '0x' || lower(hex({col})) END"
    return f"lower(CAST({col} AS VARCHAR))"


def register_udfs(con: duckdb.DuckDBPyConnection) -> None:
    """Exact hex/bytes -> decimal-string converters (NULL in -> NULL out)."""
    from duckdb.typing import BLOB, VARCHAR

    def hex_to_decstr(s: str) -> str:
        h = s[2:] if s.lower().startswith("0x") else s
        if not h:
            return "0"
        try:
            return str(int(h, 16))
        except ValueError:
            return None

    def blob_to_decstr(b) -> str:
        return str(int.from_bytes(bytes(b), "big")) if b is not None else None

    con.create_function("hex_to_decstr", hex_to_decstr, [VARCHAR], VARCHAR)
    con.create_function("blob_to_decstr", blob_to_decstr, [BLOB], VARCHAR)


# ----------------------------- Partition selection -----------------------------

def files_for_range(src: dict, lo: int, hi: int, logger) -> list[str]:
    """Parquet files of the partitions overlapping [lo, hi]. Delta-discovered
    partitions use their active-file list; plain ones are globbed."""
    files: list[str] = []
    unknown = 0
    for p in src["partitions"]:
        plo, phi = p["lo"], p["hi"]
        if plo is None:
            unknown += 1
        elif phi is None:
            if plo > hi:
                continue
        elif phi < lo or plo > hi:
            continue
        files.extend(partition_files(p))
    if unknown:
        logger.warning(
            "%d partitions had unparsable block ranges -> included unconditionally "
            "(block filter still applies, just no pruning)", unknown
        )
    return files


# ----------------------------- Source views -----------------------------

def create_trace_view(con, dec: dict, files: list[str], lo: int, hi: int) -> None:
    t = dec["trace"]
    c = t["cols"]
    block_expr = f"CAST({c['block']} AS BIGINT)"
    tx_index_expr = (
        f"CAST({c['tx_index']} AS BIGINT)" if c["tx_index"] else "CAST(NULL AS BIGINT)"
    )
    status_expr = f"TRY_CAST({c['status']} AS INTEGER)" if c["status"] else "CAST(NULL AS INTEGER)"
    error_expr = c["error"] if c["error"] else "CAST(NULL AS VARCHAR)"
    call_type_expr = f"lower(CAST({c['call_type']} AS VARCHAR))" if c["call_type"] else "CAST(NULL AS VARCHAR)"
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW trace_src AS
        SELECT
          {block_expr}                                    AS block,
          {addr_sql(c['tx_hash'], t['hash_format'])}      AS tx_hash,
          {tx_index_expr}                                 AS tx_index,
          {addr_sql(c['from'], t['addr_format'])}         AS from_addr,
          {addr_sql(c['to'], t['addr_format'])}           AS to_addr,
          {num_sql(c['value'], t['value_format'])}        AS value_wei,
          ({c['value']} IS NOT NULL)                      AS value_present,
          lower(CAST({c['trace_type']} AS VARCHAR))       AS trace_type,
          {call_type_expr}                                AS call_type,
          {status_expr}                                   AS status,
          {error_expr}                                    AS error,
          {trace_address_sort_key_sql(c['trace_address'], t['trace_address_kind'])}    AS ta_key,
          {trace_address_debug_sql(c['trace_address'], t['trace_address_kind'])}       AS trace_address_str
        FROM read_parquet({sql_path_list(files)}, union_by_name=true)
        WHERE {block_expr} BETWEEN {lo} AND {hi}
    """)


def create_trace_ok_view(con, dec: dict) -> None:
    """Value-moving traces with reverted subtrees removed."""
    mode = dec["trace"]["revert_mode"]
    if mode == "status":
        con.execute(
            "CREATE OR REPLACE TEMP VIEW trace_ok AS "
            "SELECT * FROM trace_src WHERE status = 1"
        )
    elif mode == "error_prefix":
        # A successful subcall under a later-reverting parent has no error of
        # its own -> exclude any trace whose trace_address starts with an
        # errored trace's prefix (incl. the errored trace itself).
        # ta_key is fixed-width, so "starts with prefix" is exact: an errored
        # root ('') kills the whole tx; otherwise match the trace itself or
        # any descendant (key extended by '.<elem>').
        con.execute("""
            CREATE OR REPLACE TEMP VIEW trace_ok AS
            SELECT t.* FROM trace_src t
            WHERE t.error IS NULL
              AND NOT EXISTS (
                SELECT 1
                FROM (SELECT tx_hash, ta_key FROM trace_src WHERE error IS NOT NULL) e
                WHERE e.tx_hash = t.tx_hash
                  AND (e.ta_key = '' OR t.ta_key = e.ta_key
                       OR t.ta_key LIKE e.ta_key || '.%')
              )
        """)
    else:  # 'none' — already warned loudly in Step 0
        con.execute("CREATE OR REPLACE TEMP VIEW trace_ok AS SELECT * FROM trace_src")


def create_tx_view(con, dec: dict, files: list[str], lo: int, hi: int) -> None:
    x = dec["transaction"]
    c = x["cols"]
    block_expr = f"CAST({c['block']} AS BIGINT)"
    gas_used = num_sql(c["gas_used"], x["gas_used_format"], "HUGEINT") if c["gas_used"] else "CAST(NULL AS HUGEINT)"
    eff = (
        num_sql(c["effective_gas_price"], x["gas_price_format"], "HUGEINT")
        if c["effective_gas_price"] else "CAST(NULL AS HUGEINT)"
    )
    gp = num_sql(c["gas_price"], x["gas_price_format"], "HUGEINT") if c["gas_price"] else "CAST(NULL AS HUGEINT)"
    # blob cols may be absent from decisions.json written by an older
    # inspect run (.get) or from old partitions' schemas (file probe)
    blob_used = "CAST(NULL AS HUGEINT)"
    blob_price = "CAST(NULL AS HUGEINT)"
    if c.get("blob_gas_used") and c.get("blob_gas_price") and \
            _col_in_chunk_files(c["blob_gas_used"], files):
        blob_used = num_sql(c["blob_gas_used"], x["gas_used_format"], "HUGEINT")
        blob_price = num_sql(c["blob_gas_price"], x["gas_price_format"], "HUGEINT")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW tx_src AS
        SELECT
          {block_expr}                               AS block,
          {addr_sql(c['hash'], x['hash_format'])}    AS tx_hash,
          CAST({c['tx_index']} AS BIGINT)            AS tx_index,
          {addr_sql(c['from'], x['addr_format'])}    AS from_addr,
          {gas_used}                                 AS gas_used,
          {eff}                                      AS eff_price,
          {gp}                                       AS gas_price,
          {blob_used}                                AS blob_gas_used,
          {blob_price}                               AS blob_gas_price
        FROM read_parquet({sql_path_list(files)}, union_by_name=true)
        WHERE {block_expr} BETWEEN {lo} AND {hi}
    """)


def _col_in_chunk_files(col: Optional[str], files: list[str]) -> bool:
    """Schema-drift guard: time-dependent columns (base_fee_per_gas,
    withdrawals) may be entirely absent from old partitions' schemas, and
    referencing them would fail even with union_by_name. Both only ever
    appear from a fork block onward, and files are ordered by block range,
    so probing last/first/middle file schemas is sufficient."""
    if col is None or not files:
        return False
    for f in dict.fromkeys([files[-1], files[0], files[len(files) // 2]]):
        try:
            if col in pq.read_schema(f).names:
                return True
        except Exception:
            pass
    return False


def create_block_view(con, dec: dict, files: list[str], lo: int, hi: int) -> None:
    b = dec["block"]
    c = b["cols"]
    block_expr = f"CAST({c['number']} AS BIGINT)"
    has_base_fee = _col_in_chunk_files(c["base_fee"], files)
    base_fee = (
        num_sql(c["base_fee"], b["base_fee_format"], "HUGEINT")
        if has_base_fee else "CAST(NULL AS HUGEINT)"
    )
    w_mode = b["withdrawals_mode"]
    if w_mode != "absent" and not _col_in_chunk_files(c["withdrawals"], files):
        w_mode = "absent"
    if w_mode == "list_struct":
        w_expr = c["withdrawals"]
    elif w_mode == "json_string":
        w_expr = (
            f"json_transform({c['withdrawals']}, "
            "'[{\"index\":\"BIGINT\",\"address\":\"VARCHAR\",\"amount\":\"VARCHAR\"}]')"
        )
    else:
        # NULL placeholder typed with the detected field names so the
        # withdrawal CTE still binds on chunks whose files lack the column.
        wf = b.get("withdrawal_fields") or {}
        idx_f = wf.get("index") or "index"
        addr_f = wf.get("address") or "address"
        amt_f = wf.get("amount") or "amount"
        w_expr = (
            f'CAST(NULL AS STRUCT("{idx_f}" BIGINT, "{addr_f}" VARCHAR, '
            f'"{amt_f}" VARCHAR)[])'
        )
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW block_src AS
        SELECT
          {block_expr}                              AS block,
          {addr_sql(c['miner'], b['miner_format'])} AS miner,
          {base_fee}                                AS base_fee,
          {w_expr}                                  AS withdrawals
        FROM read_parquet({sql_path_list(files)}, union_by_name=true)
        WHERE {block_expr} BETWEEN {lo} AND {hi}
    """)


# ----------------------------- Union branches -----------------------------

def value_rows_sql(dec: dict) -> str:
    """call/create/suicide value movements."""
    return f"""
      SELECT
        from_addr AS sender, to_addr AS recipient, block, value_wei AS value,
        tx_hash, tx_index, ta_key, trace_address_str,
        CASE WHEN trace_type IN ('create','create2') THEN 'create'
             WHEN trace_type IN ('suicide','selfdestruct') THEN 'suicide'
             ELSE 'call' END AS transfer_type,
        CAST(NULL AS BIGINT) AS sub_index
      FROM trace_ok
      WHERE value_wei IS NOT NULL AND value_wei > 0
        AND from_addr IS NOT NULL AND to_addr IS NOT NULL
        AND (
              -- callcode is EXCLUDED like delegatecall: it executes the
              -- callee's code in the CALLER's context, and its value
              -- transfer is caller->caller (net zero; geth implements it as
              -- a self-transfer). Treating it as a real transfer moved
              -- ~3.9M ETH from 2015-era exchange deposit wallets to their
              -- shared wallet-library address (1,339 negative balances).
              (trace_type = 'call' AND call_type = 'call')
           OR trace_type IN ('create','create2','suicide','selfdestruct')
            )
    """


def mint_rows_sql(dec: dict, flags: dict) -> str:
    """reward / genesis / daofork rows from traces.

    Rewards are hard-capped at the Merge: PoS has no execution-layer block
    rewards, but Erigon-style tracers synthesize a phantom 2-ETH reward trace
    for every post-Merge block — including them would
    overstate supply by ~2 ETH/block from block 15,537,394 on.
    """
    types = ["'daofork'"]
    if flags["include_rewards"]:
        types.append("'reward'")
    if flags["include_genesis"]:
        types.append("'genesis'")
    return f"""
      SELECT
        CASE WHEN trace_type = 'daofork' THEN from_addr ELSE NULL END AS sender,
        to_addr AS recipient, block, value_wei AS value,
        tx_hash, tx_index, ta_key, trace_address_str,
        trace_type AS transfer_type,
        CAST(NULL AS BIGINT) AS sub_index
      FROM trace_src
      WHERE value_wei IS NOT NULL AND value_wei > 0
        AND to_addr IS NOT NULL
        AND COALESCE(status, 1) = 1
        AND trace_type IN ({', '.join(types)})
        AND (trace_type != 'reward' OR block < {MERGE_BLOCK})
    """


def gas_rows_sql() -> str:
    """fee_burn / fee_tip rows for ALL txs including failed ones.

    Post-London (block.base_fee present): burned = base_fee*gas_used to NULL,
    tip = total - burned to the miner. Pre-London: single fee_tip = total.
    Post-Cancun the burn additionally includes the EIP-4844 blob fee
    (blob_gas_used * blob_gas_price — paid by the sender, fully burned,
    NOT part of effective_gas_price so it never touches the tip).
    HUGEINT math throughout: gas_used*price overflows int64.
    """
    return """
      gas_base AS (
        SELECT
          t.block, t.tx_hash, t.tx_index, t.from_addr, b.miner,
          t.gas_used * COALESCE(t.eff_price, t.gas_price) AS total_fee,
          CASE WHEN b.base_fee IS NULL THEN NULL
               ELSE b.base_fee * t.gas_used END           AS base_burned,
          CASE WHEN t.blob_gas_used IS NOT NULL AND t.blob_gas_price IS NOT NULL
               THEN t.blob_gas_used * t.blob_gas_price END AS blob_burned
        FROM tx_src t
        JOIN block_src b USING (block)
        WHERE t.gas_used IS NOT NULL
          AND COALESCE(t.eff_price, t.gas_price) IS NOT NULL
      ),
      fee_burn_rows AS (
        SELECT
          from_addr AS sender, CAST(NULL AS VARCHAR) AS recipient, block,
          CAST(COALESCE(base_burned, 0) + COALESCE(blob_burned, 0)
               AS DECIMAL(38,0)) AS value, tx_hash, tx_index,
          CAST(NULL AS VARCHAR) AS ta_key, CAST(NULL AS VARCHAR) AS trace_address_str,
          'fee_burn' AS transfer_type, CAST(NULL AS BIGINT) AS sub_index
        FROM gas_base
        WHERE COALESCE(base_burned, 0) + COALESCE(blob_burned, 0) > 0
      ),
      fee_tip_rows AS (
        SELECT
          from_addr AS sender, miner AS recipient, block,
          CAST(total_fee - COALESCE(base_burned, 0) AS DECIMAL(38,0)) AS value,
          tx_hash, tx_index,
          CAST(NULL AS VARCHAR) AS ta_key, CAST(NULL AS VARCHAR) AS trace_address_str,
          'fee_tip' AS transfer_type, CAST(NULL AS BIGINT) AS sub_index
        FROM gas_base
        WHERE miner IS NOT NULL AND total_fee - COALESCE(base_burned, 0) > 0
      )
    """


def withdrawal_rows_sql(dec: dict) -> str:
    """Beacon withdrawals exploded from block bodies, Gwei -> wei."""
    wf = dec["block"]["withdrawal_fields"]
    amt_fmt = dec["block"]["withdrawal_amount_format"]
    addr_f, amt_f, idx_f = wf.get("address"), wf.get("amount"), wf.get("index")
    amt_raw = f"struct_extract(w, '{amt_f}')"
    if amt_fmt == "hex_string":
        amt_expr = f"TRY_CAST(hex_to_decstr(CAST({amt_raw} AS VARCHAR)) AS HUGEINT)"
    elif amt_fmt == "binary":
        amt_expr = blob_num_sql(amt_raw)
    else:
        amt_expr = f"TRY_CAST({amt_raw} AS HUGEINT)"
    idx_expr = (
        f"COALESCE(TRY_CAST(struct_extract(w, '{idx_f}') AS BIGINT), CAST(pos AS BIGINT))"
        if idx_f else "CAST(pos AS BIGINT)"
    )
    if dec["block"].get("withdrawal_address_format") == "binary":
        recip_expr = f"'0x' || lower(hex(struct_extract(w, '{addr_f}')))"
    else:
        recip_expr = f"lower(CAST(struct_extract(w, '{addr_f}') AS VARCHAR))"
    return f"""
      w_exploded AS (
        SELECT block,
               unnest(withdrawals)                 AS w,
               generate_subscripts(withdrawals, 1) AS pos
        FROM block_src
        WHERE withdrawals IS NOT NULL AND len(withdrawals) > 0
      ),
      withdrawal_rows AS (
        SELECT * FROM (
          SELECT
            CAST(NULL AS VARCHAR) AS sender,
            {recip_expr} AS recipient,
            block,
            CAST({amt_expr} * 1000000000 AS DECIMAL(38,0)) AS value,
            CAST(NULL AS VARCHAR) AS tx_hash, CAST(NULL AS BIGINT) AS tx_index,
            CAST(NULL AS VARCHAR) AS ta_key, CAST(NULL AS VARCHAR) AS trace_address_str,
            'withdrawal' AS transfer_type,
            {idx_expr} AS sub_index
          FROM w_exploded
        ) WHERE value IS NOT NULL AND value > 0 AND recipient IS NOT NULL
      )
    """


def maybe_register_genesis_alloc(con, cfg: Config, dec: dict, lo: int, logger) -> bool:
    """Inject synthetic genesis rows from a geth genesis.json when the dump
    has no genesis traces. Returns True if registered."""
    if lo > 0 or not cfg.include_genesis or dec["trace"]["has_genesis_traces"]:
        return False
    if not cfg.genesis_alloc_json:
        logger.warning(
            "chunk covers block 0, dump has no genesis traces and no "
            "--genesis-alloc-json given: genesis premine NOT included "
            "(early whale balances understated, RQ-A)"
        )
        return False
    with open(cfg.genesis_alloc_json) as fh:
        doc = json.load(fh)
    alloc = doc.get("alloc", doc)
    rows = []
    for addr, entry in alloc.items():
        bal = entry.get("balance", entry) if isinstance(entry, dict) else entry
        wei = int(str(bal), 0)  # handles 0x… and decimal strings
        if wei > 0:
            a = addr.lower()
            rows.append(("0x" + a.removeprefix("0x"), str(wei)))
    import pyarrow as pa

    con.register(
        "genesis_alloc",
        pa.table({"address": [r[0] for r in rows], "value_str": [r[1] for r in rows]}),
    )
    logger.info("genesis alloc injected from %s: %d accounts", cfg.genesis_alloc_json, len(rows))
    return True


GENESIS_INJECT_SQL = """
      SELECT
        CAST(NULL AS VARCHAR) AS sender, address AS recipient, CAST(0 AS BIGINT) AS block,
        TRY_CAST(value_str AS DECIMAL(38,0)) AS value,
        CAST(NULL AS VARCHAR) AS tx_hash, CAST(NULL AS BIGINT) AS tx_index,
        CAST(NULL AS VARCHAR) AS ta_key, CAST(NULL AS VARCHAR) AS trace_address_str,
        'genesis' AS transfer_type, CAST(NULL AS BIGINT) AS sub_index
      FROM genesis_alloc
"""


# ----------------------------- Chunk processing -----------------------------

def chunk_partition_dir(cfg: Config, lo: int, hi: int) -> Path:
    return cfg.output_base / f"block_range={lo:09d}_{hi:09d}"


def chunk_done(cfg: Config, lo: int, hi: int) -> bool:
    """A chunk is done when _SUCCESS exists alongside non-empty part-*.parquet."""
    d = chunk_partition_dir(cfg, lo, hi)
    if not (d / "_SUCCESS").exists():
        return False
    return any(f.stat().st_size > 0 for f in d.glob("*.parquet"))


def open_connection(cfg: Config) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    cfg.scratch_tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit='{cfg.duckdb_memory_limit}'")
    con.execute(f"SET threads={cfg.duckdb_threads}")
    con.execute(f"SET temp_directory='{cfg.scratch_tmp}'")
    con.execute("SET preserve_insertion_order=false")
    register_udfs(con)
    return con


def _add_counts(agg: dict, key: str, n: int) -> None:
    if n:
        agg[key] = agg.get(key, 0) + n


PART_RE = re.compile(r"^part-(\d{9})_(\d{9})\.parquet$")
MIN_SUB_SPAN = 10_000  # blocks; below this an OOM is considered unrecoverable


def part_file_name(slo: int, shi: int) -> str:
    return f"part-{slo:09d}_{shi:09d}.parquet"


def existing_sub_coverage(part_dir: Path, slo: int, shi: int):
    """Part files intersecting [slo, shi] -> (files, exactly_tiles: bool).

    exactly_tiles means the files' block spans partition [slo, shi] without
    gaps or overlaps — only then is a resume-skip safe. Files with unparsable
    names (from older layouts) count as intersecting-but-unknown -> rebuild.
    """
    spans = []
    unknown = []
    for f in part_dir.glob("*.parquet"):
        m = PART_RE.match(f.name)
        if not m:
            unknown.append(f)
            continue
        flo, fhi = int(m.group(1)), int(m.group(2))
        if fhi >= slo and flo <= shi:
            spans.append((flo, fhi, f))
    if unknown:
        return [f for _, _, f in spans] + unknown, False
    spans.sort()
    files = [f for _, _, f in spans]
    if not spans:
        return files, False
    tiles = spans[0][0] == slo and spans[-1][1] == shi and all(
        spans[i + 1][0] == spans[i][1] + 1 for i in range(len(spans) - 1)
    )
    return files, tiles


def build_subchunk_adaptive(conh: list, cfg: Config, dec: dict, flags: dict,
                            slo: int, shi: int, part_dir: Path, agg: dict,
                            logger, tag: str) -> int:
    """Build [slo, shi]; on OutOfMemory split in half (fresh connection) and
    recurse — dense block ranges self-tune to whatever fits in memory."""
    out_file = part_dir / part_file_name(slo, shi)
    ts = time.perf_counter()
    try:
        rows = build_subchunk(conh[0], cfg, dec, flags, slo, shi, out_file,
                              agg, logger, tag)
        if rows:
            logger.info("%s   [%s-%s] %s rows in %.1fs", tag, f"{slo:,}",
                        f"{shi:,}", f"{rows:,}", time.perf_counter() - ts)
        return rows
    except duckdb.OutOfMemoryException as exc:
        span = shi - slo + 1
        if span <= MIN_SUB_SPAN:
            raise
        logger.warning("%s   [%s-%s] OUT OF MEMORY (%s) -> splitting sub-chunk "
                       "in half and retrying", tag, f"{slo:,}", f"{shi:,}", exc)
        try:
            conh[0].close()
        except Exception:
            pass
        conh[0] = open_connection(cfg)
        mid = slo + span // 2
        return (
            build_subchunk_adaptive(conh, cfg, dec, flags, slo, mid - 1,
                                    part_dir, agg, logger, tag)
            + build_subchunk_adaptive(conh, cfg, dec, flags, mid, shi,
                                      part_dir, agg, logger, tag)
        )


def build_subchunk(con, cfg: Config, dec: dict, flags: dict, slo: int, shi: int,
                   out_file: Path, agg: dict, logger, tag: str) -> int:
    """Build one sub-range [slo, shi] into out_file. Returns rows written.
    Accumulates input-row counts and warning counters into agg."""
    trace_files = files_for_range(dec["trace"], slo, shi, logger)
    tx_files = files_for_range(dec["transaction"], slo, shi, logger)
    block_files = files_for_range(dec["block"], slo, shi, logger)
    if not trace_files:
        logger.warning("%s   [%s-%s] no trace partitions overlap -> empty", tag,
                       f"{slo:,}", f"{shi:,}")
        return 0

    create_trace_view(con, dec, trace_files, slo, shi)
    create_trace_ok_view(con, dec)
    need_gas = bool(flags["include_gas"] and tx_files)
    if need_gas:
        create_tx_view(con, dec, tx_files, slo, shi)
    if block_files:
        create_block_view(con, dec, block_files, slo, shi)

    # trace lacks transaction_index -> borrow it from transaction/ via tx_hash
    if dec["trace"]["cols"]["tx_index"] is None and tx_files:
        if not need_gas:
            create_tx_view(con, dec, tx_files, slo, shi)
        con.execute("""
            CREATE OR REPLACE TEMP VIEW trace_ok AS
            SELECT t.* EXCLUDE (tx_index), x.tx_index
            FROM trace_ok t LEFT JOIN (SELECT DISTINCT tx_hash, tx_index FROM tx_src) x
            USING (tx_hash)
        """)

    withdrawals_on = bool(
        flags["include_withdrawals"]
        and dec["block"]["withdrawals_mode"] != "absent"
        and block_files
    )
    genesis_injected = maybe_register_genesis_alloc(con, cfg, dec, slo, logger)

    ctes = ["value_rows AS (" + value_rows_sql(dec) + ")",
            "mint_rows AS (" + mint_rows_sql(dec, flags) + ")"]
    branches = ["SELECT * FROM value_rows", "SELECT * FROM mint_rows"]
    if need_gas:
        ctes.append(gas_rows_sql().strip().lstrip())
        branches += ["SELECT * FROM fee_burn_rows", "SELECT * FROM fee_tip_rows"]
    if withdrawals_on:
        ctes.append(withdrawal_rows_sql(dec).strip())
        branches.append("SELECT * FROM withdrawal_rows")
    if genesis_injected:
        ctes.append("genesis_inject AS (" + GENESIS_INJECT_SQL + ")")
        branches.append("SELECT * FROM genesis_inject")

    out_cols = DEBUG_COLS if cfg.include_debug_columns else CANONICAL_COLS
    ctes_sql = ",\n    ".join(ctes)
    query = f"""
    WITH
    {ctes_sql},
    unioned AS (
      {' UNION ALL '.join(branches)}
    ),
    final AS (
      SELECT sender, recipient, block, value, tx_hash, transfer_type,
             trace_address_str AS trace_address,
             {log_index_window_sql()} AS log_index
      FROM unioned
    )
    SELECT {out_cols.replace('log_index', 'CAST(log_index AS INTEGER) AS log_index')}
    FROM final
    ORDER BY block, log_index
    """
    tmp_path = out_file.with_name(out_file.name + f".tmp-{os.getpid()}")
    rows = con.execute(
        f"COPY ({query}) TO '{tmp_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    ).fetchone()[0]
    os.replace(tmp_path, out_file)

    # input rows + warning counters for this sub-range
    _add_counts(agg["input_rows"], "trace",
                con.execute("SELECT count(*) FROM trace_src").fetchone()[0])
    if need_gas:
        _add_counts(agg["input_rows"], "transaction",
                    con.execute("SELECT count(*) FROM tx_src").fetchone()[0])
    if block_files:
        _add_counts(agg["input_rows"], "block",
                    con.execute("SELECT count(*) FROM block_src").fetchone()[0])
    _add_counts(agg, "value_parse_failures", con.execute(
        "SELECT count(*) FROM trace_src WHERE value_present AND value_wei IS NULL"
    ).fetchone()[0])
    if need_gas:
        _add_counts(agg, "txs_missing_gas", con.execute(
            "SELECT count(*) FROM tx_src WHERE gas_used IS NULL "
            "OR COALESCE(eff_price, gas_price) IS NULL"
        ).fetchone()[0])
        if shi >= LONDON_BLOCK:
            _add_counts(agg, "post_london_blocks_missing_base_fee", con.execute(
                f"SELECT count(*) FROM block_src WHERE block >= {LONDON_BLOCK} "
                "AND base_fee IS NULL"
            ).fetchone()[0])
    if withdrawals_on and shi >= SHAPELLA_BLOCK:
        _add_counts(agg, "post_shapella_blocks_no_withdrawals", con.execute(
            f"SELECT count(*) FROM block_src WHERE block >= {SHAPELLA_BLOCK} "
            "AND (withdrawals IS NULL OR len(withdrawals) = 0)"
        ).fetchone()[0])
    return rows


def process_chunk(cfg: Config, dec: dict, lo: int, hi: int, force: bool, logger) -> dict:
    """Build one chunk in sub-ranges of cfg.block_subchunk_size blocks, each
    written as its own part-*.parquet — bounds memory on dense block ranges
    and makes the chunk resumable mid-way (finished part files are kept).
    log_index is per-block, so any block-aligned split is safe."""
    tag = f"[{lo:,}-{hi:,}]"
    if not force and chunk_done(cfg, lo, hi):
        logger.info("%s already built (_SUCCESS present) -> skipping", tag)
        return {"lo": lo, "hi": hi, "skipped": True}

    t0 = time.perf_counter()
    flags = dict(dec["flags"])
    # CLI/env can only further restrict what Step 0 deemed possible.
    flags["include_gas"] = flags["include_gas"] and cfg.include_gas
    flags["include_withdrawals"] = flags["include_withdrawals"] and cfg.include_withdrawals
    flags["include_rewards"] = flags["include_rewards"] and cfg.include_rewards
    flags["include_genesis"] = flags["include_genesis"] and cfg.include_genesis

    part_dir = chunk_partition_dir(cfg, lo, hi)
    part_dir.mkdir(parents=True, exist_ok=True)
    (part_dir / "_SUCCESS").unlink(missing_ok=True)
    if force:
        for f in part_dir.glob("*.parquet"):
            f.unlink()
    for stale in part_dir.glob("*.tmp-*"):
        stale.unlink(missing_ok=True)

    subs = list(range(lo, hi + 1, cfg.block_subchunk_size))
    logger.info("%s start: %d sub-chunks of %s blocks, revert_mode=%s", tag,
                len(subs), f"{cfg.block_subchunk_size:,}", dec["trace"]["revert_mode"])

    agg: dict = {"input_rows": {}}
    conh = [open_connection(cfg)]  # holder: adaptive splitting may reopen it
    try:
        for slo in subs:
            shi = min(slo + cfg.block_subchunk_size - 1, hi)
            covering, tiles = existing_sub_coverage(part_dir, slo, shi)
            if tiles and not force:
                logger.info("%s   [%s-%s] %d part file(s) tile the range -> "
                            "resume-skip", tag, f"{slo:,}", f"{shi:,}", len(covering))
                continue
            for f in covering:  # partial/unknown coverage -> rebuild cleanly
                logger.info("%s   removing partial part file %s", tag, f.name)
                f.unlink()
            build_subchunk_adaptive(conh, cfg, dec, flags, slo, shi,
                                    part_dir, agg, logger, tag)

        out_glob = str(part_dir / "*.parquet")
        stats = collect_chunk_stats(conh[0], cfg, out_glob, lo, hi, flags, agg,
                                    logger, tag)

        # section 8: assert no NULL / negative values in the output
        bad = conh[0].execute(
            f"SELECT count(*) FROM read_parquet('{out_glob}') "
            "WHERE value IS NULL OR value < 0 "
            "OR (sender IS NULL AND recipient IS NULL)"
        ).fetchone()[0]
        if bad:
            raise RuntimeError(f"{tag} {bad} invalid rows (NULL/negative value or "
                               "double-NULL endpoints) — _SUCCESS withheld")

        (part_dir / "_SUCCESS").write_text(json.dumps(stats) + "\n")
        with open(cfg.output_base / "_manifest.jsonl", "a") as mf:
            mf.write(json.dumps(stats) + "\n")

        wall = time.perf_counter() - t0
        nblocks = hi - lo + 1
        logger.info(
            "%s done: %s rows in %.1fs (%.0f blocks/s, %.0f rows/s)",
            tag, f"{stats['rows_out']:,}", wall, nblocks / wall,
            stats["rows_out"] / wall if wall else 0,
        )
        stats["wall_seconds"] = round(wall, 1)
        return stats
    finally:
        conh[0].close()


def collect_chunk_stats(con, cfg, out_glob: str, lo, hi, flags, agg,
                        logger, tag) -> dict:
    """Per-chunk logging: output row counts + per-type breakdown + supply
    invariant from the written part files, plus the accumulated per-sub-chunk
    input counts and WARNING counters."""
    out = f"read_parquet('{out_glob}')"
    rows_out, minted, burned = con.execute(
        f"SELECT count(*), "
        f"COALESCE(SUM(value) FILTER (WHERE sender IS NULL), 0), "
        f"COALESCE(SUM(value) FILTER (WHERE recipient IS NULL), 0) FROM {out}"
    ).fetchone()
    minted, burned = int(minted), int(burned)
    stats = {
        "lo": lo, "hi": hi, "rows_out": rows_out,
        "minted_wei": str(minted), "burned_wei": str(burned),
        "net_supply_delta_wei": str(minted - burned),
        "flags": {k: v for k, v in flags.items()},
    }
    logger.info("%s output rows: %s | minted=%s burned=%s net_supply_delta=%s wei",
                tag, f"{rows_out:,}", minted, burned, minted - burned)

    if cfg.include_debug_columns:
        by_type = con.execute(
            f"SELECT transfer_type, count(*), SUM(value) FROM {out} GROUP BY 1 ORDER BY 1"
        ).fetchall()
        stats["by_type"] = {t: {"rows": n, "wei": str(v)} for t, n, v in by_type}
        for t, n, v in by_type:
            logger.info("%s   %-11s rows=%-12s wei=%s", tag, t, f"{n:,}", v)
    else:
        logger.info("%s per-type breakdown unavailable (INCLUDE_DEBUG_COLUMNS=False)", tag)

    stats["input_rows"] = agg["input_rows"]
    logger.info("%s input rows scanned: %s", tag,
                " ".join(f"{k}={v:,}" for k, v in agg["input_rows"].items()))

    warn_msgs = {
        "value_parse_failures":
            "%d trace values FAILED to parse (excluded)",
        "txs_missing_gas":
            "%d txs missing gas_used/effective_gas_price -> no gas rows for "
            "them (sender balances overstated)",
        "post_london_blocks_missing_base_fee":
            "%d post-London blocks missing base_fee_per_gas -> their full fee "
            "treated as tip (burn understated)",
        "post_shapella_blocks_no_withdrawals":
            "%d post-Shapella blocks with zero withdrawals (a few are "
            "legitimate; many suggests missing data)",
    }
    for key, msg in warn_msgs.items():
        if agg.get(key):
            logger.warning("%s " + msg, tag, agg[key])
            stats[key] = agg[key]
    return stats


# ----------------------------- Chunk math + main -----------------------------

def compute_chunks(cfg: Config, dec: dict) -> list[tuple[int, int]]:
    lo = cfg.block_range_start
    hi = cfg.block_range_end
    if hi is None:
        hi = dec["block_range"][1]
        if hi is None:
            raise RuntimeError("end block unknown: set BLOCK_RANGE or fix decisions.json")
    size = cfg.block_chunk_size
    return [(s, min(s + size - 1, hi)) for s in range(lo, hi + 1, size)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the ETH native transfer ledger")
    parser.add_argument("--chunk-index", type=int, help="single chunk (SLURM array task id)")
    parser.add_argument("--chunks", help="comma-separated chunk indices")
    parser.add_argument("--all", action="store_true", help="iterate all chunks (default)")
    parser.add_argument("--list-chunks", action="store_true")
    parser.add_argument("--force", action="store_true", help="rebuild even if _SUCCESS exists")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = Config.from_args(args)

    if not cfg.decisions_path.exists():
        print(f"ERROR: {cfg.decisions_path} not found — run inspect_schemas.py first (Step 0)",
              file=sys.stderr)
        return 2
    with open(cfg.decisions_path) as fh:
        dec = json.load(fh)

    chunks = compute_chunks(cfg, dec)
    if args.list_chunks:
        for i, (lo, hi) in enumerate(chunks):
            state = "done" if chunk_done(cfg, lo, hi) else "todo"
            print(f"chunk {i:4d}: blocks {lo:>12,} - {hi:>12,}  [{state}]")
        return 0

    if args.chunk_index is not None:
        selected = [args.chunk_index]
    elif args.chunks:
        selected = [int(x) for x in args.chunks.split(",")]
    else:
        selected = list(range(len(chunks)))

    job_tag = (
        f"build_chunk{selected[0]:04d}" if len(selected) == 1
        else f"build_{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    logger = setup_logging(cfg.logs_dir, job_tag)
    logger.info("config:\n%s", cfg.describe())
    logger.info("decisions: %s (generated %s)", cfg.decisions_path, dec.get("generated_at"))
    logger.info("%d total chunks, running %d", len(chunks), len(selected))

    cfg.output_base.mkdir(parents=True, exist_ok=True)
    totals: dict = {"rows": 0, "minted": 0, "burned": 0, "chunks": 0, "skipped": 0}
    for idx in selected:
        if idx < 0 or idx >= len(chunks):
            logger.error("chunk index %d out of range (0..%d)", idx, len(chunks) - 1)
            return 2
        lo, hi = chunks[idx]
        stats = process_chunk(cfg, dec, lo, hi, args.force, logger)
        if stats.get("skipped"):
            totals["skipped"] += 1
            continue
        if stats.get("empty"):
            continue
        totals["rows"] += stats["rows_out"]
        totals["minted"] += int(stats["minted_wei"])
        totals["burned"] += int(stats["burned_wei"])
        totals["chunks"] += 1

    # Final run summary
    f = dec["flags"]
    degraded = []
    if not (f["include_gas"] and cfg.include_gas):
        degraded.append("gas OFF -> active-sender balances OVERSTATED by cumulative gas")
    if not (f["include_withdrawals"] and cfg.include_withdrawals):
        degraded.append("withdrawals OFF -> staking/validator balances UNDERSTATED post-Apr-2023")
    if not (f["include_genesis"] and cfg.include_genesis) or (
        not dec["trace"]["has_genesis_traces"] and not cfg.genesis_alloc_json
    ):
        degraded.append("genesis alloc OFF/missing -> early whale balances understated")
    logger.info("=" * 70)
    logger.info("RUN SUMMARY: chunks built=%d skipped=%d rows=%s",
                totals["chunks"], totals["skipped"], f"{totals['rows']:,}")
    logger.info("  cumulative minted=%d burned=%d net_supply_delta=%d wei",
                totals["minted"], totals["burned"], totals["minted"] - totals["burned"])
    for d in degraded or ["none — all components active"]:
        logger.info("  degraded: %s", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
