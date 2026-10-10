#!/usr/bin/env python3
"""
6_0_selftest_synthetic.py

Purpose
-------
End-to-end self-test of the ETH transfer reconstruction pipeline against a
small synthetic ethereum-etl-style dump, asserting the EXACT expected ledger.

Covered:
- Variant A (string columns, status propagation, hive partitions):
  genesis traces, pre-London legacy fees, EIP-1559 burn/tip split,
  delegatecall/staticcall exclusion, reverted-tx exclusion (status=0) with
  gas still charged, create endowment, suicide refund, reward rows,
  withdrawals (Gwei->wei) incl. ordering, log_index synthesis, chunking,
  resumability, 3_0_validate_transfer_ledger.py balance + supply.
- Variant B (binary addresses/hashes, int64 values, no status column):
  error-prefix revert propagation fallback (phantom-subtree exclusion),
  binary -> 0x-string normalization, canonical 6-column output mode.

CLI
---
  $ python 6_0_selftest_synthetic.py [--keep] [--workdir DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from deltalake import write_deltalake

    HAVE_DELTA = True
except ImportError:
    HAVE_DELTA = False

HERE = Path(__file__).resolve().parent
PY = sys.executable

ETH = 10**18
GWEI = 10**9

# addresses (chosen so lexical order is stable in tie-breaks)
G1 = "0x" + "11" * 20
G2 = "0x" + "22" * 20
W = "0x" + "33" * 20
K = "0x" + "44" * 20
A = "0x" + "aa" * 20
B = "0x" + "bb" * 20
C = "0x" + "cc" * 20
M = "0x" + "ee" * 20  # miner

H1, H2, H3, H4, H5, H6, H7 = ("0x" + f"{i:02x}" * 32 for i in range(1, 8))


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"command failed: {' '.join(str(c) for c in cmd)}")
    return r


def write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# ----------------------------- Variant A dump -----------------------------

def trace_table_a(rows: list[dict]) -> pa.Table:
    cols = {
        "block_number": ("blk", pa.int64()),
        "transaction_hash": ("tx", pa.string()),
        "transaction_index": ("txi", pa.int64()),
        "from_address": ("frm", pa.string()),
        "to_address": ("to", pa.string()),
        "value": ("val", pa.string()),
        "trace_type": ("tt", pa.string()),
        "call_type": ("ct", pa.string()),
        "status": ("st", pa.int64()),
        "error": ("err", pa.string()),
        "trace_address": ("ta", pa.string()),
        "reward_type": ("rt", pa.string()),
    }
    return pa.table(
        {name: pa.array([r.get(key) for r in rows], type=t) for name, (key, t) in cols.items()}
    )


def t(blk, tt, frm, to, val, tx=None, txi=None, ta="", ct=None, st=1, rt=None):
    return dict(blk=blk, tt=tt, frm=frm, to=to, val=str(val), tx=tx, txi=txi,
                ta=ta, ct=ct, st=st, rt=rt)


def build_variant_a(base: Path) -> None:
    p1, p2 = "block_range=00000000-00000099", "block_range=00000100-00000199"

    traces_p1 = [
        # genesis premine
        t(0, "genesis", None, G1, 5 * ETH),
        t(0, "genesis", None, G2, 7 * ETH),
        # block 10, tx0: simple transfer
        t(10, "call", A, B, ETH, H1, 0, ct="call"),
        # tx1: zero-value root, value in subcall; delegatecall/staticcall noise
        t(10, "call", A, C, 0, H2, 1, ct="call"),
        t(10, "call", C, B, 2 * 10**17, H2, 1, ta="0", ct="call"),
        t(10, "call", C, B, 5 * 10**17, H2, 1, ta="1", ct="delegatecall"),  # excluded
        t(10, "call", C, B, 0, H2, 1, ta="2", ct="staticcall"),             # excluded
        # callcode = library code in caller's context, value transfer is
        # caller->caller (net zero) -- must NOT be recorded as C->B
        t(10, "call", C, B, 7 * 10**17, H2, 1, ta="3", ct="callcode"),      # excluded
        # tx2: reverted (status=0 incl. root) — value must NOT move
        t(10, "call", B, A, 3 * 10**17, H3, 2, ct="call", st=0),
        # block reward
        t(10, "reward", None, M, 2 * ETH, rt="block"),
        # block 20: create endowment + suicide refund + reward
        t(20, "create", A, K, 4 * 10**17, H4, 0),
        t(20, "call", A, K, 0, H5, 1, ct="call"),
        t(20, "suicide", K, A, 4 * 10**17, H5, 1, ta="0"),
        t(20, "reward", None, M, 2 * ETH, rt="block"),
    ]
    traces_p2 = [
        # block 150 (post-London in this synthetic chain: base_fee set)
        t(150, "call", B, C, 5 * 10**17, H6, 0, ct="call"),
        # blob tx (EIP-4844): zero-value root; blob fee burned on top
        t(150, "call", B, C, 0, H7, 1, ct="call"),
    ]
    write_parquet(base / "trace" / p1 / "data.parquet", trace_table_a(traces_p1))
    write_parquet(base / "trace" / p2 / "data.parquet", trace_table_a(traces_p2))

    def tx_table(rows):
        return pa.table({
            "block_number": pa.array([r[0] for r in rows], pa.int64()),
            "hash": pa.array([r[1] for r in rows], pa.string()),
            "transaction_index": pa.array([r[2] for r in rows], pa.int64()),
            "from_address": pa.array([r[3] for r in rows], pa.string()),
            "value": pa.array([str(10**30)] * len(rows), pa.string()),  # MUST be ignored
            "receipt_gas_used": pa.array([r[4] for r in rows], pa.int64()),
            "receipt_effective_gas_price": pa.array([r[5] for r in rows], pa.int64()),
            "gas_price": pa.array([r[6] for r in rows], pa.int64()),
            "receipt_status": pa.array([r[7] for r in rows], pa.int64()),
            "receipt_blob_gas_used": pa.array(
                [r[8] if len(r) > 8 else None for r in rows], pa.int64()),
            "receipt_blob_gas_price": pa.array(
                [r[9] if len(r) > 9 else None for r in rows], pa.int64()),
        })

    txs_p1 = [
        (10, H1, 0, A, 21_000, None, GWEI, 1),
        (10, H2, 1, A, 100_000, None, 2 * GWEI, 1),
        (10, H3, 2, B, 30_000, None, GWEI, 0),  # failed but pays gas
        (20, H4, 0, A, 50_000, None, GWEI, 1),
        (20, H5, 1, A, 40_000, None, GWEI, 1),
    ]
    txs_p2 = [(150, H6, 0, B, 21_000, 110, 100, 1),
              (150, H7, 1, B, 21_000, 110, 100, 1, 131_072, 7)]
    write_parquet(base / "transaction" / p1 / "data.parquet", tx_table(txs_p1))
    write_parquet(base / "transaction" / p2 / "data.parquet", tx_table(txs_p2))

    # amount is a big-endian BLOB, mirroring the cluster dump
    w_type = pa.list_(pa.struct([
        ("index", pa.int64()), ("validator_index", pa.int64()),
        ("address", pa.string()), ("amount", pa.binary()),
    ]))

    def block_table(rows):
        return pa.table({
            "number": pa.array([r[0] for r in rows], pa.int64()),
            "miner": pa.array([r[1] for r in rows], pa.string()),
            "base_fee_per_gas": pa.array([r[2] for r in rows], pa.int64()),
            "withdrawals": pa.array([r[3] for r in rows], w_type),
        })

    blocks_p1 = [(0, M, None, None), (10, M, None, None), (20, M, None, None)]
    blocks_p2 = [
        (150, M, 100, None),
        (160, M, 100, [
            {"index": 1, "validator_index": 7, "address": W, "amount": int_be(1_000_000_000)},
            {"index": 2, "validator_index": 8, "address": B, "amount": int_be(2_000_000_000)},
        ]),
    ]
    write_parquet(base / "block" / p1 / "data.parquet", block_table(blocks_p1))
    write_parquet(base / "block" / p2 / "data.parquet", block_table(blocks_p2))


EXPECTED_A = [
    # (block, log_index, sender, recipient, value, tx_hash, transfer_type, trace_address)
    (0, 0, None, G1, 5 * ETH, None, "genesis", ""),
    (0, 1, None, G2, 7 * ETH, None, "genesis", ""),
    (10, 0, A, B, ETH, H1, "call", ""),
    (10, 1, A, M, 21_000 * GWEI, H1, "fee_tip", None),
    (10, 2, C, B, 2 * 10**17, H2, "call", "0"),
    (10, 3, A, M, 100_000 * 2 * GWEI, H2, "fee_tip", None),
    (10, 4, B, M, 30_000 * GWEI, H3, "fee_tip", None),
    (10, 5, None, M, 2 * ETH, None, "reward", ""),
    (20, 0, A, K, 4 * 10**17, H4, "create", ""),
    (20, 1, A, M, 50_000 * GWEI, H4, "fee_tip", None),
    (20, 2, K, A, 4 * 10**17, H5, "suicide", "0"),
    (20, 3, A, M, 40_000 * GWEI, H5, "fee_tip", None),
    (20, 4, None, M, 2 * ETH, None, "reward", ""),
    (150, 0, B, C, 5 * 10**17, H6, "call", ""),
    (150, 1, B, None, 21_000 * 100, H6, "fee_burn", None),
    (150, 2, B, M, 21_000 * (110 - 100), H6, "fee_tip", None),
    # blob tx: burn = base 21000*100 + blob 131072*7; tip unchanged
    (150, 3, B, None, 21_000 * 100 + 131_072 * 7, H7, "fee_burn", None),
    (150, 4, B, M, 21_000 * (110 - 100), H7, "fee_tip", None),
    (160, 0, None, W, 1 * ETH, None, "withdrawal", None),
    (160, 1, None, B, 2 * ETH, None, "withdrawal", None),
]


def expected_balance(addr: str, up_to_block: int) -> int:
    bal = 0
    for blk, _li, snd, rcp, val, *_ in EXPECTED_A:
        if blk > up_to_block:
            continue
        if rcp == addr:
            bal += val
        if snd == addr:
            bal -= val
    return bal


# ----------------------------- Variant B dump -----------------------------

def hb(hex0x: str) -> bytes:
    return bytes.fromhex(hex0x[2:])


def int_be(v: int) -> bytes:
    """Big-endian minimal bytes, empty for 0 — like the cluster dump's BLOBs."""
    return v.to_bytes((v.bit_length() + 7) // 8, "big") if v else b""


def write_source(base: Path, name: str, table: pa.Table) -> None:
    """Write one source as a real Delta Lake table (partition=N dirs +
    _delta_log, like the cluster dump) or as a plain dir if deltalake is
    unavailable."""
    if HAVE_DELTA:
        table = table.append_column("partition", pa.array([0] * table.num_rows, pa.int64()))
        write_deltalake(str(base / name), table, partition_by=["partition"])
    else:
        write_parquet(base / name / "0" / "data.parquet", table)


def build_variant_b(base: Path) -> None:
    """Delta layout, binary addresses/hashes, BLOB big-endian values,
    NO status column (error only)."""
    h1 = "0x" + "f1" * 32
    h2 = "0x" + "f2" * 32
    rows = [
        # tx0: root ok; [0] reverted; [0,0] ok-under-reverted (must be excluded);
        # [1] ok (must be included); [2] zero-value as EMPTY blob (excluded)
        (5, h1, 0, A, B, ETH, "call", "call", None, ""),
        (5, h1, 0, A, C, 5 * 10**17, "call", "call", "Reverted", "0"),
        (5, h1, 0, C, B, 1 * 10**17, "call", "call", None, "0,0"),
        (5, h1, 0, A, C, 2 * 10**17, "call", "call", None, "1"),
        (5, h1, 0, A, C, 0, "call", "call", None, "2"),
        # [3] value ZERO-PADDED to 32 bytes (other export batches' encoding)
        # -> MUST decode, not be dropped as a parse failure
        (5, h1, 0, C, A, 12_403 * GWEI, "call", "call", None, "3"),
        # [4] genuinely overflowing value (>= 2^128) -> excluded, no crash
        (5, h1, 0, C, A, 2**200, "call", "call", None, "4"),
        # tx1: root errored -> nothing moves, gas still charged
        (5, h2, 1, B, C, 3 * 10**17, "call", "call", "Out of gas", ""),
    ]

    def enc_value(row):
        val, ta = row[5], row[9]
        if ta == "3":
            return val.to_bytes(32, "big")      # zero-padded encoding
        return int_be(val)

    trace = pa.table({
        "block_number": pa.array([r[0] for r in rows], pa.int64()),
        "transaction_hash": pa.array([hb(r[1]) for r in rows], pa.binary()),
        "transaction_index": pa.array([r[2] for r in rows], pa.int64()),
        "from_address": pa.array([hb(r[3]) for r in rows], pa.binary()),
        "to_address": pa.array([hb(r[4]) for r in rows], pa.binary()),
        "value": pa.array([enc_value(r) for r in rows], pa.binary()),
        "trace_type": pa.array([r[6] for r in rows], pa.string()),
        "call_type": pa.array([r[7] for r in rows], pa.string()),
        "error": pa.array([r[8] for r in rows], pa.string()),
        "trace_address": pa.array([r[9] for r in rows], pa.string()),
    })
    write_source(base, "trace", trace)

    tx = pa.table({
        "block_number": pa.array([5, 5], pa.int64()),
        "hash": pa.array([hb(h1), hb(h2)], pa.binary()),
        "transaction_index": pa.array([0, 1], pa.int64()),
        "from_address": pa.array([hb(A), hb(B)], pa.binary()),
        "receipt_gas_used": pa.array([21_000, 21_000], pa.int64()),
        "receipt_effective_gas_price": pa.array([GWEI, GWEI], pa.int64()),
        "gas_price": pa.array([GWEI, GWEI], pa.int64()),
    })
    write_source(base, "transaction", tx)

    blocks = pa.table({
        "number": pa.array(list(range(10)), pa.int64()),
        "miner": pa.array([hb(M)] * 10, pa.binary()),
    })
    write_source(base, "block", blocks)


EXPECTED_B = [
    (5, 0, A, B, ETH, "0x" + "f1" * 32),
    (5, 1, A, C, 2 * 10**17, "0x" + "f1" * 32),
    (5, 2, C, A, 12_403 * GWEI, "0x" + "f1" * 32),   # zero-padded 32B value
    (5, 3, A, M, 21_000 * GWEI, "0x" + "f1" * 32),
    (5, 4, B, M, 21_000 * GWEI, "0x" + "f2" * 32),
]


# ----------------------------- Assertions -----------------------------

def fetch_ledger(out_base: Path, cols: str) -> list[tuple]:
    con = duckdb.connect()
    glob = str(out_base / "block_range=*" / "*.parquet")
    rows = con.execute(
        f"SELECT {cols} FROM read_parquet('{glob}') ORDER BY block, log_index"
    ).fetchall()
    con.close()
    return rows


def check(name: str, actual, expected) -> None:
    if actual != expected:
        print(f"FAIL {name}")
        print(f"  expected: {expected}")
        print(f"  actual  : {actual}")
        raise SystemExit(1)
    print(f"  OK  {name}")


def run_variant_a(work: Path) -> None:
    print("== Variant A: ethereum-etl style (status, strings, withdrawals) ==")
    inp, out = work / "a_input", work / "a_output"
    build_variant_a(inp)
    common = ["--input-base", inp, "--output-base", out,
              "--scratch-tmp", work / "a_scratch"]
    run([PY, HERE / "0_0_inspect_schemas.py", *common])

    dec = json.loads((out / "decisions.json").read_text())
    check("A decisions: etl format", dec["looks_like_ethereum_etl"], True)
    check("A decisions: revert mode", dec["trace"]["revert_mode"], "status")
    check("A decisions: withdrawals mode", dec["block"]["withdrawals_mode"], "list_struct")
    check("A decisions: gas available", dec["flags"]["include_gas"], True)
    check("A decisions: block range (footer stats)", dec["block_range"], [0, 160])
    check("A decisions: genesis traces", dec["trace"]["has_genesis_traces"], True)

    run([PY, HERE / "1_0_build_transfers.py", *common, "--block-chunk-size", "100",
         "--block-subchunk-size", "50", "--all"])

    rows = fetch_ledger(
        out, "block, log_index, sender, recipient, CAST(value AS VARCHAR), "
             "tx_hash, transfer_type, trace_address"
    )
    expected = [
        (blk, li, s, r, str(v), tx, tt, ta) for blk, li, s, r, v, tx, tt, ta in EXPECTED_A
    ]
    check("A full ledger (18 rows, ordering + log_index)", rows, expected)

    # value column must be DECIMAL(38,0)
    con = duckdb.connect()
    types = dict(con.execute(
        "SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM "
        f"read_parquet('{out / 'block_range=*' / '*.parquet'}', hive_partitioning=false))"
    ).fetchall())
    check("A value type", types["value"], "DECIMAL(38,0)")

    # supply invariant
    minted, burned = con.execute(
        f"SELECT SUM(value) FILTER (WHERE sender IS NULL), "
        f"SUM(value) FILTER (WHERE recipient IS NULL) "
        f"FROM read_parquet('{out / 'block_range=*' / '*.parquet'}')"
    ).fetchone()
    check("A minted", int(minted), 5 * ETH + 7 * ETH + 2 * ETH + 2 * ETH + 3 * ETH)
    check("A burned (base + blob)", int(burned), 2 * 21_000 * 100 + 131_072 * 7)
    con.close()

    # sub-chunking: chunk (0,99) with subchunk 50 -> part for blocks 0-49;
    # sub 50-99 has no data and writes no file. chunk (100,160): only the
    # 150-160 sub carries data.
    chunk0 = out / "block_range=000000000_000000099"
    check("A sub-chunked output files",
          sorted(p.name for p in chunk0.glob("*.parquet")),
          ["part-000000000_000000049.parquet"])
    chunk1 = out / "block_range=000000100_000000160"
    check("A sub-chunk skips empty ranges",
          sorted(p.name for p in chunk1.glob("*.parquet")),
          ["part-000000150_000000160.parquet"])

    # resumability: second run must skip both chunks (mtime unchanged)
    part = chunk0 / "part-000000000_000000049.parquet"
    mtime = part.stat().st_mtime_ns
    r = run([PY, HERE / "1_0_build_transfers.py", *common, "--block-chunk-size", "100",
             "--block-subchunk-size", "50", "--all"])
    check("A resume: chunk skipped", part.stat().st_mtime_ns, mtime)
    if "skipped" not in (r.stdout + r.stderr):
        raise SystemExit("FAIL A resume: no 'skipped' message in output")
    check("A manifest lines", len((out / "_manifest.jsonl").read_text().splitlines()), 2)

    # validate.py balance for A, B, M at block 199
    bal_csv = work / "a_balances.csv"
    run([PY, HERE / "3_0_validate_transfer_ledger.py", "balance", "--block", "199",
         "--addresses", f"{A},{B},{M}", "--out-csv", bal_csv, *common])
    got = {line.split(",")[0]: int(line.split(",")[2])
           for line in bal_csv.read_text().splitlines()[1:]}
    for addr, label in ((A, "A"), (B, "B"), (M, "miner")):
        check(f"A validate balance({label})", got[addr], expected_balance(addr, 199))

    # validate.py supply at snapshots 99 and 199
    sup_csv = work / "a_supply.csv"
    run([PY, HERE / "3_0_validate_transfer_ledger.py", "supply", "--snapshot-blocks", "99,199",
         "--out-csv", sup_csv, *common])
    sup = {int(l.split(",")[0]): int(l.split(",")[1])
           for l in sup_csv.read_text().splitlines()[1:]}
    exp99 = sum(v for blk, _li, s, r, v, *_ in EXPECTED_A if blk <= 99 and s is None) \
        - sum(v for blk, _li, s, r, v, *_ in EXPECTED_A if blk <= 99 and r is None)
    check("A validate supply @99", sup[99], exp99)
    check("A validate supply @199", sup[199], int(minted) - int(burned))


def run_variant_b(work: Path) -> None:
    print("== Variant B: binary cols, int64 values, error-prefix fallback ==")
    inp, out = work / "b_input", work / "b_output"
    build_variant_b(inp)
    common = ["--input-base", inp, "--output-base", out,
              "--scratch-tmp", work / "b_scratch"]
    run([PY, HERE / "0_0_inspect_schemas.py", *common])

    dec = json.loads((out / "decisions.json").read_text())
    check("B decisions: partition style",
          dec["trace"]["partition_style"], "delta" if HAVE_DELTA else "plain-dirs")
    check("B decisions: revert mode", dec["trace"]["revert_mode"], "error_prefix")
    check("B decisions: addr format", dec["trace"]["addr_format"], "binary")
    check("B decisions: value format", dec["trace"]["value_format"], "binary")
    check("B decisions: withdrawals absent", dec["block"]["withdrawals_mode"], "absent")

    # canonical 6-column mode
    run([PY, HERE / "1_0_build_transfers.py", *common, "--all", "--no-debug-columns"])

    con = duckdb.connect()
    glob = str(out / "block_range=*" / "*.parquet")
    cols = [r[0] for r in con.execute(
        "SELECT column_name FROM (DESCRIBE SELECT * FROM "
        f"read_parquet('{glob}', hive_partitioning=false))"
    ).fetchall()]
    check("B canonical schema",
          cols, ["sender", "recipient", "block", "log_index", "value", "tx_hash"])
    con.close()

    rows = fetch_ledger(out, "block, log_index, sender, recipient, CAST(value AS VARCHAR), tx_hash")
    expected = [(blk, li, s, r, str(v), tx) for blk, li, s, r, v, tx in EXPECTED_B]
    check("B ledger (error-prefix propagation, binary normalization)", rows, expected)


def run_variant_c(work: Path) -> None:
    """Unit-level: Delta HEAD unreadable (non-contiguous log) -> empirical
    duplicate check decides raw-glob (append-only) vs pinned version (stale
    files present). Mirrors the cluster's 'non-contiguous log segment' state."""
    print("== Variant C: broken Delta HEAD — dup-check / version-pin logic ==")
    import importlib
    import logging as pylogging

    sys.path.insert(0, str(HERE))
    insp = importlib.import_module("0_0_inspect_schemas")
    insp.log = pylogging.getLogger("selftest_c")
    warnings: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)

    from deltalake import DeltaTable

    def tbl(lo: int, hi: int) -> pa.Table:
        return pa.table({
            "k": pa.array(list(range(lo, hi)), pa.int64()),
            "block_number": pa.array([5] * (hi - lo), pa.int64()),
            "partition": pa.array([0] * (hi - lo), pa.int64()),
        })

    def break_head(base: Path) -> None:
        (base / "_delta_log" / f"{1:020d}.json").unlink()  # gap above checkpoint

    def make_src(base: Path) -> dict:
        layout = insp.discover_partitions(base, warn)
        return {
            "base": str(base), "partitions": layout["partitions"],
            "partition_style": layout["style"],
            "delta_fallback_error": layout.get("delta_fallback_error"),
        }

    # c1: append-only -> dup check clean -> raw globbing kept (complete)
    c1 = work / "c1_trace"
    write_deltalake(str(c1), tbl(0, 10), partition_by=["partition"])
    DeltaTable(str(c1)).create_checkpoint()
    write_deltalake(str(c1), tbl(10, 20), partition_by=["partition"], mode="append")
    write_deltalake(str(c1), tbl(20, 30), partition_by=["partition"], mode="append")
    break_head(c1)
    src1 = make_src(c1)
    check("C1 HEAD unreadable -> raw fallback", bool(src1["delta_fallback_error"]), True)
    insp.resolve_delta_fallback(src1, "trace", "k", warn)
    check("C1 dup check clean", src1["dedup_check"]["duplicate_rows"], 0)
    check("C1 raw layout kept (complete)", src1["partition_style"] in ("hive", "plain-dirs"), True)
    n_files = sum(len(f) for f in
                  ([__import__('config').partition_files(p) for p in src1["partitions"]]))
    check("C1 all three commit files readable", n_files, 3)

    # c2: overwrite left stale files -> duplicates -> pinned to checkpoint version
    c2 = work / "c2_trace"
    write_deltalake(str(c2), tbl(0, 10), partition_by=["partition"])
    DeltaTable(str(c2)).create_checkpoint()
    write_deltalake(str(c2), tbl(0, 10), partition_by=["partition"], mode="overwrite")
    write_deltalake(str(c2), tbl(10, 20), partition_by=["partition"], mode="append")
    break_head(c2)
    src2 = make_src(c2)
    insp.resolve_delta_fallback(src2, "trace", "k", warn)
    check("C2 duplicates detected", src2["dedup_check"]["duplicate_rows"] > 0, True)
    check("C2 pinned to Delta version", src2["partition_style"], "delta-pinned")
    check("C2 pinned version is checkpoint", src2["delta_version"], 0)
    pinned_files = [f for p in src2["partitions"]
                    for f in __import__('config').partition_files(p)]
    check("C2 active set deduplicated", len(pinned_files), 1)


def run_random_pair_check(work: Path) -> None:
    """Unit-level: spot-check --random-pairs machinery — weighted pair
    sampling plus the single-scan batch balance query — against the known
    variant-A ledger (no RPC involved)."""
    print("== Random-pair sampling / batch balance check ==")
    import importlib
    import logging as pylog

    sys.path.insert(0, str(HERE))
    val = importlib.import_module("3_0_validate_transfer_ledger")
    from config import Config

    cfg = Config.defaults()
    cfg.output_base = work / "a_output"
    cfg.scratch_tmp = work / "a_scratch"
    cfg.duckdb_memory_limit = "2GB"
    cfg.duckdb_threads = 2
    con = val.open_con(cfg)
    lg = pylog.getLogger("selftest_pairs")

    sampled = val.sample_random_pairs(con, cfg, 10, 2.0, 0, 160, 0.5, lg)
    check("random pairs sampled", 0 < len(sampled) <= 10, True)
    check("random pairs are active pairs",
          all(any(blk == b and (s == a or r == a)
                  for blk, _li, s, r, *_ in EXPECTED_A)
              for a, b in sampled), True)

    pairs = sorted(set(sampled) | {(A, 199), (B, 199), (B, 150), (M, 20)})
    recon = val.reconstructed_balances_pairs(con, cfg, pairs)
    mismatches = [(a, b, recon[(a, b)], expected_balance(a, b))
                  for a, b in pairs if recon[(a, b)] != expected_balance(a, b)]
    if mismatches:
        print("  mismatches:", mismatches)
    check("batch pair balances match expected ledger", mismatches, [])
    con.close()


def run_aggregator_check(work: Path) -> None:
    """2_0_eth_transfer_aggregator: assert the exact exploded (address, block,
    log_index, value) ledger in 2_1 format, single-file and sharded modes."""
    print("== ETH transfer aggregator (2_1-format address ledger) ==")
    out_base = work / "a_output"
    common = ["--output-base", out_base, "--scratch-tmp", work / "a_scratch"]

    expected = []
    for blk, li, snd, rcp, val, *_ in EXPECTED_A:
        if snd is not None:
            expected.append((snd, blk, li, -val))
        if rcp is not None:
            expected.append((rcp, blk, li, val))
    expected.sort()

    def fetch(ledger_dir: Path):
        con = duckdb.connect()
        rows = con.execute(
            "SELECT address, block, log_index, CAST(value AS VARCHAR) "
            f"FROM read_parquet('{ledger_dir}/token_address=eth/*.parquet') "
            "ORDER BY address, block, log_index"
        ).fetchall()
        con.close()
        return [(a, b, li, int(v)) for a, b, li, v in rows]

    led1 = work / "a_ledger_single"
    run([PY, HERE / "2_0_eth_transfer_aggregator.py", "--output", led1,
         "--token", "eth", *common])
    check("aggregator single-file ledger", fetch(led1), expected)
    check("aggregator _SUCCESS marker",
          (led1 / "token_address=eth" / "_SUCCESS").exists(), True)

    led2 = work / "a_ledger_sharded"
    run([PY, HERE / "2_0_eth_transfer_aggregator.py", "--output", led2,
         "--token", "eth", "--shards", "3", *common])
    check("aggregator sharded ledger (3 shards)", sorted(fetch(led2)), expected)
    n_files = len(list((led2 / "token_address=eth").glob("*.parquet")))
    check("aggregator shard file count", n_files, 3)


def run_reward_merge_cap() -> None:
    """Unit-level: Erigon-style tracers synthesize phantom 2-ETH reward traces
    for post-Merge blocks; mint_rows_sql must cap rewards at the Merge."""
    print("== Reward Merge-cap unit check ==")
    import importlib

    sys.path.insert(0, str(HERE))
    bt = importlib.import_module("1_0_build_transfers")
    con = duckdb.connect()
    reward = ("CAST(NULL AS VARCHAR), CAST(NULL AS BIGINT), "
              f"CAST(NULL AS VARCHAR), '{M}', "
              "CAST(2000000000000000000 AS DECIMAL(38,0)), true, 'reward', "
              "CAST(NULL AS VARCHAR), 1, CAST(NULL AS VARCHAR), '', ''")
    con.execute(f"""
        CREATE TEMP VIEW trace_src (block, tx_hash, tx_index, from_addr,
            to_addr, value_wei, value_present, trace_type, call_type, status,
            error, ta_key, trace_address_str) AS
        SELECT 15537393, {reward}
        UNION ALL SELECT 15537394, {reward}
        UNION ALL SELECT 20000000, {reward}
    """)
    flags = {"include_rewards": True, "include_genesis": True}
    rows = con.execute(
        "SELECT block FROM (" + bt.mint_rows_sql({}, flags) + ")"
    ).fetchall()
    con.close()
    check("phantom post-Merge rewards excluded", rows, [(15537393,)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", help="reuse a fixed workdir instead of a temp dir")
    ap.add_argument("--keep", action="store_true", help="do not delete the workdir")
    args = ap.parse_args()
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="eth_selftest_"))
    if work.exists() and args.workdir:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    print(f"workdir: {work}")
    try:
        run_variant_a(work)
        run_random_pair_check(work)
        run_aggregator_check(work)
        run_variant_b(work)
        if HAVE_DELTA:
            run_variant_c(work)
        else:
            print("== Variant C skipped (deltalake not installed) ==")
        run_reward_merge_cap()
    finally:
        if not args.keep and not args.workdir:
            shutil.rmtree(work, ignore_errors=True)
    print("ALL SELF-TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
