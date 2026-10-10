#!/usr/bin/env python3
"""
4_1_verify_holdings.py

Purpose
-------
Verification of a 4_0 holdings snapshot BEFORE it feeds 5_0. Two layers:

1. [--rpc-url]  GROUND TRUTH: sample M random (wallet, block) rows from the
                holdings file and verify the ENTIRE portfolio against an archive
                node — ERC-20 via balanceOf(wallet) at the block, native ETH
                via eth_getBalance. Also probes tokens ABSENT from the
                portfolio (their chain balance must be 0) to catch missed
                credits, not just wrong ones.
2. [always]     INTERNAL INVARIANTS over the full holdings file (one scan):
                num_tokens == len(holdings), no negative/zero balances,
                no duplicate token per row, total_value_usd == sum(value_usd),
                per-block row counts.

CLI
---
  $ DUCKDB_THREADS=16 DUCKDB_MEMORY_LIMIT=64GB \
    python 4_1_verify_holdings.py \
        --holdings /path/to/recon-eth-erc20-2020_01-2022_01.parquet \
        --decimals-cache /path/to/token_decimals_cache.json \
        --rpc-url https://mainnet.infura.io/v3/<key> \
        --rpc-checks 25
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.log_utils import log  # noqa: E402

getcontext().prec = 200

ETH_PSEUDO_TOKEN = "0x" + "ee" * 20

ERC20_ABI = [
    {"constant": True, "inputs": [{"name": "_owner", "type": "address"}],
     "name": "balanceOf",
     "outputs": [{"name": "balance", "type": "uint256"}], "type": "function"},
]

def open_con(threads: int, memory: str,
             tmp_dir: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{memory}'")
    # GPFS misdetects free space -> DuckDB caps spill at a tiny size; be explicit.
    # Default lives next to the data (scratch), never in the quota-limited $HOME.
    tmp_dir = tmp_dir or os.getenv("DUCKDB_TMPDIR") or "./duckdb_tmp_4_1"
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    log(f"[duckdb] threads={threads} memory={memory} temp_directory={tmp_dir}")
    con.execute(f"SET temp_directory='{tmp_dir}'")
    con.execute("SET max_temp_directory_size='500GiB'")
    con.execute("SET preserve_insertion_order=false")
    return con


def holdings_to_map(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize a nested holdings list into {token: {balance, value_usd}}."""
    if raw is None:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for h in list(raw):
        tok = str(h.get("token_address", "")).lower()
        if tok:
            out[tok] = {"balance": h.get("balance"),
                        "value_usd": h.get("value_usd")}
    return out


def dec(x: Any) -> Optional[Decimal]:
    if x is None:
        return None
    try:
        return Decimal(str(x))
    except Exception:
        return None


# ======================= layer 1: RPC ground truth =======================

def rel_err(ours: int, chain: int) -> Decimal:
    """Relative difference between two raw-unit integers (0 when identical)."""
    if ours == chain:
        return Decimal(0)
    denom = max(abs(Decimal(ours)), abs(Decimal(chain)))
    if denom == 0:
        return Decimal("Infinity")
    return abs(Decimal(ours) - Decimal(chain)) / denom


def rpc_verify(con, holdings_path: str, rpc_url: str, decimals: Dict[str, int],
               all_tokens: List[str], n_checks: int, absent_checks: int,
               rpc_delay: float, seed: int,
               bal_rel_tol: Decimal = Decimal("1e-10")) -> bool:
    """Verify sampled portfolios against chain state.

    Comparison is in RAW integer units (wei / 10**decimals), never floats.
    A position counts as matching when it is either bit-exact or within
    `bal_rel_tol` relative error; both are reported separately so float-scale
    noise (~1e-16) stays visually distinct from a genuine reconstruction bug.
    """
    from web3 import HTTPProvider, Web3
    w3 = Web3(HTTPProvider(rpc_url))
    rng = random.Random(seed)
    worst: List[Tuple[Decimal, str]] = []

    log("=" * 74)
    log(f"[rpc] sampling {n_checks} random rows from the holdings file ...")
    sample = con.execute(f"""
        SELECT wallet_address, block_number, holdings
        FROM read_parquet('{holdings_path}')
        USING SAMPLE reservoir({n_checks} ROWS) REPEATABLE ({seed})
    """).fetchdf()

    def chain_erc20(token: str, wallet: str, block: int) -> Optional[int]:
        try:
            c = w3.eth.contract(Web3.to_checksum_address(token), abi=ERC20_ABI)
            return c.functions.balanceOf(
                Web3.to_checksum_address(wallet)).call(block_identifier=block)
        except Exception:
            return None  # no code at block / revert / non-standard

    stats = Counter()
    examples: List[str] = []
    for r in sample.itertuples(index=False):
        wallet, block = r.wallet_address, int(r.block_number)
        hmap = holdings_to_map(r.holdings)
        for tok, v in hmap.items():
            bal = dec(v["balance"])
            if tok == ETH_PSEUDO_TOKEN:
                try:
                    chain_wei = w3.eth.get_balance(
                        Web3.to_checksum_address(wallet), block_identifier=block)
                except Exception as e:
                    stats["rpc_failed"] += 1
                    continue
                ours_wei = int(bal * Decimal(10) ** 18)
                e = rel_err(ours_wei, chain_wei)
                desc = (f"ETH {wallet}@{block} ours={ours_wei} chain={chain_wei} "
                        f"delta={ours_wei - chain_wei} wei rel={e:.3e}")
                worst.append((e, desc))
                if ours_wei == chain_wei:
                    stats["eth_exact"] += 1
                elif e <= bal_rel_tol:
                    stats["eth_within_tol"] += 1
                else:
                    stats["eth_mismatch"] += 1
                    if len(examples) < 10:
                        examples.append(desc)
            else:
                d = int(decimals.get(tok, 18))
                chain_raw = chain_erc20(tok, wallet, block)
                time.sleep(rpc_delay)
                if chain_raw is None:
                    stats["erc20_skipped"] += 1
                    continue
                ours_raw = int(bal * Decimal(10) ** d)
                e = rel_err(ours_raw, chain_raw)
                desc = (f"{tok} {wallet}@{block} ours_raw={ours_raw} "
                        f"chain_raw={chain_raw} delta={ours_raw - chain_raw} "
                        f"rel={e:.3e}")
                worst.append((e, desc))
                if ours_raw == chain_raw:
                    stats["erc20_exact"] += 1
                elif e <= bal_rel_tol:
                    stats["erc20_within_tol"] += 1
                else:
                    stats["erc20_mismatch"] += 1
                    if len(examples) < 10:
                        examples.append(desc)
            time.sleep(rpc_delay)
        # absent-token probes: chain balance of tokens NOT in the portfolio
        # must be 0 (catches missed credits, not just wrong values)
        candidates = [t for t in rng.sample(all_tokens, min(20, len(all_tokens)))
                      if t not in hmap and t != ETH_PSEUDO_TOKEN][:absent_checks]
        for tok in candidates:
            chain_raw = chain_erc20(tok, wallet, block)
            time.sleep(rpc_delay)
            if chain_raw is None:
                stats["absent_skipped"] += 1
            elif chain_raw == 0:
                stats["absent_exact"] += 1
            else:
                stats["absent_mismatch"] += 1
                if len(examples) < 10:
                    examples.append(
                        f"ABSENT {tok} {wallet}@{block}: chain holds {chain_raw}, "
                        f"we hold nothing")

    log("-" * 74)
    log(f"[rpc] balance tolerance: rel <= {bal_rel_tol:.0e} "
        f"(raw integer units; exact counted separately)")
    for k in ("eth_exact", "eth_within_tol", "eth_mismatch",
              "erc20_exact", "erc20_within_tol", "erc20_mismatch",
              "erc20_skipped", "absent_exact", "absent_mismatch",
              "absent_skipped", "rpc_failed"):
        if stats[k]:
            log(f"[rpc] {k:<16}: {stats[k]}")

    # Error-magnitude histogram: float64 transport noise clusters at 1e-16..1e-17,
    # a genuine reconstruction bug shows up orders of magnitude higher.
    nonzero = sorted((e for e, _ in worst if e > 0), reverse=True)
    if nonzero:
        buckets: Counter = Counter()
        for e in nonzero:
            buckets[int(math.floor(math.log10(float(e))))] += 1
        log("[rpc] relative-error distribution (non-exact positions):")
        for exp in sorted(buckets, reverse=True):
            log(f"    1e{exp:<4} .. 1e{exp + 1:<4} : {buckets[exp]}")
        log(f"[rpc] worst relative error: {nonzero[0]:.3e}")
        log("[rpc] worst offenders:")
        for e, d in sorted(worst, key=lambda x: -x[0])[:5]:
            if e > 0:
                log(f"    {d}")
    for ex in examples:
        log(f"    e.g. {ex}")

    checked = sum(stats[k] for k in (
        "eth_exact", "eth_within_tol", "eth_mismatch",
        "erc20_exact", "erc20_within_tol", "erc20_mismatch"))
    exact = stats["eth_exact"] + stats["erc20_exact"]
    bad = stats["eth_mismatch"] + stats["erc20_mismatch"] + stats["absent_mismatch"]
    log(f"[rpc] positions checked: {checked} | bit-exact: {exact} | "
        f"beyond tolerance: {bad}")
    log(f"[rpc] RESULT: {'PASS' if bad == 0 else 'NOT_PASS'}")
    return bad == 0


# ======================= layer 2: internal invariants =======================

def invariants(con, holdings_path: str) -> bool:
    log("=" * 74)
    log("[invariants] full-file scan ...")
    t0 = time.perf_counter()
    # Single streaming scan: per-row list aggregates only (no window/unnest,
    # so nothing here needs to sort or spill).
    n_rows, bad_ntok, bad_dup, neg_bal, zero_bal, bad_total = con.execute(f"""
        WITH x AS (
          SELECT num_tokens, holdings,
                 TRY_CAST(total_value_usd AS DOUBLE) AS tot,
                 COALESCE(list_sum(list_transform(
                     holdings, h -> COALESCE(h.value_usd, 0))), 0) AS vsum
          FROM read_parquet('{holdings_path}')
        )
        SELECT count(*),
               count(*) FILTER (WHERE num_tokens <> len(holdings)),
               count(*) FILTER (
                   WHERE len(holdings) <> len(list_distinct(
                       list_transform(holdings, h -> h.token_address)))),
               COALESCE(SUM(len(list_filter(
                   holdings, h -> h.balance LIKE '-%'))), 0),
               COALESCE(SUM(len(list_filter(
                   holdings, h -> h.balance = '0'))), 0),
               count(*) FILTER (WHERE tot IS NOT NULL AND
                   abs(tot - vsum) > greatest(1e-6, abs(tot) * 1e-6))
        FROM x
    """).fetchone()
    log(f"[invariants] rows                          : {n_rows:,}  "
        f"({time.perf_counter() - t0:.0f}s)")
    log(f"[invariants] num_tokens != len(holdings)   : {bad_ntok}   (must be 0)")
    log(f"[invariants] duplicate token within a row  : {bad_dup}   (must be 0)")
    log(f"[invariants] negative balances             : {neg_bal}   (must be 0)")
    log(f"[invariants] zero balances stored          : {zero_bal}   (must be 0)")
    log(f"[invariants] total != sum(value_usd)       : {bad_total}   (must be 0)")
    passed = not any((bad_ntok, bad_dup, neg_bal, zero_bal, bad_total))
    log(f"[invariants] RESULT: {'PASS' if passed else 'NOT_PASS'}")
    return passed


# ======================= main =======================

def main() -> int:
    p = argparse.ArgumentParser(description="Verify a 4_0 holdings snapshot")
    p.add_argument("--holdings", required=True,
                   help="holdings parquet written by 4_0 (nested format)")
    p.add_argument("--rpc-url", default=os.getenv("WEB3_INFURA_URL"))
    p.add_argument("--decimals-cache", default=None,
                   help="token decimals JSON (required for --rpc-url)")
    p.add_argument("--rpc-checks", type=int, default=25,
                   help="rows whose FULL portfolio is verified on-chain")
    p.add_argument("--absent-checks", type=int, default=1,
                   help="per sampled wallet: probe this many tokens NOT held "
                        "(chain balance must be 0)")
    p.add_argument("--rpc-delay", type=float, default=0.25)
    p.add_argument("--balance-rel-tol", default="1e-10",
                   help="relative tolerance for on-chain balance comparison "
                        "(raw integer units). 0 = require bit-exact equality. "
                        "Bit-exact matches are always counted and reported "
                        "separately from within-tolerance ones.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=int(os.getenv("DUCKDB_THREADS", "16")))
    p.add_argument("--memory", default=os.getenv("DUCKDB_MEMORY_LIMIT", "64GB"))
    p.add_argument("--tmp-dir", default=None,
                   help="DuckDB spill directory (default: next to --holdings, so it "
                        "lands on scratch and not in the $HOME quota)")
    args = p.parse_args()

    tmp_dir = args.tmp_dir or os.getenv("DUCKDB_TMPDIR") or str(
        Path(args.holdings).resolve().parent / "duckdb_tmp_4_1")
    con = open_con(args.threads, args.memory, tmp_dir)
    results: Dict[str, bool] = {}

    results["invariants"] = invariants(con, args.holdings)

    if args.rpc_url:
        decimals: Dict[str, int] = {}
        if args.decimals_cache and Path(args.decimals_cache).exists():
            decimals = {k.lower(): int(v) for k, v in
                        json.loads(Path(args.decimals_cache).read_text()).items()}
        else:
            log("[rpc] WARN no decimals cache — assuming 18 for all tokens")
        # Token universe for absent-probes. Sample ROWS first (bounded reservoir),
        # THEN unnest: a percentage sample after the unnest explodes to >100M
        # strings and the DISTINCT spills.
        all_tokens = [r[0] for r in con.execute(f"""
            SELECT DISTINCT h.token_address
            FROM (
                SELECT holdings FROM read_parquet('{args.holdings}')
                USING SAMPLE reservoir(200000 ROWS) REPEATABLE ({args.seed})
            ), UNNEST(holdings) AS t(h)
        """).fetchall()]
        log(f"[rpc] token universe for absent-probes: {len(all_tokens)} tokens")
        results["rpc"] = rpc_verify(
            con, args.holdings, args.rpc_url, decimals, all_tokens,
            args.rpc_checks, args.absent_checks, args.rpc_delay, args.seed,
            Decimal(args.balance_rel_tol))

    log("=" * 74)
    log("SUMMARY: " + " | ".join(
        f"{k}={'PASS' if v else 'NOT_PASS'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
