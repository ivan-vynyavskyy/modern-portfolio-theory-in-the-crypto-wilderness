#!/usr/bin/env python3
"""
3_1_validate_flow_ledger.py

Purpose
-------
ETH analog of ../erc20_src/3_0_validation_runner.py: spot-check the aggregated
native-ETH address-flow ledger (output of 2_0_eth_transfer_aggregator.py) against
on-chain truth.

Differences from 3_0 (ERC-20) that this script accounts for:
- ONE pseudo-token directory (token_address=0xee...ee) holding N address-hash
  shard files instead of many per-token files; every file is sorted by
  (address, block, log_index), so DuckDB prunes row groups by the address
  zone maps and per-address queries stay fast without knowing the shard.
- Truth is `eth_getBalance` (native balance, archive node required), not
  ERC-20 `balanceOf` -- no contract existence probe, no decimals.
- Native reconstruction is exact integer wei, so the default tolerance is
  0 wei (any nonzero delta is a genuine defect; the one known source-data
  defect class is constant negative deltas on MEV builder/fee-recipient
  addresses).
- Sampling is activity-weighted (row-sampled) with a later-block bias,
  matching how the transfer ledger itself was validated.

Inputs
------
- <data_dir>/token_address=<token>/*.parquet
  (columns: address, block, log_index, value DECIMAL(38,0), signed flows)
- --rpc-url or $WEB3_INFURA_URL (archive endpoint)

Outputs
-------
- <data_dir>/validation_status.tsv          (token -> PASS|NOT_PASS)
- eth_spot_deltas.csv (--out-csv)           per-check comparison rows

CLI
---
  $ DUCKDB_THREADS=16 DUCKDB_MEMORY_LIMIT=64GB \
    python 3_1_validate_flow_ledger.py --data-dir /path/to/ledgers \
        --trials 200 --rpc-url https://mainnet.infura.io/v3/<key>
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
from web3 import HTTPProvider, Web3

sys.path.append(str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    LONDON_BLOCK,
    MERGE_BLOCK,
    SHAPELLA_BLOCK,
    setup_logging,
)

ETH_PSEUDO_TOKEN = "0x" + "ee" * 20
WEI = 10**18

ERAS = [
    ("pre-London", 0, LONDON_BLOCK - 1),
    ("London-Merge", LONDON_BLOCK, MERGE_BLOCK - 1),
    ("Merge-Shapella", MERGE_BLOCK, SHAPELLA_BLOCK - 1),
    ("post-Shapella", SHAPELLA_BLOCK, 10**12),
]


def era_of(block: int) -> str:
    for name, lo, hi in ERAS:
        if lo <= block <= hi:
            return name
    return "unknown"


# ----------------------------- DuckDB --------------------------------
def open_connection(threads: int, memory: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{memory}'")
    return con


def sample_pairs(
    con: duckdb.DuckDBPyConnection,
    files: list[Path],
    trials: int,
    max_block: int,
    bias_power: float,
    rng: random.Random,
    log,
) -> list[tuple[str, int]]:
    """Sample (address, block) pairs, activity-weighted with later-block bias.

    Row-sampling a shard file weights addresses by how many ledger rows they
    have (active addresses checked more -- same weighting the transfer-ledger
    spot-check used). The sampled row also gives a block where the address
    was already active; the check block is drawn in [that block, max_block]
    biased toward max_block (u^(1/power), power>1 => later blocks).
    """
    per_file = max(1, trials // max(1, min(len(files), 8)))
    chosen = rng.sample(files, k=min(len(files), max(1, trials // per_file)))
    pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for f in chosen:
        want = min(per_file, trials - len(pairs))
        if want <= 0:
            break
        rows = con.execute(
            f"SELECT address, block FROM read_parquet('{f}') "
            f"USING SAMPLE {want * 2} ROWS"
        ).fetchall()
        rng.shuffle(rows)
        for addr, first_seen in rows:
            if len(pairs) >= trials:
                break
            lo, hi = int(first_seen), max_block
            blk = lo + int((hi - lo) * (rng.random() ** (1.0 / bias_power)))
            key = (addr, blk)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    log.info("sampled %d (address, block) pairs from %d shard files",
             len(pairs), len(chosen))
    return pairs


def local_balance(
    con: duckdb.DuckDBPyConnection, glob: str, addr: str, blk: int
) -> int:
    """Cumulative signed flow (= balance) for address up to block.

    Shard files are sorted by address, so the address predicate prunes to a
    handful of row groups via parquet zone maps -- no shard lookup needed.
    """
    res = con.execute(
        f"SELECT COALESCE(SUM(CAST(value AS HUGEINT)), 0) "
        f"FROM read_parquet('{glob}') "
        f"WHERE address = ? AND block <= ?", [addr, blk]
    ).fetchone()[0]
    return int(res)


def last_local_rows(
    con: duckdb.DuckDBPyConnection, glob: str, addr: str, blk: int, n: int = 15
):
    """Last n ledger rows for address at/below block (fail diagnostics --
    an ETH address can hold millions of rows, never print them all)."""
    return con.execute(
        f"SELECT block, log_index, value FROM read_parquet('{glob}') "
        f"WHERE address = ? AND block <= ? "
        f"ORDER BY block DESC, log_index DESC LIMIT {n}", [addr, blk]
    ).fetchall()


# ----------------------------- Web3 ----------------------------------
def chain_balance(w3: Web3, addr: str, blk: int, retries: int = 6,
                  log=None) -> int | None:
    """eth_getBalance with exponential backoff on rate limits/transient
    errors (same failure modes 3_0 hit on Infura)."""
    delay = 2.0
    for attempt in range(retries):
        try:
            return w3.eth.get_balance(
                Web3.to_checksum_address(addr), block_identifier=blk
            )
        except Exception as exc:  # rate limit / transient network
            msg = str(exc)
            if attempt == retries - 1:
                if log:
                    log.warning("RPC failed for %s @ %d: %s", addr, blk, msg)
                return None
            if log and attempt == 0:
                log.debug("RPC retry for %s @ %d: %s", addr, blk, msg)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    return None


# ----------------------------- Status --------------------------------
def write_status(path: Path, token: str, status: str) -> None:
    current: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                t, s = line.split("\t", 1)
                current[t.lower()] = s.upper()
    current[token.lower()] = status.upper()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for t, s in sorted(current.items()):
            fh.write(f"{t}\t{s}\n")
    tmp.replace(path)


# ----------------------------- Main ----------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Spot-check the aggregated ETH address-flow ledger "
                    "against eth_getBalance (archive node)."
    )
    p.add_argument("--data-dir", required=True,
                   help="ledgers root containing token_address=<token>/")
    p.add_argument("--token", default=ETH_PSEUDO_TOKEN,
                   help=f"pseudo token dir name (default {ETH_PSEUDO_TOKEN})")
    p.add_argument("--trials", type=int, default=200,
                   help="number of random (address, block) checks")
    p.add_argument("--min-block", type=int, default=None)
    p.add_argument("--max-block", type=int, default=None)
    p.add_argument("--tolerance-wei", type=int, default=0,
                   help="absolute wei diff tolerated before FAIL (default 0 "
                        "-- native reconstruction is wei-exact)")
    p.add_argument("--block-bias-power", type=float, default=2.0,
                   help="later-block bias; 1.0 = uniform (default 2.0)")
    p.add_argument("--rpc-url", default=os.getenv("WEB3_INFURA_URL"),
                   help="archive JSON-RPC endpoint (or $WEB3_INFURA_URL)")
    p.add_argument("--rpc-delay", type=float, default=0.25,
                   help="seconds between RPC calls (default 0.25 = 4/s)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int,
                   default=int(os.getenv("DUCKDB_THREADS", "16")))
    p.add_argument("--memory", default=os.getenv("DUCKDB_MEMORY_LIMIT", "64GB"))
    p.add_argument("--out-csv", default="eth_spot_deltas.csv")
    p.add_argument("--status-file", default=None,
                   help="default: DATA_DIR/validation_status.tsv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    token = args.token.lower()
    token_dir = data_dir / f"token_address={token}"
    log = setup_logging(Path(__file__).resolve().parent / "logs",
                        "eth_ledger_validate")

    if not args.rpc_url:
        log.error("RPC endpoint required: --rpc-url or $WEB3_INFURA_URL")
        return 2
    files = sorted(p for p in token_dir.glob("*.parquet") if p.is_file())
    if not files:
        log.error("no shard files under %s", token_dir)
        return 1
    glob = str(token_dir / "*.parquet")
    status_file = (Path(args.status_file).expanduser().resolve()
                   if args.status_file else data_dir / "validation_status.tsv")

    con = open_connection(args.threads, args.memory)
    rng = random.Random(args.seed)
    w3 = Web3(HTTPProvider(args.rpc_url))
    log.info("validating %s: %d shard files, trials=%d, tolerance=%d wei, "
             "threads=%d, memory=%s", token, len(files), args.trials,
             args.tolerance_wei, args.threads, args.memory)

    # ledger block bounds (parquet footer stats -- instant)
    lo_blk, hi_blk = con.execute(
        f"SELECT MIN(block), MAX(block) FROM read_parquet('{glob}')"
    ).fetchone()
    lo_blk = max(int(lo_blk), args.min_block or 0)
    hi_blk = min(int(hi_blk), args.max_block or int(hi_blk))
    log.info("ledger block range used: %s - %s", f"{lo_blk:,}", f"{hi_blk:,}")

    t0 = time.perf_counter()
    pairs = sample_pairs(con, files, args.trials, hi_blk,
                         args.block_bias_power, rng, log)

    summary = Counter()
    per_era: dict[str, Counter] = {}
    results = []
    for i, (addr, blk) in enumerate(pairs, 1):
        local = local_balance(con, glob, addr, blk)
        chain = chain_balance(w3, addr, blk, log=log)
        time.sleep(args.rpc_delay)
        if chain is None:
            summary["skipped"] += 1
            continue
        delta = local - chain
        era = era_of(blk)
        ec = per_era.setdefault(era, Counter())
        ec["n"] += 1
        results.append((addr, blk, era, local, chain, delta))
        if abs(delta) <= args.tolerance_wei:
            summary["pass"] += 1
            ec["pass"] += 1
            log.info("  PASS  [%d/%d] %s @ %d", i, len(pairs), addr, blk)
        else:
            summary["fail"] += 1
            ec["fail"] += 1
            log.info("  FAIL  [%d/%d] %s @ %d", i, len(pairs), addr, blk)
            log.info("        local = %d", local)
            log.info("        chain = %d", chain)
            log.info("        delta = %d wei (%.9f ETH)", delta, delta / WEI)
            log.info("        last ledger rows at/below block:")
            for b, li, v in last_local_rows(con, glob, addr, blk):
                log.info("          block=%d log_index=%d value=%s", b, li, v)

    # CSV
    out_csv = Path(args.out_csv).expanduser().resolve()
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["address", "block", "era", "local_wei", "chain_wei",
                     "delta_wei"])
        wr.writerows(results)
    log.info("wrote %s", out_csv)

    # summary
    checked = summary["pass"] + summary["fail"]
    log.info("=" * 70)
    log.info("SUMMARY (%d checks, %.0fs)", checked, time.perf_counter() - t0)
    for name, _, _ in ERAS:
        ec = per_era.get(name)
        if ec:
            log.info("  %-14s: %3d checks, %3d exact (%.1f%%)",
                     name, ec["n"], ec["pass"], 100.0 * ec["pass"] / ec["n"])
    log.info("  overall       : %d/%d exact (%.2f%%), %d skipped (RPC)",
             summary["pass"], checked,
             100.0 * summary["pass"] / checked if checked else 0.0,
             summary["skipped"])
    if summary["fail"]:
        worst = sorted(results, key=lambda r: -abs(r[5]))[:10]
        log.info("  worst deltas:")
        for addr, blk, _, _, _, d in worst:
            if d != 0:
                log.info("    %s @ %d  delta=%+.9f ETH", addr, blk, d / WEI)
        log.info("  note: constant negative deltas on MEV builder/fee-"
                 "recipient addresses are the known source-data defect "
                 "class")

    status = "PASS" if summary["fail"] == 0 and checked > 0 else "NOT_PASS"
    write_status(status_file, token, status)
    log.info("result for %s: %s (recorded in %s)", token, status,
             status_file.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
