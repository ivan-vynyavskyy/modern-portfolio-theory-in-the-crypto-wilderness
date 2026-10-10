#!/usr/bin/env python3
"""
3_0_validate_transfer_ledger.py

Purpose
-------
Validation of the reconstructed ETH ledger — "how good is it". Three subcommands:

  balance     Reconstructed balance(addr, B) for given (addr, block) pairs
              via the supply identity.
  supply      Cumulative minted-burned at snapshot blocks, optionally
              compared against a known supply curve CSV
              (columns: block, supply_wei or supply_eth).
  spot-check  Diff reconstructed balances against eth_getBalance ground
              truth over JSON-RPC (requires an ARCHIVE node for historical
              blocks) and report the delta distribution. Two modes:
              --block B: sample addresses at one snapshot block.
              --random-pairs N: sample N (address, block) pairs from the
              ledger's own activity, with later blocks weighted more heavily
              (density ~ block^(power-1); default power 2 = linear ramp).
              Every sampled address is active at its sampled block. The
              summary is reported overall AND per era (pre-London /
              London-Merge / Merge-Shapella / post-Shapella), which reveals
              WHICH component (gas split, rewards, withdrawals) drives drift.

CLI
---
  $ python 3_0_validate_transfer_ledger.py balance --block 15000000 --addresses 0xabc...,0xdef...
  $ python 3_0_validate_transfer_ledger.py balance --block 15000000 --addresses-file addrs.txt
  $ python 3_0_validate_transfer_ledger.py supply --snapshot-blocks blocks.csv --supply-csv known.csv
  $ python 3_0_validate_transfer_ledger.py spot-check --block 15000000 --sample-size 2000 \
        --rpc-url https://... [--classify-contracts]
  $ python 3_0_validate_transfer_ledger.py spot-check --random-pairs 10000 --rpc-url https://... \
        [--block-weight-power 2] [--seed 42] [--out-csv deltas.csv]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import duckdb

sys.path.append(str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    Config,
    LONDON_BLOCK,
    MERGE_BLOCK,
    SHAPELLA_BLOCK,
    add_config_args,
    setup_logging,
)

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
    return "?"


def ledger_glob(cfg: Config) -> str:
    return str(cfg.output_base / "block_range=*" / "*.parquet")


def open_con(cfg: Config) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg.duckdb_memory_limit}'")
    con.execute(f"SET threads={cfg.duckdb_threads}")
    cfg.scratch_tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{cfg.scratch_tmp}'")
    return con


# ----------------------------- balance -----------------------------

BALANCE_SQL = """
WITH targets AS (SELECT unnest(?) AS addr),
flows AS (
  SELECT recipient AS addr, value AS inflow, CAST(0 AS DECIMAL(38,0)) AS outflow
  FROM read_parquet('{glob}')
  WHERE block <= ? AND recipient IN (SELECT addr FROM targets)
  UNION ALL
  SELECT sender AS addr, CAST(0 AS DECIMAL(38,0)), value
  FROM read_parquet('{glob}')
  WHERE block <= ? AND sender IN (SELECT addr FROM targets)
)
SELECT t.addr,
       COALESCE(SUM(f.inflow), 0) - COALESCE(SUM(f.outflow), 0) AS balance_wei
FROM targets t LEFT JOIN flows f USING (addr)
GROUP BY t.addr
ORDER BY t.addr
"""


def reconstructed_balances(con, cfg: Config, addresses: list[str], block: int) -> dict[str, int]:
    addresses = [a.strip().lower() for a in addresses if a.strip()]
    rows = con.execute(
        BALANCE_SQL.format(glob=ledger_glob(cfg)), [addresses, block, block]
    ).fetchall()
    return {a: int(b) for a, b in rows}


def cmd_balance(args, cfg: Config, log) -> int:
    if args.addresses_file:
        addresses = Path(args.addresses_file).read_text().split()
    else:
        addresses = args.addresses.split(",")
    con = open_con(cfg)
    balances = reconstructed_balances(con, cfg, addresses, args.block)
    for addr, wei in balances.items():
        log.info("%s  %26d wei  (%.6f ETH)", addr, wei, wei / WEI)
    if args.out_csv:
        with open(args.out_csv, "w") as fh:
            fh.write("address,block,balance_wei\n")
            for addr, wei in balances.items():
                fh.write(f"{addr},{args.block},{wei}\n")
        log.info("wrote %s", args.out_csv)
    return 0


# ----------------------------- supply -----------------------------

def load_snapshot_blocks(spec: str) -> list[int]:
    p = Path(spec)
    if p.exists():
        blocks = []
        for line in p.read_text().splitlines():
            tok = line.strip().split(",")[0]
            if tok and tok.isdigit():
                blocks.append(int(tok))
        return sorted(blocks)
    return sorted(int(x) for x in spec.split(",") if x.strip())


def cmd_supply(args, cfg: Config, log) -> int:
    blocks = load_snapshot_blocks(args.snapshot_blocks)
    con = open_con(cfg)
    con.execute(
        "CREATE TEMP TABLE snaps AS SELECT unnest(?) AS snap_block", [blocks]
    )
    # Per-block minted/burned first (millions of rows), then cumulative at snapshots.
    rows = con.execute(f"""
        WITH per_block AS (
          SELECT block,
                 COALESCE(SUM(value) FILTER (WHERE sender IS NULL), 0)    AS minted,
                 COALESCE(SUM(value) FILTER (WHERE recipient IS NULL), 0) AS burned
          FROM read_parquet('{ledger_glob(cfg)}')
          GROUP BY block
        ),
        cum AS (
          SELECT block,
                 SUM(minted) OVER (ORDER BY block) AS cum_minted,
                 SUM(burned) OVER (ORDER BY block) AS cum_burned
          FROM per_block
        )
        SELECT s.snap_block,
               (SELECT MAX(cum_minted) FROM cum WHERE block <= s.snap_block) AS minted,
               (SELECT MAX(cum_burned) FROM cum WHERE block <= s.snap_block) AS burned
        FROM snaps s ORDER BY s.snap_block
    """).fetchall()

    known: dict[int, int] = {}
    if args.supply_csv:
        import csv

        with open(args.supply_csv) as fh:
            for row in csv.DictReader(fh):
                blk = int(row.get("block") or row.get("block_number"))
                if "supply_wei" in row and row["supply_wei"]:
                    known[blk] = int(row["supply_wei"])
                elif "supply_eth" in row and row["supply_eth"]:
                    known[blk] = int(float(row["supply_eth"]) * WEI)

    out_lines = ["snap_block,reconstructed_supply_wei,known_supply_wei,delta_wei,delta_eth"]
    log.info("%12s | %24s | %24s | %s", "block", "reconstructed ETH", "known ETH", "delta ETH")
    for blk, minted, burned in rows:
        minted = int(minted or 0)
        burned = int(burned or 0)
        recon = minted - burned
        k = known.get(blk)
        delta = (recon - k) if k is not None else None
        log.info(
            "%12d | %24.4f | %24s | %s",
            blk, recon / WEI,
            f"{k / WEI:.4f}" if k is not None else "-",
            f"{delta / WEI:+.4f}" if delta is not None else "-",
        )
        out_lines.append(
            f"{blk},{recon},{k if k is not None else ''},"
            f"{delta if delta is not None else ''},"
            f"{delta / WEI if delta is not None else ''}"
        )
    if args.out_csv:
        Path(args.out_csv).write_text("\n".join(out_lines) + "\n")
        log.info("wrote %s", args.out_csv)
    log.info("A large gap points at missing withdrawals (post-Apr-2023) or "
             "missing pre-Merge rewards.")
    return 0


# ----------------------------- spot-check -----------------------------

def _batch_level_error(out: list):
    """Error object when the response is a rejection instead of per-call
    results. Providers differ: Infura returns a BARE error object in a list
    ([{'code': -32005, 'message': 'Too Many Requests', ...}]) — no 'error'
    wrapper and no id — others wrap it or return a plain dict."""
    for r in out:
        if isinstance(r, dict) and r.get("id") is None:
            if isinstance(r.get("error"), dict):
                return r["error"]
            if "code" in r or "message" in r:
                return r
    return None


def _is_rate_limit(err) -> bool:
    if isinstance(err, dict) and err.get("code") == -32005:
        return True
    s = str(err).lower()
    return "too many" in s or "rate" in s or "429" in s


def rpc_batch(url: str, calls: list[dict], timeout: int = 120) -> list:
    """JSON-RPC batch, robust against provider quirks: results are matched
    by id (not order); 429 / -32005 rate limits retry with exponential
    backoff (up to ~1 min); a non-rate-limit batch rejection (e.g. batch too
    large) splits the batch in half and recurses down to single calls."""
    import time as _time

    import requests

    last_err = None
    for attempt in range(7):
        try:
            resp = requests.post(url, json=calls, timeout=timeout)
        except requests.RequestException as exc:
            last_err = exc
            _time.sleep(min(60, 2 ** attempt))
            continue
        if resp.status_code == 429:
            last_err = "HTTP 429 rate limited"
            _time.sleep(min(60, 2 ** attempt))
            continue
        resp.raise_for_status()
        out = resp.json()
        if isinstance(out, dict):
            out = [out]
        by_id = {r["id"]: r for r in out
                 if isinstance(r, dict) and r.get("id") is not None}
        if len(by_id) == len(calls):
            return [by_id[c["id"]] for c in calls]
        batch_err = _batch_level_error(out)
        if _is_rate_limit(batch_err):
            last_err = batch_err
            _time.sleep(min(60, 2 ** attempt))
            continue
        if len(calls) > 1:  # e.g. batch too large for the provider -> split
            mid = len(calls) // 2
            return rpc_batch(url, calls[:mid], timeout) + \
                rpc_batch(url, calls[mid:], timeout)
        return [{"error": batch_err or {"message": f"malformed response: {out!r:.200}"}}]
    raise RuntimeError(
        f"RPC rate-limited/failing after 7 backoff attempts: {last_err} — "
        "lower --rpc-batch-size and/or raise --rpc-delay (Infura free tier "
        "sustains only a few eth_getBalance calls per second)"
    )


def sample_random_pairs(con, cfg: Config, n: int, power: float, lo: int,
                        hi: int, seed: float, log) -> list[tuple[str, int]]:
    """Weighted random sample of N (address, block) pairs from the ledger's
    own rows (so every address is active at its block), later blocks weighted
    by density ~ block^(power-1) via Efraimidis-Spirakis top-N sampling
    (key = u^(1/w), take largest keys). One full ledger scan."""
    con.execute(f"SELECT setseed({seed})")
    span = hi - lo + 1
    rows = con.execute(f"""
        SELECT addr, block FROM (
          SELECT addr, block,
                 pow(random(), 1.0 / greatest(
                     pow(CAST(block - {lo} + 1 AS DOUBLE) / {span}, {power} - 1),
                     1e-12)) AS k
          FROM (
            SELECT COALESCE(recipient, sender) AS addr, block
            FROM read_parquet('{ledger_glob(cfg)}')
            WHERE block BETWEEN {lo} AND {hi}
          )
          WHERE addr IS NOT NULL
        )
        ORDER BY k DESC
        LIMIT {n}
    """).fetchall()
    pairs = sorted({(a, b) for a, b in rows})
    log.info("sampled %d unique (address, block) pairs (requested %d, "
             "weight power %.1f, blocks %d-%d)", len(pairs), n, power, lo, hi)
    for name, elo, ehi in ERAS:
        cnt = sum(1 for _, b in pairs if elo <= b <= ehi)
        if cnt:
            log.info("  %-14s: %6d pairs", name, cnt)
    return pairs


def reconstructed_balances_pairs(con, cfg: Config,
                                 pairs: list[tuple[str, int]]) -> dict:
    """balance(addr, B) for many (addr, B) pairs in ONE ledger scan:
    gather all flows touching any sampled address, then aggregate each pair
    over flows with block <= its snapshot block."""
    import pyarrow as pa

    con.register("check_pairs", pa.table({
        "addr": [a for a, _ in pairs],
        "snap_block": pa.array([b for _, b in pairs], pa.int64()),
    }))
    rows = con.execute(f"""
        WITH flows AS (
          SELECT sender AS addr, block, -value AS delta
          FROM read_parquet('{ledger_glob(cfg)}')
          WHERE sender IN (SELECT DISTINCT addr FROM check_pairs)
          UNION ALL
          SELECT recipient, block, value
          FROM read_parquet('{ledger_glob(cfg)}')
          WHERE recipient IN (SELECT DISTINCT addr FROM check_pairs)
        )
        SELECT p.addr, p.snap_block, COALESCE(SUM(f.delta), 0)
        FROM check_pairs p
        LEFT JOIN flows f ON f.addr = p.addr AND f.block <= p.snap_block
        GROUP BY 1, 2
    """).fetchall()
    con.unregister("check_pairs")
    return {(a, int(b)): int(v) for a, b, v in rows}


def fetch_truth_rpc(pairs, rpc_url, batch_size, delay, log) -> dict:
    """eth_getBalance per (addr, block) pair — needs an ARCHIVE node.
    Paced like 3_0_validation_runner: sleep `delay` seconds between batches
    (hosted providers cap credits/second, e.g. Infura's -32005)."""
    import time as _time

    truth: dict = {}
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        calls = [
            {"jsonrpc": "2.0", "id": j, "method": "eth_getBalance",
             "params": [a, hex(b)]}
            for j, (a, b) in enumerate(chunk)
        ]
        for (a, b), r in zip(chunk, rpc_batch(rpc_url, calls)):
            if "result" in r and r["result"] is not None:
                truth[(a, b)] = int(r["result"], 16)
            else:
                log.warning("RPC error for %s @ %d: %s", a, b, r.get("error"))
        if (i // batch_size) % 20 == 19:
            log.info("  RPC progress: %d/%d pairs", min(i + batch_size, len(pairs)),
                     len(pairs))
        if delay > 0 and i + batch_size < len(pairs):
            _time.sleep(delay)
    return truth


def summarize_deltas(deltas, tol, label, log) -> None:
    """One accuracy block for a set of delta rows
    (addr, block, recon, truth, delta, rel)."""
    abs_d = sorted(abs(d) for *_, d, _r in deltas)
    n = len(abs_d)
    if not n:
        return

    def pct(p):
        return abs_d[min(n - 1, int(p * n))]

    exact = sum(1 for d in abs_d if d == 0)
    within = sum(1 for d in abs_d if d <= tol)
    over = sum(1 for *_, d, _r in deltas if d > 0)
    under = sum(1 for *_, d, _r in deltas if d < 0)
    log.info("  [%s] n=%d", label, n)
    log.info("    exact matches     : %d (%.2f%%)", exact, 100 * exact / n)
    log.info("    within %d wei: %d (%.2f%%)", tol, within, 100 * within / n)
    log.info("    median abs delta  : %.9f ETH", statistics.median(abs_d) / WEI)
    log.info("    95p    abs delta  : %.9f ETH", pct(0.95) / WEI)
    log.info("    max    abs delta  : %.9f ETH", abs_d[-1] / WEI)
    log.info("    overstated / understated: %d / %d", over, under)


def cmd_spot_check(args, cfg: Config, log) -> int:
    con = open_con(cfg)

    # ---- build the (address, block) pair list ----
    if args.random_pairs:
        end = args.end_block
        if end is None:
            end = con.execute(
                f"SELECT MAX(block) FROM read_parquet('{ledger_glob(cfg)}')"
            ).fetchone()[0]
        log.info("sampling %d random (address, block) pairs, later-block "
                 "weighted (one full ledger scan) ...", args.random_pairs)
        pairs = sample_random_pairs(con, cfg, args.random_pairs,
                                    args.block_weight_power, args.start_block,
                                    end, args.seed, log)
    else:
        block = args.block
        log.info("sampling %d addresses active up to block %d ...",
                 args.sample_size, block)
        addrs = [r[0] for r in con.execute(f"""
            SELECT DISTINCT addr FROM (
              SELECT sender AS addr FROM read_parquet('{ledger_glob(cfg)}') WHERE block <= ?
              UNION ALL
              SELECT recipient FROM read_parquet('{ledger_glob(cfg)}') WHERE block <= ?
            ) WHERE addr IS NOT NULL
            ORDER BY hash(addr) LIMIT ?
        """, [block, block, args.sample_size]).fetchall()]
        pairs = [(a, block) for a in addrs]
        log.info("sampled %d unique addresses", len(pairs))

    # ---- reconstructed balances (one scan for all pairs) ----
    log.info("computing %d reconstructed balances ...", len(pairs))
    recon = reconstructed_balances_pairs(con, cfg, pairs)

    # ---- ground truth over RPC ----
    log.info("querying eth_getBalance over RPC (archive node required, "
             "batch=%d, delay=%.2fs between batches) ...",
             args.rpc_batch_size, args.rpc_delay)
    truth = fetch_truth_rpc(pairs, args.rpc_url, args.rpc_batch_size,
                            args.rpc_delay, log)

    is_contract: dict[str, bool] = {}
    if args.classify_contracts:
        import time as _time

        log.info("classifying contracts via eth_getCode ...")
        todo = sorted({(a, b) for (a, b) in pairs if (a, b) in truth})
        for i in range(0, len(todo), args.rpc_batch_size):
            chunk = todo[i : i + args.rpc_batch_size]
            calls = [
                {"jsonrpc": "2.0", "id": j, "method": "eth_getCode",
                 "params": [a, hex(b)]}
                for j, (a, b) in enumerate(chunk)
            ]
            for (a, _b), r in zip(chunk, rpc_batch(args.rpc_url, calls)):
                is_contract[a] = is_contract.get(a, False) or \
                    len((r.get("result") or "0x")) > 2
            if args.rpc_delay > 0 and i + args.rpc_batch_size < len(todo):
                _time.sleep(args.rpc_delay)

    deltas = []
    for a, b in pairs:
        if (a, b) not in truth:
            continue
        t = truth[(a, b)]
        d = recon.get((a, b), 0) - t
        rel = abs(d) / t if t else (0.0 if d == 0 else float("inf"))
        deltas.append((a, b, recon.get((a, b), 0), t, d, rel))

    if args.out_csv:
        with open(args.out_csv, "w") as fh:
            fh.write("address,block,era,reconstructed_wei,truth_wei,"
                     "delta_wei,rel_delta,is_contract\n")
            for a, b, rc, tr, d, rel in deltas:
                fh.write(f"{a},{b},{era_of(b)},{rc},{tr},{d},{rel},"
                         f"{is_contract.get(a, '')}\n")
        log.info("wrote %s", args.out_csv)

    # -------- one-page accuracy summary --------
    if not deltas:
        log.error("no comparable pairs — RPC failed?")
        return 1
    tol = args.tolerance_wei
    log.info("=" * 70)
    log.info("ACCURACY SUMMARY (%d pairs)", len(deltas))
    summarize_deltas([(a, rc, tr, d, rel) for a, _b, rc, tr, d, rel in deltas],
                     tol, "overall", log)
    if args.random_pairs:
        for name, elo, ehi in ERAS:
            sub = [(a, rc, tr, d, rel) for a, b, rc, tr, d, rel in deltas
                   if elo <= b <= ehi]
            summarize_deltas(sub, tol, name, log)
    log.info("  systematic bias hints: overstated -> missing gas debits; "
             "understated -> missing withdrawals (validators/exchanges) or rewards")
    if is_contract:
        for label, flag in (("contracts", True), ("EOAs", False)):
            sub = [abs(d) for a, _b, *_ , d, _r in deltas
                   if is_contract.get(a) is flag]
            if sub:
                log.info("  %-9s: n=%d median abs delta %.9f ETH",
                         label, len(sub), statistics.median(sub) / WEI)
    worst = sorted(deltas, key=lambda t: -abs(t[4]))[:10]
    log.info("  worst 10 pairs:")
    for a, b, rc, tr, d, _rel in worst:
        log.info("    %s @ %d delta=%+.6f ETH%s", a, b, d / WEI,
                 " [contract]" if is_contract.get(a) else "")
    return 0


# ----------------------------- main -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the reconstructed ETH ledger")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bal = sub.add_parser("balance", help="reconstructed balances at a block")
    p_bal.add_argument("--block", type=int, required=True)
    p_bal.add_argument("--addresses", help="comma-separated 0x addresses")
    p_bal.add_argument("--addresses-file", help="file with one address per line")
    p_bal.add_argument("--out-csv")
    add_config_args(p_bal)

    p_sup = sub.add_parser("supply", help="supply invariant at snapshot blocks")
    p_sup.add_argument("--snapshot-blocks", required=True,
                       help="comma-separated blocks, or CSV/file with block per line")
    p_sup.add_argument("--supply-csv", help="known supply curve (block,supply_wei|supply_eth)")
    p_sup.add_argument("--out-csv")
    add_config_args(p_sup)

    p_spot = sub.add_parser("spot-check", help="diff vs eth_getBalance over RPC")
    p_spot.add_argument("--block", type=int,
                        help="fixed-snapshot mode: one block for all samples")
    p_spot.add_argument("--sample-size", type=int, default=2000,
                        help="addresses to sample in fixed-snapshot mode")
    p_spot.add_argument("--random-pairs", type=int,
                        help="random mode: N (address, block) pairs, "
                             "later-block weighted")
    p_spot.add_argument("--block-weight-power", type=float, default=2.0,
                        help="block density ~ block^(power-1); 1=uniform, "
                             "2=linear ramp (default), 3=quadratic")
    p_spot.add_argument("--start-block", type=int, default=0)
    p_spot.add_argument("--end-block", type=int, default=None,
                        help="default: ledger max block")
    p_spot.add_argument("--seed", type=float, default=0.42,
                        help="RNG seed in [-1,1] for reproducible sampling")
    p_spot.add_argument("--rpc-url", required=True)
    p_spot.add_argument("--rpc-batch-size", type=int, default=5,
                        help="calls per JSON-RPC batch (default 5; hosted "
                             "providers throttle by credits/second)")
    p_spot.add_argument("--rpc-delay", type=float, default=1.0,
                        help="seconds to sleep between batches (default 1.0; "
                             "0 for a local node)")
    p_spot.add_argument("--tolerance-wei", type=int, default=10**9)
    p_spot.add_argument("--classify-contracts", action="store_true")
    p_spot.add_argument("--out-csv")
    add_config_args(p_spot)

    args = parser.parse_args()
    cfg = Config.from_args(args)
    log = setup_logging(cfg.logs_dir, f"validate_{args.cmd.replace('-', '_')}")

    if args.cmd == "balance":
        if not args.addresses and not args.addresses_file:
            parser.error("balance requires --addresses or --addresses-file")
        return cmd_balance(args, cfg, log)
    if args.cmd == "supply":
        return cmd_supply(args, cfg, log)
    if (args.block is None) == (args.random_pairs is None):
        parser.error("spot-check requires exactly one of --block or --random-pairs")
    return cmd_spot_check(args, cfg, log)


if __name__ == "__main__":
    raise SystemExit(main())
