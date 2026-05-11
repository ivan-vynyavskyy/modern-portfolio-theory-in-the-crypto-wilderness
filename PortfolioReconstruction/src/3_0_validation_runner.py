#!/usr/bin/env python3
"""
3_0_validation_runner.py

Purpose
-------
Compare local cumulative balances with on-chain ERC-20 balanceOf results for spot-check validation.

What it does
------------
- Scans token Parquet files under a data directory.
- Randomly samples addresses and blocks per token and compares local vs chain balances.
- Records per-token PASS/NOT_PASS status to a TSV status file.

Inputs
------
- <data_dir>/token_address=*/<token>.parquet (Parquet; columns: address, block, value)
- --min-block / --max-block (optional int bounds; if omitted uses file min/max)
- $WEB3_INFURA_URL (optional; used if --infura-url not provided)

Outputs
-------
- <data_dir>/validation_status.tsv (TSV; columns: token, status)

CLI
---
Examples:
  $ python 3_0_validation_runner.py --data-dir /path/to/data_folder --all-tokens --trials 200 --delay 30 --delay-every 10
  $ python 3_0_validation_runner.py --data-dir /path/to/data_folder --tokens 25

Notes
-----
- Parquet files must contain address, block, and value columns.
- Validation samples addresses at random and uses a bias toward earlier blocks.
- Status values are recorded as PASS or NOT_PASS.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import Counter
from decimal import Decimal, getcontext
from pathlib import Path

# Ensure utils/ is importable when running from src/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import duckdb
import pandas as pd
from duckdb import BinderException, InvalidInputException
from web3 import HTTPProvider, Web3
from web3.exceptions import BadFunctionCallOutput, ContractLogicError

from utils.log_utils import log

# ----------------------------- Constants -----------------------------
DECIMAL_PRECISION = 80
ZERO_ADDR = "0x" + "0" * 40
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
]

getcontext().prec = DECIMAL_PRECISION

# ----------------------------- Web3 ---------------------------------
def contract_exists(w3: Web3, addr: str, block: int) -> bool:
    """Return True if contract bytecode exists at address for the block."""
    code = w3.eth.get_code(Web3.to_checksum_address(addr), block_identifier=block)
    return len(code) > 0


def chain_balance(w3: Web3, token: str, owner: str, block: int) -> int | None:
    """Return on-chain balanceOf value or None on missing contract/revert."""
    if not contract_exists(w3, token, block):
        return None
    contract = w3.eth.contract(Web3.to_checksum_address(token), abi=ERC20_ABI)
    try:
        return contract.functions.balanceOf(
            Web3.to_checksum_address(owner)
        ).call(block_identifier=block)
    except (BadFunctionCallOutput, ContractLogicError):
        return None


# ----------------------------- DuckDB --------------------------------
def local_sum(parq: Path, addr: str, blk: int) -> Decimal:
    """Return cumulative local balance for address up to block."""
    sql_base = f"FROM '{parq}' WHERE address='{addr}' AND block <= {blk}"

    # Fast path: column already numeric.
    try:
        res = duckdb.query(f"SELECT COALESCE(SUM(value),0) {sql_base}").fetchone()[0]
        return Decimal(res)
    except BinderException:
        pass
    except InvalidInputException:
        pass

    # Second try: cast VARCHAR -> HUGEINT (38-digit range).
    try:
        res = duckdb.query(
            f"SELECT COALESCE(SUM(CAST(value AS HUGEINT)),0) {sql_base}"
        ).fetchone()[0]
        return Decimal(res)
    except Exception:
        # Final fallback: Python big-int over a small slice.
        df = duckdb.query(f"SELECT value {sql_base}").fetchdf()
        total = 0
        for value in df["value"]:
            try:
                total += int(value)
            except Exception:
                pass
        return Decimal(total)


def address_min_blocks(parq: Path) -> list[tuple[str, int]]:
    """Return (address, first_block) pairs for the file."""
    return duckdb.query(
        f"SELECT address, MIN(block) AS first_blk FROM '{parq}' GROUP BY address"
    ).fetchall()


def address_rows(parq: Path, addr: str) -> pd.DataFrame:
    """Return ordered (block, value) rows for an address."""
    return duckdb.query(
        f"SELECT block,value FROM '{parq}' WHERE address='{addr}' ORDER BY block"
    ).fetchdf()


def has_rows(parq: Path) -> bool:
    """Return True if the Parquet file has at least one row."""
    try:
        return bool(
            duckdb.query(
                f"SELECT EXISTS(SELECT 1 FROM '{parq}' LIMIT 1)"
            ).fetchone()[0]
        )
    except Exception:
        return False


def block_range(parq: Path) -> tuple[int | None, int | None]:
    """Return (min_block, max_block) for the file or (None, None)."""
    min_blk, max_blk = duckdb.query(
        f"SELECT MIN(block) AS min_blk, MAX(block) AS max_blk FROM '{parq}'"
    ).fetchone()
    return (
        int(min_blk) if min_blk is not None else None,
        int(max_blk) if max_blk is not None else None,
    )


# ----------------------------- Misc ----------------------------------
def biased_block(low: int, high: int) -> int:
    """Return a random block biased toward the lower bound."""
    return int(low + math.sqrt(random.random()) * (high - low))


# ----------------------------- Status --------------------------------
def load_status(path: Path) -> dict[str, str]:
    """Load status TSV into a lowercase-token mapping."""
    if not path.exists():
        return {}
    status_map: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            token, status = line.rstrip().split("\t")
            status_map[token.lower()] = status.upper()
    return status_map


def write_status(path: Path, token: str, status: str) -> None:
    """Write status TSV atomically with updated token status."""
    tmp = path.with_suffix(".tmp")
    current = load_status(path)
    current[token.lower()] = status.upper()
    with tmp.open("w", encoding="utf-8") as fh:
        for tkn, stat in sorted(current.items()):
            fh.write(f"{tkn}\t{stat}\n")
    tmp.replace(path)


# ----------------------------- Workflow ------------------------------
def resolve_status_file(data_dir: Path, status_file_arg: str | None) -> Path:
    """Return status file path from CLI args or default location."""
    if status_file_arg:
        return Path(status_file_arg).expanduser().resolve()
    return data_dir / "validation_status.tsv"


def list_parquet_files(data_dir: Path) -> list[Path]:
    """Return sorted list of token Parquet files under data_dir."""
    return sorted(
        p for p in data_dir.glob("token_address=*/*.parquet") if p.is_file()
    )


def select_sample_files(
    parquet_files: list[Path], all_tokens: bool, tokens: int
) -> list[Path]:
    """Return sample of Parquet files based on CLI options."""
    if all_tokens:
        return parquet_files
    return random.sample(parquet_files, k=min(tokens, len(parquet_files)))


def pick_non_zero_address(
    rows: list[tuple[str, int]], max_tries: int = 10
) -> tuple[str, int] | None:
    """Return a random non-zero address and its first block, or None."""
    for _ in range(max_tries):
        addr, first_blk = random.choice(rows)
        if addr != ZERO_ADDR:
            return addr, first_blk
    return None


def compute_block_window(
    first_blk: int,
    file_blk_min: int,
    file_blk_max: int,
    blk_min_arg: int | None,
    blk_max_arg: int | None,
) -> tuple[int, int] | None:
    """Return an inclusive block window or None if invalid."""
    blk_low = max(first_blk, file_blk_min)
    if blk_min_arg is not None:
        blk_low = max(blk_low, blk_min_arg)

    blk_high = file_blk_max
    if blk_max_arg is not None:
        blk_high = min(blk_high, blk_max_arg)

    if blk_low > blk_high:
        return None
    return blk_low, blk_high


def process_token(
    parq: Path,
    idx: int,
    sample_total: int,
    status_map: dict[str, str],
    status_file: Path,
    trials: int,
    blk_min_arg: int | None,
    blk_max_arg: int | None,
    thresh: Decimal,
    w3: Web3,
    summary: Counter,
) -> bool:
    """Validate one token and update summary; return True if processed."""
    token = parq.stem.lower()

    if token in status_map:
        log(f"[{idx}] {token} - already checked ({status_map[token]}), skip")
        return False

    if not has_rows(parq):
        log(f"[{idx}] {token} - empty file, skip")
        summary["skipped"] += 1
        write_status(status_file, token, "NOT_PASS")
        return False

    rows = address_min_blocks(parq)
    file_blk_min, file_blk_max = block_range(parq)
    if file_blk_min is None or file_blk_max is None:
        log(f"[{idx}] {token} - cannot read block range, skip")
        summary["skipped"] += 1
        write_status(status_file, token, "NOT_PASS")
        return False

    log(f"\n[{idx}/{sample_total}] ===== {token} =====")
    token_ok = True

    for _ in range(trials):
        choice = pick_non_zero_address(rows)
        if choice is None:
            summary["skipped"] += 1
            continue
        addr, first_blk = choice

        window = compute_block_window(
            first_blk=first_blk,
            file_blk_min=file_blk_min,
            file_blk_max=file_blk_max,
            blk_min_arg=blk_min_arg,
            blk_max_arg=blk_max_arg,
        )
        if window is None:
            log(f"  SKIP addr={addr[:10]}... (invalid block window)")
            summary["skipped"] += 1
            continue
        blk_low, blk_high = window

        blk = biased_block(blk_low, blk_high)

        local_bal = local_sum(parq, addr, blk)
        chain_bal = chain_balance(w3, token, addr, blk)
        if chain_bal is None:
            log(f"  SKIP addr={addr[:10]}... block={blk} (no code / revert)")
            summary["skipped"] += 1
            continue

        diff = abs(chain_bal - local_bal)
        if diff <= thresh:
            log(f"  PASS  addr={addr[:10]}...  block={blk}")
            summary["pass_"] += 1
        else:
            rel = diff / chain_bal if chain_bal else Decimal(0)
            log(f"\nFAIL addr={addr} block={blk}")
            log(f"  local  = {local_bal}")
            log(f"  chain  = {chain_bal}")
            log(f"  diff   = {diff}  ({rel:.2%})")
            log("  full local rows:")
            log(address_rows(parq, addr).to_string(index=False))
            log("-" * 60)
            summary["fail"] += 1
            token_ok = False

    write_status(status_file, token, "PASS" if token_ok else "NOT_PASS")
    log(
        f"Result for {token}: {'PASS' if token_ok else 'NOT PASS'} "
        f"(recorded in {status_file.name})"
    )
    return True


def print_summary(summary: Counter, status_file: Path) -> None:
    """Print overall summary and status file path."""
    total = summary["pass_"] + summary["fail"]
    log("\n========== SUMMARY ==========")
    log(f"Total checks : {total}")
    log(f"Pass         : {summary['pass_']}")
    log(f"Fail         : {summary['fail']}")
    log(f"Skipped      : {summary['skipped']}")
    log(f"Status file  : {status_file}")
    log("=============================")


# ----------------------------- CLI ----------------------------------
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="validation_runner.py",
        description="Compare local cumulative balances with on-chain balanceOf results.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root with token_address=*/<token>.parquet",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tokens",
        type=int,
        default=25,
        help="How many token parquet files to sample",
    )
    group.add_argument(
        "--all-tokens",
        action="store_true",
        help="Validate *all* tokens under --data-dir",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of random balance checks per token",
    )
    parser.add_argument(
        "--min-block",
        type=int,
        help="Optional lower bound; default uses file min block",
    )
    parser.add_argument(
        "--max-block",
        type=int,
        help="Optional upper bound; default uses file max block",
    )
    parser.add_argument(
        "--threshold",
        type=Decimal,
        default=Decimal("1e-10"),
        help="Absolute diff tolerance before marking as FAIL",
    )
    parser.add_argument(
        "--infura-url",
        default=os.getenv("WEB3_INFURA_URL"),
        help="Ethereum JSON-RPC endpoint (defaults to $WEB3_INFURA_URL)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=60,
        help="Seconds to sleep when delay triggers (default 60)",
    )
    parser.add_argument(
        "--delay-every",
        type=int,
        default=1,
        help="Sleep only after every N tokens (default 1 = after each)",
    )
    parser.add_argument(
        "--status-file",
        help=(
            "Path to TSV with <token\\tPASS|NOT_PASS>. "
            "Default: DATA_DIR/validation_status.tsv"
        ),
    )
    return parser.parse_args()


# ----------------------------- Main ----------------------------------
def main() -> int:
    """Run validation workflow and return an exit code."""
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    trials = args.trials
    blk_min_arg = args.min_block
    blk_max_arg = args.max_block
    thresh = args.threshold
    delay_s = args.delay
    every_n = max(1, args.delay_every)
    rpc_url = args.infura_url
    if not rpc_url:
        raise SystemExit(
            "Ethereum RPC endpoint required: pass --infura-url or set $WEB3_INFURA_URL."
        )

    status_file = resolve_status_file(data_dir, args.status_file)
    status_map = load_status(status_file)

    w3 = Web3(HTTPProvider(rpc_url))
    client_name = w3.client_version.split("/")[0]
    log(
        "Start executing validation "
        f"(tolerance={thresh}, trials={trials}, rpc={client_name})"
    )

    parquet_files = list_parquet_files(data_dir)
    if not parquet_files:
        log(f"no token Parquet files under {data_dir}")
        return 1

    sample_files = select_sample_files(
        parquet_files=parquet_files,
        all_tokens=args.all_tokens,
        tokens=args.tokens,
    )

    log(f"validating {len(sample_files)} / {len(parquet_files)} tokens")

    summary = Counter(pass_=0, fail=0, skipped=0)
    processed = 0

    for idx, parq in enumerate(sample_files, 1):
        did_process = process_token(
            parq=parq,
            idx=idx,
            sample_total=len(sample_files),
            status_map=status_map,
            status_file=status_file,
            trials=trials,
            blk_min_arg=blk_min_arg,
            blk_max_arg=blk_max_arg,
            thresh=thresh,
            w3=w3,
            summary=summary,
        )

        if did_process:
            processed += 1
            if processed % every_n == 0 and idx != len(sample_files) and delay_s > 0:
                log(f"sleeping {delay_s}s ...")
                time.sleep(delay_s)

    print_summary(summary, status_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
