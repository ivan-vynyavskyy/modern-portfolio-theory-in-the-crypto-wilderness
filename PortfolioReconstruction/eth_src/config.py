#!/usr/bin/env python3
"""
config.py

Purpose
-------
Central configuration for the ETH native transfer reconstruction pipeline.

Every setting can be given as an environment variable (INPUT_BASE, OUTPUT_BASE,
SCRATCH_TMP, BLOCK_RANGE, ...; see `Config.defaults()`) or as a CLI flag added
by `add_config_args()` and consumed by `Config.from_args()`. CLI flags take
precedence over environment variables.

Notes
-----
- Chain constants (London, Merge, Shapella, DAO fork) are used only for
  warnings and expectations; the actual per-block branching (e.g. fee split)
  keys off the data itself (base_fee_per_gas NULL-ness), not block numbers.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ----------------------------- Chain constants -----------------------------

LONDON_BLOCK = 12_965_000      # EIP-1559: base fee burn
MERGE_BLOCK = 15_537_394       # last PoW block + 1; no reward traces from here
SHAPELLA_BLOCK = 17_034_870    # EIP-4895: beacon withdrawals in block bodies
DAO_FORK_BLOCK = 1_920_000     # irregular state transition

NO_TX_SENTINEL = 4_294_967_295  # transaction_index sort key for rows with no tx

# ----------------------------- Column candidates -----------------------------
# Step 0 (0_0_inspect_schemas.py) resolves canonical names against these candidate
# lists (first present wins) and records the result in decisions.json.

TRACE_CANDIDATES = {
    "block": ["block_number", "block_id", "block"],
    "tx_hash": ["transaction_hash", "tx_hash"],
    "tx_index": ["transaction_index", "tx_index"],
    "from": ["from_address", "from"],
    "to": ["to_address", "to"],
    "value": ["value"],
    "trace_type": ["trace_type", "type"],
    "call_type": ["call_type"],
    "status": ["status"],
    "error": ["error"],
    "trace_address": ["trace_address"],
    "reward_type": ["reward_type"],
}

TX_CANDIDATES = {
    "block": ["block_number", "block_id", "block"],
    "hash": ["hash", "transaction_hash", "tx_hash"],
    "tx_index": ["transaction_index", "tx_index"],
    "from": ["from_address", "from"],
    "gas_used": ["receipt_gas_used", "gas_used"],
    "effective_gas_price": ["receipt_effective_gas_price", "effective_gas_price"],
    "gas_price": ["gas_price"],
    "status": ["receipt_status", "status"],
    # EIP-4844: blob fee = blob_gas_used * blob_gas_price, paid by the
    # sender and burned (not part of effective_gas_price)
    "blob_gas_used": ["receipt_blob_gas_used", "blob_gas_used"],
    "blob_gas_price": ["receipt_blob_gas_price", "blob_gas_price"],
}

BLOCK_CANDIDATES = {
    "number": ["number", "block_number", "block_id", "block"],
    "miner": ["miner", "fee_recipient", "author", "beneficiary"],
    "base_fee": ["base_fee_per_gas", "base_fee"],
    "withdrawals": ["withdrawals"],
}

WITHDRAWAL_FIELD_CANDIDATES = {
    "address": ["address", "withdrawal_address", "recipient"],
    "amount": ["amount", "amount_gwei", "value"],
    "index": ["index", "withdrawal_index"],
}


# ----------------------------- Config dataclass -----------------------------

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _default_threads() -> int:
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            return int(slurm)
        except ValueError:
            pass
    return os.cpu_count() or 4


@dataclass
class Config:
    input_base: Path
    output_base: Path
    scratch_tmp: Path
    block_chunk_size: int
    block_subchunk_size: int
    duckdb_memory_limit: str
    duckdb_threads: int
    include_gas: bool
    include_withdrawals: bool
    include_rewards: bool
    include_genesis: bool
    include_debug_columns: bool
    block_range_start: int
    block_range_end: Optional[int]  # None -> derived from decisions.json
    genesis_alloc_json: Optional[Path]
    decisions_path: Path
    logs_dir: Path

    @staticmethod
    def defaults() -> "Config":
        # Relative defaults, resolved against the working directory. Set
        # INPUT_BASE / OUTPUT_BASE or pass --input-base / --output-base
        # (run_pipeline.py always passes both).
        input_base = Path(os.environ.get("INPUT_BASE", "eth_dump"))
        output_base = Path(os.environ.get("OUTPUT_BASE", "eth_transfers"))
        scratch = os.environ.get("SCRATCH_TMP") or os.environ.get("TMPDIR")
        scratch_tmp = Path(scratch) if scratch else output_base / "duckdb_tmp"
        block_range = os.environ.get("BLOCK_RANGE", "")
        lo, hi = 0, None
        if block_range:
            parts = block_range.replace("(", "").replace(")", "").split(",")
            lo = int(parts[0])
            if len(parts) > 1 and parts[1].strip():
                hi = int(parts[1])
        genesis_alloc = os.environ.get("GENESIS_ALLOC_JSON")
        return Config(
            input_base=input_base,
            output_base=output_base,
            scratch_tmp=scratch_tmp,
            block_chunk_size=int(os.environ.get("BLOCK_CHUNK_SIZE", 1_000_000)),
            block_subchunk_size=int(os.environ.get("BLOCK_SUBCHUNK_SIZE", 250_000)),
            duckdb_memory_limit=os.environ.get("DUCKDB_MEMORY_LIMIT", "100GB"),
            duckdb_threads=int(os.environ.get("DUCKDB_THREADS", _default_threads())),
            include_gas=_env_bool("INCLUDE_GAS", True),
            include_withdrawals=_env_bool("INCLUDE_WITHDRAWALS", True),
            include_rewards=_env_bool("INCLUDE_REWARDS", True),
            include_genesis=_env_bool("INCLUDE_GENESIS", True),
            include_debug_columns=_env_bool("INCLUDE_DEBUG_COLUMNS", True),
            block_range_start=lo,
            block_range_end=hi,
            genesis_alloc_json=Path(genesis_alloc) if genesis_alloc else None,
            decisions_path=Path(
                os.environ.get("DECISIONS_PATH", str(output_base / "decisions.json"))
            ),
            logs_dir=Path(os.environ.get("LOGS_DIR", str(Path(__file__).parent / "logs"))),
        )

    @staticmethod
    def from_args(args: argparse.Namespace) -> "Config":
        cfg = Config.defaults()
        if args.input_base:
            cfg.input_base = Path(args.input_base)
        if args.output_base:
            cfg.output_base = Path(args.output_base)
            # decisions.json follows the output base unless overridden explicitly
            if not args.decisions and not os.environ.get("DECISIONS_PATH"):
                cfg.decisions_path = cfg.output_base / "decisions.json"
            if not os.environ.get("SCRATCH_TMP") and not os.environ.get("TMPDIR") \
                    and not args.scratch_tmp:
                cfg.scratch_tmp = cfg.output_base / "duckdb_tmp"
        if args.scratch_tmp:
            cfg.scratch_tmp = Path(args.scratch_tmp)
        if args.decisions:
            cfg.decisions_path = Path(args.decisions)
        if args.block_chunk_size is not None:
            cfg.block_chunk_size = args.block_chunk_size
        if getattr(args, "block_subchunk_size", None) is not None:
            cfg.block_subchunk_size = args.block_subchunk_size
        if args.memory_limit:
            cfg.duckdb_memory_limit = args.memory_limit
        if args.threads is not None:
            cfg.duckdb_threads = args.threads
        if args.block_range:
            parts = args.block_range.split(",")
            cfg.block_range_start = int(parts[0])
            cfg.block_range_end = int(parts[1]) if len(parts) > 1 and parts[1].strip() else None
        if args.genesis_alloc_json:
            cfg.genesis_alloc_json = Path(args.genesis_alloc_json)
        for flag in (
            "include_gas",
            "include_withdrawals",
            "include_rewards",
            "include_genesis",
            "include_debug_columns",
        ):
            val = getattr(args, flag, None)
            if val is not None:
                setattr(cfg, flag, val)
        return cfg

    def describe(self) -> str:
        return (
            f"input_base={self.input_base}\n"
            f"output_base={self.output_base}\n"
            f"scratch_tmp={self.scratch_tmp}\n"
            f"decisions={self.decisions_path}\n"
            f"block_chunk_size={self.block_chunk_size:,} "
            f"(subchunk={self.block_subchunk_size:,})\n"
            f"duckdb: memory={self.duckdb_memory_limit} threads={self.duckdb_threads}\n"
            f"flags: gas={self.include_gas} withdrawals={self.include_withdrawals} "
            f"rewards={self.include_rewards} genesis={self.include_genesis} "
            f"debug_cols={self.include_debug_columns}\n"
            f"block_range=({self.block_range_start}, {self.block_range_end})"
        )


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Attach the shared config CLI flags (env vars provide the defaults)."""
    g = parser.add_argument_group("config (overrides env defaults)")
    g.add_argument("--input-base", help="read-only dump dir with trace/ transaction/ block/")
    g.add_argument("--output-base", help="writable output dir for the transfer ledger")
    g.add_argument("--scratch-tmp", help="fast scratch dir for DuckDB temp files")
    g.add_argument("--decisions", help="path to decisions.json (default <output>/decisions.json)")
    g.add_argument("--block-chunk-size", type=int, default=None)
    g.add_argument("--block-subchunk-size", type=int, default=None,
                   help="blocks per in-chunk processing unit (memory bound)")
    g.add_argument("--memory-limit", help="DuckDB memory limit, e.g. 100GB")
    g.add_argument("--threads", type=int, default=None)
    g.add_argument("--block-range", help="lo,hi (inclusive) to limit the run window")
    g.add_argument("--genesis-alloc-json", help="geth-style genesis.json for alloc injection")
    for flag in ("gas", "withdrawals", "rewards", "genesis", "debug-columns"):
        dest = f"include_{flag.replace('-', '_')}"
        g.add_argument(f"--{flag}", dest=dest, action="store_true", default=None)
        g.add_argument(f"--no-{flag}", dest=dest, action="store_false", default=None)


# ----------------------------- Partition file resolution -----------------------------

def partition_files(p: dict) -> list[str]:
    """Parquet files of one partition entry from decisions.json.

    Delta-discovered partitions carry an explicit active-file list ("files",
    from the Delta log — raw globbing could pick up stale superseded files).
    Plain partitions are globbed, always excluding _delta_log.
    """
    if p.get("files"):
        return p["files"]
    path = Path(p["path"])
    if path.is_file():
        return [str(path)]
    return [
        str(f) for f in sorted(path.rglob("*.parquet")) if "_delta_log" not in f.parts
    ]


# ----------------------------- Logging -----------------------------

def setup_logging(logs_dir: Path, tag: str, level: int = logging.INFO) -> logging.Logger:
    """Timestamped logging to stdout + a rotating file per job/chunk."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"eth_transfers.{tag}")
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:  # already configured (same process, multiple chunks)
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.handlers.RotatingFileHandler(
        logs_dir / f"{tag}.log", maxBytes=50_000_000, backupCount=3
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

