#!/usr/bin/env python3
"""
run_pipeline.py

Purpose
-------
End-to-end driver for the portfolio reconstruction pipeline. Reads a YAML
configuration file, validates that every required input exists, and shells
out to each `N_M_*.py` step in order. Fails loudly the moment any step
exits non-zero or any required path is missing.

The steps run in three streams, each in its own folder:

    erc20.*   `erc20_src/`   ERC-20 universe, prices, per-token ledgers (0_0 .. 3_1)
    eth.*     `eth_src/`     native-ETH transfer and flow ledger, ETH price (0_0 .. 5_0)
    common.*  `common_src/`  reconstruction (4_0), verification (4_1), partitioning
                             (4_2) and backtest (5_0) over both ledgers

Step ids carry the stream prefix (`eth.1_0`). A bare id (`1_2`) is accepted
wherever it is unique across the streams.

Usage
-----
    python run_pipeline.py --config pipeline_config.yaml
    python run_pipeline.py --config pipeline_config.yaml --from-step erc20.1_2
    python run_pipeline.py --config pipeline_config.yaml --from-step eth.0_0 --to-step eth.5_0
    python run_pipeline.py --config pipeline_config.yaml --dry-run
    python run_pipeline.py --list-steps

See `pipeline_config.example.yaml` for the full set of configuration knobs.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "PyYAML is required to run the pipeline driver.\n"
        "Install it with:  pip install pyyaml\n"
    )
    raise SystemExit(1) from exc


# ----------------------------- Logging -----------------------------

def _log(msg: str) -> None:
    """Print a timestamped line to stdout."""
    print(f"{datetime.now():%H:%M:%S} | run_pipeline | {msg}", flush=True)


def _die(msg: str, code: int = 1) -> "NoReturn":  # type: ignore[name-defined]
    """Print an error line and exit."""
    print(f"{datetime.now():%H:%M:%S} | run_pipeline | ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


# ----------------------------- Paths -----------------------------

ROOT_DIR = Path(__file__).resolve().parent
ERC20_SRC_DIR = ROOT_DIR / "erc20_src"    # ERC-20 universe steps (0_0 .. 3_1)
ETH_SRC_DIR = ROOT_DIR / "eth_src"        # native-ETH ledger steps (0_0 .. 5_0)
COMMON_SRC_DIR = ROOT_DIR / "common_src"  # stream-agnostic steps (4_x, 5_0)

SRC_DIRS = {"erc20": ERC20_SRC_DIR, "eth": ETH_SRC_DIR, "common": COMMON_SRC_DIR}

# Set by --dry-run: print every command without running it or touching the disk.
DRY_RUN = False


# ----------------------------- Config helpers -----------------------------

def _load_config(path: Path) -> Dict[str, Any]:
    """Load and return the YAML configuration."""
    if not path.exists():
        _die(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        _die(f"Config file {path} did not parse to a mapping at the top level.")
    return cfg


def _require(cfg: Dict[str, Any], key: str) -> Any:
    """Return cfg[key] or abort with a descriptive error."""
    value = cfg.get(key)
    if value is None or value == "":
        _die(f"Missing required config key: '{key}'")
    return value


def _resolve(base: Path, raw: str) -> Path:
    """Resolve a path that may be absolute or relative to `base`."""
    p = Path(raw)
    return p if p.is_absolute() else (base / p)


def _ensure_parent(p: Path) -> None:
    """Create the parent directory of `p` if it doesn't exist."""
    if not DRY_RUN:
        p.parent.mkdir(parents=True, exist_ok=True)


def _ensure_dir(p: Path) -> None:
    """Create directory `p` (and parents) if it doesn't exist."""
    if not DRY_RUN:
        p.mkdir(parents=True, exist_ok=True)


def _require_path(p: Path, kind: str, key: str) -> None:
    """Abort if `p` does not exist (a dry run only warns, since earlier
    steps would have created it)."""
    if p.exists():
        return
    if DRY_RUN:
        _log(f"    (dry run) {kind} for config key '{key}' does not exist yet: {p}")
        return
    _die(
        f"{kind} for config key '{key}' does not exist: {p}\n"
        f"Provide it before running the pipeline."
    )


# ----------------------------- Step plumbing -----------------------------

class Step:
    """One pipeline step that knows how to build its CLI invocation."""

    def __init__(
        self,
        stream: str,
        number: str,
        script_name: str,
        builder: Callable[[Dict[str, Any]], List[str]],
        description: str,
    ) -> None:
        self.stream = stream          # erc20 | eth | common
        self.number = number          # e.g. "1_0", unique within the stream
        self.step_id = f"{stream}.{number}"
        self.script_name = script_name
        self.builder = builder
        self.description = description
        self.src_dir = SRC_DIRS[stream]

    def script_path(self) -> Path:
        return self.src_dir / self.script_name

    def build_cmd(self, ctx: Dict[str, Any]) -> List[str]:
        return [str(ctx["python"]), str(self.script_path()), *self.builder(ctx)]


def _run_step(step: Step, cmd: List[str]) -> None:
    """Run a single step as a subprocess, aborting if it exits non-zero."""
    if not step.script_path().exists():
        _die(f"Script not found for step {step.step_id}: {step.script_path()}")

    _log(f"--- step {step.step_id}: {step.description}")
    _log(f"    $ {' '.join(shlex.quote(c) for c in cmd)}")
    if DRY_RUN:
        return

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        _die(f"step {step.step_id} failed with exit code {result.returncode} (after {elapsed:.1f}s)")

    _log(f"--- step {step.step_id} done in {elapsed:.1f}s")


# ----------------------------- Step builders -----------------------------
#
# Each step has a builder `_b_<id>(ctx) -> List[str]` that returns the step's CLI
# arguments. `ctx` holds the resolved paths (see `_build_context`) and the raw
# YAML under `ctx["raw"]`. To forward a script flag that is not exposed yet:
#
#   1. add a key to pipeline_config.example.yaml (and your pipeline_config.yaml),
#   2. read it in the builder and append the flag, e.g. in `_b_0_2`:
#
#          if ctx["raw"].get("trace_scan_require_events") is False:
#              args.append("--no-events")
#
#   3. for value-bearing flags emit them only when the key is set, so a blank
#      key falls back to the script's own default:
#
#          sleep = ctx["raw"].get("supply_fetch_sleep_seconds")
#          if sleep is not None:
#              args += ["--sleep", str(sleep)]

# ---- ERC-20 stream (erc20_src/) ----

def _b_0_0(ctx: Dict[str, Any]) -> List[str]:
    out = ctx["all_coins_csv"]
    _ensure_parent(out)
    return ["--api-key", ctx["coingecko_api_key"], "--output", str(out)]


def _b_0_1(ctx: Dict[str, Any]) -> List[str]:
    return [
        "--input", str(ctx["all_coins_csv"]),
        "--output", str(ctx["derived_coingecko_eth_tokens_csv"]),
    ]


def _b_0_2(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["traces_dir"], "Traces dir", "traces_dir")
    out = ctx["contract_addresses_tsv"]
    _ensure_parent(out)
    args = ["--traces-dir", str(ctx["traces_dir"]), "--out-tsv", str(out)]
    delta_version = ctx["raw"].get("traces_delta_version")
    if delta_version is not None:
        args += ["--delta-version", str(delta_version)]
    return args


def _b_0_3(ctx: Dict[str, Any]) -> List[str]:
    return [
        "--input", str(ctx["derived_coingecko_eth_tokens_csv"]),
        "--tsv", str(ctx["contract_addresses_tsv"]),
        "--output", str(ctx["status_eth_tokens_csv_0_3"]),
    ]


def _b_0_4(ctx: Dict[str, Any]) -> List[str]:
    out = ctx["erc20_proxy_resolved_tsv"]
    _ensure_parent(out)
    return [
        "--status-csv", str(ctx["status_eth_tokens_csv_0_3"]),
        "--rpc", ctx["rpc_url"],
        "--out-tokens", str(out),
    ]


def _b_0_5(ctx: Dict[str, Any]) -> List[str]:
    return [
        "--status-csv", str(ctx["status_eth_tokens_csv_0_3"]),
        "--resolver-tsv", str(ctx["erc20_proxy_resolved_tsv"]),
        "--output", str(ctx["status_eth_tokens_csv_0_5"]),
    ]


def _b_1_0(ctx: Dict[str, Any]) -> List[str]:
    _ensure_dir(ctx["prices_dir"])
    _ensure_parent(ctx["prices_download_status_csv"])
    return [
        "--input", str(ctx["status_eth_tokens_csv_0_5"]),
        "--output-dir", str(ctx["prices_dir"]),
        "--status-file", str(ctx["prices_download_status_csv"]),
        "--api-key", ctx["coingecko_api_key"],
        "--vs-currency", ctx["raw"].get("vs_currency", "usd"),
    ]


def _b_1_1(ctx: Dict[str, Any]) -> List[str]:
    return [
        "--local", str(ctx["status_eth_tokens_csv_0_5"]),
        "--price", str(ctx["prices_download_status_csv"]),
        "--output", str(ctx["status_eth_tokens_csv_1_1"]),
    ]


def _b_1_2(ctx: Dict[str, Any]) -> List[str]:
    return [
        "--csv", str(ctx["status_eth_tokens_csv_1_1"]),
        "--out", str(ctx["status_eth_tokens_csv_1_2"]),
        "--days", str(ctx["raw"].get("volume_lookback_days", 10)),
        "--api-key", ctx["coingecko_api_key"],
    ]


def _b_1_3(ctx: Dict[str, Any]) -> List[str]:
    out = ctx["tokens_cg_metrics_csv"]
    _ensure_parent(out)
    return [
        "--in", str(ctx["status_eth_tokens_csv_1_2"]),
        "--out", str(out),
        "--api-key", ctx["coingecko_api_key"],
    ]


def _b_1_4(ctx: Dict[str, Any]) -> List[str]:
    return [
        "--status-in", str(ctx["status_eth_tokens_csv_1_2"]),
        "--metrics", str(ctx["tokens_cg_metrics_csv"]),
        "--status-out", str(ctx["status_eth_tokens_csv_1_4"]),
    ]


def _b_2_0(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["raw_logs_dir"], "Raw logs dir", "raw_logs_dir")
    _ensure_dir(ctx["transfers_map_dir"])
    _ensure_dir(ctx["transfers_final_dir"])
    return [
        "--src-raw", str(ctx["raw_logs_dir"]),
        "--map-root", str(ctx["transfers_map_dir"]),
        "--final-root", str(ctx["transfers_final_dir"]),
        "--token-csv", str(ctx["status_eth_tokens_csv_1_4"]),
        "--batch", str(ctx["raw"].get("mapreduce_batch", 48)),
        "--map-timeout", str(ctx["raw"].get("mapreduce_timeout_seconds", 3600)),
    ]


def _b_2_1(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["transfers_final_dir"], "Transfers final dir", "transfers_final_dir")
    _ensure_dir(ctx["ledgers_dir"])
    return [
        str(ctx["transfers_final_dir"]),
        "--token-status-csv", str(ctx["status_eth_tokens_csv_1_4"]),
        "--output", str(ctx["ledgers_dir"]),
    ]


def _b_3_0(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["ledgers_dir"], "Ledgers dir", "ledgers_dir")
    return [
        "--data-dir", str(ctx["ledgers_dir"]),
        "--all-tokens",
        "--trials", str(ctx["raw"].get("validation_trials", 200)),
        "--delay", str(ctx["raw"].get("validation_delay_seconds", 30)),
        "--delay-every", str(ctx["raw"].get("validation_delay_every", 10)),
        "--infura-url", ctx["rpc_url"],
        "--status-file", str(ctx["validation_status_tsv"]),
    ]


def _b_3_1(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["validation_status_tsv"], "Validation status TSV", "validation_status_tsv")
    return [
        "--status-csv", str(ctx["status_eth_tokens_csv_1_4"]),
        "--validation-tsv", str(ctx["validation_status_tsv"]),
        "--out", str(ctx["status_eth_tokens_csv_3_1"]),
    ]


# ---- Native-ETH stream (eth_src/) ----
#
# The eth_src scripts share --input-base (raw dump), --output-base (decisions.json
# and the block_range=*/ transfer ledger) and --scratch-tmp (DuckDB spill), all
# defined in eth_src/config.py. The flow ledger goes into ledgers_dir next to the
# ERC-20 ledgers (token_address=0xee..ee/), so common.4_0 picks it up like any
# other token, and the ETH price CSV goes into prices_dir.

def _eth_dump_dir(ctx: Dict[str, Any]) -> Path:
    p = ctx["eth_dump_dir"]
    if p is None:
        _die("Missing required config key: 'eth_dump_dir' (raw Ethereum Parquet "
             "dump with trace/, transaction/ and block/, read by eth.0_0 and eth.1_0)")
    _require_path(p, "ETH dump dir", "eth_dump_dir")
    return p


def _eth_config_args(ctx: Dict[str, Any]) -> List[str]:
    _ensure_dir(ctx["eth_scratch_dir"])
    return [
        "--output-base", str(ctx["eth_transfers_dir"]),
        "--scratch-tmp", str(ctx["eth_scratch_dir"]),
    ]


def _b_eth_0_0(ctx: Dict[str, Any]) -> List[str]:
    _ensure_dir(ctx["eth_transfers_dir"])
    return ["--input-base", str(_eth_dump_dir(ctx)), *_eth_config_args(ctx)]


def _b_eth_1_0(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["eth_transfers_dir"] / "decisions.json",
                  "Schema decisions of eth.0_0", "eth_transfers_dir")
    # Without --force, chunks that already carry _SUCCESS are skipped (resumable).
    return ["--all", "--input-base", str(_eth_dump_dir(ctx)), *_eth_config_args(ctx)]


def _b_eth_2_0(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["eth_transfers_dir"], "ETH transfer ledger dir", "eth_transfers_dir")
    _ensure_dir(ctx["ledgers_dir"])
    return [
        "--output", str(ctx["ledgers_dir"]),
        "--shards", str(ctx["raw"].get("eth_ledger_shards", 64)),
        *_eth_config_args(ctx),
    ]


def _b_eth_3_0(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["eth_transfers_dir"], "ETH transfer ledger dir", "eth_transfers_dir")
    out = ctx["eth_validation_dir"] / "transfer_ledger_spot_check.csv"
    _ensure_parent(out)
    return [
        "spot-check",
        "--random-pairs", str(ctx["raw"].get("eth_validation_random_pairs", 200)),
        "--rpc-url", ctx["rpc_url"],
        "--out-csv", str(out),
        *_eth_config_args(ctx),
    ]


def _b_eth_3_1(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["ledgers_dir"], "Ledgers dir", "ledgers_dir")
    out = ctx["eth_validation_dir"] / "flow_ledger_spot_check.csv"
    _ensure_parent(out)
    # Same token -> PASS/NOT_PASS status file as erc20.3_0 (one row per token).
    return [
        "--data-dir", str(ctx["ledgers_dir"]),
        "--trials", str(ctx["raw"].get("eth_validation_trials", 1000)),
        "--rpc-url", ctx["rpc_url"],
        "--out-csv", str(out),
        "--status-file", str(ctx["validation_status_tsv"]),
    ]


def _b_eth_4_0(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["eth_transfers_dir"], "ETH transfer ledger dir", "eth_transfers_dir")
    return ["--no-scan", *_eth_config_args(ctx)]


def _b_eth_4_1(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["ledgers_dir"], "Ledgers dir", "ledgers_dir")
    return [
        "--data-dir", str(ctx["ledgers_dir"]),
        "--decisions", str(ctx["eth_transfers_dir"] / "decisions.json"),
    ]


def _b_eth_5_0(ctx: Dict[str, Any]) -> List[str]:
    _ensure_dir(ctx["prices_dir"])
    _ensure_parent(ctx["eth_price_status_csv"])
    return [
        "--output-dir", str(ctx["prices_dir"]),
        "--status-file", str(ctx["eth_price_status_csv"]),
        "--api-key", ctx["coingecko_api_key"],
        "--vs-currency", ctx["raw"].get("vs_currency", "usd"),
    ]


# ---- Common stream (common_src/) ----

def _b_4_0(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["block_to_date_csv"], "Block-to-date CSV", "block_to_date_csv")
    _require_path(ctx["input_wallets_parquet"], "Input wallets parquet", "input_wallets_parquet")
    _ensure_parent(ctx["decimals_cache_json"])
    _ensure_parent(ctx["portfolio_holdings_parquet"])

    args = [
        "--data-root", str(ctx["ledgers_dir"]),
        "--block-to-date-csv", str(ctx["block_to_date_csv"]),
        "--prices-dir", str(ctx["prices_dir"]),
        "--decimals-cache", str(ctx["decimals_cache_json"]),
        "--token-status-csv", str(ctx["status_eth_tokens_csv_3_1"]),
        "--rpc-url", ctx["rpc_url"],
        "--input-wallets-parquet", str(ctx["input_wallets_parquet"]),
        "--wallets-column", ctx["raw"].get("input_wallets_column", "address"),
        "--output", str(ctx["portfolio_holdings_parquet"]),
        "--batch-size", str(ctx["raw"].get("reconstruction_batch_size", 50_000)),
        "--duckdb-threads", str(ctx["raw"].get("reconstruction_duckdb_threads", 2)),
        "--duckdb-mem", str(ctx["raw"].get("reconstruction_duckdb_mem", "2GB")),
    ]
    procs = ctx["raw"].get("reconstruction_processes")
    if procs is not None:
        args += ["--processes", str(procs)]
    return args


def _b_4_1(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["portfolio_holdings_parquet"], "Holdings parquet", "portfolio_holdings_parquet")
    return [
        "--holdings", str(ctx["portfolio_holdings_parquet"]),
        "--decimals-cache", str(ctx["decimals_cache_json"]),
        "--rpc-url", ctx["rpc_url"],
        "--rpc-checks", str(ctx["raw"].get("holdings_verify_rpc_checks", 25)),
    ]


def _b_4_2(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["portfolio_holdings_parquet"], "Holdings parquet", "portfolio_holdings_parquet")
    # --force re-ingests the 4_0 output even if the manifest already lists it, so a
    # regenerated holdings file replaces its blocks instead of being skipped.
    return [
        "--sources", str(ctx["portfolio_holdings_parquet"]),
        "--root", str(ctx["portfolio_holdings_dataset"]),
        "--block-to-date-csv", str(ctx["block_to_date_csv"]),
        "--force",
    ]


def _b_5_0(ctx: Dict[str, Any]) -> List[str]:
    _require_path(ctx["portfolio_holdings_dataset"], "Partitioned holdings dataset",
                  "portfolio_holdings_dataset")
    _ensure_parent(ctx["backtest_summary_csv"])
    _ensure_parent(ctx["backtest_wallet_parquet"])
    args = [
        "--holdings-parquet", str(ctx["portfolio_holdings_dataset"]),
        "--block-to-date-csv", str(ctx["block_to_date_csv"]),
        "--prices-dir", str(ctx["prices_dir"]),
        "--out-csv", str(ctx["backtest_summary_csv"]),
        "--out-wallet-parquet", str(ctx["backtest_wallet_parquet"]),
        "--horizon-days", str(ctx["raw"].get("backtest_horizon_days", 20)),
    ]
    workers = ctx["raw"].get("backtest_workers")
    if workers is not None:
        args += ["--workers", str(workers)]
    blocks = ctx["raw"].get("backtest_blocks")
    if blocks is not None:
        args += ["--blocks", str(blocks)]
    return args


# ----------------------------- Step registry -----------------------------

STEPS: List[Step] = [
    # ERC-20 universe, prices and per-token ledgers
    Step("erc20", "0_0", "0_0_extract_all_coingecko_coins.py",          _b_0_0, "Download CoinGecko coin universe"),
    Step("erc20", "0_1", "0_1_extract_eth_tokens_from_coingecko.py",    _b_0_1, "Extract Ethereum-platform tokens"),
    Step("erc20", "0_2", "0_2_scan_traces_for_erc20_candidates.py",     _b_0_2, "Scan traces for ERC-20 deployments"),
    Step("erc20", "0_3", "0_3_mark_status_with_scanner_hits.py",        _b_0_3, "Mark CoinGecko tokens with scanner hits"),
    Step("erc20", "0_4", "0_4_resolve_proxies_and_verify_erc20.py",     _b_0_4, "Resolve proxies + verify ERC-20 via RPC"),
    Step("erc20", "0_5", "0_5_update_status_with_erc20_verification.py", _b_0_5, "Flip is_erc_20 with verification verdict"),
    Step("erc20", "1_0", "1_0_fetch_token_price_history.py",            _b_1_0, "Fetch CoinGecko daily-close history"),
    Step("erc20", "1_1", "1_1_update_token_status_with_price.py",       _b_1_1, "Merge price-download status"),
    Step("erc20", "1_2", "1_2_check_trading_volume_status.py",          _b_1_2, "Tag trading-volume sufficiency"),
    Step("erc20", "1_3", "1_3_fetch_supply_and_mcap_from_cg.py",        _b_1_3, "Fetch supply + FDV + market cap"),
    Step("erc20", "1_4", "1_4_update_status_fdv_mcap_vs_eth.py",        _b_1_4, "Flag FDV/mcap vs Ethereum"),
    Step("erc20", "2_0", "2_0_erc20_transfer_mapreduce.py",             _b_2_0, "Map-reduce raw logs into per-token transfers"),
    Step("erc20", "2_1", "2_1_erc20_transfer_aggregator.py",            _b_2_1, "Build per-token sorted ledgers"),
    Step("erc20", "3_0", "3_0_validation_runner.py",                    _b_3_0, "Spot-check ledger balances vs RPC"),
    Step("erc20", "3_1", "3_1_add_local_status.py",                     _b_3_1, "Append local_status to status CSV"),
    # Native-ETH transfer ledger, flow ledger and price
    Step("eth", "0_0", "0_0_inspect_schemas.py",                        _b_eth_0_0, "Discover dump schemas -> decisions.json"),
    Step("eth", "1_0", "1_0_build_transfers.py",                        _b_eth_1_0, "Build the ETH transfer ledger"),
    Step("eth", "2_0", "2_0_eth_transfer_aggregator.py",                _b_eth_2_0, "Aggregate into the address-flow ledger"),
    Step("eth", "3_0", "3_0_validate_transfer_ledger.py",               _b_eth_3_0, "Spot-check the transfer ledger vs RPC"),
    Step("eth", "3_1", "3_1_validate_flow_ledger.py",                   _b_eth_3_1, "Spot-check the flow ledger vs eth_getBalance"),
    Step("eth", "4_0", "4_0_ledger_overview.py",                        _b_eth_4_0, "Global sanity numbers of the transfer ledger"),
    Step("eth", "4_1", "4_1_ledger_stats.py",                           _b_eth_4_1, "Exact flow-ledger statistics"),
    Step("eth", "5_0", "5_0_fetch_eth_price_history.py",                _b_eth_5_0, "Fetch CoinGecko daily ETH close"),
    # Reconstruction and backtest over both ledgers
    Step("common", "4_0", "4_0_portfolio_reconstruction_pipeline.py",   _b_4_0, "Reconstruct per-wallet portfolio snapshots"),
    Step("common", "4_1", "4_1_verify_holdings.py",                     _b_4_1, "Verify holdings (invariants + archive node)"),
    Step("common", "4_2", "4_2_partition_holdings.py",                  _b_4_2, "Partition holdings by snapshot block"),
    Step("common", "5_0", "5_0_portfolio_backtest_pipeline.py",         _b_5_0, "Run MPT backtest + CAPM analytics"),
]

STEP_IDS = [s.step_id for s in STEPS]


def _resolve_step_id(raw: str) -> str:
    """Map a qualified id (`eth.1_0`) or a bare id that is unique across the
    streams (`1_2`) to the qualified id."""
    if raw in STEP_IDS:
        return raw
    matches = [s.step_id for s in STEPS if s.number == raw]
    if len(matches) == 1:
        return matches[0]
    if matches:
        _die(f"Step '{raw}' exists in several streams ({', '.join(matches)}); "
             f"add the stream prefix.")
    _die(f"Unknown step '{raw}'. Run with --list-steps to see the step ids.")


# ----------------------------- Context resolution -----------------------------

# Keys that contain filesystem paths to be resolved relative to output_root.
PATH_KEYS = (
    "all_coins_csv",
    "derived_coingecko_eth_tokens_csv",
    "erc20_proxy_resolved_tsv",
    "status_eth_tokens_csv_0_3",
    "status_eth_tokens_csv_0_5",
    "status_eth_tokens_csv_1_1",
    "status_eth_tokens_csv_1_2",
    "status_eth_tokens_csv_1_4",
    "status_eth_tokens_csv_3_1",
    "prices_dir",
    "prices_download_status_csv",
    "tokens_cg_metrics_csv",
    "contract_addresses_tsv",
    "validation_status_tsv",
    "portfolio_holdings_parquet",
    "backtest_summary_csv",
    "backtest_wallet_parquet",
)

# Path keys that the user must supply as absolute (or already-existing) paths.
USER_INPUT_PATH_KEYS = (
    "traces_dir",
    "raw_logs_dir",
    "transfers_map_dir",
    "transfers_final_dir",
    "ledgers_dir",
    "block_to_date_csv",
    "decimals_cache_json",
    "input_wallets_parquet",
)

# Optional path keys (relative to output_root unless absolute) and their defaults,
# so that a config written before the eth and partition steps existed still works.
OPTIONAL_PATH_DEFAULTS = {
    "eth_transfers_dir": "eth_transfers",            # eth.0_0/1_0: decisions.json + transfer ledger
    "eth_scratch_dir": "eth_scratch",                # DuckDB spill dir of the eth steps
    "eth_validation_dir": "eth_validation",          # spot-check CSVs of eth.3_0 and eth.3_1
    "eth_price_status_csv": "eth_price_download_status.csv",  # eth.5_0 fetch log
    "portfolio_holdings_dataset": "portfolio_holdings_dataset",  # common.4_2 output, 5_0 input
}


def _build_context(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve all paths and return the runtime context dict."""
    output_root = Path(_require(cfg, "output_root")).expanduser().resolve()
    if not DRY_RUN:
        output_root.mkdir(parents=True, exist_ok=True)

    ctx: Dict[str, Any] = {
        "raw": cfg,
        "output_root": output_root,
        "python": cfg.get("python_executable") or sys.executable,
        "coingecko_api_key": _require(cfg, "coingecko_api_key"),
        "rpc_url": _require(cfg, "rpc_url"),
    }

    # The validation_status_tsv defaults to living inside ledgers_dir.
    ledgers_raw = _require(cfg, "ledgers_dir")
    ledgers_path = Path(ledgers_raw).expanduser()
    if not ledgers_path.is_absolute():
        ledgers_path = (output_root / ledgers_path).resolve()

    for key in PATH_KEYS:
        raw = _require(cfg, key)
        if key == "validation_status_tsv":
            p = Path(raw).expanduser()
            ctx[key] = p if p.is_absolute() else (ledgers_path / p)
        else:
            ctx[key] = _resolve(output_root, str(raw)).expanduser()

    for key in USER_INPUT_PATH_KEYS:
        raw = _require(cfg, key)
        ctx[key] = Path(raw).expanduser()

    for key, default in OPTIONAL_PATH_DEFAULTS.items():
        raw = cfg.get(key) or default
        ctx[key] = _resolve(output_root, str(Path(str(raw)).expanduser()))

    # Raw Ethereum dump of the eth stream; only eth.0_0 and eth.1_0 read it, and
    # their builders abort if it is missing, so ERC-20-only runs need not set it.
    eth_dump = cfg.get("eth_dump_dir")
    ctx["eth_dump_dir"] = Path(str(eth_dump)).expanduser() if eth_dump else None

    return ctx


# ----------------------------- CLI -----------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the portfolio reconstruction pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Configuration:\n"
            "  Copy pipeline_config.example.yaml to pipeline_config.yaml and fill it in.\n\n"
            "Examples:\n"
            "  python run_pipeline.py --config pipeline_config.yaml\n"
            "  python run_pipeline.py --config pipeline_config.yaml --from-step erc20.1_2\n"
            "  python run_pipeline.py --config pipeline_config.yaml --from-step eth.0_0 --to-step eth.5_0\n"
            "  python run_pipeline.py --config pipeline_config.yaml --dry-run\n"
            "  python run_pipeline.py --list-steps\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "pipeline_config.yaml",
        help="Path to the YAML configuration file (default: pipeline_config.yaml).",
    )
    parser.add_argument(
        "--from-step",
        default=None,
        help="Skip every step before this one (useful for resuming), e.g. eth.1_0.",
    )
    parser.add_argument(
        "--to-step",
        default=None,
        help="Stop after this step, e.g. eth.5_0 to build only the ETH ledger.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command of every selected step without running it.",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="Print the ordered list of pipeline steps and exit.",
    )
    return parser


def _print_steps() -> None:
    print("Pipeline steps (in execution order):")
    for s in STEPS:
        rel = s.script_path().relative_to(ROOT_DIR)
        print(f"  {s.step_id:<11s} {str(rel):<56s}  {s.description}")


def main(argv: Optional[List[str]] = None) -> int:
    global DRY_RUN
    args = _build_arg_parser().parse_args(argv)

    if args.list_steps:
        _print_steps()
        return 0

    DRY_RUN = args.dry_run
    cfg = _load_config(args.config)
    ctx = _build_context(cfg)

    start_index, end_index = 0, len(STEPS) - 1
    if args.from_step:
        start_index = STEP_IDS.index(_resolve_step_id(args.from_step))
        _log(f"resuming from step {STEP_IDS[start_index]} (skipping {start_index} earlier step(s))")
    if args.to_step:
        end_index = STEP_IDS.index(_resolve_step_id(args.to_step))
    if end_index < start_index:
        _die(f"--to-step {STEP_IDS[end_index]} comes before --from-step {STEP_IDS[start_index]}")

    selected = STEPS[start_index:end_index + 1]
    _log(f"output_root = {ctx['output_root']}")
    _log(f"python      = {ctx['python']}")
    _log(f"steps to run: {', '.join(s.step_id for s in selected)}")
    if DRY_RUN:
        _log("dry run: commands are printed, not executed")

    pipeline_t0 = time.perf_counter()
    for step in selected:
        cmd = step.build_cmd(ctx)
        _run_step(step, cmd)

    if DRY_RUN:
        _log("DRY RUN COMPLETE (nothing was executed)")
    else:
        _log(f"PIPELINE COMPLETE in {time.perf_counter() - pipeline_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
