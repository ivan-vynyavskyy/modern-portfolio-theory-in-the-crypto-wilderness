#!/usr/bin/env python3
"""
_alloc_common.py

Description:
    Helpers shared by the allocation analyses in this folder
    (stablecoin_cash_share.py, weight_concentration.py, trading_vs_drift.py).
    Conventions follow the other Obj A scripts
    (portfolio_size_by_wealth_bucket.py, analyze_portfolio_per_block.py):

    - input is the reconstructed holdings, either one 4_0 parquet file or the
      4_2 hive root (block_number=N/*.parquet, block_number only in the
      directory name); columns wallet_address, total_value_usd, num_tokens and
      holdings = list<struct{token_address, balance, value_usd}>, where
      balance is a decimal string in token units and value_usd is
      balance x price (NULL when the token has no price at the snapshot);
    - native ETH is the pseudo-token 0xee..ee and WETH is folded into it, so
      ETH and WETH are one asset, as in 5_0 and the paper;
    - the beacon deposit contract is excluded as a holder;
    - wallets listed in --ca-tsv are contract accounts (CA), all others EOA;
    - weights are value_usd shares; positions without a price carry no weight;
    - wealth buckets use the labels of value_buckets_minimal.csv.

    Nothing here touches account-level data in Python: every per-account
    metric is computed inside DuckDB, only aggregates come back.
"""

from __future__ import annotations

import itertools
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import duckdb
import pandas as pd

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ETH_PSEUDO_TOKEN = "0x" + "ee" * 20
WETH_TOKEN = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
BEACON_DEPOSIT_CONTRACT = "0x00000000219ab540356cbb839cbe05303d7705fa"
ADDR_RE = re.compile(r"^0x[0-9a-f]{40}$")

# Same lists as distance_return_analytics/src/mpt_return_analysis.py
# (ETH_WRAPPER_ADDRESSES without WETH, which is folded into ETH here).
ETH_DERIVATIVE_TOKENS = [
    "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0",  # wstETH
    "0xbe9895146f7af43049ca1c1ae358b0541ea49704",  # cbETH
    "0xae78736cd615f374d3085123a210448e74fc6393",  # rETH
    "0xf951e335afb289353dc249e82926178eac7ded78",  # swETH
    "0xd5f7838f5c461feff7fe49ea5ebaf7728bb0adfa",  # mETH
    "0xa35b1b31ce002fbf2058d22f30f95d405200a15b",  # ETHx
    "0x5e8422345238f34275888049021821e8e08caa1f",  # frxETH
    "0xac3e018457b222d93114458476f3e3416abbe38f",  # sfrxETH
    "0xf1c9acdc66974dfb6decb12aa385b9cd01190e38",  # osETH
    "0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee",  # weETH
    "0xbf5495efe5db9ce00f80364c8b423567e58d2110",  # ezETH
    "0xa1290d69c65a6fe4df752f95823fae25cb99e5a7",  # rsETH
    "0xd9a442856c234a39a81a089c06451ebaa4306a72",  # pufETH
    "0xe95a203b1a91a908f9b9ce46459d101078c2c3cb",  # ankrETH
    "0xa2e3356610840701bdf5611a53974510ae27e2e1",  # wBETH
    "0xfe2e637202056d30016725477c5da089ab0a043a",  # sETH2
]
BTC_WRAPPER_TOKENS = [
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",  # WBTC
    "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf",  # cbBTC
    "0x18084fba666a33d37592fa2633fd49a74dd93a88",  # tBTC
    "0xeb4c2781e4eba804ce9a9803c67d0893436bb27d",  # renBTC
    "0x0316eb71485b0ab14103307bf65a021042c6d380",  # HBTC
    "0x9be89d2a4cd102d8fecc6bf9da793be995c22541",  # bBTC
    "0x8236a87084f8b84306f72007f36f2618a5634494",  # LBTC
    "0xc96de26018a54d51c097160568752c4e3bd6c364",  # FBTC
    "0x7a56e1c57c7475ccf742a1832b028f0456652f97",  # SolvBTC
    "0xf469fbd2abcd6b9de8e169d128226c0fc90a012e",  # pumpBTC
    "0x004e9c3ef86bc1ca1f0bb5c7662861ee93350568",  # uniBTC
    "0x3f67093dffd4f0af4f2918703c92b60acb7ad78b",  # 21BTC
    "0x73e0c0d45e048d25fc26fa3159b0aa04bfa4db98",  # kBTC
]

ALLOC_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ALLOC_DIR / "data"
DEFAULT_STABLECOINS_CSV = DEFAULT_DATA_DIR / "stablecoins.csv"

# Wealth buckets (USD, upper bound inclusive), labels as in value_buckets_minimal.csv
WEALTH_LABELS = ["0–1", "1–100", "100–1K", "1K–10K", "10K–100K", ">100K"]
_WEALTH_EDGES = [1, 100, 1_000, 10_000, 100_000]

# Portfolio-size buckets (number of valued assets after ETH/WETH folding)
N_LABELS = ["1", "2", "3–4", "5–9", "10+"]

ALL = "ALL"


# -----------------------------------------------------------------------------
# Logging / SQL literals
# -----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}", flush=True)


def sql_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def sql_str_list(values: Iterable[str]) -> str:
    """DuckDB list literal of lower-cased strings (typed, so an empty list works)."""
    vals = sorted({str(v).strip().lower() for v in values if str(v).strip()})
    if not vals:
        return "[]::VARCHAR[]"
    return "[" + ", ".join(sql_quote(v) for v in vals) + "]"


def wealth_code_sql(col: str) -> str:
    """UTINYINT code 0..5 into WEALTH_LABELS."""
    parts = [f"WHEN {col} <= {edge} THEN {i}" for i, edge in enumerate(_WEALTH_EDGES)]
    return f"(CASE {' '.join(parts)} ELSE {len(_WEALTH_EDGES)} END)::UTINYINT"


def nbucket_code_sql(col: str) -> str:
    """UTINYINT code 0..4 into N_LABELS."""
    return (f"(CASE WHEN {col} <= 1 THEN 0 WHEN {col} = 2 THEN 1 WHEN {col} <= 4 THEN 2 "
            f"WHEN {col} <= 9 THEN 3 ELSE 4 END)::UTINYINT")


def bin_code_sql(col: str, bins: Sequence[Tuple[str, str]]) -> str:
    """CASE expression mapping `col` to the index of the first matching bin.
    `bins` is a list of (label, condition) with `{x}` standing for the column;
    the last condition should be 'TRUE'."""
    parts = [f"WHEN {cond.format(x=col)} THEN {i}" for i, (_, cond) in enumerate(bins)]
    return f"(CASE {' '.join(parts)} END)::UTINYINT"


def grouping_sets_sql(dims: Sequence[str], always: Sequence[str] = ()) -> str:
    """GROUPING SETS clause equivalent to CUBE(dims), with `always` in every set."""
    sets = []
    for r in range(len(dims), -1, -1):
        for keep in itertools.combinations(dims, r):
            cols = list(keep) + list(always)
            sets.append("(" + ", ".join(cols) + ")")
    return "GROUPING SETS (" + ", ".join(sets) + ")"


# -----------------------------------------------------------------------------
# DuckDB / IO
# -----------------------------------------------------------------------------
def get_duck(threads: int, mem: str, temp_dir: Optional[str]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit={sql_quote(mem)}")
    con.execute("SET preserve_insertion_order=false")
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory={sql_quote(str(temp_dir))}")
    return con


def parquet_scan_target(path: str | Path) -> str:
    p = Path(path)
    if p.is_dir():
        return str(p.resolve() / "*" / "*.parquet")
    return str(p.resolve())


def list_blocks(con: duckdb.DuckDBPyConnection, path: str | Path) -> List[int]:
    """Snapshot blocks: from the block_number=N directory names of a 4_2 hive
    root, otherwise DISTINCT block_number of the parquet file(s)."""
    p = Path(path)
    if p.is_dir():
        blocks = []
        for d in p.glob("block_number=*"):
            val = d.name.split("=", 1)[1]
            if d.is_dir() and val.isdigit():
                blocks.append(int(val))
        if blocks:
            return sorted(blocks)
    rows = con.execute(
        "SELECT DISTINCT block_number::BIGINT FROM read_parquet(?, hive_partitioning=1) ORDER BY 1",
        [parquet_scan_target(p)],
    ).fetchall()
    return [int(r[0]) for r in rows]


def accounts_sql(path: str | Path, block: int) -> str:
    """SELECT returning (addr, holdings) for one snapshot, beacon deposit
    contract and NULL addresses removed. Reads only the block's partition
    directory when the input is a 4_2 hive root. Paths are inlined as quoted
    literals (not bound parameters) so the SELECT can sit inside a COPY."""
    p = Path(path)
    part = p / f"block_number={int(block)}"
    conds = ["wallet_address IS NOT NULL",
             f"lower(trim(wallet_address)) <> {sql_quote(BEACON_DEPOSIT_CONTRACT)}"]
    if p.is_dir() and part.is_dir():
        src = f"read_parquet({sql_quote(str(part.resolve() / '*.parquet'))})"
    else:
        src = f"read_parquet({sql_quote(parquet_scan_target(p))}, hive_partitioning=1)"
        conds.insert(0, f"block_number = {int(block)}")
    return f"SELECT lower(trim(wallet_address)) AS addr, holdings FROM {src} WHERE {' AND '.join(conds)}"


def load_ca_table(con: duckdb.DuckDBPyConnection, tsv_path: str | Path) -> int:
    """First column of the TSV into temp table ca(address); header tolerated."""
    addrs: Set[str] = set()
    with Path(tsv_path).open("r", encoding="utf-8") as f:
        for line in f:
            a = line.strip().split("\t")[0].strip().lower() if line.strip() else ""
            if ADDR_RE.match(a):
                addrs.add(a)
    con.register("ca_df", pd.DataFrame({"address": sorted(addrs)}))
    con.execute("CREATE OR REPLACE TEMP TABLE ca AS SELECT address FROM ca_df")
    con.unregister("ca_df")
    return len(addrs)


def label_groups(df: pd.DataFrame, dims: Sequence[str]) -> pd.DataFrame:
    """Turn DuckDB grouping output (dimension codes plus g_<dim> GROUPING flags)
    into readable labels: wallet_type (CA/EOA/ALL), wealth_bucket, n_bucket."""
    out = df.copy()
    for d in dims:
        flag = out[f"g_{d}"].astype(int) == 1
        if d == "is_ca":
            out["wallet_type"] = out["is_ca"].map({True: "CA", False: "EOA"})
            out.loc[flag, "wallet_type"] = ALL
        elif d == "wb":
            out["wealth_bucket"] = out["wb"].map(lambda c: WEALTH_LABELS[int(c)] if pd.notna(c) else ALL)
            out.loc[flag, "wealth_bucket"] = ALL
        elif d == "nb":
            out["n_bucket"] = out["nb"].map(lambda c: N_LABELS[int(c)] if pd.notna(c) else ALL)
            out.loc[flag, "n_bucket"] = ALL
    drop = [c for c in out.columns if c in dims or c.startswith("g_")]
    return out.drop(columns=drop)


def order_frame(df: pd.DataFrame, extra: Sequence[Tuple[str, Sequence[str]]] = ()) -> pd.DataFrame:
    """Stable, readable row order: ALL first, then the natural bucket order."""
    keys = []
    orders = [("wallet_type", [ALL, "EOA", "CA"]), ("wealth_bucket", [ALL] + WEALTH_LABELS),
              ("n_bucket", [ALL, "2+"] + N_LABELS)] + list(extra)
    tmp = df.copy()
    for col, order in orders:
        if col in tmp.columns:
            rank = {v: i for i, v in enumerate(order)}
            tmp[f"_o_{col}"] = tmp[col].map(lambda v: rank.get(v, len(rank)))
            keys.append(f"_o_{col}")
    lead = [c for c in ("block", "block_t", "block_t1", "cash_def") if c in tmp.columns]
    tmp = tmp.sort_values(lead + keys, kind="stable")
    return tmp.drop(columns=keys).reset_index(drop=True)


def append_csv(df: pd.DataFrame, path: Path, columns: Optional[List[str]] = None) -> None:
    if df is None or df.empty:
        return
    cols = columns or list(df.columns)
    df[cols].to_csv(path, mode="a", header=not path.exists(), index=False)


def prepare_outputs(paths: Sequence[Path], keys: Sequence[str], resume: bool) -> Set[tuple]:
    """Resume bookkeeping. Without --resume every output is deleted. With
    --resume a key (e.g. a block, or a block pair) counts as done only when it
    appears in every output that exists; rows of partially written keys are
    removed from all files so they can be recomputed without duplicates."""
    if not resume:
        for p in paths:
            p.unlink(missing_ok=True)
        return set()
    existing = [p for p in paths if p.exists()]
    if len(existing) < len(paths):
        # an output is missing entirely: nothing counts as done
        for p in existing:
            p.unlink()
        log(f"[resume] {len(paths) - len(existing)} output(s) missing, recomputing everything")
        return set()
    sets = []
    for p in existing:
        try:
            df = pd.read_csv(p, usecols=list(keys))
            sets.append({tuple(int(v) for v in r) for r in df[list(keys)].itertuples(index=False)})
        except Exception as e:  # pragma: no cover
            log(f"[resume] cannot read {p} ({e}); recomputing everything")
            for q in existing:
                q.unlink(missing_ok=True)
            return set()
    done = set.intersection(*sets) if sets else set()
    partial = set.union(*sets) - done if sets else set()
    if partial:
        for p in existing:
            df = pd.read_csv(p)
            mask = pd.Series(False, index=df.index)
            for key in partial:
                m = pd.Series(True, index=df.index)
                for k, v in zip(keys, key):
                    m &= df[k].astype("int64") == v
                mask |= m
            df[~mask].to_csv(p, index=False)
        log(f"[resume] dropped {len(partial)} partially written key(s) for recomputation: {sorted(partial)[:5]}")
    log(f"[resume] {len(done)} key(s) already complete")
    return done


def load_cash_tokens(stablecoins_csv: str | Path, cash_def: str) -> pd.DataFrame:
    """Rows of the curated stablecoin list that count as cash under `cash_def`:
    core (cash_core), extended (cash_extended) or extended_fx (cash_extended
    plus non-USD fiat stablecoins)."""
    df = pd.read_csv(stablecoins_csv, dtype=str, keep_default_na=False)
    df["contract_address"] = df["contract_address"].str.strip().str.lower()
    truthy = {"true", "1", "yes"}
    core = df["cash_core"].str.lower().isin(truthy)
    ext = df["cash_extended"].str.lower().isin(truthy)
    if cash_def == "core":
        sel = core
    elif cash_def == "extended":
        sel = ext
    elif cash_def == "extended_fx":
        sel = ext | (df["asset_class"] == "nonusd_fiat")
    else:
        raise ValueError(f"unknown cash definition {cash_def!r}")
    out = df[sel].copy()
    bad = out[~out["contract_address"].map(lambda a: bool(ADDR_RE.match(a)))]
    if not bad.empty:
        raise ValueError(f"malformed addresses in {stablecoins_csv}: {bad['contract_address'].tolist()[:5]}")
    clash = set(out["contract_address"]) & ({ETH_PSEUDO_TOKEN, WETH_TOKEN} | set(ETH_DERIVATIVE_TOKENS) | set(BTC_WRAPPER_TOKENS))
    if clash:
        raise ValueError(f"stablecoin list overlaps the ETH/BTC sets: {sorted(clash)}")
    return out
