#!/usr/bin/env python3
"""
analyze_portfolio_per_block.py

Run per-block analytics for a portfolio reconstruction Parquet file, with:

- Wallet types:
    * CA  = contract addresses (from TSV file)
    * EOA = externally owned accounts (everything else)
    * ALL = CA + EOA (overall)

- For each block and wallet_type, compute:

1) Wallet value distribution summary                 → wallet_value_distribution_per_block.csv
   - wallets_distinct
   - wallets_val_eq0
   - wallets_val_lt1
   - total_value_usd
   - % of total value held by top 1%, 5%, 10% addresses
       • top_1_share_pct, top_5_share_pct, top_10_share_pct
       • top_1_num_wallets, top_5_num_wallets, top_10_num_wallets

2) Simple averages + medians (per snapshot row)      → averages_per_block.csv
   - avg / median num_tokens (all wallets)
   - avg / median total_value_usd (all wallets)
   - avg / median num_tokens (wallets with value > 1 USD)
   - avg / median total_value_usd (wallets with value > 1 USD)

3) Asset-count buckets (#tokens per wallet)          → asset_buckets_per_block.csv
   - bucket: 0, 1, 2–5, 5–10, 10–20, 20–50, 50–100, >100
   - wallet_count, wallet_pct, value_sum_usd, value_pct

4) Value buckets + primary categories + top share    → value_buckets_with_categories_per_block.csv
   buckets: 0–1, 1–100, 100–1K, 1K–10K, 10K–100K, >100K
   For each (block, wallet_type, bucket):
   - wallet_count, wallet_pct, value_sum_usd, value_pct
   - avg / median tokens per wallet
   - avg / median top-held share (% of wallet)
   - For each primary category C:
       pct_wallets_holding__cat__C
       avg_share_among_holders__cat__C_pct
       median_share_among_holders__cat__C_pct
       bucket_value_share__cat__C_pct

5) Overall averages + medians for wallets above thresholds  → overall_averages_min_value_with_categories_per_block.csv
   For each (block, wallet_type, min_wallet_value_usd):
   - wallets_considered
   - avg / median num_tokens
   - avg / median wallet value
   - avg / median top_share_pct
   - For each primary category C:
       pct_wallets_holding__cat__C
       avg_share_among_holders__cat__C_pct
       median_share_among_holders__cat__C_pct
       cohort_value_share__cat__C_pct
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from typing import Dict, List, Set, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PRINT_WALLET_TYPES = ["ALL", "EOA", "CA"]
PRIMARY_ORDER: List[str] = [
    "Meme",
    "NFT",
    "Gaming (GameFi)",
    "Decentralized Exchange (DEX)",
    "Stablecoins",
    "Real World Assets (RWA)",
    "Artificial Intelligence (AI)",
    "Yield Farming / Liquidity",
    "Lending & Borrowing",
    "Derivatives / Perps / Options",
    "Liquid Staking / LSD",
    "SocialFi / Web3 Social",
    "Oracles",
    "Governance / DAO",
    "Launchpad / IDO",
    "Index & Basket",
    "Insurance / Risk",
    "Payments / Remittance",
    "Storage / File Storage",
    "Yield-Bearing / Interest",
    "CeFi / CEX Token",
    "Layer 2 (L2)",
    "Layer 1 (L1)",
    "Interoperability / Bridge",
    "Infrastructure / Base Layer",
    "DeFi",
    "VC / Fund Portfolio",
    "Experimental / Other",
    "Chain / App Ecosystem",
    "unknown",
]

def get_duck(threads: int = 8, mem: str = "12GB") -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with configured threads & memory limit."""
    con = duckdb.connect()
    con.execute("PRAGMA threads = ?", [threads])
    con.execute("PRAGMA memory_limit = ?", [mem])
    return con


def normalize_address(addr) -> str:
    """Normalize Ethereum address to lower-case hex string; robust to bytes/None."""
    if addr is None:
        return ""
    if isinstance(addr, (bytes, bytearray)):
        try:
            addr = addr.decode("utf-8", "ignore")
        except Exception:
            addr = ""
    return str(addr).lower()


def _dec(x):
    if x is None:
        return None
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _to_dec(x):
    # Backwards-compatible alias
    return _dec(x)


def load_contract_addresses(tsv_path: str | Path) -> Set[str]:
    """
    Load contract addresses from TSV file:
        <address>\t<block_or_other_column>
    Returns a set of lower-case addresses.
    """
    ca: Set[str] = set()
    tsv_path = Path(tsv_path)
    with tsv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            addr = normalize_address(parts[0])
            if addr:
                ca.add(addr)
    return ca


def add_wallet_type_column(df: pd.DataFrame, ca_set: Set[str]) -> pd.DataFrame:
    """
    Add 'wallet_address_norm' and 'wallet_type' columns:
      - wallet_type ∈ {"CA", "EOA"}
    """
    df = df.copy()
    df["wallet_address_norm"] = df["wallet_address"].map(normalize_address)
    df["wallet_type"] = df["wallet_address_norm"].apply(
        lambda a: "CA" if a in ca_set else "EOA"
    )
    return df


def build_wallet_type_maps(base: pd.DataFrame, ca_set: Set[str]):
    """
    Given a base DataFrame with 'wallet_address', return:
      - w_is_ca: dict[wallet_address -> bool]
      - w_type:  dict[wallet_address -> "CA"/"EOA"]
    """
    w_is_ca: Dict[str, bool] = {}
    w_type: Dict[str, str] = {}
    for wa in base["wallet_address"]:
        norm = normalize_address(wa)
        is_ca = norm in ca_set
        w_is_ca[wa] = is_ca
        w_type[wa] = "CA" if is_ca else "EOA"
    return w_is_ca, w_type


def slugify_category(cat: str) -> str:
    """
    Create a CSV-friendly slug for a category name.
    """
    s = (cat or "").lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def choose_bucket_winner(
    cat_metrics_by_bucket: Dict[str, Dict[str, float]],
    primary_rank: Dict[str, int],
) -> str | None:
    """
    Select winning category based on hold %, then bucket value share, then PRIMARY_ORDER rank, then alphabetical.
    Returns category name or None if no holder exists.
    """
    best_cat: str | None = None
    best_key: tuple | None = None
    for cat, metrics in cat_metrics_by_bucket.items():
        hold = float(metrics.get("hold_pct", 0.0) or 0.0)
        if hold <= 0:
            continue
        bucket_share = float(metrics.get("bucket_value_share_pct", 0.0) or 0.0)
        rank = primary_rank.get(cat, len(primary_rank))
        key = (-hold, -bucket_share, rank, cat)
        if best_key is None or key < best_key:
            best_key = key
            best_cat = cat
    return best_cat


def load_token_primary_categories(csv_path: str | Path) -> Tuple[Dict[str, str], List[str]]:
    """
    Load mapping from contract_address -> primary_category using PRIMARY_ORDER.
    Any category not in PRIMARY_ORDER (or empty/NaN) is mapped to "unknown".
    Returns (addr_to_category, categories=PRIMARY_ORDER).
    """
    df = pd.read_csv(csv_path)
    if "contract_address" not in df.columns or "primary_category" not in df.columns:
        raise ValueError("CSV must contain 'contract_address' and 'primary_category' columns")

    def _clean_cat(x) -> str:
        if pd.isna(x):
            return "unknown"
        c = str(x).strip()
        if not c or c not in PRIMARY_ORDER:
            return "unknown"
        return c

    df["contract_address"] = df["contract_address"].map(normalize_address)
    df["primary_category"] = df["primary_category"].map(_clean_cat)

    addr_to_cat: Dict[str, str] = {}
    for _, row in df.iterrows():
        addr = row["contract_address"]
        if not addr:
            continue
        addr_to_cat[addr] = row["primary_category"]

    return addr_to_cat, list(PRIMARY_ORDER)


def _assert_all_equals_sum(
    total_all: float, total_eoa: float, total_ca: float, label: str, epsilon: float = 1e-6
) -> None:
    target = total_eoa + total_ca
    if abs(total_all - target) > epsilon * max(1.0, abs(target)):
        raise AssertionError(f"{label}: ALL ({total_all}) != EOA+CA ({target})")


def assert_bucket_totals(
    df: pd.DataFrame,
    bucket_col: str,
    wallet_type_col: str,
    wallet_count_col: str,
    value_sum_col: str,
    epsilon: float = 1e-6,
) -> None:
    if df.empty:
        return

    totals = df.groupby(wallet_type_col).agg(
        wallets=(wallet_count_col, "sum"), value=(value_sum_col, "sum")
    )
    total_all = float(totals.loc["ALL", "wallets"]) if "ALL" in totals.index else 0.0
    total_eoa = float(totals.loc["EOA", "wallets"]) if "EOA" in totals.index else 0.0
    total_ca = float(totals.loc["CA", "wallets"]) if "CA" in totals.index else 0.0
    _assert_all_equals_sum(total_all, total_eoa, total_ca, "Wallet count totals", epsilon)

    val_all = float(totals.loc["ALL", "value"]) if "ALL" in totals.index else 0.0
    val_eoa = float(totals.loc["EOA", "value"]) if "EOA" in totals.index else 0.0
    val_ca = float(totals.loc["CA", "value"]) if "CA" in totals.index else 0.0
    _assert_all_equals_sum(val_all, val_eoa, val_ca, "Value totals", epsilon)

    for bucket in df[bucket_col].unique():
        sub = df[df[bucket_col] == bucket]
        w_all = float(sub.loc[sub[wallet_type_col] == "ALL", wallet_count_col].sum())
        w_eoa = float(sub.loc[sub[wallet_type_col] == "EOA", wallet_count_col].sum())
        w_ca = float(sub.loc[sub[wallet_type_col] == "CA", wallet_count_col].sum())
        _assert_all_equals_sum(
            w_all, w_eoa, w_ca, f"Bucket {bucket} wallet_count", epsilon
        )

        v_all = float(sub.loc[sub[wallet_type_col] == "ALL", value_sum_col].sum())
        v_eoa = float(sub.loc[sub[wallet_type_col] == "EOA", value_sum_col].sum())
        v_ca = float(sub.loc[sub[wallet_type_col] == "CA", value_sum_col].sum())
        _assert_all_equals_sum(v_all, v_eoa, v_ca, f"Bucket {bucket} value_sum", epsilon)


def append_sorted_csv(
    df: pd.DataFrame,
    out_path: Path,
    *,
    sort_cols: List[str],
    column_order: List[str] | None = None,
) -> bool:
    """
    Append DataFrame rows to a CSV, writing a header only on the first write.
    Returns True if any rows were written.
    """
    if df.empty:
        return False
    if column_order:
        df = df[column_order]
    df = df.sort_values(sort_cols)
    header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=header, index=False)
    return True


def compute_top_shares(values: np.ndarray) -> dict:
    """
    Given non-negative value array, return top 1/5/10% stats:
      - total_value_usd
      - top_1_share_pct, top_5_share_pct, top_10_share_pct
      - top_1_num_wallets, top_5_num_wallets, top_10_num_wallets
    """
    out = {
        "total_value_usd": 0.0,
        "top_1_share_pct": 0.0,
        "top_5_share_pct": 0.0,
        "top_10_share_pct": 0.0,
        "top_1_num_wallets": 0,
        "top_5_num_wallets": 0,
        "top_10_num_wallets": 0,
    }

    if values.size == 0:
        return out

    vals = np.clip(np.nan_to_num(values, nan=0.0), 0.0, None)
    total_val = float(vals.sum())
    out["total_value_usd"] = total_val

    if total_val <= 0:
        return out

    n = vals.size
    sorted_vals = np.sort(vals)[::-1]

    def calc_share(pct: float):
        k = max(1, int(np.ceil(pct * n)))
        share = float(sorted_vals[:k].sum() / total_val * 100.0)
        return k, share

    k1, s1 = calc_share(0.01)
    k5, s5 = calc_share(0.05)
    k10, s10 = calc_share(0.10)

    out["top_1_num_wallets"] = k1
    out["top_5_num_wallets"] = k5
    out["top_10_num_wallets"] = k10
    out["top_1_share_pct"] = s1
    out["top_5_share_pct"] = s5
    out["top_10_share_pct"] = s10
    return out


# ---------------------------------------------------------------------------
# 1) Wallet value distribution + top shares (per block, per wallet_type)
# ---------------------------------------------------------------------------

def summarize_wallet_value_distribution_by_type(
    path: str,
    block_number: int,
    ca_set: Set[str],
    threads: int = 8,
    mem: str = "12GB",
) -> List[dict]:
    """
    For a given block, compute per wallet_type in {ALL, CA, EOA}:
      - wallets_distinct
      - wallets_val_eq0
      - wallets_val_lt1
      - total_value_usd
      - top_1/5/10 % shares of total value
    """
    con = get_duck(threads, mem)
    q = """
    SELECT
        wallet_address,
        TRY_CAST(total_value_usd AS DOUBLE) AS val
    FROM read_parquet(?)
    WHERE block_number = ?
    """
    df = con.execute(q, [path, int(block_number)]).df()

    if df.empty:
        rows = []
        for wt in ["ALL", "CA", "EOA"]:
            rows.append(
                {
                    "block": int(block_number),
                    "wallet_type": wt,
                    "wallets_distinct": 0,
                    "wallets_val_eq0": 0,
                    "wallets_val_lt1": 0,
                    "total_value_usd": 0.0,
                    "top_1_share_pct": 0.0,
                    "top_5_share_pct": 0.0,
                    "top_10_share_pct": 0.0,
                    "top_1_num_wallets": 0,
                    "top_5_num_wallets": 0,
                    "top_10_num_wallets": 0,
                }
            )
        return rows

    df = add_wallet_type_column(df, ca_set)
    df["val"] = df["val"].astype(float)

    rows: List[dict] = []

    for wallet_type_label, sub in [
        ("ALL", df),
        ("CA", df[df["wallet_type"] == "CA"]),
        ("EOA", df[df["wallet_type"] == "EOA"]),
    ]:
        if sub.empty:
            rows.append(
                {
                    "block": int(block_number),
                    "wallet_type": wallet_type_label,
                    "wallets_distinct": 0,
                    "wallets_val_eq0": 0,
                    "wallets_val_lt1": 0,
                    "total_value_usd": 0.0,
                    "top_1_share_pct": 0.0,
                    "top_5_share_pct": 0.0,
                    "top_10_share_pct": 0.0,
                    "top_1_num_wallets": 0,
                    "top_5_num_wallets": 0,
                    "top_10_num_wallets": 0,
                }
            )
            continue

        vals = sub["val"].fillna(0.0).to_numpy()
        top_stats = compute_top_shares(vals)

        wallets_distinct = sub["wallet_address"].nunique()
        wallets_val_eq0 = sub.loc[sub["val"] == 0, "wallet_address"].nunique()
        wallets_val_lt1 = sub.loc[
            (sub["val"] < 1.0) & sub["val"].notna(), "wallet_address"
        ].nunique()

        row = {
            "block": int(block_number),
            "wallet_type": wallet_type_label,
            "wallets_distinct": int(wallets_distinct),
            "wallets_val_eq0": int(wallets_val_eq0),
            "wallets_val_lt1": int(wallets_val_lt1),
        }
        row.update(top_stats)
        rows.append(row)

        # Pretty print
        print(f"\nValue summary for block {block_number} / {wallet_type_label}:")
        print(f" - Distinct wallets:            {wallets_distinct:,}")
        print(f" - Wallets with value == 0:     {wallets_val_eq0:,}")
        print(f" - Wallets with value < 1 USD:  {wallets_val_lt1:,}")
        print(
            f" - Total value:                 ${top_stats['total_value_usd']:,.2f}\n"
            f"   Top 1%:  {top_stats['top_1_share_pct']:.2f}% "
            f"(n={top_stats['top_1_num_wallets']})\n"
            f"   Top 5%:  {top_stats['top_5_share_pct']:.2f}% "
            f"(n={top_stats['top_5_num_wallets']})\n"
            f"   Top 10%: {top_stats['top_10_share_pct']:.2f}% "
            f"(n={top_stats['top_10_num_wallets']})"
        )

    return rows


# ---------------------------------------------------------------------------
# 2) Simple averages + medians per row (per block, per wallet_type)
# ---------------------------------------------------------------------------

def summarize_averages_by_type(
    path: str,
    block_number: int,
    ca_set: Set[str],
    threads: int = 8,
    mem: str = "12GB",
) -> List[dict]:
    """
    Per-row snapshot stats per wallet_type in {ALL, CA, EOA}:
      - avg / median num_tokens (all)
      - avg / median value_usd (all)
      - avg / median num_tokens (val > 1)
      - avg / median value_usd (val > 1)
    """
    con = get_duck(threads, mem)
    q = """
    SELECT
      wallet_address,
      num_tokens::BIGINT AS num_tokens,
      TRY_CAST(total_value_usd AS DOUBLE) AS val
    FROM read_parquet(?)
    WHERE block_number = ?
    """
    df = con.execute(q, [path, int(block_number)]).df()
    if df.empty:
        rows = []
        for wt in ["ALL", "CA", "EOA"]:
            rows.append(
                {
                    "block": int(block_number),
                    "wallet_type": wt,
                    "avg_num_tokens": None,
                    "med_num_tokens": None,
                    "avg_value_usd": None,
                    "med_value_usd": None,
                    "avg_num_tokens_gt1": None,
                    "med_num_tokens_gt1": None,
                    "avg_value_usd_gt1": None,
                    "med_value_usd_gt1": None,
                }
            )
        return rows

    df = add_wallet_type_column(df, ca_set)

    rows: List[dict] = []

    def compute_stats(sub: pd.DataFrame, label: str) -> dict:
        if sub.empty:
            return {
                "block": int(block_number),
                "wallet_type": label,
                "avg_num_tokens": None,
                "med_num_tokens": None,
                "avg_value_usd": None,
                "med_value_usd": None,
                "avg_num_tokens_gt1": None,
                "med_num_tokens_gt1": None,
                "avg_value_usd_gt1": None,
                "med_value_usd_gt1": None,
            }

        nt = sub["num_tokens"].astype("float")
        v = sub["val"].astype("float")

        mask_gt1 = v > 1.0

        def safe_mean(x: pd.Series):
            return float(x.mean()) if x.notna().any() else None

        def safe_median(x: pd.Series):
            x = x.dropna()
            return float(x.median()) if not x.empty else None

        row = {
            "block": int(block_number),
            "wallet_type": label,
            "avg_num_tokens": safe_mean(nt),
            "med_num_tokens": safe_median(nt),
            "avg_value_usd": safe_mean(v),
            "med_value_usd": safe_median(v),
            "avg_num_tokens_gt1": safe_mean(nt[mask_gt1]),
            "med_num_tokens_gt1": safe_median(nt[mask_gt1]),
            "avg_value_usd_gt1": safe_mean(v[mask_gt1]),
            "med_value_usd_gt1": safe_median(v[mask_gt1]),
        }
        return row

    for label, sub in [
        ("ALL", df),
        ("CA", df[df["wallet_type"] == "CA"]),
        ("EOA", df[df["wallet_type"] == "EOA"]),
    ]:
        rows.append(compute_stats(sub, label))

        r = rows[-1]
        print(f"\nAverages / medians for block {block_number} / {label}:")
        fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
        print(f" - Avg #tokens (all):       {fmt(r['avg_num_tokens'])}")
        print(f" - Med #tokens (all):       {fmt(r['med_num_tokens'])}")
        print(f" - Avg value $ (all):       {fmt(r['avg_value_usd'])}")
        print(f" - Med value $ (all):       {fmt(r['med_value_usd'])}")
        print(f" - Avg #tokens (val>1):     {fmt(r['avg_num_tokens_gt1'])}")
        print(f" - Med #tokens (val>1):     {fmt(r['med_num_tokens_gt1'])}")
        print(f" - Avg value $ (val>1):     {fmt(r['avg_value_usd_gt1'])}")
        print(f" - Med value $ (val>1):     {fmt(r['med_value_usd_gt1'])}")

    return rows


# ---------------------------------------------------------------------------
# 3) Asset-count buckets per block, per wallet_type
# ---------------------------------------------------------------------------

def summarize_asset_buckets(
    path: str,
    block_number: int,
    ca_set: Set[str],
    *,
    threads: int = 8,
    mem: str = "12GB",
) -> pd.DataFrame:
    """
    For the given block, group wallets by num_tokens buckets and wallet_type:
      - bucket, wallet_type, wallet_count, wallet_pct, value_sum_usd, value_pct
    """
    con = get_duck(threads, mem)
    q = """
    SELECT
      wallet_address,
      CAST(num_tokens AS BIGINT) AS n,
      CAST(TRY_CAST(total_value_usd AS DOUBLE) AS DOUBLE) AS v
    FROM read_parquet(?)
    WHERE block_number = ?
    """
    df = con.execute(q, [path, int(block_number)]).df()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "wallet_type",
                "bucket",
                "wallet_count",
                "wallet_pct",
                "value_sum_usd",
                "value_pct",
            ]
        )

    df = add_wallet_type_column(df, ca_set)
    df = df[df["n"].notna()].copy()
    df["n"] = df["n"].astype("int64")
    df["v"] = df["v"].fillna(0.0).astype(float).clip(lower=0.0)

    def bucket_label(n: int) -> str:
        if n == 0:
            return "0"
        if n == 1:
            return "1"
        if 2 <= n <= 5:
            return "2–5"
        if 5 < n <= 10:
            return "5–10"
        if 10 < n <= 20:
            return "10–20"
        if 20 < n <= 50:
            return "20–50"
        if 50 < n <= 100:
            return "50–100"
        return ">100"

    def bucket_order(n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 1
        if 2 <= n <= 5:
            return 2
        if 5 < n <= 10:
            return 3
        if 10 < n <= 20:
            return 4
        if 20 < n <= 50:
            return 5
        if 50 < n <= 100:
            return 6
        return 7

    df["bucket"] = df["n"].apply(bucket_label)
    df["bucket_order"] = df["n"].apply(bucket_order)

    def compute_for_subset(sub: pd.DataFrame, wt: str) -> pd.DataFrame:
        if sub.empty:
            return pd.DataFrame()
        total_wallets = float(len(sub))
        total_value = float(sub["v"].sum()) if sub["v"].sum() != 0 else 1e-12

        grp = (
            sub.groupby(["bucket", "bucket_order"], as_index=False)
            .agg(
                wallet_count=("wallet_address", "count"),
                value_sum_usd=("v", "sum"),
            )
            .sort_values("bucket_order")
        )

        grp["wallet_pct"] = 100.0 * grp["wallet_count"] / total_wallets
        grp["value_pct"] = 100.0 * grp["value_sum_usd"] / total_value
        grp["wallet_type"] = wt
        return grp[
            [
                "wallet_type",
                "bucket",
                "wallet_count",
                "wallet_pct",
                "value_sum_usd",
                "value_pct",
            ]
        ]

    rows = [
        compute_for_subset(df, "ALL"),
        compute_for_subset(df[df["wallet_type"] == "EOA"], "EOA"),
        compute_for_subset(df[df["wallet_type"] == "CA"], "CA"),
    ]

    out = pd.concat(rows, ignore_index=True)
    out["wallet_count"] = out["wallet_count"].astype("int64")
    out["wallet_pct"] = out["wallet_pct"].round(2)
    out["value_sum_usd"] = out["value_sum_usd"].astype(float)
    out["value_pct"] = out["value_pct"].round(2)
    return out


def print_asset_buckets(df: pd.DataFrame, block_number: int):
    if df.empty:
        print(f"\nWallet distribution by # of assets (block={block_number}): no data")
        return

    for wt in PRINT_WALLET_TYPES:
        sub = df[df["wallet_type"] == wt]
        if sub.empty:
            continue
        total_wallets = int(sub["wallet_count"].sum())
        total_value = float(sub["value_sum_usd"].sum())
        print(f"\nWallet distribution by # of assets (block={block_number}, type={wt})")
        print(f"Total wallets: {total_wallets:,} | Total value: ${total_value:,.2f}\n")
        print(
            f"{'Bucket':>8} | {'Wallets':>12} | {'% of wallets':>12} | "
            f"{'Sum value $':>18} | {'% of value':>11}"
        )
        print("-" * 70)
        for _, r in sub.iterrows():
            print(
                f"{r['bucket']:>8} | {r['wallet_count']:12,d} | {r['wallet_pct']:12.2f}% | "
                f"{r['value_sum_usd']:18,.2f} | {r['value_pct']:11.2f}%"
            )


# ---------------------------------------------------------------------------
# 4) Value buckets with primary categories (per block, per wallet_type)
# ---------------------------------------------------------------------------

def summarize_value_buckets_with_categories(
    path: str,
    block_number: int,
    addr_to_cat: Dict[str, str],
    categories: List[str],
    ca_set: Set[str],
    *,
    threads: int = 8,
    mem: str = "12GB",
    arrow_batch_rows: int = 50_000,
) -> pd.DataFrame:
    """
    For the given block, per wallet_type in {ALL, CA, EOA}:
      1) Bucket wallets by total_value_usd (DuckDB).
      2) Stream holdings (Arrow) and parse JSON in Python.
      3) Aggregate per bucket:
         - wallet_count, wallet_pct
         - value_sum_usd, value_pct
         - avg_tokens, median_tokens
         - avg_top_share_pct, median_top_share_pct
         - For each primary category C:
             pct_wallets_holding__cat__{C}
             avg_share_among_holders__cat__{C}_pct
             median_share_among_holders__cat__{C}_pct
             bucket_value_share__cat__{C}_pct
             cohort_value_share__cat__{C}_pct
    """
    categories = list(categories)
    if "unknown" not in categories:
        categories.append("unknown")
    category_set = set(categories)

    con = get_duck(threads, mem)
    q = """
    WITH s AS (
      SELECT
        wallet_address,
        TRY_CAST(total_value_usd AS DOUBLE) AS v,
        TRY_CAST(num_tokens      AS BIGINT) AS n
      FROM read_parquet(?)
      WHERE block_number = ?
    ),
    cleaned AS (
      SELECT wallet_address,
             GREATEST(COALESCE(v, 0.0), 0.0) AS v,
             COALESCE(n, 0)                  AS n
      FROM s
    )
    SELECT
      wallet_address,
      v,
      n,
      CASE
        WHEN v <    1      THEN '0–1'
        WHEN v <  100      THEN '1–100'
        WHEN v < 1000      THEN '100–1K'
        WHEN v < 10000     THEN '1K–10K'
        WHEN v < 100000    THEN '10K–100K'
        ELSE '>100K'
      END AS bucket
    FROM cleaned
    """
    base = con.execute(q, [path, int(block_number)]).df()
    if base.empty:
        cols = [
            "wallet_type",
            "bucket",
            "wallet_count",
            "wallet_pct",
            "value_sum_usd",
            "value_pct",
            "avg_tokens",
            "median_tokens",
            "avg_top_share_pct",
            "median_top_share_pct",
        ]
        for c in categories:
            slug = slugify_category(c)
            cols.append(f"pct_wallets_holding__cat__{slug}")
            cols.append(f"avg_share_among_holders__cat__{slug}_pct")
            cols.append(f"median_share_among_holders__cat__{slug}_pct")
            cols.append(f"bucket_value_share__cat__{slug}_pct")
            cols.append(f"cohort_value_share__cat__{slug}_pct")
        return pd.DataFrame(columns=cols)

    # maps for bucket, totals, and wallet type
    base["wallet_address_norm"] = base["wallet_address"].map(normalize_address)
    base["is_ca"] = base["wallet_address_norm"].isin(ca_set)
    w_bucket: Dict[str, str] = dict(zip(base["wallet_address"], base["bucket"]))
    w_total: Dict[str, float] = dict(zip(base["wallet_address"], base["v"].astype(float)))
    w_tokens: Dict[str, int] = dict(zip(base["wallet_address"], base["n"].astype(int)))
    w_is_ca: Dict[str, bool] = dict(zip(base["wallet_address"], base["is_ca"]))

    buckets = ["0–1", "1–100", "100–1K", "1K–10K", "10K–100K", ">100K"]
    bucket_order = {b: i for i, b in enumerate(buckets)}
    wallet_types = PRINT_WALLET_TYPES

    # agg[wallet_type][bucket] = {...}
    agg = {
        wt: {
            b: {
                "wallet_count": 0,
                "value_sum": 0.0,
                "sum_tokens": 0.0,
                "sum_top_share_pct": 0.0,
                "tokens_values": [],
                "top_share_pct_values": [],
                "category_holders": {c: 0 for c in categories},
                "category_share_sum": {c: 0.0 for c in categories},
                "category_share_values": {c: [] for c in categories},
                "category_value_sum": {c: 0.0 for c in categories},
            }
            for b in buckets
        }
        for wt in wallet_types
    }

    # totals per wallet_type for percentages
    total_wallets = {
        "ALL": len(base),
        "CA": int(base["is_ca"].sum()),
        "EOA": int((~base["is_ca"]).sum()),
    }
    total_value = {
        "ALL": float(base["v"].sum()) if base["v"].sum() != 0 else 1e-12,
        "CA": float(base.loc[base["is_ca"], "v"].sum()) if base.loc[base["is_ca"], "v"].sum() != 0 else 1e-12,
        "EOA": float(base.loc[~base["is_ca"], "v"].sum()) if base.loc[~base["is_ca"], "v"].sum() != 0 else 1e-12,
    }

    # Step 2: stream holdings rows (Arrow)
    dataset = ds.dataset(path, format="parquet")
    filt = (ds.field("block_number") == int(block_number))
    scanner = dataset.scanner(
        columns=["wallet_address", "holdings", "block_number"],
        filter=filt,
        batch_size=arrow_batch_rows,
        use_threads=True,
    )

    for batch in scanner.to_batches():
        wa_arr = batch.column(batch.schema.get_field_index("wallet_address"))
        hold_arr = batch.column(batch.schema.get_field_index("holdings"))

        for wa, holdings_obj in zip(wa_arr.to_pylist(), hold_arr.to_pylist()):
            bucket = w_bucket.get(wa)
            if bucket is None:
                continue
            total = float(w_total.get(wa, 0.0))
            ntok = int(w_tokens.get(wa, 0))
            is_ca = bool(w_is_ca.get(wa, False))

            # Parse JSON holdings
            if holdings_obj is None:
                items = []
            elif isinstance(holdings_obj, (bytes, bytearray)):
                try:
                    items = json.loads(holdings_obj.decode("utf-8", "ignore"))
                except Exception:
                    items = []
            elif isinstance(holdings_obj, str):
                try:
                    items = json.loads(holdings_obj)
                except Exception:
                    items = []
            else:
                items = []

            cat_usd: Dict[str, Decimal] = {}
            top_val = Decimal(0)

            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    addr = normalize_address(it.get("token_address"))
                    v = _dec(it.get("value_usd"))
                    if not v or v <= 0:
                        continue

                    if v > top_val:
                        top_val = v
                    cat = addr_to_cat.get(addr, "unknown")
                    if cat not in category_set:
                        cat = "unknown"
                    cat_usd[cat] = cat_usd.get(cat, Decimal(0)) + v

            top_share = float((top_val / Decimal(total)) * Decimal(100)) if total > 0 else 0.0

            # Update aggregators for ALL and CA/EOA
            types_to_update = ["ALL", "CA"] if is_ca else ["ALL", "EOA"]
            for wt in types_to_update:
                a = agg[wt][bucket]
                a["wallet_count"] += 1
                a["value_sum"] += total
                a["sum_tokens"] += ntok
                a["sum_top_share_pct"] += top_share
                a["tokens_values"].append(ntok)
                a["top_share_pct_values"].append(top_share)

                if total > 0:
                    for cat, usd in cat_usd.items():
                        share_pct = 100.0 * float(usd / Decimal(total))
                        a["category_holders"][cat] += 1
                        a["category_share_sum"][cat] += share_pct
                        a["category_share_values"][cat].append(share_pct)
                        a["category_value_sum"][cat] += float(usd)

    # Category totals per wallet_type for cohort-level value share
    category_totals = {wt: {c: 0.0 for c in categories} for wt in wallet_types}
    for wt in wallet_types:
        for b in buckets:
            for c in categories:
                category_totals[wt][c] += agg[wt][b]["category_value_sum"][c]

    # Step 3: build final DataFrame
    rows = []
    for wt in wallet_types:
        grand_wallets = total_wallets[wt] if total_wallets[wt] > 0 else 1
        grand_value = total_value[wt] if total_value[wt] > 0 else 1e-12

        for b in buckets:
            a = agg[wt][b]
            wc = a["wallet_count"]
            vsum = a["value_sum"]

            if wc > 0:
                med_tokens = float(median(a["tokens_values"])) if a["tokens_values"] else 0.0
                med_top_share_pct = float(median(a["top_share_pct_values"])) if a["top_share_pct_values"] else 0.0
            else:
                med_tokens = med_top_share_pct = 0.0

            row = {
                "wallet_type": wt,
                "bucket": b,
                "bucket_order": bucket_order[b],
                "wallet_count": wc,
                "wallet_pct": round(100.0 * wc / grand_wallets, 2),
                "value_sum_usd": vsum,
                "value_pct": round(100.0 * vsum / grand_value, 2),
                "avg_tokens": round(a["sum_tokens"] / wc, 2) if wc else 0.0,
                "median_tokens": round(med_tokens, 2),
                "avg_top_share_pct": round(a["sum_top_share_pct"] / wc, 2) if wc else 0.0,
                "median_top_share_pct": round(med_top_share_pct, 2),
            }

            for c in categories:
                slug = slugify_category(c)
                holders = a["category_holders"].get(c, 0)
                share_sum = a["category_share_sum"].get(c, 0.0)
                med_share = (
                    float(median(a["category_share_values"][c]))
                    if a["category_share_values"][c]
                    else 0.0
                )
                row[f"pct_wallets_holding__cat__{slug}"] = (
                    round(100.0 * holders / grand_wallets, 2) if grand_wallets else 0.0
                )
                row[f"avg_share_among_holders__cat__{slug}_pct"] = (
                    round(share_sum / holders, 2) if holders else 0.0
                )
                row[f"median_share_among_holders__cat__{slug}_pct"] = round(med_share, 2)
                row[f"bucket_value_share__cat__{slug}_pct"] = (
                    round(100.0 * a["category_value_sum"][c] / max(vsum, 1e-12), 2)
                    if wc
                    else 0.0
                )
                row[f"cohort_value_share__cat__{slug}_pct"] = round(
                    100.0 * category_totals[wt][c] / max(grand_value, 1e-12), 2
                )

            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["wallet_type", "bucket_order"]).drop(columns=["bucket_order"])
    return df[[c for c in df.columns]]


def print_value_buckets_with_categories(
    df: pd.DataFrame, block_number: int, categories: List[str]
):
    if df.empty:
        print(f"\nWallet distribution by USD value (block={block_number}): no data")
        return

    categories = list(categories)
    if "unknown" not in categories:
        categories.append("unknown")
    buckets = ["0–1", "1–100", "100–1K", "1K–10K", "10K–100K", ">100K"]
    bucket_order = {b: i for i, b in enumerate(buckets)}
    primary_rank = {cat: idx for idx, cat in enumerate(categories)}

    for wt in PRINT_WALLET_TYPES:
        sub = df[df["wallet_type"] == wt]
        if sub.empty:
            continue

        total_wallets = int(sub["wallet_count"].sum())
        total_value = float(sub["value_sum_usd"].sum())
        print(
            f"\nWallet distribution by USD value + primary categories + top-share "
            f"(block={block_number}, type={wt})"
        )
        print(f"Total wallets: {total_wallets:,} | Total value: ${total_value:,.2f}\n")

        print(
            "{:<11} {:>12} {:>10} {:>18} {:>9} {:>11} {:>11} {:>18} {:>18}".format(
                "Bucket",
                "Wallets",
                "% wallets",
                "Sum value $",
                "% value",
                "Avg tok",
                "Med tok",
                "Avg top%",
                "Med top%",
            )
        )
        print("-" * 150)
        sub_sorted = sub.sort_values(
            "bucket",
            key=lambda s: s.map(lambda b: bucket_order.get(b, len(bucket_order))),
        )
        for _, r in sub_sorted.iterrows():
            print(
                "{:<11} {:>12,d} {:>10.2f}% {:>18,.2f} {:>9.2f}% {:>11.2f} {:>11.2f} {:>18.2f}% {:>18.2f}%".format(
                    r["bucket"],
                    r["wallet_count"],
                    r["wallet_pct"],
                    r["value_sum_usd"],
                    r["value_pct"],
                    r["avg_tokens"],
                    r["median_tokens"],
                    r["avg_top_share_pct"],
                    r["median_top_share_pct"],
                )
            )

        print(
            f"\nPrimary category winner per bucket — type={wt} (block={block_number})"
        )
        print(
            f"{'Bucket':<12}{'Winner category':<40}{'hold%':>10}{'avg_share':>13}"
            f"{'med_share':>12}{'bucket_value_share':>20}"
        )
        print("-" * 100)
        for _, r in sub_sorted.iterrows():
            cat_metrics: Dict[str, Dict[str, float]] = {}
            for cat in categories:
                slug = slugify_category(cat)
                cat_metrics[cat] = {
                    "hold_pct": float(r.get(f"pct_wallets_holding__cat__{slug}", 0.0) or 0.0),
                    "bucket_value_share_pct": float(r.get(f"bucket_value_share__cat__{slug}_pct", 0.0) or 0.0),
                    "avg_share_pct": float(r.get(f"avg_share_among_holders__cat__{slug}_pct", 0.0) or 0.0),
                    "median_share_pct": float(r.get(f"median_share_among_holders__cat__{slug}_pct", 0.0) or 0.0),
                }
            winner = choose_bucket_winner(cat_metrics, primary_rank)
            if winner is None:
                winner = "(none)"
                hold = avg_share = med_share = bucket_share = 0.0
            else:
                metrics = cat_metrics[winner]
                hold = metrics["hold_pct"]
                avg_share = metrics["avg_share_pct"]
                med_share = metrics["median_share_pct"]
                bucket_share = metrics["bucket_value_share_pct"]
            print(
                f"{r['bucket']:<12}{winner:<40}{hold:>10.2f}%{avg_share:>13.2f}%"
                f"{med_share:>12.2f}%{bucket_share:>20.2f}%"
            )


# ---------------------------------------------------------------------------
# 5) Overall averages + medians for wallets > min_value (per block, per wallet_type)
# ---------------------------------------------------------------------------

def overall_averages_min_value_with_categories(
    path: str,
    block_number: int,
    addr_to_cat: Dict[str, str],
    categories: List[str],
    ca_set: Set[str],
    *,
    min_wallet_value_usd: float = 1.0,
    threads: int = 8,
    mem: str = "12GB",
    arrow_batch_rows: int = 50_000,
) -> List[dict]:
    """
    Overall averages & medians for wallets with total_value_usd > min_wallet_value_usd,
    per wallet_type in {ALL, CA, EOA}:
      - avg / median num_tokens
      - avg / median wallet value (USD)
      - avg / median top-held share (%)
      - For each primary category C:
          pct_wallets_holding__cat__{C}
          avg_share_among_holders__cat__{C}_pct
          median_share_among_holders__cat__{C}_pct
          cohort_value_share__cat__{C}_pct
    """
    categories = list(categories)
    if "unknown" not in categories:
        categories.append("unknown")
    category_set = set(categories)
    wallet_types = PRINT_WALLET_TYPES

    con = get_duck(threads, mem)
    q = """
    SELECT
      wallet_address,
      TRY_CAST(total_value_usd AS DOUBLE) AS v,
      COALESCE(TRY_CAST(num_tokens AS BIGINT), 0) AS n
    FROM read_parquet(?)
    WHERE block_number = ?
      AND TRY_CAST(total_value_usd AS DOUBLE) > ?
    """
    base = con.execute(q, [path, int(block_number), float(min_wallet_value_usd)]).df()
    if base.empty:
        rows = []
        for wt in wallet_types:
            row = {
                "block": int(block_number),
                "wallet_type": wt,
                "min_wallet_value_usd": float(min_wallet_value_usd),
                "wallets_considered": 0,
                "avg_num_tokens": 0.0,
                "median_num_tokens": 0.0,
                "avg_value_usd": 0.0,
                "median_value_usd": 0.0,
                "avg_top_share_pct": 0.0,
                "median_top_share_pct": 0.0,
            }
            for c in categories:
                slug = slugify_category(c)
                row[f"pct_wallets_holding__cat__{slug}"] = 0.0
                row[f"avg_share_among_holders__cat__{slug}_pct"] = 0.0
                row[f"median_share_among_holders__cat__{slug}_pct"] = 0.0
                row[f"cohort_value_share__cat__{slug}_pct"] = 0.0
            rows.append(row)

        print(
            f"\nOverall averages @ block {block_number} "
            f"(wallets > ${min_wallet_value_usd}): 0 wallets (ALL/CA/EOA)"
        )
        return rows

    base["wallet_address_norm"] = base["wallet_address"].map(normalize_address)
    base["is_ca"] = base["wallet_address_norm"].isin(ca_set)

    # Pre-compute wallet counts per type
    wallets_considered = {
        "ALL": len(base),
        "CA": int(base["is_ca"].sum()),
        "EOA": int((~base["is_ca"]).sum()),
    }

    # maps for totals & tokens
    w_total: Dict[str, float] = dict(zip(base["wallet_address"], base["v"].astype(float)))
    w_tokens: Dict[str, int] = dict(zip(base["wallet_address"], base["n"].astype(int)))
    w_is_ca: Dict[str, bool] = dict(zip(base["wallet_address"], base["is_ca"]))

    # dataset for streaming holdings
    dataset = ds.dataset(path, format="parquet")
    filt = (ds.field("block_number") == int(block_number))
    scanner = dataset.scanner(
        columns=["wallet_address", "holdings", "block_number"],
        filter=filt,
        batch_size=arrow_batch_rows,
        use_threads=True,
    )

    # Aggregators per wallet_type
    sum_tokens = {wt: 0.0 for wt in wallet_types}
    sum_value = {wt: 0.0 for wt in wallet_types}
    sum_top_share_pct = {wt: 0.0 for wt in wallet_types}
    tokens_values = {wt: [] for wt in wallet_types}
    value_values = {wt: [] for wt in wallet_types}
    top_share_values = {wt: [] for wt in wallet_types}
    category_holders = {wt: {c: 0 for c in categories} for wt in wallet_types}
    category_share_sum = {wt: {c: 0.0 for c in categories} for wt in wallet_types}
    category_share_values = {wt: {c: [] for c in categories} for wt in wallet_types}
    category_value_sum = {wt: {c: 0.0 for c in categories} for wt in wallet_types}

    eligible = set(base["wallet_address"])
    eligible_map = {wa: True for wa in eligible}

    for batch in scanner.to_batches():
        wa_arr = batch.column(batch.schema.get_field_index("wallet_address"))
        hold_arr = batch.column(batch.schema.get_field_index("holdings"))

        for wa, holdings_obj in zip(wa_arr.to_pylist(), hold_arr.to_pylist()):
            if wa not in eligible_map:
                continue

            total = float(w_total.get(wa, 0.0))
            ntok = int(w_tokens.get(wa, 0))
            is_ca = bool(w_is_ca.get(wa, False))

            # Parse JSON holdings
            if holdings_obj is None:
                items = []
            elif isinstance(holdings_obj, (bytes, bytearray)):
                try:
                    items = json.loads(holdings_obj.decode("utf-8", "ignore"))
                except Exception:
                    items = []
            elif isinstance(holdings_obj, str):
                try:
                    items = json.loads(holdings_obj)
                except Exception:
                    items = []
            else:
                items = []

            cat_usd: Dict[str, Decimal] = {}
            top_val = Decimal(0)

            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    addr = normalize_address(it.get("token_address"))
                    v = _to_dec(it.get("value_usd"))
                    if not v or v <= 0:
                        continue

                    if v > top_val:
                        top_val = v
                    cat = addr_to_cat.get(addr, "unknown")
                    if cat not in category_set:
                        cat = "unknown"
                    cat_usd[cat] = cat_usd.get(cat, Decimal(0)) + v

            if total > 0:
                top_share = float((top_val / Decimal(total)) * Decimal(100))

                types_to_update = ["ALL", "CA"] if is_ca else ["ALL", "EOA"]
                for wt in types_to_update:
                    sum_tokens[wt] += ntok
                    sum_value[wt] += total
                    sum_top_share_pct[wt] += top_share
                    tokens_values[wt].append(ntok)
                    value_values[wt].append(total)
                    top_share_values[wt].append(top_share)

                    for cat, usd in cat_usd.items():
                        share = 100.0 * float(usd / Decimal(total))
                        category_holders[wt][cat] += 1
                        category_share_sum[wt][cat] += share
                        category_share_values[wt][cat].append(share)
                        category_value_sum[wt][cat] += float(usd)

            eligible_map.pop(wa, None)

    rows: List[dict] = []
    for wt in wallet_types:
        num_wallets = wallets_considered[wt]
        if num_wallets == 0:
            avg_num_tokens = avg_value_usd = avg_top_share_pct = 0.0
        else:
            avg_num_tokens = sum_tokens[wt] / num_wallets
            avg_value_usd = sum_value[wt] / num_wallets
            avg_top_share_pct = sum_top_share_pct[wt] / num_wallets

        med_num_tokens = float(median(tokens_values[wt])) if tokens_values[wt] else 0.0
        med_value_usd = float(median(value_values[wt])) if value_values[wt] else 0.0
        med_top_share_pct = float(median(top_share_values[wt])) if top_share_values[wt] else 0.0

        result = {
            "block": int(block_number),
            "wallet_type": wt,
            "min_wallet_value_usd": float(min_wallet_value_usd),
            "wallets_considered": num_wallets,
            "avg_num_tokens": round(avg_num_tokens, 3),
            "median_num_tokens": round(med_num_tokens, 3),
            "avg_value_usd": round(avg_value_usd, 3),
            "median_value_usd": round(med_value_usd, 3),
            "avg_top_share_pct": round(avg_top_share_pct, 3),
            "median_top_share_pct": round(med_top_share_pct, 3),
        }

        for cat in categories:
            slug = slugify_category(cat)
            holders = category_holders[wt][cat]
            avg_share = category_share_sum[wt][cat] / holders if holders else 0.0
            med_share = (
                float(median(category_share_values[wt][cat]))
                if category_share_values[wt][cat]
                else 0.0
            )
            result[f"pct_wallets_holding__cat__{slug}"] = (
                round(100.0 * holders / num_wallets, 3) if num_wallets else 0.0
            )
            result[f"avg_share_among_holders__cat__{slug}_pct"] = round(avg_share, 3)
            result[f"median_share_among_holders__cat__{slug}_pct"] = round(med_share, 3)
            result[f"cohort_value_share__cat__{slug}_pct"] = round(
                100.0 * category_value_sum[wt][cat] / max(sum_value[wt], 1e-12), 3
            )

        rows.append(result)

        print(
            f"\nOverall averages @ block {block_number} "
            f"(wallets > ${min_wallet_value_usd}, type={wt}): {num_wallets:,}\n"
            f" - Avg # tokens        : {result['avg_num_tokens']:.2f} "
            f"(median {result['median_num_tokens']:.2f})\n"
            f" - Avg wallet value    : ${result['avg_value_usd']:,.2f} "
            f"(median {result['median_value_usd']:,.2f})\n"
            f" - Avg top-held share  : {result['avg_top_share_pct']:.2f}% "
            f"(median {result['median_top_share_pct']:.2f}%)"
        )

        # Show top 8 categories by cohort value share
        def cat_rank(cat: str) -> float:
            slug = slugify_category(cat)
            return result.get(f"cohort_value_share__cat__{slug}_pct", 0.0)

        top_cats = sorted(categories, key=lambda c: (-cat_rank(c), slugify_category(c)))[:8]
        print(" - Primary categories (top by cohort value share):")
        for cat in top_cats:
            slug = slugify_category(cat)
            print(
                f"    {cat}: hold={result[f'pct_wallets_holding__cat__{slug}']:.2f}%  "
                f"avg_share(holders)={result[f'avg_share_among_holders__cat__{slug}_pct']:.2f}%  "
                f"med_share(holders)={result[f'median_share_among_holders__cat__{slug}_pct']:.2f}%  "
                f"cohort_value_share={result[f'cohort_value_share__cat__{slug}_pct']:.2f}%"
            )

    return rows


# ---------------------------------------------------------------------------
# Orchestration over all blocks
# ---------------------------------------------------------------------------

def get_all_blocks(path: str, threads: int = 8, mem: str = "12GB") -> List[int]:
    con = get_duck(threads, mem)
    q = "SELECT DISTINCT block_number FROM read_parquet(?) ORDER BY block_number"
    rows = con.execute(q, [path]).fetchall()
    return [int(r[0]) for r in rows]


def main():
    parser = argparse.ArgumentParser(
        description="Per-block analytics for portfolio reconstruction Parquet, "
                    "split by CA/EOA and with top-1/5/10% concentration."
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to portfolio reconstruction Parquet file.",
    )
    parser.add_argument(
        "--ca-tsv",
        required=True,
        help="TSV file with contract addresses (CA). First column must be address.",
    )
    parser.add_argument(
        "--out-dir",
        default="analytics_out",
        help="Output directory for CSV files.",
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="DuckDB / Arrow threads."
    )
    parser.add_argument(
        "--mem", type=str, default="12GB", help="DuckDB memory limit (e.g. '12GB')."
    )
    parser.add_argument(
        "--arrow-batch-rows",
        type=int,
        default=50_000,
        help="Arrow batch size for streaming holdings.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[1.0, 100_000.0],
        help="List of min_wallet_value_usd thresholds for overall averages.",
    )
    parser.add_argument(
        "--token-status-csv",
        default="/mnt/data/status_eth_tokens.with_primary_category.csv",
        help="CSV with token metadata (contract_address, primary_category).",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Optional limit on number of blocks to process (debugging).",
    )

    args = parser.parse_args()
    parquet_path = str(Path(args.parquet).resolve())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "value_dist": out_dir / "wallet_value_distribution_per_block.csv",
        "averages": out_dir / "averages_per_block.csv",
        "asset_buckets": out_dir / "asset_buckets_per_block.csv",
        "value_buckets": out_dir / "value_buckets_with_categories_per_block.csv",
        "overall": out_dir / "overall_averages_min_value_with_categories_per_block.csv",
    }
    # Clean previous outputs so this run mirrors the old overwrite behavior.
    for p in output_paths.values():
        p.unlink(missing_ok=True)

    # --- load CA list from TSV ---
    ca_set = load_contract_addresses(args.ca_tsv)
    print(f"Loaded {len(ca_set):,} contract addresses (CA) from {args.ca_tsv}")

    # --- load token primary categories ---
    addr_to_cat, categories = load_token_primary_categories(args.token_status_csv)
    print(
        f"Loaded {len(addr_to_cat):,} token primary categories "
        f"from {args.token_status_csv} (categories={len(categories)})"
    )

    # discover blocks
    blocks = get_all_blocks(parquet_path, threads=args.threads, mem=args.mem)
    if args.max_blocks is not None:
        blocks = blocks[: args.max_blocks]

    print(f"Found {len(blocks)} distinct blocks in {parquet_path}")

    written_flags = {k: False for k in output_paths}

    for i, block in enumerate(blocks, start=1):
        print(f"\n==================== Block {block} ({i}/{len(blocks)}) ====================")

        # 1) value distribution + top shares
        vdist_rows = summarize_wallet_value_distribution_by_type(
            parquet_path, block_number=block, ca_set=ca_set,
            threads=args.threads, mem=args.mem
        )
        vdist_df = pd.DataFrame(vdist_rows)
        written_flags["value_dist"] |= append_sorted_csv(
            vdist_df,
            output_paths["value_dist"],
            sort_cols=["block", "wallet_type"],
        )

        # 2) avg & median per row
        avg_rows = summarize_averages_by_type(
            parquet_path, block_number=block, ca_set=ca_set,
            threads=args.threads, mem=args.mem
        )
        avg_df = pd.DataFrame(avg_rows)
        avg_cols = ["block", "wallet_type"] + [
            c for c in avg_df.columns if c not in ("block", "wallet_type")
        ]
        avg_df = avg_df[avg_cols]
        written_flags["averages"] |= append_sorted_csv(
            avg_df,
            output_paths["averages"],
            sort_cols=["block", "wallet_type"],
        )

        # 3) asset-count buckets
        asset_df = summarize_asset_buckets(
            parquet_path, block, ca_set=ca_set, threads=args.threads, mem=args.mem
        )
        assert_bucket_totals(
            asset_df,
            bucket_col="bucket",
            wallet_type_col="wallet_type",
            wallet_count_col="wallet_count",
            value_sum_col="value_sum_usd",
        )
        print_asset_buckets(asset_df, block)
        if not asset_df.empty:
            asset_df = asset_df.copy()
            asset_df.insert(0, "block", block)
            written_flags["asset_buckets"] |= append_sorted_csv(
                asset_df,
                output_paths["asset_buckets"],
                sort_cols=["block", "wallet_type", "bucket"],
            )

        # 4) value buckets with primary categories
        df_buckets = summarize_value_buckets_with_categories(
            parquet_path,
            block,
            addr_to_cat,
            categories,
            ca_set=ca_set,
            threads=args.threads,
            mem=args.mem,
            arrow_batch_rows=args.arrow_batch_rows,
        )
        assert_bucket_totals(
            df_buckets,
            bucket_col="bucket",
            wallet_type_col="wallet_type",
            wallet_count_col="wallet_count",
            value_sum_col="value_sum_usd",
        )
        print_value_buckets_with_categories(df_buckets, block, categories)
        if not df_buckets.empty:
            df_buckets = df_buckets.copy()
            df_buckets.insert(0, "block", block)
            written_flags["value_buckets"] |= append_sorted_csv(
                df_buckets,
                output_paths["value_buckets"],
                sort_cols=["block", "wallet_type", "bucket"],
            )

        # 5) overall averages for thresholds
        block_overall_rows: List[dict] = []
        for thr in args.thresholds:
            overall_rows = overall_averages_min_value_with_categories(
                parquet_path,
                block,
                addr_to_cat,
                categories,
                ca_set=ca_set,
                min_wallet_value_usd=thr,
                threads=args.threads,
                mem=args.mem,
                arrow_batch_rows=args.arrow_batch_rows,
            )
            block_overall_rows.extend(overall_rows)

        if block_overall_rows:
            df_overall = pd.DataFrame(block_overall_rows)
            overall_cols = ["block", "wallet_type", "min_wallet_value_usd"] + [
                c
                for c in df_overall.columns
                if c not in ("block", "wallet_type", "min_wallet_value_usd")
            ]
            written_flags["overall"] |= append_sorted_csv(
                df_overall,
                output_paths["overall"],
                sort_cols=["min_wallet_value_usd", "block", "wallet_type"],
                column_order=overall_cols,
            )

    if any(written_flags.values()):
        print(f"\nDone. CSVs written to: {out_dir.resolve()}")
    else:
        print("\nNo output rows were written (all blocks empty).")


if __name__ == "__main__":
    main()
