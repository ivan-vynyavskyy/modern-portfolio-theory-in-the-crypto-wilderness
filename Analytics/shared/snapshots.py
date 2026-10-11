"""
snapshots.py

Description:
    Helpers shared by the analytics scripts that walk the structured
    transfers dump snapshot by snapshot:

    - load_snapshots(): the (date, block) list of reconstruction snapshots,
      read from a CSV (normally the _manifest.csv that
      4_2_partition_holdings.py writes into the holdings root), so every
      analysis covers exactly the blocks for which portfolios exist.
    - find_token_parquet_glob(): locate the ledger files of one asset in the
      dump (token_address=<addr>/<addr>[-shard-k].parquet).
"""

from __future__ import annotations

import glob
import pathlib
from typing import List, Optional

import pandas as pd

BLOCK_COLS = ["block_number", "block"]
DATE_COLS = ["date", "bound_date", "block_time_utc"]


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    return next((c for c in candidates if c in df.columns), None)


def load_snapshots(snapshots_csv: str, block_to_date_csv: Optional[str] = None,
                   verbose: bool = True) -> pd.DataFrame:
    """Read and check the (date, block) snapshot list.

    `snapshots_csv` needs a block column and may carry a date column (the
    4_2 _manifest.csv has both). Missing dates are resolved through
    `block_to_date_csv`; if both sources carry a date they are cross-checked.
    Returns a DataFrame with columns ['date' (YYYY-MM-DD), 'block'] sorted
    by block, duplicates removed.
    """
    snap = pd.read_csv(snapshots_csv)
    bcol = _find_col(snap, BLOCK_COLS)
    if bcol is None:
        raise ValueError(f"No block column ({BLOCK_COLS}) in {snapshots_csv}. Got {list(snap.columns)}")
    dcol = _find_col(snap, DATE_COLS)

    out = pd.DataFrame({"block": pd.to_numeric(snap[bcol], errors="coerce")})
    out["date"] = pd.to_datetime(snap[dcol], errors="coerce").dt.normalize() if dcol else pd.NaT
    n_bad = int(out["block"].isna().sum())
    out = out.dropna(subset=["block"]).astype({"block": "int64"})

    if block_to_date_csv:
        m = pd.read_csv(block_to_date_csv)
        mb, md = _find_col(m, BLOCK_COLS), _find_col(m, DATE_COLS)
        if mb is None or md is None:
            raise ValueError(f"Need block and date columns in {block_to_date_csv}. Got {list(m.columns)}")
        m = pd.DataFrame({"block": pd.to_numeric(m[mb], errors="coerce"),
                          "map_date": pd.to_datetime(m[md], errors="coerce").dt.normalize()})
        m = m.dropna().astype({"block": "int64"}).drop_duplicates("block")
        out = out.merge(m, on="block", how="left")
        if dcol:
            both = out["date"].notna() & out["map_date"].notna()
            n_mismatch = int((out.loc[both, "date"] != out.loc[both, "map_date"]).sum())
            if n_mismatch and verbose:
                print(f"[snapshots] WARN {n_mismatch} snapshot(s) where the date in {snapshots_csv} "
                      f"differs from {block_to_date_csv}; keeping the snapshots file's date")
        out["date"] = out["date"].fillna(out["map_date"])
        out = out.drop(columns=["map_date"])

    n_nodate = int(out["date"].isna().sum())
    if n_nodate:
        if dcol is None and not block_to_date_csv:
            raise ValueError(f"{snapshots_csv} has no date column; pass --block-to-date-csv to resolve dates")
        if verbose:
            print(f"[snapshots] WARN {n_nodate} snapshot(s) without a resolvable date are skipped: "
                  f"{out.loc[out['date'].isna(), 'block'].tolist()[:10]}")
        out = out.dropna(subset=["date"])

    n_before = len(out)
    out = out.drop_duplicates(subset=["block"]).sort_values("block").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"no usable snapshots in {snapshots_csv}")
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    if verbose:
        print(f"[snapshots] {len(out)} from {snapshots_csv}"
              f" | dropped: {n_bad} unparseable, {n_before - len(out)} duplicate block(s)"
              f" | first: {out['date'].iloc[0]} (block {out['block'].iloc[0]:,})"
              f" | last: {out['date'].iloc[-1]} (block {out['block'].iloc[-1]:,})")
    return out[["date", "block"]]


def find_token_parquet_glob(data_root: str | pathlib.Path, token_addr: str) -> Optional[str]:
    """Glob matching the ledger parquet file(s) of one asset, or None.

    Layouts written by 2_1_erc20_transfer_aggregator.py and
    2_0_eth_transfer_aggregator.py: <root>/token_address=<addr>/*.parquet
    (one file or address-hash-bucketed shards). Older single-file and
    address-named layouts are accepted as well.
    """
    root = pathlib.Path(data_root)
    addr = token_addr.lower()
    for g in (
        str(root / f"token_address={addr}" / "*.parquet"),
        str(root / f"token_address={addr}.parquet"),
        str(root / addr / "*.parquet"),
        str(root / addr / "**" / "*.parquet"),
    ):
        if glob.glob(g, recursive=True):
            return g
    return None


def cumulative_counts(hist: pd.DataFrame, target_blocks: List[int],
                      block_col: str = "block", count_col: str = "n") -> pd.Series:
    """Cumulative count at each target block from a (block -> count)
    histogram of first-seen blocks: number of entities whose first block is
    <= target. Returns a Series indexed by target block."""
    if hist.empty:
        return pd.Series([0] * len(target_blocks), index=list(target_blocks), dtype="int64")
    h = hist.sort_values(block_col)
    blocks = h[block_col].to_numpy()
    cum = h[count_col].to_numpy().cumsum()
    import numpy as np
    idx = np.searchsorted(blocks, np.asarray(target_blocks, dtype="int64"), side="right")
    vals = np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0)
    return pd.Series(vals.astype("int64"), index=list(target_blocks))
