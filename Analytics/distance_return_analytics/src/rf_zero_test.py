#!/usr/bin/env python3
"""
rf_zero_test.py

Description:
    Test the impact of changing risk_free_rate_annual from 0.05 to 0.0
    on the MPT backtest pipeline's wallet-level results. Only max-Sharpe
    fields can differ; all other strategy fields are used as a built-in
    sanity check (they should be byte-identical).

    Three-stage pipeline:
      1. Sample N wallets per block from the intersection of the source
         parquet (recon) and existing data_extended.
      2. Re-run the backtest with risk_free_rate_annual = 0.0.
      3. Inner-join old (rf=0.05) and new (rf=0.0), compute per-wallet
         and per-block diffs, and emit an overall summary.

    Stages skip if their output already exists; pass --force to re-run.

Usage:
    python rf_zero_test.py [--force]
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from locations import Location

# ----------------------------------------------------------------------------
# Paths & config
# ----------------------------------------------------------------------------

BLOCKS: List[int] = [
    9193266, 9393154, 9581792, 9782602, 9976964, 10176690, 10370274,
    10570485, 10771925, 10966874, 11167817, 11363270, 11565019, 11766939,
    11948960, 12150245, 12344945, 12545219, 12738509, 12936340, 13136427,
    13330090, 13527859, 13717847, 13916166, 14116761, 14297759, 14497034,
    14688630, 14881677, 15053226, 15253306, 15449618, 15649595, 15871480,
    16086234, 16308190, 16530248, 16730072, 16950603, 17162287, 17382266,
    17595510, 17816434, 18037988, 18251965, 18473543, 18687851, 18908895,
    19129889, 19336607, 19557289, 19771560, 19993250, 20207949, 20429973,
    20651994, 20866919, 21089069, 21303934, 21525891, 21747950, 21948292,
    22170335, 22385294, 22606143, 22820674, 23042514, 23264566, 23479244,
    23700767, 23914921,
]

SOURCE_PARQUETS: List[str] = [
    str(Location.RECON_PARQUET_2020_2022),
    str(Location.RECON_PARQUET_2023_2025),
]

DATA_EXTENDED_DIR = str(Location.MPT_DATA_EXTENDED)

PRICES_DIR        = str(Location.PRICES_DIR)
BLOCK_TO_DATE_CSV = str(Location.BLOCK_TO_DATE_CSV)
SUPPLY_CSV        = str(Location.TOKEN_INEQUALITY_CSV)

OUT_DIR                   = Path(str(Location.MPT_RF_TEST_DIR))
OUT_SAMPLE_DIR            = OUT_DIR / "sample_wallets"           # per-block parquets (resume-safe)
OUT_SAMPLE_PARQUET        = OUT_DIR / "sample_wallets.parquet"   # consolidated (built at end of stage 1)
OUT_RESULTS_DIR           = OUT_DIR / "results_rf_zero"
OUT_RESULTS_CSV           = OUT_DIR / "results_rf_zero_summary.csv"
OUT_COMPARISON_DIR        = OUT_DIR / "comparison"
OUT_COMPARISON_PARQUET    = OUT_COMPARISON_DIR / "wallet_diffs.parquet"
OUT_PER_BLOCK_SUMMARY_CSV = OUT_COMPARISON_DIR / "per_block_summary.csv"
OUT_OVERALL_SUMMARY_CSV   = OUT_COMPARISON_DIR / "overall_summary.csv"

# Knobs
WALLETS_PER_BLOCK     = 5_000
RANDOM_SEED           = 20250506
WORKERS               = 32
BATCH_SIZE            = 5_000  # one batch per block (sample is ≤5K/block)
BATCH_TIMEOUT_MIN     = 30
HORIZON_DAYS          = 20
MEAN_REVERSION_STR    = 0.5
REGIME_LOOKBACK_DAYS  = 60
REGIME_TOP_K_ASSETS   = 30

# ----------------------------------------------------------------------------
# Locate and load the production pipeline as a library
# ----------------------------------------------------------------------------

try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:  # in a notebook __file__ may be undefined
    THIS_DIR = Path.cwd()


def _load_pipeline_module() -> Any:
    """Locate ``5_0_portfolio_backtest_pipeline.py`` and import it as a module.

    Searches THIS_DIR, then cwd, then candidate dirs supplied via the
    ``RF_TEST_PIPELINE_DIR`` env var. The pipeline's own top-level code adds
    its parent dir to ``sys.path`` so ``helpers.*`` imports work afterward.
    """
    import os

    candidates: List[Path] = [THIS_DIR, Path.cwd()]
    extra = os.environ.get("RF_TEST_PIPELINE_DIR")
    if extra:
        candidates.append(Path(extra))

    name = "5_0_portfolio_backtest_pipeline.py"
    for d in candidates:
        p = d / name
        if p.exists():
            spec = importlib.util.spec_from_file_location("backtest_pipeline", str(p))
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            # Register in sys.modules BEFORE exec so the module is discoverable
            # by ProcessPoolExecutor workers when they unpickle function refs.
            # Without this, pickle stores ("backtest_pipeline", "_process_wallet_task")
            # but the worker can't find the module on import — every task fails
            # at unpickle time and the pipeline silently swallows the exceptions
            # via line 2011's `except Exception: continue`, giving processed=0.
            sys.modules["backtest_pipeline"] = mod
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod
    raise FileNotFoundError(
        f"Could not find {name!r} in any of: {[str(c) for c in candidates]}. "
        "Set RF_TEST_PIPELINE_DIR to point to its directory."
    )


backtest_pipeline = _load_pipeline_module()
from risk_config import RiskConfig  # type: ignore  # noqa: E402

# ----------------------------------------------------------------------------
# Robust parquet reading
# ----------------------------------------------------------------------------

def _decode_dict_columns(t: pa.Table) -> pa.Table:
    """Cast any dictionary-encoded columns down to their underlying value type
    so that row groups with mixed encodings can be concatenated via pandas."""
    new_arrays = []
    new_names = []
    for i in range(t.num_columns):
        arr = t.column(i)
        if pa.types.is_dictionary(arr.type):
            try:
                arr = arr.cast(arr.type.value_type)
            except Exception:
                pass
        new_arrays.append(arr)
        new_names.append(t.schema.field(i).name)
    try:
        return pa.Table.from_arrays(new_arrays, names=new_names)
    except Exception:
        return t


def _safe_read_parquet(file_path: Path, columns: List[str]) -> pd.DataFrame:
    """Read a single parquet file, robust to row groups with mismatched
    column encodings (e.g. plain int64 vs dictionary<int32>).

    ``pq.read_table`` and ``ds.dataset`` both unify the schema across all row
    groups in the file before reading any column. If two row groups disagree
    on the encoding of *any* column (even one we didn't request), the read
    fails. We try, in order:

      1. Unified ``pq.read_table`` (fast path).
      2. Per-row-group reads with column filter, then decode dicts and concat.
      3. Per-row-group reads WITHOUT column filter (slower, but bypasses any
         cross-column schema validation), then select columns post-hoc.

    Returns an empty DataFrame if every approach fails.
    """
    # 1) Unified fast path
    try:
        t = pq.read_table(str(file_path), columns=columns)
        return t.to_pandas()
    except Exception:
        pass

    # Open file once for the fallbacks
    try:
        pf = pq.ParquetFile(str(file_path))
    except Exception as e:
        print(f"[warn] could not open {file_path}: {e}")
        return pd.DataFrame()

    try:
        file_cols = pf.schema_arrow.names
    except Exception:
        try:
            file_cols = pf.schema.names  # legacy interface
        except Exception:
            file_cols = list(columns)
    avail = [c for c in columns if c in file_cols]
    if not avail:
        avail = list(columns)

    # 2) Per-row-group with column filter and dict decoding
    dfs: List[pd.DataFrame] = []
    needed_full_rg_fallback = 0
    for i in range(pf.num_row_groups):
        try:
            rg = pf.read_row_group(i, columns=avail)
            rg = _decode_dict_columns(rg)
            dfs.append(rg.to_pandas())
            continue
        except Exception:
            pass
        # 3) Per-row-group, no column filter — read everything, drop the rest.
        try:
            rg_full = pf.read_row_group(i)
            rg_full = _decode_dict_columns(rg_full)
            df_full = rg_full.to_pandas()
            in_df = [c for c in avail if c in df_full.columns]
            if in_df:
                dfs.append(df_full[in_df].copy())
            del df_full, rg_full
            needed_full_rg_fallback += 1
        except Exception as e:
            print(f"[warn] {file_path} row_group {i}: {e}")
            continue

    if needed_full_rg_fallback > 0:
        print(
            f"[info] {file_path.name}: "
            f"{needed_full_rg_fallback}/{pf.num_row_groups} row groups needed "
            "full-column fallback (column-filtered read failed for them)"
        )
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ----------------------------------------------------------------------------
# Source-parquet helpers
# ----------------------------------------------------------------------------

_SOURCE_DATASETS: List[Optional[ds.Dataset]] = []
_BLOCK_SOURCE_CACHE: Dict[int, ds.Dataset] = {}


def _ensure_source_datasets_loaded() -> None:
    """Lazily open both source datasets (idempotent)."""
    if _SOURCE_DATASETS:
        return
    for path in SOURCE_PARQUETS:
        try:
            _SOURCE_DATASETS.append(ds.dataset(path, format="parquet"))
            print(f"[source] opened {path}")
        except Exception as e:
            print(f"[warn] failed to open {path}: {e}")
            _SOURCE_DATASETS.append(None)


def get_source_for_block(block_number: int) -> Optional[ds.Dataset]:
    """Return the source dataset that contains rows for ``block_number``.

    Caches the answer per block. Returns None if no source has the block.
    """
    _ensure_source_datasets_loaded()
    if block_number in _BLOCK_SOURCE_CACHE:
        return _BLOCK_SOURCE_CACHE[block_number]
    for d in _SOURCE_DATASETS:
        if d is None:
            continue
        try:
            n = d.count_rows(filter=ds.field("block_number") == int(block_number))
            if n > 0:
                _BLOCK_SOURCE_CACHE[block_number] = d
                return d
        except Exception as e:
            print(f"[warn] count_rows failed on a source for block {block_number}: {e}")
            continue
    return None


def _normalize_wallets_fast(s: pd.Series) -> pd.Series:
    """Vectorized wallet-address normalization where possible, fallback to apply."""
    if s.empty:
        return s
    sample = s.iloc[0]
    if isinstance(sample, str):
        out = s.str.lower()
        # Ensure 0x-prefix
        no_prefix = ~out.str.startswith("0x", na=False)
        if no_prefix.any():
            out = out.where(~no_prefix, "0x" + out.where(no_prefix, ""))
        return out
    return s.apply(backtest_pipeline._normalize_wallet_address)


# ----------------------------------------------------------------------------
# data_extended helpers
# ----------------------------------------------------------------------------

def get_data_extended_wallets(block_number: int) -> Optional[pd.Index]:
    """Read the wallet_address column from data_extended for one block."""
    block_dir = Path(DATA_EXTENDED_DIR) / f"block_number={block_number}"
    if not block_dir.exists():
        return None
    parquet_files = list(block_dir.rglob("*.parquet"))
    if not parquet_files:
        return None
    wallets: List[str] = []
    for f in parquet_files:
        df = _safe_read_parquet(f, ["wallet_address"])
        if df.empty or "wallet_address" not in df.columns:
            continue
        wallets.extend(df["wallet_address"].tolist())
    if not wallets:
        return None
    return pd.Index(
        [str(w).lower() for w in wallets if w is not None and str(w)],
        name="wallet_address",
    )


# ----------------------------------------------------------------------------
# Stage 1: sample wallets
# ----------------------------------------------------------------------------

def _norm_token_address(raw: Any) -> Optional[str]:
    """Token addresses come from pyarrow as Python ``str`` OR ``bytes``
    depending on the column type (``string`` vs ``binary``). Normalize to
    canonical lowercase 0x-hex.

    A naive ``json.dumps(..., default=str)`` on bytes produces a useless
    ``"b'\\xab\\xcd...'"`` string — looks valid to ``str(item).lower()`` but
    matches no real token in the price universe.
    """
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        try:
            b = bytes(raw)
            if not b:
                return None
            return "0x" + b.hex()
        except Exception:
            return None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if not s:
            return None
        return s if s.startswith("0x") else ("0x" + s)
    # numpy scalars and pyarrow boxes
    if hasattr(raw, "as_py"):
        try:
            return _norm_token_address(raw.as_py())
        except Exception:
            return None
    if hasattr(raw, "item"):
        try:
            return _norm_token_address(raw.item())
        except Exception:
            return None
    return None


def _ensure_holdings_serializable(v: Any) -> Optional[str]:
    """Convert a holdings value to a JSON list of ``{token_address, balance}``
    dicts in the exact form the downstream pipeline expects.

    Handles all the upstream representations we've seen:
      - list of dict (typical pyarrow → pandas conversion of list<struct>)
      - already-serialized JSON string
      - numpy object array
      - dict of arrays (less common — ``{"token_address": [...], "balance": [...]}``)

    Critically, normalizes ``token_address`` through ``_norm_token_address``
    so that bytes-typed columns (the common case in recon parquets) become
    canonical ``0x...`` strings rather than the junk ``"b'\\xab...'"`` you
    get from ``str(some_bytes)``.

    Returns ``None`` if no valid items survive normalization (so the
    pipeline can drop the row).
    """
    if v is None:
        return None
    if isinstance(v, float) and not pd.notna(v):
        return None

    # Step 1: get a list of items
    arr: Any
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
        except Exception:
            return None
        if not isinstance(parsed, list):
            return None
        arr = parsed
    elif isinstance(v, np.ndarray):
        arr = v.tolist()
    elif isinstance(v, (list, tuple)):
        arr = list(v)
    elif isinstance(v, dict):
        # dict-of-arrays fallback
        keys = ("token_address", "address", "contract_address")
        bal_keys = ("balance", "amount", "value")
        tok_key = next((k for k in keys if k in v), None)
        bal_key = next((k for k in bal_keys if k in v), None)
        if tok_key is None or bal_key is None:
            return None
        try:
            arr = [
                {"token_address": t, "balance": b}
                for t, b in zip(list(v[tok_key]), list(v[bal_key]))
            ]
        except Exception:
            return None
    else:
        return None

    # Step 2: normalize each item to {token_address: 0x..., balance: float}
    out: List[Dict[str, Any]] = []
    for item in arr:
        if not isinstance(item, dict):
            try:
                item = dict(item)
            except Exception:
                continue

        raw_tok = (
            item.get("token_address")
            if "token_address" in item
            else item.get("address", item.get("contract_address"))
        )
        tok = _norm_token_address(raw_tok)
        if not tok:
            continue

        raw_bal = (
            item.get("balance")
            if "balance" in item
            else item.get("amount", item.get("value"))
        )
        if raw_bal is None:
            continue
        if hasattr(raw_bal, "as_py"):
            try:
                raw_bal = raw_bal.as_py()
            except Exception:
                pass
        if hasattr(raw_bal, "item"):
            try:
                raw_bal = raw_bal.item()
            except Exception:
                pass
        try:
            bal = float(raw_bal)
        except (TypeError, ValueError):
            try:
                bal = float(str(raw_bal))
            except Exception:
                continue
        if not (bal > 0):
            continue

        out.append({"token_address": tok, "balance": bal})

    if not out:
        return None
    return json.dumps(out, separators=(",", ":"))


def repair_sample_holdings() -> int:
    """Re-normalize the ``holdings`` column in the existing sample files.

    Use this if your stage-2 run reports ``processed=0/N`` for every block —
    that's the symptom of token-address bytes getting mangled during JSON
    serialization. This rewrites the sample parquets in place with the
    fixed encoding, so you don't need to redo stage 1.

    Returns the total number of rows after repair.
    """
    print("=" * 80)
    print("REPAIR: re-normalize holdings in existing sample parquets")
    print("=" * 80)

    n_total = 0
    n_dropped = 0

    # 1) Per-block files
    if OUT_SAMPLE_DIR.exists():
        per_block = sorted(OUT_SAMPLE_DIR.glob("block_number=*.parquet"))
        for f in per_block:
            try:
                df = pq.read_table(str(f)).to_pandas()
            except Exception as e:
                print(f"[repair] could not read {f.name}: {e}")
                continue
            n0 = len(df)
            df["holdings"] = df["holdings"].apply(_ensure_holdings_serializable)
            df = df.dropna(subset=["holdings"]).reset_index(drop=True)
            n1 = len(df)
            n_dropped += (n0 - n1)
            if n1 == 0:
                print(f"[repair] {f.name}: empty after re-normalize, leaving file")
                continue
            tmp = f.with_suffix(".repaired.parquet")
            try:
                pq.write_table(
                    pa.Table.from_pandas(df, preserve_index=False),
                    str(tmp), compression="zstd",
                )
                tmp.replace(f)
                n_total += n1
            except Exception as e:
                print(f"[repair] {f.name}: write failed: {e}")
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
        print(f"[repair] {len(per_block)} per-block files processed")

    # 2) Consolidated file — rebuild from per-block
    if OUT_SAMPLE_DIR.exists() and any(OUT_SAMPLE_DIR.glob("block_number=*.parquet")):
        _consolidate_sample_dir()
    elif OUT_SAMPLE_PARQUET.exists():
        # Fallback: repair the consolidated file directly
        df = pq.read_table(str(OUT_SAMPLE_PARQUET)).to_pandas()
        n0 = len(df)
        df["holdings"] = df["holdings"].apply(_ensure_holdings_serializable)
        df = df.dropna(subset=["holdings"]).reset_index(drop=True)
        n1 = len(df)
        n_dropped += (n0 - n1)
        n_total += n1
        tmp = OUT_SAMPLE_PARQUET.with_suffix(".repaired.parquet")
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            str(tmp), compression="zstd",
        )
        tmp.replace(OUT_SAMPLE_PARQUET)
        print(f"[repair] consolidated parquet repaired: {n0:,} → {n1:,} rows")

    # 3) Smoke test
    if OUT_SAMPLE_PARQUET.exists():
        sample = pq.read_table(
            str(OUT_SAMPLE_PARQUET), columns=["holdings"]
        ).to_pandas().head(2)
        if not sample.empty:
            test = backtest_pipeline._safe_hm(sample["holdings"].iloc[0])
            toks = list(test.keys())[:3]
            looks_ok = all(
                isinstance(t, str) and t.startswith("0x") and len(t) == 42
                for t in toks
            )
            print(f"[repair] smoke check on first row:")
            print(f"  parsed tokens : {toks}")
            print(f"  look canonical: {looks_ok}")
            if not looks_ok:
                print("[repair] WARNING: tokens still don't look right after repair.")
                print("[repair] Run diagnose_sample.py to inspect the source format.")
        # Now also clear stale stage-2 results (they were all empty anyway)
        if OUT_RESULTS_DIR.exists():
            stale_blocks = list(OUT_RESULTS_DIR.glob("block_number=*"))
            if stale_blocks:
                print(f"[repair] note: {len(stale_blocks)} block dirs in "
                      f"{OUT_RESULTS_DIR} from the failed run — pass "
                      f"force=True to stage_2_run_rf_zero to overwrite, or "
                      f"delete them manually.")

    print(f"[repair] total rows after repair: {n_total:,} (dropped {n_dropped:,})")
    return n_total


def sample_block_wallets(
    block_number: int,
    n_target: int,
    rng: np.random.Generator,
) -> Optional[pd.DataFrame]:
    """Memory-light per-block sampling.

    Strategy:
      1. Read the data_extended wallet set for this block (the universe of
         comparable wallets — they all already passed the original
         eligibility filters, so we don't need to re-check ``num_tokens``).
      2. Pre-sample ``n_target`` wallet addresses from that set.
      3. Stream the source parquet in batches and keep only rows whose
         wallet_address is in the target set. Peak memory is one batch
         (~50K rows) instead of the full block (~5M rows).

    This avoids the OOM that the previous "load full block, then filter,
    then sample" approach hit on late-2024/2025 blocks with 5M+ wallets.
    """
    src = get_source_for_block(block_number)
    if src is None:
        print(f"[block {block_number}] no source parquet contains this block; skipping")
        return None

    # 1) Universe of comparable wallets
    de = get_data_extended_wallets(block_number)
    if de is None or len(de) == 0:
        print(f"[block {block_number}] no data_extended wallets; skipping")
        return None
    de_arr = np.asarray(de.tolist(), dtype=object)
    n_universe = int(len(de_arr))

    # 2) Pre-sample target wallets (deterministic per-block via the rng arg)
    n_take = int(min(n_target, n_universe))
    if n_take < n_universe:
        idx = rng.choice(n_universe, size=n_take, replace=False)
        target_arr = de_arr[idx]
    else:
        target_arr = de_arr
    target_set = set(target_arr.tolist())
    n_targets = len(target_set)

    # 3) Source columns we want
    cols = ["wallet_address", "block_number", "holdings", "total_value_usd", "num_tokens"]
    available = [c for c in cols if c in src.schema.names]
    if "holdings" not in available or "wallet_address" not in available:
        missing = sorted(set(cols) - set(available))
        print(f"[block {block_number}] source missing required cols {missing}; skipping")
        return None

    # 4) Stream in small batches, keeping only target rows
    BATCH = 50_000
    parts: List[pd.DataFrame] = []
    n_seen = 0
    n_matched = 0
    t0 = time.perf_counter()
    try:
        scanner = src.scanner(
            filter=ds.field("block_number") == int(block_number),
            columns=available,
            batch_size=BATCH,
        )
        batch_iter = scanner.to_batches()
    except Exception:
        # Fallback: dataset.to_batches keyword API
        batch_iter = src.to_batches(
            filter=ds.field("block_number") == int(block_number),
            columns=available,
            batch_size=BATCH,
        )

    for batch in batch_iter:
        df_b = batch.to_pandas()
        n_seen += len(df_b)
        # Normalize wallet addresses to match data_extended format
        df_b["wallet_address"] = _normalize_wallets_fast(
            df_b["wallet_address"].astype("object")
        )
        df_b = df_b[df_b["wallet_address"].isin(target_set)]
        if not df_b.empty:
            parts.append(df_b.copy())
            n_matched += len(df_b)
        del batch, df_b
        # Early exit when all targets found (a wallet appears at most once per block)
        if n_matched >= n_targets:
            break

    if not parts:
        print(
            f"[block {block_number}] no rows matched after streaming "
            f"(seen {n_seen:,}); skipping"
        )
        return None

    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    # 5) Defensive eligibility re-check (data_extended wallets should already
    #    pass these, but a stale data_extended could include a now-missing row)
    if "num_tokens" in df.columns:
        df = df[pd.to_numeric(df["num_tokens"], errors="coerce").fillna(0) >= 2]
    df["__hm"] = df["holdings"].apply(backtest_pipeline._safe_hm)
    df = df[df["__hm"].map(bool)].drop(columns=["__hm"])

    # 6) Tidy schema
    df["block_number"] = int(block_number)
    df["holdings"] = df["holdings"].apply(_ensure_holdings_serializable)
    keep_order = ["wallet_address", "block_number", "holdings", "total_value_usd", "num_tokens"]
    df = df[[c for c in keep_order if c in df.columns]].reset_index(drop=True)

    elapsed = time.perf_counter() - t0
    print(
        f"[block {block_number}] sampled {len(df):,}/{n_take:,} target  "
        f"(de_n={n_universe:,}, scanned {n_seen:,} rows, {elapsed:.1f}s)"
    )
    return df


def _block_sample_path(blk: int) -> Path:
    """Path of the per-block sample parquet."""
    return OUT_SAMPLE_DIR / f"block_number={blk}.parquet"


def _consolidate_sample_dir() -> Optional[Path]:
    """Concatenate per-block parquets in OUT_SAMPLE_DIR into a single file at
    OUT_SAMPLE_PARQUET. Returns the consolidated path, or None if no files."""
    files = sorted(OUT_SAMPLE_DIR.glob("block_number=*.parquet"))
    if not files:
        return None
    tables: List[pa.Table] = []
    total = 0
    for f in files:
        try:
            t = pq.read_table(str(f))
            tables.append(t)
            total += t.num_rows
        except Exception as e:
            print(f"[warn] could not read per-block file {f.name}: {e}")
    if not tables:
        return None
    combined = pa.concat_tables(tables, promote=True) if len(tables) > 1 else tables[0]
    OUT_SAMPLE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, str(OUT_SAMPLE_PARQUET), compression="zstd")
    print(f"[stage 1] consolidated {len(files)} per-block files "
          f"({total:,} rows) → {OUT_SAMPLE_PARQUET}")
    return OUT_SAMPLE_PARQUET


def stage_1_sample(force: bool = False, blocks: Optional[List[int]] = None) -> Path:
    """Sample wallets per block and write each to its own parquet.

    Per-block files at ``OUT_SAMPLE_DIR/block_number={blk}.parquet`` are
    written immediately after each block is sampled, so a crash at block N
    leaves blocks 1..N-1 intact and re-running picks up at block N.

    After all blocks are processed, the per-block files are consolidated into
    a single parquet at ``OUT_SAMPLE_PARQUET`` for stage 2.
    """
    print("=" * 80)
    print("STAGE 1: sample wallets")
    print("=" * 80)
    OUT_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    blocks_to_run = list(blocks) if blocks is not None else list(BLOCKS)

    if (
        OUT_SAMPLE_PARQUET.exists()
        and not force
        and all(_block_sample_path(b).exists() for b in blocks_to_run)
    ):
        print(f"[stage 1] consolidated parquet already exists; skipping")
        return OUT_SAMPLE_PARQUET

    n_done = 0
    n_skipped_existing = 0
    n_failed = 0

    for i, blk in enumerate(blocks_to_run, 1):
        print(f"--- [{i}/{len(blocks_to_run)}] block {blk} ---")
        out_file = _block_sample_path(blk)
        if out_file.exists() and not force:
            try:
                n_rows = pq.read_metadata(str(out_file)).num_rows
            except Exception:
                n_rows = -1
            print(f"[block {blk}] already sampled ({n_rows} rows); skipping")
            n_skipped_existing += 1
            continue

        # Per-block deterministic RNG so re-samples are reproducible.
        rng = np.random.default_rng([RANDOM_SEED, blk])
        try:
            df = sample_block_wallets(blk, WALLETS_PER_BLOCK, rng)
        except Exception as e:
            print(f"[error] block {blk} failed: {e!r}")
            try:
                backtest_pipeline.log_exc(f"sample_block_wallets failed for block {blk}: {e}")
            except Exception:
                pass
            df = None
            n_failed += 1

        if df is not None and not df.empty:
            tmp = out_file.with_suffix(".parquet.tmp")
            try:
                pq.write_table(
                    pa.Table.from_pandas(df, preserve_index=False),
                    str(tmp), compression="zstd",
                )
                tmp.replace(out_file)  # atomic
                n_done += 1
            except Exception as e:
                print(f"[error] block {blk}: could not write {out_file}: {e}")
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
                n_failed += 1
        del df
        gc.collect()

    n_block_files = len(list(OUT_SAMPLE_DIR.glob("block_number=*.parquet")))
    print(
        f"[stage 1] {n_done} new, {n_skipped_existing} resumed, {n_failed} failed; "
        f"{n_block_files} total per-block files in {OUT_SAMPLE_DIR}"
    )

    if n_block_files == 0:
        raise RuntimeError("Stage 1 produced no per-block samples")

    # Smoke-test holdings parsing on the first available file
    first_file = sorted(OUT_SAMPLE_DIR.glob("block_number=*.parquet"))[0]
    sample_test = pq.read_table(str(first_file), columns=["holdings"]).to_pandas()
    if not sample_test.empty:
        test_map = backtest_pipeline._safe_hm(sample_test["holdings"].iloc[0])
        if not test_map:
            raise RuntimeError(
                f"First sampled row's holdings could not be parsed by _safe_hm "
                f"(in {first_file.name}). Aborting before stage 2."
            )

    # Consolidate
    consolidated = _consolidate_sample_dir()
    return consolidated if consolidated else OUT_SAMPLE_DIR


# ----------------------------------------------------------------------------
# Stage 2: re-run pipeline with rf=0
# ----------------------------------------------------------------------------

def _available_sample_blocks() -> List[int]:
    """Return the sorted list of block numbers that have a per-block sample on disk."""
    if not OUT_SAMPLE_DIR.exists():
        return []
    found = []
    for p in OUT_SAMPLE_DIR.glob("block_number=*.parquet"):
        try:
            found.append(int(p.stem.split("=", 1)[1]))
        except Exception:
            continue
    return sorted(found)


def _stage_2_already_done(blocks: List[int]) -> bool:
    if not OUT_RESULTS_DIR.exists():
        return False
    have_blocks = {
        int(p.name.split("=", 1)[1])
        for p in OUT_RESULTS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("block_number=")
    }
    return have_blocks.issuperset(set(blocks))


def stage_2_run_rf_zero(
    force: bool = False,
    blocks: Optional[List[int]] = None,
) -> Path:
    """Run the production backtest pipeline on the sample parquet with rf=0.

    Uses ``OUT_SAMPLE_PARQUET`` if it exists (consolidated single file),
    otherwise falls back to ``OUT_SAMPLE_DIR`` (per-block files). If
    ``blocks`` is None, runs only the blocks for which a sample exists on
    disk — useful when stage 1 was interrupted and you want to proceed
    with what's there instead of waiting for the rest.
    """
    print("=" * 80)
    print("STAGE 2: run pipeline with risk_free_rate_annual = 0.0")
    print("=" * 80)

    # Pick the input source: prefer single consolidated file
    if OUT_SAMPLE_PARQUET.exists():
        holdings_input = str(OUT_SAMPLE_PARQUET)
        print(f"[stage 2] reading samples from {OUT_SAMPLE_PARQUET}")
    elif OUT_SAMPLE_DIR.exists() and any(OUT_SAMPLE_DIR.glob("block_number=*.parquet")):
        holdings_input = str(OUT_SAMPLE_DIR)
        print(f"[stage 2] no consolidated parquet; reading from dir {OUT_SAMPLE_DIR}")
    else:
        raise FileNotFoundError(
            f"No sample data found. Expected one of:\n"
            f"  - {OUT_SAMPLE_PARQUET}\n"
            f"  - {OUT_SAMPLE_DIR}/block_number=*.parquet\n"
            f"Run stage_1_sample() first."
        )

    # Determine which blocks to run
    if blocks is None:
        sample_blocks = _available_sample_blocks()
        if not sample_blocks:
            raise RuntimeError("No per-block sample files found")
        blocks_to_run = sample_blocks
        if set(blocks_to_run) != set(BLOCKS):
            missing = sorted(set(BLOCKS) - set(blocks_to_run))
            print(
                f"[stage 2] running on {len(blocks_to_run)}/{len(BLOCKS)} blocks "
                f"(missing samples for {len(missing)}: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''})"
            )
        else:
            print(f"[stage 2] running on all {len(blocks_to_run)} blocks")
    else:
        blocks_to_run = list(blocks)
        print(f"[stage 2] running on {len(blocks_to_run)} caller-specified blocks")

    if _stage_2_already_done(blocks_to_run) and not force:
        print(f"[stage 2] all target blocks already present in {OUT_RESULTS_DIR}; "
              "skipping (force=True to rerun)")
        return OUT_RESULTS_DIR

    cfg = RiskConfig()
    cfg = dataclasses.replace(
        cfg,
        # Match the production cfg in 5_0_portfolio_backtest_pipeline.main()
        window_days=60,
        min_periods=45,
        clip_outliers=None,
        return_type="log",
        shrinkage_method="ledoitwolf",
        max_weight_single_asset=0.9,
        # The actual change under test:
        risk_free_rate_annual=0.0,
    )
    print(f"[cfg] {cfg}")

    backtest_pipeline.HORIZON_DAYS = HORIZON_DAYS

    if Path(SUPPLY_CSV).exists():
        print(f"[supply] loading from {SUPPLY_CSV}")
        supply_lookup = backtest_pipeline.load_supply_data(SUPPLY_CSV)
        print(f"[supply] tokens loaded: {len(supply_lookup):,}")
    else:
        print(f"[warn] supply CSV not found at {SUPPLY_CSV}; mcap weights will fall back")
        supply_lookup = {}

    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if force and OUT_RESULTS_CSV.exists():
        # Avoid mixing rows from a prior run via upsert
        OUT_RESULTS_CSV.unlink()

    t0 = time.perf_counter()
    backtest_pipeline.run_backtest_all_blocks(
        holdings_parquet=holdings_input,
        block_to_date_csv=BLOCK_TO_DATE_CSV,
        prices_dir=PRICES_DIR,
        config=cfg,
        mean_reversion_strength=MEAN_REVERSION_STR,
        blocks=blocks_to_run,
        workers=WORKERS,
        out_csv=str(OUT_RESULTS_CSV),
        out_wallet_parquet=str(OUT_RESULTS_DIR),
        regime_lookback_days=REGIME_LOOKBACK_DAYS,
        regime_top_k_assets=REGIME_TOP_K_ASSETS,
        batch_size=BATCH_SIZE,
        batch_timeout_minutes=BATCH_TIMEOUT_MIN,
        max_wallets=None,
        supply_lookup=supply_lookup,
    )
    print(f"[stage 2] pipeline finished in {(time.perf_counter() - t0) / 60.0:.1f} min")
    return OUT_RESULTS_DIR


# ----------------------------------------------------------------------------
# Stage 3: compare
# ----------------------------------------------------------------------------

def _parse_holdings_to_weights(holdings_json: Optional[str]) -> Dict[str, float]:
    """Parse a stored holdings_<strategy> JSON into ``{token: fraction}``."""
    if holdings_json is None:
        return {}
    if isinstance(holdings_json, float) and not pd.notna(holdings_json):
        return {}
    try:
        arr = json.loads(holdings_json) if isinstance(holdings_json, str) else holdings_json
    except Exception:
        return {}
    if not isinstance(arr, list):
        return {}
    out: Dict[str, float] = {}
    for item in arr:
        try:
            tok = str(item.get("token_address", "")).lower()
            pct = float(item.get("percentage", 0.0))
        except Exception:
            continue
        if tok and pct > 0:
            out[tok] = out.get(tok, 0.0) + pct / 100.0
    return out


def _l1(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    return float(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys))


def _read_block_parquet(block_dir: Path, columns: List[str]) -> pd.DataFrame:
    files = list(block_dir.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    parts: List[pd.DataFrame] = []
    for f in files:
        df = _safe_read_parquet(f, columns)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    # pandas concat tolerates dtype mismatches across files
    return pd.concat(parts, ignore_index=True, sort=False)


# Strategies whose holdings should NOT change with rf
NON_MSR_HOLDINGS = [
    "holdings_better_return",
    "holdings_safer_risk",
    "holdings_equal_weight",
    "holdings_mcap_weight",
]
# Scalar fields that should NOT change with rf
NON_MSR_SCALARS = [
    "l1_gap_better_return", "l1_gap_safer_risk",
    "beta_baseline", "beta_better_return", "beta_safer_risk",
    "beta_equal_weight", "beta_mcap_weight",
    "ret_baseline", "ret_better_return", "ret_safer_risk",
    "ret_equal_weight", "ret_mcap_weight",
]
# Fields that CAN change with rf
MSR_HOLDINGS = ["holdings_max_sharpe"]
MSR_SCALARS = [
    "l1_gap_max_sharpe", "l1_max_sharpe_vs_equal", "l1_max_sharpe_vs_mcap",
    "beta_max_sharpe", "ret_max_sharpe",
]

ALL_LOAD_COLUMNS = (
    ["wallet_address", "block_number"]
    + MSR_HOLDINGS + NON_MSR_HOLDINGS + MSR_SCALARS + NON_MSR_SCALARS
    + ["holdings_short"]  # baseline holdings, sanity
)


def stage_3_compare(blocks: Optional[List[int]] = None) -> None:
    """Inner-join old (rf=0.05) and new (rf=0.0); save per-wallet and aggregate diffs.

    If ``blocks`` is None, compare every block for which we have new (rf=0)
    results on disk.
    """
    print("=" * 80)
    print("STAGE 3: compare rf=0 vs rf=0.05")
    print("=" * 80)

    OUT_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    if blocks is None:
        if OUT_RESULTS_DIR.exists():
            available = sorted(
                int(p.name.split("=", 1)[1])
                for p in OUT_RESULTS_DIR.iterdir()
                if p.is_dir() and p.name.startswith("block_number=")
            )
        else:
            available = []
        blocks_to_compare = available if available else list(BLOCKS)
        if available and set(available) != set(BLOCKS):
            missing = sorted(set(BLOCKS) - set(available))
            print(
                f"[stage 3] comparing {len(available)}/{len(BLOCKS)} blocks "
                f"(no results for {len(missing)})"
            )
    else:
        blocks_to_compare = list(blocks)

    per_wallet: List[pd.DataFrame] = []
    per_block: List[Dict[str, Any]] = []

    for blk in blocks_to_compare:
        new_dir = OUT_RESULTS_DIR / f"block_number={blk}"
        old_dir = Path(DATA_EXTENDED_DIR) / f"block_number={blk}"

        df_new = _read_block_parquet(new_dir, ALL_LOAD_COLUMNS)
        df_old = _read_block_parquet(old_dir, ALL_LOAD_COLUMNS)

        if df_new.empty or df_old.empty:
            print(f"[block {blk}] missing: new_empty={df_new.empty}, old_empty={df_old.empty}")
            continue

        df_new["wallet_address"] = df_new["wallet_address"].astype(str).str.lower()
        df_old["wallet_address"] = df_old["wallet_address"].astype(str).str.lower()

        merged = df_new.merge(
            df_old, on="wallet_address", how="inner", suffixes=("_rf0", "_rf5"),
        )
        if merged.empty:
            print(f"[block {blk}] no wallets in common")
            continue

        # MSR weight L1 (the headline metric)
        w0 = merged["holdings_max_sharpe_rf0"].apply(_parse_holdings_to_weights)
        w5 = merged["holdings_max_sharpe_rf5"].apply(_parse_holdings_to_weights)
        merged["weight_l1_max_sharpe"] = [_l1(a, b) for a, b in zip(w0, w5)]

        # MSR scalar diffs (rf0 - rf5)
        for col in MSR_SCALARS:
            c_new, c_old = f"{col}_rf0", f"{col}_rf5"
            if c_new in merged.columns and c_old in merged.columns:
                merged[f"diff_{col}"] = merged[c_new] - merged[c_old]

        # Sanity: weight L1 for non-MSR strategies (should be ~0 — determinism check)
        for col in NON_MSR_HOLDINGS:
            c_new, c_old = f"{col}_rf0", f"{col}_rf5"
            if c_new in merged.columns and c_old in merged.columns:
                wn = merged[c_new].apply(_parse_holdings_to_weights)
                wo = merged[c_old].apply(_parse_holdings_to_weights)
                merged[f"sanity_l1_{col}"] = [_l1(a, b) for a, b in zip(wn, wo)]
        # Sanity: baseline (holdings_short) should be 100% identical
        if "holdings_short_rf0" in merged.columns and "holdings_short_rf5" in merged.columns:
            wn = merged["holdings_short_rf0"].apply(_parse_holdings_to_weights)
            wo = merged["holdings_short_rf5"].apply(_parse_holdings_to_weights)
            merged["sanity_l1_holdings_short"] = [_l1(a, b) for a, b in zip(wn, wo)]

        # Sanity: scalar diffs (should be ~0)
        for col in NON_MSR_SCALARS:
            c_new, c_old = f"{col}_rf0", f"{col}_rf5"
            if c_new in merged.columns and c_old in merged.columns:
                merged[f"sanity_diff_{col}"] = merged[c_new] - merged[c_old]

        merged["block_number"] = int(blk)

        # Slim per-wallet output
        keep = ["block_number", "wallet_address", "weight_l1_max_sharpe"]
        keep += [f"diff_{c}" for c in MSR_SCALARS if f"diff_{c}" in merged.columns]
        keep += [f"sanity_l1_{c}" for c in NON_MSR_HOLDINGS if f"sanity_l1_{c}" in merged.columns]
        if "sanity_l1_holdings_short" in merged.columns:
            keep.append("sanity_l1_holdings_short")
        keep += [f"sanity_diff_{c}" for c in NON_MSR_SCALARS if f"sanity_diff_{c}" in merged.columns]
        per_wallet.append(merged[keep].copy())

        # Per-block aggregates
        agg: Dict[str, Any] = {"block_number": int(blk), "n_compared": int(len(merged))}
        for c in ["weight_l1_max_sharpe"] + [f"diff_{x}" for x in MSR_SCALARS if f"diff_{x}" in merged.columns]:
            s = merged[c].dropna()
            if len(s):
                agg[f"{c}_median"]  = float(s.median())
                agg[f"{c}_mean"]    = float(s.mean())
                agg[f"{c}_p95_abs"] = float(np.quantile(np.abs(s), 0.95))
                agg[f"{c}_max_abs"] = float(np.abs(s).max())
        agg["share_weight_l1_gt_0p01"] = float((merged["weight_l1_max_sharpe"] > 0.01).mean())
        agg["share_weight_l1_gt_0p05"] = float((merged["weight_l1_max_sharpe"] > 0.05).mean())
        agg["share_weight_l1_gt_0p10"] = float((merged["weight_l1_max_sharpe"] > 0.10).mean())

        # max sanity across all sanity cols (should be tiny)
        sanity_cols = [c for c in merged.columns if c.startswith("sanity_")]
        if sanity_cols:
            mx = 0.0
            for c in sanity_cols:
                v = merged[c].abs().dropna()
                if len(v):
                    mx = max(mx, float(v.max()))
            agg["max_sanity_abs"] = mx

        per_block.append(agg)
        med = agg.get("weight_l1_max_sharpe_median", float("nan"))
        san = agg.get("max_sanity_abs", float("nan"))
        print(f"[block {blk}] n={len(merged):,}  weight_l1 median={med:.4g}  max_sanity={san:.4g}")

    if not per_wallet:
        print("[stage 3] no comparable wallets; aborting")
        return

    df_per_wallet = pd.concat(per_wallet, ignore_index=True)
    df_per_block  = pd.DataFrame(per_block)

    pq.write_table(
        pa.Table.from_pandas(df_per_wallet, preserve_index=False),
        str(OUT_COMPARISON_PARQUET), compression="zstd",
    )
    df_per_block.to_csv(OUT_PER_BLOCK_SUMMARY_CSV, index=False)

    # Overall summary
    overall: Dict[str, Any] = {"n_total": int(len(df_per_wallet))}
    s = df_per_wallet["weight_l1_max_sharpe"].dropna()
    if len(s):
        overall["weight_l1_max_sharpe_median"] = float(s.median())
        overall["weight_l1_max_sharpe_mean"]   = float(s.mean())
        overall["weight_l1_max_sharpe_p95"]    = float(np.quantile(s, 0.95))
        overall["weight_l1_max_sharpe_p99"]    = float(np.quantile(s, 0.99))
        overall["weight_l1_max_sharpe_max"]    = float(s.max())
    for col in MSR_SCALARS:
        c = f"diff_{col}"
        if c in df_per_wallet.columns:
            sc = df_per_wallet[c].dropna()
            if len(sc):
                overall[f"{c}_median"]  = float(sc.median())
                overall[f"{c}_mean"]    = float(sc.mean())
                overall[f"{c}_p95_abs"] = float(np.quantile(np.abs(sc), 0.95))
    overall["share_weight_l1_gt_0p01"] = float((df_per_wallet["weight_l1_max_sharpe"] > 0.01).mean())
    overall["share_weight_l1_gt_0p05"] = float((df_per_wallet["weight_l1_max_sharpe"] > 0.05).mean())
    overall["share_weight_l1_gt_0p10"] = float((df_per_wallet["weight_l1_max_sharpe"] > 0.10).mean())
    sanity_cols = [c for c in df_per_wallet.columns if c.startswith("sanity_")]
    overall["max_sanity_abs"] = float(
        max((df_per_wallet[c].abs().max() for c in sanity_cols), default=0.0)
    )

    pd.DataFrame([overall]).to_csv(OUT_OVERALL_SUMMARY_CSV, index=False)

    print()
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k:48s}: {v:.6g}")
        else:
            print(f"  {k:48s}: {v}")
    print()
    print(f"per-wallet diffs : {OUT_COMPARISON_PARQUET}")
    print(f"per-block summary: {OUT_PER_BLOCK_SUMMARY_CSV}")
    print(f"overall summary  : {OUT_OVERALL_SUMMARY_CSV}")
    print()
    print("interpretation hints")
    print("--------------------")
    print("  - max_sanity_abs ≈ 0  → pipeline determinism confirmed; only MSR fields differ.")
    print("  - weight_l1_max_sharpe is on [0, 2] (sum of |Δw_i|), so e.g. 0.05 ≈ 2.5% reallocation.")
    print("  - diff_ret_max_sharpe is in return units; multiply by 100 for percentage points.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(force: bool = False) -> int:
    t0 = time.perf_counter()

    stage_1_sample(force=force)
    stage_2_run_rf_zero(force=force)
    stage_3_compare()

    print(f"\n[total] runtime = {(time.perf_counter() - t0) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Test impact of risk_free_rate_annual=0 vs 0.05 on MPT backtest."
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-run stages even if their outputs already exist.",
    )
    args = ap.parse_args()
    raise SystemExit(main(force=args.force))