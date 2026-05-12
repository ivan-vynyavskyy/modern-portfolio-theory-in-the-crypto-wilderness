#!/usr/bin/env python3
"""
rf_shap_analysis_extended.py

Description:
    Extended RF + SHAP analysis with 6 features (month, log_value_usd,
    num_tokens, beta_baseline, holds_eth, holds_btc). Trains RF on 10M
    samples, compares 3-feature vs 6-feature models, and exports SHAP
    values. Designed for ~239M rows, 128 GB RAM, 32 cores.

Input:
    Hive-partitioned parquet (ROOT) with per-wallet returns, betas,
    and token holding columns.

Output:
    rf_shap_extended_importance.csv, rf_shap_extended_values_{strategy}.csv,
    rf_shap_extended_report.txt in OUTPUT_DIR.
"""

import os, sys, time, warnings
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from locations import Location

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
ROOT = str(Location.MPT_DATA_EXTENDED)
OUTPUT_DIR = str(Location.MPT_RF_SHAP_RESULTS)
N_CORES = 32
SEED = 42

RF_SAMPLE_SIZE = 10_000_000
SHAP_SAMPLE_SIZE = 50_000

# Original 3 features + 3 new ones
FEATURES = ["month", "log_value_usd", "num_tokens", "beta_baseline", "holds_eth", "holds_btc"]
FEATURE_NAMES_DISPLAY = [
    "Entry month", "Log portfolio value", "Token count",
    "Portfolio beta", "Holds ETH wrapper", "Holds BTC wrapper",
]

RETURN_COLS = [
    "ret_baseline", "ret_better_return", "ret_safer_risk",
    "ret_max_sharpe", "ret_equal_weight", "ret_mcap_weight",
]
RETURN_LABELS = [
    "baseline", "better_return", "safer_risk",
    "max_sharpe", "equal_weight", "mcap_weight",
]

BETA_COLS = [
    "beta_baseline", "beta_better_return", "beta_safer_risk",
    "beta_max_sharpe", "beta_equal_weight", "beta_mcap_weight",
]

# -----------------------------------------------------------------------------
# ETH and BTC wrapper token addresses (lowercase)
# -----------------------------------------------------------------------------
ETH_WRAPPER_ADDRESSES = [
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
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

BTC_WRAPPER_ADDRESSES = [
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


def _sql_holds_any(column, addresses):
    """Build SQL CASE that returns 1 if column contains any of the addresses."""
    conditions = " OR ".join(
        f"LOWER({column}) LIKE '%{addr}%'" for addr in addresses
    )
    return f"CASE WHEN ({conditions}) THEN 1 ELSE 0 END"


os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_PATH = os.path.join(OUTPUT_DIR, "rf_shap_report.txt")
log_file = open(LOG_PATH, "w", buffering=1)


def log(msg=""):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if msg else ""
    print(line, flush=True)
    log_file.write(line + "\n")


def log_df(df, title=""):
    if title:
        log(title)
    buf = df.to_string()
    print(buf, flush=True)
    log_file.write(buf + "\n")
    log("")


def save_csv(df, name):
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=True)
    log(f"  → saved {path}")


# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
log("=" * 70)
log("RF + SHAP ANALYSIS — RETURN DRIVERS (EXTENDED FEATURES)")
log("=" * 70)
log()

t0 = time.time()
con = duckdb.connect(database=":memory:")
con.execute(f"SET threads = {N_CORES}")
con.execute("SET memory_limit = '100GB'")

con.execute(f"""
    CREATE VIEW all_data AS
    SELECT *
    FROM read_parquet('{ROOT}/*/*.parquet', hive_partitioning=true)
""")

total_rows = con.sql("SELECT COUNT(*) AS n FROM all_data").fetchone()[0]
log(f"Total rows: {total_rows:,}")
log()

# -----------------------------------------------------------------------------
# 2. DRAW SAMPLE — compute holds_eth / holds_btc in SQL
# -----------------------------------------------------------------------------
log("=" * 70)
log(f"Drawing {RF_SAMPLE_SIZE/1e6:.0f}M sample for RF + SHAP")
log("=" * 70)

t1 = time.time()

holds_eth_sql = _sql_holds_any("holdings_short", ETH_WRAPPER_ADDRESSES)
holds_btc_sql = _sql_holds_any("holdings_short", BTC_WRAPPER_ADDRESSES)

base_cols = ["date", "total_value_usd", "num_tokens", "market_return", "beta_baseline"]
ret_and_beta = RETURN_COLS + BETA_COLS

select_expr = ", ".join(base_cols + ret_and_beta) + f""",
    {holds_eth_sql} AS holds_eth,
    {holds_btc_sql} AS holds_btc
"""

pct = RF_SAMPLE_SIZE / total_rows * 100
sample = con.sql(f"""
    SELECT {select_expr}
    FROM all_data
    TABLESAMPLE BERNOULLI({pct:.4f} PERCENT)
""").df()

log(f"  Raw sample: {len(sample):,} rows in {time.time()-t1:.1f}s")
log(f"  holds_eth=1: {sample['holds_eth'].sum():,}  "
    f"holds_btc=1: {sample['holds_btc'].sum():,}")

# Derive features
sample["date_dt"] = (
    pd.to_datetime(sample["date"], unit="ms")
    if sample["date"].dtype in ["int64", "float64"]
    else pd.to_datetime(sample["date"])
)
sample["month"] = (sample["date_dt"].dt.year - 2020) * 12 + sample["date_dt"].dt.month
sample["log_value_usd"] = np.log1p(sample["total_value_usd"])

# Derive returns in percent and alpha
RET_PCT_COLS = []
ALPHA_COLS = []
for ret_col, beta_col, label in zip(RETURN_COLS, BETA_COLS, RETURN_LABELS):
    pct_col = f"ret_pct_{label}"
    alpha_col = f"alpha_{label}"
    sample[pct_col] = sample[ret_col] * 100.0
    sample[alpha_col] = sample[ret_col] - sample[beta_col] * sample["market_return"]
    RET_PCT_COLS.append(pct_col)
    ALPHA_COLS.append(alpha_col)

log(f"  Sample ready: {len(sample):,} rows")
log(f"  Feature summary:")
for feat in FEATURES:
    vals = sample[feat].dropna()
    log(f"    {feat:20s}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
        f"min={vals.min():.4f}  max={vals.max():.4f}")
log()

# Winsorise at p5/p95
log("Winsorising returns/alpha at p5/p95 ...")
for col in RET_PCT_COLS + ALPHA_COLS:
    vals = sample[col].dropna()
    lo, hi = np.percentile(vals, [5, 95])
    sample[col] = sample[col].clip(lower=lo, upper=hi)
log(f"  Done. Rows unchanged: {len(sample):,}")
log()

# -----------------------------------------------------------------------------
# 3. RANDOM FOREST — 6 features
# -----------------------------------------------------------------------------
log("=" * 70)
log(f"RANDOM FOREST — {len(sample):,} rows, {len(FEATURES)} features, {N_CORES} cores")
log(f"Features: {FEATURES}")
log("=" * 70)

trained_models = {}
rf_results = []

# Train on both returns AND alpha
targets = []
for label in RETURN_LABELS:
    targets.append((f"ret_pct_{label}", f"ret_{label}"))
for label in RETURN_LABELS:
    targets.append((f"alpha_{label}", f"alpha_{label}"))

for tgt_col, tgt_name in targets:
    log(f"\n  Training RF for {tgt_name} ...")
    t1 = time.time()

    X = sample[FEATURES].values
    y = sample[tgt_col].values
    mask = ~(np.isnan(y) | np.isinf(y)) & ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=100,
        n_jobs=N_CORES,
        random_state=SEED,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp_df["target"] = tgt_name
    imp_df["R2_test"] = round(r2, 4)
    imp_df["MAE_test"] = round(mae, 2)

    log(f"  RF {tgt_name}: R²={r2:.4f}, MAE={mae:.2f}  ({time.time()-t1:.1f}s)")
    log_df(imp_df.round(4), f"  Feature importances ({tgt_name}):")

    rf_results.append(imp_df)
    trained_models[tgt_name] = (rf, X, y)

rf_all = pd.concat(rf_results)
save_csv(rf_all, "rf_shap_importance.csv")

# -----------------------------------------------------------------------------
# Compare 3-feature vs 6-feature RF
log()
log("=" * 70)
log("COMPARISON: 3-feature vs 6-feature RF")
log("=" * 70)

FEATURES_3 = ["month", "log_value_usd", "num_tokens"]
rf_results_3 = []

for tgt_col, tgt_name in targets:
    log(f"\n  Training 3-feature RF for {tgt_name} ...")
    t1 = time.time()

    X3 = sample[FEATURES_3].values
    y3 = sample[tgt_col].values
    mask3 = ~(np.isnan(y3) | np.isinf(y3)) & ~np.isnan(X3).any(axis=1)
    X3, y3 = X3[mask3], y3[mask3]

    X3_train, X3_test, y3_train, y3_test = train_test_split(
        X3, y3, test_size=0.2, random_state=SEED
    )

    rf3 = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=100,
        n_jobs=N_CORES,
        random_state=SEED,
    )
    rf3.fit(X3_train, y3_train)

    y3_pred = rf3.predict(X3_test)
    r2_3 = r2_score(y3_test, y3_pred)
    mae_3 = mean_absolute_error(y3_test, y3_pred)

    imp3_df = pd.DataFrame({
        "feature": FEATURES_3,
        "importance": rf3.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp3_df["target"] = tgt_name
    imp3_df["R2_test"] = round(r2_3, 4)
    imp3_df["MAE_test"] = round(mae_3, 2)

    log(f"  RF-3feat {tgt_name}: R²={r2_3:.4f}, MAE={mae_3:.2f}  ({time.time()-t1:.1f}s)")
    rf_results_3.append(imp3_df)

rf_3_all = pd.concat(rf_results_3)
save_csv(rf_3_all, "rf_3feat_importance.csv")

# Side-by-side comparison
log("\n  R² comparison (3-feature vs 6-feature):")
log(f"  {'Target':25s}  {'R²(3-feat)':>12s}  {'R²(6-feat)':>12s}  {'ΔR²':>8s}")
for tgt_col, tgt_name in targets:
    r2_3 = rf_3_all[rf_3_all["target"] == tgt_name]["R2_test"].iloc[0]
    r2_6 = rf_all[rf_all["target"] == tgt_name]["R2_test"].iloc[0]
    delta = r2_6 - r2_3
    log(f"  {tgt_name:25s}  {r2_3:12.4f}  {r2_6:12.4f}  {delta:+8.4f}")

# -----------------------------------------------------------------------------
# 4. SHAP VALUES
# -----------------------------------------------------------------------------
log()
log("=" * 70)
log(f"SHAP ANALYSIS — {SHAP_SAMPLE_SIZE/1e3:.0f}K subsample per model")
log("=" * 70)

try:
    import shap
    shap_available = True
    log("shap library loaded successfully")
except ImportError:
    shap_available = False
    log("WARNING: shap not installed. Run: pip install shap --break-system-packages")
    log("Skipping SHAP analysis.")

if shap_available:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # SHAP on 6-feature return models (not alpha — similar patterns expected)
    shap_targets = [(f"ret_{label}", label) for label in RETURN_LABELS]

    for tgt_name, label in shap_targets:
        if tgt_name not in trained_models:
            log(f"  Skipping {tgt_name} — model not found")
            continue

        rf, X_full, y_full = trained_models[tgt_name]
        log(f"\n  Computing SHAP for {tgt_name} ...")
        t1 = time.time()

        # Subsample for SHAP
        n = min(SHAP_SAMPLE_SIZE, len(X_full))
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(X_full), size=n, replace=False)
        X_shap = X_full[idx]

        # TreeExplainer — exact for RF
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_shap)

        log(f"  SHAP computed: {n:,} samples in {time.time()-t1:.1f}s")

        # Save raw SHAP values
        shap_df = pd.DataFrame(shap_values, columns=FEATURES)
        shap_df["expected_value"] = explainer.expected_value
        for i, feat in enumerate(FEATURES):
            shap_df[f"feat_{feat}"] = X_shap[:, i]
        save_csv(shap_df, f"rf_shap_values_{label}.csv")

        # Mean |SHAP|
        mean_abs_shap = pd.DataFrame({
            "feature": FEATURES,
            "feature_display": FEATURE_NAMES_DISPLAY,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "target": tgt_name,
        }).sort_values("mean_abs_shap", ascending=False)
        log_df(mean_abs_shap.round(4), f"  Mean |SHAP| ({tgt_name}):")

        # SHAP summary (beeswarm)
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.summary_plot(
            shap_values, X_shap,
            feature_names=FEATURE_NAMES_DISPLAY,
            show=False, plot_size=None,
        )
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"rf_shap_summary_{label}.pdf")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        log(f"  → saved {path}")

        # SHAP dependence: month
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            0, shap_values, X_shap,
            feature_names=FEATURE_NAMES_DISPLAY,
            interaction_index=None,
            show=False, ax=ax,
        )
        ax.set_xlabel("Entry month (months since Jan 2020)")
        ax.set_ylabel("SHAP value (impact on predicted return %)")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"rf_shap_month_dep_{label}.pdf")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        log(f"  → saved {path}")

        # SHAP dependence: beta_baseline
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            3, shap_values, X_shap,  # index 3 = beta_baseline
            feature_names=FEATURE_NAMES_DISPLAY,
            interaction_index=None,
            show=False, ax=ax,
        )
        ax.set_xlabel("Portfolio beta (baseline)")
        ax.set_ylabel("SHAP value (impact on predicted return %)")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"rf_shap_beta_dep_{label}.pdf")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        log(f"  → saved {path}")

        # SHAP dependence: log_value_usd
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            1, shap_values, X_shap,
            feature_names=FEATURE_NAMES_DISPLAY,
            interaction_index=None,
            show=False, ax=ax,
        )
        ax.set_xlabel("Log portfolio value (USD)")
        ax.set_ylabel("SHAP value (impact on predicted return %)")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"rf_shap_value_dep_{label}.pdf")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        log(f"  → saved {path}")

# -----------------------------------------------------------------------------
# 5. MONTH MAPPING
# -----------------------------------------------------------------------------
log()
log("=" * 70)
log("MONTH MAPPING")
log("=" * 70)

month_map = []
for m in sorted(sample["month"].dropna().unique()):
    m = int(m)
    year = 2020 + (m - 1) // 12
    mo = ((m - 1) % 12) + 1
    month_map.append({"month_int": m, "date": f"{year}-{mo:02d}"})

month_map_df = pd.DataFrame(month_map)
save_csv(month_map_df, "month_mapping.csv")
log_df(month_map_df, "Month integer → calendar date:")

# -----------------------------------------------------------------------------
# 6. SUMMARY
# -----------------------------------------------------------------------------
elapsed = time.time() - t0
log()
log("=" * 70)
log(f"RF + SHAP ANALYSIS COMPLETE — {elapsed/60:.1f} minutes")
log(f"All results saved to: {OUTPUT_DIR}")
log("=" * 70)

log()
log("Output files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    log(f"  {f:60s} {size:>10,} bytes")

log_file.close()
con.close()
