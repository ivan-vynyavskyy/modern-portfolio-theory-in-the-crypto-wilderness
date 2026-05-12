#!/usr/bin/env python3
"""
rf_shap_analysis.py

Description:
    Random Forest + SHAP analysis of return drivers. Trains RF on 10M
    samples, computes SHAP values on a 50K subsample, and exports
    per-strategy feature importances and SHAP value CSVs. Designed for
    ~239M rows, 128 GB RAM, 32 cores.

Input:
    Hive-partitioned parquet (ROOT) with per-wallet returns, month,
    log_value_usd, num_tokens columns.

Output:
    rf_shap_importance.csv, rf_shap_values_{strategy}.csv,
    rf_shap_report.txt in OUTPUT_DIR.
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

RF_SAMPLE_SIZE = 10_000_000       # 10M for RF training
SHAP_SAMPLE_SIZE = 50_000         # 50K for SHAP (TreeExplainer is O(n × trees × depth))

FEATURES = ["month", "log_value_usd", "num_tokens"]

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
log("RF + SHAP ANALYSIS — RETURN DRIVERS")
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
# 2. DRAW SAMPLE
# -----------------------------------------------------------------------------
log("=" * 70)
log(f"Drawing {RF_SAMPLE_SIZE/1e6:.0f}M sample for RF + SHAP")
log("=" * 70)

t1 = time.time()

sample_cols = (
    ["date", "total_value_usd", "num_tokens", "market_return"]
    + RETURN_COLS
    + BETA_COLS
)

pct = RF_SAMPLE_SIZE / total_rows * 100
sample = con.sql(f"""
    SELECT {', '.join(sample_cols)}
    FROM all_data
    TABLESAMPLE BERNOULLI({pct:.4f} PERCENT)
""").df()

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

log(f"Sample loaded: {len(sample):,} rows in {time.time()-t1:.1f}s")

# Winsorise at p5/p95
log("Winsorising at p5/p95 ...")
for col in RET_PCT_COLS + ALPHA_COLS:
    vals = sample[col].dropna()
    lo, hi = np.percentile(vals, [5, 95])
    sample[col] = sample[col].clip(lower=lo, upper=hi)
    log(f"  {col}: clipped to [{lo:.4f}, {hi:.4f}]")
log()

# -----------------------------------------------------------------------------
# 3. RANDOM FOREST (10M)
# -----------------------------------------------------------------------------
log("=" * 70)
log(f"RANDOM FOREST — {RF_SAMPLE_SIZE/1e6:.0f}M rows, {N_CORES} cores")
log("=" * 70)

# We'll store trained models for SHAP
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
        n_estimators=300,        # more trees than before (was 200)
        max_depth=18,            # slightly deeper (was 15)
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

    # We focus SHAP on the 6 return targets (not alpha — similar story)
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

        # TreeExplainer — efficient for RF
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_shap)

        log(f"  SHAP computed for {tgt_name}: {n:,} samples in {time.time()-t1:.1f}s")

        # Save raw SHAP values
        shap_df = pd.DataFrame(shap_values, columns=FEATURES)
        shap_df["expected_value"] = explainer.expected_value
        for i, feat in enumerate(FEATURES):
            shap_df[f"feat_{feat}"] = X_shap[:, i]
        save_csv(shap_df, f"rf_shap_values_{label}.csv")

        # SHAP summary (beeswarm)
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.summary_plot(
            shap_values, X_shap,
            feature_names=["Entry month", "Log portfolio value", "Token count"],
            show=False, plot_size=None
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
            feature_names=["Entry month", "Log portfolio value", "Token count"],
            interaction_index=None,
            show=False, ax=ax
        )
        ax.set_xlabel("Entry month (months since Jan 2020)")
        ax.set_ylabel(f"SHAP value (impact on predicted return %)")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"rf_shap_month_dep_{label}.pdf")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        log(f"  → saved {path}")

        # SHAP dependence: log_value_usd
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            1, shap_values, X_shap,
            feature_names=["Entry month", "Log portfolio value", "Token count"],
            interaction_index=None,
            show=False, ax=ax
        )
        ax.set_xlabel("Log portfolio value (USD)")
        ax.set_ylabel(f"SHAP value (impact on predicted return %)")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"rf_shap_value_dep_{label}.pdf")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        log(f"  → saved {path}")

        # Mean absolute SHAP by feature
        mean_abs_shap = pd.DataFrame({
            "feature": FEATURES,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "target": tgt_name,
        })
        log_df(mean_abs_shap.round(4), f"  Mean |SHAP| ({tgt_name}):")

# -----------------------------------------------------------------------------
# 5. MONTH DECODING — map month integer back to dates for interpretability
# -----------------------------------------------------------------------------
log()
log("=" * 70)
log("MONTH MAPPING — for interpreting SHAP month dependence plots")
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

# List output files
log()
log("Output files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    log(f"  {f:60s} {size:>10,} bytes")

log_file.close()
con.close()