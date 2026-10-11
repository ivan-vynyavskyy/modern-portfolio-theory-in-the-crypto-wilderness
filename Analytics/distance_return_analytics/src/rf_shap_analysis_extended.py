#!/usr/bin/env python3
"""
rf_shap_analysis_extended.py

Description:
    Random Forest + SHAP analysis of what drives realised 20-day returns and
    alpha. Four model variants separate the common monthly market move from
    what can actually be predicted:

      (a) random 80/20 split, snapshot-month index + 5 account features:
          the paper's original design, incl. the 3-feature comparison and SHAP.
      (b) the same 6 features, folds grouped by month: a month index cannot
          generalise to unseen months, so the drop from (a) is the share of the
          common monthly move.
      (c) month index replaced by market-state variables known before the
          snapshot (Fear & Greed, momentum, volatility, drawdown, account
          growth, previous month's median alpha), folds grouped by month, plus
          a forward-chaining test that trains on earlier years only. Leaves
          hold at least six months of rows, so the forest cannot memorise
          single months; with only 72 months it is still a lower bound, and
          the month-level time-series test reads
          month_outcomes.csv and market_state.csv.
      (d) account features plus the three frontier distances, target demeaned
          by month: does allocation predict who does better within a month?

    Sample: the same number of accounts per snapshot (--per-month), chosen by
    a seeded hash so reruns are identical. Targets are winsorised within each
    month (the old global p5/p95 clipped whole extreme months). Importances:
    impurity (as before) and permutation importance on held-out rows. Also a
    variance decomposition of returns into between- and within-month parts,
    over all rows and over the sample, and the four variance shares that
    combine it with variants (c) and (d).

Input:
    --root          5_0 wallet-level parquet (block_number=*/)
    --prices-dir    per-token price CSVs (date, daily_close_usd) incl. WETH, WBTC
    --fng-csv       fear_greed_index.csv (fetch_fear_greed_index.py)
    --accounts-csv  output_accounts_eoa_vs_ca.csv (accounts per snapshot)

Output (in --out-dir):
    Variant (a), formats unchanged: rf_shap_importance.csv (6 features),
    rf_3feat_importance.csv, rf_shap_values_baseline.csv and SHAP PDFs
    (when shap is installed), month_mapping.csv. Both importance CSVs gain
    perm_importance and perm_share columns.
    New: rf_variants_summary.csv, rf_variants_importance.csv,
    rf_forward_chaining.csv, variance_shares.csv, month_outcomes.csv,
    market_state.csv. Log: rf_shap_report.txt.

Usage:
    python rf_shap_analysis_extended.py \
        --root         <mpt-optimization-results-eth>/wallet_level \
        --out-dir      <mpt-optimization-results-eth>/rf_shap_results \
        --temp-dir     <duckdb_tmp> \
        --prices-dir   <DATA>/prices \
        --fng-csv      fear_greed_index.csv \
        --accounts-csv output_accounts_eoa_vs_ca.csv \
        --threads 32 --mem 100GB
"""

import argparse, glob, os, sys, time, warnings
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Config (all paths come from the command line, no defaults)
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="RF + SHAP analysis of return drivers, four variants."
    )
    p.add_argument("--root", type=str, required=True,
                   help="Hive-partitioned wallet-level parquet directory (block_number=*/).")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Output directory for CSVs, PDFs and rf_shap_report.txt.")
    p.add_argument("--temp-dir", type=str, required=True,
                   help="DuckDB spill directory (keep it off the home fileset).")
    p.add_argument("--prices-dir", type=str, required=True,
                   help="Directory with <token>_price.csv files incl. WETH and WBTC.")
    p.add_argument("--fng-csv", type=str, required=True,
                   help="fear_greed_index.csv from fetch_fear_greed_index.py.")
    p.add_argument("--accounts-csv", type=str, required=True,
                   help="output_accounts_eoa_vs_ca.csv (accounts per snapshot).")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--mem", type=str, default="100GB",
                   help="DuckDB memory_limit.")
    p.add_argument("--per-month", type=int, default=50_000,
                   help="Accounts sampled per snapshot.")
    p.add_argument("--folds", type=int, default=6,
                   help="Month-grouped folds for variants (b)-(d).")
    p.add_argument("--winsor", type=float, default=0.05,
                   help="Within-month winsorising quantile (0.05 = p5/p95).")
    p.add_argument("--trees", type=int, default=300,
                   help="Trees per forest (300 as in the paper's original RF).")
    return p.parse_args()


args = parse_args()
for path in (args.root, args.prices_dir):
    if not os.path.isdir(path):
        sys.exit(f"not a directory: {path}")
for path in (args.fng_csv, args.accounts_csv):
    if not os.path.isfile(path):
        sys.exit(f"not a file: {path}")
if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) < (1, 4):
    sys.exit(f"scikit-learn {sklearn.__version__}: need >= 1.4 (missing values in RandomForest)")

ROOT = args.root
OUTPUT_DIR = args.out_dir
TEMP_DIR = args.temp_dir
PRICES_DIR = args.prices_dir
N_CORES = args.threads
MEM_LIMIT = args.mem
PER_MONTH = args.per_month
FOLDS = args.folds
WINSOR = args.winsor
N_TREES = args.trees
SEED = 42

MAX_DEPTH, MIN_LEAF = 18, 100     # as in the paper's original RF
# State variables are constant within a month, so with MIN_LEAF the forest
# isolates single months and memorises the 60 training months instead of
# learning how the state relates to returns. On synthetic data with a true
# Fear & Greed effect, leaves of 1 or 3 months still gave a negative
# between-month R²; 6 months recovered the effect. Used for variant (c).
STATE_LEAF = 6 * PER_MONTH
# Exact tree SHAP is single-threaded and slow on deep 300-tree forests (old
# run: ~3 h per model for 30K rows), so only the baseline return model, which
# is the one the paper and both SHAP figures use.
SHAP_SAMPLE_SIZE = 10_000
PERM_SAMPLE = 100_000             # held-out rows used for permutation importance
PERM_REPEATS = 3
FORWARD_TEST_YEARS = [2022, 2023, 2024, 2025]   # expanding window, two-year start

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"

ACCOUNT_FEATURES = ["log_value_usd", "num_tokens", "beta_baseline", "holds_eth", "holds_btc"]
FEATURES = ["month"] + ACCOUNT_FEATURES          # (a), (b): the paper's 6 features
FEATURES_3 = ["month", "log_value_usd", "num_tokens"]
STATE_FEATURES = ["fng", "fng_extreme_fear", "fng_extreme_greed", "mom_30", "mom_90",
                  "vol_30", "dd_365", "acct_growth", "prev_median_alpha"]
DIST_FEATURES = ["l1_better_return", "l1_safer_risk", "l1_max_sharpe"]
FEATURE_NAMES_DISPLAY = [
    "Snapshot month", "Log portfolio value", "Asset count",
    "Portfolio beta", "Holds ETH", "Holds BTC wrapper",
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
L1_COLS = ["l1_gap_better_return", "l1_gap_safer_risk", "l1_gap_max_sharpe"]

# -----------------------------------------------------------------------------
# ETH and BTC token addresses (lowercase)
# -----------------------------------------------------------------------------
ETH_WRAPPER_ADDRESSES = [
    "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",  # native ETH (5_0 merges WETH into it)
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


def _sql_clip(expr, lo, hi):
    """Winsorise an SQL expression; NULL stays NULL (GREATEST skips NULLs)."""
    return f"CASE WHEN ({expr}) IS NULL THEN NULL ELSE LEAST(GREATEST({expr}, {lo}), {hi}) END"


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

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


def make_rf(min_leaf=MIN_LEAF):
    return RandomForestRegressor(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=min_leaf,
        n_jobs=N_CORES,
        random_state=SEED,
    )


def perm_imp(model, X, y):
    """Permutation importance (mean drop in R²) on up to PERM_SAMPLE held-out rows."""
    if len(X) > PERM_SAMPLE:
        idx = np.random.RandomState(SEED).choice(len(X), size=PERM_SAMPLE, replace=False)
        X, y = X[idx], y[idx]
    r = permutation_importance(model, X, y, scoring="r2", n_repeats=PERM_REPEATS,
                               random_state=SEED)
    return r.importances_mean


def perm_share(perm):
    """Permutation importances as shares of their positive total."""
    pos = np.clip(perm, 0, None)
    return pos / pos.sum() if pos.sum() > 0 else pos


def decompose(n, mean, var):
    """Between-month share of the variance, rows weighted and months weighted equally."""
    n, mean, var = (np.asarray(v, dtype=float) for v in (n, mean, var))
    grand = np.sum(n * mean) / n.sum()
    between = np.sum(n * (mean - grand) ** 2)
    pooled = between / (between + np.sum(n * var))
    equal = np.var(mean) / (np.var(mean) + np.mean(var))
    return pooled, equal


# -----------------------------------------------------------------------------
# 1. SETUP
# -----------------------------------------------------------------------------
log("=" * 70)
log("RF + SHAP ANALYSIS — RETURN DRIVERS, FOUR VARIANTS")
log("=" * 70)
log(f"Input:    {ROOT}")
log(f"Output:   {OUTPUT_DIR}")
log(f"Temp dir: {TEMP_DIR}  (threads={N_CORES}, memory_limit={MEM_LIMIT})")
log(f"Sample {PER_MONTH:,} accounts per snapshot, {FOLDS} month-grouped folds, "
    f"winsorised at p{WINSOR*100:g}/p{100-WINSOR*100:g} within each month, "
    f"{N_TREES} trees, scikit-learn {sklearn.__version__}")
log()

t0 = time.time()
con = duckdb.connect(database=":memory:")
con.execute(f"SET threads = {N_CORES}")
con.execute(f"SET memory_limit = '{MEM_LIMIT}'")
con.execute(f"SET temp_directory = '{TEMP_DIR}'")

block_dirs = sorted(glob.glob(os.path.join(ROOT, "block_number=*")),
                    key=lambda d: int(d.split("=")[-1]))
if not block_dirs:
    sys.exit(f"no block_number=* folders in {ROOT}")
log(f"Snapshots: {len(block_dirs)}")

# -----------------------------------------------------------------------------
# 2. PER-SNAPSHOT OUTCOMES OVER ALL ROWS (one snapshot per query)
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 2: Per-snapshot outcomes over all rows")
log("=" * 70)

RET_SQL = "ret_baseline * 100"
ALPHA_SQL = "(ret_baseline - beta_baseline * market_return) * 100"
qs = [WINSOR, 0.5, 1 - WINSOR]
t1 = time.time()
outcome_rows = []
for d in block_dirs:
    blk = int(d.split("=")[-1])
    src = f"read_parquet('{d}/*.parquet')"
    date, mret, n_ret, n_alpha, q_ret, q_alpha = con.sql(f"""
        SELECT ANY_VALUE(date), ANY_VALUE(market_return),
               COUNT({RET_SQL}), COUNT({ALPHA_SQL}),
               QUANTILE_CONT({RET_SQL}, {qs}), QUANTILE_CONT({ALPHA_SQL}, {qs})
        FROM {src}
    """).fetchone()
    mean_ret, var_ret, mean_alpha, var_alpha = con.sql(f"""
        SELECT AVG({_sql_clip(RET_SQL, q_ret[0], q_ret[2])}),
               VAR_POP({_sql_clip(RET_SQL, q_ret[0], q_ret[2])}),
               AVG({_sql_clip(ALPHA_SQL, q_alpha[0], q_alpha[2])}),
               VAR_POP({_sql_clip(ALPHA_SQL, q_alpha[0], q_alpha[2])})
        FROM {src}
    """).fetchone()
    outcome_rows.append({
        "block_number": blk, "date": pd.Timestamp(date).normalize(),
        "market_return_pct": mret * 100 if mret is not None else np.nan,
        "n_ret": n_ret, "median_ret_pct": q_ret[1],
        "mean_w_ret_pct": mean_ret, "var_w_ret_pct": var_ret,
        "n_alpha": n_alpha, "median_alpha_pct": q_alpha[1],
        "mean_w_alpha_pct": mean_alpha, "var_w_alpha_pct": var_alpha,
    })
outcomes = pd.DataFrame(outcome_rows).sort_values("date").reset_index(drop=True)
log(f"  Done in {time.time()-t1:.1f}s")
log_df(outcomes.round(4), "Per-snapshot outcomes (winsorised within month, all rows):")
save_csv(outcomes.set_index("block_number"), "month_outcomes.csv")

# -----------------------------------------------------------------------------
# 3. MARKET-STATE TABLE (information up to the day before each snapshot)
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 3: Market-state table")
log("=" * 70)


def load_price(token):
    df = pd.read_csv(os.path.join(PRICES_DIR, f"{token}_price.csv"),
                     usecols=["date", "daily_close_usd"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.drop_duplicates("date", keep="last").set_index("date")["daily_close_usd"].astype(float)


# Market = WETH/WBTC 50/50, as the CAPM benchmark (capm_analytics): momentum
# averages the two simple returns over the window; volatility and drawdown use
# the daily-rebalanced 50/50 index. Prices forward-filled up to 3 days.
px = pd.DataFrame({"eth": load_price(WETH), "btc": load_price(WBTC)})
px = px.reindex(pd.date_range(px.index.min(), px.index.max(), freq="D")).ffill(limit=3)
index_ret = px.pct_change(fill_method=None).mean(axis=1, skipna=True)
index_level = (1 + index_ret.fillna(0)).cumprod()

fng = pd.read_csv(args.fng_csv, parse_dates=["date"]).set_index("date").sort_index()
fng = fng.reindex(pd.date_range(fng.index.min(), fng.index.max(), freq="D")).ffill(limit=3)

acc = pd.read_csv(args.accounts_csv).sort_values("date")
acc["acct_growth"] = (acc["eoa_accounts"] + acc["ca_accounts"]).pct_change()
acct_growth = acc.set_index("block")["acct_growth"]

state_rows = []
for i, row in outcomes.iterrows():
    eve = row["date"] - pd.Timedelta(days=1)
    mom = {k: (px.reindex([eve]).iloc[0] / px.reindex([eve - pd.Timedelta(days=k)]).iloc[0] - 1).mean()
           for k in (30, 90)}
    level = index_level.loc[eve - pd.Timedelta(days=364): eve]
    fng_cls = fng["fng_class"].get(eve)
    state_rows.append({
        "block_number": row["block_number"], "date": row["date"],
        "fng": fng["fng_value"].get(eve),
        "fng_class": fng_cls,
        "fng_extreme_fear": float(fng_cls == "Extreme Fear"),
        "fng_extreme_greed": float(fng_cls == "Extreme Greed"),
        "mom_30": mom[30], "mom_90": mom[90],
        "vol_30": index_ret.loc[eve - pd.Timedelta(days=29): eve].std() * np.sqrt(365),
        "dd_365": level.iloc[-1] / level.max() - 1 if len(level) else np.nan,
        "acct_growth": acct_growth.get(row["block_number"], np.nan),
        # Last snapshot's 20-day window ends ~10 days before this one.
        "prev_median_alpha": outcomes["median_alpha_pct"].iloc[i - 1] if i > 0 else np.nan,
    })
state = pd.DataFrame(state_rows)
log_df(state.drop(columns="block_number").round(3), "Market state per snapshot:")
save_csv(state.set_index("block_number"), "market_state.csv")
missing = state[STATE_FEATURES].isna().sum()
if missing.any():
    log(f"  Missing state values (left to the forest): {missing[missing > 0].to_dict()}")

# -----------------------------------------------------------------------------
# 4. SAMPLE: the same number of accounts per snapshot
# -----------------------------------------------------------------------------
log("=" * 70)
log(f"SECTION 4: Sample of {PER_MONTH:,} accounts per snapshot")
log("=" * 70)

holds_eth_sql = _sql_holds_any("holdings_short", ETH_WRAPPER_ADDRESSES)
holds_btc_sql = _sql_holds_any("holdings_short", BTC_WRAPPER_ADDRESSES)
plain_cols = ["date", "total_value_usd", "market_return"] + RETURN_COLS + BETA_COLS + L1_COLS

t1 = time.time()
parts = []
for d in block_dirs:
    blk = int(d.split("=")[-1])
    # Seeded hash per wallet and snapshot: a random, reproducible draw that
    # differs between snapshots. Flags and N only for the drawn rows.
    parts.append(con.sql(f"""
        SELECT {blk} AS block_number, {', '.join(plain_cols)},
               CAST(json_array_length(holdings_short) AS INTEGER) AS num_tokens,
               {holds_eth_sql} AS holds_eth,
               {holds_btc_sql} AS holds_btc
        FROM (
            SELECT {', '.join(plain_cols)}, holdings_short
            FROM read_parquet('{d}/*.parquet')
            WHERE ret_baseline IS NOT NULL
            ORDER BY hash(wallet_address || '-{blk}-{SEED}')
            LIMIT {PER_MONTH}
        )
    """).df())
sample = pd.concat(parts, ignore_index=True)
log(f"  Sample: {len(sample):,} rows in {time.time()-t1:.1f}s")
log(f"  holds_eth=1: {sample['holds_eth'].sum():,}  holds_btc=1: {sample['holds_btc'].sum():,}")

sample["date"] = pd.to_datetime(sample["date"])
sample["month"] = (sample["date"].dt.year - 2020) * 12 + sample["date"].dt.month
sample["log_value_usd"] = np.log1p(sample["total_value_usd"])
for raw, name in zip(L1_COLS, DIST_FEATURES):
    sample[name] = sample[raw] / 2.0 * 100.0

# Returns and alpha in percent, winsorised within each month
RET_PCT_COLS, ALPHA_COLS = [], []
for ret_col, beta_col, label in zip(RETURN_COLS, BETA_COLS, RETURN_LABELS):
    sample[f"ret_pct_{label}"] = sample[ret_col] * 100.0
    sample[f"alpha_{label}"] = (sample[ret_col] - sample[beta_col] * sample["market_return"]) * 100.0
    RET_PCT_COLS.append(f"ret_pct_{label}")
    ALPHA_COLS.append(f"alpha_{label}")

by_month = sample.groupby("block_number")
for col in RET_PCT_COLS + ALPHA_COLS:
    lo = by_month[col].transform(lambda x: x.quantile(WINSOR))
    hi = by_month[col].transform(lambda x: x.quantile(1 - WINSOR))
    sample[col] = sample[col].clip(lower=lo, upper=hi)

# (d): targets demeaned by month
for col in ["ret_pct_baseline", "alpha_baseline"]:
    sample[f"{col}_dm"] = sample[col] - sample.groupby("block_number")[col].transform("mean")

sample = sample.merge(state[["block_number"] + STATE_FEATURES], on="block_number", how="left")

# Month-grouped folds: 72 months in random order, FOLDS folds of equal size
months = np.sort(sample["block_number"].unique())
fold_of = {m: i % FOLDS for i, m in enumerate(np.random.RandomState(SEED).permutation(months))}
sample["fold"] = sample["block_number"].map(fold_of)

log(f"  Feature summary:")
for feat in FEATURES + STATE_FEATURES + DIST_FEATURES:
    vals = sample[feat].dropna()
    log(f"    {feat:20s}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
        f"min={vals.min():.4f}  max={vals.max():.4f}")
log()

# -----------------------------------------------------------------------------
# 5. VARIANCE DECOMPOSITION: between- and within-month shares
# -----------------------------------------------------------------------------
log("=" * 70)
log("SECTION 5: Variance decomposition of baseline return and alpha")
log("=" * 70)

decomp = {}
for key, tgt_col, mean_col, var_col, n_col in [
    ("ret", "ret_pct_baseline", "mean_w_ret_pct", "var_w_ret_pct", "n_ret"),
    ("alpha", "alpha_baseline", "mean_w_alpha_pct", "var_w_alpha_pct", "n_alpha"),
]:
    all_pooled, all_equal = decompose(outcomes[n_col], outcomes[mean_col], outcomes[var_col])
    g = sample.groupby("block_number")[tgt_col]
    smp_pooled, _ = decompose(g.count(), g.mean(), g.var(ddof=0))
    decomp[key] = smp_pooled
    log(f"  {key:5s}: between-month share {all_pooled:.4f} (all rows, rows weighted), "
        f"{all_equal:.4f} (all rows, months weighted), {smp_pooled:.4f} (sample)")
    decomp[f"{key}_all_rows"] = all_pooled
    decomp[f"{key}_all_months"] = all_equal
log()

# -----------------------------------------------------------------------------
# 6. VARIANT (a): random split, month index — the paper's original design
# -----------------------------------------------------------------------------
log("=" * 70)
log(f"SECTION 6: Variant (a) — random 80/20 split, {len(sample):,} rows, {N_CORES} cores")
log("=" * 70)

targets = [(f"ret_pct_{label}", f"ret_{label}") for label in RETURN_LABELS]
targets += [(f"alpha_{label}", f"alpha_{label}") for label in RETURN_LABELS]

summary_rows, importance_rows = [], []
trained_models = {}


def variant_a(features, tag, keep_return_models):
    results = []
    for tgt_col, tgt_name in targets:
        t1 = time.time()
        X = sample[features].to_numpy(dtype=float)
        y = sample[tgt_col].to_numpy(dtype=float)
        mask = np.isfinite(y) & ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        rf = make_rf().fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        perm = perm_imp(rf, X_test, y_test)

        imp_df = pd.DataFrame({
            "feature": features,
            "importance": rf.feature_importances_,
        })
        imp_df["target"] = tgt_name
        imp_df["R2_test"] = round(r2, 4)
        imp_df["MAE_test"] = round(mae, 2)
        imp_df["perm_importance"] = perm
        imp_df["perm_share"] = perm_share(perm)
        imp_df = imp_df.sort_values("importance", ascending=False)

        log(f"  RF-{tag} {tgt_name}: R²={r2:.4f}, MAE={mae:.2f}  ({time.time()-t1:.1f}s)")
        log_df(imp_df.drop(columns=["target"]).round(4), f"  Feature importances ({tgt_name}):")
        results.append(imp_df)
        summary_rows.append({"variant": f"a_{tag}", "target": tgt_name, "scheme": "random 80/20",
                             "n_features": len(features), "min_samples_leaf": MIN_LEAF,
                             "R2": r2, "MAE": mae, "n": int(mask.sum())})
        for f, imp, p_imp, p_sh in zip(features, rf.feature_importances_, perm, perm_share(perm)):
            importance_rows.append({"variant": f"a_{tag}", "target": tgt_name, "feature": f,
                                    "impurity": imp, "perm_importance": p_imp, "perm_share": p_sh})
        if keep_return_models and tgt_name == "ret_baseline":
            trained_models[tgt_name] = (rf, X, y)
    return pd.concat(results)


rf_all = variant_a(FEATURES, "6feat", keep_return_models=True)
save_csv(rf_all, "rf_shap_importance.csv")

log()
log("=" * 70)
log("COMPARISON: 3-feature vs 6-feature RF")
log("=" * 70)
rf_3_all = variant_a(FEATURES_3, "3feat", keep_return_models=False)
save_csv(rf_3_all, "rf_3feat_importance.csv")

log("\n  R² comparison (3-feature vs 6-feature):")
log(f"  {'Target':25s}  {'R²(3-feat)':>12s}  {'R²(6-feat)':>12s}  {'ΔR²':>8s}")
for tgt_col, tgt_name in targets:
    r2_3 = rf_3_all[rf_3_all["target"] == tgt_name]["R2_test"].iloc[0]
    r2_6 = rf_all[rf_all["target"] == tgt_name]["R2_test"].iloc[0]
    log(f"  {tgt_name:25s}  {r2_3:12.4f}  {r2_6:12.4f}  {r2_6 - r2_3:+8.4f}")

# -----------------------------------------------------------------------------
# 7. SHAP VALUES (variant a, 6-feature return models)
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

    # SHAP on the 6-feature baseline return model (see SHAP_SAMPLE_SIZE)
    shap_targets = [("ret_baseline", "baseline")]

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
        # Newer shap returns a one-element array for regressors, older a scalar.
        shap_df["expected_value"] = float(np.ravel(explainer.expected_value)[0])
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

        # SHAP dependence plots: month, beta_baseline, log_value_usd
        for col_idx, name, xlabel in [
            (0, "month", "Snapshot month (months since Dec 2019)"),
            (3, "beta", "Portfolio beta (baseline)"),
            (1, "value", "Log portfolio value (USD)"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.dependence_plot(
                col_idx, shap_values, X_shap,
                feature_names=FEATURE_NAMES_DISPLAY,
                interaction_index=None,
                show=False, ax=ax,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("SHAP value (impact on predicted return %)")
            plt.tight_layout()
            path = os.path.join(OUTPUT_DIR, f"rf_shap_{name}_dep_{label}.pdf")
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            log(f"  → saved {path}")
trained_models.clear()

# -----------------------------------------------------------------------------
# 8. VARIANTS (b), (c), (d): folds grouped by month
# -----------------------------------------------------------------------------
log()
log("=" * 70)
log(f"SECTION 8: Variants (b)-(d) — {FOLDS} folds of whole months, baseline return and alpha")
log("=" * 70)


def between_r2(block, y, pred, base=None):
    """R² of the month means: how much of the between-month variation the
    predictions capture. base = benchmark mean (default: mean of the month means)."""
    m = pd.DataFrame({"m": block, "y": y, "p": pred}).groupby("m").mean()
    base = m["y"].mean() if base is None else base
    return 1 - ((m["y"] - m["p"]) ** 2).sum() / ((m["y"] - base) ** 2).sum()


def grouped_cv(variant, features, tgt_col, tgt_name, min_leaf=MIN_LEAF):
    """Train on FOLDS-1 folds of whole months, predict the held-out months."""
    t1 = time.time()
    X = sample[features].to_numpy(dtype=float)
    y = sample[tgt_col].to_numpy(dtype=float)
    acct = [f for f in features if f not in STATE_FEATURES]    # state NaNs go to the forest
    mask = np.isfinite(y) & ~np.isnan(sample[acct].to_numpy(dtype=float)).any(axis=1)
    folds = sample["fold"].to_numpy()
    oof = np.full(len(y), np.nan)
    fold_r2, imps, perms = [], [], []
    for k in range(FOLDS):
        tr, te = mask & (folds != k), mask & (folds == k)
        rf = make_rf(min_leaf).fit(X[tr], y[tr])
        oof[te] = rf.predict(X[te])
        fold_r2.append(r2_score(y[te], oof[te]))
        imps.append(rf.feature_importances_)
        perms.append(perm_imp(rf, X[te], y[te]))
    r2 = r2_score(y[mask], oof[mask])
    # (d) is demeaned by month, so its month means are ~0 and have nothing to explain.
    r2_b = (between_r2(sample["block_number"].to_numpy()[mask], y[mask], oof[mask])
            if variant != "d" else np.nan)
    mae = mean_absolute_error(y[mask], oof[mask])
    log(f"  ({variant}) {tgt_name}: out-of-fold R²={r2:.4f} (folds {min(fold_r2):.4f}..{max(fold_r2):.4f}), "
        f"between-month R²={r2_b:.4f}, MAE={mae:.2f}  ({time.time()-t1:.1f}s)")
    summary_rows.append({"variant": variant, "target": tgt_name, "scheme": f"{FOLDS} month-grouped folds",
                         "n_features": len(features), "min_samples_leaf": min_leaf,
                         "R2": r2, "R2_fold_mean": np.mean(fold_r2),
                         "R2_fold_min": min(fold_r2), "R2_fold_max": max(fold_r2),
                         "R2_between_months": r2_b, "MAE": mae, "n": int(mask.sum())})
    imp, perm = np.mean(imps, axis=0), np.mean(perms, axis=0)
    for f, i_, p_, s_ in zip(features, imp, perm, perm_share(perm)):
        importance_rows.append({"variant": variant, "target": tgt_name, "feature": f,
                                "impurity": i_, "perm_importance": p_, "perm_share": s_})
    log_df(pd.DataFrame({"impurity": imp, "perm_importance": perm, "perm_share": perm_share(perm)},
                        index=features).sort_values("perm_importance", ascending=False).round(4),
           f"  ({variant}) importances, mean over folds:")
    return r2, r2_b


def forward_chain(features, tgt_col, tgt_name):
    """(c) in real time: train on all earlier years, test on the next year.
    R²_os compares with the training mean (Campbell & Thompson 2008)."""
    years = sample["date"].dt.year.to_numpy()
    X = sample[features].to_numpy(dtype=float)
    y = sample[tgt_col].to_numpy(dtype=float)
    acct = [f for f in features if f not in STATE_FEATURES]
    mask = np.isfinite(y) & ~np.isnan(sample[acct].to_numpy(dtype=float)).any(axis=1)
    block = sample["block_number"].to_numpy()
    rows = []
    for ty in FORWARD_TEST_YEARS:
        tr, te = mask & (years < ty), mask & (years == ty)
        if not tr.any() or not te.any():
            continue
        rf = make_rf(STATE_LEAF).fit(X[tr], y[tr])
        pred = rf.predict(X[te])
        base = y[tr].mean()
        r2_os = 1 - ((y[te] - pred) ** 2).sum() / ((y[te] - base) ** 2).sum()
        rows.append({"target": tgt_name, "test_year": ty, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                     "R2_test": r2_score(y[te], pred), "R2_os": r2_os,
                     "R2_between_os": between_r2(block[te], y[te], pred, base=base)})
        log(f"  (c forward) {tgt_name} {ty}: R²_os={r2_os:.4f}, between-month R²_os={rows[-1]['R2_between_os']:.4f}")
    return rows


variant_r2 = {}
forward_rows = []
for tgt_col, tgt_name, key in [("ret_pct_baseline", "ret_baseline", "ret"),
                               ("alpha_baseline", "alpha_baseline", "alpha")]:
    log(f"\n--- {tgt_name} ---")
    variant_r2[("b", key)] = grouped_cv("b", FEATURES, tgt_col, tgt_name)
    variant_r2[("c", key)] = grouped_cv("c", STATE_FEATURES + ACCOUNT_FEATURES, tgt_col, tgt_name,
                                        min_leaf=STATE_LEAF)
    forward_rows += forward_chain(STATE_FEATURES + ACCOUNT_FEATURES, tgt_col, tgt_name)
    variant_r2[("d", key)] = grouped_cv("d", ACCOUNT_FEATURES + DIST_FEATURES, f"{tgt_col}_dm", f"{tgt_name}_demeaned")

summary_df = pd.DataFrame(summary_rows)
log_df(summary_df.round(4), "Variant summary:")
save_csv(summary_df, "rf_variants_summary.csv")
save_csv(pd.DataFrame(importance_rows), "rf_variants_importance.csv")
forward_df = pd.DataFrame(forward_rows)
log_df(forward_df.round(4), "Forward chaining (variant c):")
save_csv(forward_df, "rf_forward_chaining.csv")

# -----------------------------------------------------------------------------
# 9. FOUR VARIANCE SHARES (sample basis)
# -----------------------------------------------------------------------------
log()
log("=" * 70)
log("SECTION 9: Shares of return variance")
log("=" * 70)

share_rows = []
for key in ["ret", "alpha"]:
    s_b = decomp[key]
    r2_between_c = variant_r2[("c", key)][1]
    r2_within_d = variant_r2[("d", key)][0]
    between_pred = s_b * max(r2_between_c, 0.0)
    within_pred = (1 - s_b) * max(r2_within_d, 0.0)
    share_rows.append({
        "target": f"{key}_baseline",
        "between_month_share_all_rows": decomp[f"{key}_all_rows"],
        "between_month_share_all_months": decomp[f"{key}_all_months"],
        "between_month_share_sample": s_b,
        "between_predictable": between_pred,
        "between_unpredictable": s_b - between_pred,
        "within_predictable_allocation": within_pred,
        "within_idiosyncratic": (1 - s_b) - within_pred,
        "R2_between_c": r2_between_c,
        "R2_within_d": r2_within_d,
    })
shares_df = pd.DataFrame(share_rows).set_index("target")
log_df(shares_df.T.round(4), "Variance shares (the four predictable/unpredictable shares sum to 1):")
save_csv(shares_df, "variance_shares.csv")

# -----------------------------------------------------------------------------
# 10. MONTH MAPPING
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
# 11. SUMMARY
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
