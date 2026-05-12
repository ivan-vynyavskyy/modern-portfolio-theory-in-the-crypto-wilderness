#!/usr/bin/env python3
"""
fetch_stablecoin_lending_rates.py

Description:
    Fetch historical USDT/USDC lending APYs from DefiLlama for major
    Ethereum-based money markets (Aave v2/v3, Compound v2/v3, Morpho
    Blue, Spark). Computes monthly and overall averages to establish a
    risk-free rate benchmark.

    Pools are filtered to Ethereum-only, TVL >= $1M, and APY in
    [0, 50%] to exclude anomalies.

Output:
    stablecoin_monthly_avg_apy_ethereum.csv       (monthly by asset)
    stablecoin_overall_avg_apy_by_asset_ethereum.csv  (overall by asset)
    stablecoin_overall_avg_apy_combined_ethereum.csv  (overall combined)
"""

import requests
import pandas as pd

POOLS_URL = "https://yields.llama.fi/pools"

ETHEREUM_PROJECTS = [
    "aave-v2", "aave-v3", "compound-v2", "compound-v3",
    "morpho-blue", "spark",
]

# -----------------------------------------------------------------------------
# Fetch and filter pools
# -----------------------------------------------------------------------------
pools = requests.get(POOLS_URL, timeout=30).json()["data"]
pools_df = pd.DataFrame(pools)

stable = pools_df[
    pools_df["symbol"].str.contains("USDT|USDC", case=False, na=False)
].copy()

# Restrict to Ethereum: wallets in our population cannot bridge without
# leaving the ecosystem, so only on-chain Ethereum yields are relevant.
stable = stable[stable["chain"] == "Ethereum"]

stable = stable[
    (stable["tvlUsd"] >= 1_000_000) &
    (stable["apy"] >= 0) &
    (stable["apy"] <= 50)
]

stable = stable[stable["project"].isin(ETHEREUM_PROJECTS)]

print(f"[info] {len(stable)} Ethereum pools after filtering")

# -----------------------------------------------------------------------------
# Fetch historical APY per pool
# -----------------------------------------------------------------------------
rows = []

for _, pool in stable.iterrows():
    pool_id = pool["pool"]
    url = f"https://yields.llama.fi/chart/{pool_id}"

    try:
        hist = requests.get(url, timeout=30).json()["data"]
    except Exception:
        continue

    h = pd.DataFrame(hist)
    if h.empty:
        continue

    h["date"] = pd.to_datetime(h["timestamp"]).dt.date
    h["month"] = pd.to_datetime(h["date"]).dt.to_period("M").astype(str)

    h["pool"] = pool_id
    h["symbol"] = pool["symbol"]
    h["project"] = pool["project"]
    h["chain"] = pool["chain"]

    rows.append(h)

df = pd.concat(rows, ignore_index=True)

df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= "2021-01-01"]

df["asset"] = df["symbol"].str.extract("(USDT|USDC)", expand=False)

# -----------------------------------------------------------------------------
# Aggregate
# -----------------------------------------------------------------------------
monthly = (
    df.groupby(["asset", "month"])
      .agg(
          avg_apy=("apy", "mean"),
          median_apy=("apy", "median"),
          n_obs=("apy", "count")
      )
      .reset_index()
)

monthly.to_csv("stablecoin_monthly_avg_apy_ethereum.csv", index=False)
print(monthly.head())

overall_by_asset = (
    monthly.groupby("asset")
    .agg(
        overall_avg_apy=("avg_apy", "mean"),
        overall_median_monthly_apy=("median_apy", "median"),
        months_observed=("month", "nunique")
    )
    .reset_index()
)

overall_all_assets = pd.DataFrame({
    "asset": ["USDT_USDC_COMBINED"],
    "overall_avg_apy": [monthly["avg_apy"].mean()],
    "overall_median_monthly_apy": [monthly["median_apy"].median()],
    "months_observed": [monthly["month"].nunique()]
})

overall_by_asset.to_csv("stablecoin_overall_avg_apy_by_asset_ethereum.csv", index=False)
overall_all_assets.to_csv("stablecoin_overall_avg_apy_combined_ethereum.csv", index=False)

print("\nOverall average APY by asset:")
print(overall_by_asset)

print("\nOverall average APY across USDT + USDC:")
print(overall_all_assets)