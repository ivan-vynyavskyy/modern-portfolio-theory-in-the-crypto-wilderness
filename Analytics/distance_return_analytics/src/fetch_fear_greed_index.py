#!/usr/bin/env python3
"""
fetch_fear_greed_index.py

Description:
    Fetch the full daily history of the Crypto Fear & Greed Index from
    alternative.me (0 = extreme fear, 100 = extreme greed; daily since
    2018-02-01). Input of the market-state features in
    rf_shap_analysis_extended.py (variant c).

Output:
    fear_greed_index.csv (date, fng_value, fng_class), one row per day,
    in --out-dir (default: ../data).
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.alternative.me/fng/?limit=0&format=json"

p = argparse.ArgumentParser(description="Fetch the daily Crypto Fear & Greed Index.")
p.add_argument("--out-dir", type=str,
               default=str(Path(__file__).resolve().parents[1] / "data"))
args = p.parse_args()

data = requests.get(API_URL, timeout=60).json()["data"]
df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.date
df = (df.rename(columns={"value": "fng_value", "value_classification": "fng_class"})
        [["date", "fng_value", "fng_class"]]
        .astype({"fng_value": int})
        .drop_duplicates("date")
        .sort_values("date"))

full = pd.date_range(df["date"].min(), df["date"].max(), freq="D").date
missing = sorted(set(full) - set(df["date"]))
out = Path(args.out_dir) / "fear_greed_index.csv"
df.to_csv(out, index=False)
print(f"[info] {len(df):,} days, {df['date'].min()} .. {df['date'].max()}, "
      f"{len(missing)} missing: {', '.join(str(d) for d in missing)}")
print(f"[saved] {out}")
