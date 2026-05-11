#!/usr/bin/env python3
"""
proxy_resolver_summary.py

Purpose
-------
Summarize ERC-20 proxy-resolver verification results from a TSV into
human-readable classification and verification tables.

What it does
------------
- Reads the proxy-resolver TSV and normalizes proxy types and verification flags.
- Computes overall classification buckets (undetermined, proxy, unknown/other).
- Prints per-proxy-type breakdowns and static/runtime verification crosstabs.

Inputs
------
- TSV file (columns: address, proxy_type, verified_static, verified_runtime, symbol, decimals, name)

Outputs
-------
- Printed summary tables (classification, verification combinations, pass rates)

Notes
-----
- Proxy types are bucketed into minimal_proxy, eip1967_proxy, beacon_proxy, undetermined, or other.
- Duplicate addresses can be dropped with --drop-dupe-addresses.
"""

import argparse
import sys
import pandas as pd


PROXY_KINDS = {"minimal_proxy", "eip1967_proxy", "beacon_proxy"}
KNOWN_PROXY_TYPES = PROXY_KINDS | {"undetermined"}

def normalize_proxy_type(x):
    if not isinstance(x, str) or not x.strip():
        return "unknown"
    s = x.strip().lower()
    if s == "direct":
        return "undetermined"
    # keep known labels, bucket everything else as "other"
    if s in KNOWN_PROXY_TYPES:
        return s
    return "other"

def main():
    ap = argparse.ArgumentParser(description="Summarize ERC-20 v2 verification results for a TSV file.")
    ap.add_argument("tsv", help="Input TSV file with columns from the v2 script")
    ap.add_argument("--drop-dupe-addresses", action="store_true",
                    help="Drop duplicate rows by address, keeping the first occurrence")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.tsv, sep="\t", dtype=str, engine="python")
    except Exception as e:
        print(f"ERROR: failed to read TSV '{args.tsv}': {e}", file=sys.stderr)
        sys.exit(1)

    # Basic cleaning / normalization
    for col in ["address", "proxy_type", "verified_static", "verified_runtime", "symbol", "decimals", "name"]:
        if col not in df.columns:
            print(f"ERROR: missing required column '{col}' in input TSV.", file=sys.stderr)
            sys.exit(1)

    if args.drop_dupe_addresses and "address" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["address"], keep="first").reset_index(drop=True)
        after = len(df)
        print(f"[info] dropped {before - after} duplicate addresses; {after} unique remain")

    df["proxy_type"] = df["proxy_type"].map(normalize_proxy_type)
    df["is_proxy"]   = df["proxy_type"].isin(PROXY_KINDS)

    df["verified_static_bool"]  = df["verified_static"].astype(str).str.strip().str.upper() == "Y"
    df["verified_runtime_bool"] = df["verified_runtime"].astype(str).str.strip().str.upper() == "Y"

    total = len(df)
    print("\n=== Overview ===")
    print(f"Total rows: {total}")

    # Undetermined vs proxy vs unknown/other
    undetermined_count = (df["proxy_type"] == "undetermined").sum()
    proxy_count  = df["is_proxy"].sum()
    other_count  = total - undetermined_count - proxy_count  # unknown/other bucket

    def pct(n): 
        return f"{(100.0 * n / total):.2f}%"

    print("\n=== Classification buckets ===")
    print(f"Undetermined: {undetermined_count} ({pct(undetermined_count)})")
    print(f"Proxy       : {proxy_count} ({pct(proxy_count)})")
    print(f"Other/Unknown: {other_count} ({pct(other_count)})")

    # Breakdown by proxy_type
    print("\n=== Proxy type breakdown ===")
    proxy_breakdown = (
        df["proxy_type"]
        .value_counts(dropna=False)
        .rename_axis("proxy_type")
        .reset_index(name="count")
        .assign(share_pct=lambda d: (d["count"] / total * 100).round(2))
        .sort_values(["proxy_type"])
    )
    print(proxy_breakdown.to_string(index=False))

    # Verification flags: individual
    vs_counts = df["verified_static_bool"].value_counts().rename_axis("verified_static").reset_index(name="count")
    vr_counts = df["verified_runtime_bool"].value_counts().rename_axis("verified_runtime").reset_index(name="count")

    # Normalize boolean labels to Y/N for readability
    vs_counts["verified_static"] = vs_counts["verified_static"].map({True:"Y", False:"N"})
    vr_counts["verified_runtime"] = vr_counts["verified_runtime"].map({True:"Y", False:"N"})

    vs_counts["share_pct"] = (vs_counts["count"] / total * 100).round(2)
    vr_counts["share_pct"] = (vr_counts["count"] / total * 100).round(2)

    print("\n=== Verified (static) ===")
    print(vs_counts.sort_values("verified_static").to_string(index=False))

    print("\n=== Verified (runtime) ===")
    print(vr_counts.sort_values("verified_runtime").to_string(index=False))

    # Verification flags: combination crosstab
    combo = pd.crosstab(df["verified_static_bool"], df["verified_runtime_bool"], dropna=False)
    # Make a pretty version with Y/N headers
    combo_pretty = combo.rename(index={True:"S=Y", False:"S=N"}, columns={True:"R=Y", False:"R=N"})
    combo_pretty.loc["row_total"] = combo_pretty.sum(axis=0)
    combo_pretty["col_total"] = combo_pretty.sum(axis=1)
    print("\n=== Verification combinations (counts) ===")
    print(combo_pretty.to_string())

    # Also a percentage version
    combo_pct = (combo / combo.values.sum() * 100).round(2)
    combo_pct_pretty = combo_pct.rename(index={True:"S=Y", False:"S=N"}, columns={True:"R=Y", False:"R=N"})
    print("\n=== Verification combinations (percent) ===")
    print(combo_pct_pretty.to_string())

    # Proxy type vs runtime pass rate
    by_pt_rt = (
        df.groupby("proxy_type")["verified_runtime_bool"]
        .agg(count="size", runtime_pass="sum")
        .assign(runtime_pass_rate_pct=lambda d: (d["runtime_pass"] / d["count"] * 100).round(2))
        .reset_index()
        .sort_values(["proxy_type"])
    )
    print("\n=== Runtime pass rate by proxy_type ===")
    print(by_pt_rt.to_string(index=False))

    by_pt_st = (
        df.groupby("proxy_type")["verified_static_bool"]
        .agg(count="size", static_pass="sum")
        .assign(static_pass_rate_pct=lambda d: (d["static_pass"] / d["count"] * 100).round(2))
        .reset_index()
    )
    print("\n=== Static pass rate by proxy_type ===")
    print(by_pt_st.to_string(index=False))

if __name__ == "__main__":
    main()
