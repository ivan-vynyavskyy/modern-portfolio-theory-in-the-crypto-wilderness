#!/usr/bin/env python3
"""
block_tag_coverage.py

Joins one or more top-X% address parquets (block, address) with GraphSense
tags and reports per-block coverage + category/concept distributions.

Usage:
  python block_tag_coverage.py \
    --addresses file1.parquet file2.parquet \
    --tags      top_0_1_addresses_tags.parquet \
    --out-dir   tag_coverage_out
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addresses", required=True, nargs="+",
                    help="One or more parquet files with (block, address)")
    ap.add_argument("--tags", required=True, help="GraphSense tags parquet")
    ap.add_argument("--out-dir", default="tag_coverage_out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load and concat all address files
    addr_parts = []
    for f in args.addresses:
        part = pd.read_parquet(f)
        print(f"[info] loaded {f}: {len(part):,} rows, {part['block'].nunique()} blocks")
        addr_parts.append(part)
    addr = pd.concat(addr_parts, ignore_index=True).drop_duplicates(subset=["block", "address"])
    print(f"[info] combined addresses: {len(addr):,} rows, {addr['block'].nunique()} blocks")

    tags = pd.read_parquet(args.tags)
    print(f"[info] tags: {len(tags):,} rows, {tags['address'].nunique():,} unique addresses\n")

    # --- pre-compute per-address lookups ---
    tag_has_cat = tags.groupby("address")["category"].apply(
        lambda s: s.notna().any()
    ).rename("has_category")

    tag_has_concept = tags.groupby("address")["concepts"].apply(
        lambda s: s.explode().dropna().pipe(
            lambda x: x[x.astype(str).str.strip() != ""]
        ).shape[0] > 0
    ).rename("has_concept")

    # per-address best category (first non-null)
    tag_best_cat = tags.groupby("address")["category"].apply(
        lambda s: s.dropna().iloc[0] if s.notna().any() else None
    ).rename("best_category")

    # per-address: all concepts
    tag_all_concepts = tags.groupby("address")["concepts"].apply(
        lambda s: s.explode().dropna().pipe(
            lambda x: x[x.astype(str).str.strip() != ""]
        ).unique().tolist()
    ).rename("all_concepts")

    tagged_addresses = set(tags["address"].dropna().unique())

    tag_info = pd.DataFrame({
        "has_category": tag_has_cat,
        "has_concept": tag_has_concept,
        "best_category": tag_best_cat,
        "all_concepts": tag_all_concepts,
    })
    tag_info["has_any"] = tag_info["has_category"] | tag_info["has_concept"]

    # --- join: for each (block, address) attach tag info ---
    addr["in_tags"] = addr["address"].isin(tagged_addresses)
    addr = addr.join(tag_info, on="address")
    addr["has_category"] = addr["has_category"].fillna(False)
    addr["has_concept"] = addr["has_concept"].fillna(False)
    addr["has_any"] = addr["has_any"].fillna(False)
    addr["no_info"] = ~addr["has_any"]

    # ===================================================================
    #  A. PER-BLOCK COVERAGE TABLE
    # ===================================================================
    sep = "=" * 120
    print(sep)
    print("  A. PER-BLOCK COVERAGE")
    print(sep)

    block_stats = addr.groupby("block").agg(
        n_addresses=("address", "size"),
        n_in_tags=("in_tags", "sum"),
        n_has_category=("has_category", "sum"),
        n_has_concept=("has_concept", "sum"),
        n_has_any=("has_any", "sum"),
        n_no_info=("no_info", "sum"),
    ).astype(int)
    for c in ["in_tags", "has_category", "has_concept", "has_any", "no_info"]:
        block_stats[f"pct_{c}"] = (100.0 * block_stats[f"n_{c}"] / block_stats["n_addresses"]).round(2)

    hdr = (f"  {'block':>12s}  {'n_addr':>8s}  {'in_tags':>8s}  {'%tags':>7s}"
           f"  {'has_cat':>8s}  {'%cat':>7s}  {'has_con':>8s}  {'%con':>7s}"
           f"  {'has_any':>8s}  {'%any':>7s}  {'no_info':>8s}  {'%none':>7s}")
    print(hdr)
    print("  " + "-" * 116)
    for block, r in block_stats.iterrows():
        print(f"  {int(block):>12d}  {r['n_addresses']:>8,}  {r['n_in_tags']:>8,}  {r['pct_in_tags']:>6.2f}%"
              f"  {r['n_has_category']:>8,}  {r['pct_has_category']:>6.2f}%"
              f"  {r['n_has_concept']:>8,}  {r['pct_has_concept']:>6.2f}%"
              f"  {r['n_has_any']:>8,}  {r['pct_has_any']:>6.2f}%"
              f"  {r['n_no_info']:>8,}  {r['pct_no_info']:>6.2f}%")

    # --- averages ---
    print(f"\n  AVERAGES ACROSS BLOCKS:")
    for c in ["pct_in_tags", "pct_has_category", "pct_has_concept", "pct_has_any", "pct_no_info"]:
        s = block_stats[c]
        print(f"    {c:25s}  mean={s.mean():6.2f}%  median={s.median():6.2f}%"
              f"  min={s.min():6.2f}%  max={s.max():6.2f}%")

    # --- overall ---
    total = len(addr)
    print(f"\n  OVERALL ({total:,} block×address pairs):")
    for label, col in [("in tags", "in_tags"), ("has category", "has_category"),
                       ("has concept", "has_concept"), ("category OR concept", "has_any"),
                       ("no info", "no_info")]:
        s = int(addr[col].sum())
        print(f"    {label:25s}  {s:>10,} / {total:>10,}  ({100*s/total:6.2f}%)")

    # ===================================================================
    #  B. CATEGORY DISTRIBUTION (among matched addresses)
    # ===================================================================
    print(f"\n{sep}")
    print("  B. CATEGORY DISTRIBUTION (addresses with category, per block×address)")
    print(sep)

    cat_rows = addr[addr["has_category"]].copy()
    n_cat = len(cat_rows)
    cat_vc = cat_rows["best_category"].value_counts()
    print(f"  Total block×address with category: {n_cat:,}\n")
    print(f"  {'category':40s}  {'count':>10s}  {'%of_categorized':>16s}  {'%of_all':>10s}")
    print("  " + "-" * 80)
    for val, cnt in cat_vc.items():
        print(f"  {str(val):40s}  {cnt:>10,}  ({100*cnt/n_cat:>6.2f}%)        ({100*cnt/total:>6.2f}%)")

    # save
    cat_vc_df = cat_vc.reset_index()
    cat_vc_df.columns = ["category", "count"]
    cat_vc_df["pct_of_categorized"] = (100 * cat_vc_df["count"] / n_cat).round(4)
    cat_vc_df["pct_of_all"] = (100 * cat_vc_df["count"] / total).round(4)
    cat_vc_df.to_csv(out_dir / "category_distribution.csv", index=False)

    # ===================================================================
    #  C. CONCEPT DISTRIBUTION (among matched addresses)
    # ===================================================================
    print(f"\n{sep}")
    print("  C. CONCEPT DISTRIBUTION (addresses with concepts, per block×address)")
    print(sep)

    concept_rows = addr[addr["has_concept"]].copy()
    n_con = len(concept_rows)
    # explode concepts
    concept_flat = concept_rows.explode("all_concepts")
    concept_vc = concept_flat["all_concepts"].value_counts()
    print(f"  Total block×address with ≥1 concept: {n_con:,}\n")
    print(f"  {'concept':50s}  {'count':>10s}  {'%of_concept_addr':>16s}  {'%of_all':>10s}")
    print("  " + "-" * 90)
    for val, cnt in concept_vc.items():
        print(f"  {str(val):50s}  {cnt:>10,}  ({100*cnt/n_con:>6.2f}%)        ({100*cnt/total:>6.2f}%)")

    # save
    concept_vc_df = concept_vc.reset_index()
    concept_vc_df.columns = ["concept", "count"]
    concept_vc_df["pct_of_concept_addr"] = (100 * concept_vc_df["count"] / n_con).round(4)
    concept_vc_df["pct_of_all"] = (100 * concept_vc_df["count"] / total).round(4)
    concept_vc_df.to_csv(out_dir / "concept_distribution.csv", index=False)

    # ===================================================================
    #  D. COMBINED: effective label per address (category > concept > none)
    # ===================================================================
    print(f"\n{sep}")
    print("  D. EFFECTIVE LABEL (category preferred, concept fallback)")
    print(sep)

    def effective_label(row):
        if row["has_category"] and pd.notna(row.get("best_category")):
            return row["best_category"]
        if row["has_concept"] and isinstance(row.get("all_concepts"), list) and len(row["all_concepts"]) > 0:
            return row["all_concepts"][0]
        return "UNLABELED"

    addr["effective_label"] = addr.apply(effective_label, axis=1)
    eff_vc = addr["effective_label"].value_counts()
    print(f"\n  {'effective_label':50s}  {'count':>10s}  {'%of_all':>10s}")
    print("  " + "-" * 75)
    for val, cnt in eff_vc.head(40).items():
        print(f"  {str(val):50s}  {cnt:>10,}  ({100*cnt/total:>6.2f}%)")
    if len(eff_vc) > 40:
        print(f"  ... and {len(eff_vc) - 40} more labels")

    eff_vc_df = eff_vc.reset_index()
    eff_vc_df.columns = ["effective_label", "count"]
    eff_vc_df["pct"] = (100 * eff_vc_df["count"] / total).round(4)
    eff_vc_df.to_csv(out_dir / "effective_label_distribution.csv", index=False)

    # ===================================================================
    #  E. PER-BLOCK CATEGORY BREAKDOWN
    # ===================================================================
    print(f"\n{sep}")
    print("  E. PER-BLOCK CATEGORY BREAKDOWN (top 5 categories per block)")
    print(sep)
    for block in sorted(addr["block"].unique()):
        bsub = addr[(addr["block"] == block) & (addr["has_category"])]
        if bsub.empty:
            continue
        n_b = int((addr["block"] == block).sum())
        top5 = bsub["best_category"].value_counts().head(5)
        top_str = ", ".join(f"{c}={cnt}" for c, cnt in top5.items())
        print(f"  block {block:>12d}  (n={n_b:,}, categorized={len(bsub):,})  {top_str}")

    print(f"\n[saved] {out_dir}/category_distribution.csv")
    print(f"[saved] {out_dir}/concept_distribution.csv")
    print(f"[saved] {out_dir}/effective_label_distribution.csv")
    print("[done]")


if __name__ == "__main__":
    main()