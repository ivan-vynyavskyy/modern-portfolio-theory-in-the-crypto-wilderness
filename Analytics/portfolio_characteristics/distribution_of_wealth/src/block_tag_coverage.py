#!/usr/bin/env python3
"""
block_tag_coverage.py

Description:
    Address attribution of the top-0.1 % holders with GraphSense / iknaio
    address tags. Reproduces every statistic the paper's "Address Attribution
    of Top-0.1% Holders" subsection used (tag coverage per snapshot, category
    distribution, concept distribution, effective label) and adds the
    account-type split, the actor view, abuse tags and a tag-quality table.

    The top-0.1 % addresses of every monthly snapshot come from
    top_pct_ca_eoa_breakdown.py (--top-addresses, a (block, address) parquet);
    their distinct union was queried against the GraphSense REST API, which
    produced the three parquet files below.

Inputs:
    --summary    top_addresses_0_1_pct_eth_included_address_summary.parquet
                 One row per distinct address: is_contract (None = lookup
                 error), n_tags, labels ('|'-separated), categories / concepts /
                 abuse_concepts / actors / actor_concepts (','-separated unique
                 sets), best_* = the tag with the highest confidence level.
    --tags       top_addresses_0_1_pct_eth_included_tags.parquet (optional)
                 Raw tag rows (label, category, concepts, actor, tag_type,
                 confidence, tagpack_creator, ...); only used for the
                 tag-quality table.
    --addresses  top_addresses_0_1_pct_eth_included.parquet (optional; on the
                 cluster under $DATA/data/). (block, address) membership of the
                 top 0.1 % at each snapshot. Enables the PAIR-weighted
                 statistics (one observation per block x address, the basis of
                 the paper's numbers) and the per-snapshot coverage table.

Definitions (per address, inherited by every (block, address) pair):
    in_tags        at least one tag row              (n_tags > 0)
    has_category   at least one tag with a category
    has_concept    at least one tag with a concept
    has_any        has_category or has_concept      -> "carries a tag" in the paper
    best_category  category of the highest-confidence tag (summary column)
    effective_label best_category, else the highest-priority concept
                   (CONCEPT_PRIORITY), else UNLABELED
    type           CA / EOA from is_contract, 'unknown' when the lookup failed

Outputs (--out-dir, default ../data):
    Address-weighted (always), suffix _by_address:
        category_distribution_by_address.csv, concept_distribution_by_address.csv,
        effective_label_distribution_by_address.csv, tag_actor_distribution_by_address.csv,
        tag_actor_concepts_by_address.csv, tag_abuse_distribution_by_address.csv
    Pair-weighted (only with --addresses; the names generate_all_macros.py reads):
        category_distribution.csv, concept_distribution.csv,
        effective_label_distribution.csv, tag_actor_distribution.csv,
        tag_actor_concepts.csv, tag_abuse_distribution.csv, tag_coverage_per_block.csv
    Both bases:
        tag_coverage_summary.csv   one row per basis (address / pair)
        tag_coverage_by_type.csv   coverage per account type and basis
        tag_confidence_distribution.csv (with --tags)
        tag_attribution_report.txt the printed report

Usage:
    python block_tag_coverage.py \\
        --summary ../data/top_addresses_0_1_pct_eth_included_address_summary.parquet \\
        --tags    ../data/top_addresses_0_1_pct_eth_included_tags.parquet \\
        [--addresses /path/to/data/top_addresses_0_1_pct_eth_included.parquet] \\
        --out-dir ../data
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Concept used as effective label when an address has concepts but no
# category. Entity-type concepts first, generic ones last; any concept not
# listed ranks after the list in alphabetical order.
CONCEPT_PRIORITY: List[str] = [
    "exchange", "defi_custody", "defi_dex", "defi_lending", "defi_bridge",
    "defi_staking", "defi_dao", "defi_derivative", "defi_token", "defi_dex_pair",
    "defi", "deposit_wallet", "hot_wallet", "cold_wallet", "warm_wallet",
    "escrow_wallet", "ico_wallet", "wallet_service", "payment_processor",
    "mixing", "mixing_service", "mining_service", "miner", "gambling", "market",
    "nameservice", "messaging_service", "search_engine", "index", "hosting",
    "organization", "entity", "service", "user",
]
_PRIO = {c: i for i, c in enumerate(CONCEPT_PRIORITY)}

# Categories counted as "exchange- and DeFi-related" in the paper's text.
EXCHANGE_DEFI_CATEGORIES = [
    "exchange", "defi", "defi_dex_pair", "defi_token", "defi_dex", "defi_lending",
    "defi_bridge", "defi_staking", "defi_dao", "defi_derivative",
]

SEP = "=" * 110


class Report:
    """Collects the printed report so it can also be written to a file."""

    def __init__(self) -> None:
        self.buf = io.StringIO()

    def __call__(self, *args, **kwargs) -> None:
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.buf)


def split_set(s: pd.Series) -> pd.Series:
    """','-separated unique set column -> list column (empty list for NaN)."""
    return s.fillna("").map(lambda x: [t.strip() for t in x.split(",") if t.strip()])


def pick_concept(concepts: List[str]) -> Optional[str]:
    if not concepts:
        return None
    return sorted(concepts, key=lambda c: (_PRIO.get(c, len(_PRIO)), c))[0]


def load_summary(path: Path) -> pd.DataFrame:
    s = pd.read_parquet(path)
    s["address"] = s["address"].str.lower().str.strip()
    assert s["address"].is_unique, "address_summary must have one row per address"
    s["in_tags"] = s["n_tags"].fillna(0).astype(int) > 0
    s["has_category"] = s["categories"].notna() & (s["categories"].str.strip() != "")
    s["has_concept"] = s["concepts"].notna() & (s["concepts"].str.strip() != "")
    s["has_any"] = s["has_category"] | s["has_concept"]
    s["no_info"] = ~s["has_any"]
    s["concept_list"] = split_set(s["concepts"])
    s["abuse_list"] = split_set(s["abuse_concepts"])
    s["actor_concept_list"] = split_set(s["actor_concepts"])
    s["has_actor"] = s["best_actor"].notna()
    s["has_abuse"] = s["abuse_list"].map(len) > 0
    s["type"] = s["is_contract"].map({True: "CA", False: "EOA"}).fillna("unknown")
    fallback = s["concept_list"].map(pick_concept)
    s["effective_label"] = np.where(
        s["has_category"], s["best_category"],
        np.where(s["has_concept"], fallback, "UNLABELED"),
    )
    return s


def load_pairs(paths: List[str], summary: pd.DataFrame, rep: Report) -> pd.DataFrame:
    parts = []
    for p in paths:
        part = pd.read_parquet(p, columns=["block", "address"])
        rep(f"[info] loaded {p}: {len(part):,} rows, {part['block'].nunique()} blocks")
        parts.append(part)
    pairs = pd.concat(parts, ignore_index=True)
    pairs["address"] = pairs["address"].str.lower().str.strip()
    pairs["block"] = pairs["block"].astype("int64")
    pairs = pairs.drop_duplicates(subset=["block", "address"])
    n_missing = (~pairs["address"].isin(summary["address"])).sum()
    rep(f"[info] combined: {len(pairs):,} (block, address) pairs, {pairs['block'].nunique()} blocks, "
        f"{pairs['address'].nunique():,} distinct addresses; {n_missing:,} pairs without a summary row")
    cols = ["address", "type", "in_tags", "has_category", "has_concept", "has_any", "no_info",
            "best_category", "concept_list", "effective_label", "has_actor", "best_actor_label",
            "actor_concept_list", "has_abuse", "abuse_list"]
    pairs = pairs.merge(summary[cols], on="address", how="left")
    for c in ["in_tags", "has_category", "has_concept", "has_any", "has_actor", "has_abuse"]:
        pairs[c] = pairs[c].fillna(False).astype(bool)
    pairs["no_info"] = ~pairs["has_any"]
    pairs["type"] = pairs["type"].fillna("unknown")
    pairs["effective_label"] = pairs["effective_label"].fillna("UNLABELED")
    for c in ["concept_list", "actor_concept_list", "abuse_list"]:
        pairs[c] = pairs[c].map(lambda v: v if isinstance(v, list) else [])
    return pairs


# ----------------------------------------------------------------------------
# Distributions (identical column layout to the original block_tag_coverage.py)
# ----------------------------------------------------------------------------
def category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = df[df["has_category"]]
    vc = rows["best_category"].value_counts()
    out = vc.reset_index()
    out.columns = ["category", "count"]
    out["pct_of_categorized"] = (100 * out["count"] / max(len(rows), 1)).round(4)
    out["pct_of_all"] = (100 * out["count"] / total).round(4)
    return out


def concept_distribution(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = df[df["has_concept"]]
    vc = rows["concept_list"].explode().dropna().value_counts()
    out = vc.reset_index()
    out.columns = ["concept", "count"]
    out["pct_of_concept_addr"] = (100 * out["count"] / max(len(rows), 1)).round(4)
    out["pct_of_all"] = (100 * out["count"] / total).round(4)
    return out


def effective_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["effective_label"].value_counts()
    out = vc.reset_index()
    out.columns = ["effective_label", "count"]
    out["pct"] = (100 * out["count"] / len(df)).round(4)
    return out


def actor_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df["has_actor"]]
    vc = rows["best_actor_label"].value_counts()
    out = vc.reset_index()
    out.columns = ["actor_label", "count"]
    out["pct_of_actor_addr"] = (100 * out["count"] / max(len(rows), 1)).round(4)
    out["pct_of_all"] = (100 * out["count"] / len(df)).round(4)
    return out


def actor_concept_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df["has_actor"]]
    vc = rows["actor_concept_list"].explode().dropna().value_counts()
    out = vc.reset_index()
    out.columns = ["actor_concept", "count"]
    out["pct_of_actor_addr"] = (100 * out["count"] / max(len(rows), 1)).round(4)
    out["pct_of_all"] = (100 * out["count"] / len(df)).round(4)
    return out


def abuse_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df["has_abuse"]]
    vc = rows["abuse_list"].explode().dropna().value_counts()
    out = vc.reset_index()
    out.columns = ["abuse_concept", "count"]
    out["pct_of_abuse_addr"] = (100 * out["count"] / max(len(rows), 1)).round(4)
    out["pct_of_all"] = (100 * out["count"] / len(df)).round(4)
    return out


def coverage_row(df: pd.DataFrame, basis: str) -> Dict[str, float]:
    n = len(df)
    row: Dict[str, float] = {"basis": basis, "n": n}
    for c in ["in_tags", "has_category", "has_concept", "has_any", "has_actor", "has_abuse"]:
        k = int(df[c].sum())
        row[f"n_{c}"] = k
        row[f"pct_{c}"] = round(100 * k / n, 4)
    for t in ["CA", "EOA", "unknown"]:
        k = int((df["type"] == t).sum())
        row[f"n_{t}"] = k
        row[f"pct_{t}"] = round(100 * k / n, 4)
    cat = df[df["has_category"]]["best_category"]
    row["pct_categorized_exchange_defi"] = round(
        100 * cat.isin(EXCHANGE_DEFI_CATEGORIES).sum() / max(len(cat), 1), 4)
    return row


def coverage_by_type(df: pd.DataFrame, basis: str) -> pd.DataFrame:
    g = df.groupby("type").agg(
        n=("has_any", "size"),
        n_in_tags=("in_tags", "sum"), n_has_category=("has_category", "sum"),
        n_has_concept=("has_concept", "sum"), n_has_any=("has_any", "sum"),
        n_has_actor=("has_actor", "sum"), n_has_abuse=("has_abuse", "sum"),
    ).reset_index()
    for c in ["in_tags", "has_category", "has_concept", "has_any", "has_actor", "has_abuse"]:
        g[f"pct_{c}"] = (100 * g[f"n_{c}"] / g["n"]).round(4)
    g.insert(0, "basis", basis)
    return g


def per_block_coverage(pairs: pd.DataFrame) -> pd.DataFrame:
    g = pairs.groupby("block").agg(
        n_addresses=("address", "size"),
        n_in_tags=("in_tags", "sum"), n_has_category=("has_category", "sum"),
        n_has_concept=("has_concept", "sum"), n_has_any=("has_any", "sum"),
        n_no_info=("no_info", "sum"), n_has_actor=("has_actor", "sum"),
        n_has_abuse=("has_abuse", "sum"),
        n_CA=("type", lambda s: (s == "CA").sum()),
        n_EOA=("type", lambda s: (s == "EOA").sum()),
    )
    for c in ["in_tags", "has_category", "has_concept", "has_any", "no_info", "has_actor", "has_abuse", "CA", "EOA"]:
        g[f"pct_{c}"] = (100 * g[f"n_{c}"] / g["n_addresses"]).round(4)
    # coverage within each account type
    for t in ["CA", "EOA"]:
        sub = pairs[pairs["type"] == t].groupby("block")["has_any"].mean() * 100
        g[f"pct_has_any_{t}"] = sub.round(4)
    return g.reset_index()


def confidence_distribution(tags: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_rows = len(tags)
    for conf, cnt in tags["confidence"].fillna("NA").value_counts().items():
        rows.append({"table": "tag_rows_by_confidence", "key": conf, "count": int(cnt), "pct": round(100 * cnt / n_rows, 4)})
    for tt, cnt in tags["tag_type"].fillna("NA").value_counts().items():
        rows.append({"table": "tag_rows_by_tag_type", "key": tt, "count": int(cnt), "pct": round(100 * cnt / n_rows, 4)})
    for cr, cnt in tags["tagpack_creator"].fillna("NA").value_counts().items():
        rows.append({"table": "tag_rows_by_creator", "key": cr, "count": int(cnt), "pct": round(100 * cnt / n_rows, 4)})
    tagged = summary[summary["in_tags"]]
    for lvl, cnt in tagged["best_confidence_level"].fillna(-1).value_counts().sort_index().items():
        rows.append({"table": "tagged_addresses_by_best_confidence_level", "key": str(int(lvl)), "count": int(cnt),
                     "pct": round(100 * cnt / len(tagged), 4)})
    rows.append({"table": "tagged_addresses", "key": "n_tags_median", "count": int(tagged["n_tags"].median()), "pct": np.nan})
    rows.append({"table": "tagged_addresses", "key": "n_tags_mean", "count": round(float(tagged["n_tags"].mean()), 2), "pct": np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def print_basis_report(rep: Report, df: pd.DataFrame, basis: str, unit: str) -> None:
    total = len(df)
    rep(f"\n{SEP}\n  {basis.upper()}-WEIGHTED STATISTICS ({total:,} {unit})\n{SEP}")
    rep("  A. COVERAGE")
    for label, col in [("in tags", "in_tags"), ("has category", "has_category"), ("has concept", "has_concept"),
                       ("category OR concept", "has_any"), ("no info", "no_info"), ("has actor", "has_actor"),
                       ("abuse concept", "has_abuse")]:
        k = int(df[col].sum())
        rep(f"    {label:22s} {k:>10,} / {total:>10,}  ({100 * k / total:6.2f}%)")
    rep("\n  A2. COVERAGE BY ACCOUNT TYPE (share with category OR concept)")
    for t, g in df.groupby("type"):
        rep(f"    {t:8s} n={len(g):>10,}  in_tags={100 * g['in_tags'].mean():6.2f}%  "
            f"has_category={100 * g['has_category'].mean():6.2f}%  has_any={100 * g['has_any'].mean():6.2f}%  "
            f"has_actor={100 * g['has_actor'].mean():6.2f}%")

    cat = category_distribution(df)
    rep(f"\n  B. CATEGORY DISTRIBUTION ({int(df['has_category'].sum()):,} {unit} with a category; best_category)")
    rep(f"    {'category':28s} {'count':>10s} {'%categorized':>13s} {'%all':>8s}")
    for _, r in cat.head(15).iterrows():
        rep(f"    {r['category']:28s} {int(r['count']):>10,} {r['pct_of_categorized']:>12.2f}% {r['pct_of_all']:>7.2f}%")
    ex_defi = cat[cat["category"].isin(EXCHANGE_DEFI_CATEGORIES)]["pct_of_categorized"].sum()
    rep(f"    exchange- and DeFi-related categories: {ex_defi:.2f}% of categorized")

    con = concept_distribution(df)
    rep(f"\n  C. CONCEPT DISTRIBUTION ({int(df['has_concept'].sum()):,} {unit} with >=1 concept; multi-membership, sums > 100%)")
    rep(f"    {'concept':28s} {'count':>10s} {'%concept_addr':>14s} {'%all':>8s}")
    for _, r in con.head(15).iterrows():
        rep(f"    {r['concept']:28s} {int(r['count']):>10,} {r['pct_of_concept_addr']:>13.2f}% {r['pct_of_all']:>7.2f}%")

    eff = effective_label_distribution(df)
    rep(f"\n  D. EFFECTIVE LABEL (best_category > priority concept > UNLABELED)")
    for _, r in eff.head(12).iterrows():
        rep(f"    {r['effective_label']:28s} {int(r['count']):>10,} {r['pct']:>7.2f}%")

    act = actor_distribution(df)
    rep(f"\n  E. ACTORS ({int(df['has_actor'].sum()):,} {unit} with an actor attribution; best_actor_label)")
    for _, r in act.head(15).iterrows():
        rep(f"    {r['actor_label']:28s} {int(r['count']):>10,} {r['pct_of_actor_addr']:>7.2f}% of actor-attributed  {r['pct_of_all']:>6.2f}% of all")
    acc = actor_concept_distribution(df)
    rep("     actor concepts: " + ", ".join(f"{r['actor_concept']}={r['pct_of_actor_addr']:.1f}%" for _, r in acc.head(8).iterrows()))

    ab = abuse_distribution(df)
    rep(f"\n  F. ABUSE CONCEPTS ({int(df['has_abuse'].sum()):,} {unit}, {100 * df['has_abuse'].mean():.2f}% of all)")
    rep("     " + ", ".join(f"{r['abuse_concept']}={int(r['count']):,}" for _, r in ab.head(10).iterrows()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True, help="address_summary parquet (one row per distinct top-0.1%% address)")
    ap.add_argument("--tags", default=None, help="raw GraphSense tag rows parquet (optional, tag-quality table)")
    ap.add_argument("--addresses", nargs="*", default=None,
                    help="(block, address) parquet(s) of the top 0.1%% per snapshot (optional, enables pair-weighted stats)")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "data"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = Report()

    summary = load_summary(Path(args.summary))
    rep(f"[info] summary: {len(summary):,} distinct addresses; tagged {int(summary['in_tags'].sum()):,}; "
        f"contracts {int((summary['type'] == 'CA').sum()):,}; EOAs {int((summary['type'] == 'EOA').sum()):,}; "
        f"type unknown {int((summary['type'] == 'unknown').sum()):,}")

    summaries, by_type = [], []

    # ---- address-weighted (always) ----
    print_basis_report(rep, summary, "address", "distinct addresses")
    category_distribution(summary).to_csv(out_dir / "category_distribution_by_address.csv", index=False)
    concept_distribution(summary).to_csv(out_dir / "concept_distribution_by_address.csv", index=False)
    effective_label_distribution(summary).to_csv(out_dir / "effective_label_distribution_by_address.csv", index=False)
    actor_distribution(summary).to_csv(out_dir / "tag_actor_distribution_by_address.csv", index=False)
    actor_concept_distribution(summary).to_csv(out_dir / "tag_actor_concepts_by_address.csv", index=False)
    abuse_distribution(summary).to_csv(out_dir / "tag_abuse_distribution_by_address.csv", index=False)
    summaries.append(coverage_row(summary, "address"))
    by_type.append(coverage_by_type(summary, "address"))

    # ---- pair-weighted (with the (block, address) membership) ----
    if args.addresses:
        pairs = load_pairs(args.addresses, summary, rep)
        print_basis_report(rep, pairs, "pair", "block x address pairs")
        category_distribution(pairs).to_csv(out_dir / "category_distribution.csv", index=False)
        concept_distribution(pairs).to_csv(out_dir / "concept_distribution.csv", index=False)
        effective_label_distribution(pairs).to_csv(out_dir / "effective_label_distribution.csv", index=False)
        actor_distribution(pairs).to_csv(out_dir / "tag_actor_distribution.csv", index=False)
        actor_concept_distribution(pairs).to_csv(out_dir / "tag_actor_concepts.csv", index=False)
        abuse_distribution(pairs).to_csv(out_dir / "tag_abuse_distribution.csv", index=False)
        summaries.append(coverage_row(pairs, "pair"))
        by_type.append(coverage_by_type(pairs, "pair"))

        pb = per_block_coverage(pairs)
        pb.to_csv(out_dir / "tag_coverage_per_block.csv", index=False)
        rep(f"\n{SEP}\n  G. PER-SNAPSHOT COVERAGE ({len(pb)} blocks)\n{SEP}")
        rep(f"    {'block':>10s} {'n':>8s} {'%in_tags':>9s} {'%cat':>7s} {'%con':>7s} {'%any':>7s} {'%none':>7s} {'%CA':>7s} {'%any_CA':>8s} {'%any_EOA':>9s}")
        for _, r in pb.iterrows():
            rep(f"    {int(r['block']):>10d} {int(r['n_addresses']):>8,} {r['pct_in_tags']:>8.2f}% {r['pct_has_category']:>6.2f}% "
                f"{r['pct_has_concept']:>6.2f}% {r['pct_has_any']:>6.2f}% {r['pct_no_info']:>6.2f}% {r['pct_CA']:>6.2f}% "
                f"{r['pct_has_any_CA']:>7.2f}% {r['pct_has_any_EOA']:>8.2f}%")
        rep("    AVERAGES ACROSS SNAPSHOTS:")
        for c in ["pct_in_tags", "pct_has_category", "pct_has_concept", "pct_has_any", "pct_no_info", "pct_has_any_CA", "pct_has_any_EOA"]:
            s = pb[c]
            rep(f"      {c:18s} mean={s.mean():6.2f}%  median={s.median():6.2f}%  min={s.min():6.2f}%  max={s.max():6.2f}%")
    else:
        rep(f"\n[note] no --addresses given: pair-weighted statistics (the basis of the paper's numbers) and the "
            f"per-snapshot coverage table were NOT produced; category_distribution.csv, concept_distribution.csv and "
            f"effective_label_distribution.csv are left untouched.")

    pd.DataFrame(summaries).to_csv(out_dir / "tag_coverage_summary.csv", index=False)
    pd.concat(by_type, ignore_index=True).to_csv(out_dir / "tag_coverage_by_type.csv", index=False)

    if args.tags:
        tags = pd.read_parquet(args.tags)
        tags["address"] = tags["address"].str.lower().str.strip()
        conf = confidence_distribution(tags, summary)
        conf.to_csv(out_dir / "tag_confidence_distribution.csv", index=False)
        rep(f"\n{SEP}\n  H. TAG QUALITY ({len(tags):,} tag rows for {tags['address'].nunique():,} addresses)\n{SEP}")
        for tbl, g in conf.groupby("table", sort=False):
            rep("    " + tbl + ": " + ", ".join(f"{r['key']}={r['count']:,}" + (f" ({r['pct']:.1f}%)" if pd.notna(r['pct']) else "")
                                                 for _, r in g.head(12).iterrows()))

    (out_dir / "tag_attribution_report.txt").write_text(rep.buf.getvalue())
    rep(f"\n[saved] outputs in {out_dir.resolve()} (report: tag_attribution_report.txt)")


if __name__ == "__main__":
    main()
