#!/usr/bin/env python3
"""
generate_all_macros.py

Description:
    Rebuild every LaTeX macro used in the paper from pre-computed
    CSV / Markdown artefacts under Analytics/. Each topic owns its macros
    in a <topic>_macros.py module hooked from build(); the distance and
    return macros live in distance_return_analytics/src/
    distance_return_macros.py. The few facts with no data file in
    Analytics/ (CoinGecko universe, reconstruction pipeline, compute
    details of the reconstruction and MPT runs) are loaded from
    manual_constants.py.

    The generated file mirrors the section ordering of the paper's
    macro.tex so that diffs stay legible.

Output:
    Analytics/macro.tex

Usage:
    python Analytics/generate_all_macros.py            # write macro.tex
    python Analytics/generate_all_macros.py --verify   # compare against target
    python Analytics/generate_all_macros.py --verbose  # debug output

Verification target:
    --verify compares the generated file against a reference macro.tex.
    The path is resolved in this order:
        1. --target-tex CLI flag
        2. MACRO_TARGET_TEX env var
        3. ../BlockchainPortfolio/macro.tex (default, the paper's LaTeX
           source — not included in this public repo)
    Verification is skipped with a clear message if the target file does
    not exist.
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import re
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
BASE = pathlib.Path(__file__).resolve().parent.parent
ANALYTICS = BASE / "Analytics"
OUT_TEX = ANALYTICS / "macro.tex"
DEFAULT_TARGET_TEX = BASE / "BlockchainPortfolio" / "macro.tex"


def resolve_target_tex(cli_arg: Optional[str]) -> pathlib.Path:
    """Return the verification target path from CLI arg, env var, or default."""
    if cli_arg:
        return pathlib.Path(cli_arg).expanduser()
    env_val = os.environ.get("MACRO_TARGET_TEX")
    if env_val:
        return pathlib.Path(env_val).expanduser()
    return DEFAULT_TARGET_TEX

# Data folders
DRA_DATA = ANALYTICS / "distance_return_analytics" / "data"
PC = ANALYTICS / "portfolio_characteristics"
ACC_DATA = PC / "accounts_overview" / "data"
DOW_DATA = PC / "distribution_of_wealth" / "data"
PSZ_DATA = PC / "portfolio_size" / "data"
PBS_DATA = PC / "per_block_wallet_stats" / "data"
THC_DATA = PC / "token_holder_counts" / "data"
ALC_DATA = PC / "allocation" / "data"

# ---------------------------------------------------------------------------
# Macro emitter helper
# ---------------------------------------------------------------------------
class MacroEmitter:
    """Collect raw text lines (macros and comments) in order, then assemble."""

    def __init__(self):
        self.lines: List[str] = []
        self.macro_names: "OrderedDict[str, int]" = OrderedDict()
        self.macro_seen: Dict[str, str] = {}  # name -> raw value text

    def comment(self, text: str = "") -> None:
        for line in text.split("\n"):
            self.lines.append(line)

    def blank(self) -> None:
        self.lines.append("")

    def macro(self, name: str, value: str, slash: bool = True) -> None:
        """Emit `\def\<name>/{<value>}`.

        If `slash=False`, emit `\def\<name>{<value>}` (no trailing slash —
        a few legacy macros like \HoldersAvgYtwenty use this form).
        """
        sep = "/" if slash else ""
        line = rf"\def\{name}{sep}{{{value}}}"
        self.lines.append(line)
        # Remember last-seen value (handle redefines like \LOneSmall*).
        if name not in self.macro_seen:
            self.macro_names[name] = len(self.lines) - 1
        self.macro_seen[name] = value

    def macro_raw(self, line: str) -> None:
        """Emit a raw \def line verbatim (used for tricky cases).

        Strips any trailing inline comment.
        """
        m = re.match(r"\\def\\([A-Za-z][A-Za-z0-9]*)/?\{(.*)\}", line)
        # Drop any trailing "  % comment" so generated output stays clean.
        clean = re.sub(r"\}\s*%.*$", "}", line)
        self.lines.append(clean)
        if m:
            name = m.group(1)
            if name not in self.macro_seen:
                self.macro_names[name] = len(self.lines) - 1
            self.macro_seen[name] = m.group(2)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


# ---------------------------------------------------------------------------
# Number formatters
# ---------------------------------------------------------------------------
def fmt_num_int(v) -> str:
    """`\num{12345}`."""
    return rf"\num{{{int(round(float(v)))}}}"


def fmt_si_pct(v, dp: int = 2) -> str:
    """`\SI{12.34}{\percent}`."""
    return rf"\SI{{{round(float(v), dp):.{dp}f}}}{{\percent}}"


def fmt_pct(v, dp: int = 1) -> str:
    """`12.3\%` or similar."""
    return rf"\num{{{round(float(v), dp):.{dp}f}}}\%"


def fmt_dec(v, dp: int) -> str:
    return f"{round(float(v), dp):.{dp}f}"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Section emitters (in order of appearance in macro.tex)
# ---------------------------------------------------------------------------

def emit_token_metadata(emit: MacroEmitter, manual: Dict[str, str]) -> None:
    """\Coingecko* macros + total accounts head."""
    emit.comment("% --- Token metadata (Coingecko universe) ---")
    emit.macro("CoingeckoInitTokensN", manual["CoingeckoInitTokensN"])
    emit.macro("CoingeckoERCTokensN", manual["CoingeckoERCTokensN"])
    emit.macro("CoingeckoCleanedTokensN", manual["CoingeckoCleanedTokensN"])
    emit.blank()


def emit_accounts_overview(emit: MacroEmitter) -> None:
    """Account-count macros (\\TotalAccounts*, \\EOA*, \\CA*).

    Owned by portfolio_characteristics/accounts_overview/src/
    accounts_overview_macros.py; this is only the hook into the build
    order. ANALYTICS is on sys.path (see build()).
    """
    from portfolio_characteristics.accounts_overview.src.accounts_overview_macros import (
        emit_accounts_overview_macros,
    )
    emit_accounts_overview_macros(emit, ACC_DATA)


def emit_holders(emit: MacroEmitter) -> None:
    """Token-holder-count macros (\\Holders*, legacy form without trailing slash).

    Owned by portfolio_characteristics/token_holder_counts/src/
    token_holder_counts_macros.py; this is only the hook into the build
    order. ANALYTICS is on sys.path (see build()).
    """
    from portfolio_characteristics.token_holder_counts.src.token_holder_counts_macros import (
        emit_token_holder_counts_macros,
    )
    emit_token_holder_counts_macros(emit, THC_DATA)



def emit_distribution_of_wealth(emit: MacroEmitter) -> None:
    """Wealth-distribution macros (\\WealthBucket*, \\WealthShare*, wealth
    composition vs DeFiLlama TVL, \\TopOneWealth*, \\TopOneCA*/\\TopPointOne*).

    Owned by portfolio_characteristics/distribution_of_wealth/src/
    distribution_of_wealth_macros.py; this is only the hook into the build
    order. ANALYTICS is on sys.path (see build()).
    """
    from portfolio_characteristics.distribution_of_wealth.src.distribution_of_wealth_macros import (
        emit_distribution_of_wealth_macros,
    )
    emit_distribution_of_wealth_macros(emit, DOW_DATA)


def emit_portfolio_size_distribution(emit: MacroEmitter) -> None:
    """Portfolio-size macros (\\PortSize*).

    Owned by portfolio_characteristics/portfolio_size/src/portfolio_size_macros.py
    so the analysis folder carries its own macro definitions; this is only
    the hook into the build order. ANALYTICS is on sys.path (see build()).
    """
    from portfolio_characteristics.portfolio_size.src.portfolio_size_macros import (
        emit_portfolio_size_macros,
    )
    emit_portfolio_size_macros(emit, PSZ_DATA)


def emit_per_block_wallet_stats(emit: MacroEmitter) -> None:
    """Per-block wallet-statistics macros (\\Acct*).

    Owned by portfolio_characteristics/per_block_wallet_stats/src/
    per_block_wallet_stats_macros.py; this is only the hook into the build
    order. ANALYTICS is on sys.path (see build()).
    """
    from portfolio_characteristics.per_block_wallet_stats.src.per_block_wallet_stats_macros import (
        emit_per_block_wallet_stats_macros,
    )
    emit_per_block_wallet_stats_macros(emit, PBS_DATA)


def emit_allocation(emit: MacroEmitter) -> None:
    """Allocation macros (\\Eff*, \\Trade*): effective diversification, dust
    and trading activity.

    Owned by portfolio_characteristics/allocation/src/allocation_macros.py;
    this is only the hook into the build order. ANALYTICS is on sys.path
    (see build()).
    """
    from portfolio_characteristics.allocation.src.allocation_macros import (
        emit_allocation_macros,
    )
    emit_allocation_macros(emit, ALC_DATA)


def emit_graphsense_tags(emit: MacroEmitter, manual: Dict[str, str]) -> None:
    """GraphSense / iknaio tag attribution of the top-0.1 % holders.

    Every CSV comes from portfolio_characteristics/distribution_of_wealth/src/
    block_tag_coverage.py, which has two weighting bases:
      pair     one observation per (block, address) pair -- the basis of the
               paper's numbers; needs the cluster-side (block, address) parquet
               (files without suffix, plus tag_coverage_per_block.csv);
      address  one observation per distinct top-0.1 % address
               (files with the _by_address suffix), always available.
    Pair-level macros keep the original names (\\TagUnlabeledPct/, \\TagCat*/,
    \\TagCon*/, \\TagCoveragePct*/); address-level macros carry the suffix
    ByAddr. Coverage mean/min/max across snapshots come from
    tag_coverage_per_block.csv.
    """
    exchange_defi = ["exchange", "defi", "defi_dex_pair", "defi_token", "defi_dex", "defi_lending",
                     "defi_bridge", "defi_staking", "defi_dao", "defi_derivative"]
    cat_map = {
        "TagCatUserPct":       "user",
        "TagCatDexPairPct":    "defi_dex_pair",
        "TagCatServicePct":    "service",
        "TagCatExchangePct":   "exchange",
        "TagCatDefiTokenPct":  "defi_token",
        "TagCatAbusePct":      "abuse",
    }
    con_map = {
        "TagConDefiPct":       "defi",
        "TagConCustodyPct":    "defi_custody",
        "TagConExchangePct":   "exchange",
        "TagConDexPct":        "defi_dex",
        "TagConDepositPct":    "deposit_wallet",
    }
    bases = [("", "", "pair"), ("ByAddr", "_by_address", "address")]
    present = []
    for suffix, fsuf, basis in bases:
        files = [f"effective_label_distribution{fsuf}.csv", f"category_distribution{fsuf}.csv",
                 f"concept_distribution{fsuf}.csv"]
        if all((DOW_DATA / f).exists() for f in files):
            present.append((suffix, fsuf, basis))
    if not present:
        print(f"[generate_all_macros] GraphSense tag macros skipped: no block_tag_coverage.py outputs in {DOW_DATA}",
              file=sys.stderr)
        emit.comment("% --- GraphSense tag coverage: skipped (tag-coverage CSVs not present) ---")
        emit.blank()
        return

    def _pct(df: pd.DataFrame, key_col: str, key: str, val_col: str, dp: int = 2) -> str:
        sub = df.loc[df[key_col] == key, val_col]
        return fmt_dec(sub.iloc[0], dp) if len(sub) else fmt_dec(0.0, dp)

    cov_sum = pd.read_csv(DOW_DATA / "tag_coverage_summary.csv") if (DOW_DATA / "tag_coverage_summary.csv").exists() else None
    cov_typ = pd.read_csv(DOW_DATA / "tag_coverage_by_type.csv") if (DOW_DATA / "tag_coverage_by_type.csv").exists() else None

    for suffix, fsuf, basis in present:
        unit = "block x address pairs" if basis == "pair" else "distinct addresses"
        emit.comment(f"% --- GraphSense tag attribution of top-0.1% holders, {basis}-weighted ({unit}; macros {'without suffix' if not suffix else 'suffixed ' + suffix}) ---")

        if basis == "pair":
            # Coverage across snapshots (pair-weighted per-block table).
            pb = pd.read_csv(DOW_DATA / "tag_coverage_per_block.csv")
            emit.macro("TagCoveragePctMean", fmt_dec(pb["pct_has_any"].mean(), 2))
            emit.macro("TagCoveragePctMin",  fmt_dec(pb["pct_has_any"].min(), 2))
            emit.macro("TagCoveragePctMax",  fmt_dec(pb["pct_has_any"].max(), 2))
            if "pct_has_any_CA" in pb.columns:
                for typ in ["CA", "EOA"]:
                    col = pb[f"pct_has_any_{typ}"]
                    emit.macro(f"TagCoverage{typ}PctMean", fmt_dec(col.mean(), 2))
                    emit.macro(f"TagCoverage{typ}PctMin",  fmt_dec(col.min(), 2))
                    emit.macro(f"TagCoverage{typ}PctMax",  fmt_dec(col.max(), 2))
            emit.macro("TagSnapshotsN", str(len(pb)))

        # Observation counts and coverage shares (tag_coverage_summary.csv).
        if cov_sum is not None and (cov_sum["basis"] == basis).any():
            r = cov_sum.loc[cov_sum["basis"] == basis].iloc[0]
            emit.macro(f"TagObsN{suffix}", r"\num{%d}" % int(r["n"]))
            emit.macro(f"TagAttributedPct{suffix}",  fmt_dec(r["pct_has_any"], 2))       # category or concept
            emit.macro(f"TagCategorisedPct{suffix}", fmt_dec(r["pct_has_category"], 2))
            emit.macro(f"TagConceptPct{suffix}",     fmt_dec(r["pct_has_concept"], 2))
            emit.macro(f"TagActorPct{suffix}",       fmt_dec(r["pct_has_actor"], 2))
            emit.macro(f"TagAbusePct{suffix}",       fmt_dec(r["pct_has_abuse"], 2))
            emit.macro(f"TagContractPct{suffix}",    fmt_dec(r["pct_CA"], 2))            # is_contract flag of the tag lookup
            emit.macro(f"TagCatExchangeDefiPct{suffix}", fmt_dec(r["pct_categorized_exchange_defi"], 1))
        if cov_typ is not None:
            sub = cov_typ[cov_typ["basis"] == basis]
            emit.macro(f"TagCoverageCAPct{suffix}",  _pct(sub, "type", "CA",  "pct_has_any"))
            emit.macro(f"TagCoverageEOAPct{suffix}", _pct(sub, "type", "EOA", "pct_has_any"))

        # \TagUnlabeledPct/ -- share of top-0.1% observations with neither category nor concept.
        eff = pd.read_csv(DOW_DATA / f"effective_label_distribution{fsuf}.csv")
        emit.macro(f"TagUnlabeledPct{suffix}", _pct(eff, "effective_label", "UNLABELED", "pct"))
        emit.blank()

        # Category-level shares (within categorised observations only).
        cat = pd.read_csv(DOW_DATA / f"category_distribution{fsuf}.csv")
        for macro_name, key in cat_map.items():
            emit.macro(f"{macro_name}{suffix}", _pct(cat, "category", key, "pct_of_categorized"))
        emit.blank()

        # Concept-level shares (multi-membership, across the GraphSense concept hierarchy).
        con = pd.read_csv(DOW_DATA / f"concept_distribution{fsuf}.csv")
        for macro_name, key in con_map.items():
            emit.macro(f"{macro_name}{suffix}", _pct(con, "concept", key, "pct_of_concept_addr"))
        emit.blank()

        # Actor view: the three most frequent actor attributions and the exchange share.
        act_path = DOW_DATA / f"tag_actor_distribution{fsuf}.csv"
        if act_path.exists():
            act = pd.read_csv(act_path)
            for rank, name in enumerate(["First", "Second", "Third"]):
                if rank < len(act):
                    emit.macro(f"TagActor{name}{suffix}", str(act.at[rank, "actor_label"]))
                    emit.macro(f"TagActor{name}Pct{suffix}", fmt_dec(act.at[rank, "pct_of_actor_addr"], 1))
            acc_path = DOW_DATA / f"tag_actor_concepts{fsuf}.csv"
            if acc_path.exists():
                acc = pd.read_csv(acc_path)
                emit.macro(f"TagActorExchangePct{suffix}", _pct(acc, "actor_concept", "exchange", "pct_of_actor_addr", 1))
                emit.macro(f"TagActorDexPct{suffix}",      _pct(acc, "actor_concept", "defi_dex", "pct_of_actor_addr", 1))
            emit.blank()


def emit_recon_config(emit: MacroEmitter, manual: Dict[str, str]) -> None:
    """Reconstruction-pipeline facts and the compute details of the
    reconstruction and MPT runs. They come from the pipeline runs on the
    cluster, not from a file in Analytics/, so they stay in manual_constants.py;
    the snapshot count comes from the per-block wallet statistics."""
    emit.comment("% --- Portfolio Reconstruction pipeline: data sizes + cluster config ---")
    for k in ["ReconTransferEvents", "ReconTokensDiscovered", "ReconParquetSizeGB",
              "ReconTopTokenTransfers", "ReconTopTokenSharePct", "ReconMedianTokenTransfers",
              "ReconTokensAfterFilter", "ReconPipelineWorkers", "ReconBatchSize",
              "ReconRuns", "ReconOutputRows", "ReconOutputSizeGB",
              "ReconRowsFirstSnapshotM", "ReconRowsLastSnapshotM"]:
        emit.macro(k, manual[k])
    blocks = pd.read_csv(PBS_DATA / "averages_per_block.csv")["block"].nunique()
    emit.macro("ReconSnapshotBlocks", str(blocks))
    emit.blank()
    emit.comment("% --- MPT pipeline: cluster config, optimisation input (5_0 eligibility rule) and output size ---")
    for k in ["MptWorkers", "MptCores", "MptMultiAssetRowsM", "MptInputObservationsM",
              "MptInputMinWalletsPerBlock", "MptInputMedianWalletsPerBlock", "MptInputMaxWalletsPerBlock",
              "MptOptimisedSharePct", "MptOutputSizeGB"]:
        emit.macro(k, manual[k])
    emit.blank()


def emit_distance_return(emit: MacroEmitter) -> None:
    """Distance and return macros (\\LOne*, \\CAPM*, \\PctBeat*, \\Beat*, \\Fit*,
    \\RfExt*, \\Mpt*, ...), all computed from distance_return_analytics/data.

    Owned by distance_return_analytics/src/distance_return_macros.py; this is
    only the hook into the build order. ANALYTICS is on sys.path (see build()).
    """
    from distance_return_analytics.src.distance_return_macros import (
        emit_distance_return_macros,
    )
    emit_distance_return_macros(emit, DRA_DATA)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class _ManualLookup(dict):
    """Dict subclass that returns "" for missing keys.

    manual_constants.py only carries the entries we actually emit; this
    keeps the lookup tolerant if a key was pruned without breaking the
    rest of the build.
    """

    def __getitem__(self, key):
        return super().get(key, "")


def build(verbose: bool = False) -> str:
    sys.path.insert(0, str(ANALYTICS))
    from manual_constants import MANUAL_CONSTANTS as _RAW_MC  # local import
    MC = _ManualLookup(_RAW_MC)

    e = MacroEmitter()
    emit_token_metadata(e, MC)
    emit_accounts_overview(e)
    emit_holders(e)
    emit_distribution_of_wealth(e)
    emit_portfolio_size_distribution(e)
    emit_per_block_wallet_stats(e)
    emit_allocation(e)
    emit_graphsense_tags(e, MC)
    emit_recon_config(e, MC)
    emit_distance_return(e)

    return e.text()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
# Allow digits in macro names (e.g. \LOneMs60to80Pct/, \WealthShareCAP10P/).
# Use greedy (.*) for the value so nested braces (\num{X}) are captured.
DEF_RE = re.compile(r"\\def\\([A-Za-z][A-Za-z0-9]*)/?\{(.*)\}\s*(?:%.*)?$")


def parse_macros(text: str) -> "OrderedDict[str, str]":
    """Last-wins dict of macro_name -> raw value text, plus a list of duplicates."""
    out: "OrderedDict[str, str]" = OrderedDict()
    for line in text.splitlines():
        m = DEF_RE.match(line.strip())
        if not m:
            continue
        name, raw = m.group(1), m.group(2)
        out[name] = raw
    return out


def normalise_value(raw: str) -> Optional[str]:
    """Strip wrappers and return a normalised string for byte-equal comparison."""
    s = raw.strip()
    # Strip \num{...}
    while True:
        m = re.match(r"^\\num\{(.*)\}$", s)
        if not m:
            break
        s = m.group(1).strip()
    # Strip \SI{X}{\percent} → "X" (mark percent)
    m = re.match(r"^\\SI\{([^}]+)\}\{\\percent\}$", s)
    is_pct = False
    if m:
        s = m.group(1).strip()
        is_pct = True
    # Strip trailing \% / %
    if s.endswith(r"\%"):
        s = s[:-2].strip()
        is_pct = True
    elif s.endswith("%"):
        s = s[:-1].strip()
        is_pct = True
    return s + ("%" if is_pct else "")


def try_float(s: str) -> Optional[float]:
    s2 = s.rstrip("%").replace(",", "").replace("+", "").strip()
    if s2 == "":
        return None
    try:
        return float(s2)
    except ValueError:
        return None


def compare_values(expected: str, generated: str) -> Tuple[str, str, str]:
    """Return (status, exp_norm, gen_norm).
    Status: EXACT, NUMERIC_MATCH, ROUND_MATCH, MISMATCH."""
    en = normalise_value(expected)
    gn = normalise_value(generated)
    if en is None or gn is None:
        return ("MISMATCH", str(en), str(gn))
    if en == gn:
        return ("EXACT", en, gn)
    # Numeric tolerance
    ef = try_float(en)
    gf = try_float(gn)
    if ef is not None and gf is not None:
        if math.isclose(ef, gf, rel_tol=1e-3, abs_tol=1e-9):
            return ("NUMERIC_MATCH", en, gn)
        # ROUND_MATCH: same rounded display at some dp ∈ {0,1,2,3},
        # AND both the absolute and relative diffs are small. The
        # relative cap prevents pathological cases like 2.33 vs 1.89
        # (both round to 2 at dp=0 but the drift is 19%) from being
        # classified as "rounding noise".
        rel = abs(ef - gf) / max(abs(ef), 1e-9)
        for dp in (0, 1, 2, 3):
            if round(ef, dp) == round(gf, dp):
                if abs(ef - gf) <= max(0.5, abs(ef) * 0.02) and rel <= 0.05:
                    return ("ROUND_MATCH", en, gn)
        return ("MISMATCH", en, gn)
    # Strings: case-insensitive whitespace-collapsed
    en2 = re.sub(r"\s+", " ", en.strip())
    gn2 = re.sub(r"\s+", " ", gn.strip())
    if en2 == gn2:
        return ("EXACT", en, gn)
    return ("MISMATCH", en, gn)


def verify(generated_text: str, target_tex: pathlib.Path) -> Tuple[List[str], Dict[str, int]]:
    target_text = target_tex.read_text()
    expected = parse_macros(target_text)
    actual = parse_macros(generated_text)

    counts = {
        "EXACT": 0, "NUMERIC_MATCH": 0, "ROUND_MATCH": 0,
        "MISMATCH": 0, "MISSING_FROM_GENERATED": 0,
        "EXTRA_IN_GENERATED": 0,
    }
    lines: List[str] = []
    seen_names = set()

    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GREY = "\033[90m"
    RESET = "\033[0m"

    for name, exp_raw in expected.items():
        seen_names.add(name)
        if name not in actual:
            counts["MISSING_FROM_GENERATED"] += 1
            lines.append(f"{YELLOW}[MISSING]{RESET}     \\{name}/  expected={exp_raw}")
            continue
        gen_raw = actual[name]
        status, en, gn = compare_values(exp_raw, gen_raw)
        counts[status] += 1
        if status == "EXACT":
            lines.append(f"{GREEN}[EXACT  ]{RESET}     \\{name}/  = {en}")
        elif status == "NUMERIC_MATCH":
            lines.append(f"{YELLOW}[NUMERIC]{RESET}     \\{name}/  expected={en}  generated={gn}")
        elif status == "ROUND_MATCH":
            lines.append(f"{YELLOW}[ROUND  ]{RESET}     \\{name}/  expected={en}  generated={gn}")
        else:
            lines.append(f"{RED}[MISMATCH]{RESET}    \\{name}/  expected={en}  generated={gn}")

    for name in actual:
        if name not in seen_names:
            counts["EXTRA_IN_GENERATED"] += 1
            lines.append(f"{GREY}[EXTRA  ]{RESET}     \\{name}/  generated={actual[name]}")

    total = sum(counts.values())
    summary = []
    summary.append("")
    summary.append("=" * 78)
    summary.append("VERIFICATION SUMMARY")
    summary.append("=" * 78)
    summary.append(f"  Total macros (expected):      {len(expected)}")
    summary.append(f"  Total macros (generated):     {len(actual)}")
    summary.append(f"  EXACT:                        {counts['EXACT']}")
    summary.append(f"  NUMERIC_MATCH (rel-tol 1e-3): {counts['NUMERIC_MATCH']}")
    summary.append(f"  ROUND_MATCH:                  {counts['ROUND_MATCH']}")
    summary.append(f"  MISMATCH:                     {counts['MISMATCH']}")
    summary.append(f"  MISSING_FROM_GENERATED:       {counts['MISSING_FROM_GENERATED']}")
    summary.append(f"  EXTRA_IN_GENERATED:           {counts['EXTRA_IN_GENERATED']}")

    fail_n = counts["MISMATCH"] + counts["MISSING_FROM_GENERATED"]
    if fail_n == 0:
        summary.append(f"  RESULT: PASS (every macro present and within tolerance)")
    else:
        summary.append(f"  RESULT: FAIL ({fail_n} issue(s) need fixing)")
    summary.append("=" * 78)

    return lines + summary, counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true", help="run verification against target macro.tex")
    p.add_argument(
        "--target-tex",
        default=None,
        help=(
            "Path to reference macro.tex for --verify. Overrides the "
            "MACRO_TARGET_TEX env var. Defaults to ../BlockchainPortfolio/macro.tex."
        ),
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    txt = build(verbose=args.verbose)
    OUT_TEX.write_text(txt)
    print(f"[generate_all_macros] wrote {OUT_TEX} ({len(txt.splitlines())} lines)")

    if args.verify:
        target_tex = resolve_target_tex(args.target_tex)
        if not target_tex.exists():
            print(
                f"[generate_all_macros] --verify skipped: target file not found at {target_tex}\n"
                "  Set MACRO_TARGET_TEX or pass --target-tex to point at your reference macro.tex.",
                file=sys.stderr,
            )
            return 0
        report, counts = verify(txt, target_tex)
        print("\n".join(report) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
