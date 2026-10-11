#!/usr/bin/env python3
"""
diagnose_top_accounts.py

Description:
    Diagnostic for the wealth-concentration inputs: which accounts and which
    tokens carry the USD value at a snapshot block, and whether that value
    looks real. Runs directly on the reconstructed holdings (4_0 single
    parquet or 4_2 hive root) and reports, per snapshot block,

      A. block totals: accounts, accounts above 1 USD, total value, split by
         wallet type (EOA / CA)
      B. the top-N accounts by total_value_usd: type, number of assets, share
         of the block's value, cumulative share, and the largest positions
         (symbol and share of the account's value); known custody contracts
         are labelled (hints only, verify on Etherscan)
      C. the top-M tokens by aggregate value: holders, share of the block's
         value, largest holder and its share of the token's value, on-chain
         supply (sum of balances), implied price (value / supply) and, when
         --cg-metrics-csv is given, Coingecko circulating / total supply,
         price, market cap and FDV with the ratios on-chain/circulating and
         our-value/market-cap. The Coingecko figures are point-in-time (date
         of the 1_3 fetch), so those ratios are only meaningful for the
         snapshot closest to that date; the on-chain columns are exact for
         every block.
      D. the beacon deposit contract at every block, whether or not it makes
         the top-N: value, rank, ETH balance, share of the block's value

    Nothing big is held in Python: wallet rows stay in DuckDB and only the
    top-N accounts, their positions and the per-token aggregates come back.

Inputs:
    --parquet           4_0 parquet file or 4_2 hive root (block_number=N/...)
    --ca-tsv            contract addresses, first column (header allowed)
    --status-csv        optional: status_eth_tokens_3_1.csv (symbol, name,
                        contract_address) for symbols
    --cg-metrics-csv    optional: tokens_cg_metrics.csv (1_3 output) for
                        circulating/total supply, price, market cap, FDV;
                        its ETH row (empty contract_address) maps to 0xee..ee
    --labels-csv        optional: address,label to extend the custody labels
    --block-to-date-csv optional: block,date mapping for the headings

Outputs (in --out-dir):
    top_accounts_<block>.csv   one row per top account
    top_tokens_<block>.csv     one row per top token
    deposit_contract.csv       one row per block
    report.txt                 everything printed to stdout

Usage:
    python diagnose_top_accounts.py \\
        --parquet $DATA/recon-eth-erc20-partitioned \\
        --ca-tsv $DATA/contract_accounts_in_dataset.tsv \\
        --status-csv $DATA/status_eth_tokens_3_1.csv \\
        --cg-metrics-csv $DATA/tokens_cg_metrics.csv \\
        --blocks 9193266 11948960 18687851 21303934 23914921 \\
        --top-n 25 --top-tokens 25 --threads 32 --mem 200GB \\
        --out-dir ./diag_top_accounts
    Add --skip-tokens to leave out section C (the only full unnest of a block).
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import duckdb
import pandas as pd

ETH_PSEUDO_TOKEN = "0x" + "ee" * 20
WETH_TOKEN = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
DEPOSIT_CONTRACT = "0x00000000219ab540356cbb839cbe05303d7705fa"

# Hints only. Verify on Etherscan before relying on a label; extend with --labels-csv.
KNOWN_LABELS: Dict[str, str] = {
    DEPOSIT_CONTRACT: "Beacon deposit contract",
    WETH_TOKEN: "WETH contract",
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84": "Lido stETH contract",
    "0x8315177ab297ba92a06054ce80a67ed4dbd7ed3a": "Arbitrum One bridge",
    "0xbeb5fc579115071764c7423a4f12edde41f106ed": "Optimism portal",
    "0x49048044d57e1c92a77f79988d21fa8faf74e97e": "Base portal",
    "0x32400084c286cf3e17e7b677ea9583e60a000324": "zkSync Era diamond",
}

# 2020-01, 2021-03 (first snapshot with a >50 B EOA), 2023-12, 2024-12, 2025-12
DEFAULT_BLOCKS = [9193266, 11948960, 18687851, 21303934, 23914921]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)


# -----------------------------------------------------------------------------
# helpers (same conventions as analyze_portfolio_per_block.py)
# -----------------------------------------------------------------------------
def sql_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def normalize_address(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", "ignore")
    return str(x).strip().lower()


def get_duck(threads: int, mem: str, temp_dir: Optional[str]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit='{mem}'")
    con.execute("SET preserve_insertion_order=false")
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory={sql_quote(temp_dir)}")
    return con


def parquet_scan_target(path: str | Path) -> str:
    p = Path(path)
    if p.is_dir():
        return str(p.resolve() / "*" / "*.parquet")
    return str(p.resolve())


def load_ca_table(con: duckdb.DuckDBPyConnection, tsv_path: str | Path) -> int:
    addrs: Set[str] = set()
    with Path(tsv_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = normalize_address(line.split("\t")[0])
            if re.fullmatch(r"0x[0-9a-f]{40}", a):
                addrs.add(a)
    con.register("ca_df", pd.DataFrame({"address": sorted(addrs)}))
    con.execute("CREATE OR REPLACE TEMP TABLE ca AS SELECT address FROM ca_df")
    con.unregister("ca_df")
    return len(addrs)


def load_symbols(status_csv: Optional[str], cg_csv: Optional[str]) -> Dict[str, str]:
    sym: Dict[str, str] = {ETH_PSEUDO_TOKEN: "ETH"}
    for path in (cg_csv, status_csv):  # status wins when both are given
        if not path:
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "contract_address" not in df.columns or "symbol" not in df.columns:
            continue
        for a, s in zip(df["contract_address"], df["symbol"]):
            a = normalize_address(a)
            if re.fullmatch(r"0x[0-9a-f]{40}", a) and s:
                sym[a] = str(s)
    return sym


def load_cg_metrics(cg_csv: Optional[str]) -> pd.DataFrame:
    """tokens_cg_metrics.csv -> DataFrame indexed by token address."""
    cols = ["circulating_supply", "total_supply", "price_usd", "market_cap_usd", "fdv_usd", "cg_data_updated_at"]
    if not cg_csv:
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(cg_csv, dtype=str, keep_default_na=False)
    df["token"] = df["contract_address"].map(normalize_address)
    df.loc[df["token"] == "", "token"] = ETH_PSEUDO_TOKEN  # the ETH row
    df = df[df["token"].str.fullmatch(r"0x[0-9a-f]{40}")]
    for c in cols[:-1]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("token").set_index("token")[cols]


def load_labels(labels_csv: Optional[str]) -> Dict[str, str]:
    labels = dict(KNOWN_LABELS)
    if labels_csv:
        df = pd.read_csv(labels_csv, dtype=str, keep_default_na=False)
        for a, l in zip(df.iloc[:, 0], df.iloc[:, 1]):
            a = normalize_address(a)
            if re.fullmatch(r"0x[0-9a-f]{40}", a) and l:
                labels[a] = l
    return labels


def load_block_dates(path: Optional[str]) -> Dict[int, str]:
    if not path:
        return {}
    df = pd.read_csv(path)
    bcol = next((c for c in df.columns if "block" in c.lower()), None)
    dcol = next((c for c in df.columns if "date" in c.lower()), None)
    if bcol is None or dcol is None:
        return {}
    out = {}
    for b, d in zip(df[bcol], df[dcol]):
        try:
            out[int(b)] = str(pd.to_datetime(d).date())
        except Exception:
            pass
    return out


def usd_b(v: float) -> str:
    return f"{v / 1e9:,.2f}"


# -----------------------------------------------------------------------------
# per-block queries
# -----------------------------------------------------------------------------
def build_wallets(con: duckdb.DuckDBPyConnection, scan: str, block: int) -> None:
    """Temp table wt(addr, v, num_tokens, wallet_type) for one block."""
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE wt AS
    WITH base AS (
      SELECT lower(trim(wallet_address)) AS addr,
             GREATEST(COALESCE(TRY_CAST(total_value_usd AS DOUBLE), 0.0), 0.0) AS v,
             GREATEST(COALESCE(TRY_CAST(num_tokens AS BIGINT), 0), 0) AS num_tokens
      FROM read_parquet({sql_quote(scan)}, hive_partitioning=1)
      WHERE block_number = {int(block)} AND wallet_address IS NOT NULL
    )
    SELECT addr, v, num_tokens,
           CASE WHEN addr IN (SELECT address FROM ca) THEN 'CA' ELSE 'EOA' END AS wallet_type
    FROM base
    """)


def block_totals(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
    SELECT wallet_type, COUNT(*) AS accounts, COUNT(*) FILTER (WHERE v > 1) AS accounts_gt1_usd,
           SUM(v) AS value_usd
    FROM wt GROUP BY 1
    UNION ALL
    SELECT 'ALL', COUNT(*), COUNT(*) FILTER (WHERE v > 1), SUM(v) FROM wt
    """).df()
    df["wallet_type"] = pd.Categorical(df["wallet_type"], ["ALL", "EOA", "CA"])
    df = df.sort_values("wallet_type").reset_index(drop=True)
    total = float(df.loc[df["wallet_type"] == "ALL", "value_usd"].iloc[0])
    df["value_pct"] = 100.0 * df["value_usd"] / total if total > 0 else 0.0
    return df


def top_accounts(con: duckdb.DuckDBPyConnection, n: int) -> pd.DataFrame:
    return con.execute(f"""
    SELECT addr, wallet_type, num_tokens, v AS value_usd,
           (SELECT COUNT(*) FROM wt) AS n_accounts
    FROM wt ORDER BY v DESC LIMIT {int(n)}
    """).df()


def positions_of(con: duckdb.DuckDBPyConnection, scan: str, block: int, addrs: List[str]) -> pd.DataFrame:
    """Every position of the given accounts at the block (unnest restricted to them)."""
    con.register("sel_df", pd.DataFrame({"addr": sorted(set(addrs))}))
    df = con.execute(f"""
    SELECT addr, lower(h.token_address) AS token,
           COALESCE(h.value_usd, 0.0)::DOUBLE AS value_usd,
           TRY_CAST(h.balance AS DOUBLE) AS balance
    FROM (
      SELECT lower(trim(wallet_address)) AS addr, UNNEST(holdings) AS h
      FROM read_parquet({sql_quote(scan)}, hive_partitioning=1)
      WHERE block_number = {int(block)}
        AND lower(trim(wallet_address)) IN (SELECT addr FROM sel_df)
    )
    """).df()
    con.unregister("sel_df")
    return df


def token_aggregates(con: duckdb.DuckDBPyConnection, scan: str, block: int, m: int) -> pd.DataFrame:
    """Top-M tokens by aggregate value: one streaming unnest + group by."""
    return con.execute(f"""
    SELECT token,
           COUNT(*) FILTER (WHERE value_usd > 0)   AS holders_with_value,
           COUNT(*) FILTER (WHERE balance > 0)     AS holders,
           SUM(value_usd)                           AS value_usd,
           SUM(balance)                             AS onchain_supply,
           MAX(value_usd)                           AS max_holder_value_usd,
           arg_max(addr, value_usd)                 AS max_holder
    FROM (
      SELECT lower(trim(wallet_address)) AS addr, lower(h.token_address) AS token,
             COALESCE(h.value_usd, 0.0)::DOUBLE AS value_usd,
             COALESCE(TRY_CAST(h.balance AS DOUBLE), 0.0) AS balance
      FROM (
        SELECT wallet_address, UNNEST(holdings) AS h
        FROM read_parquet({sql_quote(scan)}, hive_partitioning=1)
        WHERE block_number = {int(block)} AND wallet_address IS NOT NULL
      )
    )
    GROUP BY token
    ORDER BY value_usd DESC
    LIMIT {int(m)}
    """).df()


def deposit_contract_row(con: duckdb.DuckDBPyConnection, pos: pd.DataFrame, block: int, total_value: float) -> dict:
    r = con.execute(f"""
    SELECT d.v, d.num_tokens, d.wallet_type,
           (SELECT COUNT(*) FROM wt WHERE v > d.v) + 1 AS rank
    FROM wt d WHERE addr = '{DEPOSIT_CONTRACT}'
    """).fetchone()
    eth = pos[(pos["addr"] == DEPOSIT_CONTRACT) & (pos["token"] == ETH_PSEUDO_TOKEN)]
    eth_bal = float(eth["balance"].iloc[0]) if len(eth) else 0.0
    eth_val = float(eth["value_usd"].iloc[0]) if len(eth) else 0.0
    if r is None:
        return dict(block=block, present=False, value_usd=0.0, rank=None, wallet_type=None,
                    num_tokens=0, eth_balance=0.0, eth_value_usd=0.0, pct_of_block_value=0.0)
    return dict(block=block, present=True, value_usd=float(r[0]), rank=int(r[3]), wallet_type=r[2],
                num_tokens=int(r[1]), eth_balance=eth_bal, eth_value_usd=eth_val,
                pct_of_block_value=100.0 * float(r[0]) / total_value if total_value > 0 else 0.0)


# -----------------------------------------------------------------------------
# assembling the tables
# -----------------------------------------------------------------------------
def describe_positions(pos: pd.DataFrame, sym: Dict[str, str], k: int = 3) -> pd.DataFrame:
    """Per account: dominant token and a 'SYM pct | SYM pct | ...' string of the top-k positions."""
    rows = []
    for addr, g in pos.groupby("addr"):
        g = g.sort_values("value_usd", ascending=False)
        tot = float(g["value_usd"].sum())
        top = g.iloc[0]
        parts = []
        for _, p in g.head(k).iterrows():
            s = sym.get(p["token"], p["token"][:10])
            pct = 100.0 * p["value_usd"] / tot if tot > 0 else 0.0
            parts.append(f"{s} {pct:.1f}%")
        bal = float(top["balance"]) if pd.notna(top["balance"]) else float("nan")
        rows.append(dict(
            addr=addr,
            n_positions=int(len(g)),
            n_positions_with_value=int((g["value_usd"] > 0).sum()),
            dominant_token=top["token"],
            dominant_symbol=sym.get(top["token"], ""),
            dominant_value_usd=float(top["value_usd"]),
            dominant_pct_of_account=100.0 * float(top["value_usd"]) / tot if tot > 0 else 0.0,
            dominant_balance=bal,
            dominant_implied_price=float(top["value_usd"]) / bal if bal and bal > 0 else float("nan"),
            positions=" | ".join(parts),
        ))
    return pd.DataFrame(rows)


def accounts_table(top: pd.DataFrame, pos: pd.DataFrame, sym: Dict[str, str], labels: Dict[str, str],
                   total_value: float, block: int) -> pd.DataFrame:
    d = describe_positions(pos[pos["addr"].isin(top["addr"])], sym)
    t = top.merge(d, on="addr", how="left")
    t.insert(0, "rank", range(1, len(t) + 1))
    t.insert(0, "block", block)
    t["label"] = t["addr"].map(labels).fillna("")
    t["pct_of_block_value"] = 100.0 * t["value_usd"] / total_value if total_value > 0 else 0.0
    t["cum_pct_of_block_value"] = t["pct_of_block_value"].cumsum()
    return t


def tokens_table(agg: pd.DataFrame, sym: Dict[str, str], labels: Dict[str, str], cg: pd.DataFrame,
                 total_value: float, block: int) -> pd.DataFrame:
    t = agg.copy()
    t.insert(0, "rank", range(1, len(t) + 1))
    t.insert(0, "block", block)
    t["symbol"] = t["token"].map(sym).fillna("")
    t["pct_of_block_value"] = 100.0 * t["value_usd"] / total_value if total_value > 0 else 0.0
    t["max_holder_pct_of_token"] = 100.0 * t["max_holder_value_usd"] / t["value_usd"].where(t["value_usd"] > 0)
    t["max_holder_label"] = t["max_holder"].map(labels).fillna("")
    t["implied_price_usd"] = t["value_usd"] / t["onchain_supply"].where(t["onchain_supply"] > 0)
    if len(cg):
        j = cg.reindex(t["token"]).reset_index(drop=True)
        t["cg_price_usd"] = j["price_usd"].values
        t["cg_circulating_supply"] = j["circulating_supply"].values
        t["cg_total_supply"] = j["total_supply"].values
        t["cg_market_cap_usd"] = j["market_cap_usd"].values
        t["cg_fdv_usd"] = j["fdv_usd"].values
        t["onchain_over_circulating"] = t["onchain_supply"] / t["cg_circulating_supply"].where(t["cg_circulating_supply"] > 0)
        t["value_over_cg_mcap"] = t["value_usd"] / t["cg_market_cap_usd"].where(t["cg_market_cap_usd"] > 0)
        flags = []
        for _, r in t.iterrows():
            f = []
            if pd.notna(r["value_over_cg_mcap"]) and r["value_over_cg_mcap"] > 2:
                f.append("VALUE>2xMCAP")
            if pd.notna(r["onchain_over_circulating"]) and r["onchain_over_circulating"] > 2:
                f.append("SUPPLY>2xCIRC")
            if pd.notna(r["max_holder_pct_of_token"]) and r["max_holder_pct_of_token"] > 50:
                f.append("TOPHOLDER>50%")
            flags.append(",".join(f))
        t["flags"] = flags
    else:
        t["flags"] = t["max_holder_pct_of_token"].map(lambda x: "TOPHOLDER>50%" if pd.notna(x) and x > 50 else "")
    return t


# -----------------------------------------------------------------------------
# report
# -----------------------------------------------------------------------------
def print_block(out: io.StringIO, block: int, date: str, totals: pd.DataFrame, acc: pd.DataFrame,
                tok: Optional[pd.DataFrame], dep: dict) -> None:
    w = out.write
    w("\n" + "=" * 120 + "\n")
    w(f"Block {block:,}{'  (' + date + ')' if date else ''}\n")
    w("=" * 120 + "\n")

    w("\nA. Block totals\n")
    a = totals.copy()
    a["value_B_usd"] = a["value_usd"].map(usd_b)
    a["value_pct"] = a["value_pct"].round(2)
    w(a[["wallet_type", "accounts", "accounts_gt1_usd", "value_B_usd", "value_pct"]].to_string(index=False) + "\n")

    w(f"\nB. Top {len(acc)} accounts by value (value in B USD; positions = share of the account's value)\n")
    b = acc.copy()
    b["value_B"] = b["value_usd"].map(usd_b)
    b["pct"] = b["pct_of_block_value"].round(2)
    b["cum_pct"] = b["cum_pct_of_block_value"].round(2)
    b["type"] = b["wallet_type"]
    b["n_assets"] = b["num_tokens"]
    w(b[["rank", "addr", "type", "label", "n_assets", "value_B", "pct", "cum_pct", "positions"]].to_string(index=False) + "\n")

    if tok is not None:
        w(f"\nC. Top {len(tok)} tokens by aggregate value (value in B USD; supply in token units)\n")
        c = tok.copy()
        c["value_B"] = c["value_usd"].map(usd_b)
        c["pct"] = c["pct_of_block_value"].round(2)
        c["top_holder_pct"] = c["max_holder_pct_of_token"].round(1)
        c["top_holder"] = c["max_holder"].str[:10] + c["max_holder_label"].map(lambda s: f" ({s})" if s else "")
        c["impl_price"] = c["implied_price_usd"].map(lambda x: f"{x:.6g}" if pd.notna(x) else "")
        cols = ["rank", "token", "symbol", "holders", "value_B", "pct", "top_holder", "top_holder_pct",
                "onchain_supply", "impl_price"]
        if "cg_market_cap_usd" in c.columns:
            c["cg_price"] = c["cg_price_usd"].map(lambda x: f"{x:.6g}" if pd.notna(x) else "")
            c["cg_mcap_B"] = c["cg_market_cap_usd"].map(lambda x: usd_b(x) if pd.notna(x) else "")
            c["cg_fdv_B"] = c["cg_fdv_usd"].map(lambda x: usd_b(x) if pd.notna(x) else "")
            c["supply/circ"] = c["onchain_over_circulating"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            c["value/mcap"] = c["value_over_cg_mcap"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            cols += ["cg_price", "cg_mcap_B", "cg_fdv_B", "supply/circ", "value/mcap"]
        cols += ["flags"]
        c["onchain_supply"] = c["onchain_supply"].map(lambda x: f"{x:,.0f}")
        w(c[cols].to_string(index=False) + "\n")

    w("\nD. Beacon deposit contract\n")
    if dep["present"]:
        w(f"   rank {dep['rank']:,} | type {dep['wallet_type']} | {dep['num_tokens']} assets | "
          f"value {usd_b(dep['value_usd'])} B USD = {dep['pct_of_block_value']:.2f}% of the block's value | "
          f"ETH balance {dep['eth_balance']:,.0f} ({usd_b(dep['eth_value_usd'])} B USD)\n")
    else:
        w("   not present at this block\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Top accounts / top tokens by value per snapshot block.")
    ap.add_argument("--parquet", required=True, help="4_0 parquet file or 4_2 hive root.")
    ap.add_argument("--ca-tsv", required=True, help="TSV with contract addresses (first column).")
    ap.add_argument("--status-csv", default=None, help="status_eth_tokens_3_1.csv for symbols.")
    ap.add_argument("--cg-metrics-csv", default=None, help="tokens_cg_metrics.csv (1_3 output).")
    ap.add_argument("--labels-csv", default=None, help="address,label CSV extending the custody labels.")
    ap.add_argument("--block-to-date-csv", default=None, help="block,date mapping for headings.")
    ap.add_argument("--blocks", type=int, nargs="*", default=DEFAULT_BLOCKS)
    ap.add_argument("--top-n", type=int, default=25, help="Accounts per block.")
    ap.add_argument("--top-tokens", type=int, default=25, help="Tokens per block.")
    ap.add_argument("--skip-tokens", action="store_true", help="Skip section C (full unnest of the block).")
    ap.add_argument("--out-dir", default="./diag_top_accounts")
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--mem", default="100GB")
    ap.add_argument("--temp-dir", default=None)
    args = ap.parse_args()

    for p in (args.parquet, args.ca_tsv, args.status_csv, args.cg_metrics_csv, args.labels_csv, args.block_to_date_csv):
        if p and not Path(p).exists():
            raise SystemExit(f"[preflight] not found: {p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scan = parquet_scan_target(args.parquet)
    con = get_duck(args.threads, args.mem, args.temp_dir)
    n_ca = load_ca_table(con, args.ca_tsv)
    sym = load_symbols(args.status_csv, args.cg_metrics_csv)
    cg = load_cg_metrics(args.cg_metrics_csv)
    labels = load_labels(args.labels_csv)
    dates = load_block_dates(args.block_to_date_csv)
    print(f"[config] parquet={scan}\n[config] {n_ca:,} contract addresses | {len(sym):,} symbols | "
          f"{len(cg):,} Coingecko rows | blocks={args.blocks}", flush=True)
    if len(cg):
        upd = cg["cg_data_updated_at"].dropna().astype(str)
        if len(upd):
            print(f"[config] Coingecko metrics dated {upd.iloc[0][:10]} (point-in-time; compare with the nearest block only)")

    out = io.StringIO()
    dep_rows = []
    for block in args.blocks:
        print(f"[block {block}] wallets ...", flush=True)
        build_wallets(con, scan, block)
        totals = block_totals(con)
        total_value = float(totals.loc[totals["wallet_type"] == "ALL", "value_usd"].iloc[0])
        top = top_accounts(con, args.top_n)
        print(f"[block {block}] positions of the top {len(top)} ...", flush=True)
        pos = positions_of(con, scan, block, list(top["addr"]) + [DEPOSIT_CONTRACT])
        acc = accounts_table(top, pos, sym, labels, total_value, block)
        acc.to_csv(out_dir / f"top_accounts_{block}.csv", index=False)
        dep = deposit_contract_row(con, pos, block, total_value)
        dep_rows.append(dep)
        tok = None
        if not args.skip_tokens:
            print(f"[block {block}] token aggregates ...", flush=True)
            tok = tokens_table(token_aggregates(con, scan, block, args.top_tokens), sym, labels, cg, total_value, block)
            tok.to_csv(out_dir / f"top_tokens_{block}.csv", index=False)
        blk = io.StringIO()
        print_block(blk, block, dates.get(block, ""), totals, acc, tok, dep)
        print(blk.getvalue(), flush=True)   # one section per block as soon as it is ready
        out.write(blk.getvalue())
        con.execute("DROP TABLE wt")

    pd.DataFrame(dep_rows).to_csv(out_dir / "deposit_contract.csv", index=False)
    (out_dir / "report.txt").write_text(out.getvalue(), encoding="utf-8")
    print(f"[ok] wrote {out_dir.resolve()} (report.txt, top_accounts_<block>.csv, top_tokens_<block>.csv, deposit_contract.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
