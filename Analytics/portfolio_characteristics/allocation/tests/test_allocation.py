#!/usr/bin/env python3
"""
test_allocation.py

Description:
    End-to-end check of the three allocation scripts on synthetic holdings in
    the exact reconstruction schema (4_2 hive root and 4_0 single file).
    Every reported number is recomputed with an independent pandas
    implementation of the definitions and compared. Runs locally in seconds:

        python tests/test_allocation.py            # from the allocation folder
        python tests/test_allocation.py --keep     # keep the work directory

    Synthetic data covers: ETH and WETH held together, unpriced tokens, a token
    that loses its price between snapshots, zero balances, the beacon deposit
    contract, contract accounts, unchanged accounts, gas-only ETH changes,
    swaps, exits and entries.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))
from _alloc_common import (  # noqa: E402
    BEACON_DEPOSIT_CONTRACT, BTC_WRAPPER_TOKENS, DEFAULT_STABLECOINS_CSV, ETH_DERIVATIVE_TOKENS,
    ETH_PSEUDO_TOKEN, WETH_TOKEN, load_cash_tokens,
)

RNG = np.random.default_rng(20260928)
ETH, WETH = ETH_PSEUDO_TOKEN, WETH_TOKEN
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
DAI = "0x6b175474e89094c44da98b954eedeac495271d0f"
SUSDE = "0x9d39a5de30e57443bff2a8307a4256c8797a3497"   # extended cash only
EURC = "0x1abaea1f7c830bd89acc67ec4af516284b1bc33c"    # non-USD fiat
WBTC = BTC_WRAPPER_TOKENS[0]
WSTETH = ETH_DERIVATIVE_TOKENS[0]


def rand_addr() -> str:
    return "0x" + "".join(RNG.choice(list("0123456789abcdef"), 40))


RISKY = [rand_addr() for _ in range(6)]
UNPRICED = rand_addr()          # never has a price
FLAKY = RISKY[5]                # priced at block 100, unpriced at 200
TOKENS = [ETH, WETH, USDC, USDT, DAI, SUSDE, EURC, WBTC, WSTETH] + RISKY + [UNPRICED]
BLOCKS = [100, 200, 300]
CAP = 0.9
TOL = 1e-7


# -----------------------------------------------------------------------------
# Synthetic data
# -----------------------------------------------------------------------------
def prices_for(block: int) -> dict:
    base = {ETH: 2000.0, WETH: 2000.0, USDC: 1.0, USDT: 1.0005, DAI: 0.999, SUSDE: 1.12, EURC: 1.08,
            WBTC: 40000.0, WSTETH: 2300.0}
    drift = {100: 1.0, 200: 1.3, 300: 0.8}[block]
    p = {}
    for t, v in base.items():
        p[t] = v if t in (USDC, USDT, DAI, SUSDE, EURC) else v * drift
    for j, t in enumerate(RISKY):
        p[t] = float(10 ** (j - 2) * (1 + 0.5 * RNG.random()) * drift)
    p[UNPRICED] = None
    if block != 100:
        p[FLAKY] = None
    return p


def gen_holdings(n_accounts: int = 3000):
    addrs = [rand_addr() for _ in range(n_accounts)]
    ca = set(RNG.choice(addrs, size=n_accounts // 10, replace=False))
    state = {}   # addr -> {token: balance string}
    rows = {b: [] for b in BLOCKS}
    for b in BLOCKS:
        px = prices_for(b)
        for a in addrs:
            if a not in state:
                if RNG.random() < 0.8 or b == BLOCKS[0]:
                    k = int(RNG.integers(1, 7))
                    toks = list(RNG.choice(TOKENS, size=k, replace=False))
                    state[a] = {t: repr(float(np.round(RNG.lognormal(2, 3), 6)) + 1e-6) for t in toks}
                else:
                    continue
            else:
                r = RNG.random()
                h = dict(state[a])
                if r < 0.30:
                    pass                                              # unchanged
                elif r < 0.45 and ETH in h:
                    h[ETH] = repr(max(float(h[ETH]) - 0.001, 1e-6))   # gas only
                elif r < 0.60:
                    t_out = RNG.choice(list(h))
                    del h[t_out]
                    t_in = RNG.choice(TOKENS)
                    h[t_in] = repr(float(np.round(RNG.lognormal(2, 3), 6)) + 1e-6)   # swap
                elif r < 0.70:
                    del state[a]                                       # exit
                    continue
                elif r < 0.80:
                    t = RNG.choice(list(h))
                    h[t] = repr(float(h[t]) * 2.0)                     # buy more of one asset
                else:
                    h = {t: repr(float(v) * 3.0) for t, v in h.items()}   # pro-rata deposit
                if h:
                    state[a] = h
                else:
                    del state[a]
                    continue
            hold = []
            for t, bal in state[a].items():
                v = None if px[t] is None else float(bal) * px[t]
                hold.append({"token_address": t, "balance": bal, "value_usd": v})
            if RNG.random() < 0.05:
                hold.append({"token_address": RNG.choice(RISKY[:5]), "balance": "0", "value_usd": 0.0})
            rows[b].append((a, hold))
        rows[b].append((BEACON_DEPOSIT_CONTRACT, [{"token_address": ETH, "balance": "5000000.0",
                                                     "value_usd": 5e6 * px[ETH]}]))
    return rows, ca


HOLD_TYPE = pa.list_(pa.struct([pa.field("token_address", pa.string()), pa.field("balance", pa.string()),
                                pa.field("value_usd", pa.float64())]))


def to_table(recs, block=None) -> pa.Table:
    cols = {"wallet_address": [], "total_value_usd": [], "num_tokens": [], "holdings": []}
    if block is not None:
        cols["block_number"] = []
    for a, h in recs:
        cols["wallet_address"].append(a.upper().replace("0X", "0x") if RNG.random() < 0.1 else a)  # mixed case
        cols["total_value_usd"].append(repr(sum(x["value_usd"] or 0.0 for x in h)))
        cols["num_tokens"].append(sum(1 for x in h if float(x["balance"]) > 0))
        cols["holdings"].append(h)
        if block is not None:
            cols["block_number"].append(block)
    fields = [pa.field("wallet_address", pa.string()), pa.field("total_value_usd", pa.string()),
              pa.field("num_tokens", pa.int32()), pa.field("holdings", HOLD_TYPE)]
    if block is not None:
        fields.insert(1, pa.field("block_number", pa.int64()))
    return pa.table({f.name: cols[f.name] for f in fields}, schema=pa.schema(fields))


def write_data(work: Path):
    rows, ca = gen_holdings()
    hive = work / "hive"
    for b, recs in rows.items():
        d = hive / f"block_number={b}"
        d.mkdir(parents=True)
        pq.write_table(to_table(recs), d / "part-0.parquet")
    single = work / "single.parquet"
    pq.write_table(pa.concat_tables([to_table(recs, b) for b, recs in rows.items()]), single)
    ca_tsv = work / "ca.tsv"
    ca_tsv.write_text("address\n" + "\n".join(sorted(ca)) + "\n")
    return rows, ca, hive, single, ca_tsv


# -----------------------------------------------------------------------------
# Independent reference implementation (plain pandas / Python)
# -----------------------------------------------------------------------------
W_EDGES = [(1, "0–1"), (100, "1–100"), (1_000, "100–1K"), (10_000, "1K–10K"), (100_000, "10K–100K")]


def wealth_label(v: float) -> str:
    for edge, lab in W_EDGES:
        if v <= edge:
            return lab
    return ">100K"


def n_label(n: int) -> str:
    return "1" if n <= 1 else "2" if n == 2 else "3–4" if n <= 4 else "5–9" if n <= 9 else "10+"


def cube(df: pd.DataFrame, dims, fn, extra=()):
    """Apply fn to every CUBE group of dims (missing dims = ALL)."""
    out = []
    for r in range(len(dims) + 1):
        for keep in itertools.combinations(dims, r):
            keys = list(keep) + list(extra)
            groups = [((), df)] if not keys else df.groupby(keys)
            for key, g in groups:
                key = key if isinstance(key, tuple) else (key,)
                rec = dict(zip(keys, key))
                for d in dims:
                    rec.setdefault(d, "ALL")
                rec.update(fn(g))
                out.append(rec)
    return pd.DataFrame(out)


def account_frame(rows, ca, block):
    recs = []
    for a, h in rows[block]:
        a = a.lower()
        if a == BEACON_DEPOSIT_CONTRACT:
            continue
        recs.append((a, [(x["token_address"], float(x["balance"]), x["value_usd"]) for x in h]))
    return recs


def ref_cash(rows, ca, cash_set):
    out = []
    for b in BLOCKS:
        for a, pos in account_frame(rows, ca, b):
            val = [(t, v) for t, _, v in pos if v is not None and v > 0]
            V = sum(v for _, v in val)
            C = sum(v for t, v in val if t in cash_set)
            E = sum(v for t, v in val if t in (ETH, WETH))
            D = sum(v for t, v in val if t in ETH_DERIVATIVE_TOKENS)
            B = sum(v for t, v in val if t in BTC_WRAPPER_TOKENS)
            n_eth = sum(1 for t, _ in val if t in (ETH, WETH))
            n_risky = sum(1 for t, _ in val if t not in cash_set) - max(n_eth - 1, 0)
            R = V - C
            sq = sum(v * v for t, v in val if t not in cash_set and t not in (ETH, WETH))
            out.append(dict(block=b, addr=a, V=V, C=C, E=E, D=D, B=B, R=R, n_risky=n_risky,
                            hhi_r=(sq + E * E) / (R * R) if R > 0 else np.nan,
                            wallet_type="CA" if a in ca else "EOA", wealth_bucket=wealth_label(V)))
    return pd.DataFrame(out)


def ref_conc(rows, ca):
    out = []
    for b in BLOCKS:
        for a, pos in account_frame(rows, ca, b):
            val = [(t, v) for t, _, v in pos if v is not None and v > 0]
            V = sum(v for _, v in val)
            if V <= 0:
                continue
            E = sum(v for t, v in val if t in (ETH, WETH))
            n_eth = sum(1 for t, _ in val if t in (ETH, WETH))
            n = len(val) - max(n_eth - 1, 0)
            others = [v for t, v in val if t not in (ETH, WETH)]
            mw = max(max(others, default=0.0), E) / V
            hhi = (sum(v * v for v in others) + E * E) / (V * V)
            out.append(dict(block=b, addr=a, V=V, n=n, mw=mw, hhi=hhi, wallet_type="CA" if a in ca else "EOA",
                            wealth_bucket=wealth_label(V), n_bucket=n_label(n)))
    return pd.DataFrame(out)


def folded_positions(rows, block):
    acc = {}
    for a, h in rows[block]:
        a = a.lower()
        if a == BEACON_DEPOSIT_CONTRACT:
            continue
        d = {}
        for x in h:
            u = float(x["balance"])
            if u <= 0:
                continue
            t = ETH if x["token_address"] in (ETH, WETH) else x["token_address"]
            v = x["value_usd"] or 0.0
            pu, pv = d.get(t, (0.0, 0.0))
            d[t] = (pu + u, pv + v)
        if d:
            acc[a] = d
    return acc


def ref_drift(rows, ca):
    out = []
    for b0, b1 in zip(BLOCKS[:-1], BLOCKS[1:]):
        P0, P1 = folded_positions(rows, b0), folded_positions(rows, b1)
        num, den = {}, {}
        for d in P1.values():
            for t, (u, v) in d.items():
                if u > 0 and v > 0:
                    num[t] = num.get(t, 0.0) + v
                    den[t] = den.get(t, 0.0) + u
        price1 = {t: num[t] / den[t] for t in num}
        for a in set(P0) | set(P1):
            d0, d1 = P0.get(a, {}), P1.get(a, {})
            toks = set(d0) | set(d1)
            w0v = {t: d0[t][1] if t in d0 and d0[t][1] > 0 else 0.0 for t in toks}
            pv = {t: d0[t][0] * price1.get(t, 0.0) if w0v[t] > 0 else 0.0 for t in toks}
            a1v = {t: d1[t][1] if t in d1 and d1[t][1] > 0 else 0.0 for t in toks}
            V0, Pv, A1 = sum(w0v.values()), sum(pv.values()), sum(a1v.values())
            u0 = {t: d0.get(t, (0.0, 0.0))[0] for t in toks}
            u1 = {t: d1.get(t, (0.0, 0.0))[0] for t in toks}
            same = {t: abs(u1[t] - u0[t]) <= 1e-9 * max(abs(u0[t]), abs(u1[t])) for t in toks}
            out.append(dict(
                block_t=b0, block_t1=b1, addr=a, V0=V0, P1=Pv, A1=A1,
                n0=sum(1 for t in toks if w0v[t] > 0),
                turnover=0.5 * sum(abs(a1v[t] / A1 - pv[t] / Pv) for t in toks) if Pv > 0 and A1 > 0 else np.nan,
                drift=0.5 * sum(abs(w0v[t] / V0 - pv[t] / Pv) for t in toks) if V0 > 0 and Pv > 0 else np.nan,
                unchanged=all(same.values()), unchanged_ex_eth=all(s for t, s in same.items() if t != ETH),
                added=sum(1 for t in toks if u0[t] <= 0 and u1[t] > 0),
                removed=sum(1 for t in toks if u0[t] > 0 and u1[t] <= 0),
                v0_unpriced=sum(w0v[t] for t in toks if w0v[t] > 0 and pv[t] <= 0),
                wallet_type="CA" if a in ca else "EOA",
            ))
    return pd.DataFrame(out)


# -----------------------------------------------------------------------------
# Comparison helpers
# -----------------------------------------------------------------------------
FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILS.append(name)


def compare(name, got: pd.DataFrame, exp: pd.DataFrame, keys, cols):
    got = got.copy()
    exp = exp.copy()
    for k in keys:
        got[k] = got[k].astype(str)
        exp[k] = exp[k].astype(str)
    m = exp.merge(got, on=keys, how="left", suffixes=("_exp", "_got"), indicator=True)
    missing = int((m["_merge"] != "both").sum())
    if missing:
        check(name, False, f"{missing} expected group(s) missing, e.g. {m.loc[m['_merge'] != 'both', keys].head(3).to_dict('records')}")
        return
    bad = []
    for c in cols:
        e, g = m[f"{c}_exp"].astype(float), m[f"{c}_got"].astype(float)
        diff = ~(np.isclose(e, g, rtol=TOL, atol=1e-9) | (e.isna() & g.isna()))
        if diff.any():
            i = diff.idxmax()
            bad.append(f"{c}: exp {e[i]} got {g[i]} at {m.loc[i, keys].to_dict()}")
    check(name, not bad, "; ".join(bad[:3]))
    extra = len(got) - len(exp)
    if extra:
        check(f"{name}: no unexpected rows", False, f"{extra} extra row(s) in script output")


def run(args, label):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        check(f"{label} runs", False, "non-zero exit")
        raise SystemExit(1)
    check(f"{label} runs", True)
    return r.stdout


def q(x, p):
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, p)) if len(x) else np.nan


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_cash(rows, ca, hive, ca_tsv, out):
    print("\nstablecoin_cash_share.py")
    run([SRC / "stablecoin_cash_share.py", "--parquet", hive, "--ca-tsv", ca_tsv, "--out-dir", out,
         "--threads", 4, "--mem", "2GB"], "cash core")
    cash_set = set(load_cash_tokens(DEFAULT_STABLECOINS_CSV, "core")["contract_address"])
    check("synthetic cash tokens are in the core list", {USDC, USDT, DAI} <= cash_set and SUSDE not in cash_set)
    ref = ref_cash(rows, ca, cash_set)
    valued = ref[ref["V"] > 0]

    def summ(g):
        c = g["C"] / g["V"]
        holders = c[g["C"] > 0]
        return dict(n_accounts=len(g), n_with_cash=int((g["C"] > 0).sum()),
                    n_cash_only=int(((g["C"] > 0) & (g["R"] <= 0)).sum()), mean_cash_share=c.mean(),
                    median_cash_share_holders=q(holders, 0.5), p90_cash_share_holders=q(holders, 0.9),
                    value_usd=g["V"].sum(), cash_value_usd=g["C"].sum(), eth_value_usd=g["E"].sum(),
                    btc_value_usd=g["B"].sum())
    exp = pd.concat([cube(g, ["wallet_type", "wealth_bucket"], summ).assign(block=b) for b, g in valued.groupby("block")])
    got = pd.read_csv(out / "cash_share_summary_core.csv")
    compare("cash summary", got, exp, ["block", "wallet_type", "wealth_bucket"],
            ["n_accounts", "n_with_cash", "n_cash_only", "mean_cash_share", "median_cash_share_holders",
             "p90_cash_share_holders", "value_usd", "cash_value_usd", "eth_value_usd", "btc_value_usd"])

    edges = [(0, "0"), (0.01, "(0,1%]"), (0.05, "(1,5%]"), (0.10, "(5,10%]"), (0.25, "(10,25%]"), (0.50, "(25,50%]"),
             (0.75, "(50,75%]"), (0.90, "(75,90%]"), (0.99, "(90,99%]")]

    def cash_bin(c):
        for e, lab in edges:
            if c <= e:
                return lab
        return "(99,100%)" if c < 1 else "100%"
    valued = valued.assign(cash_bin=(valued["C"] / valued["V"]).map(cash_bin))
    exp_h = pd.concat([cube(g, ["wallet_type", "wealth_bucket"], lambda x: dict(n_accounts=len(x), value_usd=x["V"].sum()),
                            extra=["cash_bin"]).assign(block=b) for b, g in valued.groupby("block")])
    got_h = pd.read_csv(out / "cash_share_hist_core.csv")
    compare("cash histogram", got_h, exp_h, ["block", "wallet_type", "wealth_bucket", "cash_bin"], ["n_accounts", "value_usd"])

    risky = valued[valued["R"] > 0].copy()
    mix_edges = [(0, "0"), (0.10, "(0,10%]"), (0.50, "(10,50%]"), (0.90, "(50,90%]")]
    risky["cash_bin"] = (risky["C"] / risky["V"]).map(lambda c: next((lab for e, lab in mix_edges if c <= e), "(90,100%)"))

    def mix(g):
        return dict(n_accounts=len(g), mean_eth_share=(g["E"] / g["R"]).mean(), median_eth_share=q(g["E"] / g["R"], 0.5),
                    mean_majors_share=((g["E"] + g["D"] + g["B"]) / g["R"]).mean(),
                    eth_share_value_weighted=g["E"].sum() / g["R"].sum(), mean_n_risky=g["n_risky"].mean(),
                    share_single_risky=(g["n_risky"] == 1).mean(), mean_hhi_risky=g["hhi_r"].mean(),
                    median_hhi_risky=q(g["hhi_r"], 0.5))
    exp_m = pd.concat([cube(g, ["wallet_type", "wealth_bucket", "cash_bin"], mix).assign(block=b) for b, g in risky.groupby("block")])
    got_m = pd.read_csv(out / "risky_mix_by_cash_bin_core.csv")
    compare("risky mix by cash bin", got_m, exp_m, ["block", "wallet_type", "wealth_bucket", "cash_bin"],
            ["n_accounts", "mean_eth_share", "median_eth_share", "mean_majors_share", "eth_share_value_weighted",
             "mean_n_risky", "share_single_risky", "mean_hhi_risky", "median_hhi_risky"])

    agg = pd.read_csv(out / "block_aggregates_core.csv")
    exp_a = ref.groupby("block").agg(value_usd=("V", "sum"), cash_value_usd=("C", "sum"),
                                     n_accounts_unpriced=("V", lambda v: int((v <= 0).sum()))).reset_index()
    compare("block aggregates (beacon contract excluded)", agg, exp_a, ["block"],
            ["value_usd", "cash_value_usd", "n_accounts_unpriced"])
    comp = pd.read_csv(out / "stablecoin_composition_core.csv")
    check("stablecoin composition sums to 100 %", np.allclose(comp.groupby("block")["pct_of_cash"].sum(), 100.0))

    run([SRC / "stablecoin_cash_share.py", "--parquet", hive, "--ca-tsv", ca_tsv, "--out-dir", out,
         "--cash-def", "extended", "--threads", 4, "--mem", "2GB"], "cash extended")
    ext = pd.read_csv(out / "block_aggregates_extended.csv")
    check("extended cash >= core cash", bool((ext["cash_value_usd"].values >= agg["cash_value_usd"].values - 1e-6).all()))


def test_conc(rows, ca, hive, single, ca_tsv, out):
    print("\nweight_concentration.py")
    run([SRC / "weight_concentration.py", "--parquet", hive, "--ca-tsv", ca_tsv, "--out-dir", out,
         "--threads", 4, "--mem", "2GB"], "concentration")
    ref = ref_conc(rows, ca)

    def summ(g):
        multi = g[g["n"] >= 2]
        bind = multi[multi["mw"] > CAP]
        return dict(n_accounts=len(g), value_usd=g["V"].sum(), mean_max_weight=g["mw"].mean(),
                    median_max_weight=q(g["mw"], 0.5), p90_max_weight=q(g["mw"], 0.9),
                    mean_effective_n=(1 / g["hhi"]).mean(), median_effective_n=q(1 / g["hhi"], 0.5),
                    n_cap_binding=len(bind), mean_cap_shift_binding=(bind["mw"] - CAP).mean() if len(bind) else np.nan,
                    mean_cap_shift_multi=(bind["mw"] - CAP).sum() / len(multi) if len(multi) else np.nan)
    exp = pd.concat([cube(g, ["wallet_type", "wealth_bucket", "n_bucket"], summ).assign(block=b) for b, g in ref.groupby("block")])
    exp2 = pd.concat([cube(g[g["n"] >= 2], ["wallet_type", "wealth_bucket"], summ).assign(block=b, n_bucket="2+")
                      for b, g in ref.groupby("block")])
    got = pd.read_csv(out / "weight_concentration_summary.csv")
    cols = ["n_accounts", "value_usd", "mean_max_weight", "median_max_weight", "p90_max_weight", "mean_effective_n",
            "median_effective_n", "n_cap_binding", "mean_cap_shift_binding", "mean_cap_shift_multi"]
    compare("concentration summary", got, pd.concat([exp, exp2]), ["block", "wallet_type", "wealth_bucket", "n_bucket"], cols)

    mw_edges = [(0.25, "[0,0.25)", "<"), (0.5, "[0.25,0.5)", "<"), (0.6, "[0.5,0.6)", "<"), (0.7, "[0.6,0.7)", "<"),
                (0.8, "[0.7,0.8)", "<"), (0.9, "[0.8,0.9]", "<="), (0.95, "(0.9,0.95]", "<="), (0.99, "(0.95,0.99]", "<=")]

    def mw_bin(x):
        for e, lab, op in mw_edges:
            if (x < e) if op == "<" else (x <= e):
                return lab
        return "(0.99,1]"
    multi = ref[ref["n"] >= 2].assign(max_weight_bin=lambda d: d["mw"].map(mw_bin))
    exp_h = pd.concat([cube(g, ["wallet_type", "wealth_bucket", "n_bucket"], lambda x: dict(n_accounts=len(x)),
                            extra=["max_weight_bin"]).assign(block=b) for b, g in multi.groupby("block")])
    exp_h.loc[exp_h["n_bucket"] == "ALL", "n_bucket"] = "2+"
    got_h = pd.read_csv(out / "max_weight_hist.csv")
    compare("max-weight histogram", got_h, exp_h, ["block", "wallet_type", "wealth_bucket", "n_bucket", "max_weight_bin"], ["n_accounts"])

    before = got.copy()
    run([SRC / "weight_concentration.py", "--parquet", hive, "--ca-tsv", ca_tsv, "--out-dir", out,
         "--threads", 4, "--mem", "2GB", "--resume"], "concentration --resume")
    after = pd.read_csv(out / "weight_concentration_summary.csv")
    check("--resume leaves complete outputs unchanged", after.equals(before))
    trunc = before[before["block"] != BLOCKS[-1]]
    trunc.to_csv(out / "weight_concentration_summary.csv", index=False)   # simulate a crash mid-block
    run([SRC / "weight_concentration.py", "--parquet", hive, "--ca-tsv", ca_tsv, "--out-dir", out,
         "--threads", 4, "--mem", "2GB", "--resume"], "concentration --resume after partial write")
    again = pd.read_csv(out / "weight_concentration_summary.csv")
    hist_again = pd.read_csv(out / "max_weight_hist.csv")
    check("partial block recomputed without duplicates",
          len(again) == len(before) and not again.duplicated(["block", "wallet_type", "wealth_bucket", "n_bucket"]).any()
          and not hist_again.duplicated(["block", "wallet_type", "wealth_bucket", "n_bucket", "max_weight_bin"]).any())

    out_single = out / "single"
    run([SRC / "weight_concentration.py", "--parquet", single, "--ca-tsv", ca_tsv, "--out-dir", out_single,
         "--threads", 4, "--mem", "2GB"], "concentration on a single 4_0 file")
    s = pd.read_csv(out_single / "weight_concentration_summary.csv")
    compare("single-file input gives the same summary", s, before, ["block", "wallet_type", "wealth_bucket", "n_bucket"], cols)


def test_drift(rows, ca, hive, ca_tsv, out, buckets):
    print(f"\ntrading_vs_drift.py (--buckets {buckets})")
    run([SRC / "trading_vs_drift.py", "--parquet", hive, "--ca-tsv", ca_tsv, "--out-dir", out,
         "--stage-dir", out / "stage", "--buckets", buckets, "--threads", 4, "--mem", "2GB"], f"drift, {buckets} bucket(s)")
    ref = ref_drift(rows, ca)
    at_t = ref[ref["V0"] > 0].copy()
    at_t["wealth_bucket"] = at_t["V0"].map(wealth_label)
    at_t["n_bucket"] = at_t["n0"].map(n_label)

    def summ(g):
        stay = g[g["A1"] > 0]
        meas = stay[stay["P1"] > 0]
        rec = dict(n_accounts_t=len(g), n_exit=int((g["A1"] <= 0).sum()), n_stay=len(stay), n_measured=len(meas),
                   n_units_unchanged=int(stay["unchanged"].sum()), n_units_unchanged_ex_eth=int(stay["unchanged_ex_eth"].sum()),
                   n_turnover_zero=int((meas["turnover"] <= 1e-12).sum()),
                   n_turnover_le_1pct=int((meas["turnover"] <= 0.01).sum()),
                   n_turnover_le_25pct=int((meas["turnover"] <= 0.25).sum()),
                   mean_turnover=meas["turnover"].mean() if len(meas) else np.nan,
                   mean_drift=meas["drift"].mean() if len(meas) else np.nan,
                   turnover_value_weighted=(meas["V0"] * meas["turnover"]).sum() / meas["V0"].sum() if len(meas) else np.nan,
                   n_added_any=int((stay["added"] > 0).sum()), n_removed_any=int((stay["removed"] > 0).sum()),
                   n_net_inflow_10pct=int((meas["A1"] > 1.1 * meas["P1"]).sum()),
                   share_v0_unpriced_t1=g["v0_unpriced"].sum() / g["V0"].sum())
        return rec
    exp = pd.concat([cube(g, ["wallet_type", "wealth_bucket", "n_bucket"], summ).assign(block_t=bt, block_t1=bt1)
                     for (bt, bt1), g in at_t.groupby(["block_t", "block_t1"])])
    got = pd.read_csv(out / "trading_drift_summary.csv")
    compare(f"drift summary ({buckets} bucket(s))", got, exp, ["block_t", "block_t1", "wallet_type", "wealth_bucket", "n_bucket"],
            ["n_accounts_t", "n_exit", "n_stay", "n_measured", "n_units_unchanged", "n_units_unchanged_ex_eth",
             "n_turnover_zero", "n_turnover_le_1pct", "n_turnover_le_25pct", "mean_turnover", "mean_drift",
             "turnover_value_weighted", "n_added_any", "n_removed_any", "n_net_inflow_10pct", "share_v0_unpriced_t1"])
    tot = got[(got["wallet_type"] == "ALL") & (got["wealth_bucket"] == "ALL") & (got["n_bucket"] == "ALL")]
    check("synthetic scenarios present (unchanged, gas-only, swaps, exits, lost price)",
          bool((tot["n_units_unchanged"] > 0).all() and (tot["n_units_unchanged_ex_eth"] > tot["n_units_unchanged"]).all()
               and (tot["n_exit"] > 0).all() and (tot["mean_turnover"] > 0).all() and (tot["share_v0_unpriced_t1"] > 0).any()))

    ent_ref = ref[(ref["V0"] <= 0) & (ref["A1"] > 0)].assign(wealth_bucket=lambda d: d["A1"].map(wealth_label))
    exp_e = pd.concat([cube(g, ["wallet_type", "wealth_bucket"], lambda x: dict(n_entries=len(x), value_entries_usd=x["A1"].sum()))
                       .assign(block_t=bt, block_t1=bt1) for (bt, bt1), g in ent_ref.groupby(["block_t", "block_t1"])])
    compare(f"entries ({buckets} bucket(s))", pd.read_csv(out / "trading_drift_entries.csv"), exp_e,
            ["block_t", "block_t1", "wallet_type", "wealth_bucket"], ["n_entries", "value_entries_usd"])
    hist = pd.read_csv(out / "trading_drift_hist.csv")
    tot_h = hist[(hist["wallet_type"] == "ALL") & (hist["wealth_bucket"] == "ALL") & (hist["n_bucket"] == "ALL")]
    n_meas = tot.set_index(["block_t", "metric"] if "metric" in tot else ["block_t"])["n_measured"] if False else \
        tot.set_index("block_t")["n_measured"]
    ok = all(int(tot_h[(tot_h["block_t"] == bt) & (tot_h["metric"] == m)]["n_accounts"].sum()) == int(n_meas[bt])
             for bt in n_meas.index for m in ("turnover", "drift"))
    check("histograms cover every measured account", ok)
    check("staged snapshots cleaned up", not any((out / "stage").glob("block_number=*")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="Keep the work directory.")
    args = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="alloc_test_"))
    print(f"work dir: {work}")
    try:
        rows, ca, hive, single, ca_tsv = write_data(work)
        print(f"synthetic data: {sum(len(r) for r in rows.values())} account rows in {len(BLOCKS)} snapshots")
        test_cash(rows, ca, hive, ca_tsv, work / "out_cash")
        test_conc(rows, ca, hive, single, ca_tsv, work / "out_conc")
        test_drift(rows, ca, hive, ca_tsv, work / "out_drift_4", 4)
        test_drift(rows, ca, hive, ca_tsv, work / "out_drift_1", 1)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL CHECKS PASSED' if not FAILS else f'{len(FAILS)} CHECK(S) FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
