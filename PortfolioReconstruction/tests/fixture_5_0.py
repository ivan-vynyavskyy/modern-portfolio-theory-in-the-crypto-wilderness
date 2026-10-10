#!/usr/bin/env python3
"""
fixture_5_0.py

Deterministic mock-data builder for the 5_0 backtest tests.

Generates, under a given root directory:
    prices/<token>_price.csv       daily closes for 6 tokens (incl. WETH/WBTC
                                   benchmarks and the native-ETH pseudo-token)
    holdings.parquet               single-file holdings (4_0-style schema)
    holdings_hive/block_number=N/  the same rows as a 4_2-style Hive dataset
                                   (block_number only in the directory name)
    block_to_date.csv              blocks 100 and 200 -> snapshot dates
    token_stats.csv                per-token supply (market-cap strategy)
    expected.json                  hand-computable ground truth: per block the
                                   eligible wallet set, W1's exact baseline and
                                   equal-weight forward returns, and the
                                   benchmark market return

Everything is seeded, so the fixture is bit-identical across rebuilds. The
builder skips work when <root>/FIXTURE_VERSION matches (create once, reuse).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FIXTURE_VERSION = "5_0-fixture-v3"  # v3: ETH/WETH merge (W1 priced via ETH series; W7 pair added)

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
ETH_PSEUDO = "0x" + "ee" * 20
TOKA = "0x" + "aa" * 20
TOKB = "0x" + "bb" * 20
TOKC = "0x" + "cc" * 20
NOPRICE = "0x" + "99" * 20  # deliberately has no price CSV

PRICED_TOKENS = {
    # token: (start_price, daily_drift, daily_vol)
    WETH: (2000.0, 0.002, 0.03),
    WBTC: (35000.0, 0.001, 0.025),
    ETH_PSEUDO: (2000.0, 0.002, 0.03),
    TOKA: (10.0, 0.004, 0.05),
    TOKB: (1.0, -0.001, 0.04),
    TOKC: (250.0, 0.000, 0.06),
}

BLOCKS = {100: "2021-06-01", 200: "2021-07-01"}
HORIZON_DAYS = 20

W1 = "0x" + "01" * 20  # the known-answer wallet: 100 TOKA + 2 WETH
                       # (under the ETH/WETH merge the WETH position becomes
                       # ETH, priced with the ETH pseudo-token series)
W2 = "0x" + "02" * 20  # 5 tokens
W3 = "0x" + "03" * 20  # ETH pseudo-token + TOKB
W4 = "0x" + "04" * 20  # 1 token -> ineligible (num_tokens < 2)
W5 = "0x" + "05" * 20  # TOKA + NOPRICE -> optimizer must drop it (1 priced asset)
W6 = "0x" + "06" * 20  # zero balances -> empty holdings map, filtered out
W7 = "0x" + "07" * 20  # ETH + WETH pair -> collapses to ONE asset under the
                       # merge and must yield NO record

ELIGIBLE_PROCESSED = sorted([W1, W2, W3])  # wallets that must yield records

HOLD_T = pa.list_(pa.struct([("token_address", pa.string()),
                             ("balance", pa.string()),
                             ("value_usd", pa.float64())]))
SCHEMA = pa.schema([("wallet_address", pa.string()), ("block_number", pa.int64()),
                    ("total_value_usd", pa.string()), ("num_tokens", pa.int32()),
                    ("holdings", HOLD_T)])


def _gen_prices() -> pd.DataFrame:
    """Seeded geometric random walk per token, daily 2021-03-01..2021-08-05."""
    dates = pd.date_range("2021-03-01", "2021-08-05", freq="D")
    rng = np.random.default_rng(42)
    out = {}
    for token, (p0, drift, vol) in PRICED_TOKENS.items():
        steps = rng.normal(loc=drift, scale=vol, size=len(dates) - 1)
        logp = np.concatenate([[np.log(p0)], np.log(p0) + np.cumsum(steps)])
        out[token] = np.exp(logp)
    return pd.DataFrame(out, index=dates)


def _wallet_rows(block: int) -> list[dict]:
    def h(*pairs):
        return [{"token_address": t, "balance": str(b), "value_usd": 0.0}
                for t, b in pairs]

    scale = 1.0 if block == 100 else 1.5  # different balances per block
    return [
        {"wallet_address": W1, "block_number": block, "total_value_usd": "1",
         "num_tokens": 2, "holdings": h((TOKA, 100 * scale), (WETH, 2 * scale))},
        {"wallet_address": W2, "block_number": block, "total_value_usd": "1",
         "num_tokens": 5, "holdings": h((TOKA, 50), (TOKB, 1000), (TOKC, 4),
                                        (WETH, 1), (WBTC, 0.1))},
        {"wallet_address": W3, "block_number": block, "total_value_usd": "1",
         "num_tokens": 2, "holdings": h((ETH_PSEUDO, 3), (TOKB, 500))},
        {"wallet_address": W4, "block_number": block, "total_value_usd": "1",
         "num_tokens": 1, "holdings": h((TOKA, 10))},
        {"wallet_address": W5, "block_number": block, "total_value_usd": "1",
         "num_tokens": 2, "holdings": h((TOKA, 10), (NOPRICE, 10))},
        {"wallet_address": W6, "block_number": block, "total_value_usd": "0",
         "num_tokens": 2, "holdings": h((TOKA, 0), (TOKB, 0))},
        {"wallet_address": W7, "block_number": block, "total_value_usd": "1",
         "num_tokens": 2, "holdings": h((ETH_PSEUDO, 1), (WETH, 2))},
    ]


def _expected(prices: pd.DataFrame) -> dict:
    """Ground truth mirroring 5_0's exact date conventions:
    snapshot price = last day of the lookback window = snapshot_date - 1 day;
    horizon price  = snapshot_date + HORIZON_DAYS (exact row exists);
    market return  = mean over benchmarks of ln(P(s+h) / P(s))."""
    out: dict = {"eligible_processed": ELIGIBLE_PROCESSED, "blocks": {}}
    for block, date_str in BLOCKS.items():
        s = pd.Timestamp(date_str)
        p_snap = prices.loc[s - pd.Timedelta(days=1)]
        p_hor = prices.loc[s + pd.Timedelta(days=HORIZON_DAYS)]

        scale = 1.0 if block == 100 else 1.5
        # W1 holds TOKA + WETH on chain; under the ETH/WETH merge the WETH
        # units become ETH units priced with the ETH pseudo-token series.
        units = {TOKA: 100 * scale, ETH_PSEUDO: 2 * scale}
        total0 = sum(u * p_snap[t] for t, u in units.items())
        v_hor = sum(u * p_hor[t] for t, u in units.items())
        ret_baseline = v_hor / total0 - 1.0
        ret_equal = float(np.mean([p_hor[t] / p_snap[t] for t in units])) - 1.0

        # simple holding-period return, matching compute_forward_market_return
        mkt = float(np.mean([p_hor[t] / prices.loc[s, t] - 1.0
                             for t in (WETH, WBTC)]))
        out["blocks"][str(block)] = {
            "date": date_str,
            "w1_ret_baseline": ret_baseline,
            "w1_ret_equal_weight": ret_equal,
            "market_return": mkt,
        }
    return out


def build_fixture(root: pathlib.Path) -> pathlib.Path:
    root = pathlib.Path(root)
    stamp = root / "FIXTURE_VERSION"
    if stamp.exists() and stamp.read_text().strip() == FIXTURE_VERSION:
        return root
    root.mkdir(parents=True, exist_ok=True)

    prices = _gen_prices()
    prices_dir = root / "prices"
    prices_dir.mkdir(exist_ok=True)
    for token in prices.columns:
        pd.DataFrame({
            "date": prices.index.strftime("%Y-%m-%d"),
            "daily_close_usd": prices[token].values,
        }).to_csv(prices_dir / f"{token}_price.csv", index=False)

    rows = [r for b in BLOCKS for r in _wallet_rows(b)]
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, root / "holdings.parquet")

    # 4_2-style Hive layout: block_number lives ONLY in the directory name
    for block in BLOCKS:
        d = root / "holdings_hive" / f"block_number={block}"
        d.mkdir(parents=True, exist_ok=True)
        sub = pa.Table.from_pylist(_wallet_rows(block), schema=SCHEMA)
        pq.write_table(sub.drop_columns(["block_number"]), d / "part-0.parquet")

    pd.DataFrame({"block_number": list(BLOCKS), "date": list(BLOCKS.values())}
                 ).to_csv(root / "block_to_date.csv", index=False)

    pd.DataFrame([{"block": 0, "token_address": t, "supply": 1_000_000.0}
                  for t in PRICED_TOKENS]).to_csv(root / "token_stats.csv",
                                                  index=False)

    (root / "expected.json").write_text(json.dumps(_expected(prices), indent=2))
    stamp.write_text(FIXTURE_VERSION + "\n")
    return root


if __name__ == "__main__":
    p = build_fixture(pathlib.Path(__file__).parent / "fixture_5_0")
    print(f"fixture ready at {p}")
