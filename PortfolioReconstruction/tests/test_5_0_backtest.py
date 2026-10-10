#!/usr/bin/env python3
"""
test_5_0_backtest.py

Unit + integration tests for common_src/5_0_portfolio_backtest_pipeline.py.

- Unit layer: pure functions imported directly (holdings parsing, wallet
  normalization, wealth buckets, --blocks parsing, supply lookup, market
  regime classification, summary-CSV upsert).
- Integration layer: the real CLI run as a subprocess against a small
  deterministic fixture (see fixture_5_0.py, built once and reused), with
  exact numeric checks on W1's baseline / equal-weight forward returns and
  the benchmark market return, plus the Hive-dataset input path and
  upsert idempotency.

Run:
    python tests/test_5_0_backtest.py
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

TESTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
from fixture_5_0 import (  # noqa: E402
    BLOCKS, ELIGIBLE_PROCESSED, ETH_PSEUDO, W1, W4, WETH, build_fixture,
)

SCRIPT = TESTS_DIR.parent / "common_src" / "5_0_portfolio_backtest_pipeline.py"
STRATEGIES = ["Baseline", "Max Return (Same Vol)", "Min Variance (Same Ret)",
              "Max Sharpe", "Equal Weight", "Market-Cap Weight"]
PASSED = 0


def ok(msg: str) -> None:
    global PASSED
    PASSED += 1
    print(f"  ok  {msg}")


def load_module():
    """Import 5_0 as a module (filename starts with a digit)."""
    spec = importlib.util.spec_from_file_location("backtest_pipeline_ut", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backtest_pipeline_ut"] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------- unit tests -----------------------------

def unit_tests(m) -> None:
    print("[unit]")

    hm = m.extract_holdings_map([
        {"token_address": "0xAA", "balance": "2.5"},
        {"token_address": "0xaa", "balance": 1.5},    # merges (case-insensitive)
        {"token_address": "0xbb", "balance": "0"},    # dropped (zero)
        {"token_address": "", "balance": "9"},        # dropped (no token)
    ])
    assert hm == {"0xaa": 4.0}, hm
    ok("extract_holdings_map: merge/case/zero/empty handling")

    # ETH/WETH merge: WETH balances fold into the ETH pseudo-token (and the
    # --no-merge-eth-weth escape hatch keeps them apart).
    raw = [{"token_address": WETH, "balance": "2"},
           {"token_address": ETH_PSEUDO, "balance": "3"},
           {"token_address": "0xaa", "balance": "1"}]
    assert m.extract_holdings_map(raw) == {ETH_PSEUDO: 5.0, "0xaa": 1.0}
    m.MERGE_ETH_WETH = False
    try:
        assert m.extract_holdings_map(raw) == \
            {WETH: 2.0, ETH_PSEUDO: 3.0, "0xaa": 1.0}
    finally:
        m.MERGE_ETH_WETH = True
    ok("extract_holdings_map: ETH/WETH merge + --no-merge escape hatch")

    # expected-return shrinkage toward the cross-asset mean (the paper's return regularisation)
    pa_h = m.PortfolioAnalytics(m.RiskConfig())
    mu = pd.Series({"a": 0.10, "b": 0.02, "c": 0.06})
    sh = pa_h.shrink_expected_returns_to_cross_mean(mu, 0.5)
    assert np.allclose(sh.values, [0.08, 0.04, 0.06]), sh.values
    assert pa_h.shrink_expected_returns_to_cross_mean(mu, 0.0).equals(mu)
    assert np.allclose(
        pa_h.shrink_expected_returns_to_cross_mean(mu, 1.0).values, 0.06)
    mu_nan = pd.Series({"a": 0.10, "b": np.nan, "c": 0.02})
    sh_nan = pa_h.shrink_expected_returns_to_cross_mean(mu_nan, 0.5)
    assert np.isnan(sh_nan["b"]) and np.allclose(
        [sh_nan["a"], sh_nan["c"]], [0.08, 0.04])
    ok("shrink_expected_returns_to_cross_mean: formula/no-op/collapse/NaN")

    # fallback_used flag: max_return_same_vol must report the silent
    # max-Sharpe substitution via last_mrv_fallback (and reset it per call).
    cov2 = np.array([[0.04, 0.0], [0.0, 0.09]])
    mu2 = np.array([0.05, 0.10])
    # attainable variance band for long-only 2-asset with cap 0.95 is
    # ~[0.0277, 0.0813]: 0.05 is feasible, 1e-6 is not.
    w_ok = pa_h.max_return_same_vol(cov=cov2, mu=mu2, target_var=0.05)
    assert pa_h.last_mrv_fallback is False
    assert abs(float(np.sum(w_ok)) - 1.0) < 1e-9
    w_fb = pa_h.max_return_same_vol(cov=cov2, mu=mu2, target_var=1e-6)
    assert pa_h.last_mrv_fallback is True
    assert abs(float(np.sum(w_fb)) - 1.0) < 1e-9
    pa_h.max_return_same_vol(cov=cov2, mu=mu2, target_var=0.05)
    assert pa_h.last_mrv_fallback is False, "flag must reset on next call"
    ok("max_return_same_vol: last_mrv_fallback set on infeasible target, reset after")

    # safer_risk fallback flag: an unattainable target return (above the
    # max achievable under the cap) fails both the equality and the relaxed
    # >= solve, so the global min-variance portfolio is substituted.
    w_ok2 = pa_h.min_variance_target_return(mu=mu2, cov=cov2, target_return=0.06)
    assert pa_h.last_minvar_fallback is False
    assert abs(float(np.sum(w_ok2)) - 1.0) < 1e-9
    w_fb2 = pa_h.min_variance_target_return(mu=mu2, cov=cov2, target_return=0.5)
    assert pa_h.last_minvar_fallback is True
    assert abs(float(np.sum(w_fb2)) - 1.0) < 1e-9
    pa_h.min_variance_target_return(mu=mu2, cov=cov2, target_return=0.06)
    assert pa_h.last_minvar_fallback is False, "flag must reset on next call"
    ok("min_variance_target_return: last_minvar_fallback set on unattainable return, reset after")

    n = m._normalize_wallet_address
    assert n("0xABCDEF") == "0xabcdef"
    assert n("ab" * 20) == "0x" + "ab" * 20        # bare 40-hex gets 0x
    assert n(bytes.fromhex("ab" * 20)) == "0x" + "ab" * 20
    assert n("") is None and n(None) is None
    ok("_normalize_wallet_address: str/bytes/bare-hex/empty")

    wb = m.wealth_bucket_from_value
    assert [wb(0.5), wb(1), wb(99.9), wb(100), wb(999), wb(1000), wb(9999),
            wb(10_000), wb(float("nan")), wb(-1), wb("x")] == \
        ["0-1", "1-100", "1-100", "100-1K", "100-1K", "1K-10K", "1K-10K",
         "10K+", "UNKNOWN", "UNKNOWN", "UNKNOWN"]
    ok("wealth_bucket_from_value: boundaries + invalid")

    assert m.parse_blocks_arg("100, 200,100") == [100, 200]
    assert m.parse_blocks_arg(None) is None
    blk_file = TESTS_DIR / "_blocks_tmp.txt"
    blk_file.write_text("# comment\n300\n\n400\n")
    try:
        assert m.parse_blocks_arg(str(blk_file)) == [300, 400]
    finally:
        blk_file.unlink()
    try:
        m.parse_blocks_arg("0")
        raise AssertionError("expected ValueError for block 0")
    except ValueError:
        pass
    ok("parse_blocks_arg: list, file, dedup, invalid")

    supply = {"0xaa": [(100, 10.0), (200, 20.0)]}
    g = m.get_supply_at_block
    assert g(supply, "0xAA", 150) == 10.0
    assert g(supply, "0xaa", 200) == 20.0
    assert g(supply, "0xaa", 99) is None
    assert g(supply, "0xzz", 150) is None
    ok("get_supply_at_block: bisect semantics")

    # --eth-supply-csv: daily circulating supply -> per-snapshot-block entries
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="t50_ethsup_"))
    pd.DataFrame({"date": ["2021-06-01", "2021-06-28"],
                  "circulating_supply": [1.5e6, 1.6e6]}).to_csv(tmp / "eth_supply.csv", index=False)
    bmap = pd.DataFrame({"block_number": [100, 200, 300],
                         "date": pd.to_datetime(["2021-06-01", "2021-07-01", "2021-09-01"])})
    entries, missing = m.load_eth_supply_override(str(tmp / "eth_supply.csv"), bmap)
    assert entries == [(100, 1.5e6), (200, 1.6e6)] and missing == [300], (entries, missing)
    entries, missing = m.load_eth_supply_override(str(tmp / "eth_supply.csv"), bmap, max_fill_days=1)
    assert entries == [(100, 1.5e6)] and missing == [200, 300], (entries, missing)
    ok("load_eth_supply_override: exact date, 7-day back-fill, unresolved blocks")

    idx = pd.date_range("2021-01-01", periods=80, freq="D")
    up = pd.DataFrame({"a": np.exp(np.linspace(0, 1.6, 80)),
                       "b": np.exp(np.linspace(0, 1.2, 80))}, index=idx)
    r = m.classify_market_regime_from_prices(up, lookback_days=60,
                                             benchmark_tokens=None)
    assert r["regime"] == "BULLISH", r
    down = pd.DataFrame({"a": np.exp(-np.linspace(0, 1.6, 80)),
                         "b": np.exp(-np.linspace(0, 1.2, 80))}, index=idx)
    r = m.classify_market_regime_from_prices(down, lookback_days=60,
                                             benchmark_tokens=None)
    assert r["regime"] == "BEARISH", r
    r = m.classify_market_regime_from_prices(pd.DataFrame())
    assert r["regime"] == "UNKNOWN"
    ok("classify_market_regime_from_prices: bullish/bearish/unknown")

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="t50_upsert_"))
    csv = tmp / "s.csv"
    base = pd.DataFrame([
        {"date": "2021-06-01", "block": "100", "wealth_bucket": "OVERALL",
         "strategy": "Baseline", "strategy_mean": 0.1},
        {"date": "2021-06-01", "block": "100", "wealth_bucket": "OVERALL",
         "strategy": "Max Sharpe", "strategy_mean": 0.2},
    ])
    m.upsert_summary_csv(str(csv), base)
    upd = base.iloc[[1]].copy()
    upd["strategy_mean"] = 0.9
    m.upsert_summary_csv(str(csv), upd)
    got = pd.read_csv(csv)
    assert len(got) == 2, got
    ms = got[got["strategy"] == "Max Sharpe"]["strategy_mean"].iloc[0]
    assert abs(ms - 0.9) < 1e-12, ms
    shutil.rmtree(tmp)
    ok("upsert_summary_csv: keep-last upsert, no duplicate keys")

    # The per-block ParquetWriter fixes its schema from the first batch; a
    # batch where EVERY wallet lacks an mcap weight must not infer pa.null
    # for holdings_mcap_weight (that made write_table raise on mixed runs).
    import pyarrow as pa
    base_rec = {"wallet_address": "0xw", "block_number": 100,
                "date": "2021-06-01", "total_value_usd": 5.0, "num_tokens": 2,
                "ret_baseline": 0.1, "holdings_mcap_weight": None}
    t_none = m._wallet_records_to_table([base_rec])
    t_str = m._wallet_records_to_table([dict(base_rec, holdings_mcap_weight="[]")])
    assert t_none.schema.field("holdings_mcap_weight").type == pa.string()
    assert [f.type for f in t_none.schema] == [f.type for f in t_str.schema]
    ok("_wallet_records_to_table: stable schema for all-None mcap batch")


def pushdown_tests(m, fixture: pathlib.Path) -> None:
    """num_tokens >= 2 predicate pushdown in the holdings scan."""
    print("[unit: scan pushdown]")
    batches = list(m.iter_holdings_wallet_batches_for_block(
        holdings_parquet=str(fixture / "holdings.parquet"),
        block_number=int(sorted(BLOCKS)[0]),
        batch_size=1000,
        columns=["wallet_address", "block_number", "holdings",
                 "total_value_usd", "num_tokens"],
        max_wallets=None,
    ))
    scanned = pd.concat(batches, ignore_index=True)
    assert (pd.to_numeric(scanned["num_tokens"]) >= 2).all(), \
        scanned["num_tokens"].tolist()
    assert W4 not in set(scanned["wallet_address"]), \
        "single-token wallet survived the scan-level pushdown"
    ok("iter_holdings_wallet_batches_for_block: num_tokens<2 rows never materialized")


# -------------------------- integration tests --------------------------

def run_backtest(fixture: pathlib.Path, holdings: pathlib.Path,
                 out_dir: pathlib.Path,
                 extra_args: list[str] = ()) -> subprocess.CompletedProcess:
    out_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--holdings-parquet", str(holdings),
         "--block-to-date-csv", str(fixture / "block_to_date.csv"),
         "--prices-dir", str(fixture / "prices"),
         "--supply-csv", str(fixture / "token_stats.csv"),
         "--out-csv", str(out_dir / "summary.csv"),
         "--out-wallet-parquet", str(out_dir / "wallet_level.parquet"),
         "--workers", "2", *extra_args],
        capture_output=True, text=True, timeout=900)


def check_outputs(fixture: pathlib.Path, out_dir: pathlib.Path,
                  label: str) -> None:
    expected = json.loads((fixture / "expected.json").read_text())

    wallet_root = out_dir / "wallet_level.parquet"
    for block, exp in expected["blocks"].items():
        part = wallet_root / f"block_number={block}" / "data.parquet"
        assert part.exists(), f"missing wallet parquet for block {block}"
        df = ds.dataset(str(part), format="parquet").to_table().to_pandas()

        got_wallets = sorted(df["wallet_address"])
        assert got_wallets == ELIGIBLE_PROCESSED, \
            f"block {block}: wallets {got_wallets}"

        for col in ["ret_baseline", "ret_equal_weight", "l1_gap_max_sharpe",
                    "market_return", "beta_baseline", "holdings_short",
                    "better_return_fallback_used", "safer_risk_fallback_used"]:
            assert col in df.columns, f"missing column {col}"
        for col in ["better_return_fallback_used", "safer_risk_fallback_used"]:
            assert df[col].dtype == bool, (col, df[col].dtype)

        l1_cols = ["l1_gap_better_return", "l1_gap_safer_risk",
                   "l1_gap_max_sharpe"]
        l1 = df[l1_cols].to_numpy(dtype=float)
        assert np.all((l1 >= -1e-12) & (l1 <= 2.0 + 1e-9)), l1
        assert np.isfinite(df["ret_baseline"].to_numpy(dtype=float)).all()

        w1 = df[df["wallet_address"] == W1].iloc[0]
        assert abs(w1["ret_baseline"] - exp["w1_ret_baseline"]) < 1e-9, \
            (w1["ret_baseline"], exp["w1_ret_baseline"])
        assert abs(w1["ret_equal_weight"] - exp["w1_ret_equal_weight"]) < 1e-9
        assert abs(w1["market_return"] - exp["market_return"]) < 1e-9
        ok(f"{label}: block {block} wallets, W1 exact returns, l1 bounds")

    summary = pd.read_csv(out_dir / "summary.csv",
                          dtype={"block": str, "wealth_bucket": str})
    for block in expected["blocks"]:
        blk_rows = summary[(summary["block"] == block)
                           & (summary["wealth_bucket"] == "OVERALL")]
        assert sorted(blk_rows["strategy"]) == sorted(STRATEGIES), \
            f"block {block}: {sorted(blk_rows['strategy'])}"
        base = blk_rows[blk_rows["strategy"] == "Baseline"].iloc[0]
        eq = blk_rows[blk_rows["strategy"] == "Equal Weight"].iloc[0]
        assert int(base["count_optimized"]) == len(ELIGIBLE_PROCESSED)
        assert int(eq["count_optimized"]) == len(ELIGIBLE_PROCESSED)
    # The whole-dataset OVERALL summary was removed from 5_0 (Analytics
    # computes global stats from the wallet-level parquet); make sure it
    # doesn't quietly come back.
    assert not (summary["block"] == "OVERALL").any(), "OVERALL rows resurfaced"
    assert (summary[summary["strategy"] != "Baseline"]["strategy_status"]
            .isin(["BETTER", "MIXED", "WORSE"]).all())
    # fallback_rate: populated (and a valid rate) for the two constrained
    # strategies that can silently substitute, NaN everywhere else.
    assert "fallback_rate" in summary.columns, summary.columns.tolist()
    flagged = ["Max Return (Same Vol)", "Min Variance (Same Ret)"]
    for strat in flagged:
        r = summary[summary["strategy"] == strat]["fallback_rate"]
        assert r.notna().all() and ((r >= 0) & (r <= 1)).all(), (strat, r.tolist())
    assert summary[~summary["strategy"].isin(flagged)]["fallback_rate"].isna().all()
    ok(f"{label}: summary CSV strategies/counts per block, no OVERALL rows, fallback_rate")


def eth_supply_override_tests(fixture: pathlib.Path, reference_out: pathlib.Path) -> None:
    """--eth-supply-csv must change ETH's market-cap weight (and nothing else)."""
    print("[integration: --eth-supply-csv override]")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="t50_ethsup_int_"))
    pd.DataFrame({"date": list(BLOCKS.values()),
                  "circulating_supply": [4_000_000.0] * len(BLOCKS)}).to_csv(tmp / "eth_supply.csv", index=False)
    out = pathlib.Path(tempfile.mkdtemp(prefix="t50_override_"))
    r = run_backtest(fixture, fixture / "holdings.parquet", out,
                     extra_args=["--eth-supply-csv", str(tmp / "eth_supply.csv")])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "ETH supply for 2 snapshot blocks taken from" in r.stdout, r.stdout[-2000:]
    ok("override: pipeline exits 0 and logs the override")
    for block in BLOCKS:
        ref = ds.dataset(str(reference_out / "wallet_level.parquet" / f"block_number={block}" / "data.parquet"),
                         format="parquet").to_table().to_pandas().set_index("wallet_address")
        got = ds.dataset(str(out / "wallet_level.parquet" / f"block_number={block}" / "data.parquet"),
                         format="parquet").to_table().to_pandas().set_index("wallet_address")
        # rows are written in worker-completion order, so align on the wallet
        assert sorted(got.index) == sorted(ref.index), (sorted(got.index), sorted(ref.index))
        got = got.loc[ref.index]
        # W1 holds TOKA + (W)ETH: quadrupling ETH's supply moves the mcap weights
        assert np.isfinite(got.loc[W1, "l1_baseline_vs_mcap"]) and np.isfinite(ref.loc[W1, "l1_baseline_vs_mcap"])
        assert abs(got.loc[W1, "l1_baseline_vs_mcap"] - ref.loc[W1, "l1_baseline_vs_mcap"]) > 1e-9, \
            (got.loc[W1, "l1_baseline_vs_mcap"], ref.loc[W1, "l1_baseline_vs_mcap"])
        assert np.isfinite(got.loc[W1, "ret_mcap_weight"])
        # everything that does not depend on supply is untouched
        for col in ["ret_baseline", "ret_equal_weight", "ret_max_sharpe", "l1_gap_max_sharpe", "market_return"]:
            assert np.allclose(got[col].to_numpy(dtype=float), ref[col].to_numpy(dtype=float), equal_nan=True), col
        ok(f"override: block {block} mcap weights changed for W1, other columns identical")


def integration_tests(fixture: pathlib.Path) -> None:
    print("[integration: single-file holdings]")
    out1 = pathlib.Path(tempfile.mkdtemp(prefix="t50_single_"))
    r = run_backtest(fixture, fixture / "holdings.parquet", out1)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    ok("pipeline exits 0")
    check_outputs(fixture, out1, "single-file")
    eth_supply_override_tests(fixture, out1)

    print("[integration: idempotent re-run (upsert)]")
    n_before = len(pd.read_csv(out1 / "summary.csv"))
    r = run_backtest(fixture, fixture / "holdings.parquet", out1)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    n_after = len(pd.read_csv(out1 / "summary.csv"))
    assert n_before == n_after, (n_before, n_after)
    check_outputs(fixture, out1, "re-run")
    ok(f"re-run leaves summary at {n_after} rows (no duplicates)")

    print("[integration: 4_2-style Hive dataset input]")
    out2 = pathlib.Path(tempfile.mkdtemp(prefix="t50_hive_"))
    r = run_backtest(fixture, fixture / "holdings_hive", out2)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    check_outputs(fixture, out2, "hive")

    single = pd.read_csv(out1 / "summary.csv").sort_values(
        ["block", "wealth_bucket", "strategy"]).reset_index(drop=True)
    hive = pd.read_csv(out2 / "summary.csv").sort_values(
        ["block", "wealth_bucket", "strategy"]).reset_index(drop=True)
    num = single.select_dtypes("number").columns
    pd.testing.assert_frame_equal(single[num], hive[num],
                                  check_exact=False, atol=1e-9)
    ok("hive-dataset run reproduces the single-file summary numerically")

    print("[integration: --max-wallets cap across multiple blocks]")
    # Regression: a break inside the per-block finally used to abort the
    # WHOLE block loop once the cap was hit — later blocks were silently
    # skipped and the capped block never got its summary rows.
    out3 = pathlib.Path(tempfile.mkdtemp(prefix="t50_maxw_"))
    r = run_backtest(fixture, fixture / "holdings.parquet", out3,
                     extra_args=["--blocks", ",".join(str(b) for b in BLOCKS),
                                 "--max-wallets", "2"])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for block in BLOCKS:
        part = out3 / "wallet_level.parquet" / f"block_number={block}" / "data.parquet"
        assert part.exists(), f"--max-wallets skipped block {block}"
    summary = pd.read_csv(out3 / "summary.csv", dtype={"block": str})
    assert set(summary["block"]) == {str(b) for b in BLOCKS}, \
        sorted(set(summary["block"]))
    assert int(summary["count_optimized"].max()) <= 2, \
        summary["count_optimized"].max()
    ok("--max-wallets: every block processed + summarized, cap respected")
    shutil.rmtree(out3)

    shutil.rmtree(out1)
    shutil.rmtree(out2)


def main() -> int:
    fixture = build_fixture(TESTS_DIR / "fixture_5_0")
    print(f"fixture: {fixture}")
    m = load_module()
    unit_tests(m)
    pushdown_tests(m, fixture)
    integration_tests(fixture)
    print(f"\n5_0 TESTS PASSED ({PASSED} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
