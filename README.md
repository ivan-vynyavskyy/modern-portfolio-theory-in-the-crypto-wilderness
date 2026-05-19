# Modern Portfolio Theory in the Crypto-Wilderness

This repository accompanies the paper **"Modern Portfolio Theory in the Crypto-Wilderness"**.
We reconstruct actual on-chain cryptoasset portfolios from raw Ethereum data and
study them through the lens of mean-variance theory.

## Research objectives

The paper pursues three objectives:

- **Obj A — Composition & diversification.** Characterize the composition
  and diversification patterns of actual on-chain cryptoasset portfolios.
- **Obj B — Deviation from the efficient frontier.** Quantify the extent to
  which these portfolios deviate from the mean-variance efficient frontier
  under standard optimization techniques (minimum variance, maximum return,
  maximum Sharpe ratio).
- **Obj C — Deviation effects.** Examine how such deviations affect
  portfolio performance relative to the market.

## Repository layout

```
.
├── PortfolioReconstruction/   # data pipeline: raw Ethereum data → wallet portfolio snapshots
│   ├── src/                   # 17 step scripts (0_0 → 5_0)
│   ├── helpers/, utils/       # shared code reused across steps
│   ├── run_pipeline.py        # YAML-driven orchestrator (runs all steps)
│   └── pipeline_config.example.yaml
└── Analytics/                 # downstream scripts and notebooks for Obj A / B / C
    ├── portfolio_characteristics/
    ├── distance_return_analytics/
    └── shared/
```

The two top-level folders map directly onto the contributions:

| Folder | Produces | Drives |
|---|---|---|
| `PortfolioReconstruction/` | The vetted token universe, daily prices, per-token transfer ledgers, wallet-level holdings snapshots, and the MPT backtest summaries. | The reconstruction-method contribution; supplies inputs for all three research objectives. |
| `Analytics/` | Scripts and notebooks that turn those snapshots into the figures, tables and statistics reported in the paper. | Composition stats (Obj A), ℓ₁ distance to the frontier (Obj B), CAPM / forward-return analysis (Obj C). |

---

# `PortfolioReconstruction/` – the data pipeline

## Quickstart – run the whole pipeline

The pipeline ships with `run_pipeline.py`, a Python driver that executes every
step from `0_0` through `5_0` in order, based on a single YAML config.

1. Install dependencies (CoinGecko, RPC, parquet, etc.) and `pip install pyyaml`.
2. Copy the example config and fill in your secrets / paths:

   ```bash
   cp PortfolioReconstruction/pipeline_config.example.yaml PortfolioReconstruction/pipeline_config.yaml
   $EDITOR PortfolioReconstruction/pipeline_config.yaml
   ```

   At minimum you must provide:
   - `coingecko_api_key` – CoinGecko **Pro** API key.
   - `rpc_url` – Ethereum JSON-RPC HTTP endpoint (Infura/Alchemy/your own node).
   - `output_root` – where derived datasets land.
   - The on-chain inputs the scripts cannot fetch themselves: `traces_dir`
     (Delta Lake of Ethereum traces, used by `0_2`), `raw_logs_dir`
     (raw `Transfer`/`Deposit`/`Withdrawal` logs, used by `2_0`),
     `block_to_date_csv`, `input_wallets_parquet`, and the scratch dirs for
     intermediate map/reduce output.

3. Run the pipeline:

   ```bash
   cd PortfolioReconstruction
   python run_pipeline.py --config pipeline_config.yaml
   ```

   Useful flags:
   - `--from-step 1_2` – resume after a failure starting at step `1_2`.
   - `--list-steps` – print the ordered step list and exit.

The driver fails loudly the moment a required configuration value is missing,
an input path does not exist, or any sub-script returns a non-zero exit code.

## Extending the config – running a step with a non-default flag

Each `src/N_M_*.py` script has its own CLI (`python src/<script>.py --help`).
The runner (`run_pipeline.py`) only forwards a curated subset of those flags via
the YAML config. If you want to run a step with a flag that isn't exposed yet
(for example, `--no-events` on `0_2`, `--sleep` on `1_3`, `--max-tokens` on
`4_0`), follow this 3-step recipe:

1. **Add the new key to both YAML files.** Pick a clear name and add it to
   `pipeline_config.example.yaml` (so others see it) and to your real
   `pipeline_config.yaml`.

   ```yaml
   # pipeline_config.yaml
   trace_scan_require_events: false   # new knob: pass --no-events to step 0_2
   ```

2. **Wire it into the matching builder in `run_pipeline.py`.** Each step has a
   `_b_<step_id>` function that returns the CLI args. Read your new key from
   `ctx["raw"]` and append the right flag.

   ```python
   # in _b_0_2()
   def _b_0_2(ctx):
       _require_path(ctx["traces_dir"], "Traces dir", "traces_dir")
       out = ctx["contract_addresses_tsv"]
       _ensure_parent(out)
       args = ["--traces-dir", str(ctx["traces_dir"]), "--out-tsv", str(out)]
       if ctx["raw"].get("trace_scan_require_events") is False:
           args.append("--no-events")
       return args
   ```

3. **For value-bearing flags use the "only emit if set" pattern**, so leaving
   the YAML key blank falls back to the script's own default:

   ```python
   sleep = ctx["raw"].get("supply_fetch_sleep_seconds")
   if sleep is not None:
       args += ["--sleep", str(sleep)]
   ```

Existing examples to copy from: `volume_lookback_days` → `_b_1_2`,
`mapreduce_batch` / `mapreduce_timeout_seconds` → `_b_2_0`,
`reconstruction_processes` (optional, only emitted if set) → `_b_4_0`,
`backtest_blocks` (optional path or comma-list) → `_b_5_0`.

Verify the wiring with `python run_pipeline.py --list-steps` and then resume
the pipeline from the affected step:

```bash
python run_pipeline.py --config pipeline_config.yaml --from-step 0_2
```

The same recipe + a worked example also lives at the top of the
"Step builders" section in `run_pipeline.py`.

## Portfolio reconstruction workflow

1. **Coin universe** – run [`PortfolioReconstruction/src/0_0_extract_all_coingecko_coins.py`](PortfolioReconstruction/src/0_0_extract_all_coingecko_coins.py) to dump the full CoinGecko list (with platform metadata) into `data/all_coins_coingecko.csv.gz`.
2. **Ethereum subset** – run [`PortfolioReconstruction/src/0_1_extract_eth_tokens_from_coingecko.py`](PortfolioReconstruction/src/0_1_extract_eth_tokens_from_coingecko.py) to parse the mixed `platforms` column and keep rows with Ethereum contracts, saved as `derived_coingecko_eth_tokens.csv`.
3. **On-chain discovery** – run [`PortfolioReconstruction/src/0_2_scan_traces_for_erc20_candidates.py`](PortfolioReconstruction/src/0_2_scan_traces_for_erc20_candidates.py) on trace parquet shards to spot CREATE/CREATE2 deployments whose runtime bytecode matches ERC‑20 selectors, writing a TSV of addresses plus first-seen block.
4. **Status join** – run [`PortfolioReconstruction/src/0_3_mark_status_with_scanner_hits.py`](PortfolioReconstruction/src/0_3_mark_status_with_scanner_hits.py) to join the CoinGecko table with scanner hits and flag tokens that were actually deployed on-chain.
5. **Proxy verification** – run [`PortfolioReconstruction/src/0_4_resolve_proxies_and_verify_erc20.py`](PortfolioReconstruction/src/0_4_resolve_proxies_and_verify_erc20.py) to resolve common proxy patterns via JSON-RPC, pull metadata, and check static/runtime ERC‑20 behavior.
6. **Status update** – run [`PortfolioReconstruction/src/0_5_update_status_with_erc20_verification.py`](PortfolioReconstruction/src/0_5_update_status_with_erc20_verification.py) to flip `is_erc_20` to true only when the proxy resolver confirmed both checks.
7. **Price history** – run [`PortfolioReconstruction/src/1_0_fetch_token_price_history.py`](PortfolioReconstruction/src/1_0_fetch_token_price_history.py) to download full CoinGecko daily close curves for every approved contract and append status logs.
8. **Price status merge** – run [`PortfolioReconstruction/src/1_1_update_token_status_with_price.py`](PortfolioReconstruction/src/1_1_update_token_status_with_price.py) to augment `status_eth_tokens.csv` with the latest price download metadata and summary stats.
9. **Volume sanity check** – run [`PortfolioReconstruction/src/1_2_check_trading_volume_status.py`](PortfolioReconstruction/src/1_2_check_trading_volume_status.py) to scan `status_eth_tokens.csv`, fetch 10-day vs all-time CoinGecko volumes, and flag ERC‑20 tokens whose trading activity is insufficient for downstream analysis.
10. **FDV & market-cap snapshot** – run [`PortfolioReconstruction/src/1_3_fetch_supply_and_mcap_from_cg.py`](PortfolioReconstruction/src/1_3_fetch_supply_and_mcap_from_cg.py) to pull circulating/total/maximum supply plus FDV and market-cap data (including a synthetic ETH row) for every approved token.
11. **ETH cap comparison** – run [`PortfolioReconstruction/src/1_4_update_status_fdv_mcap_vs_eth.py`](PortfolioReconstruction/src/1_4_update_status_fdv_mcap_vs_eth.py) to merge the status sheet with the CoinGecko metrics and append `fdv_mcap_lt_eth`, marking tokens whose FDV and market cap both sit below Ethereum's market cap.
12. **Transfer map-reduce** – run [`PortfolioReconstruction/src/2_0_erc20_transfer_mapreduce.py`](PortfolioReconstruction/src/2_0_erc20_transfer_mapreduce.py) to map/reduce raw log shards into per-token `Transfer` datasets, including wrapped ETH deposit/withdraw handling.
13. **Ledger builder** – run [`PortfolioReconstruction/src/2_1_erc20_transfer_aggregator.py`](PortfolioReconstruction/src/2_1_erc20_transfer_aggregator.py) to convert each token's shards into a sorted debit/credit ledger per address.
14. **Validation** – run [`PortfolioReconstruction/src/3_0_validation_runner.py`](PortfolioReconstruction/src/3_0_validation_runner.py) to sample ledger rows, compare cumulative balances to live `balanceOf()` calls, and record pass/fail statuses.
15. **Local validation status** – run [`PortfolioReconstruction/src/3_1_add_local_status.py`](PortfolioReconstruction/src/3_1_add_local_status.py) to append the `local_status` column (PASS / NOT_PASS / NO_DATA) from the validation TSV onto the token status CSV.
16. **Portfolio reconstruction** – run [`PortfolioReconstruction/src/4_0_portfolio_reconstruction_pipeline.py`](PortfolioReconstruction/src/4_0_portfolio_reconstruction_pipeline.py) to ingest the vetted token list, price curves, and transfer ledgers, reconstruct portfolios, and compute the paper's MPT metrics.
17. **MPT backtest** – run [`PortfolioReconstruction/src/5_0_portfolio_backtest_pipeline.py`](PortfolioReconstruction/src/5_0_portfolio_backtest_pipeline.py) to stream the holdings parquet block-by-block, optimize max-Sharpe / min-variance / max-return-same-vol portfolios per wallet, evaluate forward returns, and write per-block plus overall summary tables alongside a wallet-level parquet dataset.

---

# `Analytics/` – downstream scripts, notebooks and plots

`Analytics/` consumes the holdings parquet and the per-block backtest output
produced by `PortfolioReconstruction/` and turns them into the statistics,
figures and CAPM/ML analyses reported in the paper. Each subfolder targets
one of the three research objectives.

## Layout

```
Analytics/
├── generate_all_macros.py             # one-shot: rebuild macro.tex from CSV/MD outputs
├── manual_constants.py                # macros that can't be derived from data
├── macro.tex                          # generated LaTeX macros consumed by the paper
├── portfolio_characteristics/         # Obj A — composition & diversification
│   ├── accounts_overview/             #   EOA vs CA counts over time, unique-token stats
│   ├── distribution_of_wealth/        #   inequality, top-share, wealth/TVL alignment
│   ├── portfolio_size/                #   #tokens-per-wallet stats by wealth bucket
│   ├── per_block_wallet_stats/        #   per-block wallet aggregates
│   └── token_holder_counts/           #   yearly holder-count buckets per token
├── distance_return_analytics/         # Obj B & C — frontier distance + forward returns
│   ├── src/                           #   analyses: distance, CAPM, RF+SHAP, hypothesis tests
│   ├── plot/                          #   one R script per figure
│   ├── data/                          #   intermediate CSVs (input to plots and macros)
│   └── graphics/                      #   final PDF figures
├── token_metadata/                    # ad-hoc CoinGecko volume monitor
└── shared/                            # locations.py, plot_theme.R, ethereum_codec_utils.py
```

Inside each topic folder under `portfolio_characteristics/` and inside
`distance_return_analytics/` the convention is the same:

| Subdir | Holds |
|---|---|
| `src/` | Python `.py`/`.ipynb` scripts that read large parquet inputs and emit intermediate CSV/Markdown into the sibling `data/` folder. |
| `data/` | Per-topic CSVs (one per figure / table) plus a few `.txt`/`.md` analysis logs. These are the inputs that the macro generator and R plot scripts consume. |
| `plot/` | One R script per figure. Each reads a CSV from `data/`, applies `shared/plot_theme.R`, and writes a PDF into `graphics/`. |
| `graphics/` | Final PDFs that the paper imports. |

The two top-level folders feed each other:

| Folder | Produces | Drives |
|---|---|---|
| `Analytics/portfolio_characteristics/` | Composition / inequality / portfolio-size CSVs and figures. | Obj A sections of the paper. |
| `Analytics/distance_return_analytics/` | ℓ₁ distance distributions, OLS / RF / SHAP feature importances, per-block returns, CAPM betas/alphas. | Obj B (frontier distance) and Obj C (returns & CAPM). |
| `Analytics/generate_all_macros.py` | A single `macro.tex` aggregating every numeric quantity referenced in the paper. | Bridges the analyses above into the LaTeX manuscript so every number is reproducible. |

## Configuring data locations

Every script reads big-input paths from
[`Analytics/shared/locations.py`](Analytics/shared/locations.py). The
`Location.DATA_ROOT` default is the placeholder `/anonymized`; point it at
the directory that holds the parquet outputs from
`PortfolioReconstruction/` before running anything that touches the
large datasets:

```python
# Analytics/shared/locations.py
class Location:
    DATA_ROOT = Path("/your/local/data_root")
    ...
```

All other paths derive from `DATA_ROOT`, so a one-line change makes the
whole Analytics tree run against your machine.

## Analytics workflow by objective

### Obj A — Composition & diversification (`portfolio_characteristics/`)

1. **Account universe** – run [`Analytics/portfolio_characteristics/accounts_overview/src/scan_traces_for_ca_addresses.py`](Analytics/portfolio_characteristics/accounts_overview/src/scan_traces_for_ca_addresses.py) plus the three notebooks (`get_eos_ca_over_time.ipynb`, `get_unique_tokens.ipynb`, `get_wallet_counts_acquired.ipynb`) to count EOA vs CA accounts over time and the unique-token universe.
2. **Wealth distribution** – run the scripts under [`Analytics/portfolio_characteristics/distribution_of_wealth/src/`](Analytics/portfolio_characteristics/distribution_of_wealth/src) to compute value buckets per block, top-share concentration by year, CA/EOA breakdowns of the top quantiles, GraphSense tag coverage, and the inequality-measures notebook. Outputs feed the wealth/Gini/Lorenz figures.
3. **Per-block wallet stats** – run [`Analytics/portfolio_characteristics/per_block_wallet_stats/src/analyze_portfolio_per_block.py`](Analytics/portfolio_characteristics/per_block_wallet_stats/src/analyze_portfolio_per_block.py) to emit per-block wallet aggregates and asset-bucket counts.
4. **Portfolio size** – run [`Analytics/portfolio_characteristics/portfolio_size/src/portfolio_size_by_wealth_bucket.py`](Analytics/portfolio_characteristics/portfolio_size/src/portfolio_size_by_wealth_bucket.py) to produce the per-wealth-bucket #tokens histograms and statistics.
5. **Token holder counts** – run [`Analytics/portfolio_characteristics/token_holder_counts/src/analyze_holders_distribution_by_year.py`](Analytics/portfolio_characteristics/token_holder_counts/src/analyze_holders_distribution_by_year.py) for the yearly holder-bucket breakdown per token.
6. **Plot** – the matching `plot/*.R` scripts in each topic turn those CSVs into the figures used in Obj A.

### Obj B — Frontier distance (`distance_return_analytics/`)

7. **Per-block ℓ₁ distance** – run [`Analytics/distance_return_analytics/src/distance_per_block_analysis.py`](Analytics/distance_return_analytics/src/distance_per_block_analysis.py) to aggregate ℓ₁ distance statistics across all wallets per block, per strategy.
8. **By token group / wealth group** – run [`Analytics/distance_return_analytics/src/distance_per_block_by_token_group.py`](Analytics/distance_return_analytics/src/distance_per_block_by_token_group.py) and [`distance_per_wealth_group.py`](Analytics/distance_return_analytics/src/distance_per_wealth_group.py) for the stratified breakdowns + paired Wilcoxon tests.
9. **Full-data L1 analysis** – run [`Analytics/distance_return_analytics/src/mpt_distance_analysis.py`](Analytics/distance_return_analytics/src/mpt_distance_analysis.py) for the full 239M-row regression suite (OLS, quantile regression, RF importance) — needs DuckDB and ~128 GB RAM.
10. **Sub-portfolios (≥5 tokens) robustness** – same `mpt_distance_analysis.py` flow restricted to the `_gr_5` subset; outputs land in `data/distance_analysis_results_gr_5/`.
11. **Threshold sweep** – run [`Analytics/distance_return_analytics/src/hypothesis_test_threshold_sweep.py`](Analytics/distance_return_analytics/src/hypothesis_test_threshold_sweep.py) for paired Wilcoxon sweeps across various `N>=` cut-offs.
12. **Formula fit** – run [`Analytics/distance_return_analytics/src/fit_l1_formulas.py`](Analytics/distance_return_analytics/src/fit_l1_formulas.py) for the power-law fits used in the "distance vs portfolio size" claim.
13. **Plots** – distance-over-time, ridge plots, distribution buckets, token-vs-distance, ℓ₁-fit-vs-actual figures are produced by the matching scripts under [`Analytics/distance_return_analytics/plot/`](Analytics/distance_return_analytics/plot).

### Obj C — Returns & CAPM (`distance_return_analytics/`)

14. **Per-block returns / alphas** – run [`Analytics/distance_return_analytics/src/mpt_return_analysis.py`](Analytics/distance_return_analytics/src/mpt_return_analysis.py) for per-block median returns, alpha distributions, and the strategy-direction tables.
15. **Extended CAPM** – run [`Analytics/distance_return_analytics/src/analyze_extended_capm.py`](Analytics/distance_return_analytics/src/analyze_extended_capm.py) for portfolio-beta drift across strategies, beta-by-#tokens, and l1-vs-alpha binned correlations.
16. **% beating market** – run [`Analytics/distance_return_analytics/src/analyze_pct_beat_market.py`](Analytics/distance_return_analytics/src/analyze_pct_beat_market.py) for the per-block "% of wallets that beat the market" curve.
17. **RF + SHAP return drivers** – run [`Analytics/distance_return_analytics/src/rf_shap_analysis.py`](Analytics/distance_return_analytics/src/rf_shap_analysis.py) (3-feature baseline) and [`rf_shap_analysis_extended.py`](Analytics/distance_return_analytics/src/rf_shap_analysis_extended.py) (6-feature) for global feature importance + SHAP values that drive the "market entry timing dominates" claim.
18. **Risk-free-rate robustness** – run [`Analytics/distance_return_analytics/src/rf_zero_test.py`](Analytics/distance_return_analytics/src/rf_zero_test.py) for the rf=0 vs rf=5% MSR weight-distance sensitivity check.
19. **Stablecoin lending rates** – run [`Analytics/distance_return_analytics/src/fetch_stablecoin_lending_rates.py`](Analytics/distance_return_analytics/src/fetch_stablecoin_lending_rates.py) to fetch USDT/USDC APYs from DefiLlama; informs the risk-free-rate benchmark.
20. **Plots** – median-return ribbons/ridges, alpha ridges, "beat the market" quantile boxes, panel-delta-vs-market, RF feature-importance and SHAP summary figures are produced by the matching scripts under [`Analytics/distance_return_analytics/plot/`](Analytics/distance_return_analytics/plot).

### Final step — assemble the paper macros

21. **Macros** – run [`Analytics/generate_all_macros.py`](Analytics/generate_all_macros.py) once everything above has produced the CSVs under `data/`. It emits [`Analytics/macro.tex`](Analytics/macro.tex), the single LaTeX include that resolves every numeric reference in the paper.