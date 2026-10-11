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
│   ├── erc20_src/             # 15 step scripts (0_0 → 3_1) building the ERC-20 universe and ledgers
│   ├── eth_src/               # 8 step scripts (0_0 → 5_0) building the native-ETH ledger, plus the 6_0 self-test
│   ├── common_src/            # steps consuming BOTH ledgers (eth + erc20): 4_0 reconstruction,
│   │                          #   4_1 verify, 4_2 partition, 5_0 backtest (+ 4_3, 4_4 standalone checks)
│   ├── helpers/, utils/       # shared code reused across steps
│   ├── tests/                 # 5_0 backtest test on synthetic holdings
│   ├── run_pipeline.py        # YAML-driven orchestrator (runs all three streams)
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

The pipeline ships with `run_pipeline.py`, a Python driver that executes all
steps in order, based on a single YAML config. The steps form three streams,
and every step id carries its stream:

| Stream | Folder | Steps | Builds |
|---|---|---|---|
| `erc20` | `erc20_src/` | `erc20.0_0` → `erc20.3_1` | token universe, prices, per-token ledgers |
| `eth` | `eth_src/` | `eth.0_0` → `eth.5_0` | native-ETH transfer and flow ledger, ETH price |
| `common` | `common_src/` | `common.4_0` → `common.5_0` | holdings snapshots, verification, partitioning, backtest |

1. Install the dependencies (CoinGecko, RPC, parquet, PyYAML for the runner, etc.) with `pip install -r requirements.txt`.
2. Copy the example config and fill in your secrets / paths:

   ```bash
   cp PortfolioReconstruction/pipeline_config.example.yaml PortfolioReconstruction/pipeline_config.yaml
   $EDITOR PortfolioReconstruction/pipeline_config.yaml
   ```

   At minimum you must provide:
   - `coingecko_api_key` – CoinGecko **Pro** API key.
   - `rpc_url` – Ethereum JSON-RPC HTTP endpoint (Infura/Alchemy/your own
     node). The validation steps query past blocks, so it must be an archive node.
   - `output_root` – where derived datasets land.
   - The on-chain inputs the scripts cannot fetch themselves: `traces_dir`
     (Delta Lake of Ethereum traces, used by `erc20.0_2`), `raw_logs_dir`
     (raw `Transfer`/`Deposit`/`Withdrawal` logs, used by `erc20.2_0`),
     `eth_dump_dir` (raw Parquet dump with `trace/`, `transaction/` and
     `block/`, used by `eth.0_0` and `eth.1_0`), `block_to_date_csv`,
     `input_wallets_parquet`, and the scratch dirs for intermediate
     map/reduce output.

3. Run the pipeline:

   ```bash
   cd PortfolioReconstruction
   python run_pipeline.py --config pipeline_config.yaml
   ```

   Useful flags:
   - `--from-step erc20.1_2` – resume after a failure starting at that step.
     A bare id such as `1_2` works when it is unique across the streams.
   - `--to-step eth.5_0` – stop after that step, e.g.
     `--from-step eth.0_0 --to-step eth.5_0` builds only the ETH ledger.
   - `--dry-run` – print the command of every selected step without running it.
   - `--list-steps` – print the ordered step list and exit.

The driver fails loudly the moment a required configuration value is missing,
an input path does not exist, or any sub-script returns a non-zero exit code.

## Extending the config – running a step with a non-default flag

Each `N_M_*.py` step script has its own CLI (`python <dir>/<script>.py --help`;
the ERC-20 steps live in `erc20_src/`, the ETH steps in `eth_src/`, and the
steps 4_x and 5_0 over both ledgers in `common_src/`).
The runner (`run_pipeline.py`) only forwards a curated subset of those flags via
the YAML config. If you want to run a step with a flag that isn't exposed yet
(for example, `--no-events` on `erc20.0_2`, `--sleep` on `erc20.1_3`,
`--max-tokens` on `common.4_0`), follow this 3-step recipe:

1. **Add the new key to both YAML files.** Pick a clear name and add it to
   `pipeline_config.example.yaml` (so others see it) and to your real
   `pipeline_config.yaml`.

   ```yaml
   # pipeline_config.yaml
   trace_scan_require_events: false   # new knob: pass --no-events to step 0_2
   ```

2. **Wire it into the matching builder in `run_pipeline.py`.** Each step has a
   builder function that returns the CLI args (`_b_0_2` for `erc20.0_2`,
   `_b_eth_1_0` for `eth.1_0`, `_b_4_1` for `common.4_1`). Read your new key
   from `ctx["raw"]` and append the right flag.

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
`eth_ledger_shards` → `_b_eth_2_0`,
`reconstruction_processes` (optional, only emitted if set) → `_b_4_0`,
`backtest_blocks` (optional path or comma-list) → `_b_5_0`.

Check the resulting command with `--dry-run` and then resume the pipeline from
the affected step:

```bash
python run_pipeline.py --config pipeline_config.yaml --from-step erc20.0_2 --to-step erc20.0_2 --dry-run
python run_pipeline.py --config pipeline_config.yaml --from-step erc20.0_2
```

The same recipe + a worked example also lives at the top of the
"Step builders" section in `run_pipeline.py`.

## Portfolio reconstruction workflow

1. **Coin universe** – run [`PortfolioReconstruction/erc20_src/0_0_extract_all_coingecko_coins.py`](PortfolioReconstruction/erc20_src/0_0_extract_all_coingecko_coins.py) to dump the full CoinGecko list (with platform metadata) into `data/all_coins_coingecko.csv.gz`.
2. **Ethereum subset** – run [`PortfolioReconstruction/erc20_src/0_1_extract_eth_tokens_from_coingecko.py`](PortfolioReconstruction/erc20_src/0_1_extract_eth_tokens_from_coingecko.py) to parse the mixed `platforms` column and keep rows with Ethereum contracts, saved as `derived_coingecko_eth_tokens.csv`.
3. **On-chain discovery** – run [`PortfolioReconstruction/erc20_src/0_2_scan_traces_for_erc20_candidates.py`](PortfolioReconstruction/erc20_src/0_2_scan_traces_for_erc20_candidates.py) on trace parquet shards to spot CREATE/CREATE2 deployments whose runtime bytecode matches ERC‑20 selectors, writing a TSV of addresses plus first-seen block.
4. **Status join** – run [`PortfolioReconstruction/erc20_src/0_3_mark_status_with_scanner_hits.py`](PortfolioReconstruction/erc20_src/0_3_mark_status_with_scanner_hits.py) to join the CoinGecko table with scanner hits and flag tokens that were actually deployed on-chain.
5. **Proxy verification** – run [`PortfolioReconstruction/erc20_src/0_4_resolve_proxies_and_verify_erc20.py`](PortfolioReconstruction/erc20_src/0_4_resolve_proxies_and_verify_erc20.py) to resolve common proxy patterns via JSON-RPC, pull metadata, and check static/runtime ERC‑20 behavior. [`PortfolioReconstruction/helpers/proxy_resolver_summary.py`](PortfolioReconstruction/helpers/proxy_resolver_summary.py) prints the classification and verification tables of its output (standalone, not part of the runner).
6. **Status update** – run [`PortfolioReconstruction/erc20_src/0_5_update_status_with_erc20_verification.py`](PortfolioReconstruction/erc20_src/0_5_update_status_with_erc20_verification.py) to flip `is_erc_20` to true only when the proxy resolver confirmed both checks.
7. **Price history** – run [`PortfolioReconstruction/erc20_src/1_0_fetch_token_price_history.py`](PortfolioReconstruction/erc20_src/1_0_fetch_token_price_history.py) to download full CoinGecko daily close curves for every approved contract and append status logs.
8. **Price status merge** – run [`PortfolioReconstruction/erc20_src/1_1_update_token_status_with_price.py`](PortfolioReconstruction/erc20_src/1_1_update_token_status_with_price.py) to augment `status_eth_tokens.csv` with the latest price download metadata and summary stats.
9. **Volume sanity check** – run [`PortfolioReconstruction/erc20_src/1_2_check_trading_volume_status.py`](PortfolioReconstruction/erc20_src/1_2_check_trading_volume_status.py) to scan `status_eth_tokens.csv`, fetch 10-day vs all-time CoinGecko volumes, and flag ERC‑20 tokens whose trading activity is insufficient for downstream analysis.
10. **FDV & market-cap snapshot** – run [`PortfolioReconstruction/erc20_src/1_3_fetch_supply_and_mcap_from_cg.py`](PortfolioReconstruction/erc20_src/1_3_fetch_supply_and_mcap_from_cg.py) to pull circulating/total/maximum supply plus FDV and market-cap data (including a synthetic ETH row) for every approved token.
11. **ETH cap comparison** – run [`PortfolioReconstruction/erc20_src/1_4_update_status_fdv_mcap_vs_eth.py`](PortfolioReconstruction/erc20_src/1_4_update_status_fdv_mcap_vs_eth.py) to merge the status sheet with the CoinGecko metrics and append `fdv_mcap_lt_eth`, marking tokens whose FDV and market cap both sit below Ethereum's market cap.
12. **Transfer map-reduce** – run [`PortfolioReconstruction/erc20_src/2_0_erc20_transfer_mapreduce.py`](PortfolioReconstruction/erc20_src/2_0_erc20_transfer_mapreduce.py) to map/reduce raw log shards into per-token `Transfer` datasets, including wrapped ETH deposit/withdraw handling.
13. **Ledger builder** – run [`PortfolioReconstruction/erc20_src/2_1_erc20_transfer_aggregator.py`](PortfolioReconstruction/erc20_src/2_1_erc20_transfer_aggregator.py) to convert each token's shards into a sorted debit/credit ledger per address.
14. **Validation** – run [`PortfolioReconstruction/erc20_src/3_0_validation_runner.py`](PortfolioReconstruction/erc20_src/3_0_validation_runner.py) to sample ledger rows, compare cumulative balances to live `balanceOf()` calls, and record pass/fail statuses.
15. **Local validation status** – run [`PortfolioReconstruction/erc20_src/3_1_add_local_status.py`](PortfolioReconstruction/erc20_src/3_1_add_local_status.py) to append the `local_status` column (PASS / NOT_PASS / NO_DATA) from the validation TSV onto the token status CSV.
16. **Portfolio reconstruction** – run [`PortfolioReconstruction/common_src/4_0_portfolio_reconstruction_pipeline.py`](PortfolioReconstruction/common_src/4_0_portfolio_reconstruction_pipeline.py) to ingest the vetted token list, price curves, and transfer ledgers (ERC-20 **and** the native-ETH pseudo-token, see `eth_src/` below), reconstruct portfolios, and emit the per-wallet holdings snapshots.
17. **Holdings verification** – run [`PortfolioReconstruction/common_src/4_1_verify_holdings.py`](PortfolioReconstruction/common_src/4_1_verify_holdings.py) to validate the holdings parquet in two layers: internal invariants over the full file (streaming, no RPC) and ground-truth spot-checks against an archive node (`eth_getBalance` / `balanceOf` at the snapshot block, plus absent-token probes).
18. **Partitioning** – run [`PortfolioReconstruction/common_src/4_2_partition_holdings.py`](PortfolioReconstruction/common_src/4_2_partition_holdings.py) to write the holdings into one Hive-partitioned dataset (`block_number=<N>/` directories plus a `_manifest.csv` with block → date → rows → source-run provenance) that `5_0` reads with partition pruning. When `4_0` was run per time period, it merges the per-period parquets and refuses snapshot blocks that two runs both cover.
19. **MPT backtest** – run [`PortfolioReconstruction/common_src/5_0_portfolio_backtest_pipeline.py`](PortfolioReconstruction/common_src/5_0_portfolio_backtest_pipeline.py) to stream the `4_2` dataset (a single holdings parquet works too) block-by-block, optimize max-Sharpe / min-variance / max-return-same-vol portfolios per wallet, evaluate forward returns, and write per-block plus overall summary tables alongside a wallet-level parquet dataset.

`run_pipeline.py` runs steps 1–15 as `erc20.0_0` … `erc20.3_1`, then the native-ETH
steps (`eth.0_0` … `eth.5_0`, next section), then steps 16–19 as `common.4_0` …
`common.5_0`. Two scripts in `common_src/` are standalone checks outside the
runner: [`4_3_check_block_coverage.py`](PortfolioReconstruction/common_src/4_3_check_block_coverage.py)
finds duplicate, missing or unexpected snapshot blocks across per-period `4_0`
outputs before they are merged, and
[`4_4_eth_wealth_distribution.py`](PortfolioReconstruction/common_src/4_4_eth_wealth_distribution.py)
reports the distribution of native-ETH positions in the holdings.

## Native ETH holdings reconstruction (`eth_src/`)

ERC-20 balances can be rebuilt from `Transfer` event logs (steps 12–13 above),
but **native ETH leaves no logs** — it moves through internal calls, gas fees,
block rewards, beacon withdrawals and other protocol-level events. The
[`PortfolioReconstruction/eth_src/`](PortfolioReconstruction/eth_src)
sub-pipeline reconstructs ETH holdings *flow-based* from a full-history
Erigon/ethereum-etl trace dump: it enumerates every balance-changing event
(genesis allocation, value-bearing calls with correct EVM semantics for
`delegatecall`/`callcode`/`staticcall`, contract creations, self-destructs,
pre-Merge rewards, all three gas-fee regimes incl. EIP-1559 burn and EIP-4844
blob fees, post-Shapella beacon withdrawals, the DAO fork) as exact signed
wei flows, so that `balance(addr, B) = Σ credits − Σ debits` up to block `B`.

Its own numbered scripts (`0_0` schema discovery → `1_0` transfer ledger →
`2_0` address-flow aggregation → `3_x` validation → `4_x` stats →
`5_0` ETH price fetch) mirror the `erc20_src/` convention, and `run_pipeline.py`
runs them as `eth.0_0` … `eth.5_0`. They share their paths, flags and chain
constants through `eth_src/config.py`. See
[`eth_src/README.md`](PortfolioReconstruction/eth_src/README.md)
for the full run guide, the output formats and the reconstruction rules.
`6_0_selftest_synthetic.py` runs the reconstruction end to end on a generated
synthetic dump and checks the exact expected ledger, without any cluster data.

The output is a sharded per-address flow ledger (~12.9B rows over blocks
0–23,918,198, i.e. genesis → Dec 2025) in the same format as the ERC-20
ledgers, published under the pseudo-token address `0xee…ee` (18 decimals) so
step 16 (`4_0`) consumes native ETH like any other token. Accuracy is
validated wei-exactly against archive-node `eth_getBalance`: protocol supply
identities hold exactly (genesis 72,009,990.499 ETH; Σ balances = minted −
burned) and ~3,000 random (address, block) spot-checks all matched on the
final ledger.

## Tests

Two self-contained checks need no cluster data and finish in under a minute.
Both are plain scripts, run with `python`:

```bash
python PortfolioReconstruction/tests/test_5_0_backtest.py          # 5_0 on a small deterministic fixture
python PortfolioReconstruction/eth_src/6_0_selftest_synthetic.py   # ETH reconstruction on a synthetic dump
```

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
└── shared/                            # locations.py, snapshots.py, token_whitelist.py, ethereum_codec_utils.py, plot_theme.R
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

The scripts that read the large datasets take their paths as command-line
arguments (`--parquet`, `--data-root`, `--ca-tsv`, ...); each script's docstring
shows an example call with `/path/to/...` placeholders. These inputs are the
outputs of `PortfolioReconstruction/`: the partitioned holdings (`common.4_2`),
the token and ETH ledgers (`erc20.2_1`, `eth.2_0`) and the wallet-level backtest
output (`common.5_0`).
[`Analytics/shared/locations.py`](Analytics/shared/locations.py) collects the
default dataset locations under `Location.DATA_ROOT` (placeholder `/anonymized`),
which `build_address_universe.py` falls back to when its path arguments are
omitted:

```python
# Analytics/shared/locations.py
class Location:
    DATA_ROOT = Path("/your/local/data_root")
    ...
```

The plot scripts and the macro generator only read files from the topics'
`data/` folders, so they need no path configuration.

## Analytics workflow by objective

### Obj A — Composition & diversification (`portfolio_characteristics/`)

1. **Account universe** – run [`Analytics/portfolio_characteristics/accounts_overview/src/scan_traces_for_ca_addresses.py`](Analytics/portfolio_characteristics/accounts_overview/src/scan_traces_for_ca_addresses.py) then, in the same folder, `build_address_universe.py` (distinct addresses of all whitelisted ledgers, the `--input-wallets-parquet` of the reconstruction, plus the transfer count), `count_accounts_eoa_vs_ca.py` (cumulative EOA vs CA accounts per snapshot and the `contract_accounts_in_dataset.tsv` used as `--ca-tsv` downstream) and `count_token_wallets_over_time.py` (cumulative wallets per asset and snapshot). All three read the snapshot list from the `_manifest.csv` of the partitioned holdings and scan the token ledgers only (the account universe is token-active addresses); `--include-eth` adds the native-ETH ledger and thereby the ETH-only addresses.
2. **Wealth distribution** – run the scripts under [`Analytics/portfolio_characteristics/distribution_of_wealth/src/`](Analytics/portfolio_characteristics/distribution_of_wealth/src) to compute value buckets per block (`compute_value_buckets_per_block.py`), top-share concentration (`compute_top_value_concentration_per_block.py`), CA/EOA breakdowns of the top quantiles (`top_pct_ca_eoa_breakdown.py`), per-token inequality measures (`inequality_measures.py`) and the GraphSense/iknaio tag attribution of the top-0.1 % holders (`block_tag_coverage.py`). The latter reads the three `top_addresses_0_1_pct_eth_included_{address_summary,tags,contracts}.parquet` files obtained by querying the distinct addresses of `top_pct_ca_eoa_breakdown.py --top-addresses` against the GraphSense REST API; it always writes address-weighted statistics (`*_by_address.csv`) and, when the cluster-side `(block, address)` parquet is passed via `--addresses`, the pair-weighted statistics the paper's numbers rest on (`category_distribution.csv`, `concept_distribution.csv`, `effective_label_distribution.csv`, `tag_coverage_per_block.csv`), plus `tag_coverage_summary.csv`, `tag_coverage_by_type.csv`, actor/abuse/confidence tables and `tag_attribution_report.txt`. `distribution_of_wealth_macros.py` owns every LaTeX macro derived from these files (hooked from `generate_all_macros.py`; `--yearly-table` prints the yearly top-share table), `compare_erc20_only_vs_eth_included.py` writes the old-versus-new report, and `diagnose_top_accounts.py` lists the top accounts and tokens by value per snapshot. Outputs feed the wealth/Gini/Lorenz figures. The beacon deposit contract `0x00000000219ab540356cbb839cbe05303d7705fa` is excluded as an ETH holder in every value-based script (value buckets, top shares, top-quantile CA/EOA breakdown, inequality measures, per-block wallet stats, portfolio size): it custodies all staked ETH for validators whose claims live on the consensus layer, so it is custody rather than an account's wealth. The token universe is unchanged. The wealth-vs-TVL figure additionally uses the DeFiLlama Ethereum TVL series (`plot/load_defillama_tvl.R`, cached offline in `data/defillama_eth_tvl.csv`; pass `refresh = TRUE` on a machine with internet to refetch).
3. **Per-block wallet stats** – run [`Analytics/portfolio_characteristics/per_block_wallet_stats/src/analyze_portfolio_per_block.py`](Analytics/portfolio_characteristics/per_block_wallet_stats/src/analyze_portfolio_per_block.py) to emit per-block wallet aggregates and asset-bucket counts. `per_block_wallet_stats_macros.py` derives the `Acct*` macros from them (block dates for the yearly top-1 % macros come from `accounts_overview/data/output_accounts_eoa_vs_ca.csv`) and `compare_erc20_only_vs_eth_included.py` writes the old-versus-new report.
4. **Portfolio size** – run [`Analytics/portfolio_characteristics/portfolio_size/src/portfolio_size_by_wealth_bucket.py`](Analytics/portfolio_characteristics/portfolio_size/src/portfolio_size_by_wealth_bucket.py) to produce the per-wealth-bucket #tokens histograms and statistics; `portfolio_size_macros.py` derives the `PortSize*` macros and `compare_erc20_only_vs_eth_included.py` the old-versus-new report.
5. **Token holder counts** – run [`Analytics/portfolio_characteristics/token_holder_counts/src/analyze_holders_distribution_by_year.py`](Analytics/portfolio_characteristics/token_holder_counts/src/analyze_holders_distribution_by_year.py) for the yearly holder-bucket breakdown per token (reads `distribution_of_wealth/data/token_inequality_per_block.csv`, drops the native-ETH row by default); `token_holder_counts_macros.py` derives the `Holders*` macros.
6. **Allocation** – run the scripts under [`Analytics/portfolio_characteristics/allocation/src/`](Analytics/portfolio_characteristics/allocation/src): `build_stablecoin_list.py` writes the curated stablecoin list `data/stablecoins.csv` (every stablecoin-like token of the whitelist classified by hand into USD fiat, crypto, synthetic, T-bill, yield-bearing, LP, unstable, suspect (issuer-treasury supply, possible phantom value), non-USD fiat, commodity or not stable, with `cash_core` / `cash_extended` flags; `--prices-dir` adds a peg check), `stablecoin_cash_share.py` measures the stablecoin cash share per account and the composition of the risky part by cash share, the two-fund separation test (`--cash-def core|extended|extended_fx`), `weight_concentration.py` the largest weight, the effective number of assets and the share of multi-asset portfolios that the 0.9 weight cap of the optimisation changes (`--min-position-usd 1` for the dust-free view), and `trading_vs_drift.py` the active turnover between consecutive snapshots against pure price drift (each snapshot's positions are staged once to `--stage-dir` in address buckets, then joined pair by pair). All four follow the conventions of the other Obj A scripts (ETH and WETH folded, beacon deposit contract excluded, `--ca-tsv` for CAs) and run with `--threads 32 --mem 100GB --temp-dir ... --resume`; `tests/test_allocation.py` checks the three analyses against an independent pandas implementation on synthetic holdings. `allocation_macros.py` turns the concentration, dust and trading CSVs into the `\Eff*` / `\Trade*` macros (called by `generate_all_macros.py`; `--tables` prints the rows of the paper's appendix tables on effective size, dust positions and trading activity), and `plot/plot_effective_diversification.R` and `plot/plot_trading_activity.R` draw the two appendix figures.
7. **Plot** – the matching `plot/*.R` scripts in each topic turn those CSVs into the figures used in Obj A. The two per-token inequality figures (`plot_avg_ineq_combined.R`, `plot_holders_distribution_yearly.R`) drop the native-ETH row, and the former also drops the eight tokens whose supply overflows (one address with a 2^256-scale balance).

### Obj B — Frontier distance (`distance_return_analytics/`)

8. **Per-block ℓ₁ distance and portfolio size** – run [`Analytics/distance_return_analytics/src/distance_per_block_analysis.py`](Analytics/distance_return_analytics/src/distance_per_block_analysis.py) for the per-block ℓ₁ distance statistics per strategy, overall, for N < 5 vs N ≥ 5 and per N bucket (N = assets in the optimised set); the paired Wilcoxon / t tests on block-level means for thresholds N = 3..7 (N = 5 is the main-text split); and the pooled distance per exact N (`tokens_vs_dist_agg.csv`, input of the formula fit and the token-vs-distance figures).
9. **Full-data L1 analysis** – run [`Analytics/distance_return_analytics/src/mpt_distance_analysis.py`](Analytics/distance_return_analytics/src/mpt_distance_analysis.py) for the full 239M-row regression suite (OLS, quantile regression, RF importance) — needs DuckDB and ~128 GB RAM.
10. **Sub-portfolios (≥5 tokens) robustness** – same `mpt_distance_analysis.py` flow restricted to the `_gr_5` subset; outputs land in `data/distance_analysis_results_gr_5/`.
11. **Formula fit** – run [`Analytics/distance_return_analytics/src/fit_l1_formulas.py`](Analytics/distance_return_analytics/src/fit_l1_formulas.py) for the power-law fits used in the "distance vs portfolio size" claim.
12. **Plots** – distance-over-time, ridge plots, distribution buckets, token-vs-distance, ℓ₁-fit-vs-actual figures are produced by the matching scripts under [`Analytics/distance_return_analytics/plot/`](Analytics/distance_return_analytics/plot).

### Obj C — Returns & CAPM (`distance_return_analytics/`)

13. **Per-block returns / alphas** – run [`Analytics/distance_return_analytics/src/mpt_return_analysis.py`](Analytics/distance_return_analytics/src/mpt_return_analysis.py) for per-block median returns, alpha distributions, and the strategy-direction tables.
14. **Extended CAPM** – run [`Analytics/distance_return_analytics/src/analyze_extended_capm.py`](Analytics/distance_return_analytics/src/analyze_extended_capm.py) for portfolio-beta drift across strategies, beta-by-#tokens, and l1-vs-alpha binned correlations. `generate_all_macros.py` parses its console output, so save that as `data/analyze_extended_capm_output.txt` (e.g. `| tee`). `data/` keeps each cluster output folder as it comes (`distance_analysis_results/`, `distance_per_block_results/`, `return_analysis_results/`, `camp_analysis_results_full/`, `rf_shap_results/`); the generator and the plot scripts read from those folders.
15. **% beating market** – run [`Analytics/distance_return_analytics/src/analyze_pct_beat_market.py`](Analytics/distance_return_analytics/src/analyze_pct_beat_market.py) for the per-block "% of wallets that beat the market" curve; with `--prices-dir` it adds the same shares and margins against ETH's 20-day return (WETH price file, the backtest's date rule, checked against the stored market return), which is also where the forward horizon of the macros comes from. It also writes the hit rates against the wallet's own return (`pct_beat_baseline_per_block.csv`), counting a strategy portfolio within 1 % weight distance of the wallet's own as a tie.
16. **RF + SHAP return drivers** – run [`Analytics/distance_return_analytics/src/rf_shap_analysis_extended.py`](Analytics/distance_return_analytics/src/rf_shap_analysis_extended.py) for the 6- and 3-feature models, SHAP values and the month-grouped variants (market state, frontier distances) that test the "market entry timing dominates" claim; [`fetch_fear_greed_index.py`](Analytics/distance_return_analytics/src/fetch_fear_greed_index.py) provides its Fear & Greed input.
17. **Risk-free-rate robustness** – run [`Analytics/distance_return_analytics/src/rf_zero_test.py`](Analytics/distance_return_analytics/src/rf_zero_test.py) for the rf=0 vs rf=5% MSR weight-distance sensitivity check.
18. **Stablecoin lending rates** – run [`Analytics/distance_return_analytics/src/fetch_stablecoin_lending_rates.py`](Analytics/distance_return_analytics/src/fetch_stablecoin_lending_rates.py) to fetch USDT/USDC APYs from DefiLlama; informs the risk-free-rate benchmark.
19. **Plots** – median-return ribbons/ridges, alpha ridges, "beat the market" quantile boxes, panel-delta-vs-market, RF feature-importance and SHAP summary figures are produced by the matching scripts under [`Analytics/distance_return_analytics/plot/`](Analytics/distance_return_analytics/plot).

### Final step — assemble the paper macros

20. **Macros** – run [`Analytics/generate_all_macros.py`](Analytics/generate_all_macros.py) once everything above has produced the CSVs under `data/`. It emits [`Analytics/macro.tex`](Analytics/macro.tex), the single LaTeX include that resolves every numeric reference in the paper. Each topic owns its macros in a `*_macros.py` module; the distance and return macros come from [`distance_return_macros.py`](Analytics/distance_return_analytics/src/distance_return_macros.py), computed from `distance_return_analytics/data` only (the ETH-benchmark, horizon and risk-free-rate macros appear once the `--prices-dir` and `rf_zero_test.py` outputs are in `data/`). [`manual_constants.py`](Analytics/manual_constants.py) keeps only the CoinGecko universe, the reconstruction-pipeline facts and the compute details of the reconstruction and MPT runs, which have no data file in `Analytics/`.