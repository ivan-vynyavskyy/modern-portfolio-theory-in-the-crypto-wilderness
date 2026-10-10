# ETH Native Transfer Reconstruction

Reconstructs every native
ETH value movement from Erigon/ethereum-etl Parquet dumps
(`trace/`, `transaction/`, `block/`) into a directed-edge transfer ledger,
then aggregates it into the per-address flow ledger the portfolio pipeline
(`../common_src/4_0_portfolio_reconstruction_pipeline.py`) consumes.

`../run_pipeline.py` runs these steps as `eth.0_0` … `eth.5_0`, between the
ERC-20 stream and the common steps (see the root `README.md`).

## Pipeline (scripts in run order)

```
0_0_inspect_schemas.py          Step 0: schema/format discovery -> decisions.json (MANDATORY first)
1_0_build_transfers.py          transfer ledger: per-chunk trace+gas+withdrawal union -> partitioned parquet
2_0_eth_transfer_aggregator.py  transfer ledger -> address-flow ledger (2_1 format, sharded) for 4_0
3_0_validate_transfer_ledger.py transfer ledger vs chain: balance / supply / random spot-check
3_1_validate_flow_ledger.py     address-flow ledger vs eth_getBalance (3_0-runner style, PASS/NOT_PASS)
4_0_ledger_overview.py          fast global sanity numbers for the transfer ledger
4_1_ledger_stats.py             exact address/balance stats for the flow ledger (incl. negative-balance census)
5_0_fetch_eth_price_history.py  CoinGecko daily ETH close -> 0xee..ee_price.csv (bridge into the common_src pipeline)

6_0_selftest_synthetic.py       end-to-end self-test on a generated synthetic dump

config.py                       shared config/flags/chain constants (imported library)
ordering.py                     log_index synthesis / deterministic sort (imported library)
logs/                           run logs
```

Native ETH is identified downstream by the pseudo-token address
`0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee` (18 decimals).

## Transfer ledger output (1_0)

Hive-partitioned Parquet at `<OUTPUT_BASE>/block_range=<lo>_<hi>/part-*.parquet`:

| column | type | notes |
|---|---|---|
| `sender` | VARCHAR, nullable | NULL = minted (genesis/reward/withdrawal) |
| `recipient` | VARCHAR, nullable | NULL = burned (EIP-1559 base fee + EIP-4844 blob fee) |
| `block` | BIGINT | |
| `log_index` | INTEGER | **synthetic** per-block ordinal; unique only within `(block)` |
| `value` | DECIMAL(38,0) | wei, never float |
| `tx_hash` | VARCHAR, nullable | NULL for reward/withdrawal/genesis rows |

Debug columns (`transfer_type`, `trace_address`) appended by default.

Balance identity:
`balance(addr, B) = SUM(value | recipient=addr, block<=B) - SUM(value | sender=addr, block<=B)`.

## Address-flow ledger output (2_0)

`<ledgers>/token_address=0xee...ee/<token>-shard-*.parquet` with columns
`(address, block, log_index, value)` — signed flows (debit −, credit +),
address-hash sharded (oversized buckets sub-split), each file sorted by
`(address, block, log_index)`. Invariant checked before `_SUCCESS`:
`SUM(value) == minted − burned`.

## Typical run

Through the pipeline driver, with the paths from `pipeline_config.yaml`
(`eth_dump_dir`, `eth_transfers_dir`, `ledgers_dir`, `prices_dir`):

```bash
cd ..
python run_pipeline.py --config pipeline_config.yaml --from-step eth.0_0 --to-step eth.5_0
```

Or by hand, from this folder:

```bash
DUMP=/path/to/ethereum_dump       # trace/, transaction/, block/
OUT=/path/to/eth_transfers        # decisions.json + block_range=*/ transfer ledger
LEDGERS=/path/to/ledgers          # shared with the ERC-20 ledgers

# 0. schema discovery (writes decisions.json; review the STEP 0 REPORT)
python 0_0_inspect_schemas.py --input-base $DUMP --output-base $OUT

# 1. build the transfer ledger (all chunks; finished chunks are skipped, --force rebuilds them)
python 1_0_build_transfers.py --all --input-base $DUMP --output-base $OUT

# 2. aggregate into the address-flow ledger
python 2_0_eth_transfer_aggregator.py --output $LEDGERS --shards 64 --output-base $OUT

# 3. validate against an archive node
python 3_0_validate_transfer_ledger.py spot-check --random-pairs 200 --rpc-url https://... --output-base $OUT
python 3_1_validate_flow_ledger.py --data-dir $LEDGERS --trials 1000 --rpc-url https://...

# 4. global sanity + exact stats (negative-balance census)
python 4_0_ledger_overview.py --no-scan --output-base $OUT
python 4_1_ledger_stats.py --data-dir $LEDGERS --decisions $OUT/decisions.json

# 5. daily ETH close into the shared prices dir (CoinGecko key via --api-key or $COINGECKO_API_KEY)
python 5_0_fetch_eth_price_history.py --output-dir /path/to/prices \
    --status-file /path/to/eth_price_download_status.csv

# self-test (synthetic dump, no cluster data needed)
python 6_0_selftest_synthetic.py
```

## Reconstruction rules worth remembering

1. Traces are the single value source; `transaction.value` would double-count.
2. `delegatecall`/`callcode`/`staticcall` are never value transfers (callcode =
   caller→caller self-transfer; counting it fabricated 3.9M ETH — repaired).
3. Reverted subtrees excluded via propagated `status` (or error-prefix fallback);
   fees still charged.
4. Fees: pre-London all to miner; post-London base fee burned + tip to producer;
   post-Cancun blob fee (`blob_gas_used × blob_gas_price`) burned on top.
5. Beacon withdrawals come from block bodies only (Gwei→wei ×10⁹).
6. No reward rows after the Merge (phantom 2-ETH reward traces are excluded).
