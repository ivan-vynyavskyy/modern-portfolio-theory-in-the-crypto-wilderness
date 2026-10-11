"""
manual_constants.py

Description:
    LaTeX macro values that cannot be derived from a file in Analytics/.
    generate_all_macros.py imports MANUAL_CONSTANTS and emits each entry
    as \\def\\<name>/{<value>}.

    Each value is the EXACT verbatim string that appears inside the
    braces of the macro definition (preserve trailing decimals, commas,
    \\num{}, \\SI{}{} wrappers).

    Only facts with no data file in Analytics/ are kept: the size of the
    CoinGecko token universe, the reconstruction-pipeline statistics and
    the compute details of the reconstruction and MPT runs, which come
    from the pipeline runs on the cluster. The snapshot-output and
    optimisation-input counts come from a scan of the ETH-inclusive
    holdings ($DATA/recon-eth-erc20-partitioned) and wallet_level output
    on 2026-10-03; the input applies the 5_0 eligibility rule (num_tokens
    >= 2 and at least two assets with a positive balance after folding WETH
    into ETH). Every distance / return value (\\LOne*, \\CAPM*, \\Ret*,
    the \\Mpt* output counts, \\Rf*) is computed by
    distance_return_analytics/src/distance_return_macros.py, the tag
    coverage from tag_coverage_per_block.csv and the snapshot count from
    the per-block wallet statistics.
"""

MANUAL_CONSTANTS = {
    # --- Coingecko token-metadata ---
    "CoingeckoInitTokensN":        r"\num{5394}",
    "CoingeckoERCTokensN":         r"\num{5359}",
    "CoingeckoCleanedTokensN":     r"\num{4562}",  # ERC-20 whitelist after cleaning

    # --- Portfolio reconstruction pipeline metrics ---
    "ReconTransferEvents":         "1,977,546,094",
    "ReconTokensDiscovered":       "4,859",     # allow-listed tokens (Fig. 2 step 4); 4,847 of them have a ledger
    "ReconParquetSizeGB":          "42.4",
    "ReconTopTokenTransfers":      "698",
    "ReconTopTokenSharePct":       "35.3",
    "ReconMedianTokenTransfers":   "25,030",
    "ReconTokensAfterFilter":      "4,562",
    "ReconPipelineWorkers":        "48",
    "ReconBatchSize":              "20,000",
    # ETH-inclusive snapshot output (recon-eth-erc20-partitioned, 72 snapshots): five 4_0 runs by
    # snapshot period, consolidated by 4_2_partition_holdings.py; 3,316,158,544 rows in total
    "ReconRuns":                   "5",
    "ReconOutputRows":             "3,316.2",   # M account-snapshot rows
    "ReconOutputSizeGB":           "198.2",     # sum of the Parquet file sizes, 10^9 bytes
    "ReconRowsFirstSnapshotM":     "10.0",      # Jan 2020: 9,980,415
    "ReconRowsLastSnapshotM":      "90.0",      # Dec 2025: 89,984,937

    # --- MPT pipeline hardware config ---
    "MptWorkers":                  "64",
    "MptCores":                    "128",
    # Optimisation input (5_0 eligibility rule) and output size
    "MptMultiAssetRowsM":          "853.0",     # num_tokens >= 2 before folding WETH into ETH: 853,046,129
    "MptInputObservationsM":       "832.8",     # eligible after the merge: 832,791,572
    "MptInputMinWalletsPerBlock":  "3,964,445",
    "MptInputMedianWalletsPerBlock": "11,141,805",
    "MptInputMaxWalletsPerBlock":  "21,552,463",
    "MptOptimisedSharePct":        "98.2",      # 817,525,786 wallet_level rows / 832,791,572 eligible
    "MptOutputSizeGB":             "359.7",     # sum of the Parquet file sizes, 10^9 bytes (du -sh: 336G, GiB)
}
