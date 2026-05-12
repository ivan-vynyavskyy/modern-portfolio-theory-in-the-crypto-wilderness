"""
manual_constants.py

Description:
    LaTeX macro values that cannot be derived from CSVs in Analytics/.
    generate_all_macros.py imports MANUAL_CONSTANTS and emits each entry
    as \\def\\<name>/{<value>}.

    Each value is the EXACT verbatim string that appears inside the
    braces of the macro definition (preserve trailing decimals, commas,
    \\num{}, \\SI{}{} wrappers).
"""

MANUAL_CONSTANTS = {
    # --- Coingecko token-metadata ---
    "CoingeckoInitTokensN":        r"\num{5394}",
    "CoingeckoERCTokensN":         r"\num{5359}",
    "CoingeckoCleanedTokensN":     r"\num{4562}",  # ERC-20 whitelist after cleaning
    "CoingeckoRemovedTokensN":     r"\num{830}",   # = 5394 - 4562

    # --- Wealth composition (narrative round-figures) ---
    "WealthPeakTwentyone":         r"\$1\,T",
    "WealthPeakTwentyfiveB":       r"\$1.47\,T",

    # --- L1 distance dataset metadata ---
    "LOneObservationsRaw":         "239634756",

    # --- Extended OLS R-squared (base OLS + RF are now auto-derived) ---
    "LOneOlsExtBrRsq":             "0.104",
    "LOneOlsExtSrRsq":             "0.632",
    "LOneOlsExtMsRsq":             "0.132",

    # --- Naive benchmark direction macros ---
    "LOneDeltaSrEqualRatio":       "3",   # = 27.4 / 9.2 ≈ 3

    # --- CAPM dataset metadata ---
    "CAPMObsRaw":                  "235868007",
    "CAPMBlocksN":                 "72",
    "CAPMHorizonDays":             "20",

    # --- Return analysis (narrative ranges) ---
    "RetNegativePctLo":            "57",
    "RetNegativePctHi":            "59",
    "RetDistZeroMedianBaseline":   "-0.05",
    "RetDistHighMedianBaseline":   "-6.09",

    # --- GraphSense address tags (per-block coverage, not in CSV) ---
    "TagCoveragePctMean":          "23.25",
    "TagCoveragePctMin":           "10.70",
    "TagCoveragePctMax":           "35.14",

    # --- Top-wallet wealth concentration ---
    "TopOneWealthTwenty":          "97.25",
    "TopOneWealthTwentyfive":      "99.03",
    "TopFiveTenWealthMin":         "99.3",

    # --- Portfolio reconstruction pipeline metrics ---
    "ReconTransferEvents":         "1,977,546,094",
    "ReconTokensDiscovered":       "4,847",
    "ReconParquetSizeGB":          "42.4",
    "ReconTopTokenTransfers":      "698",
    "ReconTopTokenSharePct":       "35.3",
    "ReconMedianTokenTransfers":   "25,030",
    "ReconTokensAfterFilter":      "4,562",
    "ReconPipelineWorkers":        "48",
    "ReconBatchSize":              "20,000",
    "ReconOutputRowsFirst":        "477.3",
    "ReconOutputRowsSecond":       "982.2",
    "ReconOutputRows":             "1,459.5",
    "ReconOutputSizeGB":           "196.6",
    "ReconSnapshotBlocks":         "72",

    # --- Risk-free rate parameter ---
    "RfDefault":                       "5",

    # --- Robustness check sample (rf=0 vs rf=5%) ---
    "RfRobustBlocks":                  "72",
    "RfRobustWalletsPerBlock":         r"5{,}000",
    "RfRobustTotalWallets":            r"360{,}000",

    # MSR weight L1 distance, rf=0 vs rf=5%, normalised 0--100
    "RfRobustWeightLOneMedianPct":     "0",
    "RfRobustWeightLOneMeanPct":       "4.5",
    "RfRobustWeightLOneTopFivePct":        "37.1",

    # Differences in L1-style gap metrics (pp on 0--100 scale)
    "RfRobustGapDiffMedianPP":         "0",
    "RfRobustGapDiffMeanPP":           "-0.4",
    "RfRobustGapDiffTopFivePP":            "13.2",
    "RfRobustVsEqualDiffMedianPP":     "0",
    "RfRobustVsEqualDiffMeanPP":       "-0.7",
    "RfRobustVsEqualDiffTopFivePP":        "2.4",
    "RfRobustVsMcapDiffMedianPP":      "0",
    "RfRobustVsMcapDiffMeanPP":        "-2.6",
    "RfRobustVsMcapDiffTopFivePP":         "18.5",

    # --- MPT pipeline hardware config ---
    "MptObservations":             "239,634,756",
    "MptObservationsM":            "239.6",
    "MptWorkers":                  "64",
    "MptCores":                    "128",
    "MptOutputSizeGB":             "144",
    "MptStrategies":               "6",
    "MptHorizonDays":              "20",
}