"""
locations.py

Description:
    Centralized filesystem path constants for the Analytics pipeline.
    Every script that needs an absolute data path should reference
    Location.<ATTR> rather than hardcoding cluster paths.
"""

from pathlib import Path


class Location:
    """Filesystem anchors for the Analytics pipeline."""

    # ---- Root ----------------------------------------------------------------
    DATA_ROOT = Path("/anonymized")

    # ---- Shared datasets -----------------------------------------------------
    DATASETS_DIR = DATA_ROOT / "data"
    PORTFOLIO_RECONSTRUCTION_DATA = DATA_ROOT / "portfolio_reconstruction_data"

    TOKEN_DECIMALS_CACHE = DATASETS_DIR / "token_decimals_cache.json"
    TOKEN_STATUS_CSV = DATASETS_DIR / "eth_tokens_status_new_new_sorted.csv"
    TOKEN_STATUS_PRIMARY_CSV = DATASETS_DIR / "status_eth_tokens.with_primary_category.csv"
    BLOCK_TO_DATE_CSV = DATASETS_DIR / "blocks_closest_to_midnight.csv"
    WHITELIST_ADDRESSES_PARQUET = DATASETS_DIR / "whitelist_unique_addresses.parquet"
    CA_CREATION_TSV = DATASETS_DIR / "contract_accounts_created.tsv"

    # ---- Raw transfer data (accounts_overview notebooks) ---------------------
    ETH_TRANSFERS_DATA = DATA_ROOT / "eth-erc20-transfers-20251208-structured"

    # ---- Prices ----------------------------------------------------------------
    PRICES_DIR = DATA_ROOT / "prices"

    # ---- Reconstruction parquets (source data for backtest pipeline) ----------
    RECON_PARQUET_2020_2022 = DATA_ROOT / "recon-eth-erc20_2020_01-2022_12.parquet"
    RECON_PARQUET_2023_2025 = DATA_ROOT / "recon-eth-erc20_2023_01-2025_12.parquet"

    # ---- Per-token inequality (produced by inequality_measures.ipynb) ---------
    TOKEN_INEQUALITY_CSV = DATASETS_DIR / "token_inequality_per_block.csv"

    # ---- MPT optimisation results --------------------------------------------
    MPT_RESULTS_ROOT = DATA_ROOT / "mpt-optimization-results"
    MPT_DATA = MPT_RESULTS_ROOT / "data"
    MPT_DATA_EXTENDED = MPT_RESULTS_ROOT / "data_extended"
    MPT_DISTANCE_RESULTS = MPT_RESULTS_ROOT / "analysis_results"
    MPT_RETURN_RESULTS = MPT_RESULTS_ROOT / "return_analysis_results"
    MPT_RF_SHAP_RESULTS = MPT_RESULTS_ROOT / "rf_shap_results"

    # ---- Risk-free rate sensitivity test output ------------------------------
    MPT_RF_TEST_DIR = DATA_ROOT / "rf_test"
