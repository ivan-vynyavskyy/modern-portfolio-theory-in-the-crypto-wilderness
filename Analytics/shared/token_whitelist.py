"""
token_whitelist.py

Description:
    Token whitelist derived from the token status CSV, shared by the
    accounts_overview scripts (and usable by any analytics script that needs
    the set of assets in the study).

    A token is whitelisted when is_erc_20 == True, local_status == PASS and
    (by default) price_data_status == success. Native ETH is carried in the
    status CSV as the pseudo-token 0xee..ee (all checks marked passed) and is
    therefore whitelisted like any token; pass include_eth=False to drop it
    and reproduce the ERC-20-only universe.
"""

from __future__ import annotations

import os
from typing import Set

import pandas as pd

ETH_PSEUDO_TOKEN = "0x" + "ee" * 20
ZERO_ADDRESS = "0x" + "00" * 20


def _norm(s) -> str:
    return str(s).strip().lower()


def load_token_whitelist(status_csv: str, include_eth: bool = True,
                         require_price: bool = True, verbose: bool = True) -> Set[str]:
    """Return the lower-cased contract addresses passing the status checks."""
    if not os.path.exists(status_csv):
        raise FileNotFoundError(f"token status CSV not found: {status_csv}")
    df = pd.read_csv(status_csv, dtype=str, keep_default_na=False)
    if "contract_address" not in df.columns:
        raise ValueError(f"'contract_address' column missing in {status_csv}: {list(df.columns)}")

    mask = (df.get("is_erc_20", pd.Series("", index=df.index)).map(_norm) == "true") \
        & (df.get("local_status", pd.Series("", index=df.index)).map(_norm) == "pass")
    if require_price:
        mask &= df.get("price_data_status", pd.Series("", index=df.index)).map(_norm) == "success"

    addrs = df.loc[mask, "contract_address"].map(_norm)
    wl = {a for a in addrs if a.startswith("0x") and len(a) == 42}

    has_eth = ETH_PSEUDO_TOKEN in wl
    if not include_eth:
        wl.discard(ETH_PSEUDO_TOKEN)
    if verbose:
        print(f"[whitelist] {len(wl):,} assets from {status_csv}"
              f" | native ETH row present: {has_eth}"
              f"{' (excluded)' if has_eth and not include_eth else ''}")
    return wl
