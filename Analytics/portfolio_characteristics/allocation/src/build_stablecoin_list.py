#!/usr/bin/env python3
"""
build_stablecoin_list.py

Description:
    Builds the curated stablecoin list used as the on-chain "cash" asset by
    stablecoin_cash_share.py. Every token of the study's whitelist whose
    symbol or name looks like a stable asset was reviewed by hand (September
    2026) and classified below; this script joins that classification with
    the token status CSV (symbol, name, whitelist status) and writes
    ../data/stablecoins.csv.

    Classes (asset_class):
        usd_fiat       fiat-reserve USD stablecoins (USDT, USDC, PYUSD, ...)
        usd_crypto     crypto-collateralised USD stablecoins (DAI, LUSD, ...)
        usd_synthetic  delta-neutral / synthetic dollars (USDe, USDf, ...)
        usd_rwa        dollars backed by tokenised T-bills (USD0, USDV, ...)
        usd_yield      yield-bearing wrappers of dollars (sUSDe, sDAI-type,
                       cTokens, ERC-4626 vaults); price drifts above 1 by design
        usd_lp         LP tokens of pure stablecoin pools (3CRV, ...)
        usd_unstable   algorithmic or failed dollars (UST, USTC, FEI, USDD,
                       HUSD after its 2022 collapse, ...)
        usd_suspect    USD by name, but the on-chain supply looks like an issuer
                       treasury priced at 1 USD (billions of dollars in a few dozen
                       accounts): possible phantom value, never cash
        nonusd_fiat    non-USD fiat stablecoins (EURC, EURS, XSGD, JPYC, ...)
        commodity      gold-backed tokens (PAXG, XAUT, ...)
        not_stable     reviewed false positives of the name search

    Flags:
        cash_core      high-confidence USD stablecoins with fiat or crypto
                       collateral and no lasting depeg: the default cash
                       definition of the two-fund test
        cash_extended  every usd_fiat, usd_crypto, usd_synthetic, usd_rwa,
                       usd_yield and usd_lp token (sensitivity definition)
        stablecoin_cash_share.py --cash-def extended_fx additionally counts
        nonusd_fiat tokens as cash.

    The script also lists whitelist tokens whose symbol or name matches the
    stablecoin search pattern but are not in the curated table, so tokens
    added to the universe later get reviewed. With --prices-dir it checks
    every USD-pegged, non-yield token against its daily price history
    (median, minimum, share of days more than 2 % and 5 % off 1 USD) and
    flags tokens that spent more than 10 % of their days over 5 % off peg.

Input:
    --status-csv   token status CSV (symbol, name, contract_address, is_erc_20,
                   local_status, price_data_status), default
                   PortfolioReconstruction/data/status_eth_tokens_3_1.csv
    --prices-dir   optional, <addr>_price.csv files (date, daily_close_usd)

Output:
    --out-csv      default ../data/stablecoins.csv

Usage:
    python build_stablecoin_list.py
    python build_stablecoin_list.py --prices-dir $DATA/prices   # peg check (cluster)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _alloc_common import DEFAULT_STABLECOINS_CSV, log  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_STATUS_CSV = REPO_ROOT / "PortfolioReconstruction" / "data" / "status_eth_tokens_3_1.csv"

CASH_EXTENDED_CLASSES = {"usd_fiat", "usd_crypto", "usd_synthetic", "usd_rwa", "usd_yield", "usd_lp"}
CANDIDATE_SYMBOL_RE = re.compile(r"(?:usd|dai|eur|gbp|chf|jpy|sgd|aud|cad|cny|cnh|krw|try|brl|mxn|idr|zar|xau|gold)")
CANDIDATE_NAME_RE = re.compile(r"(?:usd|dollar|stable|euro|eur |pound|franc|yen|peg|gold|dai|rupiah|peso|lira)")

# (address, asset_class, peg, cash_core, confidence, note)
# confidence: high = well-known token, identity unambiguous; medium = known
# project, smaller or with a depeg history; low = identity from name only.
CURATED = [
    # ---- USD, fiat reserves -------------------------------------------------
    ("0xdac17f958d2ee523a2206206994597c13d831ec7", "usd_fiat", "USD", True, "high", "Tether USDT"),
    ("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", "usd_fiat", "USD", True, "high", "Circle USDC"),
    ("0x4fabb145d64652a948d72533023f6e7a623c7c53", "usd_fiat", "USD", True, "high", "Paxos BUSD; minting stopped Feb 2023"),
    ("0x0000000000085d4780b73119b644ae5ecd22b376", "usd_fiat", "USD", True, "high", "TrueUSD; brief depegs 2023-24"),
    ("0x8e870d67f660d95d5be530380d0ec0bd388289e1", "usd_fiat", "USD", True, "high", "Pax Dollar USDP"),
    ("0x056fd409e1d7a124bd7017459dfea2f387b6d5cd", "usd_fiat", "USD", True, "high", "Gemini Dollar"),
    ("0x6c3ea9036406852006290770bedfcaba0e23a0e8", "usd_fiat", "USD", True, "high", "PayPal USD"),
    ("0xc5f0f7b66764f6ec8c8dff7ba683102295e16409", "usd_fiat", "USD", True, "high", "First Digital USD"),
    ("0x8292bb45bf1ee4d140127049757c2e0ff06317ed", "usd_fiat", "USD", True, "high", "Ripple USD"),
    ("0x8d0d000ee44948fc98c9b98a4fa4921476f08b0d", "usd_fiat", "USD", True, "high", "World Liberty USD1"),
    ("0xe343167631d89b6ffc58b88d6b7fb0228795491d", "usd_fiat", "USD", True, "high", "Paxos Global Dollar USDG"),
    ("0x00000000efe302beaa2b3e6e1b18d08d69a9012a", "usd_fiat", "USD", True, "high", "Agora AUSD"),
    ("0xc139190f447e929f090edeb554d95abb8b18ac1c", "usd_fiat", "USD", True, "high", "Ethena USDtb, reserves in BUIDL/USDC"),
    ("0xcacd6fd266af91b8aed52accc382b4e165586e29", "usd_fiat", "USD", True, "high", "Frax USD frxUSD (2025), T-bill reserves"),
    ("0xc83e27f270cce0a3a3a29521173a83f402c1768b", "usd_fiat", "USD", False, "medium", "Quantoz USDQ"),
    ("0x7b43e3875440b44613dc3bc08e7763e6da63c8f8", "usd_fiat", "USD", False, "medium", "StablR USD"),
    ("0x4f604735c1cf31399c6e711d5962b2b3e0225ad3", "usd_fiat", "USD", False, "medium", "Glo Dollar"),
    ("0x8ccedbae4916b79da7f3f612efb2eb93a2bfd6cf", "usd_fiat", "USD", False, "medium", "MNEE"),
    ("0xc08e7e23c235073c6807c2efe7021304cb7c2815", "usd_fiat", "USD", False, "medium", "StraitsX XUSD"),
    ("0xf8750b54d86be7ae9e32b4a0c826811198d63313", "usd_fiat", "USD", False, "medium", "Hex Trust USDX"),
    ("0x5422374b27757da72d5265cc745ea906e0446634", "usd_fiat", "USD", False, "medium", "SG Forge USD CoinVertible"),
    ("0xaca92e438df0b2401ff60da7e4337b687a2435da", "usd_fiat", "USD", False, "medium", "MetaMask USD (2025)"),
    ("0x1cdd2eab61112697626f7b4bb0e23da4febf7b7c", "usd_fiat", "USD", False, "medium", "USDT bridged via Wormhole"),
    ("0xace8e719899f6e91831b18ae746c9a965c2119f1", "usd_fiat", "USD", False, "low", "Ondo U.S. Dollar Token"),
    ("0x7cd017ca5ddb86861fa983a34b5f495c6f898c41", "usd_fiat", "USD", False, "low", "Worldwide USD"),
    ("0xf9fb20b8e097904f0ab7d12e9dbee88f2dcd0f16", "usd_fiat", "USD", False, "low", "Stable Coin SBC"),
    ("0xa4335da338ec4c07c391fc1a9bf75f306adadc08", "usd_fiat", "USD", False, "low", "ARYZE eUSD"),
    ("0xdddd73f5df1f0dc31373357beac77545dc5a6f3f", "usd_fiat", "USD", False, "low", "Plume USD, wraps USDC"),
    # ---- USD, crypto collateral ---------------------------------------------
    ("0x6b175474e89094c44da98b954eedeac495271d0f", "usd_crypto", "USD", True, "high", "MakerDAO DAI"),
    ("0xdc035d45d973e3ec169d2276ddab16f1e407384f", "usd_crypto", "USD", True, "high", "Sky USDS"),
    ("0x5f98805a4e8be255a32880fdec7f6728c6568ba0", "usd_crypto", "USD", True, "high", "Liquity LUSD"),
    ("0xf939e0a03fb07f59a73314e73794be0e57ac1b4e", "usd_crypto", "USD", True, "high", "Curve crvUSD"),
    ("0x40d16fc0246ad3160ccc09b8d0d3a2cd28ae6c2f", "usd_crypto", "USD", True, "high", "Aave GHO; traded below peg 2023"),
    ("0x6440f144b7e50d6a8439336510312d2f54beb01d", "usd_crypto", "USD", True, "high", "Liquity v2 BOLD"),
    ("0x853d955acef822db058eb8505911ed77f175b99e", "usd_crypto", "USD", True, "high", "Legacy FRAX, fractional until 2023"),
    ("0x865377367054516e17014ccded1e7d814edc9ce4", "usd_crypto", "USD", False, "medium", "Inverse DOLA"),
    ("0x99d8a9c45b2eca8864373a26d1459e3dff1e17f3", "usd_crypto", "USD", False, "medium", "Abracadabra MIM; depegs 2022, 2024"),
    ("0xb01dd87b29d187f3e3a4bf6cdaebfb97f3d9ab98", "usd_crypto", "USD", False, "medium", "Legacy BOLD, superseded 2025"),
    ("0xe2f2a5c287993345a840db3b0845fbc70f5935a5", "usd_crypto", "USD", False, "medium", "mStable mUSD, basket of stablecoins"),
    ("0x4591dbff62656e7859afe5e45f6f47d3669fbb28", "usd_crypto", "USD", False, "medium", "Prisma mkUSD"),
    ("0xcfc5bd99915aaa815401c5a41a927ab7a38d29cf", "usd_crypto", "USD", False, "medium", "Threshold thUSD"),
    ("0xa0d69e286b938e21cbf7e51d71f6a4c8918f482f", "usd_crypto", "USD", False, "medium", "Lybra eUSD"),
    ("0x0000206329b97db379d5e1bf586bbdb969c63274", "usd_crypto", "USD", False, "medium", "Angle USDA"),
    ("0x085780639cc2cacd35e474e71f4d000e2405d8f6", "usd_crypto", "USD", False, "medium", "f(x) fxUSD"),
    ("0x0a5e677a6a24b2f1a2bf4f3bffc443231d2fdec8", "usd_crypto", "USD", False, "medium", "dForce USX"),
    ("0x7c1156e515aa1a2e851674120074968c905aaf37", "usd_crypto", "USD", False, "medium", "Level lvlUSD"),
    ("0x31a2e08f4232329e4eddb025c0275f43c9cd56d7", "usd_crypto", "USD", False, "low", "second token named LUSD"),
    ("0x76972f054ab43829064d31ff5f3acc5fabe57fe8", "usd_crypto", "USD", False, "low", "Tharwa USD"),
    ("0x8a60e489004ca22d775c5f2c657598278d17d9c2", "usd_crypto", "USD", False, "low", "Avalon USDa"),
    ("0x9cf12ccd6020b6888e4d4c4e4c7aca33c1eb91f8", "usd_crypto", "USD", False, "low", "Asymmetry USDaf"),
    ("0x1958853a8be062dc4f401750eb233f5850f0d0d2", "usd_crypto", "USD", False, "low", "Satoshi Stablecoin"),
    ("0x65d72aa8da931f047169112fcf34f52dbaae7d18", "usd_crypto", "USD", False, "low", "f(x) rUSD"),
    ("0x09d4214c03d01f49544c0448dbe3a27f768f2b34", "usd_crypto", "USD", False, "low", "Reservoir rUSD"),
    ("0xab5eb14c09d416f0ac63661e57edb7aecdb9befa", "usd_crypto", "USD", False, "low", "Metronome Synth USD"),
    ("0x57ab1e0003f623289cd798b1824be09a793e4bec", "usd_crypto", "USD", False, "low", "Resupply reUSD"),
    ("0x09fd37d9aa613789c517e76df1c53aece2b60df4", "usd_crypto", "USD", False, "low", "ebUSD"),
    ("0xa48f322f8b3edff967629af79e027628b9dd1298", "usd_crypto", "USD", False, "low", "Davos DUSD"),
    ("0x677ddbd918637e5f2c79e164d402454de7da8619", "usd_crypto", "USD", False, "low", "Vesper VUSD"),
    ("0x000000096cb3d4007fc2b79b935c4540c5c2d745", "usd_crypto", "USD", False, "low", "Jigsaw USD"),
    ("0xdd468a1ddc392dcdbef6db6e34e89aa338f9f186", "usd_crypto", "USD", False, "low", "Mezo USD, BTC collateral"),
    ("0x9eead9ce15383caeed975427340b3a369410cfbf", "usd_crypto", "USD", False, "low", "Alloy Tether, XAUT collateral"),
    ("0xa7a0b3fe94121e366d774d60d075f6386f750884", "usd_crypto", "USD", False, "low", "USDFI"),
    ("0xa0ed3359902eff692e5b8167038133a73d641909", "usd_crypto", "USD", False, "low", "DeFi Dollar USDFI"),
    ("0x2c2d8a078b33bf7782a16acce2c5ba6653a90d5f", "usd_crypto", "USD", False, "low", "L3USD"),
    ("0x4ea40dcee961675683e0a2e1721bd49cb9bca913", "usd_crypto", "USD", False, "low", "Ring USD"),
    ("0xcccc62962d17b8914c62d74ffb843d73b2a3cccc", "usd_crypto", "USD", False, "low", "Cap cUSD"),
    ("0xe868084cf08f3c3db11f4b73a95473762d9463f7", "usd_crypto", "USD", False, "low", "Yala YU, BTC collateral"),
    ("0x48f9e38f3070ad8945dfeae3fa70987722e3d89c", "usd_crypto", "USD", False, "low", "InfiniFi iUSD"),
    ("0x3b5f2810fb2168ffa9c73160f97bf9f2461ffa5c", "usd_crypto", "USD", False, "low", "litUSD"),
    ("0x9b3a8f7cec208e247d97dee13313690977e24459", "usd_crypto", "USD", False, "low", "Parallel USDp (2025)"),
    # ---- USD, synthetic / delta-neutral -------------------------------------
    ("0x4c9edd5852cd905f086c759e8383e09bff1e68b3", "usd_synthetic", "USD", False, "high", "Ethena USDe"),
    ("0xfa2b947eec368f42195f24f36d2af29f7c24cec2", "usd_synthetic", "USD", False, "medium", "Falcon USDf"),
    ("0x15700b564ca08d9439c58ca5053166e8317aa138", "usd_synthetic", "USD", False, "medium", "Elixir deUSD; depegged Nov 2025"),
    ("0xf3527ef8de265eaa3716fb312c12847bfba66cef", "usd_synthetic", "USD", False, "medium", "Stables Labs USDX; depegged Nov 2025"),
    ("0x4274cd7277c7bb0806bd5fe84b9adae466a8da0a", "usd_synthetic", "USD", False, "low", "Aegis YUSD"),
    ("0xe556aba6fe6036275ec1f87eda296be72c811bce", "usd_synthetic", "USD", False, "low", "Neutrl USD"),
    ("0x99999999999999cc837c997b882957dafdcb1af9", "usd_synthetic", "USD", False, "low", "Smardex wrapped USDN"),
    # ---- USD, tokenised T-bill backing ---------------------------------------
    ("0x73a15fed60bf67631dc6cd7bc5b6e8da8190acf5", "usd_rwa", "USD", False, "medium", "Usual USD0"),
    ("0x20b3b07e9c0e37815e2892ab09496559f57c3603", "usd_rwa", "USD", False, "medium", "Verified USD USDV"),
    ("0xa469b7ee9ee773642b3e93e842e5d9b5baa10067", "usd_rwa", "USD", False, "medium", "Anzen USDz"),
    # ---- USD, yield-bearing wrappers (price drifts above 1) ------------------
    ("0x96f6ef951840721adbf46ac996b59e0235cb985c", "usd_yield", "USD", False, "high", "Ondo USDY"),
    ("0x9d39a5de30e57443bff2a8307a4256c8797a3497", "usd_yield", "USD", False, "high", "Ethena staked USDe"),
    ("0xa3931d71877c0e7a3148cb7eb4463524fec27fbd", "usd_yield", "USD", False, "high", "Sky savings USDS"),
    ("0x0655977feb2f289a4ab78af67bab0d17aab84367", "usd_yield", "USD", False, "high", "Savings crvUSD"),
    ("0x39aa39c021dfbae8fac545936693ac917d5e7563", "usd_yield", "USD", False, "high", "Compound cUSDC, price about 0.02"),
    ("0x5d3a536e4d6dbd6114cc1ead35777bab948e3643", "usd_yield", "USD", False, "high", "Compound cDAI, price about 0.02"),
    ("0xad55aebc9b8c03fc43cd9f62260391c13c23e7c0", "usd_yield", "USD", False, "medium", "Compounding OpenDollar"),
    ("0xcf62f905562626cfcdd2261162a51fd02fc9c5b6", "usd_yield", "USD", False, "medium", "Staked Frax USD"),
    ("0x5c5b196abe0d54485975d1ec29617d42d9198326", "usd_yield", "USD", False, "medium", "Elixir staked deUSD; depegged Nov 2025"),
    ("0x80ac24aa929eaf5013f6436cda2a7ba190f5cc0b", "usd_yield", "USD", False, "medium", "Maple syrupUSDC"),
    ("0x356b8d89c1e1239cbbb9de4815c39a1474d5ba7d", "usd_yield", "USD", False, "medium", "Maple syrupUSDT"),
    ("0x9d1089802ee608ba84c5c98211afe5f37f96b36c", "usd_yield", "USD", False, "medium", "Fluid fUSDC"),
    ("0xd4fa2d31b7968e448877f69a96de69f5de8cd23e", "usd_yield", "USD", False, "medium", "Wrapped Aave USDC"),
    ("0x7bc3485026ac48b6cf9baf0a377477fff5703af8", "usd_yield", "USD", False, "medium", "Wrapped Aave USDT"),
    ("0x57f5e098cad7a3d1eed53991d4d66c45c9af7812", "usd_yield", "USD", False, "medium", "Mountain wrapped USDM"),
    ("0x35d8949372d46b7a3d5a56006ae77b215fc69bc0", "usd_yield", "USD", False, "low", "Usual USD0++ bond; depegged Jan 2025"),
    ("0xdbdc1ef57537e34680b898e1febd3d68c7389bcb", "usd_yield", "USD", False, "low", "InfiniFi staked iUSD"),
    ("0x12b004719fb632f1e7c010c6f5d6009fb4258442", "usd_yield", "USD", False, "low", "InfiniFi locked iUSD"),
    ("0x4737d9b4592b40d51e110b94c9c043c6654067ae", "usd_yield", "USD", False, "low", "Staked Level USD"),
    ("0x738d1115b90efa71ae468f1287fc864775e23a31", "usd_yield", "USD", False, "low", "Reservoir srUSD"),
    ("0xd3fd63209fa2d55b07a0f6db36c2f43900be3094", "usd_yield", "USD", False, "low", "Wrapped savings rUSD"),
    ("0x2b66aade1e9c062ff411bd47c44e0ad696d43bd9", "usd_yield", "USD", False, "low", "sUSDa"),
    ("0x7788a3538c5fc7f9c7c8a74eac4c898fc8d87d92", "usd_yield", "USD", False, "low", "Staked USDX; depegged Nov 2025"),
    ("0x547213367cfb08ab418e7b54d7883b2c2aa27fd7", "usd_yield", "USD", False, "low", "Staked USDz"),
    ("0xf67e2dc041b8a3c39d066037d29f500757b1e886", "usd_yield", "USD", False, "low", "seed USDN"),
    ("0xfe0ccc9942e98c963fe6b4e5194eb6e3baa4cb64", "usd_yield", "USD", False, "low", "Staked YUSD"),
    ("0x572975ff6d5136c81c8d7448b6361ef9eefe1ab0", "usd_yield", "USD", False, "low", "Wrapped staked USDT"),
    ("0x0f359fd18bda75e9c49bc027e7da59a4b01bf32a", "usd_yield", "USD", False, "low", "Relend USDC"),
    ("0xd63070114470f685b75b74d60eec7c1113d33a3d", "usd_yield", "USD", False, "low", "Morpho vault USDC"),
    ("0x7743e50f534a7f9f1791dde7dcd89f7783eefc39", "usd_yield", "USD", False, "low", "f(x) USD saving"),
    ("0x2e3c5e514eef46727de1fe44618027a9b70d92fc", "usd_yield", "USD", False, "low", "YieldFi vyUSD"),
    ("0x19ebd191f7a24ece672ba13a302212b5ef7f35cb", "usd_yield", "USD", False, "low", "YieldFi yUSD"),
    ("0x3db228fe836d99ccb25ec4dfdc80ed6d2cddcb4b", "usd_yield", "USD", False, "low", "ynUSD Max vault"),
    ("0xe2fc85bfb48c4cf147921fbe110cf92ef9f26f94", "usd_yield", "USD", False, "low", "Staked Stream USD; collapsed Nov 2025"),
    ("0x098697ba3fee4ea76294c5d6a466a4e3b3e95fe6", "usd_yield", "USD", False, "low", "USP yield stablecoin"),
    ("0xa2f8da4a55898b6c947fa392ef8d6bfd87a4ff77", "usd_yield", "USD", False, "low", "hwUSD"),
    ("0xd5255cc08ebaf6d54ac9448822a18d8a3da29a42", "usd_yield", "USD", False, "low", "Gaib AI Dollar Alpha USDC"),
    ("0xdf100b8abd3cd41fc5cc1236ad7c6629a901afbf", "usd_yield", "USD", False, "low", "Diversified USD"),
    ("0x5086bf358635b81d8c47c66d1c8b9e567db70c72", "usd_yield", "USD", False, "low", "Re Protocol reUSD"),
    ("0xddc0f880ff6e4e22e4b74632fbb43ce4df6ccc5a", "usd_yield", "USD", False, "low", "Re Protocol reUSDe"),
    ("0xbc404429558292ee2d769e57d57d6e74bbd2792d", "usd_yield", "USD", False, "low", "Savings USX"),
    ("0x1e33e98af620f1d563fcd3cfd3c75ace841204ef", "usd_yield", "USD", False, "low", "Dialectic USD vault"),
    ("0x139450c2dcef827c9a2a0bb1cb5506260940c9fd", "usd_yield", "USD", False, "low", "SuperReturn sSuperUSD"),
    ("0xd3dce716f3ef535c5ff8d041c1a41c3bd89b97ae", "usd_yield", "USD", False, "low", "Rings scUSD"),
    ("0x0d86883faf4ffd7aeb116390af37746f45b6f378", "usd_yield", "USD", False, "low", "Reserve Web 3 Dollar basket"),
    # ---- USD, stablecoin-pool LP tokens --------------------------------------
    ("0x6c3f90f043a72fa612cbac8115ee7e52bde6e490", "usd_lp", "USD", False, "high", "Curve 3pool LP (DAI/USDC/USDT)"),
    ("0x3175df0976dfa876431c2e9ee6bc45b65d3473cc", "usd_lp", "USD", False, "medium", "Curve FRAX/USDC LP"),
    # ---- USD, algorithmic or failed ------------------------------------------
    ("0xa693b19d2931d498c5b318df961919bb4aee87a5", "usd_unstable", "USD", False, "high", "TerraUSD via Wormhole; collapsed May 2022"),
    ("0xa47c8bf37f92abed4a126bda807a7b7498661acd", "usd_unstable", "USD", False, "high", "Wrapped USTC; collapsed May 2022"),
    ("0x956f47f50a910163d8bf957cf5846d573e7f87ca", "usd_unstable", "USD", False, "high", "Fei USD; wound down 2022"),
    ("0x4f8e5de400de08b164e7421b3ee387f461becd1a", "usd_unstable", "USD", False, "medium", "USDD; repeated depegs"),
    ("0xdf574c24545e5ffecb9a659c229253d4111d87e1", "usd_unstable", "USD", False, "high", "HUSD; collapsed Aug 2022"),
    ("0xc285b7e09a4584d027e5bc36571785b515898246", "usd_unstable", "USD", False, "medium", "Coin98 Dollar; depegged"),
    ("0x0a3bb08b3a15a19b4de82f8acfc862606fb69a2d", "usd_unstable", "USD", False, "medium", "iZUMi Bond USD, a bond token"),
    # ---- USD by name, supply looks like an issuer treasury (checked 2026-09-29 on
    # the extended cash-share run: flat multi-billion value in a few dozen accounts)
    ("0x40caa7912437002ee2c8415d43e7f575c733674c", "usd_suspect", "USD", False, "low",
     "Currency One USD; ~2.5 B USD from Oct 2025 in 4-47 accounts"),
    ("0xbbaec992fc2d637151daf40451f160bf85f3c8c1", "usd_suspect", "USD", False, "low",
     "USD Mapped Token; flat ~6.5 B USD since Jun 2024 in 70-95 accounts"),
    # ---- non-USD fiat ---------------------------------------------------------
    ("0x1abaea1f7c830bd89acc67ec4af516284b1bc33c", "nonusd_fiat", "EUR", False, "high", "Circle EURC"),
    ("0xdb25f211ab05b1c97d595516f45794528a807ad8", "nonusd_fiat", "EUR", False, "high", "STASIS EURS"),
    ("0xc581b735a1688071a1746c968e0798d642ede491", "nonusd_fiat", "EUR", False, "high", "Tether EURT"),
    ("0x1a7e4e63778b4f12a199c062f3efdd288afcbce8", "nonusd_fiat", "EUR", False, "medium", "Angle EURA"),
    ("0x39b8b6385416f4ca36a20319f70d28621895279d", "nonusd_fiat", "EUR", False, "medium", "Monerium EURe"),
    ("0x5f7827fdeb7c20b443265fc2f40845b715385ff2", "nonusd_fiat", "EUR", False, "medium", "SG Forge EURCV"),
    ("0x9d1a7a3191102e9f900faa10540837ba84dcbae7", "nonusd_fiat", "EUR", False, "medium", "Eurite EURI"),
    ("0x8df723295214ea6f21026eeeb4382d475f146f9f", "nonusd_fiat", "EUR", False, "medium", "Quantoz EURQ"),
    ("0x50753cfaf86c094925bf976f218d043f8791e408", "nonusd_fiat", "EUR", False, "medium", "StablR EURR"),
    ("0x888883b5f5d21fb10dfeb70e8f9722b9fb0e5e51", "nonusd_fiat", "EUR", False, "medium", "Schuman EUROP"),
    ("0x4933a85b5b5466fbaf179f72d3de273c287ec2c2", "nonusd_fiat", "EUR", False, "medium", "AllUnity EURAU"),
    ("0x3c89cd1884e7bef73ca3ef08d2ef6ec338fd8e49", "nonusd_fiat", "EUR", False, "low", "Usual EUR0"),
    ("0xba3f535bbcccca2a154b573ca6c5a49baae0a3ea", "nonusd_fiat", "EUR", False, "low", "Decentralized Euro dEURO"),
    ("0x735fa792e731a2e8f83f32eb539841b7b72e6d8f", "nonusd_fiat", "EUR", False, "low", "ARYZE eEUR"),
    ("0xa40640458fbc27b6eefedea1e9c9e17d4cee7a21", "nonusd_fiat", "EUR", False, "low", "Anchored Coins AEUR"),
    ("0xb399511642fe1666c6a07f83483e6e4feaed9a00", "nonusd_fiat", "EUR", False, "low", "The Standard EUROs"),
    ("0x96e61422b6a9ba0e068b6c5add4ffabc6a4aae27", "nonusd_fiat", "EUR", False, "low", "Iron Bank ibEUR, synthetic"),
    ("0x0f17bc9a994b87b5225cfb6a2cd4d667adb4f20b", "nonusd_fiat", "EUR", False, "low", "Jarvis jEUR, synthetic"),
    ("0x004626a008b1acdc4c74ab51644093b155e59a23", "nonusd_fiat", "EUR", False, "low", "Angle staked EURA, yield-bearing"),
    ("0xd711d7d893de57dc13ff465763218770bd42db1d", "nonusd_fiat", "GBP", False, "low", "ARYZE eGBP"),
    ("0x27f6c8289550fce67f6b50bed1f519966afe5287", "nonusd_fiat", "GBP", False, "low", "Tokenised GBP"),
    ("0x79d4f0232a66c4c91b89c76362016a1707cfbf4f", "nonusd_fiat", "CHF", False, "medium", "VNX Swiss Franc"),
    ("0xb58e61c3098d85632df34eecfb899a1ed80921cb", "nonusd_fiat", "CHF", False, "medium", "Frankencoin ZCHF"),
    ("0x53dfea0a8cc2a2a2e425e1c174bc162999723ea0", "nonusd_fiat", "CHF", False, "low", "Jarvis jCHF, synthetic"),
    ("0x2370f9d504c7a6e775bf6e14b3f12846b594cd53", "nonusd_fiat", "JPY", False, "medium", "JPY Coin v1"),
    ("0x431d5dff03120afa4bdf332c61a6e1766ef37bdb", "nonusd_fiat", "JPY", False, "medium", "JPY Coin"),
    ("0xe7c3d8c9a439fede00d2600032d5db0be71c3c29", "nonusd_fiat", "JPY", False, "medium", "JPY Coin (2025)"),
    ("0xc08512927d12348f6620a698105e1baac6ecd911", "nonusd_fiat", "JPY", False, "medium", "GMO GYEN"),
    ("0x1cfa5641c01406ab8ac350ded7d735ec41298372", "nonusd_fiat", "JPY", False, "low", "Convertible JPY"),
    ("0xcadc0acd4b445166f12d2c07eac6e2544fbe2eef", "nonusd_fiat", "CAD", False, "medium", "CAD Coin"),
    ("0x4cce605ed955295432958d8951d0b176c10720d5", "nonusd_fiat", "AUD", False, "medium", "Australian Digital Dollar"),
    ("0x70e8de73ce538da2beed35d14187f6959a8eca96", "nonusd_fiat", "SGD", False, "high", "StraitsX XSGD"),
    ("0x2c537e5624e4af88a7ae4060c022609376c8d0eb", "nonusd_fiat", "TRY", False, "medium", "BiLira TRYB"),
    ("0x0000000005c6b7c1fd10915a05f034f90d524d6e", "nonusd_fiat", "TRY", False, "low", "TRYC"),
    ("0xed03ed872159e199065401b6d0d487d78d9464aa", "nonusd_fiat", "MXN", False, "medium", "Tether MXNT"),
    ("0xf197ffc28c23e0309b5559e7a166f2c6164c80aa", "nonusd_fiat", "MXN", False, "low", "MXNB"),
    ("0xb755506531786c8ac63b756bab1ac387bacb0c04", "nonusd_fiat", "ZAR", False, "medium", "ZARP"),
    ("0xc00db6b41473d065027f5ed6fada20fde75f142e", "nonusd_fiat", "KRW", False, "low", "KRWQ"),
    ("0x01d33fd36ec67c6ada32cf36b31e88ee190b1839", "nonusd_fiat", "BRL", False, "medium", "Transfero BRZ"),
    ("0x6e109e9dd7fa1a58bc3eff667e8e41fc3cc07aef", "nonusd_fiat", "CNH", False, "medium", "Tether CNHT"),
    ("0x2925ac3be7d585874b88ea51ed50add376ad8239", "nonusd_fiat", "CNH", False, "low", "AnchorX AxCNH"),
    ("0x0dc4f92879b7670e5f4e4e6e3c801d229129d90d", "nonusd_fiat", "ARS", False, "low", "Argentine Peso wARS"),
    ("0x998ffe1e43facffb941dc337dd0468d52ba5b48a", "nonusd_fiat", "IDR", False, "medium", "Rupiah Token IDRT"),
    ("0xebf2096e01455108badcbaf86ce30b6e5a72aa52", "nonusd_fiat", "IDR", False, "medium", "StraitsX XIDR"),
    ("0xbe370ad45d44eb45174c4ec60b88839fef32c077", "nonusd_fiat", "PHP", False, "low", "PHT stablecoin"),
    # ---- gold and other commodities ------------------------------------------
    ("0x45804880de22913dafe09f4980848ece6ecbaf78", "commodity", "XAU", False, "high", "PAX Gold"),
    ("0x68749665ff8d2d112fa859aa293f07a622782f38", "commodity", "XAU", False, "high", "Tether Gold"),
    ("0x2103e845c5e135493bb6c2a4f0b8651956ea8682", "commodity", "XAU", False, "medium", "Matrixdock Gold"),
    ("0x6d57b2e05f26c26b549231c866bdd39779e4a488", "commodity", "XAU", False, "medium", "VNX Gold"),
    ("0x999903502c1c7fea3591fb17315544c6bdde7699", "commodity", "XAU", False, "low", "XAU9999"),
    ("0x86856814e74456893cfc8946bedcbb472b5fa856", "commodity", "XAU", False, "low", "Gold Token GLDT"),
    ("0x394d14d78850e516fa5eb88f843ef43196e136b0", "commodity", "XAU", False, "low", "Dignity Gold"),
    ("0x4f0ca3df1c2e6b943cf82e649d576ffe7b2fabcf", "commodity", "XAU", False, "medium", "Tokenised iShares Gold Trust"),
    ("0x2380f2673c640fb67e2d6b55b44c62f0e0e69da9", "commodity", "XAU", False, "low", "Gold xStock"),
]

# Reviewed false positives of the name search (not stable assets).
NOT_STABLE = [
    "0xda31d0d1bc934fc34f7189e38a413ca0a5e8b44f",  # BitStable governance
    "0x67c5870b4a41d4ebef24d2456547a03f1f3e094b",  # GoodDollar, not pegged
    "0x32353a6c91143bfd6c7d363b546e62a9a2489a20",  # Adventure Gold
    "0x57d3757a8c3a6612b75480158d77ed853081471f",  # AuditAI
    "0x18aaa7115705e8be94bffebde57af9bfc265b998",  # Audius
    "0xb90cb79b72eb10c39cbdf86e50b1c89f6a235f2e",  # Auditchain
    "0x777be1c6075c20184c4fd76344b7b0b7c858fe6b",  # Gold Standard BAR
    "0x44971abf0251958492fee97da3e5c5ada88b9185",  # BasedAI
    "0xeff9cdb294872b6553f4e9631fb4d1fa4f9c5e6b",  # BuildAI
    "0x50b0696468f42cab1ddc76413a1312aff3cabdf6",  # ClosedAI
    "0x571d9b73dc04ed88b4e273e048c8d4848f83b779",  # Super Closed Source
    "0xc00e94cb662c3520282e6f5717214004a7f26888",  # Compound COMP
    "0xd7075f79df19c279ba5a9eb04a00474c43a3d73e",  # DAIFUKU
    "0x1981e32c2154936741ab6541a737b87c68f13ce1",  # DAII
    "0x08d9040214c4aba39292fd2e7de20bbba7a8dce4",  # DataGold
    "0xe95076d9fe7155c17b455ac49497e78912c4ab65",  # "Dollar"
    "0xe9fa21e671bcfb04e6868784b89c19d5aa2424ea",  # EurocoinToken
    "0xd4318fa09c45cfb6355ded6085b0d698b64ec1cd",  # Federal AI
    "0xdab396ccf3d84cf2d07c4454e10c8a6f5b008d2b",  # Goldfinch
    "0x089453742936dd35134383aee9d78bee63a69b01",  # "Gold"
    "0x9deb0fc809955b79c85e82918e8586d3b7d2695a",  # GoldenBoys
    "0xd87996ff3d06858bfc20989aef50cc5fcd4d84ca",  # Golden Inu
    "0xdb57d9c14e357fc01e49035a808779df41e9b4e2",  # tokenised Goldman Sachs stock
    "0xc4f6e93aeddc11dc22268488465babcaf09399ac",  # hi Dollar, not pegged
    "0xda7c0810ce6f8329786160bb3d1734cf6661ca6e",  # Jaypeggers
    "0xe80c0cd204d654cebe8dd64a4857cab6be8345a3",  # JPEG'd
    "0x499568c250ab2a42292261d6121525d70691894b",  # KROWN
    "0xd014cb8d4bbccb75fc3a7c941528fd5e891259cf",  # Launch On USD1 meme
    "0xa3bed4e1c75d00fa6f4e5e6922db7261b5e9acd2",  # mStable governance
    "0x6069c9223e8a5da1ec49ac5525d4bb757af72cd8",  # MUSK Gold
    "0x0a2a015a15a8e019abc386fee88b8a6d7a0d90df",  # Machina
    "0xbb8954f008016055fda6aa1c27fa769578cae213",  # NeurochainAI
    "0x3d1c949a761c11e4cc50c3ae6bdb0f24fd7a39da",  # Neurahub
    "0x25517010a154ec41f76d203ab221019a37c2dab6",  # Neuralis AI
    "0x32b053f2cba79f80ada5078cb6b305da92bde6e1",  # NEURALAI
    "0xab814ce69e15f6b9660a3b184c0b0c97b9394a6b",  # Cerebrum DAO
    "0x2653891204f463fb2a2f4f412564b19e955166ae",  # Gold Fever NGL
    "0x810296898a4f870619b015c0b13c8c6e65b305e3",  # Nugget Trap Gold
    "0x323c03c48660fe31186fa82c289b0766d331ce21",  # Open Stablecoin Index
    "0x70b5fa2c02a63b1a90d23bd8166fa81614b6d463",  # Pepe Dollar
    "0x9634bdb20bbab07bb52d279fa6e0c53ccc89c879",  # Pepega
    "0x11ebe21e9d7bf541a18e1e3ac94939018ce88f0b",  # Pitch FXS
    "0x03ab458634910aad20ef5f1c8ee96f1d6ac54919",  # RAI reflex index, floats
    "0xc575bd129848ce06a460a19466c30e1d0328f52c",  # Reploy
    "0x60e254e35dd712394b3aba7a1d19114732e143dd",  # RivusDAO
    "0x4e4990e997e1df3f6b39ff49384e2e7e99bc55fe",  # Saudi Bonk
    "0xd94a8f9caed25e63ecc90edfefaf3635ea1e182a",  # Stablecomp governance
    "0x60b9c41d99fe3eb64ecc1344bad31d87f1bced6d",  # "Stable"
    "0xa6089dbfed19d1bcd43146bbdca2b8f9d9f84a9a",  # UGOLD Inc.
    "0x00c5ca160a968f47e7272a0cfcda36428f386cb6",  # USDEBT meme
    "0xecedb6f8108b9f7bbf499da843dced6c2bb6e270",  # Unstable Coin meme
    "0xd13cfd3133239a3c73a9e535a5c4dadee36b395c",  # VAIOT
    "0x7659ce147d0e714454073a5dd7003544234b6aa0",  # XCAD Network
    "0xf4cd3d3fda8d7fd6c5a500203e38640a70bf9577",  # YfDAI.finance
    "0xfdc9d2a3cae56e484a85de3c2e812784a8184d0d",  # ErectusDAO
    "0x9d409a0a012cfba9b15f6d4b36ac57a46966ab9a",  # Yearn yvBOOST
    "0xfa3e941d1f6b7b10ed84a0c211bfa8aee907965e",  # HayCoin
    "0x6958c870a6d9a7a7cca58fede8acfdf1280c12d5",  # Bean Coin
    "0xac3e018457b222d93114458476f3e3416abbe38f",  # Staked Frax Ether (ETH derivative)
    "0xc539f8e194569b3db935d70fa2e7cadd7dad7f35",  # Crypto Czar
    "0x4a7c9897ae01ff08d6e3820507a6b967bdbffa29",  # Wizard Cat
    "0x87931e7ad81914e7898d07c68f145fc0a553d8fb",  # WIZARD Vault (NFTX)
]


def load_whitelist(status_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(status_csv, dtype=str, keep_default_na=False)
    for c in ("is_erc_20", "local_status", "price_data_status"):
        df[c] = df[c].str.strip().str.lower()
    df["contract_address"] = df["contract_address"].str.strip().str.lower()
    wl = df[(df["is_erc_20"] == "true") & (df["local_status"] == "pass") & (df["price_data_status"] == "success")]
    return wl.drop_duplicates("contract_address")


def peg_check(prices_dir: Path, addr: str, start: str, end: str) -> dict:
    f = prices_dir / f"{addr}_price.csv"
    if not f.exists():
        return {"peg_days": 0}
    p = pd.read_csv(f)
    p = p[(p["date"] >= start) & (p["date"] <= end)]
    s = pd.to_numeric(p["daily_close_usd"], errors="coerce").dropna()
    if s.empty:
        return {"peg_days": 0}
    dev = (s - 1.0).abs()
    return {"peg_days": int(len(s)), "peg_median": round(float(s.median()), 4), "peg_min": round(float(s.min()), 4),
            "peg_off2_share": round(float((dev > 0.02).mean()), 4), "peg_off5_share": round(float((dev > 0.05).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the curated stablecoin list (../data/stablecoins.csv).")
    ap.add_argument("--status-csv", default=str(DEFAULT_STATUS_CSV))
    ap.add_argument("--out-csv", default=str(DEFAULT_STABLECOINS_CSV))
    ap.add_argument("--prices-dir", default=None, help="Optional price folder for the peg check.")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()

    wl = load_whitelist(Path(args.status_csv))
    wl_by_addr = wl.set_index("contract_address")
    log(f"whitelist: {len(wl):,} tokens from {args.status_csv}")

    addrs = [c[0] for c in CURATED] + NOT_STABLE
    dupes = sorted({a for a in addrs if addrs.count(a) > 1})
    if dupes:
        raise SystemExit(f"[error] address listed twice in the curated table: {dupes}")

    rows = []
    for addr, cls, peg, core, conf, note in CURATED:
        rows.append(dict(contract_address=addr, asset_class=cls, peg=peg, cash_core=bool(core),
                         cash_extended=cls in CASH_EXTENDED_CLASSES, confidence=conf, note=note))
    for addr in NOT_STABLE:
        rows.append(dict(contract_address=addr, asset_class="not_stable", peg="", cash_core=False,
                         cash_extended=False, confidence="high", note="reviewed false positive"))
    out = pd.DataFrame(rows)
    assert not (out["cash_core"] & ~out["cash_extended"]).any(), "cash_core must imply cash_extended"

    out["in_whitelist"] = out["contract_address"].isin(wl_by_addr.index)
    out["symbol"] = out["contract_address"].map(wl_by_addr["symbol"]).fillna("")
    out["name"] = out["contract_address"].map(wl_by_addr["name"]).fillna("")
    missing = out[~out["in_whitelist"]]
    if not missing.empty:
        log(f"[warn] {len(missing)} curated address(es) not in the whitelist (kept, flagged): "
            f"{missing['contract_address'].tolist()[:10]}")

    # Candidates of the name search that nobody reviewed yet
    sym, name = wl["symbol"].str.lower(), wl["name"].str.lower()
    cand = wl[sym.str.contains(CANDIDATE_SYMBOL_RE) | name.str.contains(CANDIDATE_NAME_RE)]
    unreviewed = cand[~cand["contract_address"].isin(set(out["contract_address"]))]
    if unreviewed.empty:
        log(f"name search: {len(cand)} candidates, all reviewed")
    else:
        log(f"[review] {len(unreviewed)} candidate(s) not in the curated table; classify them in CURATED or NOT_STABLE:")
        for _, r in unreviewed.iterrows():
            print(f"    {r['contract_address']}  {r['symbol']:<12s} {r['name']}")

    if args.prices_dir:
        pdir = Path(args.prices_dir)
        if not pdir.is_dir():
            raise SystemExit(f"[error] --prices-dir {pdir} is not a directory")
        checks = []
        for _, r in out.iterrows():
            if r["peg"] == "USD" and r["asset_class"] not in ("usd_yield", "usd_lp"):
                checks.append({"contract_address": r["contract_address"],
                               **peg_check(pdir, r["contract_address"], args.start, args.end)})
        if checks:
            # tokens without a price file only carry peg_days = 0, so fix the columns explicitly
            peg = pd.DataFrame(checks).reindex(columns=["contract_address", "peg_days", "peg_median", "peg_min",
                                                        "peg_off2_share", "peg_off5_share"])
            log(f"peg check: {int((peg['peg_days'] > 0).sum())} of {len(peg)} USD stablecoins have prices in {pdir}")
            out = out.merge(peg, on="contract_address", how="left")
            flag = out["peg_off5_share"].fillna(0) > 0.10
            out["peg_flag"] = flag
            for _, r in out[flag].iterrows():
                log(f"[peg] {r['symbol']:<10s} {r['asset_class']:<13s} cash_core={r['cash_core']} "
                    f"median={r['peg_median']} min={r['peg_min']} off5%={r['peg_off5_share']:.1%} ({r['note']})")

    class_order = ["usd_fiat", "usd_crypto", "usd_synthetic", "usd_rwa", "usd_yield", "usd_lp",
                   "usd_unstable", "usd_suspect", "nonusd_fiat", "commodity", "not_stable"]
    out["_o"] = out["asset_class"].map({c: i for i, c in enumerate(class_order)})
    out = out.sort_values(["_o", "cash_core", "symbol"], ascending=[True, False, True]).drop(columns="_o")
    lead = ["contract_address", "symbol", "name", "asset_class", "peg", "cash_core", "cash_extended",
            "confidence", "in_whitelist", "note"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    summary = out.groupby("asset_class").agg(tokens=("contract_address", "size"),
                                            cash_core=("cash_core", "sum"),
                                            cash_extended=("cash_extended", "sum"))
    log(f"wrote {args.out_csv} ({len(out)} rows)\n{summary.reindex(class_order).dropna().astype(int).to_string()}")


if __name__ == "__main__":
    main()
