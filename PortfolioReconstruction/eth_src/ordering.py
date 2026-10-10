#!/usr/bin/env python3
"""
ordering.py

Purpose
-------
log_index synthesis and deterministic ordering.

Native ETH has no real log index, so we synthesize a stable per-block
ordinal. Within each block rows sort by:

1. transaction_index ascending; rows with no tx (genesis, daofork, reward,
   withdrawal) get a sentinel that sorts after all real txs, ordered
   genesis -> daofork -> reward -> (fee-related-none) -> withdrawal.
2. within a tx: value-transfer rows (by trace_address compared as a tuple
   of ints, empty/root first) before that tx's fee_burn / fee_tip rows.
3. log_index = row_number() - 1 per block.

IMPORTANT: the synthesized log_index is unique only within (block) for the
ETH ledger and MAY COLLIDE with real ERC-20 log_index values. When unioning
with the ERC-20 transfers table downstream, add a source discriminator
column.

The union feeding these expressions must provide the columns:
  block, transfer_type, tx_index (BIGINT, NULL for no-tx rows),
  ta_key (VARCHAR trace_address sort key, see below; NULL for non-trace rows),
  sub_index (BIGINT, withdrawal ordinal within the block, NULL otherwise),
  recipient, value, tx_hash.

trace_address is compared as a tuple of ints via a FIXED-WIDTH string key
('' for root, else e.g. '000000001.000000003'): zero-padding makes
lexicographic order equal numeric tuple order, and a plain VARCHAR sort key
external-sorts cleanly where a LIST-typed key is memory-hungry.
"""

from __future__ import annotations

from config import NO_TX_SENTINEL

# Rows with a tx all share rank -1 here (their mutual order is decided by
# tx_index/ta/intra-rank); no-tx rows order by type.
NOTX_RANK_SQL = (
    "CASE WHEN tx_index IS NOT NULL THEN -1 ELSE "
    "CASE transfer_type WHEN 'genesis' THEN 0 WHEN 'daofork' THEN 1 "
    "WHEN 'reward' THEN 2 WHEN 'withdrawal' THEN 4 ELSE 3 END END"
)

# Within a tx: value transfers first, then fee_burn, then fee_tip.
INTRA_TX_RANK_SQL = (
    "CASE transfer_type WHEN 'fee_burn' THEN 1 WHEN 'fee_tip' THEN 2 ELSE 0 END"
)


def order_key_exprs() -> list[str]:
    """SQL sort-key expressions, in priority order, for the per-block window.

    The trailing recipient/value/tx_hash keys only break ties between rows
    that are otherwise unordered (e.g. two uncle rewards in one block) so the
    output is fully deterministic across runs.
    """
    return [
        f"COALESCE(tx_index, {NO_TX_SENTINEL})",
        NOTX_RANK_SQL,
        INTRA_TX_RANK_SQL,
        "COALESCE(ta_key, '')",
        "COALESCE(sub_index, -1)",
        "recipient NULLS LAST",
        "value",
        "tx_hash NULLS LAST",
    ]


def log_index_window_sql() -> str:
    """The window expression assigning the synthetic per-block log_index."""
    keys = ",\n               ".join(order_key_exprs())
    return (
        "row_number() OVER (\n"
        "      PARTITION BY block\n"
        f"      ORDER BY {keys}\n"
        "    ) - 1"
    )


def trace_address_sort_key_sql(col: str, kind: str) -> str:
    """SQL expression turning a raw trace_address column into the fixed-width
    VARCHAR sort key: '' for root, else zero-padded 9-digit elements joined
    by '.' (e.g. '0,12' -> '000000000.000000012'). Lexicographic order on
    this key equals numeric tuple order; a parent's key is a strict prefix
    of its children's keys.

    kind: 'string' (ethereum-etl: '' or '0,1,2'), 'list', or 'missing'.
    """
    if kind == "missing":
        return "''"
    if kind == "list":
        elems = f"list_transform({col}, x -> lpad(trim(CAST(x AS VARCHAR)), 9, '0'))"
        return (
            f"CASE WHEN {col} IS NULL THEN '' "
            f"ELSE array_to_string({elems}, '.') END"
        )
    elems = (
        f"list_transform(string_split(CAST({col} AS VARCHAR), ','), "
        f"x -> lpad(trim(x), 9, '0'))"
    )
    return (
        f"CASE WHEN {col} IS NULL OR trim(CAST({col} AS VARCHAR)) = '' THEN '' "
        f"ELSE array_to_string({elems}, '.') END"
    )


def trace_address_debug_sql(col: str, kind: str) -> str:
    """SQL expression rendering trace_address as a plain string debug column."""
    if kind == "missing":
        return "CAST(NULL AS VARCHAR)"
    if kind == "list":
        return f"array_to_string({col}, ',')"
    return f"CAST({col} AS VARCHAR)"
