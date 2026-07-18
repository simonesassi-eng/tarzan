"""Versioned executable contracts for Tarzan input files.

The dependency-free declarations in this module are the format authority for
CSV and spreadsheet inputs. Boundary enforcement lives in
:mod:`tarzan.contracts.validation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Bump when an input file's column contract changes in a way that affects how
# a user must format their file. Surfaced in docs and (optionally) checked.
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ColumnSpec:
    """One column in an input file's contract."""

    name: str                       # canonical column name (lowercased)
    required: bool                  # must be present for the file to be usable
    dtype: str                      # "date" | "string" | "number" | "enum" | "bool"
    description: str
    aliases: tuple[str, ...] = ()   # accepted alternate header spellings
    enum_values: tuple[str, ...] = ()   # for dtype == "enum"
    example: str = ""


@dataclass(frozen=True)
class FileSchema:
    """The contract for one input CSV/XLSX file."""

    file: str                       # canonical filename
    purpose: str
    columns: tuple[ColumnSpec, ...]
    version: int = SCHEMA_VERSION

    def required_columns(self) -> set[str]:
        return {c.name for c in self.columns if c.required}

    def known_columns(self) -> set[str]:
        """Every canonical name plus all accepted aliases (lowercased)."""
        out: set[str] = set()
        for c in self.columns:
            out.add(c.name)
            out.update(a.lower() for a in c.aliases)
        return out

    def to_markdown(self) -> str:
        """A human-readable column reference (for docs / error help)."""
        lines = [f"### `{self.file}` (schema v{self.version})", "",
                 self.purpose, "",
                 "| Column | Required | Type | Description |",
                 "|--------|:--------:|------|-------------|"]
        for c in self.columns:
            req = "✓" if c.required else ""
            lines.append(f"| `{c.name}` | {req} | {c.dtype} | {c.description} |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The order list — the single source of truth for the whole report.
# Mirrors exactly what tarzan.data.loader._parse_order_row reads.
# ---------------------------------------------------------------------------

ORDER_LIST_SCHEMA = FileSchema(
    file="order_list.csv",
    purpose=("The dated journal of every buy/sell/transfer/coupon/dividend. "
             "The snapshot, allocations and XIRR/TWROR are all derived from it."),
    columns=(
        ColumnSpec("date", True, "date",
                   "Settlement/value date (YYYY-MM-DD preferred).", example="2025-01-15"),
        ColumnSpec("type", True, "enum",
                   "Movement kind.",
                   enum_values=("buy", "sell", "coupon", "dividend",
                                "transfer_in", "transfer_out"),
                   example="buy"),
        ColumnSpec("isin", True, "string",
                   "12-char ISIN (format-checked, not mod-10). Required unless "
                   "a ticker uniquely identifies the instrument.", example="IE00B4L5Y983"),
        ColumnSpec("quantity", True, "number",
                   "Units, signed by direction (+ buy/transfer_in, − sell; 0 for "
                   "coupon/dividend).", example="100"),
        ColumnSpec("gross_eur", True, "number",
                   "Signed gross amount in EUR before fees.", example="10000"),
        ColumnSpec("net_eur", True, "number",
                   "Signed net bank cash flow in EUR (− for buys).", example="-10000"),
        ColumnSpec("instrument_kind", False, "enum",
                   "Exact mechanics kind. Required for order-derived valuation "
                   "when provider evidence is unavailable; category/name/price "
                   "never substitute for it.",
                   enum_values=("STOCK", "ETF", "BOND", "CASH"),
                   example="ETF"),
        ColumnSpec("instrument_equivalence_group", False, "string",
                   "Explicit shared identity for documented equivalent identifiers "
                   "such as cum/ex variants. Blank keeps cost basis isolated by full ISIN.",
                   example="BTP-VALORE-2028-CUM-EX"),
        ColumnSpec("trade_date", False, "date",
                   "Order/market-exposure date; defaults to `date` if absent.",
                   example="2025-01-13"),
        ColumnSpec("name", False, "string", "Instrument display name."),
        ColumnSpec("ticker", False, "string", "Yahoo-style ticker, if known.",
                   example="VWCE.DE"),
        ColumnSpec("currency", False, "string",
                   "ISO-4217 code of the instrument; defaults to EUR.", example="USD"),
        ColumnSpec("price_native", False, "number",
                   "Trade price in the instrument's own currency."),
        ColumnSpec("fx_rate", False, "number",
                   "Native-per-EUR FX at trade (Fineco 'Cambio')."),
        ColumnSpec("fees_eur", False, "number", "Fees in EUR; defaults to 0."),
        ColumnSpec("source", False, "string", "Provenance tag; defaults to 'fineco'."),
    ),
)


# ---------------------------------------------------------------------------
# Per-holding rebalancing targets (optional side file).
# ---------------------------------------------------------------------------

TARGETS_PER_HOLDING_SCHEMA = FileSchema(
    file="targets_per_holding.csv",
    purpose=("Optional per-instrument rebalancing targets, joined to the "
             "order-derived snapshot by ISIN or ticker."),
    columns=(
        ColumnSpec("isin", False, "string",
                   "Instrument ISIN. At least one of isin/ticker is required per row."),
        ColumnSpec("ticker", False, "string",
                   "Instrument ticker. At least one of isin/ticker is required per row."),
        ColumnSpec("name", False, "string", "Display name."),
        ColumnSpec("target_equities", False, "number",
                   "Target weight as % of the equity sleeve."),
        ColumnSpec("target_fixed_income", False, "number",
                   "Target weight as % of the fixed-income sleeve."),
        ColumnSpec("target_portfolio", False, "number",
                   "Target weight as % of the whole invested portfolio "
                   "(per-holding-only mode)."),
        ColumnSpec("no_buy_no_sell", False, "bool",
                   "If true, freeze this holding (exclude from rebalancing)."),
    ),
)


ALL_SCHEMAS: tuple[FileSchema, ...] = (
    ORDER_LIST_SCHEMA,
    TARGETS_PER_HOLDING_SCHEMA,
)


def schemas_markdown() -> str:
    """Full input-format reference for all files (docs / --help output)."""
    return "\n\n".join(s.to_markdown() for s in ALL_SCHEMAS)
