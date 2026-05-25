"""Data ingestion: load holdings from CSV/XLSX and investor config from CSV.

Handles column validation, number parsing, and per-holding target columns.
Also loads InvestorConfig from CSV or returns defaults.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional, Union

import pandas as pd

from tarzan.models.holding import Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.exceptions import DataIngestionError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = frozenset({
    "isin", "ticker", "quantity", "cost_basis_eur", "market_value_eur", "currency",
})


def load_holdings(source: Union[str, io.BytesIO], filename: str = "") -> list[Holding]:
    """Load holdings from a file path or uploaded BytesIO buffer.

    Args:
        source: File path (str) or BytesIO from st.file_uploader.
        filename: Original filename (needed for BytesIO to detect extension).

    Returns:
        List of validated Holding objects.
    """
    df = _read_source(source, filename)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataIngestionError(f"Missing required columns: {', '.join(sorted(missing))}")

    holdings: list[Holding] = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            holding = _parse_row(idx, row, df.columns)
            if holding is not None:
                holdings.append(holding)
            else:
                skipped += 1
        except Exception as e:
            logger.warning("Row %d: unexpected error '%s', skipping", idx, e)
            skipped += 1

    if skipped:
        logger.info("Skipped %d rows with invalid data", skipped)
    logger.info("Loaded %d holdings from %s", len(holdings), filename or str(source))
    return holdings


def load_config(source: Optional[Union[str, io.BytesIO]] = None) -> InvestorConfig:
    """Load investor config from CSV path, BytesIO, or return defaults."""
    if source is None:
        logger.info("No config file, using defaults")
        return InvestorConfig()

    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Config file not found: {source}")
        return InvestorConfig.from_csv(source)

    # BytesIO from file uploader
    import csv as csv_mod
    source.seek(0)
    text = source.read().decode("utf-8")
    reader = csv_mod.DictReader(io.StringIO(text))
    # Only ``key`` and ``value`` columns are required. Extra columns
    # (e.g. ``description``) are tolerated so the config file can be
    # self-documenting without breaking the parser.
    rows = {
        row["key"].strip(): row["value"].strip()
        for row in reader
        if row.get("key") and "value" in row
    }
    return InvestorConfig.from_dict(rows)


def _read_source(source: Union[str, io.BytesIO], filename: str) -> pd.DataFrame:
    """Read a holdings file from path or BytesIO."""
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Holdings file not found: {source}")
        ext = os.path.splitext(source)[1].lower()
        if ext == ".xlsx":
            return pd.read_excel(source, sheet_name=0)
        elif ext == ".csv":
            return pd.read_csv(source)
        raise DataIngestionError(f"Unsupported format: {ext}")

    # BytesIO
    ext = os.path.splitext(filename)[1].lower() if filename else ".csv"
    source.seek(0)
    if ext == ".xlsx":
        return pd.read_excel(source, sheet_name=0)
    return pd.read_csv(source)


def _parse_row(idx: int, row: pd.Series, columns) -> Optional[Holding]:
    """Parse a single row into a Holding, returning None if invalid.

    A row with ``quantity <= 0`` is normally skipped, but kept when it
    carries a strictly positive target (``target_equities`` or
    ``target_fixed_income``). This lets users declare a holding they
    want to start building — the rebalancer will then propose buys
    into it from cash or from over-target holdings.
    """
    qty = _parse_number_safe(row["quantity"], "quantity", idx)

    # Detect "target-only" placeholder rows: qty == 0 (or invalid)
    # but at least one of the target columns is positive.
    has_positive_target = False
    for col in ("target_equities", "target_fixed_income"):
        if col in columns:
            try:
                val = _parse_number(row[col])
                if val is not None and val > 0:
                    has_positive_target = True
                    break
            except (ValueError, TypeError):
                continue

    if qty is None or qty <= 0:
        if not has_positive_target:
            if qty is not None:
                logger.warning("Row %d: quantity %.4f <= 0, skipping", idx, qty)
            return None
        # qty <= 0 but the user has expressed a target — keep the row
        # as a zero-balance placeholder so the optimizer can buy into
        # it. Force qty/cost/value to 0 to avoid downstream surprises.
        logger.info(
            "Row %d: quantity 0 with positive target — kept as target placeholder.",
            idx,
        )
        qty = 0.0
        cost_basis = 0.0
        market_value = 0.0
    else:
        cost_basis = _parse_number_safe(row["cost_basis_eur"], "cost_basis_eur", idx)
        if cost_basis is None:
            return None

        market_value = _parse_number_safe(row["market_value_eur"], "market_value_eur", idx)
        if market_value is None:
            return None

    isin = str(row.get("isin", "")).strip()
    ticker = str(row.get("ticker", "")).strip()
    currency = str(row.get("currency", "EUR")).strip().upper()

    # Handle NaN values from pandas (empty CSV cells)
    if isin.lower() == "nan":
        isin = ""
    if ticker.lower() == "nan":
        ticker = ""
    if currency.lower() == "NAN":
        currency = "EUR"

    if not ticker and isin:
        ticker = isin

    holding = Holding(
        isin=isin, ticker=ticker, quantity=qty,
        cost_basis_eur=cost_basis, market_value_eur=market_value, currency=currency,
    )

    # Parse optional target columns
    if "target_equities" in columns:
        try:
            te = _parse_number(row["target_equities"])
            if te >= 0:
                holding.target_equities = te
        except (ValueError, TypeError):
            pass
    if "target_fixed_income" in columns:
        try:
            tfi = _parse_number(row["target_fixed_income"])
            if tfi >= 0:
                holding.target_fixed_income = tfi
        except (ValueError, TypeError):
            pass

    # Parse no_buy_no_sell flag
    if "no_buy_no_sell" in columns:
        val = str(row.get("no_buy_no_sell", "")).strip().lower()
        holding.no_buy_no_sell = val in ("true", "1", "yes")

    return holding


def _parse_number_safe(val, field_name: str, row_idx: int) -> Optional[float]:
    try:
        return _parse_number(val)
    except (ValueError, TypeError):
        logger.warning("Row %d: invalid %s '%s', skipping", row_idx, field_name, val)
        return None


def _parse_number(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(",", "")
    return float(s)