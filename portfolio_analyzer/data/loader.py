"""Data ingestion: load holdings from CSV/XLSX and investor config from CSV.

Handles column validation, number parsing, optional geo columns,
and per-holding target columns. Also loads InvestorConfig from CSV
or returns defaults.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional, Union

import pandas as pd

from portfolio_analyzer.models.holding import Geography, Holding
from portfolio_analyzer.models.investor_config import InvestorConfig
from portfolio_analyzer.exceptions import DataIngestionError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = frozenset({
    "isin", "ticker", "quantity", "cost_basis_eur", "market_value_eur", "currency",
})

_GEO_COLUMNS = {
    "usa": "USA",
    "emerging_markets": "Emerging Markets",
    "eurozone_emu": "Eurozone EMU",
    "japan": "Japan",
    "dev_ex_usa_ex_emu_ex_jp": "Dev ex-USA ex-EMU ex-JP",
}


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
    rows = {
        row["key"].strip(): row["value"].strip()
        for row in reader
        if "key" in row and "value" in row
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
    """Parse a single row into a Holding, returning None if invalid."""
    qty = _parse_number_safe(row["quantity"], "quantity", idx)
    if qty is None or qty <= 0:
        if qty is not None:
            logger.warning("Row %d: quantity %.4f <= 0, skipping", idx, qty)
        return None

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

    # Parse optional geography columns from input
    geo = _parse_input_geo(row, columns)
    if geo:
        holding.input_geo = geo
        holding.input_geo_source = (
            str(row.get("source_and_timestamp", "input_csv")).strip() or "input_csv"
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


def _parse_input_geo(row: pd.Series, columns) -> Optional[dict]:
    geo_lookup = {g.value: g for g in Geography}
    result = {}
    has_any = False
    for col_name, geo_value in _GEO_COLUMNS.items():
        if col_name in columns:
            try:
                pct = _parse_number(row[col_name])
                if pct > 0:
                    geo = geo_lookup.get(geo_value)
                    if geo:
                        result[geo] = pct
                        has_any = True
            except (ValueError, TypeError):
                pass
    return result if has_any else None