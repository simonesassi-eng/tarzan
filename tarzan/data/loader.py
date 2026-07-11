"""Data ingestion: load the order list and investor config from CSV.

Handles column validation, number parsing, and per-holding target columns.
The portfolio snapshot is derived from the order list (see
``returns_builder.build_holdings_from_orders``); this module no longer
ingests a standalone holdings snapshot.
"""

from __future__ import annotations

import io
import logging
import math
import os
from typing import Optional, Union

import pandas as pd

from tarzan.models.order import Order, OrderType
from tarzan.models.investor_config import InvestorConfig
from tarzan.exceptions import DataIngestionError

logger = logging.getLogger(__name__)

# Minimal set the order list must carry for returns. The full schema has
# more columns (trade_date, name, price_native, fx_rate, fees_eur,
# source); those are optional and default when absent.
ORDER_REQUIRED_COLUMNS = frozenset({
    "date", "type", "isin", "quantity", "gross_eur", "net_eur",
})


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


def load_orders(source: Union[str, io.BytesIO], filename: str = "") -> list[Order]:
    """Load and validate the order list into typed Order objects.

    Reuses ``_read_source`` (path/.csv/.xlsx/BytesIO), lowercases columns,
    validates the canonical schema, and skips malformed or unknown-type
    rows with a warning rather than failing the whole load.

    A missing file path returns ``[]`` (not an exception) so the
    orchestrator can log and exit gracefully.

    Args:
        source: File path (str) or BytesIO buffer.
        filename: Original filename (needed for BytesIO extension detection).

    Returns:
        List of validated Order objects (possibly empty).
    """
    # Missing path → empty list, let the caller decide it's non-fatal.
    if isinstance(source, str) and not os.path.exists(source):
        logger.warning("Order list not found at %s; treating as no orders.", source)
        return []

    df = _read_source(source, filename)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = ORDER_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataIngestionError(
            f"Order list missing required columns: {', '.join(sorted(missing))}"
        )

    orders: list[Order] = []
    skipped = 0
    for idx, row in df.iterrows():
        try:
            order = _parse_order_row(idx, row)
            if order is not None:
                orders.append(order)
            else:
                skipped += 1
        except Exception as e:
            logger.warning("Order row %d: unexpected error '%s', skipping", idx, e)
            skipped += 1

    if skipped:
        logger.info("Skipped %d order row(s) with invalid/unknown data", skipped)
    logger.info("Loaded %d orders from %s", len(orders), filename or str(source))
    return orders


def _parse_order_row(idx: int, row: pd.Series) -> Optional[Order]:
    """Parse one order-list row into an Order, or None to skip it."""
    otype = OrderType.from_raw(row.get("type"))
    if otype is None:
        logger.warning("Order row %d: unknown type %r, skipping", idx, row.get("type"))
        return None

    order_date = _parse_date_safe(row.get("date"))
    if order_date is None:
        logger.warning("Order row %d: invalid date %r, skipping", idx, row.get("date"))
        return None
    trade_date = _parse_date_safe(row.get("trade_date")) or order_date

    isin = str(row.get("isin", "")).strip()
    if not isin or isin.lower() == "nan":
        logger.warning("Order row %d: missing ISIN, skipping", idx)
        return None

    quantity = _parse_number_safe(row.get("quantity"), "quantity", idx)
    gross_eur = _parse_number_safe(row.get("gross_eur"), "gross_eur", idx)
    net_eur = _parse_number_safe(row.get("net_eur"), "net_eur", idx)
    if quantity is None or gross_eur is None or net_eur is None:
        return None

    currency = str(row.get("currency", "EUR")).strip().upper() or "EUR"
    if currency.lower() == "nan":
        currency = "EUR"

    return Order(
        date=order_date,
        trade_date=trade_date,
        type=otype,
        isin=isin,
        name=_clean_str(row.get("name")),
        ticker=_clean_str(row.get("ticker")),
        quantity=quantity,
        currency=currency,
        price_native=_parse_number_optional(row.get("price_native")),
        fx_rate=_parse_number_optional(row.get("fx_rate")),
        gross_eur=gross_eur,
        fees_eur=_parse_number_safe(row.get("fees_eur"), "fees_eur", idx) or 0.0,
        net_eur=net_eur,
        source=_clean_str(row.get("source")) or "fineco",
    )


def _clean_str(val) -> str:
    """Return a stripped string, mapping NaN/None to empty string."""
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def load_targets_per_holding(
    source: Union[str, io.BytesIO], filename: str = ""
) -> dict[str, dict]:
    """Load per-holding rebalancing targets keyed by ISIN.

    The order list carries no target columns, so when the snapshot is
    derived from orders (order-only mode) the rebalancer's per-holding
    targets come from this side file instead. Expected columns:
    ``name, isin, ticker, target_equities, target_fixed_income,
    no_buy_no_sell`` (only ``isin`` is required; the rest are optional and
    blanks map to "no target").

    A missing path returns ``{}`` (non-fatal) so the pipeline runs without
    it — holdings simply carry no per-instrument target.

    Columns (case-insensitive): ``name, isin, ticker, target_equities,
    target_fixed_income, target_portfolio, no_buy_no_sell``. ``target_portfolio``
    is the instrument's target weight as a % of the whole invested portfolio
    (used by the per-holding-only rebalancing mode). At least one of ``isin``
    / ``ticker`` must be present per row.

    Each row is keyed by its ISIN when present, otherwise by its uppercased
    ticker — so ticker-only rows (e.g. a not-yet-held instrument to buy) are
    kept, and the caller can match holdings by ISIN or ticker.

    Returns:
        ``{key: {"isin", "ticker", "name", "target_equities",
                 "target_fixed_income", "target_portfolio", "no_buy_no_sell"}}``.
    """
    if isinstance(source, str) and not os.path.exists(source):
        logger.info("No per-holding targets at %s; none applied.", source)
        return {}

    df = _read_source(source, filename)
    df.columns = [c.strip().lower() for c in df.columns]
    if "isin" not in df.columns and "ticker" not in df.columns:
        logger.warning("Per-holding targets file has no 'isin'/'ticker' column; ignoring.")
        return {}

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        isin = _clean_str(row.get("isin"))
        ticker = _clean_str(row.get("ticker"))
        key = isin or ticker.upper()
        if not key:
            continue
        nbns = str(row.get("no_buy_no_sell", "")).strip().lower()
        result[key] = {
            "isin": isin,
            "ticker": ticker,
            "name": _clean_str(row.get("name")),
            "target_equities": _parse_number_optional(row.get("target_equities")),
            "target_fixed_income": _parse_number_optional(row.get("target_fixed_income")),
            "target_portfolio": _parse_number_optional(row.get("target_portfolio")),
            "no_buy_no_sell": nbns in ("true", "1", "yes"),
        }
    logger.info("Loaded per-holding targets for %d instrument(s)", len(result))
    return result


def _parse_date_safe(val):
    """Parse a value into a date, or None on failure/empty."""
    if val is None or (isinstance(val, str) and not val.strip()):
        return None
    try:
        ts = pd.to_datetime(val, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except (ValueError, TypeError):
        return None


def _parse_number_optional(val) -> Optional[float]:
    """Parse a number, returning None for blanks/NaN (not 0)."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return _parse_number(val)
    except (ValueError, TypeError):
        return None


def _read_source(source: Union[str, io.BytesIO], filename: str) -> pd.DataFrame:
    """Read an input file (order list / targets) from path or BytesIO."""
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Input file not found: {source}")
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


def _parse_number_safe(val, field_name: str, row_idx: int) -> Optional[float]:
    try:
        return _parse_number(val)
    except (ValueError, TypeError):
        logger.warning("Row %d: invalid %s '%s', skipping", row_idx, field_name, val)
        return None


def _parse_number(val) -> float:
    """Parse a numeric cell tolerant of both US and European notation.

    Handles thousands separators and either decimal mark by treating the
    *rightmost* '.' or ',' as the decimal separator and stripping the
    other as a grouping separator:

        "1,234.56"  → 1234.56   (US: dot decimal, comma grouping)
        "1.234,56"  → 1234.56   (European: comma decimal, dot grouping)
        "1234,56"   → 1234.56   (European, no grouping)
        "1,5"       → 1.5       (European decimal)
        "99,750"    → 99.75     (European 3-decimal price — NOT 99750)
        "1,234,567" → 1234567   (US grouping — unambiguous, multiple commas)
        "1234"      → 1234.0

    A SINGLE comma with no dot is the ambiguous case ("1,234" could be US
    1234 or European 1.234). Tarzan's upstream is Fineco (European), and —
    critically — real numeric cells never reach this string path at all:
    pandas types them as float64 and they take the ``isinstance`` fast path
    above, while the config layer parses its own numbers with plain
    ``float()``. Only a hand-edited quoted string lands here, so we read a
    single comma as the European decimal mark. Multiple commas are
    unambiguous US grouping (European grouping uses dots), so those are
    stripped. NOTE: a single DOT is always kept as a decimal mark (real ETF
    prices are genuinely 3-decimal, e.g. 31.975), so it is never treated as
    grouping.
    """
    if isinstance(val, (int, float)):
        # A blank numeric cell arrives here as numpy/pandas NaN (which is a
        # float), and a bad computation can yield inf. Neither is a valid
        # monetary/quantity value: reject it so the caller skips the row
        # rather than silently building an Order(quantity=nan) that poisons
        # NAV, weights and XIRR to NaN with no error surfaced.
        f = float(val)
        if not math.isfinite(f):
            raise ValueError("non-finite")
        return f
    s = str(val).strip()
    if not s:
        raise ValueError("empty")

    # Preserve a leading sign, operate on the magnitude.
    sign = ""
    if s[0] in "+-":
        sign, s = s[0], s[1:]

    has_dot = "." in s
    has_comma = "," in s
    if has_dot and has_comma:
        # The rightmost of the two is the decimal separator.
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        grp = "," if dec == "." else "."
        s = s.replace(grp, "").replace(dec, ".")
    elif has_comma:
        # Only commas present. Multiple commas are unambiguous US grouping
        # (European grouping uses dots), so strip them. A single comma is the
        # ambiguous case — read it as the European decimal mark, since the
        # Fineco source is European and a comma-3-digit value like "99,750" is
        # a 99.75 price far more often than ninety-nine thousand. (Real data
        # never reaches here; see the docstring.)
        if s.count(",") > 1:
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif has_dot:
        # Only dots present. A single dot with 3 trailing digits is
        # ambiguous (1.234 could be 1234 EU-grouped or 1.234 US-decimal);
        # default to the US/standard reading (decimal) since our pipelines
        # emit US-format numbers. Multiple dots → grouping.
        if s.count(".") > 1:
            s = s.replace(".", "")
    return float(sign + s)