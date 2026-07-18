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

from tarzan.runtime import data_quality as dq
from tarzan.instruments.registry import InstrumentKind
from tarzan.models.order import Order, OrderType
from tarzan.models.investor_config import InvestorConfig
from tarzan.contracts.exceptions import DataIngestionError
from tarzan.contracts.targets import (
    TargetError,
    TargetSetOutcome,
    TargetSetStatus,
    ValidatedTargetRow,
)
from tarzan.contracts.validation import (
    check_order_sign,
    currency_is_known,
    isin_format_error,
    normalize_currency,
    normalize_isin,
)

logger = logging.getLogger(__name__)

# Data-quality report source tag for the order-list ingestion stage.
_DQ_SOURCE = "order_load"

# Minimal set the order list must carry for returns. Derived from the single
# declarative schema (tarzan.contracts.schema.ORDER_LIST_SCHEMA) so there is ONE source
# of truth for the format; kept as a module constant for back-compat.
from tarzan.contracts.schema import ORDER_LIST_SCHEMA as _ORDER_SCHEMA
ORDER_REQUIRED_COLUMNS = frozenset(_ORDER_SCHEMA.required_columns())


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


def load_orders(source: Union[str, io.BytesIO], filename: str = "",
                strict: bool = False) -> list[Order]:
    """Load and validate the order list into typed Order objects.

    Reuses ``_read_source`` (path/.csv/.xlsx/BytesIO), lowercases columns,
    validates them against the declared :data:`tarzan.contracts.schema.ORDER_LIST_SCHEMA`,
    and skips malformed or unknown-type rows with a warning rather than
    failing the whole load.

    A missing file path returns ``[]`` (not an exception) so the
    orchestrator can log and exit gracefully.

    Args:
        source: File path (str) or BytesIO buffer.
        filename: Original filename (needed for BytesIO extension detection).
        strict: When True, an unrecognized column is a hard error (the posture
            a multi-tenant product wants — reject a mis-formatted file with an
            actionable message). Default False keeps the historical lenient
            behavior (unknown columns tolerated with a warning), so the
            automated newsletter and existing files are unaffected.

    Returns:
        List of validated Order objects (possibly empty).
    """
    # Missing path → empty list, let the caller decide it's non-fatal.
    if isinstance(source, str) and not os.path.exists(source):
        logger.warning("Order list not found at %s; treating as no orders.", source)
        return []

    df = _read_source(source, filename)
    df.columns = [c.strip().lower() for c in df.columns]

    # Validate columns against the explicit, versioned schema. Missing required
    # columns are always fatal; unknown columns warn (lenient) or reject
    # (strict). Warnings go to the per-run data-quality report.
    from tarzan.contracts.schema import ORDER_LIST_SCHEMA
    from tarzan.contracts.validation import validate_columns
    fatal, col_warnings = validate_columns(df.columns, ORDER_LIST_SCHEMA, strict=strict)
    for w in col_warnings:
        logger.warning(w)
        dq.warning(_DQ_SOURCE, w)
    if fatal:
        raise DataIngestionError(fatal)

    orders: list[Order] = []
    skipped = 0
    for idx, row in df.iterrows():
        try:
            order = _parse_order_row(idx, row, strict=strict)
            if order is not None:
                orders.append(order)
            elif strict:
                raise DataIngestionError(
                    f"order row {int(idx) + 2} violates the executable type/value contract"
                )
            else:
                skipped += 1
        except DataIngestionError:
            raise
        except Exception as e:
            if strict:
                raise DataIngestionError(
                    f"order row {int(idx) + 2} violates the executable contract: {e}"
                ) from e
            logger.warning("Order row %d: unexpected error '%s', skipping", idx, e)
            skipped += 1

    if skipped:
        logger.info("Skipped %d order row(s) with invalid/unknown data", skipped)

    _assign_order_ids(orders)
    logger.info("Loaded %d orders from %s", len(orders), filename or str(source))
    return orders


def _assign_order_ids(orders: list[Order]) -> None:
    """Give every order a stable, unique ``order_id`` and surface duplicates.

    The order ledger is append-only and two identical movements on the same
    day CAN be legitimate, so we never drop rows. But an accidentally
    duplicated input block (a user pasting the same rows twice) would silently
    double-count every position and cash flow — so we detect repeated natural
    keys, disambiguate the id with an ordinal (``<key>#2`` …), and emit ONE
    data-quality WARNING per repeated key with its count, making the
    duplication visible for the user to confirm or fix.
    """
    seen: dict[str, int] = {}
    dup_counts: dict[str, int] = {}
    dup_example: dict[str, Order] = {}
    for o in orders:
        nk = o.natural_key()
        n = seen.get(nk, 0) + 1
        seen[nk] = n
        o.order_id = nk if n == 1 else f"{nk}#{n}"
        if n == 2:
            dup_example[nk] = o
        if n >= 2:
            dup_counts[nk] = n
    for nk, n in dup_counts.items():
        o = dup_example[nk]
        msg = (f"{n} identical order rows for {o.isin} on {o.trade_date} "
               f"({o.type.value}, qty {o.quantity:g}, net €{o.net_eur:,.2f}) — "
               "kept all (identical trades can be legitimate); verify this is "
               "not an accidentally duplicated input block, which would "
               "double-count the position and cash flow.")
        logger.warning("Order dedup: %s", msg)
        dq.warning(_DQ_SOURCE, msg, context=o.isin)


def _parse_instrument_kind(
    raw: object,
    *,
    idx: int,
    isin: str,
    strict: bool,
) -> Optional[InstrumentKind]:
    """Parse the optional exact mechanics kind without heuristic fallback."""
    value = _clean_str(raw).upper()
    if not value:
        return None
    try:
        return InstrumentKind(value)
    except ValueError:
        message = (
            f"row {idx} ({isin}): unsupported instrument_kind {value!r}; "
            "order-derived valuation remains unavailable unless an exact "
            "provider kind is resolved"
        )
        if strict:
            raise DataIngestionError(message)
        logger.warning("Order %s", message)
        dq.warning(_DQ_SOURCE, message, context=isin)
        return None


def _parse_order_row(
    idx: int,
    row: pd.Series,
    *,
    strict: bool = False,
) -> Optional[Order]:
    """Parse one order-list row into an Order, or None to skip it.

    Validation policy (see ``tarzan.contracts.validation``): structural failures
    (unknown type, invalid date, malformed/missing ISIN, unparseable required
    number) SKIP the row; a wrong-sign quantity is NORMALIZED (not dropped);
    an unknown currency is KEPT with an EUR fallback. Every skip/coercion is
    recorded in the per-run data-quality report so the user can review it.
    """
    otype = OrderType.from_raw(row.get("type"))
    if otype is None:
        msg = f"row {idx}: unknown order type {row.get('type')!r} — skipped"
        logger.warning("Order %s", msg)
        dq.warning(_DQ_SOURCE, msg, context=f"row {idx}")
        return None

    order_date = _parse_date_safe(row.get("date"))
    if order_date is None:
        msg = f"row {idx}: invalid/missing date {row.get('date')!r} — skipped"
        logger.warning("Order %s", msg)
        dq.warning(_DQ_SOURCE, msg, context=f"row {idx}")
        return None
    trade_date = _parse_date_safe(row.get("trade_date")) or order_date

    # ISIN — format-only validation (no mod-10; legit bond ISINs fail it).
    isin = normalize_isin(row.get("isin"))
    isin_err = isin_format_error(isin)
    if isin_err:
        msg = f"row {idx}: {isin_err} — skipped"
        logger.warning("Order %s", msg)
        dq.warning(_DQ_SOURCE, msg, context=f"row {idx}")
        return None

    quantity = _parse_number_safe(row.get("quantity"), "quantity", idx)
    gross_eur = _parse_number_safe(row.get("gross_eur"), "gross_eur", idx)
    net_eur = _parse_number_safe(row.get("net_eur"), "net_eur", idx)
    if quantity is None or gross_eur is None or net_eur is None:
        bad = [f for f, v in (("quantity", quantity), ("gross_eur", gross_eur),
                              ("net_eur", net_eur)) if v is None]
        msg = f"row {idx} ({isin}): unparseable/blank {', '.join(bad)} — skipped"
        logger.warning("Order %s", msg)
        dq.warning(_DQ_SOURCE, msg, context=isin)
        return None

    # Sign/type consistency — normalize a clear mismatch rather than drop it.
    sign = check_order_sign(otype, quantity)
    if sign.message:
        msg = f"row {idx} ({isin}): {sign.message}"
        logger.warning("Order %s", msg)
        dq.warning(_DQ_SOURCE, msg, context=isin)
    quantity = sign.quantity

    # Currency — keep an unknown code but fall back to EUR and flag it.
    currency = normalize_currency(row.get("currency"))
    if not currency_is_known(currency):
        msg = (f"row {idx} ({isin}): unrecognized currency {currency!r} — "
               "kept as-is (verify; foreign holdings need a valid code to "
               "value in EUR)")
        logger.warning("Order %s", msg)
        dq.warning(_DQ_SOURCE, msg, context=isin)

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
        instrument_kind=_parse_instrument_kind(
            row.get("instrument_kind"),
            idx=idx,
            isin=isin,
            strict=strict,
        ),
    )


def _clean_str(val) -> str:
    """Return a stripped string, mapping NaN/None to empty string."""
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def load_targets_per_holding(
    source: Union[str, io.BytesIO], filename: str = ""
) -> TargetSetOutcome:
    """Load a row-preserving, domain-validated target set.

    Blank optional cells remain ``None`` and notional percentages may exceed
    100. Every malformed/negative/non-finite value or duplicate canonical key
    invalidates the complete planning target set while retaining source rows.
    """
    if isinstance(source, str) and not os.path.exists(source):
        logger.info("No per-holding targets at %s; none applied.", source)
        return TargetSetOutcome.absent()

    df = _read_source(source, filename)
    df.columns = [c.strip().lower() for c in df.columns]
    errors: list[TargetError] = []
    rows: list[ValidatedTargetRow] = []
    if "isin" not in df.columns and "ticker" not in df.columns:
        error = TargetError(
            code="MISSING_TARGET_IDENTIFIER_COLUMN",
            message="target input requires an ISIN or ticker column",
            canonical_key="",
            source_rows=(),
        )
        dq.error("target_load", error.message, context=filename or "targets")
        return TargetSetOutcome(status=TargetSetStatus.INVALID, errors=(error,))

    from tarzan.models.instrument_key import instrument_key, normalize_isin

    def parse_target(raw, field_name: str, source_row: int, canonical_key: str):
        text = _clean_str(raw)
        if not text:
            return None
        try:
            value = float(_parse_number(raw))
        except (TypeError, ValueError):
            errors.append(TargetError(
                code="MALFORMED_TARGET_VALUE",
                message=f"{field_name} must be a finite nonnegative number",
                canonical_key=canonical_key,
                source_rows=(source_row,),
            ))
            return None
        if not math.isfinite(value) or value < 0:
            errors.append(TargetError(
                code="INVALID_TARGET_DOMAIN",
                message=f"{field_name} must be finite and nonnegative",
                canonical_key=canonical_key,
                source_rows=(source_row,),
            ))
            return None
        return value

    grouped: dict[str, list[ValidatedTargetRow]] = {}
    for index, row in df.iterrows():
        source_row = int(index) + 2
        raw_isin = _clean_str(row.get("isin"))
        ticker = _clean_str(row.get("ticker"))
        isin = normalize_isin(raw_isin)
        canonical_key = instrument_key(isin, ticker)
        if not canonical_key:
            errors.append(TargetError(
                code="MISSING_TARGET_IDENTIFIER",
                message="target row requires an ISIN or ticker",
                canonical_key="",
                source_rows=(source_row,),
            ))
            continue
        nbns = _clean_str(row.get("no_buy_no_sell")).lower()
        entry = {
            "isin": isin,
            "ticker": ticker,
            "name": _clean_str(row.get("name")),
            "target_equities": parse_target(row.get("target_equities"), "target_equities", source_row, canonical_key),
            "target_fixed_income": parse_target(row.get("target_fixed_income"), "target_fixed_income", source_row, canonical_key),
            "target_portfolio": parse_target(row.get("target_portfolio"), "target_portfolio", source_row, canonical_key),
            "no_buy_no_sell": nbns in ("true", "1", "yes"),
        }
        validated = ValidatedTargetRow(
            source_name=filename or "targets",
            source_row=source_row,
            canonical_key=canonical_key,
            value=entry,
        )
        rows.append(validated)
        grouped.setdefault(canonical_key, []).append(validated)

    for canonical_key, duplicates in grouped.items():
        if len(duplicates) > 1:
            source_rows = tuple(item.source_row for item in duplicates)
            errors.append(TargetError(
                code="DUPLICATE_TARGET_ROW",
                message=f"duplicate target {canonical_key} at source rows {source_rows}",
                canonical_key=canonical_key,
                source_rows=source_rows,
            ))

    if errors:
        for error in errors:
            dq.error(
                "target_load",
                f"{error.code}: {error.message}; planning suppressed for the target set.",
                context=error.canonical_key or filename or "targets",
            )
        return TargetSetOutcome(
            status=TargetSetStatus.INVALID,
            rows=tuple(rows),
            errors=tuple(errors),
        )

    result: dict[str, dict] = {}
    for validated in rows:
        entry = dict(validated.value)
        legacy_key = entry["isin"] or entry["ticker"].upper()
        result[legacy_key] = entry

    logger.info("Loaded per-holding targets for %d instrument(s)", len(rows))
    return TargetSetOutcome(
        status=TargetSetStatus.VALID,
        rows=tuple(rows),
        mapping=result,
    )


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