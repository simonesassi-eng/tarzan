"""Input-boundary validators for the order list.

Pure, side-effect-free checks used by the loader to catch malformed rows at
ingestion and produce *actionable* diagnostics instead of silently feeding
corrupt data into the valuation/returns math. Each validator returns a small
result object; the loader decides whether to skip, keep, or normalize based on
it, and records the reason in the per-run data-quality report.

Policy (per the reviewed design):
  * ISIN — **format only** (12 chars, 2-letter country prefix, alphanumeric
    body). The ISO 6166 mod-10 check digit is intentionally NOT enforced:
    several legitimately-held bond ISINs (US Treasuries, BTPs) fail it as the
    data vendor sees them, and rejecting real positions is worse than the
    disease. A structural format failure is a skip.
  * Currency — ISO-4217 membership, but a non-member is a **warn-and-keep**
    (falls back to EUR at the loader), because dropping a real buy over a
    currency typo would lose a position.
  * Order sign/type — the quantity sign must match the movement direction.
    A clear mismatch (e.g. a SELL with positive quantity) is **normalized**
    to the correct sign and flagged, rather than skipped, so a data-entry
    slip does not silently corrupt position tracking or drop the trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tarzan.models.order import OrderType

# ---------------------------------------------------------------------------
# ISIN — format only (no mod-10 check digit; see module docstring)
# ---------------------------------------------------------------------------

_ISIN_LEN = 12


def normalize_isin(raw: Optional[str]) -> str:
    """Uppercase and strip an ISIN (also removing internal spaces/hyphens a
    hand-edited cell might carry). Returns '' for None/blank/'nan'."""
    if raw is None:
        return ""
    s = str(raw).strip().upper().replace(" ", "").replace("-", "")
    return "" if s in ("", "NAN") else s


def isin_format_error(isin: str) -> Optional[str]:
    """Return a human-readable reason the ISIN is malformed, or None if it is
    structurally valid. Format only: <2-letter country><9 alphanumeric><digit>,
    12 chars total, without enforcing the mod-10 check digit."""
    if not isin:
        return "missing ISIN"
    if len(isin) != _ISIN_LEN:
        return f"ISIN '{isin}' has {len(isin)} chars (expected {_ISIN_LEN})"
    if not (isin[0:2].isalpha()):
        return f"ISIN '{isin}' does not start with a 2-letter country code"
    if not isin.isalnum():
        return f"ISIN '{isin}' has non-alphanumeric characters"
    if not isin[-1].isdigit():
        return f"ISIN '{isin}' does not end with a check digit"
    return None


# ---------------------------------------------------------------------------
# Currency — ISO-4217 membership (warn-and-keep on miss)
# ---------------------------------------------------------------------------

# The ISO-4217 active codes a retail multi-asset portfolio can plausibly hold.
# Not exhaustive of every world currency, but broad enough that a code outside
# it is almost certainly a typo worth flagging (still kept, defaulted to EUR).
_ISO_4217 = frozenset({
    "EUR", "USD", "GBP", "GBX", "GBp", "CHF", "JPY", "CAD", "AUD", "NZD",
    "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "RON", "BGN", "HRK", "ISK",
    "TRY", "RUB", "UAH", "ILS", "ILa", "AED", "SAR", "QAR", "KWD", "ZAR",
    "ZAc", "EGP", "NGN", "KES", "MAD", "HKD", "SGD", "CNY", "CNH", "TWD",
    "KRW", "INR", "IDR", "THB", "MYR", "PHP", "VND", "BRL", "MXN", "ARS",
    "CLP", "COP", "PEN", "MOP",
})


def normalize_currency(raw: Optional[str]) -> str:
    """Uppercase/strip a currency code; '' → 'EUR'. Minor-unit codes (GBp,
    ZAc, ILa) are preserved as-is (case-sensitive) because the enricher keys
    its minor-unit rescale on them."""
    if raw is None:
        return "EUR"
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return "EUR"
    # Preserve known minor-unit codes verbatim (mixed case is significant).
    if s in ("GBp", "GBX", "ZAc", "ZAC", "ILa", "ILA"):
        return s
    return s.upper()


def currency_is_known(currency: str) -> bool:
    """True if the (already-normalized) currency is a recognized ISO-4217
    code or minor-unit variant."""
    return currency in _ISO_4217


# ---------------------------------------------------------------------------
# Order sign / type consistency
# ---------------------------------------------------------------------------

_POSITIVE_QTY_TYPES = frozenset({OrderType.BUY, OrderType.TRANSFER_IN})
_NEGATIVE_QTY_TYPES = frozenset({OrderType.SELL, OrderType.TRANSFER_OUT})
_ZERO_QTY_TYPES = frozenset({OrderType.COUPON, OrderType.DIVIDEND})


@dataclass
class SignCheck:
    """Result of checking a quantity's sign against its movement type.

    ``quantity`` is the (possibly sign-corrected) value the loader should use;
    ``message`` is non-None when something was off and worth surfacing.
    """

    quantity: float
    message: Optional[str] = None


def check_order_sign(otype: OrderType, quantity: float) -> SignCheck:
    """Reconcile a quantity's sign with its movement direction.

    BUY/TRANSFER_IN must be > 0, SELL/TRANSFER_OUT must be < 0, and
    COUPON/DIVIDEND carry no position change (0). A clear mismatch is
    normalized to the correct sign (not dropped) and reported, so a
    data-entry slip does not silently corrupt position tracking — while a
    genuinely zero position-changing quantity is left as-is for the caller's
    own handling.
    """
    if otype in _ZERO_QTY_TYPES:
        # A distribution should not move a position; a non-zero quantity is a
        # mild anomaly but harmless (position change gate ignores these types).
        return SignCheck(quantity=quantity)

    if quantity == 0.0:
        # A position-changing order with zero quantity moves nothing — leave
        # it; the loader's downstream logic treats it as a no-op.
        return SignCheck(quantity=quantity)

    if otype in _POSITIVE_QTY_TYPES and quantity < 0:
        return SignCheck(
            quantity=abs(quantity),
            message=(f"{otype.value} has a negative quantity ({quantity:g}); "
                     "sign corrected to positive"),
        )
    if otype in _NEGATIVE_QTY_TYPES and quantity > 0:
        return SignCheck(
            quantity=-abs(quantity),
            message=(f"{otype.value} has a positive quantity ({quantity:g}); "
                     "sign corrected to negative"),
        )
    return SignCheck(quantity=quantity)


# ---------------------------------------------------------------------------
# Column-level schema validation (file boundary)
# ---------------------------------------------------------------------------

def validate_columns(present_columns, schema, *, strict: bool = False):
    """Validate a file's columns against its declared ``FileSchema``.

    Returns ``(fatal_error_or_None, warnings_list)``:

      * A **missing required column** is always a fatal error (the file cannot
        be used) — returned as a message string; the caller raises.
      * An **unknown column** (not a canonical name or accepted alias) is a
        WARNING in the default *lenient* mode (tolerated, e.g. Fineco adds an
        extra column), but a fatal error in ``strict`` mode — the posture a
        multi-tenant product wants, so a user's mis-formatted file is rejected
        with an actionable message rather than silently half-read.

    Pure: it only inspects column names; it never mutates and never raises.
    The loader decides what to do with the result (raise vs record+continue).
    """
    present = {str(c).strip().lower() for c in present_columns}
    known = schema.known_columns()

    missing = sorted(schema.required_columns() - present)
    fatal = None
    if missing:
        fatal = (f"{schema.file}: missing required column(s): "
                 f"{', '.join(missing)}. Expected columns — see the "
                 f"{schema.file} schema (v{schema.version}).")

    warnings = []
    unknown = sorted(present - known)
    if unknown:
        msg = (f"{schema.file}: unrecognized column(s): {', '.join(unknown)} "
               "(ignored)." if not strict else
               f"{schema.file}: unrecognized column(s): {', '.join(unknown)} "
               "— rejected in strict mode.")
        if strict:
            # In strict mode an unknown column is fatal; surface it as the
            # error (missing-required still takes precedence if both apply).
            fatal = fatal or msg
        else:
            warnings.append(msg)

    return fatal, warnings
