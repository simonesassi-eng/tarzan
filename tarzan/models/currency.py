"""Minor currency units, and the one place their rescale lives.

Some venues quote in a MINOR unit worth a hundredth of the major one: an
LSE-listed instrument priced "653.3082 GBp" costs £6.533082, not £653. There is no
FX pair for a minor code, so a price must be rescaled to the major unit before any
conversion — and the enricher's own comment on the subject says what happens
otherwise: "If we skip this step, current_value explodes by 100x."

It skipped it. Three modules each knew the code list separately — the enricher's
rescale, the contract validator's allowlist, and the stress generator's cash leg —
and a fourth place needed it and had none: ``returns_builder._seed_market_value``
valued a 79-unit GBp position at €60,013 against a €619 cost basis, a +9,593% gain
that was 73% of the portfolio total. Its own order carried ``currency="GBp"``.

So the fact lives here, in the models layer below both data and engine, and every
reader imports it rather than restating it.

Convention: a minor-unit code is the two-letter currency followed by a lowercase
letter (GBp, ZAc, ILa). Case is significant, and the uppercase variants some
providers emit are listed alongside.
"""

from __future__ import annotations

from typing import Optional

#: Minor-unit code -> the major ISO code it is a hundredth of.
MINOR_TO_MAJOR: dict[str, str] = {
    "GBp": "GBP",   # British pence
    "GBX": "GBP",   # British pence (alternate code, uppercase)
    "ZAc": "ZAR",   # South African cents
    "ZAC": "ZAR",
    "ILa": "ILS",   # Israeli agorot
    "ILA": "ILS",
    "ZWL": "ZWL",   # edge case: not a minor unit, kept so lookups do not rescale it
}

#: The codes that genuinely need a /100. ``ZWL`` maps to itself, so excluding it by
#: comparing against the mapping is what keeps it from being divided.
MINOR_UNITS: frozenset = frozenset(
    code for code, major in MINOR_TO_MAJOR.items() if code != major
)


def major_code(currency: str) -> str:
    """The major ISO code for ``currency`` — itself when it is not a minor unit."""
    return MINOR_TO_MAJOR.get(currency, currency)


def is_minor(currency: Optional[str]) -> bool:
    """Whether ``currency`` is a minor unit needing a /100 rescale."""
    return str(currency or "") in MINOR_UNITS


def to_major(price: float, currency: Optional[str]) -> float:
    """``price`` expressed in the MAJOR unit of ``currency``.

    A no-op for an ordinary code, so a caller can apply it unconditionally rather
    than deciding whether this particular instrument needs it — which is the
    decision every one of the divergent copies got to make separately.
    """
    return price / 100.0 if is_minor(currency) else price
