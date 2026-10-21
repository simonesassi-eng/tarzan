"""Exact corporate-action declarations; price jumps are never event evidence."""

from dataclasses import dataclass
from datetime import date
import math
from types import MappingProxyType


@dataclass(frozen=True)
class DeclaredSplit:
    symbol: str
    isin: str
    effective_date: date
    units_after_per_unit_before: float
    announced_on: date
    event_source: str

    def __post_init__(self):
        if (
            not math.isfinite(self.units_after_per_unit_before)
            or self.units_after_per_unit_before <= 0
            or self.announced_on > self.effective_date
        ):
            raise ValueError("invalid declared split")


# The 300 factor was already curated in the original CL2 repair and independently
# checked against its Paris ruler. The exchange's public Milan announcement now
# corroborates the exact event, unchanged ISIN, and effective date; its PUBLIC
# abstract does not expose the numerical factor. Do not claim it does.
# No factor is inferred by the repair routine, and no bare-ticker alias matches.
CURATED_SPLITS = MappingProxyType({
    "CL2.MI": (
        DeclaredSplit(
            "CL2.MI", "FR0010755611", date(2023, 10, 10), 300.0, date(2023, 10, 4),
            "https://live.euronext.com/ajax/getNoticePublicAbstract/1891658",
        ),
    ),
})
