"""Prepared immutable display references; no lookup belongs in a section builder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from tarzan.models.instrument_key import normalize_isin, normalize_ticker
from tarzan.contracts.immutable import freeze_json


@dataclass(frozen=True)
class ReportReference:
    names: Mapping = field(default_factory=dict)
    taxonomy: Mapping = field(default_factory=dict)
    benchmarks: Mapping = field(default_factory=dict)
    benchmark_identities: tuple = ()
    watchlist_names: tuple[str, ...] = ()
    metric_ratings: Mapping = field(default_factory=dict)
    benchmark_alpha_beta: str = ""
    benchmark_geo: str = ""

    def __post_init__(self):
        for name in ("names", "taxonomy", "benchmarks", "metric_ratings"):
            object.__setattr__(self, name, freeze_json(getattr(self, name)))
        object.__setattr__(
            self, "benchmark_identities",
            tuple(tuple(item) for item in self.benchmark_identities),
        )
        object.__setattr__(
            self, "watchlist_names",
            tuple(sorted({str(name).strip().lower() for name in self.watchlist_names})),
        )

    @classmethod
    def prepare(cls):
        """Read the reference authorities once at the analysis boundary."""
        from tarzan import config

        return cls(
            names=config.name_lookup(),
            taxonomy=config.instrument_taxonomy(),
            benchmarks=config.benchmarks(),
            benchmark_identities=config.benchmark_identities(),
            watchlist_names=tuple(config.watchlist_names()),
            metric_ratings=config.metric_ratings(),
            benchmark_alpha_beta=config.benchmark_beta_name(),
            benchmark_geo=config.benchmark_geo_allocation(),
        )

    def name_for(self, isin, ticker):
        for key in (normalize_isin(isin), normalize_ticker(ticker)):
            if key and key in self.names:
                return self.names[key]
        return None


def reference_for(metrics) -> ReportReference:
    """Compatibility boundary for a legacy cube, not a section-level resolver."""
    reference = getattr(metrics, "report_reference", None)
    return reference if reference is not None else ReportReference.prepare()
