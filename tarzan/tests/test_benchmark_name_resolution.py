"""The benchmark display names come from the curated taxonomy, not literals.

``build_context`` DEFAULTED to the strings "S&P 500" / "MSCI ACWI" and resolved
nothing — the lookup lived only in ``render_newsletter``. The charts find their
benchmark series in ``metrics.benchmark_histories`` BY that name, and the real
geo benchmark on this book is "iShares MSCI ACWI", so a caller reaching
build_context directly matched no key and silently lost the benchmark line from
every panel that draws it. One resolver now, and no literal index name as a
default anywhere.
"""

from __future__ import annotations

from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics


class TestTheBenchmarkIsNamedByTheTaxonomy:
    """The charts look their benchmark series up in ``benchmark_histories`` BY
    the display name, so a hardcoded name is not a cosmetic default — it matches
    no key and silently removes the line from all three panels.
    """

    def test_build_context_resolves_the_geo_benchmark_from_config(self):
        from tarzan import config as cfg
        from tarzan.export.newsletter import _benchmark_names

        _ab, geo = _benchmark_names(None, None)

        assert geo == cfg.benchmark_geo_allocation()
        assert geo in cfg.chart_benchmarks() or geo == cfg._default_benchmark_name()

    def test_an_explicit_name_still_wins(self):
        from tarzan.export.newsletter import _benchmark_names
        assert _benchmark_names("Beta One", "Geo One") == ("Beta One", "Geo One")

    def test_the_context_default_is_not_a_literal_index_name(self):
        """A context built directly must draw NO benchmark rather than look up a
        name the taxonomy never produced."""
        from tarzan.export.newsletter._constants import _NewsletterContext
        ctx = _NewsletterContext(metrics=PortfolioMetrics(),
                                 config=InvestorConfig())
        assert ctx.benchmark_geo is None
        assert ctx.benchmark_alpha_beta is None
