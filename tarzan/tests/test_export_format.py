"""Tests for the shared export formatting taxonomy (colors, ordering,
benchmark labels) — the single source of truth consumed by both the
Excel dashboard and the HTML newsletter."""

from __future__ import annotations

from tarzan.export import _format
from tarzan.export import excel as xl
from tarzan.export import newsletter as nl


class TestColorTaxonomySingleSource:
    def test_excel_uses_shared_asset_colors(self):
        # Excel binds its ASSET_COLORS to the shared taxonomy object.
        assert xl.ASSET_COLORS is _format.ASSET_CLASS_COLORS

    def test_newsletter_matches_excel_per_class(self):
        # For every asset class, Excel (bare hex) and the newsletter
        # (#-prefixed) must resolve to the SAME color — this is the
        # regression that previously diverged for Crypto/Alternative.
        for klass, bare in _format.ASSET_CLASS_COLORS.items():
            assert nl.ASSET_COLORS[klass] == _format.css(bare)

    def test_newsletter_matches_excel_per_region(self):
        for region, bare in _format.GEO_COLORS.items():
            assert nl.GEO_COLORS[region] == _format.css(bare)

    def test_crypto_and_alternative_distinct(self):
        # Sanity: the two classes that had drifted are defined and
        # Alternative is its own (slate) color, not a copy of Crypto.
        assert _format.ASSET_CLASS_COLORS["Crypto"] != _format.ASSET_CLASS_COLORS["Alternative"]


class TestCssHelper:
    def test_prefixes_bare_hex(self):
        assert _format.css("1D4ED8") == "#1D4ED8"

    def test_leaves_prefixed_untouched(self):
        assert _format.css("#1D4ED8") == "#1D4ED8"

    def test_none_falls_back(self):
        assert _format.css(None).startswith("#")


class TestAssetClassOrder:
    def test_base_order_returned_without_arg(self):
        order = _format.asset_class_order()
        assert order[0] == "Equities"
        assert set(order) == set(_format.ASSET_CLASS_COLORS)

    def test_unknown_class_is_appended_not_dropped(self):
        order = _format.asset_class_order(["Equities", "Private Equity"])
        assert "Private Equity" in order  # never silently dropped

    def test_newsletter_order_covers_all_classes(self):
        # Every defined asset class must appear in the newsletter order so
        # no holding's class is dropped from the email.
        assert set(_format.ASSET_CLASS_COLORS).issubset(set(nl.ASSET_CLASS_ORDER))


class TestRiskProfileBenchmarkLabels:
    """_build_risk_profile must honor the configured benchmark names,
    not hardcoded 'S&P 500' / 'MSCI ACWI' literals."""

    def _ctx(self, ab_name, geo_name):
        import pandas as pd
        from tarzan.models.portfolio import PortfolioMetrics
        from tarzan.models.investor_config import InvestorConfig

        m = PortfolioMetrics()
        m.performance_full = {"cagr": 5.0, "volatility": 12.0, "sharpe": 1.0,
                              "sortino": 1.2, "max_drawdown": -10.0,
                              "var_95": -1.0, "cvar_95": -1.5,
                              "alpha": 0.5, "beta": 0.9}
        m.benchmark_comparison = pd.DataFrame([
            {"benchmark": ab_name, "cagr": 4.0, "volatility": 15.0, "sharpe": 0.8,
             "sortino": 1.0, "max_drawdown": -20.0, "var_95": -1.5, "cvar_95": -2.0,
             "alpha": 0.0, "beta": 1.0},
            {"benchmark": geo_name, "cagr": 4.5, "volatility": 14.0, "sharpe": 0.9,
             "sortino": 1.1, "max_drawdown": -18.0, "var_95": -1.4, "cvar_95": -1.9,
             "alpha": 0.2, "beta": 0.95},
        ])
        return nl._NewsletterContext(
            metrics=m, config=InvestorConfig(),
            benchmark_alpha_beta=ab_name, benchmark_geo=geo_name,
        )

    def test_headers_use_configured_names(self):
        ctx = self._ctx("MSCI World", "FTSE All-World")
        profile = nl._build_risk_profile(ctx)
        assert profile["available"]
        assert profile["headers"] == ["Metric", "You", "MSCI World", "FTSE All-World"]

    def test_benchmark_values_resolved_by_configured_name(self):
        # The benchmark columns must be populated from the configured
        # rows, not blank (which is what happened when the lookup used a
        # hardcoded literal that didn't match the configured benchmark).
        ctx = self._ctx("MSCI World", "FTSE All-World")
        profile = nl._build_risk_profile(ctx)
        cagr_row = next(r for r in profile["rows"] if r["label"] == "CAGR")
        assert cagr_row["sp500"] != "—"
        assert cagr_row["acwi"] != "—"


class TestShortInstrumentName:
    def _s(self, name, **kw):
        return _format.short_instrument_name(name, **kw)

    def test_preserves_real_alnum_token(self):
        # "3M" must survive — it is the company name, not a share class.
        assert self._s("3M Company") == "3M Company"

    def test_strips_trailing_share_class(self):
        out = self._s("iShares Core MSCI World UCITS ETF 1C")
        assert "1C" not in out
        assert "MSCI World" in out

    def test_drops_fund_series_roman_keeps_issuer(self):
        out = self._s("Xtrackers II Global Govt Bond UCITS ETF 1C")
        assert out.startswith("Xtr.")
        assert " II " not in f" {out} "

    def test_fallback_when_all_boilerplate(self):
        # Stripping everything would blank the cell; keep the original.
        assert self._s("UCITS ETF Acc") == "UCITS ETF Acc"

    def test_empty_input(self):
        assert self._s("") == ""
        assert self._s(None) == ""

    def test_truncation_with_ellipsis(self):
        out = self._s("Some Extremely Long Instrument Name That Exceeds", max_len=20)
        assert len(out) <= 20
        assert out.endswith("…")

    def test_issuer_abbreviation(self):
        assert self._s("Invesco Physical Gold ETC").startswith("Inv.")
