"""Tests for the shared export formatting taxonomy (colors, ordering,
benchmark labels) — the single source of truth (``_format``) the HTML
newsletter binds to."""

from __future__ import annotations

from tarzan.export import _format
from tarzan.export import newsletter as nl


class TestColorTaxonomySingleSource:
    def test_newsletter_matches_shared_format_per_class(self):
        # For every asset class, the newsletter (#-prefixed) must resolve to
        # the SAME color as the shared _format source (bare hex) — this is the
        # regression that previously diverged for Crypto/Alternative.
        for klass, bare in _format.ASSET_CLASS_COLORS.items():
            assert nl.ASSET_COLORS[klass] == _format.css(bare)

    def test_newsletter_matches_shared_format_per_region(self):
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
        from tarzan.models.portfolio import PortfolioMetrics
        from tarzan.models.investor_config import InvestorConfig

        def _metrics(cagr, vol, sh, so, mdd, ui, var, cvar, a, b):
            return {"cagr": cagr, "volatility": vol, "sharpe": sh, "sortino": so,
                    "max_drawdown": mdd, "ulcer_index": ui, "var_95": var,
                    "cvar_95": cvar, "alpha": a, "beta": b}

        m = PortfolioMetrics()
        # New contract: the Historical risk profile reads metrics.historical_risk
        # (per-instrument full history + a current-weight portfolio backtest).
        m.historical_risk = {
            "available": True,
            "portfolio": {
                "label": "Your portfolio", "ticker": None, "span_label": "3.0Y",
                "note": None, "is_portfolio": True,
                "metrics": _metrics(5.0, 12.0, 1.0, 1.2, -10.0, 6.0, -1.0, -1.5, 0.5, 0.9),
            },
            "instruments": [
                {"label": ab_name, "ticker": "SWDA.MI", "span_label": "10.0Y",
                 "note": None, "is_portfolio": False,
                 "metrics": _metrics(4.0, 15.0, 0.8, 1.0, -20.0, 9.0, -1.5, -2.0, 0.0, 1.0)},
                {"label": geo_name, "ticker": "VWCE.MI", "span_label": "9.0Y",
                 "note": None, "is_portfolio": False,
                 "metrics": _metrics(4.5, 14.0, 0.9, 1.1, -18.0, 8.0, -1.4, -1.9, 0.2, 0.95)},
            ],
        }
        return nl._NewsletterContext(
            metrics=m, config=InvestorConfig(),
            benchmark_alpha_beta=ab_name, benchmark_geo=geo_name,
        )

    def test_benchmarks_become_rows_with_configured_names(self):
        ctx = self._ctx("MSCI World", "FTSE All-World")
        profile = nl._build_risk_profile(ctx)
        assert profile["available"]
        # Transposed layout: metrics are columns, series are rows.
        labels = [r["label"] for r in profile["rows"]]
        assert labels[0] == "Your portfolio"
        assert profile["rows"][0]["is_portfolio"] is True
        assert "MSCI World" in labels and "FTSE All-World" in labels
        # 10 metric columns now (Ulcer added), with α/β carrying the marker.
        col_labels = [c["label"] for c in profile["columns"]]
        assert col_labels[0] == "CAGR" and len(profile["columns"]) == 10
        assert "Ulcer" in col_labels
        assert profile["columns"][-2]["note"] == "*"  # α
        assert profile["columns"][-1]["note"] == "*"  # β
        assert "MSCI World" in profile["alpha_beta_note"]
        # Ticker pins are shortened (exchange suffix stripped).
        ab_row = next(r for r in profile["rows"] if r["label"] == "MSCI World")
        assert ab_row["ticker"] == "SWDA"
        assert ab_row["span_label"] == "10.0Y"

    def test_benchmark_values_populated_in_rows(self):
        # Each instrument row must carry real values (not blank), proving
        # the metrics are read straight from historical_risk.
        ctx = self._ctx("MSCI World", "FTSE All-World")
        profile = nl._build_risk_profile(ctx)
        bench_rows = [r for r in profile["rows"] if not r["is_portfolio"]]
        assert bench_rows
        for r in bench_rows:
            assert len(r["cells"]) == 10
            assert r["cells"][0] != "—"  # CAGR populated


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
