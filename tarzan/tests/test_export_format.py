"""Tests for the shared export formatting taxonomy (colors, ordering,
benchmark labels) — the single source of truth (``_format``) the HTML
newsletter binds to."""

from __future__ import annotations

import re

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


class TestColorizePct:
    """_colorize_pct colours signed % AND signed pp (divergence note uses pp)."""

    def _spans(self, text):
        import re
        from tarzan.export.newsletter import _colorize_pct
        return re.findall(r'color:(#[0-9A-Fa-f]+)[^>]*>([^<]+)</span>',
                          _colorize_pct(text))

    def test_colors_percent_and_pp(self):
        spans = self._spans("gap -4.53pp, picks +0.92pp, day +0.5% vs -0.8%")
        toks = [t for _, t in spans]
        assert any("pp" in t for t in toks)   # pp figures coloured
        assert any("%" in t for t in toks)    # % figures coloured
        # sign drives colour (PALETTE holds "#RRGGBB" directly)
        from tarzan.export.newsletter import PALETTE
        colors = {t.strip(): c for c, t in spans}
        assert colors.get("-4.53pp") == PALETTE["red"]
        assert colors.get("+0.92pp") == PALETTE["green"]

    def test_colors_spelled_out_percentage_points(self):
        # The AI sometimes writes "percentage points" instead of "pp".
        spans = self._spans("costing -4.53 percentage points of the gap")
        assert any("percentage point" in t for _, t in spans)

    def test_bare_beta_not_coloured(self):
        # A bare number like a beta "0.69" must stay neutral (no unit).
        assert self._spans("beta of 0.69") == []


class TestColorizePctLines:
    """_colorize_pct_lines turns the news digest's newline-separated items
    (each optionally starting with a '[tag]' time marker) into <tr><td>
    rows - a <table>, not a native <ul>, renders consistently across email
    clients (Outlook in particular)."""

    def _lines(self, text):
        from tarzan.export.newsletter import _colorize_pct_lines
        return _colorize_pct_lines(text)

    def test_one_row_per_line(self):
        html = self._lines("[14:30] First headline.\n[Yesterday] Second +0.5%.")
        assert html.count("<tr>") == 2 and html.count("</tr>") == 2

    def test_tag_extracted_and_styled_separately_from_headline(self):
        html = self._lines("[14:30] Markets rallied.")
        assert "14:30" in html
        assert html.index("14:30") < html.index("Markets rallied")

    def test_line_without_tag_still_renders(self):
        html = self._lines("No tag on this one.")
        assert "No tag on this one." in html
        assert html.count("<tr>") == 1

    def test_last_row_has_no_border(self):
        html = self._lines("First.\nSecond.\nThird.")
        rows = html.split("<tr>")[1:]
        assert "border-bottom" in rows[0] and "border-bottom" in rows[1]
        assert "border-bottom" not in rows[2]

    def test_empty_lines_dropped(self):
        html = self._lines("First.\n\nSecond.\n")
        assert html.count("<tr>") == 2

    def test_percentage_still_coloured(self):
        from tarzan.export.newsletter import PALETTE
        html = self._lines("[09:00] Oil slipped -1.2% on supply data.")
        assert PALETTE["red"] in html

    def test_empty_input_returns_empty_string(self):
        assert self._lines("") == ""
        assert self._lines(None) == ""


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

    def test_portfolio_metrics_become_tiles(self):
        """The section is the portfolio's own ten metrics, as tiles.

        It used to be a 36-row table -- the portfolio plus every reference
        instrument, over eleven columns -- which answered a comparison nobody
        asked for at a quarter of the issue's height. The reference benchmarks
        are no longer rendered here; ``alpha_beta_note`` still names the index
        alpha and beta are measured against, because that one is not a
        comparison but the definition of two of the figures.
        """
        ctx = self._ctx("MSCI World", "FTSE All-World")
        profile = nl._build_risk_profile(ctx)
        assert profile["available"]
        # Greek letters are wrapped so a CSS uppercase transform cannot fold
        # them onto capitals drawn like Latin A and B, so strip the markup here.
        labels = [re.sub(r"<[^>]+>", "", t["label"]) for t in profile["tiles"]]
        assert labels == ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD",
                          "Ulcer", "VaR 95%", "CVaR 95%",
                          "\u03b1*", "\u03b2*"], labels
        alpha = next(t["label"] for t in profile["tiles"]
                     if "\u03b1" in t["label"])
        assert "text-transform:none" in alpha
        assert "MSCI World" in profile["alpha_beta_note"]

    def test_tiles_carry_the_portfolio_values(self):
        ctx = self._ctx("MSCI World", "FTSE All-World")
        tiles = {t["label"]: t["value"] for t in nl._build_risk_profile(ctx)["tiles"]}
        # The fixture's portfolio row, not a benchmark's.
        assert tiles["CAGR"] == "5.00%"
        assert tiles["Sharpe"] == "1.00"
        assert tiles["Sortino"] == "1.20"

    def test_rated_metrics_get_a_gauge_and_beta_does_not(self):
        """A gauge is drawn wherever constants.yaml rates the metric, so the
        scale shown is the configured one rather than one invented here."""
        ctx = self._ctx("MSCI World", "FTSE All-World")
        tiles = {t["label"]: t["gauge"] for t in nl._build_risk_profile(ctx)["tiles"]}
        for label in ("Volatility", "Sharpe", "Sortino", "Max DD", "CAGR"):
            assert "<svg" in tiles[label], label
        # Lower-is-better metrics put the strong zone on the left, so the end
        # captions swap with it.
        assert tiles["Volatility"].index("strong") < tiles["Volatility"].index("weak")
        assert tiles["Sharpe"].index("weak") < tiles["Sharpe"].index("strong")


class TestShortInstrumentName:
    def _s(self, name, **kw):
        return _format.short_instrument_name(name, **kw)

    def test_preserves_real_alnum_token(self):
        # "3M" must survive — it is the company name, not a share class.
        assert self._s("3M Company") == "3M Company"

    def test_strips_trailing_share_class(self):
        out = self._s("iShares Core MSCI World UCITS ETF 1C")
        assert "1C" not in out
        # "World" is abbreviated to "Wrld" (see _WORD_ABBREVIATIONS); the point
        # of this test is that the trailing share class is gone and the index
        # survives, now in its abbreviated form.
        assert out == "iSh. Core MSCI Wrld"

    def test_abbreviates_common_words(self):
        assert self._s("Xtrackers MSCI World Momentum") == "Xtr. MSCI Wrld Mom."
        assert self._s("L&G Market Neutral Commodities") == "L&G Mkt Neutral Comm."

    def test_keeps_hedged_abbreviated_not_stripped(self):
        # "Hedged" is a real distinction, kept as "Hdgd" rather than dropped.
        assert self._s("Xtrackers S&P 500 Swap EUR Hedged") == \
            "Xtr. S&P 500 Swap EUR Hdgd"

    def test_drops_trailing_ticker_echo(self):
        assert self._s("Amundi Euro Government Bond 25+ (LMTH)") == \
            "Amu. Euro Govt Bond 25+"

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


class TestSignedZeroIsNotSigned:
    """A value that rounds to zero must not print a sign.

    A residual weight of -0.004% used to render as "−0.0%", which reads as a
    real move in the negative direction rather than as nothing. The optimizer's
    After column showed it on the real order list: positions being sold to zero
    printed "−0.0%" next to siblings printing "0.0%".
    """

    def test_pct_drops_the_sign_when_it_rounds_to_zero(self):
        from tarzan.export.newsletter._format import _pct

        assert _pct(-0.001, 1, True) == "0.0%"
        assert _pct(0.0, 1, True) == "0.0%"

    def test_pct_keeps_the_sign_on_a_real_value(self):
        from tarzan.export.newsletter._format import _pct

        assert _pct(-1.2, 1, True) == "\u22121.2%"
        assert _pct(2.5, 2, True) == "+2.50%"

    def test_a_value_below_the_printed_precision_is_still_signed_if_it_shows(self):
        """-0.004 rounds away at 1 decimal but survives at 3, so the sign must
        come back rather than being dropped on magnitude alone."""
        from tarzan.export.newsletter._format import _pct

        assert _pct(-0.004, 3, True) == "\u22120.004%"

    def test_compact_and_smart_and_pp_agree(self):
        from tarzan.export.newsletter._format import (
            _pct_compact, _pct_smart, _signed_pp)

        assert _pct_compact(-0.0004) == "0.00%"
        assert _pct_compact(-1.23) == "\u22121.23%"
        assert _pct_smart(-0.004, 1, True) == "0%"
        assert _pct_smart(-1.6, 1, True) == "\u22121.6%"
        assert _signed_pp(-0.02) == "0.0"
        assert _signed_pp(-0.9) == "\u22120.9"
