"""Unit tests for enricher FX helpers (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from tarzan.data.enricher import _as_fraction, _normalize_minor_currency


class TestYieldFractionNormalization:
    """yield_pct/ter are stored as FRACTIONS; the metrics layer ×100 them.
    yfinance mixes fraction (yield=0.021) and percent (dividendYield=2.4)
    conventions, so _as_fraction must normalize the percent ones."""

    def test_fraction_field_unchanged(self):
        assert _as_fraction(0.021) == pytest.approx(0.021)

    def test_percent_field_divided_by_100(self):
        # dividendYield=2.4 (percent) must become 0.024 so ×100 gives 2.4%.
        assert _as_fraction(2.4) == pytest.approx(0.024)

    def test_small_ter_fraction_unchanged(self):
        assert _as_fraction(0.0007) == pytest.approx(0.0007)

    def test_none_and_nonpositive_and_nan_map_to_none(self):
        assert _as_fraction(None) is None
        assert _as_fraction(0) is None
        assert _as_fraction(-1.0) is None
        assert _as_fraction(float("nan")) is None

    def test_weighted_yield_no_longer_100x_inflated(self):
        # 50/50 book: one fraction-field holding, one percent-field holding.
        weights = [50.0, 50.0]
        yields = [_as_fraction(0.021), _as_fraction(2.4)]
        total_w = sum(weights)
        wy = sum((y or 0.0) * w for y, w in zip(yields, weights)) / total_w * 100
        # ~ (2.1% + 2.4%)/2 = 2.25%, not 121%.
        assert wy == pytest.approx(2.25, abs=0.01)


class TestNormalizeMinorCurrency:
    def test_gbp_pence_rescaled_to_gbp(self):
        # 28450 GBp ≡ 284.50 GBP
        prices = pd.Series([28450.0, 28580.0, 28700.0])
        rescaled, currency = _normalize_minor_currency(prices, "GBp")
        assert currency == "GBP"
        assert rescaled.iloc[0] == 284.50
        assert rescaled.iloc[1] == 285.80
        assert rescaled.iloc[2] == 287.00

    def test_gbx_alternate_code_rescaled(self):
        prices = pd.Series([100.0])
        rescaled, currency = _normalize_minor_currency(prices, "GBX")
        assert currency == "GBP"
        assert rescaled.iloc[0] == 1.0

    def test_zac_rescaled_to_zar(self):
        prices = pd.Series([5000.0])
        rescaled, currency = _normalize_minor_currency(prices, "ZAc")
        assert currency == "ZAR"
        assert rescaled.iloc[0] == 50.0

    def test_ila_rescaled_to_ils(self):
        prices = pd.Series([200.0])
        rescaled, currency = _normalize_minor_currency(prices, "ILa")
        assert currency == "ILS"
        assert rescaled.iloc[0] == 2.0

    def test_major_currencies_unchanged(self):
        prices = pd.Series([100.0, 101.0])
        for cur in ("USD", "EUR", "GBP", "JPY", "CHF"):
            rescaled, currency = _normalize_minor_currency(prices, cur)
            assert currency == cur
            assert rescaled.iloc[0] == 100.0
            assert rescaled.iloc[1] == 101.0

    def test_unknown_currency_passthrough(self):
        prices = pd.Series([42.0])
        rescaled, currency = _normalize_minor_currency(prices, "XYZ")
        assert currency == "XYZ"
        assert rescaled.iloc[0] == 42.0


class TestConvertToEurFxFailure:
    """A total FX failure must NOT value a non-EUR holding 1:1 as EUR."""

    def test_no_fx_returns_empty_not_native_prices(self, monkeypatch):
        import tarzan.data.enricher as enricher
        # Simulate total FX failure (both pairs throttled, no cache).
        monkeypatch.setattr(enricher, "_get_fx_series",
                            lambda ccy: pd.Series(dtype=float))
        prices = pd.Series(
            [500.0, 510.0],
            index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
        )
        out = enricher.convert_to_eur(prices, "USD")
        # Empty → caller drops the (unconvertible) price and falls back to the
        # last-known EUR anchor, rather than booking $500 as €500.
        assert out.empty

    def test_eur_passthrough_unaffected(self):
        import tarzan.data.enricher as enricher
        prices = pd.Series([100.0, 101.0])
        out = enricher.convert_to_eur(prices, "EUR")
        assert list(out) == [100.0, 101.0]

    def test_idempotent_on_major_after_rescale(self):
        prices = pd.Series([28450.0])
        # First call: GBp → GBP, divide by 100
        p1, c1 = _normalize_minor_currency(prices, "GBp")
        # Second call on already-major currency: no change
        p2, c2 = _normalize_minor_currency(p1, c1)
        assert c2 == "GBP"
        assert p2.iloc[0] == 284.50


# ---------------------------------------------------------------------------
# Deterministic ISIN → ticker resolution
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402  (already imported above; harmless re-import)

from tarzan.data import enricher  # noqa: E402
from tarzan.data.enricher import (  # noqa: E402
    _Candidate,
    _name_match_score,
    _name_tokens,
    _rank_key,
    _suffix_priority,
)


def _cand(symbol, *, price=10.0, currency="EUR", name=""):
    """Build a candidate carrying only the info needed for ranking."""
    return _Candidate(
        symbol=symbol,
        info={"currency": currency},
        price=price,
        currency=currency,
        name=name,
    )


class TestNameMatching:
    def test_stopwords_dropped(self):
        toks = _name_tokens("Xtrackers MSCI World ex USA UCITS ETF 1C EUR Acc")
        assert "msci" in toks and "world" in toks
        assert "ucits" not in toks and "etf" not in toks and "1c" not in toks

    def test_matching_name_scores_high(self):
        # Same instrument family → strong overlap with canonical name.
        score = _name_match_score(
            "Xtrackers MSCI World ex USA UCITS ETF", "X MSCI WORLD EX USA 1C"
        )
        assert score >= 0.75

    def test_collision_name_scores_zero(self):
        # Unrelated fund sharing the bare symbol → no token overlap.
        score = _name_match_score(
            "Nomura Focused International", "X MSCI WORLD EX USA 1C"
        )
        assert score == 0.0

    def test_unknown_canonical_is_neutral(self):
        assert _name_match_score("Anything At All", "") == 0.5


class TestSuffixPriority:
    def test_config_order_respected(self):
        suffixes = enricher.ISIN_EXCHANGE_SUFFIXES
        # Earlier suffix in config ranks strictly before a later one.
        assert _suffix_priority(f"ABC{suffixes[0]}") < _suffix_priority(f"ABC{suffixes[-1]}")

    def test_unknown_suffix_ranks_last(self):
        assert _suffix_priority("ABC.XYZ") == len(enricher.ISIN_EXCHANGE_SUFFIXES)


class TestRankKey:
    def test_name_match_beats_currency_match(self):
        # The right instrument (name match) must win even if a collision
        # candidate happens to match the expected currency.
        right = _cand("EXUS.MI", currency="EUR", name="Xtrackers MSCI World ex USA UCITS ETF")
        wrong = _cand("EXUS", currency="EUR", name="Nomura Focused International")
        canon = "X MSCI WORLD EX USA 1C"
        assert _rank_key(right, canon, "EUR") > _rank_key(wrong, canon, "EUR")

    def test_currency_breaks_tie_when_names_equal(self):
        a = _cand("ABC.F", currency="USD", name="Same Fund")
        b = _cand("ABC.L", currency="EUR", name="Same Fund")
        # Expected EUR → the EUR listing ranks higher.
        assert _rank_key(b, "Same Fund", "EUR") > _rank_key(a, "Same Fund", "EUR")


class TestResolveIsinDeterminism:
    """_resolve_isin must be a pure function of the candidate set:
    identical inputs → identical winner, regardless of probe order."""

    def _patch(self, monkeypatch, candidates_by_symbol, canonical_name, openfigi_syms):
        monkeypatch.setattr(enricher, "_openfigi_name", lambda isin: canonical_name)
        monkeypatch.setattr(enricher, "_openfigi_lookup", lambda isin: list(openfigi_syms))

        def fake_fetch(symbol):
            return candidates_by_symbol.get(symbol)

        monkeypatch.setattr(enricher, "_fetch_candidate_meta", fake_fetch)
        # History is fetched only for the winner; stub it out (no network).
        monkeypatch.setattr(enricher, "_fetch_history", lambda symbol: pd.DataFrame())

    def test_collision_rejected_by_name(self, monkeypatch):
        isin = "IE0006WW1TQ4"
        cands = {
            "EXUS": _cand("EXUS", price=26.77, currency="USD",
                          name="Nomura Focused International"),
            "EXUS.MI": _cand("EXUS.MI", price=38.05, currency="EUR",
                             name="Xtrackers MSCI World ex USA UCITS ETF"),
        }
        # OpenFIGI lists the bare colliding symbol first (the bug trigger).
        self._patch(monkeypatch, cands, "X MSCI WORLD EX USA 1C", ["EXUS"])
        result = enricher._resolve_isin(isin, hint_ticker="EXUS.MI", expected_currency="EUR")
        assert result is not None
        _, symbol = result
        assert symbol == "EXUS.MI"

    def test_idempotent_across_calls(self, monkeypatch):
        isin = "IE0006WW1TQ4"
        cands = {
            "EXUS": _cand("EXUS", price=26.77, currency="USD",
                          name="Nomura Focused International"),
            "EXUS.MI": _cand("EXUS.MI", price=38.05, currency="EUR",
                             name="Xtrackers MSCI World ex USA UCITS ETF"),
        }
        self._patch(monkeypatch, cands, "X MSCI WORLD EX USA 1C", ["EXUS"])
        results = {
            enricher._resolve_isin(isin, hint_ticker="EXUS.MI", expected_currency="EUR")[1]
            for _ in range(5)
        }
        assert results == {"EXUS.MI"}

    def test_same_isin_different_hints_same_winner(self, monkeypatch):
        """Holdings path (hint EXUS.MI) and order path (hint = bare ISIN)
        must resolve to the same symbol."""
        isin = "IE0006WW1TQ4"
        cands = {
            "EXUS": _cand("EXUS", price=26.77, currency="USD",
                          name="Nomura Focused International"),
            "EXUS.MI": _cand("EXUS.MI", price=38.05, currency="EUR",
                             name="Xtrackers MSCI World ex USA UCITS ETF"),
        }
        self._patch(monkeypatch, cands, "X MSCI WORLD EX USA 1C", ["EXUS"])
        holdings_win = enricher._resolve_isin(isin, hint_ticker="EXUS.MI", expected_currency="EUR")[1]
        order_win = enricher._resolve_isin(isin, hint_ticker=isin, expected_currency="EUR")[1]
        assert holdings_win == order_win == "EXUS.MI"

    def test_returns_none_when_nothing_priced(self, monkeypatch):
        self._patch(monkeypatch, {}, "", [])
        assert enricher._resolve_isin("XX0000000000") is None

    def test_prefers_candidate_with_usable_history(self, monkeypatch):
        """A higher-ranked quote-only listing (no price history) must lose to a
        lower-ranked listing that actually serves a series — otherwise the
        returns build is forced onto the carry-flat fallback. Regression for
        IE00BKFB6L02 resolving to a dead .SG line instead of UEQC.DE."""
        isin = "IE00BKFB6L02"
        # .SG ranks higher on suffix priority (index 1) than .DE (index 3), but
        # only .DE serves a history.
        cands = {
            f"{isin}.SG": _cand(f"{isin}.SG", price=127.0, currency="EUR",
                                 name="UBS CMCI Commodity Carry"),
            "UEQC.DE": _cand("UEQC.DE", price=127.0, currency="EUR",
                             name="UBS CMCI Commodity Carry"),
        }
        monkeypatch.setattr(enricher, "_openfigi_name", lambda i: "UBS CMCI Commodity Carry")
        monkeypatch.setattr(enricher, "_openfigi_lookup", lambda i: ["UEQC.DE"])
        monkeypatch.setattr(enricher, "_fetch_candidate_meta",
                            lambda s: cands.get(s))
        # Only UEQC.DE has a real series; the .SG line quotes but has no history.
        hist = pd.DataFrame({"Close": [1.0, 1.1]},
                            index=pd.to_datetime(["2026-07-14", "2026-07-15"]))
        monkeypatch.setattr(enricher, "_fetch_history",
                            lambda s: hist if s == "UEQC.DE" else pd.DataFrame())
        monkeypatch.setattr(enricher.price_cache, "store_resolution",
                            lambda *a, **k: None)
        result = enricher._resolve_isin(isin, hint_ticker=isin, expected_currency="EUR")
        assert result is not None
        _, symbol = result
        assert symbol == "UEQC.DE"


class TestTaxonomyBidirectionalMatch:
    """_apply_taxonomy_override must match a taxonomy row keyed by one of
    (ISIN, ticker) when the holding knows only the other, via the learned
    ISIN↔ticker xref. Regression for the Fineco ISIN-only UEQC order missing
    the ticker-keyed UEQC taxonomy row."""

    def test_isin_only_holding_matches_ticker_keyed_row(self, monkeypatch):
        from tarzan.models.holding import Holding

        # Taxonomy row keyed by ticker only (blank ISIN), like the UEQC row.
        monkeypatch.setattr(
            enricher.cfg, "instrument_taxonomy",
            lambda: {"UEQC": ("Commodities", "Carry")})
        # xref knows UEQC.DE ↔ IE00BKFB6L02 (learned during resolution).
        monkeypatch.setattr(enricher.price_cache, "load_ticker_isin_reverse",
                            lambda isin: "UEQC.DE" if isin == "IE00BKFB6L02" else None)
        monkeypatch.setattr(enricher.price_cache, "load_ticker_isin",
                            lambda t: None)

        # Holding known ONLY by ISIN (no ticker) — the Fineco import case.
        h = Holding(isin="IE00BKFB6L02", ticker="", quantity=40.0,
                    cost_basis_eur=5090.0, market_value_eur=4982.0, currency="EUR",
                    name="UBS CMCI USD-A-AC")
        assert enricher._apply_taxonomy_override(h) is True
        assert h.asset_class.value == "Commodities"
        assert h.role == "Carry"


# ---------------------------------------------------------------------------
# Network layer — retry/backoff, currency matching, per-run memoization
# ---------------------------------------------------------------------------


class TestTransientClassification:
    def test_429_is_transient(self):
        assert enricher._is_transient_error(Exception("HTTP Error 429: Too Many Requests"))

    def test_timeout_is_transient(self):
        assert enricher._is_transient_error(Exception("connection timed out"))

    def test_not_found_is_not_transient(self):
        assert not enricher._is_transient_error(Exception("404 Not Found"))


class TestRetry:
    def test_returns_on_first_success(self, monkeypatch):
        calls = []
        monkeypatch.setattr(enricher._time, "sleep", lambda s: None)
        out = enricher._retry(lambda: calls.append(1) or "ok", what="x")
        assert out == "ok" and len(calls) == 1

    def test_retries_then_succeeds_on_transient(self, monkeypatch):
        monkeypatch.setattr(enricher._time, "sleep", lambda s: None)
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] < 2:
                raise Exception("HTTP 429 rate limit")
            return "recovered"

        assert enricher._retry(flaky, what="x") == "recovered"
        assert state["n"] == 2

    def test_no_retry_on_definitive_error(self, monkeypatch):
        monkeypatch.setattr(enricher._time, "sleep", lambda s: None)
        state = {"n": 0}

        def notfound():
            state["n"] += 1
            raise Exception("404 Not Found")

        assert enricher._retry(notfound, what="x") is None
        assert state["n"] == 1  # not retried

    def test_gives_up_after_max_attempts(self, monkeypatch):
        from tarzan.data import _yf_net
        monkeypatch.setattr(_yf_net._time, "sleep", lambda s: None)
        state = {"n": 0}

        def always_throttled():
            state["n"] += 1
            raise Exception("timeout")

        assert enricher._retry(always_throttled, what="x") is None
        assert state["n"] == _yf_net._MAX_FETCH_ATTEMPTS


class TestCurrencyMatches:
    def test_exact_match(self):
        assert enricher._currency_matches("EUR", "EUR")

    def test_minor_unit_matches_major(self):
        # yfinance "GBp" must match a declared "GBP" holding currency.
        assert enricher._currency_matches("GBp", "GBP")
        assert enricher._currency_matches("ZAc", "ZAR")

    def test_mismatch(self):
        assert not enricher._currency_matches("USD", "EUR")

    def test_empty_is_false(self):
        assert not enricher._currency_matches("", "EUR")
        assert not enricher._currency_matches("EUR", "")


class TestOpenFigiMemoization:
    def test_single_network_call_per_isin(self, monkeypatch):
        enricher.reset_run_caches()
        monkeypatch.setattr(enricher._time, "sleep", lambda s: None)
        calls = {"n": 0}

        def fake_urlopen(req, timeout=10):
            calls["n"] += 1
            raise Exception("404 Not Found")  # definitive → no retry

        monkeypatch.setattr(enricher, "urlopen", fake_urlopen)
        # Three logical lookups for the same ISIN…
        enricher._openfigi_raw("IE0006WW1TQ4")
        enricher._openfigi_raw("IE0006WW1TQ4")
        enricher._openfigi_raw("IE0006WW1TQ4")
        # …collapse to a single network call thanks to per-run memoization.
        assert calls["n"] == 1

    def test_reset_clears_memo(self, monkeypatch):
        enricher.reset_run_caches()
        monkeypatch.setattr(enricher._time, "sleep", lambda s: None)
        calls = {"n": 0}

        def fake_urlopen(req, timeout=10):
            calls["n"] += 1
            raise Exception("404 Not Found")

        monkeypatch.setattr(enricher, "urlopen", fake_urlopen)
        enricher._openfigi_raw("IE0006WW1TQ4")
        enricher.reset_run_caches()
        enricher._openfigi_raw("IE0006WW1TQ4")
        assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Geo-breakdown memo — run-scoped & thread-safe
# ---------------------------------------------------------------------------

from tarzan.models.holding import Geography  # noqa: E402


class TestGeoBreakdownMemo:
    def test_scrape_once_then_memoized(self, monkeypatch):
        enricher.reset_run_caches()
        calls = {"n": 0}
        breakdown = {Geography.USA: 100.0}

        def fake_scrape(ticker, isin=""):
            calls["n"] += 1
            return breakdown, "justetf"

        monkeypatch.setattr(enricher, "_scrape_geo_breakdown", fake_scrape)
        first = enricher.get_geo_breakdown("XDWD.MI", "IE000")
        second = enricher.get_geo_breakdown("XDWD.MI", "IE000")
        assert first == second == (breakdown, "justetf")
        assert calls["n"] == 1  # second call served from memo

    def test_reset_forces_rescrape(self, monkeypatch):
        enricher.reset_run_caches()
        calls = {"n": 0}

        def fake_scrape(ticker, isin=""):
            calls["n"] += 1
            return {Geography.USA: 100.0}, "justetf"

        monkeypatch.setattr(enricher, "_scrape_geo_breakdown", fake_scrape)
        enricher.get_geo_breakdown("XDWD.MI")
        enricher.reset_run_caches()
        enricher.get_geo_breakdown("XDWD.MI")
        assert calls["n"] == 2  # memo cleared between runs → fresh scrape

    def test_classify_geography_uses_memoized_breakdown(self, monkeypatch):
        enricher.reset_run_caches()
        # Seed the memo via the public API, then classify should pick the
        # dominant region without re-scraping.
        monkeypatch.setattr(
            enricher, "_scrape_geo_breakdown",
            lambda ticker, isin="": ({Geography.JAPAN: 80.0, Geography.USA: 20.0}, "justetf"),
        )
        enricher.get_geo_breakdown("XMJP.MI")
        from tarzan.models.holding import Holding
        h = Holding(isin="IE000", ticker="XMJP.MI", quantity=1.0, cost_basis_eur=0.0,
                    market_value_eur=0.0, currency="EUR")
        geo = enricher.classify_geography({"quoteType": "ETF"}, "XMJP.MI", h)
        assert geo == Geography.JAPAN


class TestBacktestPeriod:
    def test_set_and_read_roundtrip(self):
        original = enricher._backtest_period()
        try:
            enricher.set_portfolio_backtest_period("3y")
            assert enricher._backtest_period() == "3y"
        finally:
            enricher.set_portfolio_backtest_period(original)


class TestYfCallSpacing:
    def test_spacing_enforced_between_calls(self, monkeypatch):
        from tarzan.data import _yf_net
        enricher.reset_run_caches()
        slept = []
        # Freeze monotonic so the gate always sees zero elapsed time and
        # therefore must wait the full interval on the second call.
        monkeypatch.setattr(_yf_net._time, "monotonic", lambda: 0.0)
        monkeypatch.setattr(_yf_net._time, "sleep", lambda s: slept.append(s))
        enricher._space_yf_call()  # first call: last=0 set, may or may not wait
        slept.clear()
        enricher._space_yf_call()  # second call: 0 elapsed → must wait full interval
        assert slept and slept[0] == _yf_net._YF_MIN_INTERVAL

    def test_reset_clears_yf_timestamp(self):
        from tarzan.data import _yf_net
        _yf_net._last_call[0] = 123.0
        enricher.reset_run_caches()
        assert _yf_net._last_call[0] == 0.0


from tarzan.instruments.registry import InstrumentKind  # noqa: E402


class TestBondFxConversion:
    """The Borsa Italiana bond fallback must convert the native clean
    price to EUR for ANY currency (USD Treasury, ZAR note, GBP gilt),
    not just EUR bonds — the regression that inflated a ZAR note 19x."""

    def _holding(self, isin, qty, currency, market_value):
        from tarzan.models.holding import Holding
        return Holding(isin=isin, ticker=isin, quantity=qty, cost_basis_eur=0.0,
                       market_value_eur=market_value, currency=currency)

    def test_eur_bond_unchanged(self, monkeypatch):
        import tarzan.data.bond_fetcher as bf
        monkeypatch.setattr(bf, "fetch_bond_price",
                            lambda isin: {"price": 103.84, "source": "borsa_italiana/mot/btp"})
        h = self._holding("IT0005542359", qty=4000.0, currency="EUR", market_value=4150.0)
        enricher._try_terrapin_fallback(h, InstrumentKind.BOND)
        # 4000 * 103.84 / 100 = 4153.60, current_price EUR-per-unit = 1.0384
        assert h.current_value == pytest.approx(4153.60)
        assert h.current_price == pytest.approx(1.0384)
        assert h.data_source.startswith("borsa_italiana")

    def test_zar_bond_converted_to_eur(self, monkeypatch):
        import tarzan.data.bond_fetcher as bf
        monkeypatch.setattr(bf, "fetch_bond_price",
                            lambda isin: {"price": 98.14, "source": "borsa_italiana/mot/btp"})
        # FX: EUR per 1 ZAR ≈ 1/19.2 ≈ 0.05208
        monkeypatch.setattr(enricher, "_get_fx_series",
                            lambda ccy: pd.Series([1.0 / 19.2]))
        h = self._holding("XS2105803527", qty=110000.0, currency="ZAR", market_value=5624.0)
        enricher._try_terrapin_fallback(h, InstrumentKind.BOND)
        # 110000 * (98.14/19.2) / 100 ≈ 5623 EUR — NOT 110000*98.14/100 = 107954
        assert h.current_value == pytest.approx(110000 * (98.14 / 19.2) / 100.0, rel=1e-6)
        assert 4000 < h.current_value < 8000
        # current_price is EUR-per-unit
        assert h.current_price == pytest.approx((98.14 / 19.2) / 100.0, rel=1e-6)

    def test_usd_treasury_converted_to_eur(self, monkeypatch):
        import tarzan.data.bond_fetcher as bf
        monkeypatch.setattr(bf, "fetch_bond_price",
                            lambda isin: {"price": 84.25, "source": "borsa_italiana/mot/btp"})
        # FX: EUR per 1 USD ≈ 0.92
        monkeypatch.setattr(enricher, "_get_fx_series",
                            lambda ccy: pd.Series([0.92]))
        h = self._holding("US91282CGJ45", qty=2800.0, currency="USD", market_value=2170.0)
        enricher._try_terrapin_fallback(h, InstrumentKind.BOND)
        # 2800 * (84.25*0.92) / 100 ≈ 2170 EUR
        assert h.current_value == pytest.approx(2800 * (84.25 * 0.92) / 100.0, rel=1e-6)


class TestFixedIncomeEtfCurrentValuation:
    """Exposure labels cannot select bond per-100 valuation mechanics."""

    def test_fixed_income_etf_uses_unit_pricing(self, monkeypatch):
        from tarzan.models.holding import AssetClass, Holding

        history = pd.DataFrame(
            {"Close": [105.25]},
            index=pd.to_datetime(["2025-06-29"]),
        )
        monkeypatch.setattr(
            enricher,
            "_fetch_ticker_data",
            lambda ticker: {
                "info": {
                    "category": "Fixed Income",
                    "currency": "EUR",
                    "longName": "Bond Strategy Fixed Income ETF",
                },
                "history": history,
            },
        )
        holding = Holding(
            isin="ETF-FI-TEST",
            ticker="FIETF",
            quantity=10.0,
            cost_basis_eur=1000.0,
            market_value_eur=999.0,
            currency="EUR",
            security_type=InstrumentKind.ETF.value,
            asset_class=AssetClass.FIXED_INCOME,
        )

        enriched, _ = enricher._enrich_single(holding)

        assert enriched.current_price == pytest.approx(105.25)
        assert enriched.current_value == pytest.approx(1052.50)
        assert enriched.security_type == InstrumentKind.ETF.value


class TestGeoBreakdownDiskCache:
    """get_geo_breakdown must reuse the on-disk geo cache so a justETF
    outage (scrape returns None) does not degrade geography."""

    def test_falls_back_to_disk_cache_when_scrape_fails(self, tmp_path, monkeypatch):
        from tarzan.data import enricher, price_cache
        from tarzan.models.holding import Geography

        # Enable the cache against a throwaway dir (conftest disables it).
        monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
        enricher.reset_run_caches()

        # Seed the disk cache for an ISIN, then force the live scrape to fail.
        price_cache.store_geo(
            "IE00BL25JP72", {"USA": 79.0, "Japan": 4.0}, "justetf"
        )
        monkeypatch.setattr(enricher, "_scrape_geo_breakdown", lambda *a, **k: None)

        result = enricher.get_geo_breakdown("XDEM.MI", "IE00BL25JP72")
        assert result is not None
        breakdown, source = result
        assert breakdown.get(Geography.USA) == pytest.approx(79.0)
        assert source == "justetf"

    def test_returns_none_when_no_cache_and_scrape_fails(self, tmp_path, monkeypatch):
        from tarzan.data import enricher
        monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
        enricher.reset_run_caches()
        monkeypatch.setattr(enricher, "_scrape_geo_breakdown", lambda *a, **k: None)
        assert enricher.get_geo_breakdown("ZZZ.MI", "ZZ0000000000") is None


class TestFetchHistoryProvenance:
    def _patch_history_sources(self, monkeypatch, cached, fresh):
        enricher._history_memo.clear()
        monkeypatch.setattr(enricher.price_cache, "load_history", lambda symbol: cached)
        monkeypatch.setattr(enricher, "_retry", lambda fn, *, what: fresh)
        monkeypatch.setattr(enricher.price_cache, "store_history", lambda *args: None)

    def test_failed_live_tail_keeps_selected_cache_row_provenance(
        self, monkeypatch
    ):
        cached_day = pd.Timestamp.now().normalize() - pd.Timedelta(days=10)
        fresh_day = pd.Timestamp.now().normalize()
        cached = pd.DataFrame({"Close": [100.0]}, index=[cached_day])
        fresh = pd.DataFrame({"Close": [float("nan")]}, index=[fresh_day])
        self._patch_history_sources(monkeypatch, cached, fresh)

        history = enricher._fetch_history("CACHE-STALE")
        holding = TestSetPriceDataRobustness()._holding()
        enricher._set_price_data(holding, history, {}, "EUR")

        assert holding.current_price == 100.0
        assert holding.price_observation_timestamp == cached_day.to_pydatetime()
        assert holding.price_is_fallback is True
        assert holding.data_source == "price_cache:CACHE-STALE"

    def test_valid_fresh_tail_remains_primary(self, monkeypatch):
        cached_day = pd.Timestamp.now().normalize() - pd.Timedelta(days=2)
        fresh_day = pd.Timestamp.now().normalize()
        cached = pd.DataFrame({"Close": [100.0]}, index=[cached_day])
        fresh = pd.DataFrame({"Close": [101.0]}, index=[fresh_day])
        self._patch_history_sources(monkeypatch, cached, fresh)

        history = enricher._fetch_history("LIVE-FRESH")
        holding = TestSetPriceDataRobustness()._holding()
        enricher._set_price_data(holding, history, {}, "EUR")

        assert holding.current_price == 101.0
        assert holding.price_observation_timestamp == fresh_day.to_pydatetime()
        assert holding.price_is_fallback is False
        assert holding.data_source is None

    def test_timestamped_live_quote_supersedes_older_cache(
        self, monkeypatch
    ):
        from tarzan import runtime

        monkeypatch.setattr(runtime, "as_of", lambda: None)
        cached_day = pd.Timestamp.now().normalize() - pd.Timedelta(days=10)
        quote_time = pd.Timestamp.now(tz="UTC").floor("s")
        history = pd.DataFrame({"Close": [100.0]}, index=[cached_day])
        history.attrs[enricher._HISTORY_ORIGINS_ATTR] = {
            enricher._history_timestamp_key(cached_day): (
                enricher._HISTORY_ORIGIN_CACHE
            )
        }
        history.attrs[enricher._HISTORY_SYMBOL_ATTR] = "CACHE-WITH-LIVE"
        holding = TestSetPriceDataRobustness()._holding()

        enricher._set_price_data(
            holding,
            history,
            {
                "regularMarketPrice": 105.0,
                "regularMarketTime": int(quote_time.timestamp()),
            },
            "EUR",
        )

        assert holding.current_price == 105.0
        assert holding.price_observation_timestamp == quote_time.to_pydatetime()
        assert holding.price_is_fallback is False
        assert holding.data_source is None

    def test_pinned_as_of_ignores_current_quote(self, monkeypatch):
        import datetime

        from tarzan import runtime

        monkeypatch.setattr(runtime, "as_of", lambda: datetime.date(2025, 1, 2))
        history_day = pd.Timestamp("2025-01-02")
        history = pd.DataFrame({"Close": [100.0]}, index=[history_day])
        holding = TestSetPriceDataRobustness()._holding()

        enricher._set_price_data(
            holding,
            history,
            {
                "regularMarketPrice": 105.0,
                "regularMarketTime": int(pd.Timestamp("2025-01-03", tz="UTC").timestamp()),
            },
            "EUR",
        )

        assert holding.current_price == 100.0
        assert holding.price_observation_timestamp == history_day.to_pydatetime()


class TestSetPriceDataRobustness:
    """_set_price_data must never propagate a NaN/non-positive close as a
    price. A throttled live quote (yfinance returning NaN) previously slipped
    past the ``is not None`` guard and collapsed the portfolio total."""

    def _holding(self):
        from tarzan.models.holding import Holding
        return Holding(isin="IE00B4L5Y983", ticker="SWDA.MI", quantity=10.0,
                       cost_basis_eur=1000.0, market_value_eur=1200.0, currency="EUR")

    def test_uses_last_valid_close(self):
        h = self._holding()
        hist = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
        enricher._set_price_data(h, hist, {}, "EUR")
        assert h.current_price == 102.0

    def test_trailing_nan_close_ignored(self):
        # Last row is NaN (throttled today) → use the last *valid* close.
        h = self._holding()
        hist = pd.DataFrame({"Close": [100.0, 101.0, float("nan")]})
        enricher._set_price_data(h, hist, {}, "EUR")
        assert h.current_price == 101.0

    def test_all_nan_history_uses_undated_info_only_as_fallback(self):
        h = self._holding()
        hist = pd.DataFrame({"Close": [float("nan"), float("nan")]})
        enricher._set_price_data(
            h,
            hist,
            {
                "regularMarketPrice": 99.5,
                "regularMarketTime": float("nan"),
            },
            "EUR",
        )
        assert h.current_price == 99.5
        assert h.price_observation_timestamp is None
        assert h.price_is_fallback is True
        assert h.data_source == "yfinance:regularMarketPrice (undated)"

    def test_previous_close_is_explicit_fallback(self):
        h = self._holding()
        enricher._set_price_data(
            h,
            pd.DataFrame(),
            {"previousClose": 98.5},
            "EUR",
        )
        assert h.current_price == 98.5
        assert h.price_is_fallback is True
        assert h.data_source == "yfinance:previousClose"

    def test_cached_fx_degrades_timestamped_live_quote(self, monkeypatch):
        quote_time = pd.Timestamp.now(tz="UTC").floor("s")
        fx_time = quote_time - pd.Timedelta(days=10)
        fx = pd.Series(
            [0.9, float("nan")],
            index=pd.DatetimeIndex([fx_time, quote_time]),
        )
        fx.attrs[enricher._HISTORY_ORIGINS_ATTR] = {
            enricher._history_timestamp_key(fx_time): (
                enricher._HISTORY_ORIGIN_CACHE
            ),
            enricher._history_timestamp_key(quote_time): (
                enricher._HISTORY_ORIGIN_PRIMARY
            ),
        }
        monkeypatch.setattr(enricher, "_get_fx_series", lambda currency: fx)
        h = self._holding()

        enricher._set_price_data(
            h,
            pd.DataFrame(),
            {
                "regularMarketPrice": 100.0,
                "regularMarketTime": int(quote_time.timestamp()),
            },
            "USD",
        )

        assert h.current_price == pytest.approx(90.0)
        assert h.price_observation_timestamp == fx_time.to_pydatetime()
        assert h.price_is_fallback is True
        assert h.data_source == "price_cache:FX"

    def test_previous_close_preserves_cached_fx_source(self, monkeypatch):
        fx_time = pd.Timestamp.now(tz="UTC").floor("s") - pd.Timedelta(days=2)
        fx = pd.Series([0.9], index=pd.DatetimeIndex([fx_time]))
        fx.attrs[enricher._HISTORY_ORIGINS_ATTR] = {
            enricher._history_timestamp_key(fx_time): (
                enricher._HISTORY_ORIGIN_CACHE
            )
        }
        monkeypatch.setattr(enricher, "_get_fx_series", lambda currency: fx)
        h = self._holding()

        enricher._set_price_data(
            h,
            pd.DataFrame(),
            {"previousClose": 100.0},
            "USD",
        )

        assert h.current_price == pytest.approx(90.0)
        assert h.price_is_fallback is True
        assert h.data_source == "price_cache:FX"

    def test_undated_quote_with_fresh_fx_keeps_quote_provenance(
        self, monkeypatch
    ):
        fx_time = pd.Timestamp.now(tz="UTC").floor("s")
        fx = pd.Series([0.9], index=pd.DatetimeIndex([fx_time]))
        fx.attrs[enricher._HISTORY_ORIGINS_ATTR] = {
            enricher._history_timestamp_key(fx_time): (
                enricher._HISTORY_ORIGIN_PRIMARY
            )
        }
        monkeypatch.setattr(enricher, "_get_fx_series", lambda currency: fx)
        h = self._holding()

        enricher._set_price_data(
            h,
            pd.DataFrame(),
            {"regularMarketPrice": 100.0},
            "USD",
        )

        assert h.current_price == pytest.approx(90.0)
        assert h.price_observation_timestamp is None
        assert h.price_is_fallback is True
        assert h.data_source == "yfinance:regularMarketPrice (undated)"

    def test_no_price_anywhere_leaves_none(self):
        # All-NaN history and no usable info quote → current_price stays None
        # so the caller's seed/last-known fallback can take over.
        h = self._holding()
        hist = pd.DataFrame({"Close": [float("nan")]})
        enricher._set_price_data(h, hist, {"regularMarketPrice": float("nan")}, "EUR")
        assert h.current_price is None

    def test_nonpositive_close_ignored(self):
        h = self._holding()
        hist = pd.DataFrame({"Close": [100.0, 0.0, -3.0]})
        enricher._set_price_data(h, hist, {}, "EUR")
        assert h.current_price == 100.0


class TestEnrichSingleSafetyNet:
    """_enrich_single must never leave a holding with a None/NaN
    current_value; it seeds the last-known EUR anchor so the portfolio
    total cannot collapse when the live price is missing."""

    def test_seeds_market_value_when_no_price(self, monkeypatch):
        from tarzan.models.holding import Holding
        h = Holding(isin="IE00B4L5Y983", ticker="SWDA.MI", quantity=10.0,
                    cost_basis_eur=1000.0, market_value_eur=1234.0, currency="EUR")

        # Simulate a fully throttled fetch: no resolution, empty history/info,
        # and no bond quote either (kept fully offline).
        import tarzan.data.bond_fetcher as bf
        monkeypatch.setattr(enricher, "_resolve_isin", lambda *a, **k: None)
        monkeypatch.setattr(enricher, "_fetch_ticker_data",
                            lambda t: {"info": {}, "history": pd.DataFrame()})
        monkeypatch.setattr(bf, "fetch_bond_price", lambda isin: None)
        # No price anywhere → fallback lands on the CSV anchor.
        out, _ = enricher._enrich_single(h)
        assert out.current_value is not None
        assert out.current_value == out.current_value  # not NaN
        assert out.current_value == pytest.approx(1234.0)


class TestExactKindEvidenceConflicts:
    def test_provider_cannot_override_conflicting_order_kinds(self, monkeypatch):
        from tarzan.models.holding import Holding

        history = pd.DataFrame(
            {"Close": [105.25]},
            index=pd.to_datetime(["2025-06-29"]),
        )
        monkeypatch.setattr(
            enricher,
            "_fetch_ticker_data",
            lambda ticker: {
                "info": {"quoteType": "ETF", "currency": "EUR"},
                "history": history,
            },
        )
        holding = Holding(
            isin="KIND-CONFLICT",
            ticker="KIND-CONFLICT",
            quantity=10.0,
            cost_basis_eur=1000.0,
            market_value_eur=999.0,
            currency="EUR",
            instrument_kind_evidence=("BOND", "ETF"),
        )

        enriched, _ = enricher._enrich_single(holding)

        assert enriched.security_type is None
        assert enriched.current_value == pytest.approx(999.0)
        assert enriched.data_source == "last-known (instrument kind unavailable)"

    def test_openfigi_etf_kind_does_not_fabricate_equity_category(self):
        classified = enricher._classify_figi_item(
            sec_type="etf",
            market_sector="equity",
            name="Fixed Income ETF",
            kw={},
            figi_fi=[],
            figi_eq=[],
            figi_etf=["etf"],
        )

        assert classified["instrument_type"] == "ETF"
        assert "asset_class" not in classified
