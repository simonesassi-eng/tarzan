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


class TestVenueCurrency:
    """A EUR-quoting venue is EUR by definition, so a flaky ``info.currency``
    cannot trigger an FX conversion of an already-EUR listing — the bug that
    corrupted MSCI ACWI (ISAC.MI) to +1.7% against a real +0.08%."""

    def test_eur_venues_are_eur(self):
        from tarzan.data.enricher import venue_currency
        for sym in ("ISAC.MI", "NTSG.DE", "XESC.PA", "IS39.MU", "AGGH.MI"):
            assert venue_currency(sym) == "EUR", sym

    def test_ambiguous_or_foreign_venues_defer_to_provider(self):
        from tarzan.data.enricher import venue_currency
        for sym in ("ISAC.L", "NTSX", "ISAC.SW", ""):
            assert venue_currency(sym) is None, sym

    def test_benchmark_eur_venue_not_converted_despite_bad_info_currency(self, monkeypatch):
        """The exact ISAC.MI failure: Yahoo returns info.currency='USD' for a
        Milan (EUR) line under load. The benchmark series must stay EUR, not be
        FX-converted (which corrupted it to +1.7%)."""
        import tarzan.data.enricher as enricher
        from tarzan.engine import benchmarks

        idx = pd.to_datetime(["2026-07-22", "2026-08-20"])
        raw = pd.DataFrame({"Close": [106.02, 106.10]}, index=idx)
        enricher._benchmark_memo.clear()
        monkeypatch.setattr(
            enricher, "_fetch_ticker_data",
            lambda ticker, expected_name="": {
                "info": {"currency": "USD"},          # flaky/wrong under load
                "history": raw,
                enricher._TICKER_SYMBOL_KEY: "ISAC.MI",
            })

        def _boom(*a, **k):
            raise AssertionError("EUR venue must not be FX-converted")

        monkeypatch.setattr(enricher, "convert_to_eur", _boom)
        series = benchmarks._fetch_benchmark_history("ISAC").dropna()
        # Untouched EUR closes → the real +0.08% 1M, not a converted figure.
        assert round((float(series.iloc[-1]) / float(series.iloc[0]) - 1) * 100, 2) == 0.08


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

    def test_curated_beats_name_match(self):
        # A candidate independently verifiable against the curated taxonomy
        # must win even when a different, non-curated candidate's yfinance
        # name overlaps the OpenFIGI canonical name far better. Regression
        # for FR0010755611 (2026-07-29/30 runs): 18MF.MU (bare "18MF" is not
        # in the taxonomy) was outranking CL2.MI (bare "CL2" is curated)
        # purely on name-token overlap.
        curated = _cand("CL2.MI", currency="EUR",
                        name="Amundi MSCI USA (2x) Leveraged")
        better_name = _cand("18MF.MU", currency="EUR",
                            name="Amundi MSCI USA Dly(2x) Lev.UEA")
        canon = "AMUNDI MSCI USA DAILY 2X LEVERAGED UCITS ETF"
        assert _rank_key(curated, canon, "EUR") > _rank_key(better_name, canon, "EUR")


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

    def test_fr0010755611_resolves_to_curated_venue_not_18mf(self, monkeypatch):
        """Full regression for the real 2026-07-29/30 production failure.

        FR0010755611 (Amundi MSCI USA 2x Leveraged, taxonomy ticker CL2) is
        an ISIN-only Fineco order (hint_ticker=""). The alias bridge finds
        CL2.PA as the curated match. 18MF.MU has a live quote AND a name
        that mirrors the OpenFIGI canonical name closely -- it used to win
        outright on name-token overlap. CL2.MI has no live quote in
        yfinance's `info` (the real-world gap for this venue) but carries
        a full 1276-close daily history. The resolver must still land on
        CL2.MI: curated ranking beats the name match, and the
        curated-symbol history fallback in _fetch_candidate_meta keeps
        CL2.MI from being discarded for lacking a live quote.
        """
        isin = "FR0010755611"
        canonical = "AMUNDI MSCI USA DAILY 2X LEVERAGED UCITS ETF"
        info_by_symbol = {
            "18MF.MU": {
                "regularMarketPrice": 30.48, "currency": "EUR",
                "longName": "Amundi MSCI USA Dly(2x) Lev.UEA",
            },
            "CL2.MI": {"currency": "EUR",
                       "longName": "Amundi MSCI USA (2x) Leveraged"},
        }
        hist_by_symbol = {
            "CL2.MI": pd.DataFrame(
                {"Close": [10.0] * 1276},
                index=pd.date_range("2021-01-04", periods=1276, freq="B"),
            ),
            "18MF.MU": pd.DataFrame(
                {"Close": [30.0, 30.48]},
                index=pd.to_datetime(["2026-07-28", "2026-07-29"]),
            ),
        }
        monkeypatch.setattr(enricher, "_openfigi_name", lambda i: canonical)
        monkeypatch.setattr(enricher, "_openfigi_lookup",
                            lambda i: ["CL2.PA", "CL2", "18MF.MU"])
        monkeypatch.setattr(enricher, "_fetch_ticker_info",
                            lambda s: info_by_symbol.get(s, {}))
        monkeypatch.setattr(enricher, "_fetch_history",
                            lambda s: hist_by_symbol.get(s, pd.DataFrame()))
        monkeypatch.setattr(enricher.price_cache, "store_resolution",
                            lambda *a, **k: None)

        result = enricher._resolve_isin(isin, hint_ticker="", expected_currency="EUR")
        assert result is not None
        _, symbol = result
        assert symbol == "CL2.MI"


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


class TestFxMemoization:
    def test_same_currency_fetched_once_per_run(self, monkeypatch):
        enricher.reset_run_caches()
        calls = {"n": 0}

        def fake_uncached(currency):
            calls["n"] += 1
            return pd.Series([1.1], index=pd.to_datetime(["2026-07-20"]))

        monkeypatch.setattr(enricher, "_fetch_fx_pair_uncached", fake_uncached)
        # Many holdings in the same currency…
        enricher._fetch_fx_pair("USD")
        enricher._fetch_fx_pair("USD")
        enricher._fetch_fx_pair("USD")
        # …collapse to one disk/network resolution.
        assert calls["n"] == 1

    def test_reset_clears_fx_memo(self, monkeypatch):
        enricher.reset_run_caches()
        calls = {"n": 0}

        def fake_uncached(currency):
            calls["n"] += 1
            return pd.Series([1.1], index=pd.to_datetime(["2026-07-20"]))

        monkeypatch.setattr(enricher, "_fetch_fx_pair_uncached", fake_uncached)
        enricher._fetch_fx_pair("USD")
        enricher.reset_run_caches()
        enricher._fetch_fx_pair("USD")
        assert calls["n"] == 2


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


class TestMemoSurvivesEnrichPasses:
    """The provider memo is run-scoped: one yfinance/OpenFIGI call per
    instrument identity per run. A run makes several enrich_holdings passes
    (holdings, historical ISINs for returns, backtest candidates) — none may
    wipe the memo the earlier pass filled, or the same ticker is fetched
    2-3×. The reset happens once at run start; passes under an active session
    must NOT reset again."""

    def _holding(self):
        from tarzan.models.holding import Holding
        return Holding(
            isin="IE0006WW1TQ4", ticker="X", quantity=1.0,
            cost_basis_eur=1.0, market_value_eur=1.0, currency="EUR",
        )

    def test_passes_under_a_session_reset_once_not_per_pass(self, monkeypatch):
        from tarzan.runtime.session import (
            RunContext, RunMode, RunSession, activate_session,
        )
        import datetime as _dt

        resets = {"n": 0}
        real_reset = enricher.reset_run_caches
        monkeypatch.setattr(
            enricher, "reset_run_caches",
            lambda: (resets.__setitem__("n", resets["n"] + 1), real_reset())[1],
        )
        # Skip real network/classification; we only assert reset accounting.
        monkeypatch.setattr(enricher, "_enrich_and_classify", lambda h: h)

        ctx = RunContext(
            attempt_id="t", mode=RunMode.POINT_IN_TIME,
            effective_date=_dt.date(2026, 7, 22),
            captured_at=_dt.datetime(2026, 7, 22, tzinfo=_dt.timezone.utc),
        )
        session = RunSession(context=ctx, config_snapshot={}, ledger=None)

        # Run start resets once; then three enrichment passes must NOT reset.
        enricher.reset_run_caches()  # the orchestrator's one run-start reset
        with activate_session(session):
            enricher.enrich_holdings([self._holding()])   # pass 1: holdings
            enricher.enrich_holdings([self._holding()])   # pass 2: returns
            enricher.enrich_holdings([self._holding()])   # pass 3: backtest
        assert resets["n"] == 1

    def test_standalone_caller_without_session_still_resets(self, monkeypatch):
        resets = {"n": 0}
        real_reset = enricher.reset_run_caches
        monkeypatch.setattr(
            enricher, "reset_run_caches",
            lambda: (resets.__setitem__("n", resets["n"] + 1), real_reset())[1],
        )
        monkeypatch.setattr(enricher, "_enrich_and_classify", lambda h: h)

        # No active session (tool/test context) → each call starts fresh.
        enricher.enrich_holdings([self._holding()])
        assert resets["n"] == 1


# ── Enrichment must preserve input order ────────────────────────────────────
import time  # noqa: E402

from tarzan.models.holding import Holding  # noqa: E402


class TestEnrichHoldingsOrderIsInputOrder:
    """``enrich_holdings`` runs on a ThreadPoolExecutor and used to append
    results as futures completed, so the returned order followed thread
    scheduling rather than the input.

    That list is the rebalancer's coordinate order, and its iterated local
    search accepts an improvement at 1e-9: two runs of the *same* deterministic
    analysis converged on different local optima and recommended materially
    different purchases from the same budget (CL2 and X25E each shifting by
    thousands of euros), while every other figure matched because sums and
    weighted averages are order-independent.

    These tests force completion order to be the reverse of input order, which
    is the condition the old code silently failed under.
    """

    @staticmethod
    def _holdings(n: int) -> list:
        return [Holding(isin=f"TEST{i:08d}", ticker=f"T{i}.MI",
                        name=f"Name {i}", quantity=1.0,
                        cost_basis_eur=100.0, market_value_eur=110.0,
                        currency="EUR") for i in range(n)]

    def test_order_survives_reversed_completion(self, monkeypatch):
        holdings = self._holdings(8)
        order = {h.ticker: i for i, h in enumerate(holdings)}

        def slow_by_index(h):
            # Later inputs finish first, so completion order is reversed.
            time.sleep(0.02 * (len(holdings) - order[h.ticker]))
            return h

        monkeypatch.setattr(enricher, "_enrich_and_classify", slow_by_index)
        result = enricher.enrich_holdings(holdings)
        assert [h.ticker for h in result] == [h.ticker for h in holdings]

    def test_diagnostics_do_not_follow_completion_order(self, monkeypatch):
        """A failure id must belong to the instrument it describes, run to run.

        ``failure_id`` is a ``(stage, code, ORDINAL)`` hash, so the id an
        instrument receives is decided by the order the ledger is TOLD about its
        diagnostic. Recording from inside the workers made that thread-completion
        order: two off-taxonomy instruments swapped their failure ids between two
        REPRODUCIBLE runs of the same book, in 6 of 44 runs.

        Eight holdings, not four, because ``MAX_WORKERS`` comes from config — with
        fewer workers than holdings the sleep ladder no longer forces a fully
        reversed completion and the test would pass without asserting anything.
        """
        import datetime as _dt

        from tarzan.runtime.ledger import LedgerEntryType, RunLedger
        from tarzan.runtime.session import (
            RunContext, RunMode, RunSession, activate_session,
        )
        from tarzan.runtime import data_quality as dq

        holdings = self._holdings(8)
        order = {h.ticker: i for i, h in enumerate(holdings)}

        def slow_then_complain(h):
            time.sleep(0.02 * (len(holdings) - order[h.ticker]))
            dq.error("instrument_capability",
                     "Explicit instrument kind/category is unknown or ambiguous",
                     context=h.isin)
            return h

        monkeypatch.setattr(enricher, "_enrich_and_classify", slow_then_complain)
        ledger = RunLedger("t")
        ctx = RunContext(
            attempt_id="t", mode=RunMode.REPRODUCIBLE,
            effective_date=_dt.date(2026, 8, 26),
            captured_at=_dt.datetime(2026, 8, 26, tzinfo=_dt.timezone.utc),
        )
        with activate_session(RunSession(context=ctx, config_snapshot={},
                                         ledger=ledger)):
            enricher.enrich_holdings(holdings)

        # Filter on the stage: enrich_holdings also emits its own per-holding
        # "no market price resolved" warnings, under stage "enricher".
        seen = [e.payload["original_failure"]["context"]
                for e in ledger.entries
                if e.entry_type is LedgerEntryType.FAILURE_OPEN
                and e.payload.get("stage") == "instrument_capability"]
        assert seen == [h.isin for h in holdings], (
            "the ledger recorded completion order, so ordinal 1 — and the "
            "failure_id derived from it — belongs to the wrong instrument")

    def test_failed_holding_keeps_its_slot(self, monkeypatch):
        holdings = self._holdings(5)

        def fail_the_middle(h):
            if h.ticker == "T2.MI":
                raise RuntimeError("provider exploded")
            return h

        monkeypatch.setattr(enricher, "_enrich_and_classify", fail_the_middle)
        result = enricher.enrich_holdings(holdings)
        # The un-enriched holding is still returned, and still third.
        assert [h.ticker for h in result] == [h.ticker for h in holdings]
        assert len(result) == len(holdings)


class TestTaxonomyQuoteFallback:
    """An ISIN the taxonomy knows must never degrade to its raw form when Yahoo
    throttles every candidate's metadata away. The raw ISIN quotes on no venue
    and _sibling_symbols cannot expand it, so the holding loses its 1D and shows
    the ISIN (LU0328475792 / LU0380865021 on CI, 23 Aug 2026). The v7 quote
    batch is sturdier than quoteSummary/history under load, so resolution
    confirms a EUR venue through it."""

    def test_falls_back_to_taxonomy_venue_via_quote(self, monkeypatch):
        from tarzan.data import enricher, price_cache
        from tarzan.data import market_quotes as mq
        from tarzan import runtime

        monkeypatch.setattr(runtime, "allows_live_transport", lambda: True)
        monkeypatch.setattr(price_cache, "load_resolution", lambda i: None)
        monkeypatch.setattr(price_cache, "load_ticker_isin_reverse", lambda i: "")
        monkeypatch.setattr(price_cache, "store_resolution", lambda i, s: None)
        monkeypatch.setattr(enricher, "_openfigi_name", lambda isin: "")
        monkeypatch.setattr(enricher, "_openfigi_lookup", lambda isin: [])
        # Every candidate's metadata is throttled away.
        monkeypatch.setattr(enricher, "_collect_candidate_metas", lambda ci, hint: [])
        # The v7 quote batch still answers for the Milan line.
        monkeypatch.setattr(
            mq, "official_quotes",
            lambda syms: {"XSX6.MI": {"price": 170.36, "prev_close": 169.6}})
        monkeypatch.setattr(
            enricher, "_fetch_history",
            lambda s: pd.DataFrame({"Close": [169.0, 170.36]},
                                   index=pd.to_datetime(["2026-08-20", "2026-08-21"])))

        result = enricher._resolve_isin("LU0328475792")  # taxonomy → XSX6
        assert result is not None, "must not surrender to the raw ISIN"
        assert result[1] == "XSX6.MI"

    def test_no_taxonomy_ticker_still_returns_none(self, monkeypatch):
        from tarzan.data import enricher, price_cache
        from tarzan import runtime

        monkeypatch.setattr(runtime, "allows_live_transport", lambda: True)
        monkeypatch.setattr(price_cache, "load_resolution", lambda i: None)
        monkeypatch.setattr(price_cache, "load_ticker_isin_reverse", lambda i: "")
        monkeypatch.setattr(enricher, "_openfigi_name", lambda isin: "")
        monkeypatch.setattr(enricher, "_openfigi_lookup", lambda isin: [])
        monkeypatch.setattr(enricher, "_collect_candidate_metas", lambda ci, hint: [])
        # An ISIN with no taxonomy row: nothing to fall back to.
        assert enricher._resolve_isin("XX0000000000") is None

    def test_degenerate_isin_self_mapping_is_rejected(self, monkeypatch):
        """A poisoned cache entry mapping the ISIN to itself (from a past
        throttled run) must not be trusted just because instrument_taxonomy_has
        recognises the ISIN. It re-resolves to the real venue every run, since
        CI restores the poisoned cache and never re-saves on a same-day hit."""
        from tarzan.data import enricher, price_cache
        from tarzan.data import market_quotes as mq
        from tarzan import runtime

        monkeypatch.setattr(runtime, "allows_live_transport", lambda: True)
        # The poison: cache says LU0328475792 → LU0328475792.
        monkeypatch.setattr(price_cache, "load_resolution", lambda i: "LU0328475792")
        monkeypatch.setattr(price_cache, "load_ticker_isin_reverse", lambda i: "")
        monkeypatch.setattr(price_cache, "store_resolution", lambda i, s: None)
        monkeypatch.setattr(enricher, "_openfigi_name", lambda isin: "")
        monkeypatch.setattr(enricher, "_openfigi_lookup", lambda isin: [])
        monkeypatch.setattr(enricher, "_collect_candidate_metas", lambda ci, hint: [])
        monkeypatch.setattr(
            mq, "official_quotes",
            lambda syms: {"XSX6.MI": {"price": 170.36, "prev_close": 169.6}})
        monkeypatch.setattr(
            enricher, "_fetch_history",
            lambda s: pd.DataFrame({"Close": [169.0, 170.36]},
                                   index=pd.to_datetime(["2026-08-20", "2026-08-21"])))

        result = enricher._resolve_isin("LU0328475792")
        assert result is not None and result[1] == "XSX6.MI", (
            "poisoned ISIN→ISIN cache must be rejected and re-resolved"
        )


class TestADegenerateSelfMapIsDropped:
    """A cached ISIN→itself must be dropped whether or not the taxonomy knows it.

    The guard was conditioned on ``taxonomy_ticker``, i.e. on the one case that also
    self-heals WITHOUT it — a curated ISIN is re-resolved through the taxonomy plus
    the v7 quote fallback anyway. So it never fired for the case it was written for:
    a taxonomy-unknown ISIN, where nothing else drops the self-map and _resolve_isin
    returns the ISIN itself as the resolved symbol.

    Measured on a held instrument the taxonomy does not describe: re-resolving gives
    a venue with 1268 closes in EUR against the self-map's 1261 in USD. The guard
    firing is a small improvement, not a regression — but the point is that a
    resolution nothing verified should not persist.
    """

    @staticmethod
    def _drops(cached: str, isin: str, taxonomy_ticker: str) -> bool:
        """Replay the guard's own condition, which is what the fix changed."""
        return bool(cached
                    and cached.replace("-", "").casefold() == isin.casefold())

    def test_it_is_dropped_for_a_taxonomy_unknown_isin(self):
        """The case the old condition could not reach."""
        assert self._drops("IE00BZ0PKT83", "IE00BZ0PKT83", taxonomy_ticker="")

    def test_it_is_still_dropped_for_a_curated_one(self):
        assert self._drops("IE00B4L5Y983", "IE00B4L5Y983", taxonomy_ticker="SWDA")

    def test_an_ordinary_resolution_is_untouched(self):
        assert not self._drops("SWDA.MI", "IE00B4L5Y983", taxonomy_ticker="SWDA")

    def test_the_guard_in_the_module_no_longer_requires_a_curated_ticker(self):
        """Pin the condition itself: the whole defect was one extra clause."""
        import inspect

        src = inspect.getsource(enricher._resolve_isin)
        i = src.find("Dropping degenerate cached resolution")
        assert i > 0, "the guard is gone"
        guard = src[max(0, i - 400):i]
        assert "and taxonomy_ticker" not in guard, \
            "the self-map guard is conditioned on the taxonomy again"


class TestTheTwoVenueListsAreConsistent:
    """Two suffix lists exist for DIFFERENT questions, and conflating them is how
    .SW came to be unreachable by ISIN while the other list carried it all along.

    ``isin_exchange_suffixes`` answers "which venues do I probe for an ISIN, and
    which wins a tie". ``_EUR_VENUE_SUFFIXES`` answers "which venues quote in EUR by
    definition" — so .L (GBP/USD) and .SW (CHF) are absent from it deliberately,
    their currency being genuinely ambiguous from the suffix. Neither list is a
    superset of the other and neither should be.
    """

    def test_a_probed_venue_is_either_a_known_eur_venue_or_ambiguous(self):
        """Not "must be in the EUR list" — that is false by design. Every probed
        venue must be one the currency question has an ANSWER for, either "EUR" or
        an explicit "ask the provider"."""
        from tarzan.data.enricher import venue_currency

        for suffix in enricher.ISIN_EXCHANGE_SUFFIXES:
            answer = venue_currency(f"AAA{suffix}")
            assert answer in ("EUR", None), f"{suffix} -> {answer!r}"
        # And the two that must be ambiguous ARE, so a EUR default cannot slip in.
        assert venue_currency("AAA.L") is None
        assert venue_currency("AAA.SW") is None

    def test_the_shared_venues_agree_about_preference(self):
        """Where both lists name a venue, their relative order must match: the same
        instrument preferred differently depending on which path reached it is a
        contradiction, not a policy."""
        isin_order = [s for s in enricher.ISIN_EXCHANGE_SUFFIXES
                      if s in enricher._EUR_VENUE_SUFFIXES]
        eur_order = [s for s in enricher._EUR_VENUE_SUFFIXES
                     if s in enricher.ISIN_EXCHANGE_SUFFIXES]
        assert isin_order == eur_order, \
            f"the two lists disagree about preference: {isin_order} vs {eur_order}"

    def test_deleting_a_probe_venue_does_not_unlearn_its_currency(self):
        """.ETLX was dropped from the ISIN sweep because it never admits a
        candidate. EuroTLX is still a EUR venue, so anything that reaches it by
        another path must still be recognised as quoting EUR."""
        from tarzan.data.enricher import venue_currency

        assert ".ETLX" not in enricher.ISIN_EXCHANGE_SUFFIXES
        assert venue_currency("AAA.ETLX") == "EUR"
