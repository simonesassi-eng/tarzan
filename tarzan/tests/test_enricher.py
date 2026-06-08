"""Unit tests for enricher FX helpers (no network)."""

from __future__ import annotations

import pandas as pd

from tarzan.data.enricher import _normalize_minor_currency


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


def _cand(symbol, *, price=10.0, currency="EUR", name="", has_history=True):
    """Build a candidate with a minimal data dict for ranking tests."""
    return _Candidate(
        symbol=symbol,
        data={"info": {"currency": currency}, "history": pd.DataFrame()},
        price=price,
        currency=currency,
        name=name,
        has_history=has_history,
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
