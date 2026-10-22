"""A discovery limit bounds failed probes too, without guessing listing identity."""

import pytest

from tarzan.data import enricher as E


@pytest.fixture
def discovery(monkeypatch):
    monkeypatch.setattr(E, "_openfigi_raw", lambda isin: [])
    monkeypatch.setattr(E, "_openfigi_lookup", lambda isin: ["ALIASA", "ALIASB"])
    seen = []
    monkeypatch.setattr(E, "_fetch_candidate_meta", lambda symbol: seen.append(symbol))
    return seen


def test_missing_candidates_cannot_exceed_the_actual_attempt_budget(discovery):
    assert E._collect_candidate_metas("IE0000000001", "HINT") == []
    assert len(discovery) == E._MAX_RESOLVE_FETCHES
    assert len(set(discovery)) == len(discovery)


def test_partial_answers_do_not_extend_the_attempt_budget(monkeypatch, discovery):
    def fetch(symbol):
        discovery.append(symbol)
        if len(discovery) % 3 == 0:
            return E._Candidate(symbol, {}, 10.0, "EUR", "Synthetic Fund")
        return None

    monkeypatch.setattr(E, "_fetch_candidate_meta", fetch)
    answers = E._collect_candidate_metas("IE0000000001", "HINT")
    assert len(discovery) == E._MAX_RESOLVE_FETCHES
    assert len(answers) == E._MAX_RESOLVE_FETCHES // 3


def test_us_listing_is_discovered_from_its_verified_mic_not_isin_country(monkeypatch, discovery):
    # An Irish ISIN can have a suffixless US listing. A US ISIN can have only
    # European listings: issuer domicile is not a Yahoo venue identifier.
    monkeypatch.setattr(E, "_openfigi_lookup", lambda isin: ["OTHER", "REAL"])
    monkeypatch.setattr(E, "_openfigi_raw", lambda isin: [{
        "data": [{"ticker": "REAL", "exchCode": "UNKNOWN", "micCode": "XNAS"}],
    }])
    E._collect_candidate_metas("IE0000000001", "")
    assert discovery[0] == "REAL"


def test_unverified_bare_alias_does_not_displace_a_verified_listing(monkeypatch, discovery):
    monkeypatch.setattr(E, "_openfigi_lookup", lambda isin: ["COLLISION", "REAL.MI"])
    monkeypatch.setattr(E, "_openfigi_raw", lambda isin: [{
        "data": [
            {"ticker": "COLLISION", "exchCode": "UNKNOWN"},
            {"ticker": "REAL", "micCode": "XMIL"},
        ],
    }])
    E._collect_candidate_metas("US0000000001", "")
    assert discovery[0] == "REAL.MI"


def test_curated_hint_keeps_its_preferred_venue_ahead_of_unrelated_aliases(monkeypatch, discovery):
    monkeypatch.setattr(E, "_openfigi_lookup", lambda isin: [f"ALIAS{i}.SW" for i in range(12)])
    monkeypatch.setattr(E, "_openfigi_raw", lambda isin: [{
        "data": [{"ticker": f"ALIAS{i}", "micCode": "XSWX"} for i in range(12)],
    }])
    E._collect_candidate_metas("FR0010755611", "CL2.PA")
    assert discovery[:2] == ["CL2.PA", "CL2"]
    assert "CL2.MI" in discovery
    assert len(discovery) == E._MAX_RESOLVE_FETCHES


def test_unknown_venue_never_fabricates_a_suffixless_verified_symbol():
    assert E._map_figi_to_yfinance("REAL", "UNKNOWN", "UNKNOWN") is None
