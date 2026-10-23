"""Geography reuses the owned metadata observation, never another run's data."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import datetime, timezone
from threading import Barrier, Lock
import time

import pytest

from tarzan.data import _yf_net, enricher, geo_resolver
from tarzan.models.holding import Geography
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session


def _session(name):
    context = RunContext(name, RunMode.LIVE, None, datetime.now(timezone.utc))
    return RunSession(context, {}, RunLedger(name))


@pytest.fixture
def metadata(monkeypatch):
    calls = []
    responses = {}
    lock = Lock()

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def info(self):
            with lock:
                calls.append(self.symbol)
            # Exercise both an in-flight miss and a completed memo hit.
            time.sleep(0.02)
            value = responses[self.symbol]
            if isinstance(value, Exception):
                raise value
            return value

    monkeypatch.setattr(_yf_net, "ticker", Ticker)
    monkeypatch.setattr(_yf_net, "space_yf_call", lambda: None)
    monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
    return responses, calls


@pytest.mark.parametrize(("info", "isin", "expected"), [
    ({"quoteType": "EQUITY", "country": "United States"}, "", {Geography.USA: 100.0}),
    ({"quoteType": "EQUITY", "country": "Switzerland"}, "", {Geography.DEVELOPED_EX_USA_EMU_JP: 100.0}),
    ({"quoteType": "EQUITY"}, "CH0000000000", {Geography.DEVELOPED_EX_USA_EMU_JP: 100.0}),
    ({"quoteType": "ETF", "country": "Ireland"}, "IE0000000000", None),
    ({"quoteType": "EQUITY", "country": "Atlantis"}, "", None),
    ({}, "", None),
    (ValueError("synthetic metadata unavailable"), "", None),
])
def test_owned_geography_reuses_enrichment_observation(metadata, info, isin, expected):
    responses, calls = metadata
    responses["SYNTH.MI"] = info
    with activate_session(_session("reuse")):
        enricher._fetch_ticker_info("SYNTH.MI")
        result = geo_resolver._geo_from_own_country("SYNTH.MI", isin)
    assert (result[0] if result is not None else None) == expected
    assert calls == ["SYNTH.MI"]


def test_geography_and_enrichment_share_a_concurrent_miss(metadata):
    responses, calls = metadata
    responses["SYNTH"] = {"quoteType": "EQUITY", "country": "United States"}
    barrier = Barrier(2)

    def fetch(function):
        barrier.wait(timeout=5)
        return function("SYNTH")

    with activate_session(_session("concurrent")), ThreadPoolExecutor(max_workers=2) as pool:
        info = pool.submit(copy_context().run, fetch, enricher._fetch_ticker_info)
        geo = pool.submit(copy_context().run, fetch, geo_resolver._geo_from_own_country)
        assert info.result(timeout=5)["country"] == "United States"
        assert geo.result(timeout=5)[0] == {Geography.USA: 100.0}
    assert calls == ["SYNTH"]


def test_next_run_refetches_and_cannot_replace_another_runs_metadata(metadata):
    responses, calls = metadata
    first, second = _session("first"), _session("second")
    responses["SYNTH"] = {"quoteType": "EQUITY", "country": "United States"}
    with activate_session(first):
        assert geo_resolver._geo_from_own_country("SYNTH")[0] == {Geography.USA: 100.0}
    responses["SYNTH"] = {"quoteType": "EQUITY", "country": "Switzerland"}
    with activate_session(second):
        assert geo_resolver._geo_from_own_country("SYNTH")[0] == {
            Geography.DEVELOPED_EX_USA_EMU_JP: 100.0,
        }
    with activate_session(first):
        assert geo_resolver._geo_from_own_country("SYNTH")[0] == {Geography.USA: 100.0}
    assert calls == ["SYNTH", "SYNTH"]


def test_distinct_listings_do_not_share_metadata(metadata):
    responses, calls = metadata
    responses.update({
        "SYNTH.MI": {"quoteType": "EQUITY", "country": "United States"},
        "SYNTH.DE": {"quoteType": "EQUITY", "country": "Switzerland"},
    })
    with activate_session(_session("listings")):
        enricher._fetch_ticker_info("SYNTH.MI")
        assert geo_resolver._geo_from_own_country("SYNTH.DE")[0] == {
            Geography.DEVELOPED_EX_USA_EMU_JP: 100.0,
        }
    assert calls == ["SYNTH.MI", "SYNTH.DE"]


def test_geography_cannot_corrupt_the_shared_metadata_observation(metadata):
    responses, calls = metadata
    responses["SYNTH"] = {
        "quoteType": "EQUITY", "country": "United States", "evidence": {"source": ["original"]},
    }
    with activate_session(_session("isolation")):
        info = enricher._fetch_ticker_info("SYNTH")
        info["country"] = "Switzerland"
        info["evidence"]["source"].append("changed")
        assert geo_resolver._geo_from_own_country("SYNTH")[0] == {Geography.USA: 100.0}
        assert enricher._fetch_ticker_info("SYNTH")["evidence"] == {"source": ["original"]}
    assert calls == ["SYNTH"]


def test_standalone_geography_keeps_its_direct_fetch_contract(monkeypatch, metadata):
    responses, calls = metadata
    responses["SYNTH"] = {"quoteType": "EQUITY", "country": "United States"}
    monkeypatch.setattr(
        enricher, "_fetch_ticker_info",
        lambda *args: pytest.fail("standalone geography must not consume legacy enrichment state"),
    )
    assert geo_resolver._geo_from_own_country("SYNTH")[0] == {Geography.USA: 100.0}
    assert calls == ["SYNTH"]


def test_pinned_geography_does_not_consume_live_memo_or_transport(monkeypatch):
    monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: False)
    monkeypatch.setattr(
        enricher, "_fetch_ticker_info", lambda *args: pytest.fail("live memo"),
    )
    monkeypatch.setattr(_yf_net, "fetch_yf", lambda *args, **kwargs: pytest.fail("transport"))
    with activate_session(_session("pinned-guard")):
        assert geo_resolver._geo_from_own_country("SYNTH") is None
