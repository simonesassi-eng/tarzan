"""Independent primary acquisitions overlap under one existing native limit."""

import threading

import pandas as pd
import pytest

from tarzan import config
from tarzan.data import _yf_net, enricher, market_quotes
from tarzan.runtime import data_quality as dq
from tarzan.runtime.session import current_session
from tarzan.tests.test_essential_data_priority import owned
from tarzan.tests.test_priority_quote_prefetch import (
    configure, engine, required_catalog,
)


def frame():
    return pd.DataFrame(
        {"Close": [100., 101.]}, index=pd.date_range("2026-10-21", periods=2),
    )


def stage(session):
    return next(
        entry.payload for entry in reversed(session.ledger.entries)
        if entry.payload.get("stage") == "provider_prefetch:essential_quotes"
    )


@pytest.mark.parametrize("maximum", [2, 3, 4, 5])
def test_independent_native_branches_overlap_without_multiplying_workers(monkeypatch, maximum):
    configure(monkeypatch)
    monkeypatch.setattr(config, "max_workers", lambda: maximum)
    monkeypatch.setattr(_yf_net, "space_yf_call", lambda: None)
    started_bars, started_macros = threading.Event(), threading.Event()
    release = threading.Event()
    guard = threading.Lock()
    active = 0
    peak = 0
    seen = []

    with owned() as (session, clock):
        def acquire(symbol, kind):
            nonlocal active, peak
            assert current_session() is session
            # Essential evidence must not inherit an optional-work cap.
            assert session.transport.call("yahoo", lambda cap: cap) == 20.
            with guard:
                active += 1
                peak = max(peak, active)
                seen.append((symbol, kind))
                (started_bars if kind == "bars" else started_macros).set()
                if started_bars.is_set() and started_macros.is_set():
                    release.set()
            try:
                assert release.wait(timeout=3), "independent branch was serialized"
                return frame()
            finally:
                with guard:
                    active -= 1

        monkeypatch.setattr(_yf_net, "ticker", lambda symbol: type("Ticker", (), {
            "history": lambda self, **kw: acquire(symbol, "bars"),
        })())
        monkeypatch.setattr(enricher, "_fetch_history", lambda symbol: acquire(symbol, "macro"))
        subject = engine()
        subject._prefetch_priority_quotes(required_catalog())
        assert stage(session)["outcome"] == "SUCCEEDED"
    assert started_bars.is_set() and started_macros.is_set()
    assert 2 <= peak <= maximum
    assert {symbol for symbol, kind in seen if kind == "macro"} == {"MACRO-A", "MACRO-B"}
    assert {symbol for symbol, kind in seen if kind == "bars"} == {
        "MACRO-A", "MACRO-B", "HELD.MI", "TARGET.DE", "CORE.MI",
    }
    assert subject.holdings[0].ticker == "HELD.MI"
    assert subject.holdings[0].target_portfolio == 10


def test_single_worker_keeps_serial_caller_semantics(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(config, "max_workers", lambda: 1)
    seen, parameters = [], []

    def bars(symbols, **kwargs):
        seen.append("bars")
        parameters.append(kwargs)
        return frame()

    monkeypatch.setattr(_yf_net, "fetch_download", bars)
    monkeypatch.setattr(enricher, "_fetch_history", lambda symbol: seen.append(symbol) or frame())
    with owned() as (session, _):
        engine()._prefetch_priority_quotes(required_catalog())
        assert stage(session)["outcome"] == "SUCCEEDED"
    assert seen == ["bars", "MACRO-A", "MACRO-B"]
    assert parameters[0]["threads"] == 1


@pytest.mark.parametrize("maximum", [2, 3, 7])
def test_an_empty_macro_universe_keeps_the_full_bar_worker_allowance(monkeypatch, maximum):
    configure(monkeypatch)
    monkeypatch.setattr(config, "max_workers", lambda: maximum)
    monkeypatch.setattr(market_quotes, "MARKETS", ())
    parameters = []
    monkeypatch.setattr(_yf_net, "fetch_download", lambda *a, **kw:
                        parameters.append(kw) or frame())
    monkeypatch.setattr(enricher, "_fetch_history", lambda symbol: pytest.fail("unexpected macro"))
    with owned() as (session, _):
        engine()._prefetch_priority_quotes(required_catalog())
        assert stage(session)["outcome"] == "SUCCEEDED"
    assert parameters[0]["threads"] == maximum


def test_macro_diagnostics_follow_input_order_not_completion_order(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(config, "max_workers", lambda: 4)
    monkeypatch.setattr(_yf_net, "fetch_download", lambda *a, **kw: frame())
    second_done = threading.Event()
    completion = []

    def history(symbol):
        if symbol == "MACRO-A":
            assert second_done.wait(timeout=3)
        dq.warning("synthetic_prefetch", "synthetic diagnostic", context=symbol)
        completion.append(symbol)
        if symbol == "MACRO-B":
            second_done.set()
        return frame()

    monkeypatch.setattr(enricher, "_fetch_history", history)
    with owned() as (session, _):
        engine()._prefetch_priority_quotes(required_catalog())
        assert stage(session)["outcome"] == "SUCCEEDED"
        assert [issue.context for issue in session.diagnostics] == ["MACRO-A", "MACRO-B"]
    assert completion == ["MACRO-B", "MACRO-A"]


def test_cancellation_retains_already_buffered_macro_evidence(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(config, "max_workers", lambda: 4)
    observed = threading.Event()

    def bars(*args, **kwargs):
        assert observed.wait(timeout=3)
        raise KeyboardInterrupt()

    def history(symbol):
        dq.warning("synthetic_prefetch", "acquired before cancellation", context=symbol)
        observed.set()
        return frame()

    monkeypatch.setattr(_yf_net, "fetch_download", bars)
    monkeypatch.setattr(enricher, "_fetch_history", history)
    with owned() as (session, _):
        with pytest.raises(KeyboardInterrupt):
            engine()._prefetch_priority_quotes(required_catalog())
        assert stage(session)["outcome"] == "INTERRUPTED"
        assert [issue.context for issue in session.diagnostics] == ["MACRO-A", "MACRO-B"]


def test_one_failed_macro_keeps_the_other_branch_and_original_quality_outcome(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(config, "max_workers", lambda: 4)
    seen = []
    monkeypatch.setattr(_yf_net, "fetch_download", lambda *a, **kw: seen.append("bars") or frame())

    def history(symbol):
        seen.append(symbol)
        if symbol == "MACRO-A":
            raise TimeoutError("synthetic native failure")
        return frame()

    monkeypatch.setattr(enricher, "_fetch_history", history)
    with owned() as (session, _):
        engine()._prefetch_priority_quotes(required_catalog())
        assert stage(session)["outcome"] == "DEGRADED"
    assert set(seen) == {"bars", "MACRO-A", "MACRO-B"}


def test_expired_authority_starts_no_requests_in_either_branch(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(config, "max_workers", lambda: 4)
    monkeypatch.setattr(_yf_net, "space_yf_call", lambda: None)
    monkeypatch.setattr(_yf_net, "ticker", lambda symbol: pytest.fail("late native bars"))
    with owned() as (session, clock):
        clock[0] = 30
        monkeypatch.setattr(enricher, "_fetch_history", lambda symbol:
                            session.transport.call("yahoo", lambda cap: pytest.fail("late native macro")))
        engine()._prefetch_priority_quotes(required_catalog())
        assert stage(session)["outcome"] == "DEGRADED"


@pytest.mark.parametrize("maximum", [0, -1, True, "4"])
def test_invalid_worker_allowance_fails_before_any_provider_acquisition(monkeypatch, maximum):
    configure(monkeypatch)
    monkeypatch.setattr(config, "max_workers", lambda: maximum)
    monkeypatch.setattr(_yf_net, "fetch_download", lambda *a, **kw: pytest.fail("invalid bar worker limit"))
    monkeypatch.setattr(enricher, "_fetch_history", lambda *a: pytest.fail("invalid macro worker limit"))
    with owned() as (session, _):
        engine()._prefetch_priority_quotes(required_catalog())
        assert stage(session)["outcome"] == "DEGRADED"
