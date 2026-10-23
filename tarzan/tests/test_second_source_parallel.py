"""Synthetic owner/worker integration; all network adapters are forbidden."""

from dataclasses import replace
from datetime import date
from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace

import pandas as pd
import pytest

from tarzan import config, runtime
from tarzan.data import enricher, justetf, listing_identity, listing_aliases, market_quotes
from tarzan.engine import cross_check as xc, metrics, third_source_votes as votes
from tarzan.models.investor_config import InvestorConfig
from tarzan.runtime import data_quality as dq
from tarzan.runtime.session import current_session
from tarzan.tests.test_owned_third_source_votes import no_network, owner  # noqa: F401


class SyntheticSourceStop(BaseException):
    pass


@pytest.fixture
def ports(monkeypatch):
    """Mark only this isolated fixture's fake ports as native for scheduling."""
    trace = {"prepare": [], "adopt": [], "fetch": [], "qualify": [], "prefetch": []}
    trace_lock = Lock()
    worker = {"fetch": None}
    faults = {"prepare": None, "adopt": None, "source": None}
    parent = get_ident()
    tape = pd.Series([100.0, 110.0], index=pd.to_datetime(["2026-10-15", "2026-10-22"]))
    monkeypatch.setattr(config, "_load_indexes_csv", lambda: pd.DataFrame())
    monkeypatch.setattr(runtime, "today", lambda: date(2026, 10, 23))
    monkeypatch.setattr(listing_identity, "quote_identities", lambda symbols: {})
    monkeypatch.setattr(listing_identity, "ambiguous_relative_signatures", lambda *a: set())
    monkeypatch.setattr(listing_aliases, "verified_siblings", lambda *a, **kw: ())
    monkeypatch.setattr(
        market_quotes, "_sibling_symbols",
        lambda ticker, **kw: [
            ticker, ticker.replace(".MI", ".DE"), ticker.replace(".MI", ".PA"),
            ticker.replace(".MI", ".L"), ticker.replace(".MI", ".AS"),
        ],
    )

    def prefetch(isins):
        trace["prefetch"].append(tuple(isins))

    def series(isin):
        return tape

    def quote(isin):
        return {}

    def check_windows(own_tape, ref, ticker, **kw):
        assert get_ident() == parent
        trace["prepare"].append(ticker)
        if faults["prepare"] is not None:
            faults["prepare"](ticker)
        return {
            window: {
                "status": xc.DIVERGED if window in ("5d", "1m") else xc.OK,
                "ours": 10.0, "alt": 20.0,
                "anchor": date(2026, 10, 15), "at": date(2026, 10, 22),
                "window": window, "ticker": ticker,
            } for window in xc.LONG_WINDOWS
        }

    def check_one_day(*a, **kw):
        pytest.fail("a prior-session tape must not buy a current-session comparison")

    def qualify(own_tape, closes, *, ticker, symbol, **kw):
        with trace_lock:
            trace["qualify"].append((ticker, symbol))
        # Neither currency-incompatible nor absent venues can supply a vote.
        return closes if symbol.endswith(".DE") else None

    def settle(verdict, own_tape, thirds):
        assert get_ident() == parent
        trace["adopt"].append((verdict["ticker"], verdict["window"], len(thirds)))
        if faults["adopt"] is not None:
            faults["adopt"](verdict["ticker"])
        return dict(verdict, status=xc.CORRECTED, corrected_value=20.0, agreed=20.0)

    def fetch(symbol):
        with trace_lock:
            trace["fetch"].append((symbol, get_ident(), current_session()))
        if faults["source"] is not None:
            faults["source"](symbol)
        if worker["fetch"] is not None:
            worker["fetch"](symbol)
        dq.info("synthetic_vote", "candidate examined", context=symbol)
        return None if symbol.endswith(".AS") else tape

    for name, port in (("prefetch", prefetch), ("series", series), ("quote", quote)):
        monkeypatch.setattr(justetf, name, port)
    for name, port in (
        ("check_windows", check_windows), ("check_one_day", check_one_day),
        ("settle", settle), ("qualify_sibling", qualify),
    ):
        monkeypatch.setattr(xc, name, port)
    monkeypatch.setattr(justetf, "_NATIVE_REFEREE_PORTS", (prefetch, series, quote))
    monkeypatch.setattr(xc, "_NATIVE_REFEREE_PORTS", (check_windows, check_one_day, settle, qualify))
    monkeypatch.setattr(enricher, "_sibling_close_series", fetch)
    monkeypatch.setattr(enricher, "_NATIVE_SIBLING_CLOSE_SERIES", fetch)

    def subject(count=4, *, engine_type=metrics.MetricsEngine):
        holdings = [
            SimpleNamespace(
                ticker=f"AAA{index}.MI", isin=f"IE00000000{index:02}",
                price_history=tape, price_history_native=tape, price_currency="EUR",
            ) for index in range(count)
        ]
        hp = pd.DataFrame([
            {"ticker": h.ticker, "type": "In portfolio", "1d": 1.0,
             **{window: 10.0 for window in xc.LONG_WINDOWS}}
            for h in holdings
        ])
        return engine_type(holdings, InvestorConfig()), {"holding_performance": hp}

    return subject, trace, worker, faults, parent


@pytest.mark.parametrize("capacity", [1, 2, 4, 8])
@pytest.mark.parametrize("count", [1, 4, 32])
def test_rows_keep_every_candidate_qualification_and_canonical_adoption(monkeypatch, ports, capacity, count):
    subject, trace, worker, faults, parent = ports
    engine, ctx = subject(count)
    with owner(monkeypatch, capacity=capacity, workers=4) as (session, _):
        engine._second_source(ctx)
        assert all(s is session for _, _, s in trace["fetch"])
        assert session.transport.coordinator.snapshot()["active"] == 0
        recorded = dq.issues()
    expected = [f"AAA{index}.MI" for index in range(count)]
    assert trace["prepare"] == expected
    assert trace["adopt"] == [(ticker, window, 1) for ticker in expected for window in ("5d", "1m")]
    for ticker in expected:
        assert [symbol for symbol, _, _ in trace["fetch"] if symbol.startswith(ticker[:-3] + ".")] == [
            ticker.replace(".MI", ending) for ending in (".DE", ".PA", ".L", ".AS")
        ]
    assert len(trace["fetch"]) == count * 4
    assert len(trace["qualify"]) == count * 3
    assert ctx["holding_performance"]["5d"].tolist() == [20.0] * count
    expected_dq = [
        ticker.replace(".MI", ending) for ticker in expected for ending in (".DE", ".PA", ".L", ".AS")
    ]
    assert [issue.context for issue in recorded] == expected_dq
    if capacity == 1:
        assert all(thread == parent for _, thread, _ in trace["fetch"])
    else:
        assert all(thread != parent for _, thread, _ in trace["fetch"])


def test_rows_overlap_without_financial_writes_before_join(monkeypatch, ports):
    subject, trace, worker, faults, parent = ports
    engine, ctx = subject()
    rendezvous = Barrier(4)

    def simultaneous(symbol):
        if symbol.endswith(".DE"):
            assert ctx["holding_performance"]["xc_5d"].tolist() == [xc.UNAVAILABLE] * 4
            assert ctx["holding_performance"]["5d"].tolist() == [10.0] * 4
            rendezvous.wait(timeout=3)

    worker["fetch"] = simultaneous
    with owner(monkeypatch):
        engine._second_source(ctx)
    assert len(trace["adopt"]) == 8


@pytest.mark.parametrize("fault_type", [ValueError, KeyboardInterrupt, SyntheticSourceStop])
def test_later_setup_fault_adopts_only_completed_earlier_prefix(monkeypatch, ports, fault_type):
    subject, trace, worker, faults, parent = ports
    engine, ctx = subject()
    fault = fault_type("synthetic row setup stop")

    def setup(ticker):
        if ticker == "AAA2.MI":
            raise fault

    faults["prepare"] = setup
    with owner(monkeypatch):
        if isinstance(fault, Exception):
            engine._second_source(ctx)
        else:
            with pytest.raises(fault_type) as observed:
                engine._second_source(ctx)
            assert observed.value is fault
    assert ctx["holding_performance"]["5d"].tolist() == [20.0, 20.0, 10.0, 10.0]
    assert trace["adopt"] == [(ticker, window, 1) for ticker in ("AAA0.MI", "AAA1.MI")
                              for window in ("5d", "1m")]
    assert len(trace["fetch"]) == 8


@pytest.mark.parametrize("fault_type", [KeyboardInterrupt, SyntheticSourceStop])
def test_source_interruption_drains_accepted_rows_but_does_not_adopt_later_rows(monkeypatch, ports, fault_type):
    subject, trace, worker, faults, parent = ports
    engine, ctx = subject()
    fault = fault_type("synthetic third venue stop")
    rendezvous = Barrier(4)
    finished = []

    def source(symbol):
        if symbol.endswith(".DE"):
            rendezvous.wait(timeout=3)
            if symbol == "AAA1.DE":
                raise fault
        if symbol.endswith(".AS"):
            finished.append(symbol)

    faults["source"] = source
    with owner(monkeypatch):
        with pytest.raises(fault_type) as observed:
            engine._second_source(ctx)
        assert observed.value is fault
        recorded = dq.issues()
    assert sorted(finished) == ["AAA0.AS", "AAA2.AS", "AAA3.AS"]
    assert ctx["holding_performance"]["5d"].tolist() == [20.0, 10.0, 10.0, 10.0]
    assert [issue.context for issue in recorded] == [
        f"AAA0{ending}" for ending in (".DE", ".PA", ".L", ".AS")
    ]


@pytest.mark.parametrize("adoption_type, expected_type", [
    (ValueError, SyntheticSourceStop), (KeyboardInterrupt, KeyboardInterrupt),
])
def test_true_interruption_is_not_masked_by_ordinary_adoption_fault(monkeypatch, ports, adoption_type, expected_type):
    subject, trace, worker, faults, parent = ports
    engine, ctx = subject()
    source_error = SyntheticSourceStop("synthetic source primary")
    adoption_error = adoption_type("synthetic owner adoption")
    rendezvous = Barrier(4)

    def source(symbol):
        if symbol.endswith(".DE"):
            rendezvous.wait(timeout=3)
            if symbol == "AAA3.DE":
                raise source_error

    def adopt(ticker):
        if ticker == "AAA0.MI":
            raise adoption_error

    faults["source"], faults["adopt"] = source, adopt
    with owner(monkeypatch):
        with pytest.raises(expected_type) as observed:
            engine._second_source(ctx)
    assert observed.value is (source_error if adoption_type is ValueError else adoption_error)
    assert ctx["holding_performance"]["5d"].tolist() == [10.0] * 4
    assert len(trace["fetch"]) == 13


@pytest.mark.parametrize("injection", [
    "prefetch", "series", "quote", "check_windows", "check_one_day", "settle",
    "qualify_sibling", "sibling", "own_tape", "subclass", "optional", "same_attempt_foreign_context",
    "legacy_justetf", "legacy_cross_check", "legacy_sibling", "malformed_marker",
])
def test_injected_or_foreign_authority_preserves_serial_row_boundaries(monkeypatch, ports, injection):
    subject, trace, worker, faults, parent = ports

    class CustomEngine(metrics.MetricsEngine):
        pass

    engine, ctx = subject(engine_type=CustomEngine if injection == "subclass" else metrics.MetricsEngine)
    if injection in ("prefetch", "series", "quote"):
        original = getattr(justetf, injection)
        monkeypatch.setattr(justetf, injection, lambda *a, **kw: original(*a, **kw))
    elif injection in ("check_windows", "check_one_day", "settle", "qualify_sibling"):
        original = getattr(xc, injection)
        monkeypatch.setattr(xc, injection, lambda *a, **kw: original(*a, **kw))
    elif injection == "sibling":
        original = enricher._sibling_close_series
        monkeypatch.setattr(enricher, "_sibling_close_series", lambda *a: original(*a))
    elif injection == "own_tape":
        original = engine._own_tape
        engine._own_tape = lambda *a: original(*a)
    elif injection in ("legacy_justetf", "legacy_cross_check", "legacy_sibling"):
        module, marker = {
            "legacy_justetf": (justetf, "_NATIVE_REFEREE_PORTS"),
            "legacy_cross_check": (xc, "_NATIVE_REFEREE_PORTS"),
            "legacy_sibling": (enricher, "_NATIVE_SIBLING_CLOSE_SERIES"),
        }[injection]
        monkeypatch.delattr(module, marker)
    elif injection == "malformed_marker":
        monkeypatch.setattr(justetf, "_NATIVE_REFEREE_PORTS", (justetf.prefetch,))
    monkeypatch.setattr(votes, "ThreadPoolExecutor", lambda *a, **kw: pytest.fail("custom caller was batched"))
    with owner(monkeypatch) as (session, _):
        if injection == "optional":
            with session.transport.optional_work("synthetic_optional"):
                engine._second_source(ctx)
        elif injection == "same_attempt_foreign_context":
            with runtime.bind_context(replace(session.context)):
                engine._second_source(ctx)
        else:
            engine._second_source(ctx)
    assert len(trace["adopt"]) == 8
    assert all(thread == parent for _, thread, _ in trace["fetch"])


def test_ordinary_venue_failure_keeps_searching_the_remaining_venues(monkeypatch, ports):
    subject, trace, worker, faults, parent = ports
    engine, ctx = subject(1)

    def fail_one(symbol):
        if symbol.endswith(".PA"):
            raise TimeoutError("synthetic slow venue")

    faults["source"] = fail_one
    with owner(monkeypatch):
        engine._second_source(ctx)
    assert [symbol for symbol, _, _ in trace["fetch"]] == ["AAA0.DE", "AAA0.PA", "AAA0.L", "AAA0.AS"]
    assert trace["adopt"] == [("AAA0.MI", "5d", 1), ("AAA0.MI", "1m", 1)]


def test_unaccepted_prepared_row_preserves_original_owner_fault(monkeypatch, ports, caplog):
    subject, trace, worker, faults, parent = ports
    engine, ctx = subject()
    original = RuntimeError("synthetic callback setup interrupted")

    def before_accept(callbacks, **kw):
        next(iter(callbacks))
        return votes.ThirdSourceBatch((), original)

    monkeypatch.setattr(votes, "acquire_third_source_votes", before_accept)
    with owner(monkeypatch):
        engine._second_source(ctx)
    assert "synthetic callback setup interrupted" in caplog.text
    assert "list index out of range" not in caplog.text
    assert trace["fetch"] == trace["adopt"] == []
    assert ctx["holding_performance"]["5d"].tolist() == [10.0] * 4
