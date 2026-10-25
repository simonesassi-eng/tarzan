"""Bounded Yahoo batches preserve per-symbol budgets, identity and context."""

from dataclasses import replace
from datetime import datetime, timezone
import threading
from types import SimpleNamespace

import pandas as pd
import pytest
import yfinance

from tarzan import runtime
from tarzan.data import _yf_net
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session, current_session
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings


_NATIVE_DOWNLOAD = yfinance.download


@pytest.fixture
def owner(monkeypatch):
    # A pre-fix adapter would reach the SDK batch instead of the injected
    # per-symbol boundary. Never let a regression test open a real socket.
    monkeypatch.setattr(yfinance, "download", lambda *args, **kwargs: pd.DataFrame())
    context = RunContext("synthetic-batch", RunMode.LIVE, None, datetime.now(timezone.utc))
    budget = ProviderBudget(ProviderTransportSettings.configured(), RunLedger(context.attempt_id))
    session = RunSession(context, {}, budget.ledger, transport=budget)
    with runtime.bind_context(context), activate_session(session):
        try:
            yield session
        finally:
            budget.close()


def history(symbol, *, zone="Europe/Rome"):
    return pd.DataFrame(
        {"Open": [10.0, 11.0], "Close": [11.0, 12.0], "Volume": [100, 200]},
        index=pd.date_range("2026-10-21 10:00", periods=2, freq="15min", tz=zone),
    ).rename_axis("Datetime")


def test_batch_has_bounded_parallel_workers_with_the_exact_run_owner(monkeypatch, owner):
    import yfinance
    from tarzan import config

    monkeypatch.setattr(config, "max_workers", lambda: 4)
    monkeypatch.setattr(yfinance, "download", lambda *args, **kwargs: pytest.fail("shared SDK batch state"))
    barrier = threading.Barrier(4, timeout=3)
    observed = []
    guard = threading.Lock()

    def ticker(symbol):
        def fetch(**kwargs):
            with guard:
                observed.append((symbol, current_session(), kwargs))
            barrier.wait()
            return history(symbol)
        return SimpleNamespace(history=fetch)

    monkeypatch.setattr(_yf_net, "ticker", ticker)
    frame = _yf_net.download(["D.MI", "B.MI", "C.MI", "A.MI"], interval="15m", group_by="ticker")
    assert len(observed) == 4
    assert all(session is owner for _, session, _ in observed)
    assert set(frame.columns.get_level_values("Ticker")) == {"A.MI", "B.MI", "C.MI", "D.MI"}
    assert frame.index.tz is timezone.utc or str(frame.index.tz) == "UTC"


def test_batch_preserves_successful_symbols_when_one_provider_answer_fails(monkeypatch, owner):
    def ticker(symbol):
        def fetch(**kwargs):
            if symbol == "MISSING.MI":
                raise TimeoutError("synthetic provider failure")
            return history(symbol)
        return SimpleNamespace(history=fetch)

    monkeypatch.setattr(_yf_net, "ticker", ticker)
    frame = _yf_net.download(["MISSING.MI", "VALID.MI"], interval="15m", group_by="ticker")
    assert frame["VALID.MI"]["Close"].dropna().tolist() == [11.0, 12.0]
    assert frame["MISSING.MI"]["Close"].dropna().empty


@pytest.mark.parametrize("interval,ignore_tz", [("15m", False), ("1h", False), ("1d", True)])
@pytest.mark.parametrize("group_by", ["ticker", "column"])
def test_batch_shape_and_instants_match_native_sdk_with_stable_utc_contract(
    monkeypatch, owner, interval, ignore_tz, group_by,
):
    import yfinance
    import yfinance.multi as multi

    class Ticker:
        def __init__(self, symbol, *args, **kwargs):
            self.symbol = symbol
            # The native SDK batch inspects the outcome of the completed
            # per-symbol history call, including successful calls.
            self._price_history = SimpleNamespace(_last_error=None)

        def history(self, **kwargs):
            return history(self.symbol)

    monkeypatch.setattr(yfinance, "Ticker", Ticker)
    monkeypatch.setattr(multi, "Ticker", Ticker)
    expected = _NATIVE_DOWNLOAD(["A.MI", "B.MI"], interval=interval, threads=False,
                               group_by=group_by, progress=False)
    # The newer native SDK retains a shared listing timezone. Tarzan's
    # established intraday contract is UTC; convert instants, never strip tz.
    if not ignore_tz:
        expected.index = expected.index.tz_convert("UTC")
    actual = _yf_net.download(["A.MI", "B.MI"], interval=interval, group_by=group_by)
    pd.testing.assert_frame_equal(actual, expected, check_like=True)
    assert (actual.index.tz is None) is ignore_tz
    if not ignore_tz:
        assert str(actual.index.tz) == "UTC"


@pytest.mark.parametrize("group_by", ["ticker", "column"])
def test_batch_preserves_native_instants_across_different_exchange_timezones(
    monkeypatch, owner, group_by,
):
    import yfinance.multi as multi

    class Ticker:
        def __init__(self, symbol, *args, **kwargs):
            self.symbol = symbol
            self._price_history = SimpleNamespace(_last_error=None)

        def history(self, **kwargs):
            zone = "America/New_York" if self.symbol == "US" else "Europe/Rome"
            return history(self.symbol, zone=zone)

    monkeypatch.setattr(yfinance, "Ticker", Ticker)
    monkeypatch.setattr(multi, "Ticker", Ticker)
    expected = _NATIVE_DOWNLOAD(["EU", "US"], interval="15m", threads=False,
                               group_by=group_by, progress=False)
    expected.index = expected.index.tz_convert("UTC")
    actual = _yf_net.download(["EU", "US"], interval="15m", group_by=group_by)
    pd.testing.assert_frame_equal(actual, expected, check_like=True)
    assert str(actual.index.tz) == "UTC"


@pytest.mark.parametrize("group_by", ["ticker", "column"])
def test_single_symbol_non_multiindex_native_contract(monkeypatch, owner, group_by):
    monkeypatch.setattr(_yf_net, "ticker", lambda symbol: SimpleNamespace(history=lambda **kwargs: history(symbol)))
    frame = _yf_net.download("A.MI", interval="15m", group_by=group_by, multi_level_index=False)
    assert not isinstance(frame.columns, pd.MultiIndex)
    assert frame.columns.name is None
    assert frame["Close"].tolist() == [11.0, 12.0]


def test_serial_caller_request_is_respected_and_public_progress_is_never_used(monkeypatch, owner):
    ids = []
    monkeypatch.setattr(_yf_net, "ticker", lambda symbol: SimpleNamespace(
        history=lambda **kwargs: ids.append(threading.get_ident()) or history(symbol),
    ))
    frame = _yf_net.download(["A.MI", "B.MI"], threads=False, progress=True, group_by="ticker")
    assert len(set(ids)) == 1
    assert len(frame) == 2


def test_batch_does_not_apply_one_symbol_timeout_to_the_entire_catalog(monkeypatch, owner):
    # Deterministic fake time, serial batch: each symbol uses 1.5s of a 2s
    # operation. Three good answers span 4.5s, not one 2s operation.
    from tarzan import config
    clock = [0.0]
    owner.transport.clock = lambda: clock[0]
    owner.transport.deadline = 30.0
    owner.transport.settings = replace(owner.transport.settings, run_budget_seconds=30.0)
    policy = replace(owner.transport.policies["yahoo"], timeout_seconds=2.0)
    owner.transport.policies["yahoo"] = policy
    monkeypatch.setattr(config, "max_workers", lambda: 1)
    monkeypatch.setattr(_yf_net, "space_yf_call", lambda: None)
    caps = []

    def ticker(symbol):
        def fetch(**kwargs):
            def request(timeout):
                caps.append(timeout)
                clock[0] += 1.5
                return history(symbol)
            return owner.transport.call("yahoo", request)
        return SimpleNamespace(history=fetch)

    monkeypatch.setattr(_yf_net, "ticker", ticker)
    frame = _yf_net.fetch_download(["A.MI", "B.MI", "C.MI"], group_by="ticker", what="synthetic batch")
    assert all(frame[symbol]["Close"].dropna().tolist() == [11.0, 12.0] for symbol in ("A.MI", "B.MI", "C.MI"))
    assert caps == [2.0, 2.0, 2.0]
    assert clock[0] == 4.5


def test_each_child_keeps_the_native_operation_cap(monkeypatch, owner):
    clock = [0.0]
    owner.transport.clock = lambda: clock[0]
    owner.transport.deadline = 30.0
    owner.transport.policies["yahoo"] = replace(owner.transport.policies["yahoo"], timeout_seconds=2.0)
    caps = []

    def ticker(symbol):
        def fetch(**kwargs):
            def request(timeout):
                caps.append(timeout)
                clock[0] += 2.1
                return history(symbol)
            return owner.transport.call("yahoo", request)
        return SimpleNamespace(history=fetch)

    monkeypatch.setattr(_yf_net, "ticker", ticker)
    frame = _yf_net.download(["A.MI"], group_by="ticker")
    assert frame["A.MI"]["Close"].dropna().empty
    assert caps == [2.0]


def test_exhausted_run_starts_no_child_native_requests(monkeypatch, owner):
    owner.transport.deadline = 0
    monkeypatch.setattr(_yf_net, "ticker", lambda symbol: pytest.fail("started after the run deadline"))
    frame = _yf_net.download(["A.MI", "B.MI"], group_by="ticker")
    assert frame.empty


def test_overlapping_batches_reuse_owned_history_and_return_defensive_copies(monkeypatch, owner):
    seen = []
    monkeypatch.setattr(_yf_net, "ticker", lambda symbol: SimpleNamespace(
        history=lambda **kwargs: seen.append(symbol) or history(symbol),
    ))
    first = _yf_net.download(["A.MI"], period="5d", interval="15m", group_by="ticker")
    first.loc[:, ("A.MI", "Close")] = 999.0
    second = _yf_net.download(["A.MI", "B.MI"], period="5d", interval="15m", group_by="ticker")
    assert sorted(seen) == ["A.MI", "B.MI"]
    assert second["A.MI"]["Close"].tolist() == [11.0, 12.0]
    assert second["B.MI"]["Close"].tolist() == [11.0, 12.0]


def test_batch_memo_never_combines_different_history_parameters(monkeypatch, owner):
    seen = []
    monkeypatch.setattr(_yf_net, "ticker", lambda symbol: SimpleNamespace(
        history=lambda **kwargs: seen.append((symbol, kwargs["period"])) or history(symbol),
    ))
    _yf_net.download(["A.MI"], period="5d", interval="15m")
    _yf_net.download(["A.MI"], period="1d", interval="15m")
    _yf_net.download(["A.MI"], period="5d", interval="15m", group_by="ticker")
    assert seen == [("A.MI", "5d"), ("A.MI", "1d")]


def test_batch_memo_belongs_to_the_run_not_the_module(monkeypatch, owner):
    from contextvars import Context

    seen = []
    monkeypatch.setattr(_yf_net, "ticker", lambda symbol: SimpleNamespace(
        history=lambda **kwargs: seen.append(symbol) or history(symbol),
    ))
    _yf_net.download(["A.MI"], period="5d", interval="15m")
    context = RunContext("independent-batch", RunMode.LIVE, None, datetime.now(timezone.utc))
    budget = ProviderBudget(ProviderTransportSettings.configured(), RunLedger(context.attempt_id))
    other = RunSession(context, {}, budget.ledger, transport=budget)
    def independent_run():
        with runtime.bind_context(context), activate_session(other):
            _yf_net.download(["A.MI"], period="5d", interval="15m")
            _yf_net.download(["A.MI"], period="5d", interval="15m")

    try:
        Context().run(independent_run)
    finally:
        budget.close()
    assert seen == ["A.MI", "A.MI"]


def test_prefetched_facts_survive_expired_io_authority_without_new_requests(monkeypatch, owner):
    seen = []
    monkeypatch.setattr(_yf_net, "ticker", lambda symbol: SimpleNamespace(
        history=lambda **kwargs: seen.append(symbol) or history(symbol),
    ))
    _yf_net.download(["A.MI"], period="5d", interval="15m", group_by="ticker")
    owner.transport.deadline = 0
    monkeypatch.setattr(_yf_net, "space_yf_call", lambda:
                        owner.transport.call("yahoo", lambda timeout: None))
    frame = _yf_net.fetch_download(
        ["A.MI", "MISSING.MI"], period="5d", interval="15m",
        group_by="ticker", what="expired synthetic batch",
    )
    assert frame["A.MI"]["Close"].tolist() == [11.0, 12.0]
    assert frame["MISSING.MI"]["Close"].dropna().empty
    assert seen == ["A.MI"]


def test_concurrent_overlapping_batches_coalesce_the_same_native_history(monkeypatch, owner):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    first_started, second_started, release = (threading.Event() for _ in range(3))
    seen = []

    def ticker(symbol):
        def fetch(**kwargs):
            assert current_session() is owner
            seen.append(symbol)
            if symbol == "A.MI":
                first_started.set()
                assert release.wait(timeout=3)
            else:
                second_started.set()
            return history(symbol)
        return SimpleNamespace(history=fetch)

    monkeypatch.setattr(_yf_net, "ticker", ticker)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(copy_context().run, _yf_net.download, ["A.MI"], interval="15m")
        assert first_started.wait(timeout=3)
        second = pool.submit(copy_context().run, _yf_net.download, ["A.MI", "B.MI"], interval="15m")
        try:
            assert second_started.wait(timeout=3)
        finally:
            release.set()
        assert not first.result(timeout=3).empty
        assert not second.result(timeout=3).empty
    assert sorted(seen) == ["A.MI", "B.MI"]
