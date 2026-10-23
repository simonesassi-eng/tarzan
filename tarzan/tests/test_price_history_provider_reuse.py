"""Exact raw history observations may be shared, never synthetic price tapes."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from datetime import date, datetime, timezone
from threading import Barrier, Lock
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from tarzan import runtime
from tarzan.data import _yf_net, enricher, price_cache
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session


def _session(name):
    context = RunContext(
        name, RunMode.LIVE, None, datetime(2026, 1, 15, tzinfo=timezone.utc),
    )
    return RunSession(context, {}, RunLedger(name))


@contextmanager
def _owned(session):
    with runtime.bind_context(session.context), activate_session(session):
        try:
            yield session
        finally:
            session.close_transport()


def _frame():
    result = pd.DataFrame(
        {
            "Open": [100., 98., 99.],
            "High": [101., 99., 100.],
            "Low": [99., 97., 98.],
            "Close": [100., 98., 99.],
            "Adj Close": [98., 98., 99.],
            "Volume": [10, 20, 30],
            "Dividends": [0., 2., 0.],
            "Stock Splits": [0., 0., 0.],
        },
        index=pd.date_range("2025-01-01", periods=3),
    )
    result.attrs["provider_metadata"] = {"observations": ["original"]}
    return result


@pytest.fixture
def provider(monkeypatch):
    calls, paced = [], []
    lock = Lock()
    state = SimpleNamespace(value=_frame(), callback=None, delay=0)

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **arguments):
            with lock:
                calls.append((self.symbol, dict(arguments)))
            if state.delay:
                time.sleep(state.delay)
            if state.callback is not None:
                return state.callback(arguments)
            if isinstance(state.value, BaseException):
                raise state.value
            return state.value

    monkeypatch.setattr(_yf_net, "ticker", Ticker)
    monkeypatch.setattr(enricher, "_space_yf_call", lambda: paced.append(1))
    monkeypatch.setattr(enricher, "_fill_today_from_sibling", lambda symbol, frame: frame)
    return state, calls, paced


def test_owned_identical_downloads_preserve_both_price_rulers_and_actions(provider):
    state, calls, paced = provider
    original = state.value.copy(deep=True)
    with _owned(_session("same")):
        first = enricher._download_price_history("SYNTH.MI")
        second = enricher._download_price_history("SYNTH.MI")
    pd.testing.assert_frame_equal(first, second)
    assert first["Price Close"].tolist() == [100, 98, 99]
    assert first["Close"].tolist() == [98, 98, 99]
    assert first["Open"].tolist() == [98, 98, 99]
    assert first["Dividends"].tolist() == [0, 2, 0]
    assert first["Stock Splits"].tolist() == [0, 0, 0]
    assert "Adj Close" not in first
    pd.testing.assert_frame_equal(state.value, original)
    assert calls == [("SYNTH.MI", {"interval": "1d", "auto_adjust": False, "period": "5y"})]
    assert paced == [1]


@pytest.mark.parametrize("reverse", [False, True])
def test_sibling_and_own_history_reuse_only_the_raw_provider_response(provider, reverse):
    _, calls, _ = provider
    with _owned(_session("shared")):
        if reverse:
            history = enricher._fetch_history("SYNTH.DE")
            sibling = enricher._sibling_close_series("SYNTH.DE")
        else:
            sibling = enricher._sibling_close_series("SYNTH.DE")
            history = enricher._fetch_history("SYNTH.DE")
    pd.testing.assert_series_equal(history["Close"], sibling)
    assert history["Price Close"].tolist() == [100, 98, 99]
    assert len(calls) == 1


def test_shared_response_cannot_propagate_a_synthesized_close(provider, monkeypatch):
    _, calls, _ = provider
    synthetic_date = pd.Timestamp("2025-01-04")

    def synthesize(symbol, frame):
        result = frame.copy()
        result.loc[synthetic_date] = float("nan")
        result.loc[synthetic_date, "Close"] = 777
        result.attrs[enricher._HISTORY_SYNTHETIC_ATTR] = {"synthetic": {"price": 777}}
        return result

    monkeypatch.setattr(enricher, "_fill_today_from_sibling", synthesize)
    with _owned(_session("no-chain")):
        own = enricher._fetch_history("SYNTH.DE")
        sibling = enricher._sibling_close_series("SYNTH.DE")
        raw = enricher._download_price_history("SYNTH.DE")
    assert own.loc[synthetic_date, "Close"] == 777
    assert synthetic_date not in sibling.index
    assert synthetic_date not in raw.index
    assert enricher._HISTORY_SYNTHETIC_ATTR not in raw.attrs
    assert len(calls) == 1


def test_download_results_and_nested_metadata_are_isolated(provider):
    state, calls, _ = provider
    with _owned(_session("copies")):
        first = enricher._download_price_history("SYNTH")
        first.iloc[0, first.columns.get_loc("Close")] = 777
        first.attrs["provider_metadata"]["observations"].append("consumer mutation")
        state.value.iloc[0, state.value.columns.get_loc("Close")] = 888
        state.value.attrs["provider_metadata"]["observations"].append("source mutation")
        second = enricher._download_price_history("SYNTH")
    assert second["Close"].iloc[0] == 98
    assert second["Price Close"].iloc[0] == 100
    assert second.attrs["provider_metadata"] == {"observations": ["original"]}
    assert len(calls) == 1


@pytest.mark.parametrize("value", [
    None, pd.DataFrame(), pd.DataFrame({"Volume": [10]}),
    pd.DataFrame({"Close": [float("nan")]}),
    pd.DataFrame({"Close": [float("inf")]}),
    pd.DataFrame({"Close": [-1.]}),
    pd.DataFrame({"Close": [0.]}),
    pd.DataFrame({"Close": [100.], "Adj Close": [float("nan")]}),
])
def test_missing_or_unusable_data_is_not_retained_as_a_completed_observation(provider, value):
    state, calls, _ = provider
    state.value = value
    with _owned(_session("negative")):
        first = enricher._download_price_history("SYNTH")
        second = enricher._download_price_history("SYNTH")
        if first is not None:
            pd.testing.assert_frame_equal(first, second)
            first["consumer"] = 1
        third = enricher._download_price_history("SYNTH")
        assert third is None or "consumer" not in third
    assert len(calls) == 3
    with _owned(_session("next-negative")):
        enricher._download_price_history("SYNTH")
    assert len(calls) == 4


@pytest.mark.parametrize("negative", [False, True])
def test_concurrent_download_misses_fetch_and_pace_once(provider, negative):
    state, calls, paced = provider
    state.delay = 0.03
    if negative:
        state.value = pd.DataFrame()
    barrier = Barrier(6)

    def fetch():
        barrier.wait(timeout=5)
        return enricher._download_price_history("SYNTH")

    with _owned(_session("concurrent")), ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(copy_context().run, fetch) for _ in range(6)]
        results = [future.result(timeout=5) for future in futures]
    assert len(calls) == 1
    assert paced == [1]
    assert all(result.equals(results[0]) for result in results)
    results[0]["consumer"] = 1
    assert "consumer" not in results[1]


@pytest.mark.parametrize("missing", [None, pd.DataFrame(), pd.DataFrame({"Volume": [10]})])
def test_an_inconclusive_sibling_observation_cannot_hide_a_later_price_recovery(
    provider, missing,
):
    state, calls, _ = provider
    state.value = missing
    with _owned(_session("recovery")):
        assert enricher._sibling_close_series("SYNTH.DE") is None
        state.value = _frame()
        own = enricher._fetch_history("SYNTH.DE")
    assert own["Close"].tolist() == [98, 98, 99]
    assert own["Price Close"].tolist() == [100, 98, 99]
    assert len(calls) == 2


@pytest.mark.parametrize("error_code", ["Internal Server Error", "Service Unavailable"])
def test_installed_sdk_masks_provider_error_as_empty_data_and_recovery_remains_possible(
    provider, monkeypatch, error_code,
):
    import yfinance as yf
    from yfinance.scrapers.history import PriceHistory

    state, calls, _ = provider
    native_calls = []

    def unavailable(**arguments):
        native_calls.append(1)
        return SimpleNamespace(
            text="synthetic JSON response",
            json=lambda: {"chart": {
                "result": None,
                "error": {"code": error_code, "description": "synthetic provider outage"},
            }},
        )

    monkeypatch.setattr(yf.config.debug, "hide_exceptions", True)
    sdk = PriceHistory(SimpleNamespace(get=unavailable), "SYNTH", "Europe/Rome")
    masked = sdk.history(period="5y")
    assert masked.empty
    assert native_calls == [1]
    state.value = masked
    with _owned(_session("sdk-recovery")):
        assert enricher._download_price_history("SYNTH.DE").empty
        state.value = _frame()
        recovered = enricher._download_price_history("SYNTH.DE")
    assert recovered["Price Close"].tolist() == [100, 98, 99]
    assert len(calls) == 2


def test_different_requests_and_period_settings_do_not_alias(provider):
    _, calls, _ = provider
    start = datetime(2025, 1, 1)
    with _owned(_session("ranges")):
        enricher._download_price_history("SYNTH.MI")
        enricher._download_price_history("SYNTH.MI", start)
        enricher._download_price_history("SYNTH.MI", start)
        enricher._download_price_history("SYNTH.MI", datetime(2025, 1, 2))
        enricher._download_price_history("SYNTH.DE")
        enricher.set_portfolio_backtest_period("1y")
        enricher._download_price_history("SYNTH.MI")
        enricher._download_price_history("SYNTH.MI", start)
    assert calls == [
        ("SYNTH.MI", {"interval": "1d", "auto_adjust": False, "period": "5y"}),
        ("SYNTH.MI", {"interval": "1d", "auto_adjust": False, "start": start}),
        ("SYNTH.MI", {"interval": "1d", "auto_adjust": False, "start": datetime(2025, 1, 2)}),
        ("SYNTH.DE", {"interval": "1d", "auto_adjust": False, "period": "5y"}),
        ("SYNTH.MI", {"interval": "1d", "auto_adjust": False, "period": "1y"}),
    ]


def test_each_owned_run_refetches_without_replacing_another_runs_response(provider):
    state, calls, _ = provider
    first, second = _session("first"), _session("second")
    with _owned(first):
        old = enricher._download_price_history("SYNTH")
    state.value = state.value.copy()
    state.value["Adj Close"] = [97., 97., 97.]
    with _owned(second):
        new = enricher._download_price_history("SYNTH")
    with _owned(first):
        pd.testing.assert_frame_equal(enricher._download_price_history("SYNTH"), old)
    assert new["Close"].tolist() == [97, 97, 97]
    assert len(calls) == 2


def test_reset_permits_a_real_refresh_without_touching_other_runs(provider):
    _, calls, _ = provider
    first, second = _session("first"), _session("second")
    with _owned(first):
        enricher._download_price_history("SYNTH")
    with _owned(second):
        enricher._download_price_history("SYNTH")
        enricher.reset_run_caches()
        enricher._download_price_history("SYNTH")
    with _owned(first):
        enricher._download_price_history("SYNTH")
    assert len(calls) == 3


@pytest.mark.parametrize("error", [ValueError("synthetic unavailable"), CancelledError()])
def test_failed_or_cancelled_download_does_not_poison_the_memo(provider, error):
    state, calls, _ = provider
    state.value = error
    with _owned(_session("error")):
        with pytest.raises(type(error)) as caught:
            enricher._download_price_history("SYNTH")
        assert caught.value is error
        state.value = _frame()
        assert enricher._download_price_history("SYNTH")["Close"].tolist() == [98, 98, 99]
        enricher._download_price_history("SYNTH")
    assert len(calls) == 2


def test_standalone_downloads_retain_their_direct_fresh_fetch_contract(provider):
    _, calls, paced = provider
    enricher._download_price_history("SYNTH")
    enricher._download_price_history("SYNTH")
    assert len(calls) == 2
    assert paced == [1, 1]


def test_pinned_history_does_not_consume_an_owned_live_download(provider):
    _, calls, _ = provider
    with _owned(_session("pinned-guard")):
        enricher._download_price_history("SYNTH")
        runtime.configure(as_of=date(2025, 1, 2), deterministic=True)
        assert enricher._fetch_history("SYNTH").empty
    assert len(calls) == 1


def test_an_adjusted_batch_observation_cannot_satisfy_a_raw_price_request(provider):
    _, calls, _ = provider
    owner = _session("separate-rulers")
    with _owned(owner):
        owner.providers.store("yahoo.batch_history")[
            ("SYNTH", (("interval", "1d"), ("auto_adjust", True), ("period", "5y")))
        ] = pd.DataFrame({"Close": [777.]})
        result = enricher._download_price_history("SYNTH")
    assert result["Price Close"].tolist() == [100, 98, 99]
    assert len(calls) == 1


@pytest.mark.parametrize("factor", [0.98, 0.5, 1.1])
def test_revised_adjustment_forces_a_new_full_read_not_an_older_memo(
    provider, monkeypatch, tmp_path, factor,
):
    state, calls, _ = provider
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(price_cache, "covers_period", lambda *args: True)
    days = pd.date_range("2025-01-01", periods=10)
    old = pd.DataFrame(
        {"Close": [100.] * 9, "Price Close": [100.] * 9}, index=days[:9],
    )
    price_cache.store_history("SYNTH.MI", old)
    state.value = pd.DataFrame(
        {"Close": [100.] * 9, "Adj Close": [100.] * 9}, index=days[:9],
    )
    revised = pd.DataFrame(
        {"Close": [100.] * 9 + [100 * factor], "Adj Close": [100 * factor] * 10},
        index=days,
    )
    with _owned(_session("revised")):
        enricher._download_price_history("SYNTH.MI")
        state.callback = lambda arguments: (
            revised.loc[pd.Timestamp(arguments["start"]):]
            if "start" in arguments else revised
        )
        result = enricher._fetch_history("SYNTH.MI")
    assert result["Close"].tolist() == pytest.approx([100 * factor] * 10)
    assert result["Price Close"].iloc[0] == 100
    assert len(calls) == 3  # initial full observation, revised tail, mandatory new full read
    assert ["start" in arguments for _, arguments in calls] == [False, True, False]


@pytest.mark.parametrize("unavailable_full", [
    pd.DataFrame(),
    pd.DataFrame({"Close": [float("nan")], "Adj Close": [98.]},
                 index=pd.date_range("2025-01-01", periods=1)),
    pd.DataFrame({"Close": [float("inf")], "Adj Close": [98.]},
                 index=pd.date_range("2025-01-01", periods=1)),
    pd.DataFrame({"Close": [-100.], "Adj Close": [98.]},
                 index=pd.date_range("2025-01-01", periods=1)),
    pd.DataFrame({"Close": [100.], "Adj Close": [float("nan")]},
                 index=pd.date_range("2025-01-01", periods=1)),
    pd.DataFrame({"Close": [100.], "Adj Close": [0.]},
                 index=pd.date_range("2025-01-01", periods=1)),
])
def test_a_failed_forced_refresh_cannot_be_hidden_by_a_previous_full_observation(
    provider, monkeypatch, tmp_path, unavailable_full,
):
    state, calls, _ = provider
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(price_cache, "covers_period", lambda *args: True)
    days = pd.date_range("2025-01-01", periods=10)
    old = pd.DataFrame(
        {"Close": [100.] * 9, "Price Close": [100.] * 9}, index=days[:9],
    )
    price_cache.store_history("SYNTH.MI", old)
    state.value = pd.DataFrame(
        {"Close": [100.] * 9, "Adj Close": [100.] * 9}, index=days[:9],
    )
    revised = pd.DataFrame(
        {"Close": [100.] * 9 + [98.], "Adj Close": [98.] * 10}, index=days,
    )
    with _owned(_session("refresh-failed")):
        enricher._download_price_history("SYNTH.MI")
        state.callback = lambda arguments: (
            revised.loc[pd.Timestamp(arguments["start"]):]
            if "start" in arguments else unavailable_full
        )
        result = enricher._fetch_history("SYNTH.MI")
    pd.testing.assert_frame_equal(result, old)
    assert set(result.attrs[enricher._HISTORY_ORIGINS_ATTR].values()) == {
        enricher._HISTORY_ORIGIN_CACHE,
    }
    assert len(calls) == 3
