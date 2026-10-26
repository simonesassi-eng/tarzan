"""Completed benchmark consumers do not require another acquisition budget.

Native SDK decoders consume synthetic raw responses. These are no-extra-raw-
request tests, not physical HTTP or newsletter speedup measurements.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from functools import wraps
from threading import Event

import pandas as pd
import pytest
import yfinance as yf

from tarzan.data import _yf_net, enricher as en
from tarzan.engine import benchmarks as bm
from tarzan.runtime.memo import clear_provider_stores, memoized
from tarzan.tests.test_benchmark_history_capability import (
    owned, raw_market, resolution,
)
from tarzan.tests.test_native_isin_listing_identity import (
    ISIN, independent_synthetic_mapping, install_raw_source, owned as isin_owned,
)


@contextmanager
def expired(owner, boundary="optional"):
    budget = owner.transport
    if boundary == "whole-run":
        previous = budget.deadline
        budget.deadline = budget.clock() - 1
        try:
            yield
        finally:
            budget.deadline = previous
        return
    with budget.optional_work("completed-facts-outer"):
        previous = budget.clock
        instant = previous() + budget.settings.optional_budget_seconds + 1
        budget.clock = lambda: instant
        try:
            yield
        finally:
            budget.clock = previous


def rates(owner, currency, history):
    major = en._MINOR_TO_MAJOR_CURRENCY.get(currency, currency)
    fx = pd.Series(
        [.9, .91, .92], index=history.index, name="Close",
    )
    fx.attrs[en._HISTORY_SYMBOL_ATTR] = f"FX_{major}"
    fx.attrs[en._HISTORY_ORIGINS_ATTR] = {
        en._history_timestamp_key(index): en._HISTORY_ORIGIN_CACHE
        for index in fx.index
    }
    memoized("enricher.fx", major, lambda: fx, legacy={})
    return fx


def assert_matches(catalog, expected, symbol):
    assert set(catalog) == {"complete"}
    resolved = catalog["complete"]
    assert resolved.ticker == expected.attrs["resolved_ticker"]
    assert resolved.requested_ticker == symbol
    assert resolved.currency == expected.attrs[bm._CURRENCY_ATTR]
    canonical = expected.dropna().copy()
    canonical.attrs.pop(bm._NATIVE_ATTR)
    pd.testing.assert_series_equal(resolved.history, canonical)
    pd.testing.assert_series_equal(
        resolved.history_native,
        expected.attrs[bm._NATIVE_ATTR].rename(resolved.ticker),
    )
    assert resolved.history.attrs == canonical.attrs


@pytest.mark.parametrize("suffix", en._EUR_VENUE_SUFFIXES)
@pytest.mark.parametrize("boundary", ["optional", "whole-run"])
def test_all_declared_eur_venues_reuse_completed_native_history(
    raw_market, suffix, boundary,
):
    calls, _state = raw_market
    symbol = "COMPLETED" + suffix
    with owned() as owner:
        en._fetch_history(symbol)
        expected = bm._fetch_benchmark_uncached(symbol)
        before = len(calls)
        with expired(owner, boundary):
            catalog, errors = bm.preprocess_benchmarks({"complete": symbol}, workers=2)
        assert errors == ()
        assert_matches(catalog, expected, symbol)
        assert len(calls) == before
        assert not owner.providers.flight._pending


@pytest.mark.parametrize("symbol,currency", [
    ("COMPLETED.L", "USD"), ("COMPLETED.L", "GBp"),
    ("COMPLETED.SW", "CHF"), ("COMPLETED", "USD"),
])
def test_ambiguous_currency_preserves_native_minor_units_fx_and_provenance(
    raw_market, symbol, currency,
):
    calls, state = raw_market
    state["currency"] = currency
    with owned() as owner:
        history = en._fetch_history(symbol)
        en._fetch_ticker_info(symbol)
        fx = rates(owner, currency, history)
        expected = bm._fetch_benchmark_uncached(symbol)
        before = len(calls)
        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks({"complete": symbol}, workers=2)
        assert errors == ()
        assert_matches(catalog, expected, symbol)
        scale = 100 if currency == "GBp" else 1
        assert catalog["complete"].history.tolist() == pytest.approx(
            history["Close"].mul(fx).div(scale).tolist(),
        )
        assert len(calls) == before


@pytest.mark.parametrize("currency", ["", None])
def test_completed_unknown_currency_remains_unconverted_and_unlabelled(
    raw_market, currency,
):
    calls, state = raw_market
    state["currency"] = currency
    symbol = "COMPLETED.UNKNOWN"
    with owned() as owner:
        expected = bm._fetch_benchmark_uncached(symbol)
        before = len(calls)
        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks({"complete": symbol})
        assert errors == ()
        assert_matches(catalog, expected, symbol)
        assert catalog["complete"].currency == ""
        assert len(calls) == before


@pytest.mark.parametrize("missing", ["metadata", "fx", "history"])
def test_incomplete_dependencies_never_reopen_acquisition(raw_market, missing):
    calls, _state = raw_market
    symbol = "COMPLETED.L"
    with owned() as owner:
        history = en._fetch_history(symbol)
        if missing != "metadata":
            en._fetch_ticker_info(symbol)
        if missing != "fx":
            rates(owner, "USD", history)
        if missing == "history":
            clear_provider_stores("enricher.history")
        before = len(calls)
        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks({"complete": symbol})
        assert catalog == {}
        assert errors == ("complete: optional provider budget exhausted",)
        assert len(calls) == before
        assert not owner.providers.flight._pending


@pytest.mark.parametrize("invalid", [
    "empty", "one-row", "foreign-symbol", "no-close", "future", "partial-future", "nat",
])
def test_unusable_or_wrong_identity_history_is_not_a_completed_benchmark(
    raw_market, invalid,
):
    calls, _state = raw_market
    symbol = "COMPLETED.MI"
    with owned() as owner:
        history = en._fetch_history(symbol)
        if invalid == "empty":
            history = history.iloc[:0]
        elif invalid == "one-row":
            history = history.iloc[:1]
        elif invalid == "foreign-symbol":
            history.attrs[en._HISTORY_SYMBOL_ATTR] = "FOREIGN.MI"
        elif invalid == "no-close":
            history = history.drop(columns="Close")
        else:
            index = list(history.index)
            if invalid == "future":
                index = [point + pd.Timedelta(days=10000) for point in index]
            elif invalid == "partial-future":
                index[-1] += pd.Timedelta(days=10000)
            else:
                index[-1] = pd.NaT
            history.index = pd.DatetimeIndex(index)
        owner.providers.stores["enricher.history"][symbol] = history
        before = len(calls)
        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks({"complete": symbol})
        assert catalog == {}
        assert errors == ("complete: optional provider budget exhausted",)
        assert len(calls) == before


@pytest.mark.parametrize("symbol", ["COMPLETED.MI", "COMPLETED.L"])
def test_already_derived_benchmark_does_not_need_source_dependencies(raw_market, symbol):
    calls, _state = raw_market
    with owned() as owner:
        history = en._fetch_history(symbol)
        if symbol.endswith(".L"):
            rates(owner, "USD", history)
        expected = bm._fetch_benchmark_history(symbol)
        clear_provider_stores("enricher.history", "enricher.info", "enricher.fx")
        before = len(calls)
        with expired(owner, "whole-run"):
            catalog, errors = bm.preprocess_benchmarks({"complete": symbol})
        assert errors == ()
        assert_matches(catalog, expected, symbol)
        assert len(calls) == before


def test_bare_alias_reuses_only_current_completed_selected_listing(
    monkeypatch, raw_market, resolution,
):
    calls, _state = raw_market
    entries, now = resolution
    entries["TICKER:COMPLETED"] = {"symbol": "COMPLETED.DE", "ts": now}
    monkeypatch.setattr(en.cfg, "name_for", lambda *_: "Synthetic")
    with owned() as owner:
        en._fetch_history("COMPLETED.DE")
        expected = bm._fetch_benchmark_uncached("COMPLETED")
        before = len(calls)
        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks({"complete": "COMPLETED"})
        assert errors == ()
        assert_matches(catalog, expected, "COMPLETED")
        assert catalog["complete"].ticker == "COMPLETED.DE"
        assert len(calls) == before


@pytest.mark.parametrize("mapping", [None, "expired", "foreign-root"])
def test_unresolved_alias_cannot_discover_new_venues_after_expiry(
    monkeypatch, raw_market, resolution, mapping,
):
    calls, _state = raw_market
    entries, now = resolution
    if mapping is not None:
        entries["TICKER:COMPLETED"] = {
            "symbol": "FOREIGN.MI" if mapping == "foreign-root" else "COMPLETED.MI",
            "ts": 0 if mapping == "expired" else now,
        }
    monkeypatch.setattr(en.cfg, "name_for", lambda *_: "Synthetic")
    with owned() as owner:
        en._fetch_history("COMPLETED.MI")
        before = len(calls)
        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks({"complete": "COMPLETED"})
        assert catalog == {}
        assert errors == ("complete: optional provider budget exhausted",)
        assert len(calls) == before


@pytest.mark.parametrize("listing,currency", [
    ("COMPLETED.MI", "EUR"), ("COMPLETED.L", "GBp"),
])
def test_completed_native_isin_keeps_actual_listing_without_another_search(
    monkeypatch, listing, currency,
):
    calls = install_raw_source(monkeypatch, listing=listing, currency=currency)
    with isin_owned() as owner:
        data = en._fetch_ticker_data(ISIN)
        if currency != "EUR":
            fx = pd.Series(.9, index=data["history"].index, name="Close")
            memoized("enricher.fx", "GBP", lambda: fx, legacy={})
        expected = bm._fetch_benchmark_uncached(ISIN)
        before = len(calls)
        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks({"complete": ISIN})
        assert errors == ()
        assert_matches(catalog, expected, ISIN)
        assert catalog["complete"].ticker == listing
        assert len(calls) == before


def test_explicit_injected_fetch_is_never_called_after_expiry(raw_market):
    calls, _state = raw_market
    injected = []
    with owned() as owner:
        bm._fetch_benchmark_history("COMPLETED.MI")
        before = len(calls)
        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks(
                {"complete": "COMPLETED.MI"},
                fetch_history=lambda symbol: injected.append(symbol),
            )
        assert not catalog and errors
        assert injected == []
        assert len(calls) == before


@pytest.mark.parametrize("target,name", [
    (bm, "_fetch_benchmark_history"), (bm, "_fetch_benchmark_uncached"),
    (en, "_fetch_benchmark_data"), (en, "_fetch_ticker_data"),
    (en, "_fetch_history"), (en, "convert_to_eur"), (_yf_net, "ticker"),
])
def test_injected_entry_ports_decline_completed_only_optimization(
    monkeypatch, raw_market, target, name,
):
    calls, _state = raw_market
    with owned() as owner:
        en._fetch_history("COMPLETED.MI")
        original = getattr(target, name)

        @wraps(original)
        def wrapper(*args, **kwargs):
            pytest.fail("injected acquisition entry must not run")

        monkeypatch.setattr(target, name, wrapper)
        before = len(calls)
        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks({"complete": "COMPLETED.MI"})
        assert not catalog and errors
        assert len(calls) == before


def test_unknown_sdk_declines_without_constructing_a_client(monkeypatch, raw_market):
    calls, _state = raw_market
    with owned() as owner:
        en._fetch_history("COMPLETED.MI")
        monkeypatch.setattr(yf, "__version__", "1.7.1")
        before = len(calls)
        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks({"complete": "COMPLETED.MI"})
        assert not catalog and errors
        assert len(calls) == before


def test_completed_snapshot_and_returned_series_are_defensive(raw_market):
    calls, state = raw_market
    state["currency"] = "GBp"
    with owned() as owner:
        history = en._fetch_history("COMPLETED.L")
        en._fetch_ticker_info("COMPLETED.L")
        rates(owner, "GBp", history)
        expected = bm._fetch_benchmark_uncached("COMPLETED.L")
        data, fx = en._completed_benchmark_inputs("COMPLETED.L")
        data["history"].iloc[0, 0] = -1
        data["info"]["currency"] = "JPY"
        fx.iloc[0] = -1
        before = len(calls)
        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks({"complete": "COMPLETED.L"})
            assert errors == ()
            assert_matches(catalog, expected, "COMPLETED.L")
            catalog["complete"].history.iloc[0] = -1
            catalog["complete"].history_native.iloc[0] = -1
            second, errors = bm.preprocess_benchmarks({"complete": "COMPLETED.L"})
        assert errors == ()
        assert_matches(second, expected, "COMPLETED.L")
        assert len(calls) == before


def test_completed_read_does_not_create_a_fresh_transport_client(raw_market):
    calls, _state = raw_market
    with owned() as owner:
        en._fetch_history("COMPLETED.MI")
        owner.transport.close()
        assert owner.transport._yahoo_client is None
        before = len(calls)
        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks({"complete": "COMPLETED.MI"})
        assert catalog and not errors
        assert owner.transport._yahoo_client is None
        assert len(calls) == before


def test_reset_removes_authority_to_reuse_the_old_fact(raw_market):
    calls, _state = raw_market
    with owned() as owner:
        en._fetch_history("COMPLETED.MI")
        clear_provider_stores("enricher.history")
        before = len(calls)
        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks({"complete": "COMPLETED.MI"})
        assert not catalog and errors
        assert len(calls) == before


def test_expired_consumer_does_not_join_an_uncompleted_history_flight(
    monkeypatch, raw_market,
):
    _calls, _state = raw_market
    entered, release = Event(), Event()
    with owned() as owner:
        history = en._fetch_history("COMPLETED.MI")
        clear_provider_stores("enricher.history")

        def load(_symbol):
            entered.set()
            assert release.wait(3), "test leader must be released"
            return history

        monkeypatch.setattr(en, "_fetch_history_uncached", load)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(copy_context().run, en._fetch_history, "COMPLETED.MI")
            assert entered.wait(3), "test leader must start"
            try:
                with expired(owner):
                    catalog, errors = bm.preprocess_benchmarks({"complete": "COMPLETED.MI"})
                assert not catalog and errors
                assert not future.done()
            finally:
                release.set()
            pd.testing.assert_frame_equal(future.result(), history)
        assert not owner.providers.flight._pending


def test_parallel_mixed_catalog_preserves_order_and_failed_members(raw_market):
    calls, _state = raw_market
    definitions = {
        "first": "COMPLETED.MI", "missing": "ABSENT.L", "last": "COMPLETED.DE",
    }
    with owned() as owner:
        en._fetch_history("COMPLETED.MI")
        en._fetch_history("COMPLETED.DE")
        before = len(calls)
        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks(definitions, workers=4)
        assert list(catalog) == ["first", "last"]
        assert errors == ("missing: optional provider budget exhausted",)
        assert len(calls) == before
        assert not owner.providers.flight._pending


def test_unowned_call_cannot_borrow_a_previous_runs_facts(raw_market):
    with owned():
        bm._fetch_benchmark_history("COMPLETED.MI")
    assert bm._completed_benchmark_history("COMPLETED.MI") is None
    with owned():
        assert bm._completed_benchmark_history("COMPLETED.MI") is None


@pytest.mark.parametrize("boundary", ["optional", "whole-run"])
@pytest.mark.parametrize("entry", ["default", "explicit-native", "native-engine"])
def test_actual_native_metrics_entry_consumes_completed_benchmarks(
    monkeypatch, raw_market, boundary, entry,
):
    from tarzan.engine import metrics
    from tarzan.models.investor_config import InvestorConfig

    calls, _state = raw_market
    symbol = "COMPLETEDENTRY.MI"
    with owned() as owner:
        en._fetch_history(symbol)
        expected = bm._fetch_benchmark_uncached(symbol)
        definitions = {"complete": symbol}
        if entry == "native-engine":
            definitions.update({
                en.cfg.benchmark_beta_name(): "COMPLETEBETA.MI",
                en.cfg.benchmark_geo_allocation(): "COMPLETEGEO.MI",
            })
            for required_symbol in tuple(definitions.values())[1:]:
                en._fetch_history(required_symbol)
            monkeypatch.setattr(metrics, "BENCHMARKS", definitions)
        before = len(calls)
        with expired(owner, boundary):
            if entry == "native-engine":
                context = {}
                missing = metrics.MetricsEngine(
                    [], InvestorConfig(),
                )._prepare_benchmark_context(context)
                assert missing == ()
                catalog = context["_benchmark_catalog"]
                errors = context["benchmark_resolution_errors"]
            else:
                options = (
                    {"fetch_history": metrics._fetch_benchmark_history}
                    if entry == "explicit-native" else {}
                )
                catalog, errors = bm.preprocess_benchmarks(
                    definitions, workers=2, **options,
                )
        assert errors == ()
        assert set(catalog) == set(definitions)
        assert_matches({"complete": catalog["complete"]}, expected, symbol)
        assert len(calls) == before
        assert not owner.providers.flight._pending


def test_wrapped_explicit_native_fetch_remains_uninvoked_after_expiry(raw_market):
    calls, _state = raw_market
    with owned() as owner:
        en._fetch_history("COMPLETED.MI")
        before = len(calls)

        @wraps(bm._fetch_benchmark_history)
        def wrapper(_symbol):
            pytest.fail("an injected fetch is not a completed native fact consumer")

        with expired(owner):
            catalog, errors = bm.preprocess_benchmarks(
                {"complete": "COMPLETED.MI"}, fetch_history=wrapper,
            )
        assert not catalog and errors
        assert len(calls) == before
