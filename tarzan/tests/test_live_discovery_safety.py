"""Recovered sources must remain discoverable; caches are not listing authority.

Synthetic providers only. Exercise real discovery, owned memos, disk caches,
sealed snapshots, deadlines and bounded concurrent workloads.
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
from threading import Lock
from types import SimpleNamespace
import time

from hypothesis import HealthCheck, given, settings, strategies as st
import pandas as pd
import pytest

from tarzan import runtime
from tarzan.data import bond_history as bonds, enricher as en, price_cache, protected_cache
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session
from tarzan.runtime import transport


ROOT, MI, DE = "SYNRECOVER", "SYNRECOVER.MI", "SYNRECOVER.DE"
NAME = "Synthetic Recovery Venue Fund"
ISIN = "IT0000000001"
CAPTURED = datetime(2026, 10, 22, 13, tzinfo=timezone.utc)
START, END = date(2026, 1, 5), date(2026, 1, 9)
ABSENCE_NAMESPACE = "yahoo_info_not_found_v1"  # old unpublished schema


def frame(price=100., factor=1.):
    values = [price - 2., price - 1., price]
    return pd.DataFrame(
        {"Close": values, "Adj Close": [v * factor for v in values]},
        index=pd.to_datetime(["2026-10-20", "2026-10-21", "2026-10-22"]),
    )


def metadata(price=100., **overrides):
    return {"regularMarketPrice": price, "longName": NAME, "currency": "EUR", **overrides}


def seed_absence(symbol):
    price_cache.store_timed(ABSENCE_NAMESPACE, symbol, {
        "source": "yahoo", "status": 404, "operation": "quoteSummary", "symbol": symbol,
    })


@contextmanager
def owned(name="recovery", *, budget=None, mode=RunMode.LIVE, captured_at=CAPTURED):
    context = RunContext(
        name, mode, None if mode is RunMode.LIVE else captured_at.date(), captured_at,
    )
    session = RunSession(context, {}, RunLedger(name), transport=budget)
    with runtime.bind_context(context), activate_session(session):
        try:
            yield session
        finally:
            session.close_transport()


def receipt_clock(monkeypatch, completed_at):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return (
                completed_at.astimezone(tz) if tz is not None
                else completed_at.replace(tzinfo=None)
            )

    monkeypatch.setattr(bonds, "datetime", Clock)


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    monkeypatch.setattr(en.cfg, "instrument_taxonomy_has", lambda symbol: False)
    monkeypatch.setattr(en, "ISIN_EXCHANGE_SUFFIXES", [".MI", ".DE", ".PA"])
    monkeypatch.setattr(en, "_space_yf_call", lambda: None)
    monkeypatch.setattr(en, "_fill_today_from_sibling", lambda symbol, value: value)
    state = SimpleNamespace(
        info={}, histories={}, info_calls=[], history_calls=[], delay=0.,
        active=0, peak=0,
    )
    lock = Lock()

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def info(self):
            with lock:
                state.info_calls.append(self.symbol)
            value = state.info.get(self.symbol, {})
            return deepcopy(value() if callable(value) else value)

        def history(self, **arguments):
            with lock:
                state.history_calls.append((self.symbol, dict(arguments)))
                state.active += 1
                state.peak = max(state.peak, state.active)
            try:
                if state.delay:
                    time.sleep(state.delay)
                value = state.histories.get(self.symbol, pd.DataFrame())
                return deepcopy(value() if callable(value) else value)
            finally:
                with lock:
                    state.active -= 1

    monkeypatch.setattr(en._yf_net, "ticker", Ticker)
    return state


def resolve(root=ROOT):
    return en._resolve_bare_ticker(root, expected_name=NAME, expected_currency="EUR")


def selected(value):
    return None if value is None else (
        value[en._TICKER_SYMBOL_KEY], value["history"]["Price Close"].iloc[-1],
    )


def test_primary_sole_venue_recovers_before_old_absence_expires(source):
    seed_absence(MI)
    source.info[MI], source.histories[MI] = metadata(), frame()
    with owned():
        assert selected(resolve()) == (MI, 100.)
    assert MI in source.info_calls
    assert [symbol for symbol, _ in source.history_calls] == [MI]


def test_01_preferred_recovered_venue_wins_even_with_healthy_secondary(source):
    seed_absence(MI)
    source.info = {MI: metadata(), DE: metadata(102.)}
    source.histories = {MI: frame(), DE: frame(102.)}
    with owned():
        assert selected(resolve()) == (MI, 100.)
    assert source.info_calls.count(MI) == source.info_calls.count(DE) == 1
    assert [symbol for symbol, _ in source.history_calls] == [MI]


def test_02_missing_first_venue_cannot_exclude_healthy_later_or_other_root(source):
    seed_absence(MI)
    source.info = {DE: metadata(102.), "OTHERREC.MI": metadata(103.)}
    source.histories = {DE: frame(102.), "OTHERREC.MI": frame(103.)}
    with owned():
        assert selected(resolve()) == (DE, 102.)
        assert selected(resolve("OTHERREC")) == ("OTHERREC.MI", 103.)
    assert MI in source.info_calls
    assert {symbol for symbol, _ in source.history_calls} == {DE, "OTHERREC.MI"}


def test_03_legacy_chart_absence_does_not_exclude_fresh_sibling_prices(source):
    price_cache.store_timed("no_listing", MI, True)
    source.histories[MI] = frame()
    with owned():
        assert en._sibling_close_series(MI).iloc[-1] == 100.
    assert [symbol for symbol, _ in source.history_calls] == [MI]


def test_04_legacy_isin_absence_does_not_exclude_all_bond_venues(source, monkeypatch):
    price_cache.store_timed("no_listing", ISIN, True)
    monkeypatch.setattr(price_cache, "load_instrument_profile", lambda *a, **k: {
        "status": "VERIFIED", "kind": "BOND",
    })
    monkeypatch.setattr(en, "_openfigi_name", lambda isin: NAME)
    monkeypatch.setattr(en, "_openfigi_lookup", lambda isin: [])
    monkeypatch.setattr(en, "_resolution_candidate_symbols", lambda *a: [MI, DE])
    source.info[MI], source.histories[MI] = metadata(), frame()
    with owned():
        result = en._resolve_isin(ISIN, "")
        assert result is not None
        assert result[1] == MI
        assert selected(result[0]) == (MI, 100.)


def bond_packet(*, unavailable=False):
    return {
        "exchange": "XMIL", "fetched_at": "2026-10-22T12:40:00+00:00",
        "payload": {
            "status": 2304 if unavailable else 0, "entityID": "euronext",
            "period": "5Y", "adjustement": False, "addDayLastPrice": False,
            "transco": None if unavailable else {
                "code": ISIN, "codification": "ISIN", "exchCode": "XMIL",
            },
            "history": {"currency": "EUR", "instrType": "BOND", "historyDt": [] if unavailable else [
                {"dt": f"2026010{day}", "closePx": 93 + day} for day in range(5, 10)
            ]},
        },
    }


def authorize_bond(monkeypatch):
    monkeypatch.setattr(price_cache, "load_instrument_profile", lambda *a, **k: {
        "isin": ISIN, "kind": "BOND", "status": "VERIFIED", "confidence": "HIGH",
        "source": "OpenFIGI", "observed_at": "2026-10-21T12:00:00+00:00",
        "provenance": {"provider": "OpenFIGI", "idType": "ID_ISIN"},
    })


def test_05_native_bond_source_recovers_inside_former_24_hour_exclusion(source, monkeypatch):
    authorize_bond(monkeypatch)
    old = bonds.parse_packet(bond_packet(unavailable=True), ISIN)
    price_cache.store_timed(bonds.NAMESPACE, ISIN, old)
    calls = []
    monkeypatch.setattr(bonds, "_request_raw", lambda isin: calls.append(isin) or bond_packet())
    with owned():
        fresh = bonds.raw_history(ISIN, START, END)
        assert fresh["state"] == "AVAILABLE"
        assert bonds.eur_per_nominal(fresh, as_of=END).iloc[-1] == 1.02
    assert calls == [ISIN]


def test_06_curated_identity_and_qualified_prices_survive_missing_metadata(source, monkeypatch):
    seed_absence(MI)
    source.histories[MI] = frame()
    monkeypatch.setattr(en.cfg, "instrument_taxonomy_has", lambda symbol: symbol == MI)
    with owned():
        assert en._fetch_candidate_meta(MI).price == 100.
        assert selected(en._fetch_ticker_data(MI)) == (MI, 100.)
    assert source.info_calls.count(MI) == 1


@pytest.mark.parametrize("bad", [
    None, False, True, 0, -1, float("nan"), float("inf"), float("-inf"),
    [], {}, "invalid", 10 ** 400,
])
def test_07_invalid_primary_quote_cannot_mask_valid_previous_close(source, bad):
    source.info[MI] = metadata(bad, previousClose=100.)
    with owned():
        candidate = en._fetch_candidate_meta(MI)
    assert candidate is not None and candidate.price == 100.


@pytest.mark.parametrize("bad", [None, False, True, [], ["bad"], "bad", 42])
def test_07_malformed_sdk_metadata_cannot_abort_other_venue_discovery(source, bad):
    source.info = {MI: bad, DE: metadata(102.)}
    source.histories[DE] = frame(102.)
    with owned():
        assert selected(resolve()) == (DE, 102.)


@pytest.mark.parametrize("field", ["longName", "shortName", "name", "currency"])
@pytest.mark.parametrize("bad", [True, 42, ["invalid"], {"invalid": "value"}])
def test_07_malformed_identity_fields_fail_closed_without_aborting_search(source, field, bad):
    source.info = {MI: {"regularMarketPrice": 100., field: bad}, DE: metadata(102.)}
    source.histories[DE] = frame(102.)
    with owned():
        assert selected(resolve()) == (DE, 102.)


def test_08_identity_collision_and_currency_remain_part_of_venue_ranking(source):
    source.info = {
        MI: metadata(100., longName="Unrelated Biotechnology Corporation"),
        DE: metadata(102.),
        ROOT + ".PA": metadata(103., currency="USD"),
    }
    source.histories = {MI: frame(), DE: frame(102.), ROOT + ".PA": frame(103.)}
    with owned():
        assert selected(resolve()) == (DE, 102.)
    assert [symbol for symbol, _ in source.history_calls] == [DE]


def test_09_sealed_legacy_cache_restores_prices_but_does_not_block_recovery(
    source, monkeypatch, tmp_path,
):
    price_cache.store_timed("no_listing", MI, True)
    price_cache.store_history("POSITIVE.DE", frame())
    initial = price_cache._timed_path("no_listing").read_bytes()
    staging = tmp_path / "sealed"
    key = b"synthetic-key-is-not-a-secret-00"
    assert protected_cache.export_snapshot(staging, key) == 2
    assert MI.encode() not in (staging / protected_cache.FILENAME).read_bytes()
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "restored"))
    assert protected_cache.import_snapshot(staging, key) == 2
    assert price_cache._timed_path("no_listing").read_bytes() == initial
    assert price_cache.load_history("POSITIVE.DE")["Close"].iloc[-1] == 100.
    source.histories[MI] = frame()
    with owned():
        assert en._sibling_close_series(MI).iloc[-1] == 100.


def test_10_concurrent_exact_requests_share_one_observation_not_mutable_results(source):
    source.histories[MI], source.delay = frame(factor=.98), .025
    with owned(), ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda context: context.run(en._download_price_history, MI),
            [copy_context() for _ in range(32)],
        ))
    assert len(source.history_calls) == 1
    assert source.peak == 1
    results[0].iloc[-1, results[0].columns.get_loc("Close")] = 777.
    assert all(result["Close"].iloc[-1] == 98. for result in results[1:])
    assert all(result["Price Close"].iloc[-1] == 100. for result in results)


def test_11_new_runs_reobserve_the_same_symbol_and_do_not_share_old_misses(source):
    with owned("first"):
        assert en._fetch_candidate_meta(MI) is None
    source.info[MI], source.histories[MI] = metadata(), frame()
    with owned("second"):
        assert selected(resolve()) == (MI, 100.)
    assert source.info_calls.count(MI) == 2
    assert price_cache.load_timed(ABSENCE_NAMESPACE, MI, 1) is None


@pytest.mark.parametrize("miss", [pd.DataFrame(), pd.DataFrame({"Close": [float("nan")]})])
def test_12_unusable_raw_history_is_retried_when_the_source_recovers(source, miss):
    answers = iter([miss, frame()])
    source.histories[MI] = lambda: next(answers)
    with owned():
        assert en._sibling_close_series(MI) is None
        assert en._sibling_close_series(MI).iloc[-1] == 100.
    assert len(source.history_calls) == 2
    assert price_cache.load_timed("no_listing", MI, 7) is None


def test_12_native_bond_transport_failure_is_not_a_persistent_or_run_owned_absence(
    source, monkeypatch,
):
    authorize_bond(monkeypatch)
    calls = []

    def request(isin):
        calls.append(isin)
        if len(calls) == 1:
            raise TimeoutError("synthetic native failure")
        return bond_packet()

    monkeypatch.setattr(bonds, "_request_raw", request)
    with owned():
        assert bonds.raw_history(ISIN, START, END) is None
        assert bonds.raw_history(ISIN, START, END)["state"] == "AVAILABLE"
    assert calls == [ISIN, ISIN]


def test_13_exhausted_budget_cannot_start_new_discovery(source, monkeypatch):
    authorize_bond(monkeypatch)
    ticks = [0.]
    budget = transport.ProviderBudget(
        transport.ProviderTransportSettings.configured(),
        clock=lambda: ticks[0], sleep=lambda seconds: None,
    )
    monkeypatch.setattr(bonds, "_request_raw", lambda *a: pytest.fail("I/O after deadline"))
    ticks[0] = budget.deadline + 1
    with owned(budget=budget):
        assert en._fetch_candidate_meta(MI) is None
        assert bonds.raw_history(ISIN, START, END) is None
    assert source.info_calls == source.history_calls == []


def test_13_pinned_context_cannot_probe_even_with_old_absence_and_recovered_provider(source):
    seed_absence(MI)
    source.info[MI], source.histories[MI] = metadata(), frame()
    with owned(mode=RunMode.POINT_IN_TIME):
        assert resolve() is None
    assert source.info_calls == source.history_calls == []


def test_14_force_refresh_bypasses_an_old_raw_observation_on_adjustment_revision(source):
    source.histories[MI] = frame()
    with owned():
        old = en._download_price_history(MI)
        source.histories[MI] = frame(factor=.5)
        tail = frame(factor=.5).rename(columns={"Close": "Price Close", "Adj Close": "Close"})
        cached, refreshed = en._coherent_price_inputs(MI, old, tail, "2026-10-20")
        assert cached is None
        assert refreshed["Close"].iloc[-1] == 50.
        assert refreshed["Price Close"].iloc[-1] == 100.
        # The prior observation remains a separate fact, not mutated in place.
        assert old["Close"].iloc[-1] == 100.
    assert len(source.history_calls) == 2


@settings(
    max_examples=75, deadline=None, derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    available=st.lists(st.booleans(), min_size=3, max_size=3),
    absent=st.lists(st.booleans(), min_size=3, max_size=3),
    price=st.integers(min_value=10, max_value=10000),
)
def test_15_generative_same_provider_facts_choose_same_venue_with_or_without_absences(
    source, monkeypatch, tmp_path, available, absent, price,
):
    symbols = [MI, DE, ROOT + ".PA"]
    source.info = {
        symbol: metadata(float(price + i)) for i, symbol in enumerate(symbols) if available[i]
    }
    source.histories = {
        symbol: frame(float(price + i)) for i, symbol in enumerate(symbols) if available[i]
    }
    # Each example starts from a new logical cache; no learned positive winner.
    case = "".join(str(int(value)) for value in [*available, *absent])
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / f"case-{case}-{price}"))
    monkeypatch.setattr(price_cache, "load_resolution", lambda key: None)
    with monkeypatch.context() as patch:
        patch.setenv("TARZAN_DISABLE_CACHE", "1")
        with owned("without"):
            baseline = selected(resolve())
    for i, symbol in enumerate(symbols):
        if absent[i]:
            seed_absence(symbol)
    with owned("with"):
        assert selected(resolve()) == baseline


def test_16_scaled_workload_has_bounded_native_calls_and_workers(source):
    symbols = [f"SYNLOAD{i}.MI" for i in range(32)]
    source.histories = {symbol: frame(100. + i) for i, symbol in enumerate(symbols)}
    source.delay = .002
    with owned(), ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(copy_context().run, en._download_price_history, symbols[i % 32])
            for i in range(1024)
        ]
        results = [future.result() for future in futures]
    counts = Counter(symbol for symbol, _ in source.history_calls)
    assert counts == {symbol: 1 for symbol in symbols}
    assert source.peak <= 4
    for i, result in enumerate(results):
        assert result["Price Close"].iloc[-1] == 100. + i % 32
    results[0].iloc[-1, results[0].columns.get_loc("Price Close")] = 999.
    assert results[32]["Price Close"].iloc[-1] == 100.


def test_bonus_native_packet_is_reused_across_windows_without_persisting_absences(
    source, monkeypatch,
):
    authorize_bond(monkeypatch)
    calls = []
    monkeypatch.setattr(
        bonds, "_request_raw", lambda isin: calls.append(isin) or bond_packet(unavailable=True),
    )
    with owned("unavailable"):
        assert bonds.raw_history(ISIN, START, END) is None
        assert bonds.raw_history(ISIN, START, date(2026, 1, 12)) is None
    assert calls == [ISIN]
    assert price_cache.load_timed(bonds.NAMESPACE, ISIN, 36525) is None
    monkeypatch.setattr(bonds, "_request_raw", lambda isin: calls.append(isin) or bond_packet())
    with owned("recovered"):
        assert bonds.raw_history(ISIN, START, END)["state"] == "AVAILABLE"
    assert calls == [ISIN, ISIN]


@pytest.mark.parametrize("unavailable", [False, True])
def test_bonus_future_native_receipt_cannot_postdate_its_acquisition_or_enter_disk_cache(
    source, monkeypatch, unavailable,
):
    authorize_bond(monkeypatch)
    completed_at = CAPTURED + timedelta(seconds=10)
    receipt_clock(monkeypatch, completed_at)
    raw = bond_packet(unavailable=unavailable)
    raw["fetched_at"] = (completed_at + timedelta(microseconds=1)).isoformat()
    monkeypatch.setattr(bonds, "_request_raw", lambda isin: deepcopy(raw))
    with owned():
        assert bonds.raw_history(ISIN, START, END) is None
    assert price_cache.load_timed(bonds.NAMESPACE, ISIN, 36525) is None


@pytest.mark.parametrize("delay", [
    timedelta(microseconds=1), timedelta(seconds=5), timedelta(minutes=3),
])
@pytest.mark.parametrize("unavailable", [False, True])
def test_bonus_native_receipt_after_run_start_is_valid_and_shared_without_backdating(
    source, monkeypatch, delay, unavailable,
):
    authorize_bond(monkeypatch)
    fetched_at = CAPTURED + delay
    receipt_clock(monkeypatch, fetched_at)
    raw = bond_packet(unavailable=unavailable)
    raw["fetched_at"] = fetched_at.isoformat()
    calls = []
    monkeypatch.setattr(
        bonds, "_request_raw", lambda isin: calls.append(isin) or deepcopy(raw),
    )
    with owned():
        first = bonds.raw_history(ISIN, START, END)
        second = bonds.raw_history(ISIN, START, date(2026, 2, 1))
    assert calls == [ISIN]
    stored = price_cache.load_timed(bonds.NAMESPACE, ISIN, 36525)
    if unavailable:
        assert first is second is stored is None
    else:
        assert first == second == stored
        assert first["state"] == "AVAILABLE"
        assert first["fetched_at"] == fetched_at.isoformat()


def test_bonus_real_native_port_stamps_a_post_start_receipt_without_losing_closed_prices(
    source, monkeypatch,
):
    authorize_bond(monkeypatch)
    completed_at = CAPTURED + timedelta(seconds=15)
    receipt_clock(monkeypatch, completed_at)
    reads = []
    view = (
        f'<chart-allinone code="{ISIN}" exchcode="XMIL" '
        'url="https://grafici.borsaitaliana.it/api/" '
        'token="synthetic-public-view"></chart-allinone>'
    )

    def read(opener, url, **kwargs):
        reads.append("view" if "interactive-chart/" in url else "history")
        return (
            view.encode() if reads[-1] == "view"
            else json.dumps(bond_packet()["payload"]).encode()
        )

    monkeypatch.setattr(bonds, "_read_body", read)
    with owned():
        result = bonds.raw_history(ISIN, START, END)
    assert reads == ["view", "history"]
    assert result is not None
    assert result["prices"][-1]["date"] == "2026-01-09"
    assert result["fetched_at"] == completed_at.isoformat()


@pytest.mark.parametrize("captured_at,expected_last", [
    (datetime(2026, 10, 22, 21, 59, 59, tzinfo=timezone.utc), "2026-10-21"),
    (datetime(2026, 10, 22, 22, 0, 0, tzinfo=timezone.utc), "2026-10-22"),
])
def test_bonus_native_receipt_midnight_rollover_does_not_move_the_closed_market_cutoff(
    source, monkeypatch, captured_at, expected_last,
):
    authorize_bond(monkeypatch)
    fetched_at = captured_at + timedelta(seconds=10)
    receipt_clock(monkeypatch, fetched_at)
    raw = bond_packet()
    raw["fetched_at"] = fetched_at.astimezone(timezone(timedelta(hours=2))).isoformat()
    raw["payload"]["history"]["historyDt"].extend([
        {"dt": "20261021", "closePx": 101},
        {"dt": "20261022", "closePx": 102},
        {"dt": "20261023", "closePx": 103},
    ])
    monkeypatch.setattr(bonds, "_request_raw", lambda isin: deepcopy(raw))
    with owned(captured_at=captured_at):
        result = bonds.raw_history(ISIN, START, date(2026, 10, 23))
    assert result is not None
    assert result["prices"][-1]["date"] == expected_last
    assert result["fetched_at"] == fetched_at.isoformat()


def test_bonus_native_bond_packet_coalesces_concurrent_different_windows(source, monkeypatch):
    authorize_bond(monkeypatch)
    calls = []
    lock = Lock()

    def request(isin):
        with lock:
            calls.append(isin)
        time.sleep(.025)
        return bond_packet()

    monkeypatch.setattr(bonds, "_request_raw", request)
    with owned(), ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                copy_context().run, bonds.raw_history, ISIN, START, date(2026, 2, i + 1),
            )
            for i in range(20)
        ]
        results = [future.result(timeout=5) for future in futures]
    assert calls == [ISIN]
    assert all(result["state"] == "AVAILABLE" for result in results)
    results[0]["prices"][0]["close"] = 777.
    assert all(result["prices"][0]["close"] == 98. for result in results[1:])


def test_bonus_acquired_bond_packet_remains_usable_without_io_after_deadline(source, monkeypatch):
    authorize_bond(monkeypatch)
    ticks, calls = [0.], []
    budget = transport.ProviderBudget(
        transport.ProviderTransportSettings.configured(),
        clock=lambda: ticks[0], sleep=lambda seconds: None,
    )
    monkeypatch.setattr(bonds, "_request_raw", lambda isin: calls.append(isin) or bond_packet())
    with owned(budget=budget):
        first = bonds.raw_history(ISIN, START, END)
        ticks[0] = budget.deadline + 1
        later = bonds.raw_history(ISIN, START, date(2026, 1, 12))
    assert later == first
    assert calls == [ISIN]


def test_bonus_native_metadata_retry_count_and_timeout_stay_bounded(source):
    ticks, native = [0.], []
    budget = transport.ProviderBudget(
        transport.ProviderTransportSettings.configured(),
        clock=lambda: ticks[0],
        sleep=lambda seconds: ticks.__setitem__(0, ticks[0] + seconds),
    )

    def request(timeout):
        native.append(timeout)
        if len(native) == 1:
            raise TimeoutError("synthetic native timeout")
        return metadata()

    source.info[MI] = lambda: transport.provider_call("yahoo", request, timeout=3.)
    with owned(budget=budget):
        for _ in range(100):
            assert en._fetch_candidate_meta(MI).price == 100.
    assert native == [3., 3.]
    assert source.info_calls == [MI]


@pytest.mark.parametrize("bad", [True, False, "invalid", float("nan"), float("inf"), 10 ** 400])
def test_bonus_bad_quotes_without_valid_fallback_do_not_become_candidate_prices(source, bad):
    source.info[MI] = metadata(bad)
    with owned():
        assert en._fetch_candidate_meta(MI) is None


def test_bonus_numeric_quote_text_and_valid_secondary_name_are_accepted(source):
    source.info[MI] = metadata("100", longName=["invalid"], shortName=NAME)
    source.histories[MI] = frame()
    with owned():
        assert selected(resolve()) == (MI, 100.)


@pytest.mark.parametrize("group", [
    True, "invalid", {"data": None}, {"data": "invalid"}, {"data": {"ticker": ROOT}},
    {"data": [True]}, {"data": [{"ticker": [ROOT]}]},
    {"data": [{"name": [NAME]}]}, {"data": [{"exchCode": {"invalid": "value"}}]},
    {"data": [{"micCode": ["XMIL"]}]}, {"data": [{"securityType": 42}]},
    {"data": [{"marketSector": True}]}, {"data": [{"shareClassFIGI": ["invalid"]}]},
])
@pytest.mark.parametrize("legacy_cached", [False, True])
def test_bonus_malformed_figi_mapping_is_not_reused_as_identity_across_runs(
    source, monkeypatch, group, legacy_cached,
):
    if legacy_cached:
        price_cache.store_timed("openfigi", ISIN, [group])
    original_retry = en._retry
    calls = []
    valid = {"data": [{"ticker": ROOT, "name": NAME, "micCode": "XMIL"}]}

    def retry(fn, *, what, provider="yahoo"):
        if provider != "openfigi":
            return original_retry(fn, what=what, provider=provider)
        calls.append(provider)
        return [valid if legacy_cached or len(calls) > 1 else group]

    monkeypatch.setattr(en, "_retry", retry)
    with owned("bad-figi"):
        if legacy_cached:
            assert en._openfigi_raw(ISIN) == [valid]
        else:
            assert en._openfigi_raw(ISIN) == []
            assert price_cache.load_timed("openfigi", ISIN, 30) is None
    with owned("recovered-figi"):
        assert en._openfigi_name(ISIN) == NAME
        assert en._openfigi_lookup(ISIN)[0] == MI
    assert len(calls) == (1 if legacy_cached else 2)


@pytest.mark.parametrize("field", ["ticker", "name", "figi", "micCode"])
@pytest.mark.parametrize("legacy_cached", [False, True])
def test_bonus_blank_figi_fields_never_become_thirty_day_identity_authority(
    source, monkeypatch, field, legacy_cached,
):
    group = {"data": [{field: " \t\n "}]}
    if legacy_cached:
        price_cache.store_timed("openfigi", ISIN, [group])
    original_retry = en._retry
    calls = []
    valid = {"data": [{"ticker": ROOT, "name": NAME, "micCode": "XMIL"}]}

    def retry(fn, *, what, provider="yahoo"):
        if provider != "openfigi":
            return original_retry(fn, what=what, provider=provider)
        calls.append(provider)
        return [valid if legacy_cached or len(calls) > 1 else group]

    monkeypatch.setattr(en, "_retry", retry)
    with owned("blank-figi"):
        assert en._openfigi_raw(ISIN) == ([valid] if legacy_cached else [group])
        if not legacy_cached:
            assert price_cache.load_timed("openfigi", ISIN, 30) is None
    with owned("recovered-figi"):
        assert en._openfigi_name(ISIN) == NAME
        assert en._openfigi_lookup(ISIN)[0] == MI
    assert len(calls) == (1 if legacy_cached else 2)


def test_bonus_neutral_name_score_is_not_positive_isin_identity_evidence(source, monkeypatch):
    monkeypatch.setattr(en, "_openfigi_name", lambda isin: "")
    monkeypatch.setattr(en, "_openfigi_lookup", lambda isin: [])
    monkeypatch.setattr(en, "_openfigi_raw", lambda isin: [])
    monkeypatch.setattr(en, "_resolution_candidate_symbols", lambda *a: [MI])
    source.info[MI], source.histories[MI] = metadata(currency="USD"), frame()
    with owned():
        assert en._resolve_isin(ISIN, "", expected_currency="EUR") is None
    assert source.history_calls == []  # never price an unidentified security


def test_bonus_missing_name_preserves_real_currency_corroboration(source, monkeypatch):
    monkeypatch.setattr(en, "_openfigi_name", lambda isin: "")
    monkeypatch.setattr(en, "_openfigi_lookup", lambda isin: [])
    monkeypatch.setattr(en, "_resolution_candidate_symbols", lambda *a: [MI])
    source.info[MI], source.histories[MI] = metadata(), frame()
    with owned():
        result = en._resolve_isin(ISIN, "", expected_currency="EUR")
    assert result is not None and result[1] == MI


def test_bonus_exact_isin_to_venue_mapping_is_identity_even_without_name_or_currency(
    source, monkeypatch,
):
    monkeypatch.setattr(en, "_openfigi_name", lambda isin: "")
    monkeypatch.setattr(en, "_openfigi_lookup", lambda isin: [MI])
    monkeypatch.setattr(en, "_openfigi_raw", lambda isin: [{
        "data": [{"ticker": ROOT, "micCode": "XMIL"}],
    }])
    monkeypatch.setattr(en, "_resolution_candidate_symbols", lambda *a: [MI])
    source.info[MI], source.histories[MI] = metadata(currency="USD"), frame()
    with owned():
        # Expected settlement currency alone is not instrument identity.
        result = en._resolve_isin(ISIN, "", expected_currency="EUR")
    assert result is not None and result[1] == MI
