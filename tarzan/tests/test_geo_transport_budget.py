"""Geographic justETF scrapes must obey the same native budget as other data."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import requests

from tarzan import runtime
from tarzan.data import geo_resolver, price_cache
from tarzan.runtime.ledger import LedgerEntryType, RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings


ISIN = "IE0000000001"
PROFILE = "https://www.justetf.com/en/etf-profile.html?isin=" + ISIN
HTML = "The fund seeks to track the Synthetic World index. Total expense ratio: 0.20 %"


class Clock:
    def __init__(self):
        self.value = 0.
        self.waits = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.value += seconds


@contextmanager
def _owned(*, timeout=3., retries=1, total=20.):
    clock = Clock()
    settings = replace(
        ProviderTransportSettings.configured(), run_budget_seconds=total,
        optional_budget_seconds=2., publication_reserve_seconds=1.,
        providers=tuple(
            replace(policy, timeout_seconds=timeout, retry_budget=retries)
            if policy.provider == "justetf" else policy
            for policy in ProviderTransportSettings.configured().providers
        ),
    )
    context = RunContext("geo-native", RunMode.LIVE, None, datetime.now(timezone.utc))
    ledger = RunLedger(context.attempt_id)
    budget = ProviderBudget(settings, ledger, clock=clock.now, sleep=clock.sleep)
    session = RunSession(context, {}, ledger, transport=budget)
    with runtime.bind_context(context), activate_session(session):
        try:
            yield budget, clock
        finally:
            budget.close()


@pytest.fixture
def ter_cache(monkeypatch):
    values, writes = {}, []
    monkeypatch.setattr(price_cache, "has_ter", lambda isin: isin in values)
    monkeypatch.setattr(price_cache, "load_ter", lambda isin: values.get(isin))

    def store(isin, value):
        values[isin] = value
        writes.append((isin, value))

    monkeypatch.setattr(price_cache, "store_ter", store)
    return values, writes


@pytest.fixture(params=["justetf_index_name", "justetf_ter"])
def scrape(request, ter_cache):
    return getattr(geo_resolver, request.param)


def _answer(status=200, html=HTML, closed=None):
    return SimpleNamespace(
        status_code=status, text=html,
        close=lambda: closed.append(status) if closed is not None else None,
    )


def _expected(scrape):
    return "Synthetic World" if scrape.__name__ == "justetf_index_name" else pytest.approx(0.002)


@pytest.mark.parametrize("configured_timeout", [1.5, 30.])
def test_scrape_uses_configured_native_timeout_and_records_only_safe_fields(
    monkeypatch, scrape, configured_timeout,
):
    calls = []

    def get(url, **arguments):
        calls.append((url, arguments))
        return _answer()

    monkeypatch.setattr(requests, "get", get)
    expected_timeout = min(configured_timeout, 15.)
    with _owned(timeout=configured_timeout, retries=0, total=60.) as (budget, _):
        assert scrape(ISIN) == _expected(scrape)
    assert calls == [(PROFILE, {
        "headers": {"User-Agent": "Mozilla/5.0"}, "timeout": expected_timeout,
    })]
    attempts = [
        entry.payload for entry in budget.ledger.entries
        if entry.entry_type is LedgerEntryType.PROVIDER_ATTEMPT
    ]
    assert len(attempts) == 1
    assert attempts[0]["source"] == "justetf"
    assert attempts[0]["operation"] == "native_http"
    assert attempts[0]["outcome"] == "HTTP_200"
    assert attempts[0]["timeout_seconds"] == expected_timeout
    assert all(ISIN not in str(entry.to_dict()) and PROFILE not in str(entry.to_dict())
               for entry in budget.ledger.entries)


def test_remaining_run_deadline_caps_the_native_scrape(monkeypatch, scrape):
    caps = []
    monkeypatch.setattr(
        requests, "get",
        lambda url, **arguments: caps.append(arguments["timeout"]) or _answer(),
    )
    with _owned(total=0.5) as (_, clock):
        clock.value = 0.3
        assert scrape(ISIN) == _expected(scrape)
    assert caps == pytest.approx([0.2])


@pytest.mark.parametrize("optional", [False, True])
def test_exhausted_budget_starts_no_request_and_does_not_cache_a_ter_miss(
    monkeypatch, scrape, ter_cache, optional,
):
    monkeypatch.setattr(requests, "get", lambda *a, **kw: pytest.fail("unbudgeted I/O"))
    with _owned() as (budget, clock):
        if optional:
            with budget.optional_work("synthetic-geography"):
                clock.value = 2.1
                assert scrape(ISIN) is None
        else:
            clock.value = 20.
            assert scrape(ISIN) is None
        codes = [failure.stable_code for failure in budget.ledger.failure_records()]
    assert codes == [
        "OPTIONAL_PROVIDER_BUDGET_EXHAUSTED" if optional else "PROVIDER_BUDGET_EXHAUSTED",
    ]
    assert ter_cache[1] == []


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_statuses_retry_under_one_deadline_then_preserve_the_parser(
    monkeypatch, scrape, status,
):
    caps, closed = [], []
    with _owned() as (budget, clock):
        def get(url, **arguments):
            caps.append(arguments["timeout"])
            clock.value += 0.1
            return _answer(status if len(caps) == 1 else 200, closed=closed)

        monkeypatch.setattr(requests, "get", get)
        assert scrape(ISIN) == _expected(scrape)
    assert caps == pytest.approx([3., 2.65])
    assert closed == [status]
    assert clock.waits == [0.25]
    attempts = [
        entry.payload for entry in budget.ledger.entries
        if entry.entry_type is LedgerEntryType.PROVIDER_ATTEMPT
    ]
    assert [attempt["outcome"] for attempt in attempts] == [f"HTTP_{status}", "HTTP_200"]
    assert [attempt["ordinal"] for attempt in attempts] == [1, 2]


@pytest.mark.parametrize("failure", ["timeout", "http"])
def test_exhausted_retries_cannot_poison_the_persistent_ter_cache(
    monkeypatch, scrape, ter_cache, failure,
):
    calls = []

    def get(url, **arguments):
        calls.append(arguments["timeout"])
        if failure == "timeout":
            raise requests.Timeout("synthetic private request")
        return _answer(503)

    monkeypatch.setattr(requests, "get", get)
    with _owned() as (budget, _):
        assert scrape(ISIN) is None
    assert len(calls) == 2
    assert ter_cache[1] == []
    assert [failure.stable_code for failure in budget.ledger.failure_records()] == [
        "PROVIDER_RETRY_BUDGET_EXHAUSTED",
    ]


def test_a_real_not_found_response_is_not_retried(monkeypatch, scrape):
    calls = []
    monkeypatch.setattr(requests, "get", lambda *a, **kw: calls.append(1) or _answer(404))
    with _owned() as (budget, clock):
        assert scrape(ISIN) is None
    assert calls == [1]
    assert clock.waits == []
    assert budget.ledger.failure_records() == ()


def test_standalone_scrape_keeps_its_existing_15_second_timeout(monkeypatch, scrape):
    caps = []
    monkeypatch.setattr(
        requests, "get",
        lambda url, **arguments: caps.append(arguments["timeout"]) or _answer(),
    )
    assert scrape(ISIN) == _expected(scrape)
    assert caps == [15]


def test_empty_isin_starts_no_transport(monkeypatch, scrape):
    monkeypatch.setattr(requests, "get", lambda *a, **kw: pytest.fail("empty ISIN request"))
    with _owned() as (budget, _):
        assert scrape("") is None
    assert budget.ledger.entries == ()


def test_pinned_scrape_never_opens_a_live_transport(monkeypatch, scrape, ter_cache):
    monkeypatch.setattr(runtime, "allows_live_transport", lambda: False)
    monkeypatch.setattr(requests, "get", lambda *a, **kw: pytest.fail("pinned request"))
    with _owned() as (budget, _):
        assert scrape(ISIN) is None
    assert budget.ledger.entries == ()
    assert ter_cache[1] == []


@pytest.mark.parametrize("value", [0.002, None])
def test_cached_ter_hit_or_miss_does_not_need_new_network_authority(
    monkeypatch, ter_cache, value,
):
    ter_cache[0][ISIN] = value
    monkeypatch.setattr(requests, "get", lambda *a, **kw: pytest.fail("cached TER request"))
    with _owned() as (budget, clock):
        clock.value = 20.
        assert geo_resolver.justetf_ter(ISIN) == value
    assert budget.ledger.entries == ()
    assert ter_cache[1] == []


def test_confirmed_not_found_ter_absence_remains_a_cached_miss(monkeypatch, ter_cache):
    calls = []
    monkeypatch.setattr(
        requests, "get", lambda *a, **kw: calls.append(1) or _answer(404, "No TER published"),
    )
    with _owned():
        assert geo_resolver.justetf_ter(ISIN) is None
        assert geo_resolver.justetf_ter(ISIN) is None
    assert calls == [1]
    assert ter_cache[1] == [(ISIN, None)]


@pytest.mark.parametrize(
    "status,body",
    [(200, "No TER published"), (200, "Verify you are human"),
     (200, "Total expense ratio: 99.00 %"), (401, HTML), (403, HTML)],
)
def test_an_ambiguous_profile_or_error_cannot_cache_an_absent_ter(
    monkeypatch, ter_cache, status, body,
):
    calls = []
    monkeypatch.setattr(
        requests, "get", lambda *a, **kw: calls.append(1) or _answer(status, body),
    )
    with _owned():
        assert geo_resolver.justetf_ter(ISIN) is None
        assert geo_resolver.justetf_ter(ISIN) is None
    assert calls == [1, 1]
    assert ter_cache[1] == []


def test_transport_failure_keeps_default_ter_available_without_storing_false_absence(
    monkeypatch, ter_cache,
):
    monkeypatch.setattr(
        requests, "get", lambda *a, **kw: pytest.fail("request after deadline"),
    )
    with _owned() as (_, clock):
        clock.value = 20.
        assert geo_resolver.resolve_ter(ISIN, "Fixed Income") == pytest.approx(0.0015)
    assert ter_cache[1] == []
