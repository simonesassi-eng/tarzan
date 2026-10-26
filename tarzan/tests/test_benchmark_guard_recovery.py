"""A failed capability probe must retain ordinary native benchmark recovery."""

from urllib.parse import urlsplit

from curl_cffi.requests.exceptions import HTTPError, Timeout
import pytest
from yfinance import cache as sdk_cache
from yfinance.data import YfData
from yfinance.exceptions import YFDataException, YFRateLimitError

from tarzan.data import enricher as en
from tarzan.runtime.transport import ProviderBudgetExceeded, ProviderRequestFailure
from tarzan.tests.test_native_isin_listing_identity import (
    ISIN, install_raw_source, owned,
)
from tarzan.verification.tape import UnrecordedInput


def first_search_error(monkeypatch, error, *, listing="SYNRECOVERY.L", currency="USD"):
    """Keep Ticker/Search/Quote/history native; substitute raw input only."""
    sdk_cache.get_isin_cache().store(ISIN, None)
    YfData.cache_get.cache_clear()
    calls = install_raw_source(monkeypatch, listing=listing, currency=currency)
    original = YfData.get
    searches = []

    def get(instance, url, *arguments, **keywords):
        if urlsplit(url).path == "/v1/finance/search":
            searches.append(1)
            if len(searches) == 1:
                raise error
        return original(instance, url, *arguments, **keywords)

    monkeypatch.setattr(YfData, "get", get)
    return calls, searches


@pytest.mark.parametrize("listing,currency", [
    ("SYNRECOVERY.MI", "EUR"), ("SYNRECOVERY.L", "USD"),
])
@pytest.mark.parametrize("error", [
    YFDataException("synthetic source unavailable"),
    YFRateLimitError(),
    ProviderRequestFailure("synthetic bounded request failure"),
    RuntimeError("synthetic parser failure"),
])
def test_native_guard_failure_keeps_original_full_metadata_and_history(
    monkeypatch, listing, currency, error,
):
    calls, searches = first_search_error(
        monkeypatch, error, listing=listing, currency=currency,
    )
    with owned() as session:
        assert en._owns_native_isin_data_request(ISIN) is True
        data = en._fetch_benchmark_data(ISIN)
        assert not session.providers.flight._pending
    assert len(searches) == 2
    assert data[en._TICKER_SYMBOL_KEY] == listing
    assert data["info"]["currency"] == currency
    assert data["info"]["regularMarketPrice"] == 101.0
    assert data["history"]["Close"].tolist() == [100.0, 101.0]
    assert data["history"].attrs[en._HISTORY_SYMBOL_ATTR] == listing
    assert "/v7/finance/quote" in calls
    assert any("/v10/finance/quoteSummary/" in path for path in calls)
    assert not any("/fundamentals-timeseries/" in path for path in calls)


@pytest.mark.parametrize("error", [
    HTTPError("synthetic HTTP failure"), Timeout("synthetic timeout"),
])
@pytest.mark.parametrize("listing,currency", [
    ("SYNRECOVERY.MI", "EUR"), ("SYNRECOVERY.L", "USD"),
])
def test_already_handled_transport_errors_keep_the_existing_recovery(
    monkeypatch, listing, currency, error,
):
    _calls, searches = first_search_error(
        monkeypatch, error, listing=listing, currency=currency,
    )
    with owned():
        data = en._fetch_benchmark_data(ISIN)
    assert len(searches) == 2
    assert data[en._TICKER_SYMBOL_KEY] == listing
    assert data["info"]["currency"] == currency
    assert data["history"]["Close"].tolist() == [100.0, 101.0]


@pytest.mark.parametrize("error", [
    KeyboardInterrupt("synthetic interruption"),
    SystemExit("synthetic cancellation"),
    UnrecordedInput("synthetic unrecorded source"),
])
def test_guard_never_masks_fatal_interruption_or_missing_frozen_authority(
    monkeypatch, error,
):
    _calls, searches = first_search_error(monkeypatch, error)
    with owned(), pytest.raises(type(error)) as caught:
        en._fetch_benchmark_data(ISIN)
    assert caught.value is error
    assert len(searches) == 1


def test_guard_failure_does_not_reopen_an_expired_native_acquisition_budget(monkeypatch):
    calls, searches = first_search_error(
        monkeypatch, ProviderBudgetExceeded("synthetic exhausted budget"),
    )
    with owned() as session:
        session.transport.deadline = session.transport.clock() - 1
        data = en._fetch_benchmark_data(ISIN)
        assert not session.providers.flight._pending
    # The injected first constructor failure occurs before any raw response.
    # Existing operation admission must prevent all subsequent acquisition.
    assert len(searches) == 1
    assert calls == []
    assert data["info"] == {}
    assert data["history"].empty
    assert data[en._TICKER_SYMBOL_KEY] == ISIN
