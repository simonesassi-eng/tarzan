"""Native metadata must retain request/listing identity before SDK flattening."""

from copy import deepcopy
import json
from urllib.parse import urlencode, urlsplit

from curl_cffi.requests import Response
from curl_cffi.requests.models import Request
import pytest
from yfinance.data import YfData

from tarzan.data import enricher as en
from tarzan.runtime.yahoo_response import YahooChartResponse
from tarzan.tests.test_native_isin_listing_identity import CAPTURED, owned


SYMBOL = "SYNIDENTITY.MI"
OTHER = "UNRELATED.DE"
GOOD = {
    "symbol": SYMBOL, "longName": "Synthetic independent identity",
    "currency": "EUR", "quoteType": "EQUITY", "regularMarketPrice": 101.0,
    "previousClose": 100.0, "regularMarketTime": int(CAPTURED.timestamp()),
}
SUMMARY = {
    "symbol": SYMBOL,
    "price": {**GOOD, "regularMarketPrice": 100.0},
    "assetProfile": {"country": "Ireland"},
    "quoteType": {"symbol": SYMBOL, "quoteType": "EQUITY"},
}
FOREIGN = {**GOOD, "symbol": OTHER, "regularMarketPrice": 999.0, "currency": "USD"}
FOREIGN_SUMMARY = {
    "symbol": OTHER, "price": dict(FOREIGN),
    "assetProfile": {"country": "Unrelated"},
    "quoteType": {"symbol": OTHER, "quoteType": "EQUITY"},
}


@pytest.fixture
def native_source(monkeypatch):
    YfData.cache_get.cache_clear()
    packets = {"summary": [deepcopy(SUMMARY)], "quote": [deepcopy(GOOD)]}
    calls = []

    def get(client, url, params=None, *arguments, **keywords):
        route = urlsplit(url).path
        if "/quoteSummary/" in route:
            operation, envelope = "summary", "quoteSummary"
        elif route == "/v7/finance/quote":
            assert params["symbols"] == SYMBOL
            operation, envelope = "quote", "quoteResponse"
        else:
            raise AssertionError("undeclared independent metadata source")
        calls.append(operation)
        response = client._session.response_class()
        response.url = url.rstrip("?") + ("?" + urlencode(params) if params else "")
        response.request = Request(response.url, {}, "GET")
        response.content = json.dumps({
            envelope: {"result": deepcopy(packets[operation]), "error": None},
        }).encode()
        response.status_code, response.ok = 200, True
        response.encoding = "utf-8"
        return response

    monkeypatch.setattr(YfData, "get", get)
    yield packets, calls
    YfData.cache_get.cache_clear()


def native_info():
    with owned() as session:
        result = en._fetch_ticker_info(SYMBOL)
        assert not session.providers.flight._pending
    return result


@pytest.mark.parametrize("bad_symbol", [OTHER, None, "", True, [SYMBOL]])
def test_unqualified_quote_does_not_override_independent_summary(native_source, bad_symbol):
    packets, calls = native_source
    packets["quote"] = [{**FOREIGN, "symbol": bad_symbol}]
    result = native_info()
    assert result["regularMarketPrice"] == 100.0
    assert result["currency"] == "EUR"
    assert result["country"] == "Ireland"
    assert calls == ["summary", "quote"]


@pytest.mark.parametrize("location", ["top", "price", "quoteType"])
def test_foreign_summary_does_not_contribute_profile_to_valid_quote(native_source, location):
    packets, _calls = native_source
    record = packets["summary"][0]
    if location == "top":
        record["symbol"] = OTHER
    else:
        record[location]["symbol"] = OTHER
    result = native_info()
    assert result["regularMarketPrice"] == 101.0
    assert result["currency"] == "EUR"
    assert "country" not in result


def test_both_foreign_sources_do_not_become_usable_metadata(native_source):
    packets, _calls = native_source
    packets.update(summary=[deepcopy(FOREIGN_SUMMARY)], quote=[deepcopy(FOREIGN)])
    assert native_info() == {}


@pytest.mark.parametrize("operation,first,correct,price", [
    ("summary", FOREIGN_SUMMARY, SUMMARY, 100.0),
    ("quote", FOREIGN, GOOD, 101.0),
])
def test_correct_later_record_is_not_hidden_by_relabelled_first_row(
    native_source, operation, first, correct, price,
):
    packets, _calls = native_source
    packets.update(summary=[], quote=[])
    packets[operation] = [deepcopy(first), deepcopy(correct)]
    result = native_info()
    assert result["regularMarketPrice"] == price
    assert result["currency"] == "EUR"


def test_conflicting_quotes_retain_summary_not_an_arbitrary_first_price(native_source):
    packets, _calls = native_source
    packets["quote"] = [deepcopy(GOOD), {**GOOD, "regularMarketPrice": 777.0}]
    result = native_info()
    assert result["regularMarketPrice"] == 100.0
    assert result["currency"] == "EUR"


def test_conflicting_summaries_retain_quote_not_an_arbitrary_first_profile(native_source):
    packets, _calls = native_source
    changed = deepcopy(SUMMARY)
    changed["assetProfile"]["country"] = "Conflicting"
    packets["summary"].append(changed)
    result = native_info()
    assert result["regularMarketPrice"] == 101.0
    assert "country" not in result


def test_identical_duplicates_and_missing_summary_header_remain_usable(native_source):
    packets, _calls = native_source
    del packets["summary"][0]["symbol"]
    packets["summary"] *= 2
    packets["quote"] *= 2
    result = native_info()
    assert result["regularMarketPrice"] == 101.0
    assert result["country"] == "Ireland"


def test_rejection_does_not_persist_into_later_run(native_source):
    packets, calls = native_source
    packets.update(summary=[], quote=[deepcopy(FOREIGN)])
    assert native_info() == {}
    packets["quote"] = [deepcopy(GOOD)]
    recovered = native_info()
    assert recovered["regularMarketPrice"] == 101.0
    assert calls == ["summary", "quote", "summary", "quote"]


def decoded(document, url, *, method="GET", request=True, factory=YahooChartResponse, **kwargs):
    response = factory()
    response.url = url
    if request:
        response.request = Request(url, {}, method)
    response.content = json.dumps(document).encode()
    response.encoding = "utf-8"
    original = response.content
    result = response.json(**kwargs)
    assert response.content == original
    return result


def test_batched_quotes_keep_only_each_requested_unambiguous_identity():
    third = "SYNTHIRD.L"
    document = {"quoteResponse": {"result": [
        dict(FOREIGN), dict(GOOD), {**GOOD, "symbol": third, "currency": "GBp"},
        {**GOOD, "regularMarketPrice": 777.0},
    ], "error": None}}
    result = decoded(
        document,
        f"https://query2.finance.yahoo.com/v7/finance/quote?symbols={SYMBOL},{third}",
    )
    assert result["quoteResponse"]["result"] == [
        {**GOOD, "symbol": third, "currency": "GBp"},
    ]


@pytest.mark.parametrize("url,method", [
    ("https://example.org/v7/finance/quote?symbols=SYNIDENTITY.MI", "GET"),
    ("http://query1.finance.yahoo.com/v7/finance/quote?symbols=SYNIDENTITY.MI", "GET"),
    ("https://query1.finance.yahoo.com/v7/finance/quote", "GET"),
    ("https://query1.finance.yahoo.com/v7/finance/quote?symbols=", "GET"),
    ("https://query1.finance.yahoo.com/v7/finance/quote?symbols=A&symbols=B", "GET"),
    ("https://query1.finance.yahoo.com/v7/finance/quote?symbols=SYNIDENTITY.MI", "POST"),
])
def test_uninspected_source_keeps_original_decoder_contract(url, method):
    document = {"quoteResponse": {"result": [dict(FOREIGN)], "error": None}}
    assert decoded(document, url, method=method) == document


def test_plain_public_response_does_not_change_its_contract():
    document = {"quoteResponse": {"result": [dict(FOREIGN)], "error": None}}
    assert decoded(
        document, f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={SYMBOL}",
        factory=Response,
    ) == document


def test_completed_response_url_without_request_is_valid_replay_authority():
    document = {"quoteResponse": {"result": [dict(FOREIGN)], "error": None}}
    assert decoded(
        document, f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={SYMBOL}",
        request=False,
    )["quoteResponse"]["result"] == []


def test_custom_decoder_dictionary_keeps_uninspected_contract():
    class Custom(dict):
        pass

    document = {"quoteResponse": {"result": [dict(FOREIGN)], "error": None}}
    result = decoded(
        document, f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={SYMBOL}",
        object_hook=Custom,
    )
    assert type(result) is Custom
    assert result == document


def test_native_errors_and_repeated_decoding_keep_bytes_and_report_rejection_once(caplog):
    response = YahooChartResponse()
    response.url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={SYMBOL}"
    response.content = json.dumps({
        "quoteResponse": {"result": [dict(FOREIGN)], "error": None},
    }).encode()
    original = response.content
    assert response.json()["quoteResponse"]["result"] == []
    assert response.json()["quoteResponse"]["result"] == []
    assert response.content == original
    warnings = [
        record for record in caplog.records
        if getattr(record, "public_code", "") == "YAHOO_METADATA_IDENTITY_REJECTED"
    ]
    assert len(warnings) == 1
    assert SYMBOL not in warnings[0].getMessage() and OTHER not in warnings[0].getMessage()
    response.content = b"synthetic malformed JSON"
    with pytest.raises(ValueError):
        response.json()


def test_optional_warning_failure_cannot_discard_independent_valid_quote(monkeypatch):
    from tarzan.runtime import yahoo_metadata

    def warning(*args, **kwargs):
        raise OSError("synthetic optional log handler failure")

    monkeypatch.setattr(yahoo_metadata.logger, "warning", warning)
    document = {"quoteResponse": {"result": [dict(FOREIGN), dict(GOOD)], "error": None}}
    assert decoded(
        document, f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={SYMBOL}",
    )["quoteResponse"]["result"] == [GOOD]


def test_fatal_cancellation_is_not_relabelled_as_optional_warning_failure(monkeypatch):
    from tarzan.runtime import yahoo_metadata

    def warning(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(yahoo_metadata.logger, "warning", warning)
    document = {"quoteResponse": {"result": [dict(FOREIGN)], "error": None}}
    with pytest.raises(KeyboardInterrupt):
        decoded(
            document, f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={SYMBOL}",
        )
