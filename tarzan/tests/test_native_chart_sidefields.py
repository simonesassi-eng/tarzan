"""Unusable chart dates or volume cannot erase independent valid prices."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from decimal import Decimal
import json

import pandas as pd
import pytest

from curl_cffi.requests import Headers, Response
from curl_cffi.requests.models import Request
from yfinance.cache import get_tz_cache

from tarzan.data import _yf_net, chart_actions, enricher, price_cache, proxy_data
from tarzan.runtime import data_quality as dq
from tarzan.runtime.yahoo_response import YahooChartResponse
from tarzan.tests.test_native_chart_price_types import chart_document, native_packets
from tarzan.tests.test_yahoo_core_info import owned


FIELD_REJECTION_ATTR = "tarzan_rejected_chart_fields_v1"
_NATIVE_STORE_HISTORY = price_cache.store_history


@pytest.mark.parametrize("invalid", [
    True, False, [1], {"raw": 1}, "not an epoch", 10 ** 400,
], ids=["true", "false", "array", "object", "nonnumeric", "overflow"])
def test_unusable_raw_timestamp_keeps_a_different_valid_dated_price(native_packets, invalid):
    documents, calls, responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["timestamp"][1] = invalid
    documents["SYNTH.MI"] = deepcopy(document)
    with owned("native-sidefield-timestamp") as session:
        observed = enricher._download_price_history("SYNTH.MI")
        receipt = session.transport.coordinator.snapshot()
    assert isinstance(observed, pd.DataFrame)
    assert observed["Price Close"].tolist() == [100.]
    assert list(observed.index.date) == [pd.Timestamp("2026-10-23").date()]
    assert receipt["acquired"] == receipt["released"] == len(calls) == 1
    assert receipt["active"] == 0
    assert json.loads(responses[0].content) == documents["SYNTH.MI"] == document


@pytest.mark.parametrize("invalid", [
    [100], {"raw": 100}, "not a volume", 10 ** 400,
], ids=["array", "object", "nonnumeric", "overflow"])
def test_unusable_volume_keeps_independent_valid_native_price_rows(native_packets, invalid):
    documents, calls, responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = invalid
    documents["SYNTH.MI"] = deepcopy(document)
    with owned("native-sidefield-volume") as session:
        observed = enricher._download_price_history("SYNTH.MI")
        receipt = session.transport.coordinator.snapshot()
    assert isinstance(observed, pd.DataFrame)
    assert observed["Close"].tolist() == observed["Price Close"].tolist() == [100., 101.]
    assert receipt["acquired"] == receipt["released"] == len(calls) == 1
    assert receipt["active"] == 0
    assert json.loads(responses[0].content) == documents["SYNTH.MI"] == document


@pytest.mark.parametrize("volume", [None, 0, 100, 100.])
def test_valid_native_volume_keeps_all_original_price_rows(native_packets, volume):
    documents, calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = volume
    documents["SYNTH.MI"] = document
    with owned("native-sidefield-volume-valid"):
        observed = enricher._download_price_history("SYNTH.MI")
    assert observed["Close"].tolist() == observed["Price Close"].tolist() == [100., 101.]
    assert len(calls) == 1


def test_missing_native_timestamp_keeps_only_the_independent_dated_row(native_packets):
    documents, calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["timestamp"][1] = None
    documents["SYNTH.MI"] = document
    with owned("native-sidefield-missing-date"):
        observed = enricher._download_price_history("SYNTH.MI")
    assert observed["Close"].tolist() == observed["Price Close"].tolist() == [100.]
    assert list(observed.index.date) == [pd.Timestamp("2026-10-23").date()]
    assert len(calls) == 1


@pytest.mark.parametrize("field", ["timestamp", "volume"])
def test_rejected_auxiliary_evidence_is_visible_and_allows_later_recovery(native_packets, field):
    documents, calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    result = document["chart"]["result"][0]
    if field == "timestamp":
        result["timestamp"][1] = True
    else:
        result["indicators"]["quote"][0]["volume"][1] = {"raw": 100}
    documents["SYNTH.MI"] = document
    with owned("native-sidefield-visible-recovery") as session:
        uncertain = enricher._download_price_history("SYNTH.MI")
        assert uncertain.attrs[FIELD_REJECTION_ATTR]["fields"] == (field,)
        assert not enricher._reusable_price_observation(uncertain)
        issues = [issue for issue in dq.issues() if "chart dates or volume" in issue.message]
        assert len(issues) == 1 and issues[0].severity == dq.WARNING
        documents["SYNTH.MI"] = chart_document("SYNTH.MI", [100., 102.])
        recovered = enricher._download_price_history("SYNTH.MI")
        reused = enricher._download_price_history("SYNTH.MI")
        receipt = session.transport.coordinator.snapshot()
    assert recovered["Price Close"].tolist() == [100., 102.]
    assert FIELD_REJECTION_ATTR not in recovered.attrs
    pd.testing.assert_frame_equal(recovered, reused)
    assert receipt["active"] == 0 and receipt["acquired"] == receipt["released"] == len(calls) == 2


def test_rejected_volume_does_not_overwrite_existing_valid_disk_bytes(native_packets, monkeypatch, tmp_path):
    documents, _calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = [100]
    documents["SYNTH.MI"] = document
    monkeypatch.setenv("TARZAN_DISABLE_CACHE", "0")
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "market"))
    with owned("native-sidefield-disk"):
        uncertain = enricher._download_price_history("SYNTH.MI")
    clean = uncertain.copy()
    clean.attrs.pop(FIELD_REJECTION_ATTR)
    _NATIVE_STORE_HISTORY("SYNTH.MI", clean)
    target = price_cache._history_path("SYNTH.MI")
    original_bytes = target.read_bytes()
    _NATIVE_STORE_HISTORY("SYNTH.MI", uncertain)
    assert target.read_bytes() == original_bytes


def test_repeated_merge_keeps_bounded_auxiliary_uncertainty(native_packets):
    documents, _calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = True
    documents["SYNTH.MI"] = document
    with owned("native-sidefield-merge"):
        uncertain = enricher._download_price_history("SYNTH.MI")
    expected = uncertain.attrs[FIELD_REJECTION_ATTR]
    clean = uncertain.copy()
    clean.attrs.pop(FIELD_REJECTION_ATTR)
    merged = clean
    for _ in range(100):
        merged = price_cache.merge_history(merged, uncertain)
        assert merged.attrs[FIELD_REJECTION_ATTR] == expected
        assert not enricher._reusable_price_observation(merged)


@pytest.mark.parametrize("field_names", [
    None, True, {"timestamp": True}, ("timestamp", ["volume"]),
])
def test_malformed_auxiliary_receipt_fields_cannot_abort_valid_price_merge(field_names):
    index = pd.to_datetime(["2026-10-23", "2026-10-26"])
    clean = pd.DataFrame({"Close": [100., 101.]}, index=index)
    uncertain = clean.copy()
    uncertain.attrs[FIELD_REJECTION_ATTR] = {
        "symbol": "SYNTH.MI", "dates": ("2026-10-26",),
        "unknown_date": False, "fields": field_names,
    }
    merged = price_cache.merge_history(clean, uncertain)
    assert merged["Close"].tolist() == [100., 101.]
    assert chart_actions.has_rejected_chart_evidence(merged)
    assert merged.attrs[FIELD_REJECTION_ATTR]["unknown_date"] is True


@pytest.mark.parametrize("volume", [9_223_372_036_854_775_807, "100", 100.5])
def test_valid_native_volume_domains_keep_exact_decoded_raw_cells(volume):
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = volume
    raw = json.dumps(document).encode()
    native, admitted = Response(), YahooChartResponse()
    native.content = admitted.content = raw
    native.url = admitted.url = "https://query1.finance.yahoo.com/v8/finance/chart/SYNTH.MI"
    assert admitted.json() == native.json() == document


@pytest.mark.parametrize("invalid", ["100.0", "1e2"])
def test_native_noninteger_volume_strings_do_not_erase_valid_prices(native_packets, invalid):
    documents, _calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = invalid
    documents["SYNTH.MI"] = document
    with owned("native-sidefield-volume-integer-string"):
        observed = enricher._download_price_history("SYNTH.MI")
    assert observed["Price Close"].tolist() == [100., 101.]
    assert FIELD_REJECTION_ATTR in observed.attrs


@pytest.mark.parametrize("stamp", [9_223_372_036.5, -9_223_372_036.5])
def test_native_representable_fractional_epoch_boundaries_keep_their_exact_cells(stamp):
    from yfinance.utils import parse_quotes

    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["timestamp"][1] = stamp
    native = parse_quotes(document["chart"]["result"][0])
    response = YahooChartResponse()
    response.url = "https://query1.finance.yahoo.com/v8/finance/chart/SYNTH.MI"
    response.content = json.dumps(document).encode()
    admitted = response.json()
    assert admitted == document
    pd.testing.assert_frame_equal(parse_quotes(admitted["chart"]["result"][0]), native)


def test_auxiliary_uncertainty_survives_fx_run_memo_then_recovers(native_packets):
    documents, calls, _responses, _stored = native_packets
    document = chart_document("USDEUR=X", [.5, .51])
    document["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = {"raw": 100}
    documents["USDEUR=X"] = document
    with owned("native-sidefield-fx-memo"):
        first = enricher._fetch_fx_pair("USD")
        documents["USDEUR=X"] = chart_document("USDEUR=X", [.5, .52])
        second = enricher._fetch_fx_pair("USD")
        reused = enricher._fetch_fx_pair("USD")
    assert first.tolist() == [.5, .51] and FIELD_REJECTION_ATTR in first.attrs
    assert second.tolist() == [.5, .52] and FIELD_REJECTION_ATTR not in second.attrs
    pd.testing.assert_series_equal(second, reused)
    assert len(calls) == 2


def test_auxiliary_uncertainty_survives_proxy_returns_then_recovers(native_packets, monkeypatch):
    documents, calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = {"raw": 100}
    documents["SYNTH.MI"] = document
    monkeypatch.setattr(price_cache, "load_history", lambda _key: None)
    monkeypatch.setattr(_yf_net, "space_yf_call", lambda: None)
    with owned("native-sidefield-proxy-memo"):
        first = proxy_data._returns("SYNTH.MI")
        documents["SYNTH.MI"] = chart_document("SYNTH.MI", [100., 102.])
        second = proxy_data._returns("SYNTH.MI")
    assert first.iloc[-1] == pytest.approx(.01) and FIELD_REJECTION_ATTR in first.attrs
    assert second.iloc[-1] == pytest.approx(.02) and FIELD_REJECTION_ATTR not in second.attrs
    assert len(calls) == 2


@pytest.mark.parametrize("group_by,multi_level_index", [
    ("ticker", True), ("column", True), ("column", False),
])
def test_auxiliary_uncertainty_survives_batch_projection_and_later_recovery(
    native_packets, group_by, multi_level_index,
):
    documents, calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = True
    documents["SYNTH.MI"] = document
    documents["USDEUR=X"] = chart_document("USDEUR=X", [.5, .51])
    with owned("native-sidefield-batch"):
        first = _yf_net.download(
            ["SYNTH.MI", "USDEUR=X"], period="5y", interval="1d",
            group_by=group_by, multi_level_index=multi_level_index,
            auto_adjust=False, threads=False,
        )
        assert FIELD_REJECTION_ATTR in first.attrs
        child = first["SYNTH.MI"] if group_by == "ticker" else first["Close"]
        assert FIELD_REJECTION_ATTR in child.attrs
        documents["SYNTH.MI"] = chart_document("SYNTH.MI", [100., 102.])
        recovered = _yf_net.download(
            ["SYNTH.MI", "USDEUR=X"], period="5y", interval="1d",
            group_by=group_by, multi_level_index=multi_level_index,
            auto_adjust=False, threads=False,
        )
    assert FIELD_REJECTION_ATTR not in recovered.attrs
    assert len(calls) == 3  # Healthy sibling retains its original positive memo.


def test_both_timestamp_and_volume_receipts_remain_bounded_and_visible(native_packets):
    documents, _calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    result = document["chart"]["result"][0]
    result["timestamp"][1] = True
    result["indicators"]["quote"][0]["volume"][0] = {"raw": 100}
    documents["SYNTH.MI"] = document
    with owned("native-sidefield-two-fields"):
        observed = enricher._download_price_history("SYNTH.MI")
    receipt = observed.attrs[FIELD_REJECTION_ATTR]
    assert observed["Price Close"].tolist() == [100.]
    assert receipt["fields"] == ("timestamp", "volume")
    assert receipt["unknown_date"] is True and receipt["dates"] == ("2026-10-23",)


@pytest.mark.parametrize("future_count", [1, 64, 65, 128])
def test_future_rejected_volume_cannot_contaminate_an_older_window(native_packets, future_count):
    documents, _calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    result = document["chart"]["result"][0]
    first = int(pd.Timestamp("2026-10-27", tz="UTC").timestamp())
    result["timestamp"].extend(first + day * 86400 for day in range(future_count))
    quote = result["indicators"]["quote"][0]
    for field in ("open", "high", "low", "close"):
        quote[field].extend([102.] * future_count)
    quote["volume"].extend([{"raw": 100}] * future_count)
    result["indicators"]["adjclose"][0]["adjclose"].extend([102.] * future_count)
    documents["SYNTH.MI"] = document
    with owned("native-sidefield-future-scope"):
        observed = enricher._download_price_history("SYNTH.MI")
        issues = dq.issues()
    assert observed["Price Close"].tolist() == [100., 101.]
    assert FIELD_REJECTION_ATTR not in observed.attrs
    assert not any("chart dates or volume" in issue.message for issue in issues)


@pytest.mark.parametrize("url", [
    "", "https://example.invalid/v8/finance/chart/SYNTH.MI",
    "https://query1.finance.yahoo.com/v7/finance/quote",
])
def test_auxiliary_guard_without_a_completed_native_chart_keeps_prior_decoder_contract(url):
    document = chart_document("SYNTH.MI", [100., 101.])
    result = document["chart"]["result"][0]
    result["timestamp"] = [False, True]
    result["indicators"]["quote"][0]["volume"] = [False, True]
    response = YahooChartResponse()
    response.url, response.content = url, json.dumps(document).encode()
    assert response.json() == document


@pytest.mark.parametrize("host", ["query1.finance.yahoo.com", "query2.finance.yahoo.com"])
def test_native_original_request_qualifies_a_chart_with_a_different_effective_url(host):
    document = chart_document("SYNTH.MI", [100., 101.])
    result = document["chart"]["result"][0]
    result["timestamp"][1] = True
    raw = json.dumps(document).encode()
    response = YahooChartResponse()
    response.request = Request(
        f"https://{host}/v8/finance/chart/SYNTH.MI", Headers(), "GET",
    )
    response.url = "http://127.0.0.1:49152/synthetic-chart-probe"
    response.content = raw
    admitted = response.json()
    assert admitted["chart"]["result"][0]["timestamp"] == [result["timestamp"][0], None]
    assert response.content is raw and json.loads(response.content) == document
    assert response.request.method == "GET"


@pytest.mark.parametrize("url,method", [
    ("https://query1.finance.yahoo.com/v8/finance/chart/SYNTH.MI", "POST"),
    ("https://example.invalid/v8/finance/chart/SYNTH.MI", "GET"),
    ("https://query1.finance.yahoo.com/v7/finance/quote", "GET"),
    ("https://query1.finance.yahoo.com/v8/finance/chart/", "GET"),
    ("http://query1.finance.yahoo.com/v8/finance/chart/SYNTH.MI", "GET"),
])
def test_unqualified_native_request_cannot_be_overridden_by_the_effective_url(url, method):
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["timestamp"][1] = True
    response = YahooChartResponse()
    response.request = Request(url, Headers(), method)
    response.url = "https://query1.finance.yahoo.com/v8/finance/chart/SYNTH.MI"
    response.content = json.dumps(document).encode()
    assert response.json() == document


def test_unknown_request_shape_retains_its_native_decoder_contract():
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["timestamp"][1] = True
    response = YahooChartResponse()
    response.request = {"url": "https://query1.finance.yahoo.com/v8/finance/chart/SYNTH.MI"}
    response.url = "https://query1.finance.yahoo.com/v8/finance/chart/SYNTH.MI"
    response.content = json.dumps(document).encode()
    assert response.json() == document


@pytest.mark.parametrize("hooks", [
    {"parse_float": Decimal}, {"parse_int": Decimal},
    {"parse_float": Decimal, "parse_int": Decimal},
])
def test_custom_native_json_numeric_hooks_keep_the_original_decoded_contract(hooks):
    document = chart_document("SYNTH.MI", [100., 101.5])
    response, control = YahooChartResponse(), Response()
    response.url = control.url = "https://query1.finance.yahoo.com/v8/finance/chart/SYNTH.MI"
    response.content = control.content = json.dumps(document).encode()
    assert response.json(**hooks) == control.json(**hooks)


def test_auxiliary_decodes_are_independent_and_do_not_change_native_response_facts():
    document = chart_document("SYNTH.MI", [100., 101.])
    result = document["chart"]["result"][0]
    result["timestamp"][1] = True
    result["indicators"]["quote"][0]["volume"][0] = {"raw": 100}
    response = YahooChartResponse()
    raw = json.dumps(document).encode()
    response.content, response.status_code, response.reason = raw, 200, "OK"
    response.url = "https://query1.finance.yahoo.com/v8/finance/chart/SYNTH.MI"
    first = response.json()
    first["chart"]["result"][0]["timestamp"][0] = 0
    first["chart"]["result"][0]["meta"]["_tarzan_rejected_chart_fields_v1"].clear()
    second = response.json()["chart"]["result"][0]
    assert second["timestamp"] == [result["timestamp"][0], None]
    assert second["indicators"]["quote"][0]["volume"] == [None, 100]
    assert set(second["meta"]["_tarzan_rejected_chart_fields_v1"]) == {"timestamp", "volume"}
    assert response.content is raw and json.loads(response.text) == document
    assert response.status_code == 200 and response.reason == "OK"


@pytest.mark.parametrize("volume", [
    -(2 ** 63), -(2 ** 63) + 1, -1, "-1", "9223372036854775807",
])
def test_native_signed_integer_volume_domain_is_not_reinterpreted(volume):
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = volume
    response = YahooChartResponse()
    response.url = "https://query1.finance.yahoo.com/v8/finance/chart/SYNTH.MI"
    response.content = json.dumps(document).encode()
    assert response.json() == document


@pytest.mark.parametrize("field", ["timestamp", "volume"])
@pytest.mark.parametrize("invalid", [
    float("nan"), float("inf"), -float("inf"), "NaN", "Infinity",
    2 ** 63, -(2 ** 63) - 1,
])
def test_unrepresentable_native_auxiliary_values_keep_independent_valid_prices(
    native_packets, field, invalid,
):
    documents, calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    result = document["chart"]["result"][0]
    if field == "timestamp":
        result["timestamp"][1] = invalid
    else:
        result["indicators"]["quote"][0]["volume"][1] = invalid
    documents["SYNTH.MI"] = document
    with owned("native-sidefield-numeric-domain"):
        observed = enricher._download_price_history("SYNTH.MI")
    assert observed["Price Close"].tolist() == ([100.] if field == "timestamp" else [100., 101.])
    assert FIELD_REJECTION_ATTR in observed.attrs and len(calls) == 1


def test_no_valid_native_dates_cannot_invent_prices_and_allows_a_later_valid_answer(native_packets):
    documents, calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["timestamp"] = [True, {"raw": 1}]
    documents["SYNTH.MI"] = document
    with owned("native-sidefield-invalid-dates-recovery"):
        first = enricher._download_price_history("SYNTH.MI")
        assert first is None or not enricher._history_has_market_evidence(first)
        documents["SYNTH.MI"] = chart_document("SYNTH.MI", [100., 102.])
        recovered = enricher._download_price_history("SYNTH.MI")
    assert recovered["Price Close"].tolist() == [100., 102.] and len(calls) == 2


def test_no_valid_direct_fx_dates_still_tries_the_independent_inverse(native_packets):
    documents, calls, _responses, _stored = native_packets
    document = chart_document("USDEUR=X", [.5, .51])
    document["chart"]["result"][0]["timestamp"] = [True, {"raw": 1}]
    documents["USDEUR=X"] = document
    documents["EURUSD=X"] = chart_document("EURUSD=X", [2., 2.])
    with owned("native-sidefield-invalid-dates-inverse"):
        observed = enricher._fetch_fx_pair_uncached("USD")
    assert observed.tolist() == [.5, .5]
    assert [symbol for symbol, _arguments in calls] == ["USDEUR=X", "EURUSD=X"]


def test_concurrent_native_auxiliary_rejection_keeps_healthy_siblings_and_drains(native_packets):
    documents, calls, responses, _stored = native_packets
    symbols = [f"SIDEFIELD{index:02d}.MI" for index in range(8)]
    for index, symbol in enumerate(symbols):
        get_tz_cache().store(symbol, "UTC")
        packet = chart_document(symbol, [100. + index, 101. + index])
        if index % 2:
            packet["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = True
        documents[symbol] = packet
    originals = deepcopy(documents)
    with owned("native-sidefield-concurrent") as session:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(copy_context().run, enricher._download_price_history, symbol)
                for symbol in symbols
            ]
            frames = [future.result(timeout=10) for future in futures]
        receipt = session.transport.coordinator.snapshot()
    for index, frame in enumerate(frames):
        assert frame["Price Close"].tolist() == [100. + index, 101. + index]
        assert (FIELD_REJECTION_ATTR in frame.attrs) is bool(index % 2)
    assert documents == originals
    assert {json.loads(response.content)["chart"]["result"][0]["meta"]["symbol"]
            for response in responses} == set(symbols)
    assert receipt["acquired"] == receipt["released"] == len(calls) == 8
    assert receipt["active"] == receipt["queued_essential"] == receipt["queued_optional"] == 0
    assert receipt["peak"] <= receipt["capacity"] == 4
