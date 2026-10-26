"""Raw split fields never acquire authority by losing their scalar types."""

from copy import deepcopy
from decimal import Decimal
import json

import pandas as pd
import pytest

from tarzan.data import _yf_net, enricher, price_cache, proxy_data
from tarzan.runtime import data_quality as dq
from tarzan.tests.test_native_chart_price_types import chart_document, native_packets
from tarzan.tests.test_yahoo_core_info import owned


_NATIVE_STORE_HISTORY = price_cache.store_history
REJECTION_ATTR = "tarzan_rejected_split_evidence_v1"


def split_document(values, numerator, denominator, *, symbol="SYNTH.MI"):
    document = chart_document(symbol, values)
    result = document["chart"]["result"][0]
    result["meta"]["instrumentType"] = "EQUITY"
    stamp = result["timestamp"][1]
    result["events"] = {"splits": {str(stamp): {
        "date": stamp, "numerator": numerator, "denominator": denominator,
        "splitRatio": "synthetic-independent-control",
    }}}
    return document


@pytest.mark.parametrize("numerator,denominator,values", [
    (True, 2., [100., 200.]),
    (2., True, [100., 50.]),
], ids=["boolean-numerator", "boolean-denominator"])
def test_raw_boolean_split_cannot_rescale_valid_native_prices(
    native_packets, numerator, denominator, values,
):
    documents, calls, responses, _stored = native_packets
    document = split_document(values, numerator, denominator)
    documents["SYNTH.MI"] = deepcopy(document)
    with owned("native-split-boolean") as session:
        observed = enricher._download_price_history("SYNTH.MI")
        repaired = price_cache.repair_split_jumps(observed, symbol="SYNTH.MI")
        receipt = session.transport.coordinator.snapshot()
    assert observed["Price Close"].tolist() == values
    assert repaired["Price Close"].tolist() == repaired["Close"].tolist() == values
    assert len(calls) == 1
    assert receipt["active"] == 0 and receipt["acquired"] == receipt["released"] == 1
    assert documents["SYNTH.MI"] == document
    assert json.loads(responses[0].content) == document


def test_rejected_split_is_visible_and_cannot_be_positive_memoized(native_packets):
    documents, calls, _responses, _stored = native_packets
    documents["SYNTH.MI"] = split_document([100., 101.], True, 2.)
    with owned("native-split-visible-recovery") as session:
        rejected = enricher._download_price_history("SYNTH.MI")
        assert REJECTION_ATTR in rejected.attrs
        assert rejected.attrs[REJECTION_ATTR]["dates"] == ("2026-10-26",)
        assert not enricher._reusable_price_observation(rejected)
        issues = [issue for issue in dq.issues() if "split factors" in issue.message]
        assert len(issues) == 1 and issues[0].severity == dq.WARNING
        assert issues[0].context == "SYNTH.MI"
        documents["SYNTH.MI"] = split_document([100., 50.], 2., 1.)
        recovered = enricher._download_price_history("SYNTH.MI")
        reused = enricher._download_price_history("SYNTH.MI")
        receipt = session.transport.coordinator.snapshot()
    assert REJECTION_ATTR not in recovered.attrs
    pd.testing.assert_frame_equal(recovered, reused)
    assert price_cache.repair_split_jumps(recovered)["Price Close"].tolist() == [50., 50.]
    assert len(calls) == 2
    assert receipt["active"] == 0 and receipt["acquired"] == receipt["released"] == 2


def test_uncertain_action_is_not_persisted_over_a_valid_existing_cache(
    native_packets, monkeypatch, tmp_path,
):
    documents, _calls, _responses, _stored = native_packets
    documents["SYNTH.MI"] = split_document([100., 101.], True, 2.)
    monkeypatch.setenv("TARZAN_DISABLE_CACHE", "0")
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "market"))
    with owned("native-split-cache-receipt"):
        uncertain = enricher._download_price_history("SYNTH.MI")
    clean = uncertain.copy()
    clean.attrs.pop(REJECTION_ATTR)
    _NATIVE_STORE_HISTORY("SYNTH.MI", clean)
    target = price_cache._history_path("SYNTH.MI")
    assert target.is_file()
    original_bytes = target.read_bytes()
    _NATIVE_STORE_HISTORY("SYNTH.MI", uncertain)
    assert target.read_bytes() == original_bytes
    restored = price_cache.load_history("SYNTH.MI")
    assert restored is not None and REJECTION_ATTR not in restored.attrs


def test_merge_cannot_erase_or_recursively_expand_action_uncertainty(native_packets):
    documents, _calls, _responses, _stored = native_packets
    documents["SYNTH.MI"] = split_document([100., 101.], True, 2.)
    with owned("native-split-merge-receipt"):
        uncertain = enricher._download_price_history("SYNTH.MI")
    clean = uncertain.copy()
    clean.attrs.pop(REJECTION_ATTR)
    merged = price_cache.merge_history(clean, uncertain)
    expected = uncertain.attrs[REJECTION_ATTR]
    assert merged.attrs[REJECTION_ATTR] == expected
    for _ in range(100):
        merged = price_cache.merge_history(merged, uncertain)
        assert merged.attrs[REJECTION_ATTR] == expected
    assert not enricher._reusable_price_observation(merged)


def test_rejected_future_action_cannot_degrade_an_earlier_price_window(native_packets):
    documents, _calls, _responses, _stored = native_packets
    document = split_document([100., 101.], True, 2.)
    result = document["chart"]["result"][0]
    event = next(iter(result["events"]["splits"].values()))
    event["date"] = int(pd.Timestamp("2026-10-27", tz="UTC").timestamp())
    documents["SYNTH.MI"] = document
    with owned("native-split-future-boundary"):
        observed = enricher._download_price_history("SYNTH.MI")
        issues = dq.issues()
    assert observed["Price Close"].tolist() == [100., 101.]
    assert REJECTION_ATTR not in observed.attrs
    assert not any("split factors" in issue.message for issue in issues)


def test_bounded_many_future_rejections_cannot_contaminate_an_older_window(native_packets):
    documents, _calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    result = document["chart"]["result"][0]
    result["meta"]["instrumentType"] = "EQUITY"
    start = int(pd.Timestamp("2026-10-27", tz="UTC").timestamp())
    result["events"] = {"splits": {
        str(start + day * 86400): {
            "date": start + day * 86400, "numerator": True, "denominator": 2.,
        }
        for day in range(65)
    }}
    documents["SYNTH.MI"] = document
    with owned("native-split-bounded-future"):
        observed = enricher._download_price_history("SYNTH.MI")
        issues = dq.issues()
    assert observed["Close"].tolist() == observed["Price Close"].tolist() == [100., 101.]
    assert REJECTION_ATTR not in observed.attrs
    assert not any("split factors" in issue.message for issue in issues)


@pytest.mark.parametrize("start,count", [
    ("2026-07-01", 65), ("2026-02-01", 128),
    ("2026-10-27", 64), ("2026-10-27", 128),
])
def test_bounded_rejected_events_preserve_known_outside_window_scope(
    native_packets, start, count,
):
    documents, _calls, _responses, _stored = native_packets
    document = split_document([100., 101.], True, 2.)
    result = document["chart"]["result"][0]
    first = int(pd.Timestamp(start, tz="UTC").timestamp())
    result["events"]["splits"] = {
        str(first + day * 86400): {
            "date": first + day * 86400, "numerator": True, "denominator": 2.,
        }
        for day in range(count)
    }
    documents["SYNTH.MI"] = document
    with owned("native-split-bounded-outside"):
        observed = enricher._download_price_history("SYNTH.MI")
        issues = dq.issues()
    assert observed["Price Close"].tolist() == [100., 101.]
    assert REJECTION_ATTR not in observed.attrs
    assert not any("split factors" in issue.message for issue in issues)


def test_omitted_in_window_rejection_remains_uncertain_when_date_list_is_full(native_packets):
    documents, _calls, _responses, _stored = native_packets
    document = split_document([100., 101.], True, 2.)
    result = document["chart"]["result"][0]
    first = int(pd.Timestamp("2026-10-27", tz="UTC").timestamp())
    result["events"]["splits"] = {
        str(first + day * 86400): {
            "date": first + day * 86400, "numerator": True, "denominator": 2.,
        }
        for day in range(64)
    }
    result["events"]["splits"]["omitted-in-window"] = {
        "date": result["timestamp"][-1], "numerator": True, "denominator": 2.,
    }
    documents["SYNTH.MI"] = document
    with owned("native-split-truncated-overlap"):
        observed = enricher._download_price_history("SYNTH.MI")
        issues = dq.issues()
    assert observed["Price Close"].tolist() == [100., 101.]
    assert observed.attrs[REJECTION_ATTR]["unknown_date"] is True
    assert not enricher._reusable_price_observation(observed)
    assert any("split factors" in issue.message for issue in issues)


@pytest.mark.parametrize("date", [None, True, "2026-10-26", 0, -1, 10 ** 400])
def test_unknown_rejected_action_date_cannot_certify_an_older_window(native_packets, date):
    documents, _calls, _responses, _stored = native_packets
    document = split_document([100., 101.], True, 2.)
    event = next(iter(document["chart"]["result"][0]["events"]["splits"].values()))
    event["date"] = date
    documents["SYNTH.MI"] = document
    with owned("native-split-unknown-date"):
        observed = enricher._download_price_history("SYNTH.MI")
    assert observed["Price Close"].tolist() == [100., 101.]
    assert observed.attrs[REJECTION_ATTR]["unknown_date"] is True
    assert not enricher._reusable_price_observation(observed)


@pytest.mark.parametrize("numerator,denominator", [
    (False, 2.), (2., False), (0., 1.), (1., 0.), (-1., 2.), (2., -1.),
    (float("inf"), 1.), (1., float("nan")), (1e308, 1e-308), (1e-308, 1e308),
])
def test_invalid_or_nonrepresentable_split_factor_never_repairs_prices(
    native_packets, numerator, denominator,
):
    documents, _calls, _responses, _stored = native_packets
    documents["SYNTH.MI"] = split_document([100., 101.], numerator, denominator)
    with owned("native-split-domain"):
        observed = enricher._download_price_history("SYNTH.MI")
    repaired = price_cache.repair_split_jumps(observed)
    assert repaired["Price Close"].tolist() == repaired["Close"].tolist() == [100., 101.]
    assert REJECTION_ATTR in observed.attrs


def test_invalid_split_does_not_drop_an_independent_valid_event(native_packets):
    documents, _calls, _responses, _stored = native_packets
    document = split_document([100., 50.], 2., 1.)
    result = document["chart"]["result"][0]
    result["events"]["splits"]["invalid-neighbour"] = {
        "date": result["timestamp"][0], "numerator": True, "denominator": 2.,
    }
    documents["SYNTH.MI"] = document
    with owned("native-split-valid-neighbour"):
        observed = enricher._download_price_history("SYNTH.MI")
    assert observed["Stock Splits"].tolist() == [0., 2.]
    assert price_cache.repair_split_jumps(observed)["Price Close"].tolist() == [50., 50.]
    assert REJECTION_ATTR in observed.attrs


def test_proxy_consumer_retains_uncertainty_and_allows_later_recovery(native_packets, monkeypatch):
    documents, calls, _responses, _stored = native_packets
    documents["SYNTH.MI"] = split_document([100., 101.], True, 2.)
    monkeypatch.setattr(price_cache, "load_history", lambda _key: None)
    monkeypatch.setattr(_yf_net, "space_yf_call", lambda: None)
    with owned("native-split-proxy-recovery"):
        first = proxy_data._returns("SYNTH.MI")
        assert REJECTION_ATTR in first.attrs
        documents["SYNTH.MI"] = split_document([100., 102.], 2., 1.)
        second = proxy_data._returns("SYNTH.MI")
    assert first.iloc[-1] == pytest.approx(.01)
    assert second.iloc[-1] == pytest.approx(.02)
    assert REJECTION_ATTR not in second.attrs and len(calls) == 2


def test_fx_consumer_keeps_split_uncertainty_instead_of_persistable_rates(native_packets):
    documents, _calls, _responses, stored = native_packets
    document = split_document([.5, .51], True, 2., symbol="USDEUR=X")
    document["chart"]["result"][0]["meta"]["instrumentType"] = "CURRENCY"
    documents["USDEUR=X"] = document
    with owned("native-split-fx-receipt"):
        observed = enricher._fetch_fx_pair_uncached("USD")
    assert observed.tolist() == [.5, .51]
    assert REJECTION_ATTR in observed.attrs
    assert len(stored) == 1 and REJECTION_ATTR in stored[0][1].attrs


def test_fx_run_memo_must_not_hide_later_valid_action_evidence(native_packets):
    documents, calls, _responses, _stored = native_packets
    document = split_document([.5, .51], True, 2., symbol="USDEUR=X")
    document["chart"]["result"][0]["meta"]["instrumentType"] = "CURRENCY"
    documents["USDEUR=X"] = document
    with owned("native-split-fx-memo-recovery"):
        first = enricher._fetch_fx_pair("USD")
        document = chart_document("USDEUR=X", [.5, .52])
        documents["USDEUR=X"] = document
        recovered = enricher._fetch_fx_pair("USD")
        reused = enricher._fetch_fx_pair("USD")
    assert first.tolist() == [.5, .51] and REJECTION_ATTR in first.attrs
    assert recovered.tolist() == [.5, .52] and REJECTION_ATTR not in recovered.attrs
    pd.testing.assert_series_equal(recovered, reused)
    assert len(calls) == 2


def test_batch_child_does_not_reuse_an_uncertain_action_fact(native_packets):
    documents, calls, _responses, _stored = native_packets
    documents["SYNTH.MI"] = split_document([100., 101.], True, 2.)
    with owned("native-split-batch-recovery") as session:
        first = _yf_net.download(
            ["SYNTH.MI"], period="5y", interval="1d", auto_adjust=False,
            group_by="ticker", threads=False,
        )
        documents["SYNTH.MI"] = split_document([100., 102.], 2., 1.)
        second = _yf_net.download(
            ["SYNTH.MI"], period="5y", interval="1d", auto_adjust=False,
            group_by="ticker", threads=False,
        )
        receipt = session.transport.coordinator.snapshot()
    assert first["SYNTH.MI"]["Close"].tolist() == [100., 101.]
    assert second["SYNTH.MI"]["Close"].tolist() == [100., 102.]
    assert len(calls) == 2
    assert receipt["active"] == 0 and receipt["acquired"] == receipt["released"] == 2


def test_multi_symbol_batch_cannot_erase_uncertain_child_evidence(native_packets):
    documents, calls, _responses, _stored = native_packets
    documents["SYNTH.MI"] = split_document([100., 101.], True, 2.)
    documents["USDEUR=X"] = chart_document("USDEUR=X", [.5, .51])
    with owned("native-split-batch-projection") as session:
        observed = _yf_net.download(
            ["SYNTH.MI", "USDEUR=X"], period="5y", interval="1d",
            auto_adjust=False, group_by="ticker", threads=False,
        )
        receipt = session.transport.coordinator.snapshot()
    assert observed["SYNTH.MI"]["Close"].tolist() == [100., 101.]
    assert observed["USDEUR=X"]["Close"].tolist() == [.5, .51]
    assert REJECTION_ATTR in observed.attrs
    assert REJECTION_ATTR in observed["SYNTH.MI"].attrs
    assert len(calls) == 2
    assert receipt["active"] == 0 and receipt["acquired"] == receipt["released"] == 2


@pytest.mark.parametrize("numerator,denominator,values,expected", [
    (1., 2., [100., 200.], [200., 200.]),
    (2., 1., [100., 50.], [50., 50.]),
    (1, 2, [100., 200.], [200., 200.]),
    (2, 1, [100., 50.], [50., 50.]),
], ids=["reverse-float", "forward-float", "reverse-int", "forward-int"])
def test_valid_split_declarations_keep_native_and_local_rulers(
    native_packets, numerator, denominator, values, expected,
):
    documents, calls, responses, _stored = native_packets
    document = split_document(values, numerator, denominator)
    documents["SYNTH.MI"] = deepcopy(document)
    with owned("native-split-valid"):
        observed = enricher._download_price_history("SYNTH.MI")
        repaired = price_cache.repair_split_jumps(observed, symbol="SYNTH.MI")
    assert repaired["Price Close"].tolist() == repaired["Close"].tolist() == expected
    assert len(calls) == 1
    assert documents["SYNTH.MI"] == json.loads(responses[0].content) == document


@pytest.mark.parametrize("numerator,denominator", [
    ([2.], 1.), (2., {"raw": 1.}), ("not a ratio", 1.), (10 ** 400, 1.),
], ids=["array-numerator", "object-denominator", "nonnumeric-ratio",
        "overflow-numerator"])
def test_bad_split_fields_do_not_discard_independent_native_price_rows(
    native_packets, numerator, denominator,
):
    documents, calls, _responses, _stored = native_packets
    documents["SYNTH.MI"] = split_document([100., 101.], numerator, denominator)
    with owned("native-split-scalar"):
        observed = enricher._download_price_history("SYNTH.MI")
    assert isinstance(observed, pd.DataFrame)
    assert observed["Price Close"].dropna().tolist() == [100., 101.]
    assert observed["Close"].dropna().tolist() == [100., 101.]
    assert len(calls) == 1


@pytest.mark.parametrize("numerator,denominator", [(2, 1), (2., 1.), (1., 2.)])
def test_valid_raw_split_document_is_entirely_unchanged(numerator, denominator):
    from tarzan.runtime.yahoo_response import YahooChartResponse

    document = split_document([100., 101.], numerator, denominator)
    response = YahooChartResponse()
    raw = json.dumps(document).encode()
    response.content = raw
    assert response.json() == document
    assert response.content is raw


def test_rejected_event_receipt_preserves_raw_bytes_and_independent_decodes():
    from tarzan.runtime.yahoo_response import SPLIT_REJECTIONS_META, YahooChartResponse

    document = split_document([100., 101.], True, 2.)
    response = YahooChartResponse()
    raw = json.dumps(document).encode()
    response.content = raw
    first = response.json()["chart"]["result"][0]
    assert first["events"]["splits"] == {}
    expected = {
        "count": 1, "dates": [first["timestamp"][-1]], "unknown_date": False,
        "date_bounds": [first["timestamp"][-1]] * 2, "dates_truncated": False,
    }
    assert first["meta"][SPLIT_REJECTIONS_META] == expected
    first["meta"][SPLIT_REJECTIONS_META]["dates"].append(1)
    second = response.json()["chart"]["result"][0]
    assert second["meta"][SPLIT_REJECTIONS_META] == expected
    assert response.content is raw and json.loads(response.content) == document


def test_split_admission_retains_optional_native_decimal_decoder_contract():
    from curl_cffi.requests import Response
    from tarzan.runtime.yahoo_response import YahooChartResponse

    document = split_document([100., 101.], 2., 1.)
    raw = json.dumps(document).encode()
    native, admitted = Response(), YahooChartResponse()
    native.content = admitted.content = raw
    try:
        original = native.json(parse_float=Decimal)
    except TypeError as error:
        with pytest.raises(type(error)):
            admitted.json(parse_float=Decimal)
        return
    parsed = admitted.json(parse_float=Decimal)
    assert parsed == original
    event = next(iter(parsed["chart"]["result"][0]["events"]["splits"].values()))
    assert type(event["numerator"]) is type(event["denominator"]) is Decimal
