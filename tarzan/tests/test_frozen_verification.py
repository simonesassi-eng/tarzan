"""Synthetic tests of private, transport-level KPI verification."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from tarzan.runtime.io_utils import canonical_json_bytes
from tarzan.verification.portfolio import compare_checkpoints, differences, portfolio_kpis
from tarzan.verification.portfolio import _snapshot_cache_clock
from tarzan.verification.tape import HTTPAnswer, ProviderTape, UnrecordedInput, decode, encode


def test_frame_tape_preserves_provider_columns_timezone_and_metadata():
    frame = pd.DataFrame(
        {"Close": [100.125, 50.5], "Stock Splits": [0.0, 2.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="D", tz="Europe/Rome"),
    )
    frame.attrs["source"] = {"provider": "synthetic", "boundary": date(2025, 1, 2)}
    roundtrip = decode(encode(frame))
    assert_frame_equal(roundtrip, frame)
    assert roundtrip.attrs == frame.attrs
    assert b"NaN" not in canonical_json_bytes(encode(frame))


def test_series_and_nonfinite_numpy_values_have_strict_json_roundtrip():
    series = pd.Series(
        [1.0, np.nan, 3.0], index=pd.date_range("2025-01-01", periods=3), name="synthetic",
    )
    assert_series_equal(decode(encode(series)), series)
    value = {"when": datetime(2025, 1, 1, tzinfo=timezone.utc), 2: (np.float64(4.5), b"bytes")}
    assert decode(encode(value)) == value
    assert np.isnan(decode(encode(float("nan"))))
    assert decode(encode(float("inf"))) == float("inf")


def test_replay_has_no_provider_fallback_and_answers_are_isolated():
    captured = ProviderTape()
    original = {"price": 42.0, "nested": [1, 2]}
    answer = captured.call("synthetic.price", {"symbol": "SYNTH"}, lambda: original)
    original["nested"].append(3)
    answer["nested"].append(4)
    replay = ProviderTape(records=captured.records, replay=True)
    selected = replay.call("synthetic.price", {"symbol": "SYNTH"}, lambda: pytest.fail("network"))
    assert selected == {"price": 42.0, "nested": [1, 2]}
    with pytest.raises(UnrecordedInput):
        replay.call("synthetic.price", {"symbol": "OTHER"}, lambda: pytest.fail("network"))
    assert len(replay.misses) == 1


def test_transport_tuning_does_not_change_the_frozen_market_observation():
    tape = ProviderTape()
    tape.call("synthetic", {"symbol": "SYNTH", "timeout": 10}, lambda: 42)
    replay = ProviderTape(records=tape.records, replay=True)
    assert replay.call("synthetic", {"symbol": "SYNTH", "timeout": 1}, None) == 42
    with pytest.raises(UnrecordedInput):
        replay.call("synthetic", {"symbol": "SYNTH", "period": "1y"}, None)


def test_equivalent_quote_batches_and_mapping_order_reuse_the_same_answer():
    tape = ProviderTape()
    tape.call("yf.http", {"url": "https://example.invalid", "params": {"symbols": "B,A", "lang": "en"}},
              lambda: {"quote": [42, 43]})
    replay = ProviderTape(records=tape.records, replay=True)
    assert replay.call(
        "yf.http", {"params": {"lang": "en", "symbols": "A,B"}, "url": "https://example.invalid"}, None,
    ) == {"quote": [42, 43]}
    with pytest.raises(UnrecordedInput):
        replay.call("yf.http", {"url": "https://example.invalid", "params": {"symbols": "A,C", "lang": "en"}}, None)


@pytest.mark.parametrize("arguments", [
    {"url": "https://example.invalid/?token=secret"},
    {"json": {"api_key": "secret"}},
    {"cookies": {"session": "secret"}},
])
def test_credential_bearing_requests_are_not_recorded(arguments):
    tape = ProviderTape()
    with pytest.raises(ValueError, match="credential-bearing"):
        tape.call("synthetic", arguments, lambda: 42)
    assert not tape.records


def test_concurrent_consumers_select_one_answer_per_semantic_request():
    tape = ProviderTape()
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(
            lambda number: tape.call("synthetic", {"symbol": "SYNTH"}, lambda: number),
            range(8),
        ))
    assert len(set(answers)) == 1
    assert len(tape.records) == 1


def test_http_answers_support_all_used_provider_response_interfaces():
    answer = HTTPAnswer(body=b'{"quote":42}', url="https://example.invalid",
                        headers={"content-type": "application/json"})
    assert answer.json() == {"quote": 42}
    assert answer.text == '{"quote":42}'
    assert answer.ok
    assert answer.headers["content-type"] == "application/json"
    assert answer.read() == answer.content
    answer.raise_for_status()
    import requests
    with pytest.raises(requests.HTTPError):
        HTTPAnswer(body=b"", status=503).raise_for_status()


def test_kpi_difference_ignores_only_numerical_roundoff_not_unavailability():
    assert differences({"pnl": 100.0}, {"pnl": 100.0 + 1e-10}) == []
    assert differences({"pnl": None}, {"pnl": 0.0}) == [
        {"kpi": "pnl", "before": None, "after": 0.0},
    ]
    assert differences(False, 0) == [{"kpi": "", "before": False, "after": 0}]
    assert differences({"pnl": 100}, {"pnl": 110})[0]["delta"] == 10


@pytest.mark.parametrize("field", ["input_sha256", "captured_at", "observation_digest", "identity_cache_digest"])
def test_comparison_rejects_different_underlying_evidence(field):
    left = {name: "synthetic-a" for name in (
        "input_sha256", "captured_at", "observation_digest", "identity_cache_digest",
    )}
    left["kpis"] = {"total_value": 1000.0}
    assert compare_checkpoints(left, dict(left)) == []
    right = {**left, field: "synthetic-b"}
    with pytest.raises(ValueError, match=field):
        compare_checkpoints(left, right)


def test_kpi_projection_keeps_financial_outputs_not_ai_or_attempt_noise():
    from tarzan.models.portfolio import PortfolioMetrics
    metrics = PortfolioMetrics(
        total_value=1000.0,
        holdings_df=pd.DataFrame([{"isin": "US0000000001", "cost_basis_eur": 800.0}]),
        pnl_series=pd.Series([0.0, 200.0], index=pd.date_range("2025-01-01", periods=2)),
    )
    result = portfolio_kpis(metrics)
    assert result["unrealized_pnl_eur"] == 200.0
    assert result["unrealized_pnl_pct"] == 25.0
    assert "analysis_id" not in result
    assert "attempt_id" not in result
    assert result["pnl_series"]["values"] == [0.0, 200.0]


def test_snapshot_ttls_follow_capture_even_when_host_time_moves_months_forward(monkeypatch, tmp_path):
    import time
    from types import SimpleNamespace
    from tarzan.data import price_cache

    captured = datetime(2025, 1, 1, tzinfo=timezone.utc)
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(price_cache, "time", SimpleNamespace(time=lambda: captured.timestamp()))
    price_cache.store_resolution("US0000000001", "SYNTH.MI")
    price_cache.store_timed("openfigi", "US0000000001", ["SYNTH"])
    host = SimpleNamespace(time=lambda: captured.timestamp() + 100 * 86400)
    monkeypatch.setattr(price_cache, "time", host)
    original_process_clock = time.time
    assert price_cache.load_resolution("US0000000001") is None
    with _snapshot_cache_clock(captured):
        assert price_cache.load_resolution("US0000000001") == "SYNTH.MI"
        assert price_cache.load_timed("openfigi", "US0000000001", 7) == ["SYNTH"]
        assert time.time is original_process_clock
    assert price_cache.time is host
    assert price_cache.load_timed("openfigi", "US0000000001", 7) is None
