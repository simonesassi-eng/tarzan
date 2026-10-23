"""Evidence labels must not turn watchlist/chart coverage into portfolio value."""

from types import SimpleNamespace

import pandas as pd
import pytest

from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.export.newsletter._sections_alloc import _build_ticker_sources
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime.diagnostics import _quality


def test_private_quality_distinguishes_missing_holdings_from_missing_watchlist():
    metrics = PortfolioMetrics(
        total_value=1000.,
        holdings_df=pd.DataFrame({"ticker": ["HELD.MI", "HELDGAP.MI"]}),
        intraday_requested_tickers=["HELD.MI", "HELDGAP.MI", "WATCHGAP.MI"],
        intraday_quotes={"HELD.MI": {"intraday_series": object()}},
        intraday_evidence={
            "HELD.MI": {"state": "LAGGED_CURRENT_SESSION", "native_bar_count": 1},
            "HELDGAP.MI": {"state": "MISSING", "native_bar_count": 0},
            "WATCHGAP.MI": {"state": "MISSING", "native_bar_count": 0},
        },
        returns_coverage_pct=100.,
    )
    quality = _quality(metrics)
    assert quality["intraday_requested"] == 3
    assert quality["intraday_with_bars"] == 1
    assert quality["intraday_missing"] == ["HELDGAP.MI", "WATCHGAP.MI"]
    assert quality["intraday_current_holdings_missing"] == ["HELDGAP.MI"]
    assert quality["intraday_current_holdings_lagged"] == ["HELD.MI"]
    assert quality["intraday_single_observation"] == ["HELD.MI"]
    assert quality["returns_coverage_pct"] == 100.


@pytest.mark.parametrize("state,count,label", [
    ("LAGGED_CURRENT_SESSION", 2, "Lagged bars · 09:45 UTC"),
    ("LAST_COMPLETED_SESSION", 2, "Last session · 09:45 UTC"),
    ("FRESH_CANONICAL", 1, "Single bar · 09:45 UTC"),
    ("MISSING", 0, "No current-session bars"),
])
def test_source_audit_preserves_native_isin_venue_and_actual_bar_time(
    monkeypatch, state, count, label,
):
    monkeypatch.setattr("tarzan.config.benchmark_identities", lambda: ())
    ticker = "PROVEN.MI"
    record = {
        "canonical_ticker": ticker, "isin": "XX0000000001", "name": "Synthetic",
        "portfolio_presence": "Current + Historical",
        "history_ticker": ticker, "intraday_effective_ticker": ticker if count else "",
        "historical_price_sources": (
            ("XX0000000002", "XMIL", "SYNTHETIC_LIQUIDATION_AUTHORITY"),
        ),
    }
    metrics = SimpleNamespace(
        ticker_resolutions=(record,), benchmark_tickers={},
        intraday_requested_tickers=(ticker,),
        intraday_quotes={ticker: {}} if count else {},
        intraday_evidence={ticker: {
            "state": state, "native_bar_count": count,
            "observed_at": "2026-10-23T09:45:00+00:00",
        }},
    )
    audit = _build_ticker_sources(_NewsletterContext(metrics=metrics, config=None))
    row, = audit["rows"]
    assert row["isin"] == "XX0000000001"
    assert row["hist_ric"] == "XX0000000002 · XMIL"
    assert row["history_note"] == "Official clean liquidation price"
    assert row["intraday_note"] == label
    assert "not portfolio valuation coverage" in audit["quality_note"]
    assert ("1 lagged" in audit["quality_note"]) is (state == "LAGGED_CURRENT_SESSION")
