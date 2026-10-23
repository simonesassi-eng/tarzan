"""The second-source prefetch must match its actual row/tape/currency demand."""

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from tarzan.engine import cross_check
from tarzan.engine.metrics import MetricsEngine
from tarzan.models.investor_config import InvestorConfig


def _tape(points=2):
    return pd.Series(
        [100.0 + index for index in range(points)],
        index=pd.date_range("2025-01-01", periods=points),
    )


def _holding(ticker, isin, currency="EUR", *, eur=None, native=None):
    return SimpleNamespace(
        ticker=ticker, isin=isin, price_currency=currency,
        price_history=eur, price_history_native=native,
    )


@pytest.fixture
def ports(monkeypatch):
    calls = {"prefetch": [], "series": [], "quote": []}
    monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
    monkeypatch.setattr("tarzan.runtime.today", lambda: date(2026, 10, 23))
    monkeypatch.setattr("tarzan.config._load_indexes_csv", lambda: pd.DataFrame())
    monkeypatch.setattr(
        "tarzan.data.justetf.prefetch",
        lambda isins: calls["prefetch"].append(tuple(isins)),
    )

    def series(isin):
        calls["series"].append(isin)
        return None

    def quote(isin):
        calls["quote"].append(isin)
        return None

    monkeypatch.setattr("tarzan.data.justetf.series", series)
    monkeypatch.setattr("tarzan.data.justetf.quote", quote)
    monkeypatch.setattr(
        cross_check, "check_windows",
        lambda *args, **kwargs: {
            window: {"status": cross_check.UNAVAILABLE, "alt": None}
            for window in cross_check.LONG_WINDOWS
        },
    )
    return calls


def _run(holdings, rows, *, seeds=(), catalog=None):
    hp = pd.DataFrame(rows, columns=["ticker", "type", "1d", "5d"])
    before = hp.copy(deep=True)
    ctx = {"holding_performance": hp, "_benchmark_catalog": catalog or {}}
    engine = MetricsEngine(
        list(holdings), InvestorConfig(), rebalance_seeds=list(seeds),
    )
    engine._second_source(ctx)
    pd.testing.assert_frame_equal(ctx["holding_performance"][before.columns], before)
    return ctx["holding_performance"]


@pytest.mark.parametrize("currency", ["EUR", "eur", "eUr"])
def test_listing_eur_is_demanded_without_changing_financial_rows(ports, currency):
    holding = _holding("AAA.MI", "IE0000000001", currency, eur=_tape())
    result = _run([holding], [("AAA.MI", "In portfolio", 1.0, 2.0)])
    assert ports["prefetch"] == [("IE0000000001",)]
    assert ports["series"] == ["IE0000000001"]
    assert result.loc[0, "xc_5d"] == cross_check.UNAVAILABLE


@pytest.mark.parametrize("currency", ["USD", "GBP", "", None, "UNKNOWN"])
def test_non_eur_listing_is_not_bought_even_with_eur_valuation_tape(ports, currency):
    holding = _holding(
        "AAA.US", "IE0000000001", currency, eur=_tape(), native=_tape(),
    )
    _run([holding], [("AAA.US", "In portfolio", 1.0, 2.0)])
    assert ports["prefetch"] == [()]
    assert ports["series"] == []
    assert ports["quote"] == []


def test_same_isin_eur_and_usd_keeps_eur_demand_only(ports):
    eur = _holding("AAA.MI", "IE0000000001", eur=_tape(), native=_tape())
    usd = _holding("AAA.US", "IE0000000001", "USD", eur=_tape(), native=_tape())
    result = _run(
        [usd, eur],
        [("AAA.US", "In portfolio", 1.0, 2.0),
         ("AAA.MI", "In portfolio", 1.0, 2.0)],
    )
    assert ports["prefetch"] == [("IE0000000001",)]
    assert ports["series"] == ["IE0000000001"]
    assert result.loc[0, "xc_5d"] == cross_check.UNAVAILABLE


def test_absent_native_uses_the_same_legacy_eur_tape_predicate(ports):
    holding = _holding("AAA.MI", "IE0000000001", eur=_tape())
    _run([holding], [("AAA.MI", "In portfolio", 1.0, 2.0)])
    assert ports["prefetch"] == [("IE0000000001",)]


def test_single_native_observation_retains_the_existing_eur_fallback(ports):
    holding = _holding("AAA.MI", "IE0000000001", eur=_tape(), native=_tape(1))
    _run([holding], [("AAA.MI", "In portfolio", 1.0, 2.0)])
    assert ports["prefetch"] == [("IE0000000001",)]


def test_none_tape_is_not_demanded(ports):
    holding = _holding("AAA.MI", "IE0000000001")
    _run([holding], [("AAA.MI", "In portfolio", 1.0, 2.0)])
    assert ports["prefetch"] == [()]
    assert ports["series"] == []


def test_empty_tape_preserves_the_existing_consumer_predicate(ports):
    # Empty is not None. Changing that here would disagree with the consumer.
    holding = _holding("AAA.MI", "IE0000000001", eur=_tape(0))
    _run([holding], [("AAA.MI", "In portfolio", 1.0, 2.0)])
    assert ports["prefetch"] == [("IE0000000001",)]


def test_subject_not_in_printed_rows_is_not_demanded(ports):
    holdings = [
        _holding("AAA.MI", "IE0000000001", eur=_tape()),
        _holding("BBB.MI", "IE0000000002", eur=_tape()),
    ]
    _run(holdings, [("AAA.MI", "In portfolio", 1.0, 2.0)])
    assert ports["prefetch"] == [("IE0000000001",)]
    assert ports["series"] == ["IE0000000001"]


def test_row_type_must_match_the_subject_not_just_the_ticker(ports):
    holding = _holding("AAA.MI", "IE0000000001", eur=_tape())
    _run([holding], [("AAA.MI", "Target not held", 1.0, 2.0)])
    assert ports["prefetch"] == [()]
    assert ports["series"] == []


def test_seed_and_portfolio_share_one_prefetch_but_keep_both_rows(ports):
    holding = _holding("AAA.MI", "IE0000000001", eur=_tape())
    result = _run(
        [holding],
        [("AAA.MI", "In portfolio", 1.0, 2.0),
         ("AAA.MI", "Target not held", 1.0, 2.0)],
        seeds=[holding],
    )
    assert ports["prefetch"] == [("IE0000000001",)]
    assert len(result) == 2
    assert ports["series"] == ["IE0000000001", "IE0000000001"]


def test_benchmark_demand_uses_its_listing_currency(ports, monkeypatch):
    monkeypatch.setattr(
        "tarzan.config._load_indexes_csv",
        lambda: pd.DataFrame({
            "ticker": ["AAA", "BBB"], "isin": ["IE0000000001", "IE0000000002"],
        }),
    )
    catalog = {
        "eur": SimpleNamespace(
            ticker="AAA.MI", history=_tape(), history_native=_tape(), currency="EUR",
        ),
        "usd": SimpleNamespace(
            ticker="BBB.US", history=_tape(), history_native=_tape(), currency="USD",
        ),
    }
    _run(
        [], [("AAA.MI", "Benchmark index", 1.0, 2.0),
             ("BBB.US", "Benchmark index", 1.0, 2.0)],
        catalog=catalog,
    )
    assert ports["prefetch"] == [("IE0000000001",)]
    assert ports["series"] == ["IE0000000001"]


def test_prefetch_failure_keeps_rows_and_unavailable_verdicts(ports, monkeypatch):
    def failed(_isins):
        raise TimeoutError("synthetic referee outage")

    monkeypatch.setattr("tarzan.data.justetf.prefetch", failed)
    holding = _holding("AAA.MI", "IE0000000001", eur=_tape())
    result = _run([holding], [("AAA.MI", "In portfolio", 1.0, 2.0)])
    assert result.loc[0, "xc_5d"] == cross_check.UNAVAILABLE
    assert ports["series"] == []


def test_transport_forbidden_does_not_prefetch_or_consume(ports, monkeypatch):
    monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: False)
    holding = _holding("AAA.MI", "IE0000000001", eur=_tape())
    _run([holding], [("AAA.MI", "In portfolio", 1.0, 2.0)])
    assert ports == {"prefetch": [], "series": [], "quote": []}
