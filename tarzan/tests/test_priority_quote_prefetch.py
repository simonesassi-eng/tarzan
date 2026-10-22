"""Primary bars and macro histories precede quota-consuming optional discovery."""

from types import SimpleNamespace

import pandas as pd
import pytest

from tarzan import config, runtime
from tarzan.data import _yf_net, enricher, market_quotes
from tarzan.engine import benchmarks, metrics
from tarzan.models.investor_config import InvestorConfig
from tarzan.runtime.session import current_session
from tarzan.tests.test_essential_data_priority import owned, prices


def required_catalog():
    return {
        "Critical": benchmarks.ResolvedBenchmark(
            "Critical", "CORE", "CORE.MI", prices("CORE"),
        ),
    }


def engine():
    holdings = [
        SimpleNamespace(ticker="HELD.MI", isin="IE0000000001", target_portfolio=10),
        SimpleNamespace(ticker="", isin="IT0000000002", target_portfolio=0),
        SimpleNamespace(ticker="IT0000000003", isin="IT0000000003", target_portfolio=0),
    ]
    seeds = [
        SimpleNamespace(ticker="TARGET.DE", isin="IE0000000004", target_portfolio=20),
        SimpleNamespace(ticker="UNUSED.DE", isin="IE0000000005", target_portfolio=0),
    ]
    return metrics.MetricsEngine(holdings, InvestorConfig(), rebalance_seeds=seeds)


def configure(monkeypatch):
    monkeypatch.setattr(config, "benchmark_beta_name", lambda: "Critical")
    monkeypatch.setattr(config, "benchmark_geo_allocation", lambda: "Critical")
    monkeypatch.setattr(market_quotes, "MARKETS", (
        ("Macro A", "MACRO-A", "index"), ("Macro B", "MACRO-B", "index"),
    ))


def test_prefetch_hook_runs_after_required_resolution_and_before_optional_scope():
    seen = []
    with owned(total=30, optional=1) as (session, clock):
        def fetch(ticker):
            seen.append(ticker)
            return prices(ticker)

        def prefetch(catalog):
            assert list(catalog) == ["Critical"]
            assert catalog["Critical"].ticker == "CORE.MI"
            seen.append("prefetch")
            # Core I/O must not inherit the one-second optional ceiling.
            assert session.transport.call("yahoo", lambda cap: cap) == 20
            clock[0] += 2

        catalog, errors = benchmarks.preprocess_benchmarks(
            {"Extra": "EXTRA", "Critical": "CORE"},
            required_names=("Critical",), fetch_history=fetch,
            before_optional=prefetch,
        )
        assert seen == ["CORE", "prefetch", "EXTRA"]
        assert list(catalog) == ["Extra", "Critical"]
        assert errors == ()


def test_prefetch_hook_receives_only_successfully_resolved_required_references():
    received = []
    catalog, errors = benchmarks.preprocess_benchmarks(
        {"Missing": "MISSING", "Extra": "EXTRA", "Critical": "CORE"},
        required_names=("Missing", "Critical"),
        fetch_history=lambda ticker: prices(ticker) if ticker != "MISSING" else None,
        before_optional=lambda required: received.append(required),
    )
    assert list(received[0]) == ["Critical"]
    assert list(catalog) == ["Extra", "Critical"]
    assert len(errors) == 1


def test_owned_prefetch_uses_only_canonical_holdings_targets_references_and_fixed_macros(monkeypatch):
    configure(monkeypatch)
    downloads, histories, owners = [], [], []
    monkeypatch.setattr(_yf_net, "fetch_download", lambda symbols, **kwargs:
                        downloads.append((tuple(symbols), kwargs)) or pd.DataFrame())
    monkeypatch.setattr(enricher, "_fetch_history", lambda symbol:
                        histories.append(symbol) or owners.append(current_session())
                        or pd.DataFrame({"Close": [1.0, 2.0]}))
    subject = engine()
    with owned() as (session, clock):
        subject._prefetch_priority_quotes(required_catalog())
        assert set(downloads[0][0]) == {
            "HELD.MI", "TARGET.DE", "CORE.MI", "MACRO-A", "MACRO-B",
        }
        assert downloads[0][1]["period"] == "5d"
        assert downloads[0][1]["interval"] == "15m"
        assert downloads[0][1]["group_by"] == "ticker"
        assert set(histories) == {"MACRO-A", "MACRO-B"}
        assert all(owner is session for owner in owners)
        # I/O prefetch never rewrites holdings or their financial fields.
        assert subject.holdings[0].ticker == "HELD.MI"
        assert subject.holdings[0].target_portfolio == 10


@pytest.mark.parametrize("mode", ["unowned", "offline", "missing_required"])
def test_prefetch_starts_no_io_without_owned_live_authority_and_required_identity(monkeypatch, mode):
    configure(monkeypatch)
    monkeypatch.setattr(_yf_net, "fetch_download", lambda *a, **k: pytest.fail("unexpected I/O"))
    monkeypatch.setattr(enricher, "_fetch_history", lambda *a, **k: pytest.fail("unexpected I/O"))
    if mode == "unowned":
        engine()._prefetch_priority_quotes(required_catalog())
    else:
        with owned():
            if mode == "offline":
                monkeypatch.setattr(runtime, "allows_live_transport", lambda: False)
            engine()._prefetch_priority_quotes(
                {} if mode == "missing_required" else required_catalog(),
            )


def test_prefetch_provider_failure_is_best_effort_and_does_not_consume_optional_scope(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(_yf_net, "fetch_download", lambda *a, **k: (_ for _ in ()).throw(
        TimeoutError("synthetic private provider failure"),
    ))
    monkeypatch.setattr(enricher, "_fetch_history", lambda symbol: pd.DataFrame())
    with owned() as (session, clock):
        engine()._prefetch_priority_quotes(required_catalog())
        assert session.transport.call("yahoo", lambda cap: cap) == 20
        stages = [entry.payload for entry in session.ledger.entries
                  if entry.payload.get("stage") == "provider_prefetch:essential_quotes"]
        assert stages and stages[-1]["outcome"] == "DEGRADED"


def test_engine_wiring_prefetches_core_before_any_optional_lookup(monkeypatch):
    configure(monkeypatch)
    seen = []
    monkeypatch.setattr(metrics, "BENCHMARKS", {"Extra": "EXTRA", "Critical": "CORE"})
    monkeypatch.setattr(metrics, "_fetch_benchmark_history", lambda ticker:
                        seen.append(ticker) or prices(ticker))
    monkeypatch.setattr(_yf_net, "fetch_download", lambda *a, **k:
                        seen.append("core bars") or pd.DataFrame())
    monkeypatch.setattr(enricher, "_fetch_history", lambda symbol:
                        seen.append(symbol) or pd.DataFrame({"Close": [1.0, 2.0]}))
    with owned():
        context = {}
        engine()._preprocess_benchmarks(context)
        assert seen[0] == "CORE"
        assert seen.index("core bars") < seen.index("EXTRA")
        assert all(seen.index(symbol) < seen.index("EXTRA")
                   for symbol in ("MACRO-A", "MACRO-B"))
        assert list(context["benchmark_tickers"]) == ["Extra", "Critical"]


def test_one_macro_exception_does_not_hide_other_facts_or_cancellation(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(_yf_net, "fetch_download", lambda *a, **k: pd.DataFrame())

    def fetch(symbol):
        if symbol == "MACRO-A":
            raise TimeoutError("synthetic ordinary failure")
        raise KeyboardInterrupt()

    monkeypatch.setattr(enricher, "_fetch_history", fetch)
    with owned() as (session, clock):
        with pytest.raises(KeyboardInterrupt):
            engine()._prefetch_priority_quotes(required_catalog())
        stages = [entry.payload for entry in session.ledger.entries
                  if entry.payload.get("stage") == "provider_prefetch:essential_quotes"]
        assert stages[-1]["outcome"] == "INTERRUPTED"


def test_cancellation_of_prefetch_is_not_recorded_as_success(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(_yf_net, "fetch_download", lambda *a, **k: (_ for _ in ()).throw(
        KeyboardInterrupt(),
    ))
    with owned() as (session, clock):
        with pytest.raises(KeyboardInterrupt):
            engine()._prefetch_priority_quotes(required_catalog())
        stages = [entry.payload for entry in session.ledger.entries
                  if entry.payload.get("stage") == "provider_prefetch:essential_quotes"]
        assert stages[-1]["outcome"] == "INTERRUPTED"
