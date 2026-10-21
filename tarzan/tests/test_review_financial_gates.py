"""Synthetic regression coverage for review R01–R03 and R06."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from tarzan import orchestrator
from tarzan.contracts.exceptions import DataIngestionError
from tarzan.contracts.scalars import parse_boolean
from tarzan.contracts.targets import TargetSetStatus
from tarzan.data import enricher, loader
from tarzan.engine.metrics import MetricsEngine
from tarzan.models.holding import Holding, AssetClass
from tarzan.models.investor_config import InvestorConfig
from tarzan.runtime.session import last_run_result
from tarzan.runtime.summary import SummaryProjector


def _holding():
    return Holding(
        isin="US0000000001", ticker="SYNTH", quantity=10.0,
        cost_basis_eur=1000.0, market_value_eur=1000.0, currency="EUR",
        current_price=100.0, current_value=1000.0,
        asset_class=AssetClass.EQUITIES,
    )


def test_invalid_target_set_blocks_a_standalone_engine_even_when_mapping_is_empty():
    targets = loader.load_targets_per_holding(io.BytesIO(
        b"isin,ticker,target_portfolio\nUS0000000001,SYNTH,50\nUS0000000001,SYNTH,50\n",
    ))
    holdings = [_holding()]
    orchestrator._apply_per_holding_targets(holdings, targets)
    engine = MetricsEngine(holdings, InvestorConfig(target_use_per_holding_only=True))
    context = {"total_value": 1000.0, "classification_available": True}
    engine._rebalancing(context)
    assert targets.status is TargetSetStatus.INVALID
    assert context["rebalancing_plans"] is None
    assert context["rebalancing_suggestions"] is None
    assert context["planning_availability"] == "UNAVAILABLE"


def test_invalid_targets_block_real_orchestration_but_not_independent_analytics(monkeypatch):
    from tarzan.tests.test_golden_master import _AS_OF, _ORDERS_CSV, _stub_enrich
    monkeypatch.setattr(enricher, "enrich_holdings", _stub_enrich)
    monkeypatch.setattr("tarzan.engine.metrics._fetch_benchmark_history", lambda *a, **k: pd.Series(dtype=float))
    metrics, _ = orchestrator.run(
        config_source=io.BytesIO(b"key,value\ntarget_use_per_holding_only,true\n"),
        orders_source=io.BytesIO(_ORDERS_CSV.encode()),
        targets_per_holding_source=io.BytesIO(
            b"isin,target_portfolio\nIE00B4L5Y983,100\nIE00B4L5Y983,100\n",
        ),
        deterministic=True, as_of=_AS_OF,
    )
    assert metrics.total_value == pytest.approx(19650.0)  # synthetic book
    assert metrics.rebalancing_plans is None
    assert metrics.planning_availability == "UNAVAILABLE"
    summary = SummaryProjector.project(last_run_result()).to_dict()
    assert summary["sections"]["planning"]["availability"] == "UNAVAILABLE"
    assert summary["sections"]["planning"]["value"] is None
    assert summary["sections"]["planning"]["failure_refs"]


def test_target_absence_is_not_an_unreadable_target_file():
    absent = orchestrator._load_targets_or_empty(None, "")
    invalid = orchestrator._load_targets_or_empty(
        io.BytesIO(b'isin,target_portfolio\n"unterminated,100\n'), "synthetic.csv",
    )
    assert absent.status is TargetSetStatus.ABSENT and absent.planning_eligible
    assert invalid.status is TargetSetStatus.INVALID and not invalid.planning_eligible
    assert invalid.errors[0].code == "UNREADABLE_TARGET_INPUT"


def test_absent_required_per_holding_plan_cannot_authorize_liquidation():
    engine = MetricsEngine(
        [_holding()], InvestorConfig(target_use_per_holding_only=True),
        target_plan=orchestrator._load_targets_or_empty(None, ""),
    )
    context = {"total_value": 1000.0}
    engine._rebalancing(context)
    assert context["rebalancing_plans"] is None


@pytest.mark.parametrize("value", ["treu", "maybe", "2", "-1", "truth"])
def test_unknown_protection_flag_invalidates_the_whole_plan(value):
    targets = loader.load_targets_per_holding(io.BytesIO(
        f"isin,target_portfolio,no_buy_no_sell\nUS0000000001,100,{value}\n".encode(),
    ))
    assert not targets.planning_eligible
    assert targets.rows[0].value["no_buy_no_sell"] is True
    assert any(error.code == "INVALID_TRADING_PROTECTION" for error in targets.errors)


@pytest.mark.parametrize("value,expected", [
    ("TRUE", True), ("yes", True), ("1", True), ("on", True),
    ("FALSE", False), ("no", False), ("0", False), ("off", False), ("", False),
])
def test_closed_protection_domain_preserves_explicit_true_false_and_optional_blank(value, expected):
    assert parse_boolean(value) is expected
    targets = loader.load_targets_per_holding(io.BytesIO(
        f"isin,target_portfolio,no_buy_no_sell\nUS0000000001,100,{value}\n".encode(),
    ))
    assert targets.planning_eligible
    assert targets["US0000000001"]["no_buy_no_sell"] is expected


@pytest.mark.parametrize("key", ["rebalancing_no_sell", "rebalancing_auto_relax", "target_use_per_holding_only"])
def test_configuration_booleans_reject_typos_and_blank_keeps_default(key):
    with pytest.raises(DataIngestionError, match="boolean"):
        InvestorConfig.from_dict({key: "treu"})
    assert getattr(InvestorConfig.from_dict({key: ""}), key) == getattr(InvestorConfig(), key)


@pytest.mark.parametrize("value", ["NaN", "inf", "-1", "not-a-number"])
@pytest.mark.parametrize("key", [
    "target_cash_buffer_eur", "rebalancing_transaction_fee_buy_eur",
    "target_invested_allocation_equities_pctg", "target_equity_geo_usa_pctg",
])
def test_configuration_numeric_domains_do_not_silently_fall_back(key, value):
    with pytest.raises(DataIngestionError):
        InvestorConfig.from_dict({key: value})


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("kind,net,fees", [
    ("buy", 100.0, 0.0), ("sell", -100.0, 0.0),
    ("coupon", -100.0, 0.0), ("dividend", -100.0, 0.0),
    ("transfer_in", 100.0, 0.0), ("transfer_out", 100.0, 0.0),
])
def test_incoherent_cash_direction_rejects_the_ledger_without_guessing_or_dropping_positions(kind, net, fees, strict):
    source = io.BytesIO(
        ("date,type,isin,quantity,gross_eur,net_eur,fees_eur,currency,instrument_kind\n"
         f"2025-01-02,{kind},US0000000001,10,100,{net},{fees},EUR,ETF\n").encode(),
    )
    with pytest.raises(DataIngestionError, match="net_eur|proceeds"):
        loader.load_orders(source, strict=strict)


@pytest.mark.parametrize("kind,net,fees", [
    ("buy", -102.0, 2.0), ("sell", 98.0, 2.0),
    ("dividend", 10.0, 0.0), ("coupon", 10.0, 0.0),
    ("transfer_in", -2.0, 2.0), ("transfer_out", -2.0, 2.0),
    ("sell", -1.0, 2.0), ("transfer_in", 0.0, 0.0),
])
def test_legitimate_flows_fee_only_transfers_and_net_debit_sales_remain_valid(kind, net, fees):
    source = io.BytesIO(
        ("date,type,isin,quantity,gross_eur,net_eur,fees_eur,currency,instrument_kind\n"
         f"2025-01-02,{kind},US0000000001,10,100,{net},{fees},EUR,ETF\n").encode(),
    )
    orders = loader.load_orders(source, strict=True)
    assert len(orders) == 1
    assert orders[0].net_eur == net
    assert orders[0].fees_eur == fees


def test_optional_absent_fee_is_zero_without_a_spurious_skip_warning(caplog):
    source = io.BytesIO(
        b"date,type,isin,quantity,gross_eur,net_eur\n2025-01-02,buy,US0000000001,10,100,-100\n",
    )
    assert loader.load_orders(source, strict=True)[0].fees_eur == 0.0
    assert "invalid fees_eur" not in caplog.text


@pytest.mark.parametrize("content", [
    "key,value\nrebalancing_no_sell,true\nrebalancing_no_sell,false\n",
    "name,value\nrebalancing_no_sell,true\n",
    'key,value\nrebalancing_no_sell,"unterminated\n',
    "key,value\nrebalancing_no_sell\n",
    "key,value\nrebalancing_no_sell,false,extra\n",
])
def test_present_invalid_configuration_cannot_become_defaults(content, tmp_path):
    with pytest.raises(DataIngestionError):
        loader.load_config(io.BytesIO(content.encode()))
    path = tmp_path / "synthetic.csv"
    path.write_text(content)
    with pytest.raises(DataIngestionError):
        loader.load_config(str(path))


def test_configuration_bom_and_description_column_share_the_same_parser(tmp_path):
    content = "\ufeffkey,value,description\nrebalancing_no_sell,true,synthetic protection\n"
    path = tmp_path / "synthetic.csv"
    path.write_text(content)
    uploaded = loader.load_config(io.BytesIO(content.encode()))
    local = loader.load_config(str(path))
    assert uploaded.rebalancing_no_sell is local.rebalancing_no_sell is True
