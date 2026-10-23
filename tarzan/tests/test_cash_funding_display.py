"""Synthetic source-to-HTML checks for strategic reserve versus bank funding."""

from __future__ import annotations

import copy
from html import unescape
from html.parser import HTMLParser
from itertools import product

from markupsafe import Markup
import numpy as np
import pandas as pd
import pytest

from tarzan.contracts.reference import ReportReference
from tarzan.engine import rebalancer
from tarzan.export.newsletter import build_context, render_newsletter
from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.export.newsletter import _sections_alloc as sections
from tarzan.export.newsletter._semantic import validate_newsletter_semantics
from tarzan.models.holding import AssetClass, Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.verification.portfolio import portfolio_kpis


_PROOF_FIELDS = (
    "initial_settlement_cash_eur", "ending_settlement_cash_eur",
    "cash_equivalent_securities_before_eur", "cash_equivalent_securities_after_eur",
    "initial_cash_equivalents_eur", "ending_cash_equivalents_eur",
    "cash_equivalents_target_eur", "protected_cash_eur",
    "external_contribution_eur", "gross_sales_eur", "gross_purchases_eur",
    "estimated_tax_eur", "fees_eur", "residual_eur",
)
_PROOF_FLAGS = (
    "position_invariants_satisfied", "protected_cash_satisfied",
    "settlement_cash_satisfied", "frozen_positions_satisfied",
    "no_sell_satisfied", "serialized_actions_satisfied",
    "action_identity_satisfied", "units_invariants_satisfied",
    "wealth_invariants_satisfied", "cash_mechanics_satisfied",
    "non_cost_only_progress_satisfied",
)
_SCOPE = (
    "Funding, position and policy checks use selected prices only. "
    "Broker lot sizes/fractional orders, settlement timing and execution prices "
    "are not verified. These are modeled proposals, not broker instructions."
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("TARZAN_DISABLE_AI", "1")
    monkeypatch.setattr("tarzan.runtime.today", lambda: pd.Timestamp("2026-10-21").date())

    def forbidden(*args, **kwargs):
        pytest.fail("cash funding display must not call AI, provider or SMTP")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("smtplib.SMTP", forbidden)
    monkeypatch.setattr("tarzan.export.ai_summary.generate_summary", forbidden)


def _reference():
    return ReportReference(
        names={"MONEY.MI": "Synthetic reserve ETF", "EQ.MI": "Synthetic equity"},
        taxonomy={"MONEY.MI": ("Cash & Cash Equivalents", "Cash / Money Market")},
    )


def _holding(ticker, *, kind="ETF", value=1000.0, price=10.0, cash=False,
             gain=0.0, locked=False):
    return Holding(
        isin="SYNTHETIC:" + ticker, ticker=ticker, name="Synthetic " + ticker,
        quantity=value / price if price else 0.0, currency="EUR",
        current_price=price, current_value=value, market_value_eur=value,
        cost_basis_eur=value / (1.0 + gain / 100.0),
        gain_pct=gain, security_type=kind, no_buy_no_sell=locked,
        asset_class=AssetClass.CASH_EQUIVALENTS if cash else AssetClass.EQUITIES,
    )


def _config(*, buffer=900.0, contribution=0.0, no_sell=False, fee=1.0):
    return InvestorConfig(
        invested_allocation_targets_pctg={"Equities": 100.0},
        equity_geo_targets_pctg={}, target_cash_buffer_eur=buffer,
        rebalancing_lump_sum_amount_eur=contribution, rebalancing_no_sell=no_sell,
        rebalancing_transaction_fee_buy_eur=fee,
        rebalancing_transaction_fee_sell_eur=fee,
        rebalancing_capital_gains_tax_standard_pctg=26.0,
    )


def _source(*, bank=0.0, locked=False, contribution=0.0, sell=False,
            no_sell=False, unknown=False, buy=True, buffer=900.0, fee=1.0):
    holdings = [
        _holding("EQ.MI"),
        _holding("MONEY.MI", kind=None if unknown else "ETF", cash=True, gain=100.0),
    ]
    if bank or locked:
        holdings.append(_holding("BANK", kind="CASH", value=bank, price=1.0,
                                 cash=True, locked=locked))
    config = _config(buffer=buffer, contribution=contribution, no_sell=no_sell, fee=fee)
    values = np.array([holding.current_value for holding in holdings], dtype=float)
    model = rebalancer._ObjectiveModel(holdings, config, values)
    vector = np.zeros(len(holdings))
    if buy:
        vector[0] = 85.0 if sell else 100.0
    if sell:
        vector[1] = -100.0
    actions = rebalancer._extract_actions(vector, holdings, model, values)
    after, funding = rebalancer._finalize_action_proof(
        holdings=holdings, values=values, actions=actions, config=config,
        external_contribution=contribution, protected_cash=buffer,
    )
    checks = rebalancer._verify(
        after, holdings, config, model.geo_frac, model.all_geos,
        unallocated_settlement_cash=funding["unallocated_settlement_cash_eur"],
        cash_mechanics=model.cash_mechanics,
    )
    return holdings, config, actions, checks + [funding]


def _metrics(holdings, config, actions, verifications, *, plans=True):
    frame = pd.DataFrame([{
        "isin": holding.isin, "ticker": holding.ticker, "name": holding.name,
        "quantity": holding.quantity, "current_price": holding.current_price,
        "current_value": holding.current_value, "cost_basis_eur": holding.cost_basis_eur,
        "gain_pct": holding.gain_pct or 0.0,
        "gain_eur": holding.current_value - holding.cost_basis_eur,
        "asset_class": holding.asset_class.value, "currency": "EUR",
        "geography": "Synthetic", "ter": 0.0, "yield_pct": 0.0,
        "weight_pct": holding.current_value / sum(h.current_value for h in holdings) * 100.0,
    } for holding in holdings])
    total = sum(holding.current_value for holding in holdings)
    reserve = sum(holding.current_value for holding in holdings
                  if holding.asset_class is AssetClass.CASH_EQUIVALENTS)
    return PortfolioMetrics(
        total_value=total, invested_value=total - reserve, cash_value=reserve,
        cash_target_eur=config.target_cash_buffer_eur, holdings_df=frame,
        allocation_by_class=pd.DataFrame({"category": ["Equities"], "weight_pct": [100.0]}),
        performance={"1d": 1.0}, performance_full={"1d": 1.0, "5d": 0.0},
        holding_performance=pd.DataFrame({
            "ticker": [holding.ticker for holding in holdings],
            "name": [holding.name for holding in holdings],
            "type": ["In portfolio"] * len(holdings),
            "asset_class": [holding.asset_class.value for holding in holdings],
            "1d": [0.25] * len(holdings),
        }),
        rebalancing_suggestions=actions, rebalancing_verifications=verifications,
        rebalancing_plans=[{
            "label": "Synthetic plan", "no_sell": config.rebalancing_no_sell,
            "suggestions": actions, "verifications": verifications,
        }] if plans else None,
        intraday_requested_tickers=tuple(holding.ticker for holding in holdings),
        intraday_evidence={}, intraday_quotes={}, market_quotes=(),
        report_reference=_reference(),
    )


def _cube(**kwargs):
    holdings, config, actions, checks = _source(**kwargs)
    return _metrics(holdings, config, actions, checks), config


def _context(metrics, config):
    return _NewsletterContext(metrics=metrics, config=config, reference=metrics.report_reference)


def _render(metrics, config):
    audit = {}
    html = render_newsletter(metrics, config, ai_summary="", semantic_audit=audit)
    return html, audit


class _FundingDocument(HTMLParser):
    """Independent DOM inspection; does not call renderer or guard helpers."""

    _VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
             "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.frames = []
        self.plans = {}
        self.feed(html)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        parent = self.frames[-1] if self.frames else (None, None, None)
        plan = parent[1]
        field = parent[2]
        if "data-tarzan-funding-plan" in attrs:
            plan = int(attrs["data-tarzan-funding-plan"])
            self.plans[plan] = {"attrs": attrs, "fields": {}}
        if "data-tarzan-funding-field" in attrs and plan is not None:
            field = attrs["data-tarzan-funding-field"]
            self.plans[plan]["fields"][field] = {"attrs": attrs, "text": ""}
        if tag not in self._VOID:
            self.frames.append((tag, plan, field))

    def handle_endtag(self, tag):
        for index in range(len(self.frames) - 1, -1, -1):
            if self.frames[index][0] == tag:
                del self.frames[index:]
                break

    def handle_data(self, text):
        if self.frames:
            _, plan, field = self.frames[-1]
            if plan is not None and field is not None:
                self.plans[plan]["fields"][field]["text"] += text


def _plan_document(html):
    document = _FundingDocument(html)
    assert set(document.plans) == {0}
    return document.plans[0]


def _proof(metrics):
    return next(item for item in reversed(metrics.rebalancing_verifications)
                if item.get("kind") == "funding")


def test_money_remains_a_priced_etf_and_does_not_become_spendable_bank_cash():
    metrics, config = _cube()
    before = portfolio_kpis(metrics)
    html, _ = _render(metrics, config)
    proof = _proof(metrics)
    doc = _plan_document(html)
    assert doc["attrs"]["data-tarzan-funding-status"] == "NON_EXECUTABLE"
    assert float(doc["fields"]["initial_settlement_cash_eur"]["attrs"]["data-eur"]) == 0.0
    assert float(doc["fields"]["cash_equivalent_securities_before_eur"]["attrs"]["data-eur"]) == 1000.0
    assert proof["initial_settlement_cash_eur"] == 0.0
    assert "MONEY.MI" in html and "Cash &amp; Cash Equivalents" in html
    assert "bank cash" in html.lower() and "strategic reserve" in html.lower()
    assert "not broker instructions" in html
    assert portfolio_kpis(metrics) == before
    money = metrics.holdings_df.loc[metrics.holdings_df.ticker == "MONEY.MI"].iloc[0]
    assert money.quantity == 100.0 and money.current_value == 1000.0
    assert money.gain_pct == 100.0


@pytest.mark.parametrize("case,kwargs,expected", [
    ("no_bank_cash", {}, "NON_EXECUTABLE"),
    ("contribution", {"contribution": 101.0}, "EXECUTABLE"),
    ("explicit_etf_sale", {"sell": True}, "EXECUTABLE"),
    ("no_sell", {"sell": True, "no_sell": True}, "NON_EXECUTABLE"),
    ("unknown_mechanics", {"unknown": True}, "NON_EXECUTABLE"),
    ("locked_cash", {"bank": 1000.0, "locked": True}, "NON_EXECUTABLE"),
])
def test_all_displayed_components_come_from_the_final_source_proof(case, kwargs, expected):
    metrics, config = _cube(**kwargs)
    before = copy.deepcopy(metrics.rebalancing_verifications)
    html, _ = _render(metrics, config)
    proof = _proof(metrics)
    doc = _plan_document(html)
    assert proof["status"] == expected
    assert doc["attrs"]["data-tarzan-funding-status"] == expected
    for key in _PROOF_FIELDS:
        assert float(doc["fields"][key]["attrs"]["data-eur"]) == proof[key]
    if case == "locked_cash":
        for key in ("locked_settlement_cash_eur", "initial_free_settlement_cash_eur",
                    "ending_free_settlement_cash_eur"):
            assert float(doc["fields"][key]["attrs"]["data-eur"]) == proof[key]
        assert "locked" in html.lower() and "free bank cash" in html.lower()
    assert metrics.rebalancing_verifications == before


def test_executable_means_only_selected_price_model_checks_not_broker_acceptance():
    metrics, config = _cube(contribution=101.0)
    html, _ = _render(metrics, config)
    doc = _plan_document(html)
    assert doc["attrs"]["data-tarzan-broker-execution-verified"] == "false"
    assert _SCOPE in unescape(html)
    assert "executable as it stands" not in html
    assert "executable as they stand" not in html
    assert "funding_initial_cash_eur" not in html
    assert "CHECKS PASSED" in html


def test_invested_allocation_denominator_is_named_as_strategic_not_spending_cash():
    metrics, config = _cube()
    allocation = sections._build_allocation(_context(metrics, config))
    reserve = next(row for row in allocation["rows"] if row.get("is_eur"))
    assert "reserve" in reserve["name"].lower()
    diversification = sections._build_diversification(_context(metrics, config))["html"]
    assert "Strategic reserve" in unescape(diversification)
    assert "not bank cash available for trades" in unescape(diversification)
    assert "Cash is outside invested capital." not in diversification


@pytest.mark.parametrize("key,value", list(product(
    _PROOF_FIELDS, (None, float("nan"), float("inf"), True, "unknown"),
)))
def test_missing_or_nonfinite_amount_is_unavailable_never_invented_zero(key, value):
    metrics, config = _cube(contribution=101.0)
    _proof(metrics)[key] = value
    html, _ = _render(metrics, config)
    doc = _plan_document(html)
    field = doc["fields"][key]
    assert field["attrs"]["data-eur"] == ""
    assert field["text"].strip() == "—"
    assert doc["attrs"]["data-tarzan-funding-status"] == "UNKNOWN"
    assert "CHECKS PASSED" not in html


@pytest.mark.parametrize("key", _PROOF_FLAGS)
@pytest.mark.parametrize("value", [False, None, "true"])
def test_unproven_invariant_never_reads_as_success(key, value):
    metrics, config = _cube(contribution=101.0)
    _proof(metrics)[key] = value
    optimizer = sections._build_optimizer(_context(metrics, config))
    assert optimizer["plans"][0]["executable"] is False
    html, _ = _render(metrics, config)
    assert "CHECKS PASSED" not in html
    assert "Draft" in html or "UNKNOWN" in html


@pytest.mark.parametrize("proof_status", [None, "", "UNKNOWN", "<img src=x onerror=alert(1)>"])
def test_unknown_proof_status_never_becomes_active_executable_instructions(proof_status):
    metrics, config = _cube(contribution=101.0)
    _proof(metrics)["status"] = proof_status
    html, _ = _render(metrics, config)
    assert _plan_document(html)["attrs"]["data-tarzan-funding-status"] == "UNKNOWN"
    assert "CHECKS PASSED" not in html
    assert "<img src=x" not in html


def test_missing_final_proof_is_visible_as_unknown_not_a_successful_plan():
    metrics, config = _cube()
    metrics.rebalancing_verifications = [
        item for item in metrics.rebalancing_verifications if item["kind"] != "funding"
    ]
    metrics.rebalancing_plans[0]["verifications"] = metrics.rebalancing_verifications
    html, _ = _render(metrics, config)
    doc = _plan_document(html)
    assert doc["attrs"]["data-tarzan-funding-status"] == "UNKNOWN"
    assert "CHECKS PASSED" not in html
    assert "funding proof unavailable" in html.lower()


def test_only_last_final_proof_controls_display_not_an_earlier_draft():
    metrics, config = _cube()
    previous = copy.deepcopy(_proof(metrics))
    previous["status"] = "EXECUTABLE"
    previous["ending_settlement_cash_eur"] = 999999.0
    metrics.rebalancing_verifications.insert(0, previous)
    html, _ = _render(metrics, config)
    doc = _plan_document(html)
    assert doc["attrs"]["data-tarzan-funding-status"] == "NON_EXECUTABLE"
    assert float(doc["fields"]["ending_settlement_cash_eur"]["attrs"]["data-eur"]) == -101.0
    assert "999999" not in html


@pytest.mark.parametrize("actions", [False, True])
def test_non_executable_proof_takes_precedence_over_trade_count(actions):
    metrics, config = _cube(buy=actions, buffer=3000.0)
    context = build_context(metrics, config, ai_summary="")
    assert context["hero"]["rebal_label"] == "Not executable"
    assert context["hero"]["rebal_executable"] is False
    assert "not executable" in context["headline"]["text"]
    assert "not executable" in context["preheader"]
    html, _ = _render(metrics, config)
    assert "Already on target" not in html
    assert "draft only" in html.lower()
    assert _plan_document(html)["attrs"]["data-tarzan-funding-status"] == "NON_EXECUTABLE"


def test_no_action_with_residual_target_drift_is_not_fabricated_alignment():
    metrics, config = _cube(buy=False)
    assert _proof(metrics)["status"] == "EXECUTABLE"
    assert any(check["status"] == "⚠ PARTIAL" for check in metrics.rebalancing_verifications
               if check["kind"] != "funding")
    context = build_context(metrics, config, ai_summary="")
    assert context["hero"]["rebal_label"] != "Aligned"
    assert "allocation is on target" not in context["headline"]["text"]
    html, _ = _render(metrics, config)
    assert "No trades proposed" in html
    assert "Already on target" not in html
    assert _plan_document(html)["attrs"]["data-tarzan-target-alignment"] == "unproven"


def test_zero_actions_can_be_aligned_only_with_nonempty_target_evidence():
    metrics, config = _cube(buy=False, buffer=1000.0)
    context = build_context(metrics, config, ai_summary="")
    assert context["hero"]["rebal_label"] == "Aligned"
    html, _ = _render(metrics, config)
    assert "Checked targets within tolerance" in html
    assert _plan_document(html)["attrs"]["data-tarzan-target-alignment"] == "proven"
    metrics.rebalancing_verifications[:] = [_proof(metrics)]
    context = build_context(metrics, config, ai_summary="")
    assert context["hero"]["rebal_label"] != "Aligned"


@pytest.mark.parametrize("defect", ["absent", "empty", "partial", "unknown", "no_solution"])
def test_missing_or_failed_target_evidence_does_not_prove_alignment(defect):
    metrics, config = _cube(buy=False, buffer=1000.0)
    target = next(item for item in metrics.rebalancing_verifications if item["kind"] == "cash")
    if defect == "absent":
        metrics.rebalancing_verifications[:] = [_proof(metrics)]
    elif defect == "empty":
        for item in metrics.rebalancing_verifications:
            item["items"] = []
    elif defect == "partial":
        target["status"] = "⚠ PARTIAL"
    elif defect == "unknown":
        target["status"] = "UNKNOWN"
    else:
        target["no_solution"] = True
    html, _ = _render(metrics, config)
    assert _plan_document(html)["attrs"]["data-tarzan-target-alignment"] == "unproven"
    assert "Checked targets within tolerance" not in html


def test_single_legacy_no_action_funding_failure_is_not_silently_hidden():
    metrics, config = _cube(buy=False, buffer=3000.0)
    metrics.rebalancing_plans = None
    html, _ = _render(metrics, config)
    assert _plan_document(html)["attrs"]["data-tarzan-funding-status"] == "NON_EXECUTABLE"
    assert "Not executable" in html


def test_positive_subcent_residual_is_not_negative_cash_or_a_shortfall():
    metrics, config = _cube(contribution=101.002)
    proof = _proof(metrics)
    assert proof["status"] == "EXECUTABLE" and proof["ending_settlement_cash_eur"] > 0
    html, _ = _render(metrics, config)
    doc = _plan_document(html)
    assert float(doc["fields"]["ending_settlement_cash_eur"]["attrs"]["data-eur"]) > 0
    assert doc["attrs"]["data-tarzan-funding-status"] == "EXECUTABLE"
    assert "shortfall" not in html.lower()


@pytest.mark.parametrize("make_label", [str, Markup])
def test_plan_labels_are_escaped_once_and_cannot_inject_funding_markup(make_label):
    metrics, config = _cube(contribution=101.0)
    payload = '<img src=x onerror="bad()"><div data-tarzan-funding-plan="99">A & B</div>'
    metrics.rebalancing_plans[0]["label"] = make_label(payload)
    html, _ = _render(metrics, config)
    assert "<img src=x" not in html
    assert "onerror=&quot;" in html or "onerror=&#34;" in html
    assert "A &amp; B" in html and "A &amp;amp; B" not in html
    assert payload in unescape(html)
    assert set(_FundingDocument(html).plans) == {0}


def test_ui_change_does_not_modify_session_euro_or_any_financial_output():
    metrics, config = _cube()
    before = portfolio_kpis(metrics)
    hero = sections._build_hero(_context(metrics, config))
    session = next(tile for tile in hero["tiles"] if tile["label"] == "Session")
    expected_eur = sections._eur_smart(metrics.invested_value * .01 / 1.01, signed=True)
    assert expected_eur in session["caption"]
    _render(metrics, config)
    assert portfolio_kpis(metrics) == before


@pytest.mark.parametrize("tamper", [
    "status", "amount", "missing_field", "alignment", "broker_verified",
    "scope", "missing_plan", "trade_count", "visible_amount", "visible_badge",
    "action_dropped", "duplicate_field", "duplicate_plan",
])
def test_independent_semantic_guard_rejects_funding_or_scope_misrepresentation(tamper):
    metrics, config = _cube(contribution=101.0)
    html, audit = _render(metrics, config)
    assert validate_newsletter_semantics(metrics, audit, html) == ()
    if tamper == "status":
        html = html.replace('data-tarzan-funding-status="EXECUTABLE"',
                            'data-tarzan-funding-status="UNKNOWN"', 1)
    elif tamper == "amount":
        html = html.replace('data-tarzan-funding-field="initial_settlement_cash_eur" data-eur="0.0"',
                            'data-tarzan-funding-field="initial_settlement_cash_eur" data-eur="1000.0"', 1)
    elif tamper == "missing_field":
        html = html.replace('data-tarzan-funding-field="initial_settlement_cash_eur"',
                            'data-tarzan-funding-field="synthetic_wrong_field"', 1)
    elif tamper == "alignment":
        html = html.replace('data-tarzan-target-alignment="unproven"',
                            'data-tarzan-target-alignment="proven"', 1)
    elif tamper == "broker_verified":
        html = html.replace('data-tarzan-broker-execution-verified="false"',
                            'data-tarzan-broker-execution-verified="true"', 1)
    elif tamper == "scope":
        html = html.replace("are not verified", "are verified", 1)
    elif tamper == "missing_plan":
        html = html.replace("data-tarzan-funding-plan", "synthetic-invisible-plan")
    elif tamper == "trade_count":
        html = html.replace('data-tarzan-action-count="1"', 'data-tarzan-action-count="0"', 1)
    elif tamper == "visible_amount":
        html = html.replace('data-eur="0.0">€0.00</span>',
                            'data-eur="0.0">€1,000.00</span>', 1)  # synthetic tampered amount
    elif tamper == "visible_badge":
        html = html.replace(">CHECKS PASSED</span>", ">BROKER VERIFIED</span>", 1)
    elif tamper == "action_dropped":
        html = html.replace('data-tarzan-funding-action="true"', "", 1)
    elif tamper == "duplicate_field":
        html = html.replace('data-tarzan-funding-badge="true"',
                            'data-tarzan-funding-field="initial_settlement_cash_eur" '
                            'data-eur="0.0" data-tarzan-funding-badge="true"', 1)
    else:
        html = html.replace("</body>", '<div data-tarzan-funding-plan="0"></div></body>', 1)
    errors = validate_newsletter_semantics(metrics, audit, html)
    assert any("funding" in error.lower() or "broker" in error.lower() for error in errors)


@pytest.mark.parametrize("value", [pd.NA, np.bool_(True), -float("inf"), {}, []])
def test_additional_missing_types_never_bypass_unknown_or_semantic_checks(value):
    metrics, config = _cube(contribution=101.0)
    _proof(metrics)["initial_settlement_cash_eur"] = value
    html, audit = _render(metrics, config)
    doc = _plan_document(html)
    assert doc["attrs"]["data-tarzan-funding-status"] == "UNKNOWN"
    assert doc["fields"]["initial_settlement_cash_eur"]["text"].strip() == "—"
    assert validate_newsletter_semantics(metrics, audit, html) == ()


@pytest.mark.parametrize("value", [pd.NA, np.bool_(True), {}, [], 1])
def test_nonstring_status_is_unknown_not_an_exception(value):
    metrics, config = _cube(contribution=101.0)
    _proof(metrics)["status"] = value
    html, audit = _render(metrics, config)
    assert _plan_document(html)["attrs"]["data-tarzan-funding-status"] == "UNKNOWN"
    assert validate_newsletter_semantics(metrics, audit, html) == ()


@pytest.mark.parametrize("key", [
    "locked_settlement_cash_eur", "initial_free_settlement_cash_eur",
    "ending_free_settlement_cash_eur", "locked_cash_balances_satisfied",
])
def test_partial_locked_bank_proof_never_claims_success(key):
    metrics, config = _cube(bank=1000.0, locked=True, contribution=101.0)
    assert _proof(metrics)["status"] == "EXECUTABLE"
    _proof(metrics).pop(key)
    html, audit = _render(metrics, config)
    assert _plan_document(html)["attrs"]["data-tarzan-funding-status"] == "UNKNOWN"
    assert "CHECKS PASSED" not in html
    assert validate_newsletter_semantics(metrics, audit, html) == ()


def test_locked_cash_flag_cannot_stand_in_for_missing_bank_components():
    metrics, config = _cube(bank=1000.0, locked=True, contribution=101.0)
    for key in ("locked_settlement_cash_eur", "initial_free_settlement_cash_eur",
                "ending_free_settlement_cash_eur"):
        _proof(metrics).pop(key)
    html, audit = _render(metrics, config)
    doc = _plan_document(html)
    assert doc["attrs"]["data-tarzan-funding-status"] == "UNKNOWN"
    for key in ("locked_settlement_cash_eur", "initial_free_settlement_cash_eur",
                "ending_free_settlement_cash_eur"):
        assert doc["fields"][key]["text"].strip() == "—"
    assert validate_newsletter_semantics(metrics, audit, html) == ()


@pytest.mark.parametrize("defect", ["status", "alignment_text", "missing_outcome"])
def test_no_trade_visible_outcome_has_its_own_independent_guard(defect):
    metrics, config = _cube(buy=False, buffer=1000.0)
    html, audit = _render(metrics, config)
    assert validate_newsletter_semantics(metrics, audit, html) == ()
    if defect == "status":
        html = html.replace('data-tarzan-funding-outcome="aligned"',
                            'data-tarzan-funding-outcome="unproven"', 1)
    elif defect == "alignment_text":
        html = html.replace("Checked targets within tolerance; no trades proposed.",
                            "Already on target: no trades needed.", 1)
    else:
        html = html.replace('data-tarzan-funding-outcome="aligned"', "", 1)
    assert validate_newsletter_semantics(metrics, audit, html)


@pytest.mark.parametrize("state", ["unproven", "unknown", "not-executable"])
def test_unknown_or_failed_no_trade_text_cannot_claim_alignment(state):
    metrics, config = _cube(buy=False, buffer=3000.0 if state == "not-executable" else 900.0)
    if state == "unknown":
        _proof(metrics)["status"] = "UNKNOWN"
    html, audit = _render(metrics, config)
    assert validate_newsletter_semantics(metrics, audit, html) == ()
    if state == "not-executable":
        source_text = "No executable trade set was produced; draft only."
    else:
        source_text = "No trades proposed. Target alignment has not been proven."
    original_html = html
    html = html.replace(source_text, "Checked targets within tolerance; no trades proposed.", 1)
    assert html != original_html
    assert validate_newsletter_semantics(metrics, audit, html)


def test_two_plans_keep_separate_proofs_and_active_is_not_broker_acceptance():
    metrics, config = _cube(contribution=101.0)
    alternative = copy.deepcopy(metrics.rebalancing_plans[0])
    alternative["label"] = "Synthetic alternative"
    alternative["no_sell"] = not config.rebalancing_no_sell
    funding = next(check for check in reversed(alternative["verifications"])
                   if check["kind"] == "funding")
    funding["no_sell_satisfied"] = False
    funding["status"] = "NON_EXECUTABLE"
    metrics.rebalancing_plans.append(alternative)
    before = portfolio_kpis(metrics)
    html, audit = _render(metrics, config)
    document = _FundingDocument(html)
    assert set(document.plans) == {0, 1}
    assert document.plans[0]["attrs"]["data-tarzan-funding-status"] == "EXECUTABLE"
    assert document.plans[1]["attrs"]["data-tarzan-funding-status"] == "NON_EXECUTABLE"
    assert "2 modeled proposals; 1 passed funding" in html
    assert "ACTIVE marks the configured proposal, not a broker order." in html
    assert validate_newsletter_semantics(metrics, audit, html) == ()
    assert portfolio_kpis(metrics) == before


def test_no_planner_or_benchmark_contract_remains_legacy_compatible():
    metrics, config = _cube(buy=False)
    metrics.rebalancing_plans = None
    metrics.rebalancing_verifications = []
    html, audit = _render(metrics, config)
    assert not _FundingDocument(html).plans
    assert validate_newsletter_semantics(metrics, audit, html) == ()


@pytest.mark.parametrize("kind,field", [
    ("cash", "actual_eur"), ("cash", "target_eur"),
    ("asset", "actual_pct"), ("asset", "target_pct"),
])
@pytest.mark.parametrize("value", [None, pd.NA, float("nan"), float("inf"), True])
def test_nonempty_target_rows_still_need_known_numeric_evidence(kind, field, value):
    metrics, config = _cube(buy=False, buffer=1000.0)
    target = next(check for check in metrics.rebalancing_verifications if check["kind"] == kind)
    target["items"][0][field] = value
    context = build_context(metrics, config, ai_summary="")
    assert context["hero"]["rebal_label"] != "Aligned"
    html, audit = _render(metrics, config)
    assert _plan_document(html)["attrs"]["data-tarzan-target-alignment"] == "unproven"
    assert validate_newsletter_semantics(metrics, audit, html) == ()


@pytest.mark.parametrize("value", [np.int64(2**53 + 17), np.int64(-(2**63)),
                                   np.float64(0.002)])
def test_numeric_source_scalar_types_keep_exact_raw_markers_and_cent_display(value):
    metrics, config = _cube()
    _proof(metrics)["initial_settlement_cash_eur"] = value
    html, audit = _render(metrics, config)
    field = _plan_document(html)["fields"]["initial_settlement_cash_eur"]
    assert field["attrs"]["data-eur"] == str(value)
    display_value = int(value) if isinstance(value, np.integer) else float(value)
    sign = "−" if round(display_value, 2) < 0 else ""
    expected = (
        f"{sign}€{abs(display_value):,}.00" if isinstance(value, np.integer)
        else f"{sign}€{abs(display_value):,.2f}"
    )
    assert field["text"].strip() == expected
    assert validate_newsletter_semantics(metrics, audit, html) == ()
