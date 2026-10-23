"""Pre-delivery semantic invariants for the portfolio newsletter.

The renderer is allowed to format canonical analytical data, but it is not
allowed to resolve another market symbol or relabel a chart with a return from
a different window.  This module verifies those rules on the completed render
before any durable delivery claim or SMTP invocation.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from datetime import date
from html.parser import HTMLParser
from numbers import Integral, Real

logger = logging.getLogger("tarzan.newsletter")


# The sign is OPTIONAL. ``_pct(..., signed=True)`` deliberately omits it when
# the value rounds to zero (``_sign_for`` returns "" so a residual \u22120.004%
# prints as "0.00%" rather than as a signed zero reading like a real move
# down). Requiring a sign here made that correct rendering unparseable, and an
# unparseable label is reported as disagreeing with its endpoint \u2014 blocking
# delivery over a line that was drawn and labelled correctly.
_DISPLAYED_PERCENT_RE = re.compile(r"([+\-\u2212]?)(\d+(?:\.\d+)?)%$")
_EXPLICIT_EUR_AMOUNT_RE = re.compile(
    r"\bEUR\s*[+\-\u2212]?\d|[+\-\u2212]?\d[\d.,]*\s*EUR\b",
)


def _text(value: object) -> str:
    return str(value or "").strip()


def _identity_map(frame, name_column: str, ticker_column: str) -> dict[str, set[str]]:
    """Project a frame into ``name -> exact ticker set`` without coercing NaN."""
    if frame is None or getattr(frame, "empty", True):
        return {}
    if name_column not in frame.columns or ticker_column not in frame.columns:
        return {}
    identities: dict[str, set[str]] = {}
    for _, row in frame.iterrows():
        name = _text(row.get(name_column))
        ticker = _text(row.get(ticker_column))
        if name:
            identities.setdefault(name, set()).add(ticker)
    return identities


def _check_projection(
    errors: list[str],
    projection: str,
    actual: Mapping[str, set[str]],
    expected: Mapping[str, str],
) -> None:
    for name, ticker in expected.items():
        seen = actual.get(name)
        if seen is None:
            errors.append(f"{projection} is missing benchmark {name}")
        elif seen != {ticker}:
            errors.append(
                f"{projection} uses {sorted(seen)!r} for {name}; expected {ticker!r}"
            )


def _displayed_percent(label: str) -> float | None:
    match = _DISPLAYED_PERCENT_RE.search(label)
    if match is None:
        return None
    value = float(match.group(2))
    return -value if match.group(1) in {"-", "\u2212"} else value


class _FundingMarkup(HTMLParser):
    """Read the delivered HTML, independently of section/context builders."""

    _VOID = frozenset({
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    })

    def __init__(self, document: str):
        super().__init__(convert_charrefs=True)
        self.plans: list[dict] = []
        self.frames: list[tuple] = []
        self.feed(document)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        _, plan, field, component = (
            self.frames[-1] if self.frames else (None, None, None, None)
        )
        if "data-tarzan-funding-plan" in attrs:
            plan = len(self.plans)
            field = component = None
            self.plans.append({
                "attrs": attrs, "fields": {}, "components": {}, "actions": 0,
            })
        if plan is not None:
            record = self.plans[plan]
            if "data-tarzan-funding-field" in attrs:
                name = attrs["data-tarzan-funding-field"]
                field = {"attrs": attrs, "text": ""}
                record["fields"].setdefault(name, []).append(field)
            for kind in ("badge", "scope", "outcome"):
                if f"data-tarzan-funding-{kind}" in attrs:
                    component = {"attrs": attrs, "text": ""}
                    record["components"].setdefault(kind, []).append(component)
            if attrs.get("data-tarzan-funding-action") == "true":
                record["actions"] += 1
        if tag not in self._VOID:
            self.frames.append((tag, plan, field, component))

    def handle_endtag(self, tag):
        for index in range(len(self.frames) - 1, -1, -1):
            if self.frames[index][0] == tag:
                del self.frames[index:]
                break

    def handle_data(self, text):
        if self.frames:
            _, _, field, component = self.frames[-1]
            if field is not None:
                field["text"] += text
            if component is not None:
                component["text"] += text


def _validate_funding_markup(metrics, document: str) -> list[str]:
    """Reconcile model-proof labels/amounts without sharing renderer helpers.

    Cash & Cash Equivalents is a strategic classification, not bank cash.
    Funding/position/policy success never proves broker lot sizes, settlement
    timing or execution prices. This guard reads only the final source proof;
    it does not recalculate trades, target percentages or portfolio returns.
    """
    source_plans = getattr(metrics, "rebalancing_plans", None)
    if not source_plans:
        actions = getattr(metrics, "rebalancing_suggestions", None) or []
        checks = getattr(metrics, "rebalancing_verifications", None) or []
        if not actions and not checks:
            return []
        source_plans = [{"suggestions": actions, "verifications": checks}]

    parsed = _FundingMarkup(document)
    errors: list[str] = []
    if len(parsed.plans) != len(source_plans):
        return ["funding display plan count differs from source plans"]
    amount_fields = (
        "initial_settlement_cash_eur", "ending_settlement_cash_eur",
        "cash_equivalent_securities_before_eur", "cash_equivalent_securities_after_eur",
        "initial_cash_equivalents_eur", "ending_cash_equivalents_eur",
        "cash_equivalents_target_eur", "protected_cash_eur",
        "external_contribution_eur", "gross_sales_eur", "gross_purchases_eur",
        "estimated_tax_eur", "fees_eur", "residual_eur",
    )
    required_flags = (
        "position_invariants_satisfied", "protected_cash_satisfied",
        "settlement_cash_satisfied", "frozen_positions_satisfied",
        "no_sell_satisfied", "serialized_actions_satisfied",
        "action_identity_satisfied", "units_invariants_satisfied",
        "wealth_invariants_satisfied", "cash_mechanics_satisfied",
        "non_cost_only_progress_satisfied",
    )
    locked_fields = (
        "locked_settlement_cash_eur", "initial_free_settlement_cash_eur",
        "ending_free_settlement_cash_eur",
    )
    target_kinds = {
        "asset", "geography", "per_holding_equity", "per_holding_fi",
        "per_holding_portfolio", "cash",
    }
    expected_scope = (
        "Funding, position and policy checks use selected prices only. "
        "Broker lot sizes/fractional orders, settlement timing and execution prices "
        "are not verified. These are modeled proposals, not broker instructions."
    )

    def numeric(value) -> bool:
        if isinstance(value, bool) or not isinstance(value, Real):
            return False
        try:
            return math.isfinite(value)
        except (OverflowError, TypeError, ValueError):
            return False

    for index, (source, rendered) in enumerate(zip(source_plans, parsed.plans)):
        prefix = f"funding plan {index}"
        if not isinstance(source, Mapping):
            errors.append(f"{prefix} source is not a plan record")
            continue
        checks = source.get("verifications") or []
        actions = source.get("suggestions") or []
        proof = None
        target_checks = []
        uncertain_targets = False
        source_infeasible = False
        for check in checks:
            if not isinstance(check, Mapping):
                uncertain_targets = True
                continue
            kind = check.get("kind")
            if isinstance(kind, str):
                if kind == "funding":
                    proof = check
                elif kind in target_kinds:
                    target_checks.append(check)
            if check.get("no_solution") is not None and check.get("no_solution") is not False:
                uncertain_targets = True
                source_infeasible = True

        for target in target_checks:
            items = target.get("items")
            if not isinstance(items, (list, tuple)):
                uncertain_targets = True
                continue
            expected_fields = (
                ("actual_eur", "target_eur") if target["kind"] == "cash"
                else ("actual_pct", "target_pct")
            )
            if any(
                not isinstance(item, Mapping)
                or any(not numeric(item.get(name)) for name in expected_fields)
                for item in items
            ):
                uncertain_targets = True

        fields, flags = amount_fields, required_flags
        if proof is not None and (
            "locked_cash_balances_satisfied" in proof
            or any(name in proof for name in locked_fields)
        ):
            fields += locked_fields
            flags += ("locked_cash_balances_satisfied",)
        status = "UNKNOWN"
        if proof is not None:
            source_status = proof.get("status")
            if isinstance(source_status, str) and source_status == "NON_EXECUTABLE":
                status = "NON_EXECUTABLE"
            elif (
                isinstance(source_status, str) and source_status == "EXECUTABLE"
                and all(proof.get(name) is True for name in flags)
                and all(numeric(proof.get(name)) for name in fields)
            ):
                status = "EXECUTABLE"
        aligned = bool(
            not actions and status == "EXECUTABLE" and not uncertain_targets
            and target_checks
            and any(isinstance(check.get("items"), (list, tuple)) and check["items"]
                    for check in target_checks)
            and all(isinstance(check.get("status"), str) and check["status"] == "✓ OK"
                    for check in target_checks)
        )
        expected_attributes = {
            "data-tarzan-funding-plan": str(index),
            "data-tarzan-funding-status": status,
            "data-tarzan-target-alignment": "proven" if aligned else "unproven",
            "data-tarzan-broker-execution-verified": "false",
            "data-tarzan-action-count": str(len(actions)),
        }
        for name, expected in expected_attributes.items():
            if rendered["attrs"].get(name) != expected:
                errors.append(f"{prefix} {name} disagrees with final source proof")
        if rendered["actions"] != len(actions):
            errors.append(f"{prefix} rendered trade count disagrees with source actions")
        if set(rendered["fields"]) != set(fields):
            errors.append(f"{prefix} displayed amount fields differ from final source proof")
        for name in fields:
            cells = rendered["fields"].get(name, [])
            if len(cells) != 1:
                errors.append(f"{prefix} amount {name} is missing or duplicated")
                continue
            value = proof.get(name) if proof is not None else None
            available = numeric(value)
            expected_raw = str(value) if available else ""
            expected_text = "—"
            if available:
                display_value = int(value) if isinstance(value, Integral) else value
                sign = "−" if round(display_value, 2) < 0 else ""
                expected_text = (
                    f"{sign}€{abs(display_value):,}.00" if isinstance(value, Integral)
                    else f"{sign}€{abs(display_value):,.2f}"
                )
            if (
                cells[0]["attrs"].get("data-eur") != expected_raw
                or cells[0]["text"].strip() != expected_text
            ):
                errors.append(f"{prefix} displayed amount {name} differs from final source proof")
        badge = (
            "CHECKS PASSED" if status == "EXECUTABLE"
            else "NOT EXECUTABLE · DRAFT" if status == "NON_EXECUTABLE"
            else "UNKNOWN · DRAFT"
        )
        badges = rendered["components"].get("badge", [])
        if len(badges) != 1 or badges[0]["text"].strip() != badge:
            errors.append(f"{prefix} visible model-check badge disagrees with source proof")
        scopes = rendered["components"].get("scope", [])
        if (
            len(scopes) != 1
            or scopes[0]["attrs"].get("data-tarzan-funding-scope") != "limited"
            or " ".join(scopes[0]["text"].split()) != expected_scope
        ):
            errors.append(f"{prefix} broker execution disclaimer is missing or incorrect")
        outcomes = rendered["components"].get("outcome", [])
        if actions:
            if outcomes:
                errors.append(f"{prefix} no-trade/alignment outcome contradicts source actions")
        else:
            outcome = (
                "not-executable" if status == "NON_EXECUTABLE"
                else "aligned" if aligned else "unproven"
            )
            if status == "NON_EXECUTABLE":
                expected_outcome_text = "No executable trade set was produced; draft only."
            elif aligned:
                expected_outcome_text = "Checked targets within tolerance; no trades proposed."
            else:
                expected_outcome_text = (
                    "No trades proposed"
                    + ("; no feasible plan was produced" if source_infeasible else "")
                    + ". Target alignment has not been proven."
                )
            if (
                len(outcomes) != 1
                or outcomes[0]["attrs"].get("data-tarzan-funding-outcome") != outcome
                or outcomes[0]["text"].strip() != expected_outcome_text
            ):
                errors.append(f"{prefix} visible no-trade/alignment outcome disagrees with proof")
    return errors


class _SessionMarkup(HTMLParser):
    """Inspect the actual session tile, not a renderer's context audit."""

    def __init__(self, document):
        super().__init__(convert_charrefs=True)
        self.records = []
        self.frames = []
        self.feed(document)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        _, record, component = self.frames[-1] if self.frames else (None, None, None)
        if "data-tarzan-session-move" in attrs:
            record = {"attrs": attrs, "components": {}, "duplicate_attrs": len(attrs) != len(attributes)}
            self.records.append(record)
            component = None
        if record is not None:
            for kind in ("percent", "caption"):
                if f"data-tarzan-session-{kind}" in attrs:
                    component = {"text": "", "duplicate_attrs": len(attrs) != len(attributes)}
                    record["components"].setdefault(kind, []).append(component)
        if tag not in _FundingMarkup._VOID:
            self.frames.append((tag, record, component))

    def handle_endtag(self, tag):
        for index in range(len(self.frames) - 1, -1, -1):
            if self.frames[index][0] == tag:
                del self.frames[index:]
                break

    def handle_data(self, text):
        if self.frames and self.frames[-1][2] is not None:
            self.frames[-1][2]["text"] += text


def _validate_session_markup(metrics, document):
    """Independently reconcile euro attribution with its original NAV facts."""
    import pandas as pd
    from tarzan.contracts.session_move import DailyReturnStep, SessionMoveEvidence
    from tarzan.engine.stats import normalize_session_index, window_anchor

    def finite(value):
        try:
            return (
                isinstance(value, Real) and not isinstance(value, bool)
                and math.isfinite(value)
            )
        except (TypeError, ValueError, OverflowError):
            return False

    def equal(first, second, *, absolute=1e-9):
        return finite(first) and finite(second) and math.isclose(
            float(first), float(second), rel_tol=1e-12, abs_tol=absolute,
        )

    percent = (getattr(metrics, "performance", None) or {}).get("1d")
    records = _SessionMarkup(document).records
    if not finite(percent):
        return ["session display claims an unavailable return"] if records else []
    if len(records) != 1:
        return ["session display is missing or duplicates the source return"]
    record = records[0]
    errors = []
    attrs = record["attrs"]
    components = record["components"]
    if record["duplicate_attrs"] or attrs.get("data-tarzan-session-move") != "true":
        errors.append("session display has an ambiguous evidence marker")
    try:
        raw_percent = float(attrs.get("data-tarzan-session-return", ""))
    except (TypeError, ValueError, OverflowError):
        raw_percent = float("nan")
    if not equal(raw_percent, percent, absolute=1e-10):
        errors.append("session display return differs from source percentage")
    percent_parts, caption_parts = components.get("percent", []), components.get("caption", [])
    if len(percent_parts) != 1 or len(caption_parts) != 1:
        return errors + ["session display is missing or duplicates visible components"]
    if any(part["duplicate_attrs"] for part in (*percent_parts, *caption_parts)):
        errors.append("session display has ambiguous visible components")
    percent_text = percent_parts[0]["text"].strip()
    sign = (
        "" if abs(round(float(percent), 2)) < 1e-9
        else "+" if percent > 0 else "\u2212"
    )
    expected_percent = f"{sign}{abs(float(percent)):.2f}%"
    if _DISPLAYED_PERCENT_RE.fullmatch(percent_text) is None or percent_text != expected_percent:
        errors.append("session visible percentage differs from source return")
    caption = caption_parts[0]["text"].strip()
    proof = getattr(metrics, "session_move_evidence", None)
    if type(proof) is not SessionMoveEvidence or proof.availability not in ("AVAILABLE", "DEGRADED"):
        if (
            attrs.get("data-tarzan-session-availability") != "UNAVAILABLE"
            or attrs.get("data-tarzan-session-profit") != ""
            or not caption.startswith("EUR attribution unavailable")
            or "€" in caption
            or _EXPLICIT_EUR_AMOUNT_RE.search(caption) is not None
            or any(attrs.get(f"data-tarzan-session-{field}", "") for field in (
                "anchor", "end", "source-anchor", "source-end",
            ))
        ):
            errors.append("session display invents unavailable euro attribution")
        return errors

    dates = (proof.anchor_date, proof.end_date, proof.source_anchor_date, proof.source_end_date)
    values = (proof.return_pct, proof.profit_eur, proof.external_flow_eur)
    if not all(type(value) is date for value in dates) or not all(finite(value) for value in values):
        return errors + ["session source evidence contains unavailable dates or amounts"]
    if not (
        proof.anchor_date < proof.end_date and proof.source_anchor_date < proof.source_end_date
        and equal(proof.return_pct, percent, absolute=1e-10)
        and getattr(metrics, "history_availability", "AVAILABLE") in ("AVAILABLE", "DEGRADED")
        and not (
            getattr(metrics, "history_availability", "AVAILABLE") == "DEGRADED"
            and proof.availability != "DEGRADED"
        )
        and proof.source_basis == getattr(metrics, "history_source_basis", "UNPREPARED")
    ):
        errors.append("session source evidence contradicts its return window")
    source_attrs = {
        "availability": proof.availability, "basis": proof.source_basis,
        "anchor": proof.anchor_date.isoformat(), "end": proof.end_date.isoformat(),
        "source-anchor": proof.source_anchor_date.isoformat(),
        "source-end": proof.source_end_date.isoformat(),
    }
    for field, expected in source_attrs.items():
        if attrs.get(f"data-tarzan-session-{field}") != expected:
            errors.append(f"session display {field} differs from source evidence")
    try:
        raw_profit = float(attrs.get("data-tarzan-session-profit", ""))
    except (TypeError, ValueError, OverflowError):
        raw_profit = float("nan")
    if not equal(raw_profit, proof.profit_eur):
        errors.append("session display euros differ from paired source evidence")
    # Deliberately independent of the newsletter's currency formatting helper.
    money = ("+" if proof.profit_eur >= 0 else "\u2212") + f"€{abs(proof.profit_eur):,.2f}"
    prefix = [
        money,
        "flow-adjusted NAV" if proof.source_basis == "ORDER_FLOW_ADJUSTED_NAV"
        else "historical holdings basket",
    ]
    if proof.availability == "DEGRADED":
        prefix.append("estimated historical data")
    source_notes = []
    if (
        proof.source_anchor_date != proof.anchor_date
        or proof.source_end_date != proof.end_date
    ):
        source_notes.append(f"source {proof.source_anchor_date} to {proof.source_end_date}")
        prefix.extend(source_notes)
    if (
        not caption.startswith(" \u00b7 ".join(prefix) + " \u00b7 ")
        or re.findall(r"source \S+ to \S+", caption) != source_notes
        or caption.count("€") != 1
        or _EXPLICIT_EUR_AMOUNT_RE.search(caption) is not None
    ):
        errors.append("session visible euro amount differs from source evidence")
    if proof.availability == "DEGRADED" and "estimated historical data" not in caption:
        errors.append("session display hides degraded historical evidence")
    history = getattr(metrics, "portfolio_history", None)
    if not (
        isinstance(history, pd.Series) and isinstance(history.index, pd.DatetimeIndex)
        and history.index.is_unique and history.index.is_monotonic_increasing
    ):
        return errors + ["session source history cannot prove its displayed window"]
    observed = history.dropna()
    anchor = window_anchor(observed, "1d")
    if anchor is None or observed.empty:
        return errors + ["session source history has no measured window"]
    start_raw, end_raw = observed.loc[:anchor].iloc[-1], observed.iloc[-1]
    if not finite(start_raw) or not finite(end_raw):
        return errors + ["session source history contains unavailable endpoint values"]
    start, end = float(start_raw), float(end_raw)
    if not (
        finite(start) and start > 0 and finite(end) and end >= 0
        and anchor.date() == proof.anchor_date
        and observed.index[-1].date() == proof.end_date
        and equal((end / start - 1.0) * 100.0, percent, absolute=1e-10)
    ):
        errors.append("session displayed return is not its original history window")
    if proof.source_basis == "HOLDINGS_PRICE_AGGREGATE":
        if (
            proof.steps or proof.external_flow_eur != 0
            or proof.source_anchor_date != proof.anchor_date
            or proof.source_end_date != proof.end_date
            or not equal(proof.profit_eur, end - start)
        ):
            errors.append("session euro attribution differs from the historical holdings basket")
        if "historical holdings basket" not in caption:
            errors.append("session display mislabels its historical holdings basis")
        return errors
    if proof.source_basis != "ORDER_FLOW_ADJUSTED_NAV":
        return errors + ["session euro attribution has an unsupported source basis"]
    if "flow-adjusted NAV" not in caption:
        errors.append("session display mislabels its flow-adjusted basis")
    actual = getattr(metrics, "actual_value_series", None)
    flows = getattr(metrics, "external_flows", None)
    if not (
        isinstance(actual, pd.Series) and isinstance(actual.index, pd.DatetimeIndex)
        and actual.index.is_unique and actual.index.is_monotonic_increasing
        and isinstance(flows, Mapping)
    ):
        return errors + ["session euro attribution lacks original values or capital flows"]
    if any(type(day) is not date or not finite(amount) for day, amount in flows.items()):
        return errors + ["session original capital flows contain unavailable dates or amounts"]
    actual = actual.copy(deep=False)
    actual.index = normalize_session_index(actual.index)
    if not actual.index.is_unique:
        return errors + ["session original values contain duplicate native sessions"]
    try:
        native = pd.Series(list(actual.index.date), index=actual.index, dtype=object)
        native = native.where(actual.notna()).resample("B").last()
        if (
            native.loc[pd.Timestamp(proof.anchor_date)] != proof.source_anchor_date
            or native.loc[pd.Timestamp(proof.end_date)] != proof.source_end_date
        ):
            errors.append("session attribution changes its native observation dates")
        cursor, factor, total_profit, total_flow = proof.source_anchor_date, 1.0, 0.0, 0.0
        previous_value = None
        for step in proof.steps:
            if not (
                type(step) is DailyReturnStep and step.state == "MEASURED"
                and type(step.start_date) is date and type(step.end_date) is date
                and step.start_date == cursor and cursor < step.end_date <= proof.source_end_date
                and all(finite(value) for value in (
                    step.starting_value_eur, step.ending_value_eur, step.external_flow_eur,
                    step.factor, step.profit_eur,
                ))
                and step.starting_value_eur > 0 and step.ending_value_eur >= 0
                and step.factor > 0
            ):
                errors.append("session euro attribution contains an unmeasured NAV step")
                break
            before_flow = step.ending_value_eur - step.external_flow_eur
            expected_flow = sum(
                amount for day, amount in flows.items()
                if step.start_date < day <= step.end_date
            )
            if (
                before_flow <= 0
                or not equal(step.starting_value_eur, actual.loc[pd.Timestamp(step.start_date)])
                or not equal(step.ending_value_eur, actual.loc[pd.Timestamp(step.end_date)])
                or not equal(step.external_flow_eur, expected_flow)
                or not equal(step.factor, before_flow / step.starting_value_eur, absolute=1e-12)
                or not equal(step.profit_eur, before_flow - step.starting_value_eur)
                or (previous_value is not None and not equal(step.starting_value_eur, previous_value))
            ):
                errors.append("session euro attribution contradicts original NAV values or flows")
            cursor, previous_value = step.end_date, step.ending_value_eur
            factor *= step.factor
            total_profit += step.profit_eur
            total_flow += step.external_flow_eur
        if (
            not proof.steps or cursor != proof.source_end_date
            or not equal((factor - 1.0) * 100.0, percent, absolute=1e-10)
            or not equal(total_profit, proof.profit_eur)
            or not equal(total_flow, proof.external_flow_eur)
        ):
            errors.append("session euro attribution does not cover the exact return window")
    except (TypeError, ValueError, KeyError, OverflowError):
        errors.append("session euro attribution cannot reconcile original NAV facts")
    return errors


def validate_newsletter_semantics(
    metrics,
    semantic_audit: Mapping[str, object] | None,
    newsletter_html: str,
) -> tuple[str, ...]:
    """Return every identity/window violation that must block delivery.

    Empty legacy/test metrics without benchmark or planner contracts are ignored. A real
    MetricsEngine result always exposes either resolved benchmark tickers,
    explicit resolution errors, or a degraded preprocessing computer.
    """
    errors = _validate_funding_markup(metrics, newsletter_html)
    errors.extend(_validate_session_markup(metrics, newsletter_html))
    resolution_errors = tuple(
        _text(error)
        for error in (getattr(metrics, "benchmark_resolution_errors", ()) or ())
        if _text(error)
    )
    selected = {
        _text(name): _text(ticker)
        for name, ticker in (getattr(metrics, "benchmark_tickers", {}) or {}).items()
        if _text(name)
    }
    degraded = set(getattr(metrics, "degraded_computers", ()) or ())
    hp = getattr(metrics, "holding_performance", None)
    comparison = getattr(metrics, "benchmark_comparison", None)
    histories = getattr(metrics, "benchmark_histories", {}) or {}
    historical_risk = getattr(metrics, "historical_risk", None) or {}
    has_contract = bool(
        selected
        or resolution_errors
        or "_preprocess_benchmarks" in degraded
        or histories
        or not getattr(comparison, "empty", True)
    )
    if not has_contract:
        return tuple(errors)

    from tarzan.contracts.reference import reference_for
    reference = reference_for(metrics)

    definitions = {_text(name): _text(ticker)
                   for name, ticker in reference.benchmarks.items()}
    expected_names = set(definitions)
    selected_names = set(selected)

    # Only the benchmarks the portfolio is actually MEASURED against — the geo
    # and alpha/beta references drawn in the "vs market" chart — are required.
    # The other ~50 configured names are tracked for reference; a transient
    # Yahoo outage on one of them (X25E returned no history on 20 Aug 2026 under
    # runner throttling) must drop that one row, NOT block the whole digest —
    # three sends were lost to exactly this. Availability degrades; identity is
    # still fully enforced below on whatever DID resolve.
    critical = {_text(reference.benchmark_geo), _text(reference.benchmark_alpha_beta)}
    critical.discard("")
    unresolved = expected_names - selected_names
    missing_critical = sorted(unresolved & critical)
    if missing_critical:
        errors.append(
            f"critical benchmark(s) did not resolve: {missing_critical!r}"
        )
    dropped = sorted(unresolved - critical)
    if dropped:
        logger.warning(
            "Benchmark(s) unavailable this run, dropped from the digest (not a "
            "delivery blocker): %s",
            dropped,
        )
    extra = sorted(selected_names - expected_names)
    if extra:
        errors.append(
            f"benchmark ticker catalog has unexpected entries: extra={extra!r}"
        )
    # A resolution error blocks only when it concerns a CRITICAL benchmark.
    critical_resolution_errors = [
        error for error in resolution_errors
        if any(name and name in error for name in critical)
    ]
    if critical_resolution_errors:
        errors.append(
            "critical benchmark preprocessing did not resolve: "
            + "; ".join(critical_resolution_errors)
        )
    if "_preprocess_benchmarks" in degraded:
        errors.append("benchmark preprocessing computer failed")
    if "_live_1d" in degraded:
        errors.append("intraday preprocessing source validation failed")
    for name, ticker in selected.items():
        if not ticker:
            errors.append(f"benchmark {name} has an empty provider ticker")

    benchmark_rows = hp
    if hp is not None and not getattr(hp, "empty", True) and "type" in hp.columns:
        benchmark_rows = hp[
            hp["type"].astype(str).str.contains("enchmark", case=False, na=False)
        ]
    _check_projection(
        errors,
        "holding_performance",
        _identity_map(benchmark_rows, "name", "ticker"),
        selected,
    )
    _check_projection(
        errors,
        "benchmark_comparison",
        _identity_map(comparison, "benchmark", "ticker"),
        selected,
    )

    risk_rows: dict[str, set[str]] = {}
    for row in historical_risk.get("instruments", ()) or ():
        name = _text(row.get("label"))
        if name:
            risk_rows.setdefault(name, set()).add(_text(row.get("ticker")))
    _check_projection(errors, "historical_risk", risk_rows, selected)

    for name, ticker in selected.items():
        series = histories.get(name)
        if series is None:
            errors.append(f"benchmark_histories is missing benchmark {name}")
            continue
        series_name = _text(getattr(series, "name", ""))
        resolved = _text(getattr(series, "attrs", {}).get("resolved_ticker"))
        requested = _text(getattr(series, "attrs", {}).get("requested_ticker"))
        if series_name != ticker:
            errors.append(
                f"benchmark_histories[{name!r}].name is {series_name!r}; expected {ticker!r}"
            )
        if resolved != ticker:
            errors.append(
                f"benchmark_histories[{name!r}] resolved ticker is {resolved!r}; "
                f"expected {ticker!r}"
            )
        if requested != definitions.get(name):
            errors.append(
                f"benchmark_histories[{name!r}] provenance is {requested!r}; "
                f"expected input {definitions.get(name)!r}"
            )

    # History/current must use the canonical identity. Intraday is a separate
    # data capability: a guarded same-root EUR sibling is allowed only when the
    # preprocessing provenance explicitly records that fallback.
    from tarzan.data.market_quotes import _sibling_symbols

    for record in getattr(metrics, "ticker_resolutions", ()) or ():
        canonical = _text(record.get("canonical_ticker"))
        status = _text(record.get("status"))
        if status == "CONFLICT":
            errors.append(
                f"holding ticker resolution conflict for "
                f"{_text(record.get('name')) or canonical}"
            )
            continue
        if not canonical:
            continue
        for field in ("history_ticker", "current_ticker"):
            value = _text(record.get(field))
            if not value:
                continue
            identities = {
                part.strip() for part in value.split(" / ") if part.strip()
            }
            if identities != {canonical}:
                errors.append(
                    f"holding {canonical} has alternate "
                    f"{field}={sorted(identities)!r}"
                )

        intraday_value = _text(record.get("intraday_effective_ticker"))
        intraday_reason = _text(record.get("intraday_reason"))
        intraday_identities = {
            part.strip()
            for part in intraday_value.split(" / ")
            if part.strip()
        }
        allowed_intraday = {canonical, *_sibling_symbols(canonical)}
        invalid_intraday = intraday_identities - allowed_intraday
        if invalid_intraday:
            errors.append(
                f"holding {canonical} has invalid intraday source(s) "
                f"{sorted(invalid_intraday)!r}"
            )
        if (
            any(ticker != canonical for ticker in intraday_identities)
            and "sibling fallback" not in intraday_reason.casefold()
        ):
            errors.append(
                f"holding {canonical} has an undocumented intraday fallback"
            )

    audit = semantic_audit or {}
    intraday = audit.get("performance_intraday", {})
    if not isinstance(intraday, Mapping):
        intraday = {}
    origin = _text(intraday.get("origin"))
    requested = tuple(
        _text(ticker)
        for ticker in intraday.get("requested_tickers", ())
        if _text(ticker)
    )
    returned = tuple(
        _text(ticker)
        for ticker in intraday.get("returned_tickers", ())
        if _text(ticker)
    )
    rendered_sources_raw = intraday.get("source_tickers", {})
    rendered_sources = (
        {
            _text(canonical): _text(source)
            for canonical, source in rendered_sources_raw.items()
            if _text(canonical)
        }
        if isinstance(rendered_sources_raw, Mapping)
        else {}
    )
    preprocessed_requested = tuple(
        _text(ticker)
        for ticker in (
            getattr(metrics, "intraday_requested_tickers", ()) or ()
        )
        if _text(ticker)
    )
    preprocessed_quotes = dict(
        getattr(metrics, "intraday_quotes", {}) or {}
    )

    # The intraday request is built from ``holding_performance``, which carries
    # only holdings with a usable price history (>= 2 closes) and drops those
    # whose order mechanics are unavailable. ``holdings_df`` keeps every
    # valuation-accepted holding, history or not — so unioning the two demands
    # intraday for tickers the request set structurally cannot contain, and a
    # freshly-bought or thinly-listed instrument blocks delivery on its first
    # run. The analytical ticker set for THIS check is the performance frame.
    expected_intraday: set[str] = set()
    hp_frame = hp
    if hp_frame is None or getattr(hp_frame, "empty", True) or "ticker" not in getattr(hp_frame, "columns", ()):
        # No performance frame at all: fall back to the holdings snapshot so a
        # renderer that skipped preprocessing entirely is still caught.
        hp_frame = getattr(metrics, "holdings_df", None)
    if hp_frame is not None and not getattr(hp_frame, "empty", True) and "ticker" in getattr(hp_frame, "columns", ()):
        expected_intraday.update(
            _text(ticker) for ticker in hp_frame["ticker"].dropna() if _text(ticker)
        )
    # Plus the TARGET's own sleeves, seeds included, because ``_live_1d`` requests
    # them: the 1D panel blends the target's session from their bars and refuses
    # partial coverage. On the reference book they are all in the frame anyway (every
    # target sleeve is also a tracked benchmark, and the frame carries those), so
    # this widens the expectation for a shape that is possible rather than one that
    # is present — but the request and the expectation must move together or every
    # run fails on "candidates differ", which blocks delivery.
    expected_intraday.update(
        _text(ticker)
        for ticker in (getattr(metrics, "target_weights", {}) or {})
        if _text(ticker)
    )

    if origin != "metrics_preprocessing":
        errors.append("performance intraday data did not originate in preprocessing")
    if set(preprocessed_requested) != expected_intraday:
        errors.append(
            "intraday preprocessing candidates differ from the analytical "
            f"ticker set (requested={sorted(set(preprocessed_requested))!r}, "
            f"expected={sorted(expected_intraday)!r})"
        )
    if set(requested) != set(preprocessed_requested):
        errors.append("renderer did not consume the preprocessing request set")
    if len(requested) != len(set(requested)):
        errors.append("intraday preprocessing request contains duplicate symbols")
    if set(returned) != set(preprocessed_quotes):
        errors.append("renderer intraday rows differ from the preprocessed catalog")
    if not set(returned).issubset(set(requested)):
        errors.append(
            "preprocessing returned an unrequested canonical ticker "
            f"{sorted(set(returned) - set(requested))!r}"
        )

    expected_sources: dict[str, str] = {}
    for canonical, quote in preprocessed_quotes.items():
        canonical_text = _text(canonical)
        if not isinstance(quote, Mapping):
            errors.append(
                f"preprocessed intraday quote {canonical_text} is not structured"
            )
            continue
        source = _text(
            quote.get("intraday_source_ticker")
            or quote.get("source_ticker")
            or canonical_text
        )
        expected_sources[canonical_text] = source
        if source not in {canonical_text, *_sibling_symbols(canonical_text)}:
            errors.append(
                f"preprocessed intraday quote {canonical_text} uses invalid "
                f"source {source!r}"
            )
        series = quote.get("intraday_series")
        if series is None or len(series) < 2:
            errors.append(
                f"preprocessed intraday quote {canonical_text} has no usable series"
            )
        try:
            baseline = float(quote.get("intraday_baseline"))
        except (TypeError, ValueError):
            baseline = math.nan
        if not math.isfinite(baseline) or baseline == 0:
            errors.append(
                f"preprocessed intraday quote {canonical_text} has invalid baseline"
            )
    if rendered_sources != expected_sources:
        errors.append("renderer intraday sources differ from preprocessing provenance")

    # Recompute EVERY window panel independently from the raw metrics, then
    # compare each with the renderer's endpoint and its visible rounded label.
    # This catches a label sourced from a generic return bucket while the line
    # ends on a different shared close -- the failure this gate was built for,
    # now applied to all six windows rather than only the 30-day one.
    if getattr(metrics, "actual_value_series", None) is not None:
        from tarzan.export._perf_series import _perf_window
        # 1D is the SESSION, so its window comes from the intraday bars rather than
        # from daily closes (see ``_perf_intraday_window`` for why a 1D daily window
        # is degenerate). It returns the same shape, so the loop below verifies it
        # through the identical code path as every other bucket -- the endpoint, the
        # legend value, the visible rounded label and its presence in the HTML.
        from tarzan.export.newsletter._sections_perf import _perf_intraday_window

        geo_name = reference.benchmark_geo
        #: The lines a window panel is DESIGNED to draw. Kept here as well as in
        #: the renderer on purpose: the check that the drawn set EQUALS the
        #: resolvable set is what stops a line from quietly disappearing, and a
        #: gate reading its expectation out of the audit it is checking would
        #: verify nothing. Total and Unrealized P&L are absent by design -- they
        #: live in the matrix, which the money-figure checks above cover.
        panel_keys = ("twr", "target", "acwi")
        window_audit = audit.get("performance_windows")
        if not isinstance(window_audit, Mapping):
            # Fall back to the single-window entry so an older audit shape is
            # reported as a missing audit rather than silently passing.
            window_audit = {"1m": audit.get("performance_30d", {})}
        for bucket in ("1d", "5d", "1m", "3m", "ytd", "1y"):
            expected_window = (
                _perf_intraday_window(metrics, geo_name) if bucket == "1d"
                else _perf_window(metrics, 30, geo_name, bucket=bucket))
            panel_audit = window_audit.get(bucket)
            if expected_window is None:
                # No session (1D) or the book does not reach back that far. Either
                # way the renderer must not have drawn a panel for it -- for 1D it
                # states the figures instead, which carry no audited line.
                if panel_audit:
                    errors.append(
                        f"{bucket} panel was rendered for an unavailable window")
                continue
            expected_endpoints = expected_window.get("endpoints", {})
            resolvable = {k for k in panel_keys
                          if expected_endpoints.get(k) is not None}
            if not resolvable:
                continue
            if bucket == "1m" and not isinstance(panel_audit, Mapping):
                errors.append("30-day performance render audit is missing")
                continue
            if not isinstance(panel_audit, Mapping):
                errors.append(f"{bucket} performance render audit is missing")
                continue
            drawn = set(panel_audit.get("drawn") or ())
            if drawn != resolvable:
                errors.append(
                    f"{bucket} panel drew {sorted(drawn)} where "
                    f"{sorted(resolvable)} resolved")
            rendered_endpoints = panel_audit.get("endpoints", {})
            legend_values = panel_audit.get("legend_values", {})
            legend_labels = panel_audit.get("legend_labels", {})
            # One rule for all six windows: the label IS the drawn endpoint. 1D was the
            # exception until its lines were made to END on the tape's figure instead of
            # being labelled with it -- an exception that printed a number at a dot that
            # was not the dot's value, and left the labels in a different order than the
            # dots they belonged to.
            expected_labels: dict = {}
            for key in sorted(resolvable):
                expected_value = expected_endpoints.get(key)
                expected_label = expected_labels.get(key, expected_value)
                rendered_value = rendered_endpoints.get(key)
                label_value = legend_values.get(key)
                label = _text(legend_labels.get(key))
                if rendered_value is None or not math.isclose(
                    float(rendered_value), float(expected_value), abs_tol=1e-9
                ):
                    errors.append(
                        f"{bucket} {key} line endpoint differs from the "
                        f"shared-close endpoint"
                    )
                if label_value is None or not math.isclose(
                    float(label_value), float(expected_label), abs_tol=1e-9
                ):
                    errors.append(f"{bucket} {key} legend uses a different figure")
                displayed = _displayed_percent(label)
                if displayed is None or not math.isclose(
                    displayed, float(expected_label), abs_tol=0.0051
                ):
                    errors.append(
                        f"{bucket} {key} visible label {label!r} disagrees with "
                        f"{float(expected_label):+.6f}%"
                    )
                if label and label not in newsletter_html:
                    errors.append(
                        f"{bucket} {key} audited label is absent from rendered HTML")

    return tuple(errors)
