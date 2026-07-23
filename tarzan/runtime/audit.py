"""Run-owned rebalancing audit compatibility projection.

The active :class:`RunSession` and append-only :class:`RunLedger` are the
production authorities for every deterministic plan input, final action, and
verification. A context-local list preserves the established read/render API
for versioned consumers and tests; initialized CLI/email runs publish it only
through ``LocalArtifactWriter``. Recording remains best-effort so evidence
collection cannot alter optimizer behavior.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class _Audit:
    records: list[dict] = field(default_factory=list)


# Context-local compatibility projection; active runs mirror records to their
# session-owned ledger.
_audit: ContextVar[Optional[_Audit]] = ContextVar("tarzan_rebalancing_audit", default=None)


def _current_audit() -> _Audit:
    audit = _audit.get()
    if audit is None:
        audit = _Audit()
        _audit.set(audit)
    return audit


def reset() -> None:
    """Start a fresh audit trail in the current run context."""
    _audit.set(_Audit())


def record_rebalancing_plan(
    label: str,
    *,
    no_sell: bool,
    total_value: float,
    lump_sum: Optional[float],
    config: Any,
    holdings: list,
    suggestions: Optional[list],
    verifications: Optional[list],
) -> None:
    """Record one rebalancing plan (inputs + outputs). Best-effort.

    ``holdings`` are the actual holdings the solver saw (real + seeds); we
    capture the value and targets that drove the plan, plus the solver's
    friction knobs, so the plan is reconstructable from the record alone.
    """
    try:
        cfg_snapshot = {
            "target_tolerance_pctg": getattr(config, "rebalancing_target_tolerance_pctg", None),
            "no_sell": no_sell,
            "lump_sum_eur": float(lump_sum) if lump_sum else 0.0,
            "cash_buffer_eur": getattr(config, "target_cash_buffer_eur", None),
            "cgt_standard_pctg": getattr(config, "rebalancing_capital_gains_tax_standard_pctg", None),
            "cgt_government_pctg": getattr(config, "rebalancing_capital_gains_tax_government_pctg", None),
            "fee_buy_eur": getattr(config, "rebalancing_transaction_fee_buy_eur", None),
            "fee_sell_eur": getattr(config, "rebalancing_transaction_fee_sell_eur", None),
            "use_per_holding_only": getattr(config, "target_use_per_holding_only", None),
        }
        holdings_snapshot = []
        for h in holdings or []:
            val = getattr(h, "current_value", None)
            if val is None:
                val = getattr(h, "market_value_eur", 0.0)
            holdings_snapshot.append({
                "isin": getattr(h, "isin", None),
                "ticker": getattr(h, "ticker", None),
                "asset_class": (h.asset_class.value
                                if getattr(h, "asset_class", None) is not None else None),
                "value_eur": round(float(val or 0.0), 2),
                "target_equities": getattr(h, "target_equities", None),
                "target_fixed_income": getattr(h, "target_fixed_income", None),
                "target_portfolio": getattr(h, "target_portfolio", None),
                "no_buy_no_sell": getattr(h, "no_buy_no_sell", None),
                "is_seeded_target": getattr(h, "is_seeded_target", False),
            })
        record = {
            "plan": label,
            "total_value_eur": round(float(total_value or 0.0), 2),
            "config": cfg_snapshot,
            "holdings": holdings_snapshot,
            "actions": suggestions or [],
            "verifications": verifications or [],
        }
        _current_audit().records.append(record)
        from tarzan.runtime.ledger import LedgerEntryType
        from tarzan.runtime.session import current_session
        session = current_session()
        if session is not None:
            session.audit.append(record)
            session.ledger.append(LedgerEntryType.PLAN, record)
    except Exception as e:  # noqa: BLE001 — audit must never break the pipeline
        logger.debug("Rebalancing audit record failed: %s", e)


def records() -> list[dict]:
    return list(_current_audit().records)
