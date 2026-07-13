"""Append-only rebalancing audit trail.

A durable, machine-readable record of every rebalancing plan the engine
produced in a run — the inputs it saw (per-holding value + target, config
knobs, lump sum) and the outputs it emitted (the buy/sell actions and the
post-trade verification per ambit). Today Tarzan keeps no trace of *why* a
given buy/sell was suggested once the report is closed; this gives that
traceability (a MiFID-style "reconstruct the decision" record).

Design mirrors ``tarzan.data_quality``:
  * process-global collector, reset at the top of ``orchestrator.run``;
  * **best-effort** — recording or writing must never raise into the
    pipeline (an audit trail that breaks the run is worse than none);
  * written by the CLI after the run as JSON Lines (one record per plan),
    so it is greppable and appendable without parsing the whole file.

The rebalancer is deterministic given identical inputs (fixed
``LSParams.seed``), so a record fully determines the plan it describes.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class _Audit:
    records: list[dict] = field(default_factory=list)


# Process-global, reset per run.
_audit = _Audit()


def reset() -> None:
    """Start a fresh audit trail for a new run."""
    global _audit
    _audit = _Audit()


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
        _audit.records.append({
            "plan": label,
            "total_value_eur": round(float(total_value or 0.0), 2),
            "config": cfg_snapshot,
            "holdings": holdings_snapshot,
            "actions": suggestions or [],
            "verifications": verifications or [],
        })
    except Exception as e:  # noqa: BLE001 — audit must never break the pipeline
        logger.debug("Rebalancing audit record failed: %s", e)


def records() -> list[dict]:
    return list(_audit.records)


def write_report(output_dir: str, filename: str = "rebalancing_audit.jsonl") -> Optional[str]:
    """Write the audit trail as JSON Lines to ``output_dir/filename``.

    One JSON object per line (per plan). Best-effort: returns None on any
    I/O error rather than breaking the run. Not written when empty (a run
    with no rebalancing produced no plans to audit).
    """
    if not _audit.records:
        return None
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for rec in _audit.records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str))
                f.write("\n")
        return path
    except Exception as e:  # noqa: BLE001
        logger.debug("Rebalancing audit write failed: %s", e)
        return None
