"""Asset-class and equity-geography targets DERIVED from the per-instrument plan.

Tarzan used to hold three target files that had to agree by hand: the class weights and
the geography weights in ``targets.csv``, and the per-instrument weights in
``targets_per_holding.csv``. Nothing checked that they described the same portfolio, and
on the reference book they had stopped:

    class target        plan implies
    Equities     75%    74.5%   ok
    Gold         10%    10.0%   ok
    Commodities   5%     5.0%   ok
    Fixed Income 28%    21.0%   7pp apart
    Alternative  11%    15.0%   4pp apart

The consequence was not cosmetic. Every drift figure, every semaphore and the whole
Allocation section measured the book against numbers the plan does not produce: the
fixed-income gap read -13.2pp where the plan's own instruments imply -6.2pp, and the
USA gap read -14.3pp against a real -21.8pp.

So when ``targets_per_holding.csv`` carries any weight, IT is the plan, and the class
and geography targets are computed from it. The two lists cannot disagree afterwards
because there is only one list.

The derivation reuses the SAME functions that produce the ACTUAL figures --
``config.class_breakdown_for`` for notional class exposure and the geo resolver's
taxonomy lookup for the equity split -- so target and actual are one computation applied
to two sets of weights. That identity is the point: any future change to how exposure is
modelled moves both sides together.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

#: Weights closer than this are the same number written twice.
_WEIGHT_EPS = 0.01


def plan_weights(target_rows: dict) -> tuple[dict[str, float], list[str]]:
    """``({ticker: weight %}, notes)`` for the instruments the plan names.

    Deduplicated by INSTRUMENT, not by the file's ISIN key. Two share classes or
    listings of one fund are two rows with two ISINs and one economic identity: the
    reference plan lists CL2 and MFEH twice each, so the raw file sums to 115.5% where
    the plan is 100%, and deriving from it would have counted those two sleeves double
    (a 16% leveraged US sleeve instead of 8%).

    A repeated instrument carrying DIFFERENT weights is a contradiction in the input,
    not something to average: the larger is kept and the conflict is reported.
    """
    from tarzan.models.instrument_key import normalize_ticker

    out: dict[str, float] = {}
    notes: list[str] = []
    for row in (target_rows or {}).values():
        weight = float((row or {}).get("target_portfolio") or 0.0)
        if weight <= 0:
            continue
        ticker = str((row or {}).get("ticker") or "").strip()
        isin = str((row or {}).get("isin") or "").strip()
        key = (normalize_ticker(ticker) if ticker else "") or isin.upper()
        if not key:
            notes.append("a target row names neither a ticker nor an ISIN; skipped")
            continue
        seen = out.get(key)
        if seen is None:
            out[key] = weight
        elif abs(seen - weight) > _WEIGHT_EPS:
            notes.append(
                f"{key} is listed more than once with different weights "
                f"({seen:g}% and {weight:g}%); kept {max(seen, weight):g}%")
            out[key] = max(seen, weight)
    return out, notes


def _primary_class(ticker: str) -> Optional[str]:
    """The instrument's curated asset class, needed as the breakdown's fallback."""
    from tarzan import config as cfg

    row = (cfg.instrument_taxonomy() or {}).get(str(ticker).split(".")[0].upper())
    if isinstance(row, dict):
        return row.get("asset_class") or None
    if isinstance(row, (tuple, list)) and row:
        return str(row[0]) or None
    return None


def derive_class_targets(plan: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Notional class targets implied by the plan, and what could not be resolved.

    NOT normalised to 100. These are notional exposures and leverage is exactly why
    they exceed it: the reference plan implies 125.5% of capital, which is the same
    statement as "the book runs at 1.25x". Normalising would erase the one fact the
    class table exists to show.
    """
    from tarzan import config as cfg

    out: dict[str, float] = {}
    notes: list[str] = []
    for ticker, weight in plan.items():
        breakdown = cfg.class_breakdown_for(None, ticker, _primary_class(ticker)) or {}
        if not breakdown:
            # Contributing nothing is the honest outcome: defaulting it into some class
            # would move a target the reader cannot trace to any instrument.
            notes.append(
                f"{ticker} ({weight:g}% of the plan) has no asset class and no exp_* "
                f"row in the taxonomy, so it contributes to no class target")
            continue
        for klass, exposure in breakdown.items():
            name = getattr(klass, "value", str(klass))
            out[name] = out.get(name, 0.0) + weight * float(exposure) / 100.0
    return out, notes


def derive_geo_targets(plan: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Equity-geography targets implied by the plan, normalised to 100.

    Each sleeve contributes its EQUITY NOTIONAL — not its weight — split by the
    taxonomy's own geography row, which is how the actual geography table is built. A
    90/60 efficient core therefore contributes 90% of its weight and a 2x US tracker
    200% of its own.

    Normalised, unlike the class targets: geography is a distribution WITHIN the equity
    sleeve, and ``InvestorConfig`` validates that it sums to 100.

    Sleeves with equity notional but no geography row are excluded and reported. The
    remainder is then normalised over what IS known, which changes the answer — hence
    the note rather than a silent renormalisation.
    """
    from tarzan import config as cfg
    from tarzan.data.geo_resolver import _lookup_asset_geo

    raw: dict[str, float] = {}
    notes: list[str] = []
    equity_total = 0.0
    for ticker, weight in plan.items():
        breakdown = cfg.class_breakdown_for(None, ticker, _primary_class(ticker)) or {}
        equity = next((float(v) for k, v in breakdown.items()
                       if getattr(k, "value", str(k)) == "Equities"), 0.0)
        if equity <= 0:
            continue
        notional = weight * equity / 100.0
        found = _lookup_asset_geo("", ticker)
        if not found:
            notes.append(
                f"{ticker} carries {notional:.1f}pp of equity notional with no "
                f"geography row in the taxonomy; excluded from the geography targets")
            continue
        equity_total += notional
        for region, share in (found[0] or {}).items():
            name = getattr(region, "value", str(region))
            raw[name] = raw.get(name, 0.0) + notional * float(share) / 100.0
    if equity_total <= 0:
        return {}, notes
    scale = sum(raw.values())
    if scale <= 0:
        return {}, notes
    return {k: v / scale * 100.0 for k, v in raw.items()}, notes


def derive_target_leverage(plan: dict[str, float]) -> dict[str, Optional[float]]:
    """``{class: notional / physical}`` for the plan, the same ratio production shows
    for the actual book.

    ``None`` means the plan holds no physical capital in that class, so its exposure is
    entirely synthetic — true of fixed income here, whose whole target comes from an
    efficient core's bond overlay while that fund's capital counts as equity. Printing
    a number there would invent one.

    This only became a meaningful figure once the class targets were derived. While they
    came from a separate file the ratio mixed two sources: Alternative read 11/15 =
    0.73x, which is not a leverage but the distance between two lists that disagreed.
    Derived from one plan, the same class reads 15/15 = 1.00x and matches its actual.
    """
    from tarzan import config as cfg

    notional, _ = derive_class_targets(plan)
    physical: dict[str, float] = {}
    for ticker, weight in plan.items():
        primary = _primary_class(ticker)
        if not primary:
            # No primary class: its capital cannot be attributed, so it is left out of
            # every denominator rather than inflating one arbitrarily.
            continue
        physical[primary] = physical.get(primary, 0.0) + weight
    out: dict[str, Optional[float]] = {}
    for klass, exposure in notional.items():
        base = physical.get(klass, 0.0)
        out[klass] = (exposure / base) if base > 0 else None
    return out


def apply_to_config(config, target_rows: dict) -> Optional[dict]:
    """Replace the config's class and geography targets with the plan's own.

    Returns a report of what was derived, or ``None`` when the per-holding file carries
    no weights — in which case the declared targets stand and nothing changes.

    The trigger is the FILE, not ``target_use_per_holding_only``. That flag already
    tells the optimizer to rebalance toward the per-instrument plan, so with it on and
    this off the issue reported drift against class targets the optimizer was ignoring:
    two parts of one run disagreeing about what the plan is.
    """
    plan, notes = plan_weights(target_rows)
    if not plan:
        return None

    classes, class_notes = derive_class_targets(plan)
    geo, geo_notes = derive_geo_targets(plan)
    notes += class_notes + geo_notes
    if not classes:
        # Nothing resolvable: keep what the config declared rather than blank the
        # targets, and say so. A silent {} would make every class read "no target".
        notes.append("no class target could be derived; declared targets kept")
        return {"applied": False, "plan": plan, "notes": notes}

    before_classes = dict(getattr(config, "invested_allocation_targets_pctg", {}) or {})
    before_geo = dict(getattr(config, "equity_geo_targets_pctg", {}) or {})

    # Every class the config knew about stays a key, at 0 when the plan does not reach
    # it: dropping the key would hide the class from the table, and a class the plan
    # abandoned still needs to show its holdings drifting to zero.
    merged = {name: 0.0 for name in before_classes}
    merged.update(classes)
    config.invested_allocation_targets_pctg = merged
    if geo:
        merged_geo = {name: 0.0 for name in before_geo}
        merged_geo.update(geo)
        config.equity_geo_targets_pctg = merged_geo

    report = {
        "applied": True,
        "plan": plan,
        "target_leverage": derive_target_leverage(plan),
        "classes": merged,
        "geo": dict(config.equity_geo_targets_pctg),
        "declared_classes": before_classes,
        "declared_geo": before_geo,
        "notional_pct": sum(classes.values()),
        "notes": notes,
    }
    logger.info(
        "Targets derived from the %d-instrument plan: notional %.1f%% of capital "
        "(%s). Declared class/geo targets in targets.csv are not used.",
        len(plan), report["notional_pct"],
        ", ".join(f"{k} {v:.1f}%" for k, v in sorted(
            classes.items(), key=lambda kv: -kv[1])),
    )
    if geo:
        logger.info("Equity geography derived: %s", ", ".join(
            f"{k} {v:.1f}%" for k, v in sorted(geo.items(), key=lambda kv: -kv[1])))
    for note in notes:
        logger.warning("Target derivation: %s", note)
    return report
