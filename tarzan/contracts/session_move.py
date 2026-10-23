"""Paired euro evidence for the exact return window, never a cash-weight guess."""

from dataclasses import dataclass
from datetime import date
import math
from numbers import Real

import numpy as np
import pandas as pd

from tarzan.runtime.ledger import Availability


@dataclass(frozen=True)
class DailyReturnStep:
    """Facts captured inside the original NAV chain without changing its math."""

    end_date: date
    start_date: date
    starting_value_eur: float
    ending_value_eur: float
    external_flow_eur: float
    factor: float | None
    profit_eur: float | None
    state: str


@dataclass(frozen=True)
class SessionMoveEvidence:
    """One exact source window; absent euro evidence is not numeric zero."""

    availability: str = Availability.UNAVAILABLE.value
    source_basis: str = "UNPREPARED"
    anchor_date: date | None = None
    end_date: date | None = None
    return_pct: float | None = None
    profit_eur: float | None = None
    external_flow_eur: float | None = None
    steps: tuple[DailyReturnStep, ...] = ()
    reason: str = "NO_PAIRED_SOURCE"
    # A business-day resample labels a weekend bin Friday but uses its last
    # calendar observation. Preserve that native authority, without changing
    # the published window or its existing return.
    source_anchor_date: date | None = None
    source_end_date: date | None = None


def paired_session_move(
    series, return_pct, *, source_basis=None, steps=(), daily_nav=None,
    history_availability=Availability.AVAILABLE.value,
) -> SessionMoveEvidence:
    """Use the same endpoints and original valid NAV factors as ``1d``.

    A fixed holdings basket uses its own endpoint delta, including priced
    CashEq securities. An order NAV sums the original chain's euro profits,
    not today's value times a percentage. Suspended/unpriced/re-entry steps
    and sparse fallback valuations cannot prove that conversion.
    """
    from tarzan.engine.stats import normalize_session_index, window_anchor

    empty = SessionMoveEvidence(
        source_basis=source_basis or "UNPREPARED",
        reason="NO_PAIRED_SOURCE",
    )
    if (
        history_availability not in (Availability.AVAILABLE.value, Availability.DEGRADED.value)
        or source_basis not in ("HOLDINGS_PRICE_AGGREGATE", "ORDER_FLOW_ADJUSTED_NAV")
        or not isinstance(series, pd.Series)
        or not isinstance(series.index, pd.DatetimeIndex)
        or not series.index.is_monotonic_increasing or not series.index.is_unique
        or not isinstance(return_pct, Real) or isinstance(return_pct, (bool, np.bool_))
    ):
        return empty
    try:
        pct = float(return_pct)
        observed = series.dropna()
        if not math.isfinite(pct) or len(observed) < 2:
            return empty
        anchor = window_anchor(observed, "1d")
        if anchor is None:
            return empty
        end = observed.index[-1]
        start_value = float(observed.loc[:anchor].iloc[-1])
        end_value = float(observed.iloc[-1])
        if not (
            math.isfinite(start_value) and start_value > 0
            and math.isfinite(end_value) and end_value >= 0
        ):
            return empty
        expected_pct = (end_value / start_value - 1.0) * 100.0
        if not math.isclose(expected_pct, pct, rel_tol=1e-12, abs_tol=1e-10):
            return SessionMoveEvidence(source_basis=source_basis, reason="RETURN_SOURCE_MISMATCH")
        start_date, end_date = anchor.date(), end.date()
        source_start, source_end = start_date, end_date
        if source_basis == "HOLDINGS_PRICE_AGGREGATE":
            profit = end_value - start_value
            selected, flow = (), 0.0
        else:
            if daily_nav is not None:
                if (
                    not isinstance(daily_nav, pd.Series)
                    or not isinstance(daily_nav.index, pd.DatetimeIndex)
                    or not daily_nav.index.is_monotonic_increasing
                    or not daily_nav.index.is_unique
                ):
                    return SessionMoveEvidence(source_basis=source_basis, reason="UNKNOWN_NAV_ENDPOINTS")
                calendar = daily_nav.copy(deep=False)
                calendar.index = normalize_session_index(calendar.index)
                if not calendar.index.is_unique:
                    return SessionMoveEvidence(source_basis=source_basis, reason="UNKNOWN_NAV_ENDPOINTS")
                native_dates = pd.Series(
                    list(calendar.index.date), index=calendar.index, dtype=object,
                ).where(calendar.notna())
                source_dates = native_dates.resample("B").last().dropna()
                source_start = source_dates.loc[pd.Timestamp(start_date)]
                source_end = source_dates.loc[pd.Timestamp(end_date)]
                # The dates alone cannot certify that another NAV was used.
                if (
                    float(calendar.loc[pd.Timestamp(source_start)]) != start_value
                    or float(calendar.loc[pd.Timestamp(source_end)]) != end_value
                ):
                    return SessionMoveEvidence(source_basis=source_basis, reason="NAV_ENDPOINTS_MISMATCH")
            selected = tuple(step for step in steps if source_start < step.end_date <= source_end)
            cursor = source_start
            factor = 1.0
            previous_value = None
            for step in selected:
                if (
                    not isinstance(step, DailyReturnStep)
                    or step.state != "MEASURED" or step.start_date != cursor
                    or step.end_date <= cursor
                    or step.factor is None or step.factor <= 0
                    or step.profit_eur is None
                    or not all(
                        isinstance(value, Real)
                        and not isinstance(value, (bool, np.bool_))
                        and math.isfinite(value)
                        for value in (
                        step.starting_value_eur, step.ending_value_eur,
                        step.external_flow_eur, step.factor, step.profit_eur,
                        )
                    )
                ):
                    return SessionMoveEvidence(source_basis=source_basis, reason="UNMEASURED_NAV_STEP")
                before_flow = step.ending_value_eur - step.external_flow_eur
                if (
                    step.starting_value_eur <= 0 or step.ending_value_eur < 0
                    or before_flow <= 0
                    or not math.isclose(
                        step.factor, before_flow / step.starting_value_eur,
                        rel_tol=1e-12, abs_tol=1e-12,
                    )
                    or not math.isclose(
                        step.profit_eur, before_flow - step.starting_value_eur,
                        rel_tol=1e-12, abs_tol=1e-9,
                    )
                    or (
                        previous_value is not None
                        and not math.isclose(
                            step.starting_value_eur, previous_value,
                            rel_tol=1e-12, abs_tol=1e-9,
                        )
                    )
                ):
                    return SessionMoveEvidence(source_basis=source_basis, reason="NAV_FACTS_MISMATCH")
                cursor = step.end_date
                previous_value = step.ending_value_eur
                factor *= step.factor
            if not selected or cursor != source_end:
                return SessionMoveEvidence(source_basis=source_basis, reason="INCOMPLETE_NAV_WINDOW")
            if not math.isclose((factor - 1.0) * 100.0, pct, rel_tol=1e-12, abs_tol=1e-10):
                return SessionMoveEvidence(source_basis=source_basis, reason="NAV_FACTOR_MISMATCH")
            profit = sum(step.profit_eur for step in selected)
            flow = sum(step.external_flow_eur for step in selected)
        if not math.isfinite(profit) or not math.isfinite(flow):
            return empty
        return SessionMoveEvidence(
            availability=history_availability, source_basis=source_basis,
            anchor_date=start_date, end_date=end_date, return_pct=pct,
            profit_eur=profit, external_flow_eur=flow, steps=selected, reason="",
            source_anchor_date=source_start, source_end_date=source_end,
        )
    except (TypeError, ValueError, AttributeError, KeyError, OverflowError):
        return empty
