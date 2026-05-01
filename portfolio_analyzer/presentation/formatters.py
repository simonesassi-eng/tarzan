"""Formatting utilities for the Streamlit UI.

Handles number formatting (EUR, %, bp), gain/loss coloring,
and metric rating labels.
"""

from __future__ import annotations

from portfolio_analyzer import config as cfg


def fmt_eur(val: float) -> str:
    """Format as EUR currency: €12,345.67"""
    if val is None:
        return "—"
    return f"€{val:,.2f}"


def fmt_pct(val, decimals: int = 1) -> str:
    """Format as percentage with sign."""
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    return f"{val:+.{decimals}f}%"


def fmt_pct_plain(val, decimals: int = 1) -> str:
    """Format as percentage without sign."""
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    return f"{val:.{decimals}f}%"


def fmt_ratio(val, decimals: int = 2) -> str:
    """Format a ratio (Sharpe, Sortino, Beta)."""
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    return f"{val:.{decimals}f}"


def gain_color(val) -> str:
    """Return CSS color for gain/loss value."""
    if val is None or (isinstance(val, float) and val != val):
        return "#8b949e"
    if val > 0:
        return "#3fb950"
    if val < 0:
        return "#f85149"
    return "#8b949e"


def rate_metric(metric_key: str, value) -> tuple[str, str, str]:
    """Rate a metric value and return (label, color, emoji).

    Uses thresholds from config/constants.yaml metric_ratings.
    Returns ('—', '#8b949e', '') if metric not configured or value is NaN.
    """
    if value is None or (isinstance(value, float) and value != value):
        return "—", "#8b949e", ""

    ratings = cfg.metric_ratings()
    spec = ratings.get(metric_key)
    if not spec:
        return "—", "#8b949e", ""

    thresholds = spec.get("thresholds", [0, 0])
    labels = spec.get("labels", ["Good", "Fair", "Poor"])
    invert = spec.get("invert", False)

    good_t, warn_t = thresholds[0], thresholds[1]

    if invert:
        if value <= good_t:
            return labels[0], "#3fb950", "🟢"
        elif value <= warn_t:
            return labels[1], "#f0883e", "🟡"
        else:
            return labels[2], "#f85149", "🔴"
    else:
        if value >= good_t:
            return labels[0], "#3fb950", "🟢"
        elif value >= warn_t:
            return labels[1], "#f0883e", "🟡"
        else:
            return labels[2], "#f85149", "🔴"
