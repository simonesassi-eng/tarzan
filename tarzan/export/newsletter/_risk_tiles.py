"""The portfolio's risk metrics as STATE tiles, and the legend that defines them.

These used to be section [11] RISK: ten tiles, each with a weak/fair/strong gauge drawn
beside the figure. The section is gone and the metrics moved up into [01] STATE, beside
the value and the return measures — the ten figures answer "what shape was the ride",
which is a property of the state and not a separate topic.

The gauge went with the section. What it carried survives without a drawing: the
configured bands already NAME themselves per metric ("Contained/Moderate/Severe" for a
drawdown, "Defensive/Neutral/Aggressive" for beta), so the band becomes the tile's
caption and its colour becomes the figure's colour. A bar that has to be measured
against its own track is a slower way to say the same thing, in 92px per tile.

Both the tiles and the legend read ``metric_ratings`` from constants.yaml, with its
citations, so the bands quoted are the ones the ratings are computed from and there is
no second copy to drift.

Beta is coloured NEUTRAL on purpose. Its bands rate market exposure, which is a property
to know rather than a score to win: painting "defensive" green would state an opinion the
project has deliberately declined to make. It keeps the band word, without the verdict.
"""

from __future__ import annotations

from typing import Optional

from tarzan.export.newsletter._format import _pct, is_missing

#: Tile labels for the two Greek metrics, spelled out.
#:
#: The STATE grid uppercases its labels in CSS, which folds α onto Α -- drawn like a
#: Latin A in most fonts. ``greek_safe`` exists for exactly that and wraps the letter in
#: a ``text-transform:none`` span, but a STATE tile's label is ESCAPED on the way out
#: (asset-class names and "Total P&L" carry ampersands), so the span arrived as visible
#: text: ``<span style="text-transform:none;">α</span>*``. Spelling the names out needs
#: no markup and no exception. The legend keeps the letters -- its column is not
#: uppercased, so nothing folds there.
_TILE_LABELS: dict[str, str] = {"alpha": "Alpha", "beta": "Beta"}

#: ``(label, metrics key, is_pct, footnote, ratings key)``, in display order.
#:
#: Full names, not abbreviations: "Vol", "VaR" and "CVaR" were squeezed for an
#: eleven-column table that no longer exists, and the confidence level on the two tail
#: measures is part of their definition rather than a footnote.
#:
#: The asterisk on alpha and beta points at the note naming the index they are measured
#: against, which is their definition and not a comparison.
#:
#: CAGR is deliberately NOT here. The risk block computes one, but STATE already shows a
#: CAGR from a different measure -- the book's own annualized TWROR -- and on the
#: reference book the two read +15.68% and 24.27%. Two tiles labelled "CAGR" with
#: different numbers, six tiles apart, is worse than one of them being absent; and the
#: risk figure is the more misleading of the pair, being a backtest at TODAY's weights
#: rather than anything the book actually earned. Sharpe and Sortino still carry it
#: implicitly as their numerator, which is where a backtest CAGR belongs.
RISK_METRICS: tuple[tuple[str, str, bool, str, Optional[str]], ...] = (
    ("Volatility", "volatility", True, "", "volatility"),
    ("Sharpe", "sharpe", False, "", "sharpe"),
    ("Sortino", "sortino", False, "", "sortino"),
    ("Max DD", "max_drawdown", True, "", "max_drawdown"),
    ("Ulcer", "ulcer_index", True, "", "ulcer_index"),
    ("VaR 95%", "var_95", True, "", "var_pct"),
    ("CVaR 95%", "cvar_95", True, "", "cvar_pct"),
    ("α", "alpha", True, "*", "alpha"),
    ("β", "beta", False, "*", "beta"),
)

#: Beta is rated but not judged — see the module docstring.
_UNJUDGED = frozenset({"beta"})

#: What each metric IS, in one clause. Calibration is the band word and the colour on
#: the tile, so the prose does not repeat it ("equity indexes ~15-20%", ">1 is good").
_DESCRIPTIONS: dict[str, str] = {
    "volatility": "Yearly standard deviation of daily returns.",
    "sharpe": "Return above cash per unit of volatility.",
    "sortino": "Return above cash per unit of DOWNSIDE volatility only.",
    "max_drawdown": "Deepest peak-to-trough fall over the window.",
    "ulcer_index": "Root mean square of every drawdown, so depth and duration both "
                   "count.",
    "var_pct": "The daily loss the worst 5% of days exceed.",
    "cvar_pct": "The average loss on those worst 5% of days.",
    "alpha": "Return not explained by the benchmark's own move.",
    "beta": "How far the portfolio moves for a 1% move in the benchmark.",
}

#: A word or two on the tile itself, under the figure: what the number is measured on.
#: Short enough for a third of the STATE grid, and never a restatement of the band word
#: that sits beside it.
_UNITS: dict[str, str] = {
    "volatility": "annualized",
    "sharpe": "per unit of vol",
    "sortino": "per unit of downside",
    "max_drawdown": "peak to trough",
    "ulcer_index": "drawdown RMS",
    "var_pct": "daily, worst 5%",
    "cvar_pct": "average of that 5%",
    "alpha": "annualized",
    "beta": "market exposure",
}


def _band_index(value: float, thresholds, invert: bool) -> Optional[int]:
    """Which of the three configured bands ``value`` falls in, or None.

    ``invert: true`` means "a smaller magnitude is better", and the thresholds are
    written against the metric's ABSOLUTE value: max_drawdown is banded at [-15, -30]
    and VaR at [0.8, 1.5] while both carry the flag. So an inverted metric compares
    abs(value) against abs(threshold), which is the same reading the gauge used.
    """
    if not thresholds or len(thresholds) < 2:
        return None
    good, warn = float(thresholds[0]), float(thresholds[1])
    v = float(value)
    if invert:
        v, good, warn = abs(v), abs(good), abs(warn)
        if v <= good:
            return 0
        return 1 if v <= warn else 2
    if v >= good:
        return 0
    return 1 if v >= warn else 2


#: Band index to STATE tile tone. ``warn`` is amber, which the STATE partial did not
#: have before these tiles arrived: the return tiles are judged by SIGN, two-valued, and
#: a health band has three.
_TONES = ("pos", "warn", "neg")


def health(value, rating: Optional[dict], *, judged: bool = True) -> tuple[str, str]:
    """``(tone, band label)`` for one figure against its configured band.

    ``judged=False`` returns the band's word with a neutral tone, for a metric whose
    bands describe a property rather than score it.
    """
    band = rating or {}
    idx = _band_index(value, band.get("thresholds"), bool(band.get("invert", False)))
    if idx is None:
        return "flat", ""
    labels = band.get("labels") or ()
    word = str(labels[idx]).lower() if len(labels) > idx else ""
    return (_TONES[idx] if judged else "flat"), word


def risk_tiles(metrics, tile) -> list[dict]:
    """The ten risk figures as STATE tiles, built through the caller's own ``tile``.

    ``tile(label, value, caption, tone)`` is passed in rather than imported so the tiles
    are escaped and shaped by exactly the helper the rest of STATE uses; a second
    constructor here is how one tile ends up unescaped.

    Returns ``[]`` when the historical-risk block is unavailable, so STATE simply ends
    at the return measures rather than showing ten em-dashes.
    """
    from tarzan import config as cfg

    hr = getattr(metrics, "historical_risk", None) or {}
    if not hr.get("available"):
        return []
    port = hr.get("portfolio") or {}
    values = port.get("metrics") or {}
    if not values:
        return []
    ratings = cfg.metric_ratings() or {}

    out: list[dict] = []
    for label, key, is_pct, note, rating_key in RISK_METRICS:
        value = values.get(key)
        if is_missing(value):
            continue
        tone, word = health(value, ratings.get(rating_key or ""),
                            judged=rating_key not in _UNJUDGED)
        unit = _UNITS.get(rating_key or "", "")
        caption = " · ".join(p for p in (word, unit) if p)
        out.append(tile(
            f"{_TILE_LABELS.get(rating_key or '', label)}{note}",
            _pct(float(value)) if is_pct else f"{float(value):.2f}",
            caption, tone))
    return out


def risk_legend() -> list[dict]:
    """``[{label, strong, fair, weak, description}]`` — one row per metric.

    The bands come from ``metric_ratings`` so the numbers quoted are the ones the tile
    colours are computed from. It sits at the FOOT of the issue now: the tiles it
    defines are in STATE at the top, and a reader who needs the definition of Sortino
    needs it once, not above every reading of it.
    """
    from tarzan import config as cfg

    ratings = cfg.metric_ratings() or {}
    rows: list[dict] = []
    for label, _key, _is_pct, note, rating_key in RISK_METRICS:
        band = ratings.get(rating_key or "") or {}
        thresholds = band.get("thresholds") or []
        labels = band.get("labels") or ("", "", "")
        unit = band.get("unit") or ""
        invert = bool(band.get("invert", False))
        # An inverted band is written against the MAGNITUDE -- max_drawdown at [-15, -30]
        # means "shallower than 15%" -- so the comparator has to be applied to abs(),
        # or the row reads "≤ −15%", which admits −20% and says the opposite of the band.
        def _n(t):
            return f"{abs(float(t)) if invert else float(t):g}{unit}"

        good = _n(thresholds[0]) if len(thresholds) > 0 else ""
        warn = _n(thresholds[1]) if len(thresholds) > 1 else ""
        rows.append({
            # The letters themselves here: this column is not uppercased, so there is
            # nothing to fold and no wrapper to escape.
            "label": f"{label}{note}",
            # The comparator belongs with the number: "≥ 7%" and "≤ 15%" are different
            # claims and the bands are written in both directions.
            "strong": (f'{"≤" if invert else "≥"} {good}'
                       if good else ""),
            "fair": (f'{"≤" if invert else "≥"} {warn}' if warn else ""),
            "weak": (f'{">" if invert else "<"} {warn}' if warn else ""),
            "strong_label": str(labels[0]) if len(labels) > 0 else "",
            "fair_label": str(labels[1]) if len(labels) > 1 else "",
            "weak_label": str(labels[2]) if len(labels) > 2 else "",
            "description": _DESCRIPTIONS.get(rating_key or "", ""),
        })
    return rows


def risk_notes(ctx) -> dict:
    """The three disclosures that used to travel with section [11].

    They are not decoration: the metrics are a backtest at TODAY's weights over a
    common window, alpha and beta are measured against a specific index, and the
    backtest may have had to leave a holding out. Deleting the section without
    rehousing these would have kept the figures and dropped their terms, so they sit
    with the legend at the foot of the issue.

    Empty strings where the engine has nothing to say, so the template renders nothing
    rather than a heading over a blank.
    """
    hr = getattr(ctx.metrics, "historical_risk", None) or {}
    port = hr.get("portfolio") or {}
    if not hr.get("available"):
        return {"window": "", "alpha_beta": "", "backtest": ""}
    return {
        "window": (
            f'Measured on the portfolio over {port.get("span_label") or "its"} of '
            f'history: a backtest at today\u2019s weights held constant, over the '
            f'longest window where every holding with a year or more of history '
            f'overlaps.'),
        "alpha_beta": (
            f'\u03b1 and \u03b2 are computed against '
            f'{ctx.benchmark_alpha_beta or "the market index"}.'),
        "backtest": port.get("note") or "",
    }
