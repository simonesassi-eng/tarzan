"""The semantic gate must read every label the renderer can legitimately draw.

The gate blocks delivery when a chart's visible label disagrees with the
endpoint the line was drawn from. That comparison depends on parsing the label
back into a number — so a label the parser cannot read is indistinguishable
from a label that lies, and the newsletter does not send.

``_pct(..., signed=True)`` deliberately omits the sign when a value rounds to
zero: ``_sign_for`` returns "" so a residual −0.004% prints as "0.00%" rather
than a signed zero that reads as a real move down. The gate's regex required a
sign, so that correct rendering was unparseable and blocked the send with

    30-day unreal_pct visible label '0.00%' disagrees with endpoint +0.001600%

which is exactly one violation — the endpoint and legend checks compare floats
and pass. Pure string/format math, network-free.
"""

from __future__ import annotations

import math

import pytest

from tarzan.export.newsletter._format import _pct
from tarzan.export.newsletter._semantic import _displayed_percent


# Values that round to zero at 2dp print WITHOUT a sign, by design.
@pytest.mark.parametrize("value", [0.0, 0.001, -0.001, -0.0001, 0.004, -0.004])
def test_rounds_to_zero_label_is_parseable(value):
    label = _pct(value, signed=True)
    assert label == "0.00%", "the unsigned zero rendering is the thing under test"
    parsed = _displayed_percent(label)
    assert parsed is not None, (
        f"{label!r} must parse; an unparseable label is reported as disagreeing "
        "with its endpoint and blocks delivery"
    )
    # The same tolerance the gate applies.
    assert math.isclose(parsed, value, abs_tol=0.0051)


@pytest.mark.parametrize("value", [1.23, -1.23, 12.5, -0.97, 100.0])
def test_signed_labels_still_round_trip(value):
    parsed = _displayed_percent(_pct(value, signed=True))
    assert parsed is not None
    assert math.isclose(parsed, value, abs_tol=0.0051)


def test_gate_still_catches_a_real_mismatch():
    """Accepting an unsigned zero must not make the gate blind."""
    parsed = _displayed_percent("+1.23%")
    assert parsed is not None
    assert not math.isclose(parsed, 5.67, abs_tol=0.0051)


def test_sign_is_not_dropped_from_a_nonzero_label():
    """A negative that does NOT round to zero keeps its minus."""
    assert _displayed_percent(_pct(-1.5, signed=True)) == -1.5
    assert _displayed_percent("−0.97%") == -0.97


# ── The intraday "analytical ticker set" is the PERFORMANCE frame ─────────────
# The intraday request is built from ``holding_performance``, which carries only
# holdings with >= 2 closes (and drops those whose order mechanics are
# unavailable). ``holdings_df`` keeps every valuation-accepted holding. Unioning
# the two demanded intraday for a ticker the request set structurally cannot
# contain, so a freshly-bought or thinly-listed instrument blocked the send on
# its first run:
#
#   intraday preprocessing candidates differ from the analytical ticker set
#   (requested=[... no 18MF.MU ...], expected=[... 18MF.MU ...])
#
# FR0010755611 was bought 2026-07-27 and resolved to 18MF.MU, which had too
# little history to enter holding_performance. One violation, delivery blocked.

import pandas as pd

from tarzan.export.newsletter._semantic import validate_newsletter_semantics


class _M:
    """The narrowest metrics stand-in that reaches the intraday check."""
    benchmark_resolution_errors = ()
    benchmark_tickers = {}
    degraded_computers = ()
    benchmark_comparison = None
    historical_risk = {}
    ticker_resolutions = ()

    def __init__(self, hp, holdings_df, requested, quotes):
        self.holding_performance = hp
        self.holdings_df = holdings_df
        self.intraday_requested_tickers = requested
        self.intraday_quotes = quotes
        # A non-empty history set gives the gate a benchmark "contract" so it
        # does not early-return before the intraday check.
        s = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-07-01", "2026-07-02"]))
        s.name = "ACWI.X"
        s.attrs["resolved_ticker"] = "ACWI.X"
        s.attrs["requested_ticker"] = "ACWI.X"
        self.benchmark_histories = {"probe": s}


def _mismatch_errors(errors):
    return [e for e in errors if "intraday preprocessing candidates differ" in e]


def test_historyless_holding_does_not_block_intraday_check():
    """A holding absent from holding_performance must not be demanded."""
    hp = pd.DataFrame([
        {"ticker": "AVEM.DE", "type": "In portfolio"},
        {"ticker": "CNDX.L", "type": "Benchmark index"},
    ])
    # 18MF.MU is valuation-accepted but has too little history for hp.
    holdings_df = pd.DataFrame([{"ticker": "AVEM.DE"}, {"ticker": "18MF.MU"}])
    requested = ("AVEM.DE", "CNDX.L")
    quotes = {"AVEM.DE": {"intraday_source_ticker": "AVEM.DE",
                          "intraday_series": [1.0, 2.0],
                          "intraday_baseline": 1.0},
              "CNDX.L": {"intraday_source_ticker": "CNDX.L",
                         "intraday_series": [1.0, 2.0],
                         "intraday_baseline": 1.0}}
    audit = {"performance_intraday": {
        "origin": "metrics_preprocessing",
        "requested_tickers": requested,
        "returned_tickers": tuple(quotes),
        "source_tickers": {k: k for k in quotes},
    }}
    errors = validate_newsletter_semantics(
        _M(hp, holdings_df, requested, quotes), audit, "")
    assert _mismatch_errors(errors) == [], (
        f"a historyless holding must not block delivery; got {_mismatch_errors(errors)}"
    )


def test_renderer_dropping_a_performance_ticker_is_still_caught():
    """The check must keep its teeth: a request set missing an hp ticker fails."""
    hp = pd.DataFrame([
        {"ticker": "AVEM.DE", "type": "In portfolio"},
        {"ticker": "CL2.MI", "type": "In portfolio"},
    ])
    requested = ("AVEM.DE",)  # CL2.MI silently dropped
    quotes = {"AVEM.DE": {"intraday_source_ticker": "AVEM.DE",
                          "intraday_series": [1.0, 2.0],
                          "intraday_baseline": 1.0}}
    audit = {"performance_intraday": {
        "origin": "metrics_preprocessing",
        "requested_tickers": requested,
        "returned_tickers": tuple(quotes),
        "source_tickers": {k: k for k in quotes},
    }}
    errors = validate_newsletter_semantics(
        _M(hp, pd.DataFrame([{"ticker": "AVEM.DE"}]), requested, quotes), audit, "")
    assert _mismatch_errors(errors), "the gate went blind to a dropped ticker"


# ── Benchmark availability degrades; only critical benchmarks block ──────────
# A single tracked index (X25E) with no history under Yahoo throttling used to
# block the whole digest and cost three sends. Only the geo/alpha-beta
# reference the "vs market" chart is drawn against is required now.

class _Bench:
    """Minimal metrics whose only contract is the benchmark universe."""
    degraded_computers = ()
    benchmark_comparison = None
    holding_performance = None
    historical_risk = {}
    ticker_resolutions = ()

    def __init__(self, selected, resolution_errors):
        from tarzan import config as cfg
        self.benchmark_tickers = selected
        self.benchmark_resolution_errors = resolution_errors
        defs = cfg.benchmarks()
        histories = {}
        for name, ticker in selected.items():
            s = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-07-01", "2026-07-02"]))
            s.name = ticker
            s.attrs["resolved_ticker"] = ticker
            s.attrs["requested_ticker"] = defs.get(name, ticker)
            histories[name] = s
        self.benchmark_histories = histories


def _critical_name():
    from tarzan import config as cfg
    return cfg.benchmark_geo_allocation()


def test_unavailable_tracked_benchmark_does_not_block():
    crit = _critical_name()
    from tarzan import config as cfg
    selected = {crit: cfg.benchmarks()[crit]}  # only the critical one resolved
    errors = validate_newsletter_semantics(
        _Bench(selected, ("Xtrackers II Eurozone Government Bond 25+: no usable history for X25E",)),
        {}, "")
    joined = " | ".join(errors)
    assert "X25E" not in joined, joined
    assert "did not resolve" not in joined, joined
    assert "catalog" not in joined, joined


def test_unresolved_critical_benchmark_blocks():
    errors = validate_newsletter_semantics(
        _Bench({}, ("iShares MSCI ACWI: no usable history for ISAC",)), {}, "")
    joined = " | ".join(errors)
    assert "critical benchmark(s) did not resolve" in joined, joined
