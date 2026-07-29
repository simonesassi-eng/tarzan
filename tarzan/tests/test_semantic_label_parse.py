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
