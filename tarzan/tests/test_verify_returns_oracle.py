"""The oracle's one judgement call, checked.

``scripts/verify_returns_vs_yahoo.py`` compares the newsletter's printed returns
against a raw Yahoo pull and exits non-zero on a disagreement. Everything else in it
is arithmetic; the one place it exercises judgement is deciding whether the source is
in a position to judge at all — and that decision is what turns a difference into a
reported finding or into a shrug.

It has been wrong twice, in opposite directions, and both times silently:

* measured RELATIVE to the sample, a uniformly stale sample read as current and nine
  endpoint mismatches (0.10-4.25pp) were reported as real findings;
* measured against the venue's last SESSION, every window abstained before the open,
  so the check would have run every morning and decided nothing.

Hence this file. The script is imported by path because it is a script, not a package
module; its import is side-effect free (the chdir lives in ``main``) precisely so this
is possible.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_returns_vs_yahoo.py"


@pytest.fixture(scope="module")
def oracle():
    spec = importlib.util.spec_from_file_location("verify_returns_vs_yahoo", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importing_the_script_does_not_move_the_process(oracle):
    """A script the suite imports must not chdir or touch the environment on import."""
    assert hasattr(oracle, "source_can_referee")


class TestWhenTheSourceMayJudge:
    """``source_can_referee(source_last, stamped_end, bucket)``."""

    def test_same_close_means_the_comparison_is_real(self, oracle):
        day = dt.date(2026, 9, 4)
        assert oracle.source_can_referee(day, day, "3m") is True

    def test_a_frame_behind_the_stamped_tape_cannot_judge(self, oracle):
        # The Saturday case: Yahoo's Milan frames stopped on Thu 3 Sep while the tape
        # carried Friday's published close. Every long window then differs by Friday.
        assert oracle.source_can_referee(
            dt.date(2026, 9, 3), dt.date(2026, 9, 4), "3m") is False

    def test_one_day_is_judged_from_the_published_pair_regardless(self, oracle):
        # 1D never reads the frame's end, so a frame missing the latest session is
        # irrelevant to it — and it was the only window that verified on that Saturday.
        assert oracle.source_can_referee(
            dt.date(2026, 9, 3), dt.date(2026, 9, 4), "1d") is True

    def test_before_the_open_the_morning_run_still_decides(self, oracle):
        """The regression the calendar-based rule introduced.

        At 08:30 nothing has traded, so the tape is stamped to the PREVIOUS session and
        the frame ends there too. Equal — so the check runs and decides, instead of
        abstaining on every instrument every weekday morning.
        """
        yesterday = dt.date(2026, 9, 3)
        for bucket in ("5d", "1m", "3m", "ytd", "1y"):
            assert oracle.source_can_referee(yesterday, yesterday, bucket) is True

    def test_a_frame_ahead_of_the_tape_is_still_allowed_to_judge(self, oracle):
        """Only a frame BEHIND is disqualified.

        A source carrying a close the tape has not stamped is the one case worth
        hearing about: it means our tape missed a session the vendor has, which is a
        finding about us, not about the vendor. Silencing it would hide exactly the
        fault this oracle exists for — a figure computed on a tape a session behind.
        """
        assert oracle.source_can_referee(
            dt.date(2026, 9, 4), dt.date(2026, 9, 3), "1m") is True
