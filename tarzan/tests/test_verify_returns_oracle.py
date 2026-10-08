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


def _section_html(ordinal: str, label: str, body: str = "") -> str:
    """The shape the template emits for a section header."""
    return (f'<span style="x">[{ordinal}]</span>&nbsp;&nbsp;'
            f'<span style="y">{label}</span>{body}')


class TestFindingTheReturnsTable:
    """The oracle locates sections by LABEL, and this is why.

    It used to slice between the literal strings "[06]" and "[07]". When Portfolio
    movers took fourth place, every ordinal after it moved by one: Returns became [07]
    and the slice landed on Allocation. The step then read no period columns and
    printed "0 figures compared, 0 disagreeing" — a green line for a check that had
    compared nothing.
    """

    def test_it_finds_the_section_by_its_label(self, oracle):
        html = (_section_html("06", "Allocation", "ALLOC-BODY")
                + _section_html("07", "Returns", "RETURNS-BODY")
                + _section_html("08", "Watchlist", "WATCH-BODY"))
        assert "RETURNS-BODY" in oracle._section(html, "Returns")
        assert "ALLOC-BODY" not in oracle._section(html, "Returns")
        assert "WATCH-BODY" not in oracle._section(html, "Returns")

    def test_the_ordinal_may_move_without_breaking_it(self, oracle):
        """The regression itself: same document, Returns renumbered."""
        for ordinal in ("06", "07", "11"):
            html = (_section_html("05", "Allocation", "ALLOC")
                    + _section_html(ordinal, "Returns", "RETURNS-BODY"))
            assert "RETURNS-BODY" in oracle._section(html, "Returns"), ordinal

    def test_a_missing_section_yields_nothing_rather_than_the_whole_page(self, oracle):
        html = _section_html("06", "Allocation", "ALLOC")
        assert oracle._section(html, "Returns") == ""
        assert oracle._header_keys(html) == []
        assert oracle._rendered_rows(html) == {}

    def test_the_period_columns_come_from_the_table_header(self, oracle):
        html = _section_html("07", "Returns", (
            "<table><tr>"
            "<td>Instrument</td><td>1D</td><td>5D</td><td>1M</td><td>3Y</td>"
            "</tr></table>"))
        assert oracle._header_keys(html) == ["1d", "5d", "1m", "3y"]


class TestIntradayEveryVenue:
    """Check 3: a holding's live 1D against the CONSENSUS of the instrument's venues.

    Rebuilt from 8 Oct 2026. At 11:08 the issue printed -4.38% for a sleeve whose Xetra
    line had made one off-market trade at 09:21; Paris, trading since, said -0.07%.
    Check 2 could never see this — Yahoo's own pair for that venue said -4.38% too.

    The first version of this check measured distance from the SPAN of the venues and
    could never fire: the span always contains the engine's own venue. These tests use
    the real 11:08 numbers, which is what would have caught that.
    """

    SIGMA_PP = 0.88   # the fund's daily volatility, ~14% a year

    @staticmethod
    def _at(h, m):
        return dt.datetime(2026, 10, 8, h, m, tzinfo=dt.timezone.utc)

    def test_the_8_oct_figure_is_a_finding(self, oracle):
        v = oracle.intraday_offside(
            -4.38, {"X.DE": (-4.38, self._at(7, 21)), "X.PA": (-0.07, self._at(8, 26))},
            self.SIGMA_PP)
        assert v["consensus"] == pytest.approx(-0.07)
        assert v["off"] > v["allowance"], v

    def test_the_corrected_figure_is_not(self, oracle):
        v = oracle.intraday_offside(
            -0.07, {"X.DE": (-4.38, self._at(7, 21)), "X.PA": (-0.07, self._at(8, 26))},
            self.SIGMA_PP)
        assert v["off"] == pytest.approx(0.0)

    def test_a_dispute_between_live_venues_is_reported_and_resolved_small(self, oracle):
        """11:18 the same day: 43 minutes apart, so both venues vote; 1.55pp apart, so
        the figure is disputed and the smaller mover is the consensus."""
        venues = {"X.DE": (-1.67, self._at(8, 35), 403), "X.PA": (-0.12, self._at(9, 18), 473)}
        v = oracle.intraday_offside(-0.12, venues, self.SIGMA_PP)
        assert v["disputed"] and v["consensus"] == pytest.approx(-0.12)
        assert v["off"] == pytest.approx(0.0)
        # ...and the engine's ORIGINAL -1.67% would now be a finding.
        bad = oracle.intraday_offside(-1.67, venues, self.SIGMA_PP)
        assert bad["off"] > bad["allowance"]

    def test_the_busier_venue_is_the_consensus_even_with_the_larger_move(self, oracle):
        """The second fund that day: the smaller mover's previous close was one share."""
        venues = {"Y.DE": (-3.65, self._at(9, 44), 188), "Y.PA": (-0.92, self._at(9, 10), 175)}
        v = oracle.intraday_offside(-3.65, venues, 1.25)
        assert v["disputed"] and v["consensus"] == pytest.approx(-3.65)
        assert v["off"] == pytest.approx(0.0)

    def test_agreeing_venues_are_not_a_dispute(self, oracle):
        v = oracle.intraday_offside(
            -0.85, {"Y.MI": (-0.79, self._at(9, 40)), "Y.DE": (-0.91, self._at(9, 42))},
            0.8)
        assert not v["disputed"] and v["off"] <= v["allowance"]

    def test_the_clock_between_two_observations_is_tolerated(self, oracle):
        """The case the oracle's own history documents: a 2x line read -0.36% on our
        tape and +0.94% on a quote fetched moments later. That 1.30pp is the clock,
        and a 2x line's sigma carries it."""
        v = oracle.intraday_offside(-0.36, {"CL.MI": (0.94, self._at(7, 33))}, 2.4)
        assert v["off"] == pytest.approx(1.30, abs=0.01)
        assert v["off"] <= v["allowance"]

    def test_a_low_volatility_line_still_gets_the_floor(self, oracle):
        """A money-market fund's sigma is a few basis points; 1.5 sigmas of it would
        flag any quote two minutes apart. The 1pp floor is what stops that."""
        v = oracle.intraday_offside(0.01, {"M.MI": (0.0, self._at(9, 0))}, 0.01)
        assert v["allowance"] == 1.0
