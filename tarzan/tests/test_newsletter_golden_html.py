"""Golden-master gate for the rendered newsletter MARKUP.

``test_golden_master`` pins every reported *number*; nothing pinned the HTML.
The newsletter's visual layer is ~70% Python string builders behind
``|safe`` injections, and the only automated pre-send check on the markup is
the semantic gate, which inspects a handful of chart labels. A redesign — or a
refactor that silently drops a section — was therefore invisible to CI.

This renders the newsletter from the same network-free deterministic pipeline
as the numeric golden and diffs it against a committed file, so any markup
change is a reviewed diff instead of a surprise in an inbox.

Regenerate deliberately, in the same commit as the change that moves it:

    UPDATE_NEWSLETTER_GOLDEN=1 python -m pytest \\
        tarzan/tests/test_newsletter_golden_html.py -q
"""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path

import pytest

from tarzan.export.newsletter import render_newsletter

# Reuse the numeric golden's synthetic portfolio so one dataset backs both
# gates: a data change that moves a number also moves this markup.
from tarzan.tests.test_golden_master import _AS_OF, _ORDERS_CSV, _stub_enrich

GOLDEN_PATH = Path(__file__).parent / "golden" / "newsletter.html"


def _render(metrics, config) -> str:
    # ai_summary="" forces the market-context block off explicitly rather than
    # relying on the absence of a key, so the golden cannot depend on the
    # environment.
    return render_newsletter(
        metrics=metrics,
        config=config,
        issue_number=31,
        ai_summary="",
    )


@pytest.fixture
def rendered(tmp_path, monkeypatch):
    import pandas as pd

    from tarzan import orchestrator

    monkeypatch.setattr("tarzan.data.enricher.enrich_holdings", _stub_enrich)
    empty = pd.Series(dtype=float)
    monkeypatch.setattr("tarzan.engine.metrics._fetch_benchmark_history",
                        lambda *a, **k: empty)
    orders = tmp_path / "order_list.csv"
    orders.write_text(_ORDERS_CSV)
    metrics, config = orchestrator.run(
        config_source=None, orders_source=str(orders),
        targets_per_holding_source=None,
        deterministic=True, as_of=_AS_OF,
    )
    return _render(metrics, config), metrics, config


class TestNewsletterGoldenHtml:
    def test_render_is_reproducible(self, rendered):
        """Two renders of one analysis must be byte-identical.

        Guards the deterministic-id contract: ``build_context`` resets the SVG
        clipPath counters per render, and the footer stamp comes from the
        run-owned clock, so nothing may leak process state into the markup.
        """
        html, metrics, config = rendered
        assert _render(metrics, config) == html

    def test_the_masthead_is_stamped_from_the_run_clock(self, rendered):
        """The issue date must be the effective date, not the wall clock.

        Two defects in one: under ``--as_of`` a masthead stamped "today" dates
        the issue to a day none of the figures below it describe, and the golden
        above then failed on every day except the one it was regenerated on — a
        gate that expires overnight instead of catching a regression.
        """
        html, _metrics, _config = rendered
        assert _AS_OF.strftime("%a, %d %b %Y") in html, (
            "masthead is not stamped from the run-owned clock")

    def test_markup_matches_golden(self, rendered):
        html, _metrics, _config = rendered
        if os.environ.get("UPDATE_NEWSLETTER_GOLDEN") == "1":
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(html, encoding="utf-8")
            pytest.skip(f"golden regenerated: {GOLDEN_PATH}")
        assert GOLDEN_PATH.exists(), (
            f"missing golden {GOLDEN_PATH}; regenerate with "
            "UPDATE_NEWSLETTER_GOLDEN=1"
        )
        expected = GOLDEN_PATH.read_text(encoding="utf-8")
        if html != expected:
            diff = "\n".join(list(difflib.unified_diff(
                expected.splitlines(), html.splitlines(),
                fromfile="golden", tofile="rendered", lineterm="", n=1,
            ))[:60])
            pytest.fail(
                "rendered newsletter differs from the committed golden.\n"
                "If the change is intended, regenerate it in the same commit:\n"
                "  UPDATE_NEWSLETTER_GOLDEN=1 python -m pytest "
                f"{Path(__file__).name} -q\n\nFirst differences:\n{diff}"
            )

    def test_golden_carries_every_section(self, rendered):
        """A structural floor, so a golden regenerated from a broken render
        cannot quietly pass. These are the section labels the template emits on
        the deterministic fixture.
        """
        html, _m, _c = rendered
        # Optimizer is absent on this fixture (no rebalance suggestions), so it
        # is not part of the floor; the ordering test covers it when present.
        for label in ("State", "Portfolio", "Allocation", "The book",
                      "Returns", "Watchlist", "Attribution", "Risk",
                      "Appendix"):
            assert f">{label}</span>" in html, (
                f"section {label!r} vanished from the render")

    def test_section_ordinals_match_the_concept_order(self, rendered):
        """The order is the argument the document makes: what the portfolio is,
        what it did, whether that beat the alternative, and so on. A section that
        moves changes the reading, so the sequence is pinned rather than left to
        whoever edits the template next.
        """
        html, _m, _c = rendered
        labels = re.findall(
            r'\[\d\d\]</span>&nbsp;&nbsp;<span[^>]*>([^<]+)</span>', html)
        expected = ["State", "Portfolio", "Vs the market", "Markets",
                    "Allocation", "Returns", "Watchlist", "The book",
                    "Attribution", "Risk", "Optimizer", "Strategy", "Appendix"]
        # Sections the deterministic fixture cannot fill are absent, not
        # reordered, so compare as a subsequence of the intended order.
        assert labels == [x for x in expected if x in labels], labels

    def test_email_safety_floor(self, rendered):
        """No script, no remote asset, no webfont — the properties that make
        this HTML safe to put in a mail client. Never checked before.
        """
        html, _m, _c = rendered
        assert "<script" not in html.lower()
        assert not re.search(r'src\s*=\s*["\']https?://', html, re.I)
        assert not re.search(r'@import|<link[^>]+stylesheet', html, re.I)

    def test_kpi_tiles_never_collapse_to_one_column(self, rendered):
        """The STATE grid must hold three columns at every width it is actually
        read at, with no narrow-viewport rule collapsing it further.

        A narrow-pane fallback used to stack the tiles one per line below
        520px — meant for a pane narrower than the card, but every phone
        viewport is narrower than that, so it fired unconditionally on the
        one device this mail is read on: nine tiles, one per line, every
        time. Three 33.33% columns hold their structure at any width down
        to an iPhone SE, so no such rule should exist any more.
        """
        html, _m, _c = rendered
        breakpoints = {
            int(width): body
            for width, body in re.findall(
                r'@media only screen and \(max-width: (\d+)px\)\s*\{(.*?)\n    \}',
                html, re.S,
            )
        }
        stack = [w for w, body in breakpoints.items() if "kpi-cell" in body]
        reflow = [w for w, body in breakpoints.items() if ".container" in body]
        assert not stack, f"the STATE grid must not collapse at any width: {stack}"
        assert len(reflow) == 1, breakpoints.keys()

    def test_state_grid_does_not_depend_on_the_breakpoint_alone(self, rendered):
        """The tiles must hold three columns even where media queries do not run.

        Fixed table layout (declared widths are authoritative) and
        text-size-adjust:100% (declared font sizes are the rendered ones)
        both keep a client from auto-sizing a column to an inflated
        monospace value and overflowing the row — independent of whether
        any media query in the document runs at all.
        """
        html, _m, _c = rendered
        assert "-webkit-text-size-adjust: 100%" in html, (
            "without this Apple Mail inflates the tile values and overflows the row"
        )
        # Every STATE tile row is a fixed-layout table with explicit cell widths.
        rows = re.findall(
            r'<table[^>]*style="table-layout:fixed;[^"]*"[^>]*>\s*<tr>(.*?)</tr>',
            html, re.S,
        )
        tile_rows = [r for r in rows if 'width="33.33%"' in r]
        assert tile_rows, "the STATE grid must use a fixed table layout"
        for row in tile_rows:
            cells = re.findall(r'<td[^>]*style="width:([0-9.]+)%', row)
            assert len(cells) >= 1, "each tile cell needs an explicit CSS width"
            assert all(abs(float(c) - 33.33) < 0.01 for c in cells), cells


class TestSectionNumbering:
    """Section ordinals must run 1..N over the sections actually rendered.

    The kicker macro increments a namespace counter rather than taking a fixed
    number per section, because optional blocks (market context, optimizer) come
    and go: a static ordinal left holes in the sequence — [02] with no [01] —
    whenever one was unavailable.
    """

    @staticmethod
    def _ordinals(html: str) -> list[int]:
        return [int(n) for n in re.findall(
            r'\[(\d\d)\]</span>&nbsp;&nbsp;', html)]

    def test_sequence_has_no_holes_without_optional_blocks(self, rendered):
        html, _m, _c = rendered
        ordinals = self._ordinals(html)
        assert ordinals, "no numbered section kickers rendered"
        assert ordinals == list(range(1, len(ordinals) + 1)), ordinals

    def test_sequence_absorbs_an_optional_block(self, rendered):
        """Adding the market-context block shifts the rest by one and still
        leaves a gapless run."""
        _html, metrics, config = rendered
        with_ai = render_newsletter(
            metrics=metrics, config=config, issue_number=31,
            ai_summary="A market context note.",
        )
        ordinals = self._ordinals(with_ai)
        assert ordinals == list(range(1, len(ordinals) + 1)), ordinals
        assert len(ordinals) == len(self._ordinals(_html)) + 1


class TestReturnsHeat:
    """Conditional formatting must be scaled on each column's own extremes.

    The property that matters is saturation: within a column, the table's most
    negative value has to reach the saturated red end and its most positive the
    saturated green end. A scale pooled across columns, or across both grids,
    leaves every cell pale and the formatting says nothing.
    """

    def test_column_extremes_saturate(self, rendered):
        from tarzan.export import _heat

        html, _m, _c = rendered
        # Cell backgrounds in the returns grids carry tabular-nums; row and
        # group surfaces do not, which keeps this off the table furniture.
        tinted = re.findall(
            r'background:(#[0-9A-Fa-f]{6});[^"]*font-variant-numeric', html)
        assert tinted, "no returns cell carries a conditional-format tint"
        saturated = {
            _heat.heat_bg(1.0, neg=-1.0, pos=1.0).upper(),
            _heat.heat_bg(-1.0, neg=-1.0, pos=1.0).upper(),
        }
        seen = {t.upper() for t in tinted}
        assert seen & saturated, (
            "no column reaches the end of its ramp; the scale is not "
            f"normalised per column. saturated={sorted(saturated)}"
        )

    def test_blank_cell_is_not_tinted(self):
        from tarzan.export import _heat
        assert _heat.heat_bg(None, neg=-5.0, pos=5.0) is None

    def test_near_zero_stays_on_the_surface(self):
        """A value a hair off zero must not pick up a tint that reads as a
        signal — the ramp has a dead zone at the bottom."""
        from tarzan.export import _heat
        assert _heat.heat_bg(0.01, neg=-20.0, pos=20.0) is None

    def test_day_column_is_damped(self):
        """The 1D cell also carries the session sparkline, so its tint is
        capped to keep the chart on top readable."""
        from tarzan.export import _heat
        full = _heat.heat_bg(5.0, neg=-5.0, pos=5.0)
        damped = _heat.heat_bg(5.0, neg=-5.0, pos=5.0, damp=_heat.DAY_DAMP)
        assert full != damped

    def test_figure_colour_takes_two_values_and_never_the_sign(self):
        """Across a grid only the BACKGROUND may vary.

        The figures used to be sign-coloured, so a returns row carried the sign
        twice -- once in the cell, once in the text -- and the text colour
        changed from row to row for a reason unrelated to the tint. The ramp now
        returns the figure colour with the background, and it has exactly two
        values: ink on a cell strong enough to hold it, a mid grey otherwise.
        """
        from tarzan.export import _heat
        from tarzan.export._palette import PALETTE

        colours = {
            _heat.heat(v, neg=-8.0, pos=8.0)[1]
            for v in (0.0, 0.4, 2.0, 5.0, 8.0, -0.4, -2.0, -5.0, -8.0)
        }
        assert len(colours) == 2, colours
        assert PALETTE["ink"] in colours
        # Never the signal green or red: those are figure colours elsewhere, and
        # in a tinted grid they would restate the background.
        assert PALETTE["green"] not in colours
        assert PALETTE["red"] not in colours

    def test_the_ramp_ends_are_held_back_from_the_signal_colours(self):
        """A cell at the end of its ramp is a surface with a number on it, not a
        block of the same green the number would be written in."""
        from tarzan.export import _heat
        from tarzan.export._palette import PALETTE

        end_up = _heat.heat(8.0, neg=-8.0, pos=8.0)[0]
        end_down = _heat.heat(-8.0, neg=-8.0, pos=8.0)[0]
        assert end_up.upper() != PALETTE["green"].upper()
        assert end_down.upper() != PALETTE["red"].upper()

    def test_the_returns_grid_has_no_alternating_stripe(self, rendered):
        """The zebra stripe fights the heat: under a tinted matrix an uncoloured
        cell has to be one surface, or the stripe reads as a signal too."""
        html, _m, _c = rendered
        start = html.find(">Returns</span>")
        end = html.find(">Watchlist</span>")
        assert start != -1 and end > start
        from tarzan.export._palette import PALETTE
        assert PALETTE["zebra"] not in html[start:end]


class TestNoRaggedTables:
    """Every row of every table must span the same number of columns.

    A row one cell short does not fail to render: HTML lays it out happily and
    every figure after the gap moves one column left, under a heading that is
    not its own. The diversification total and cash rows shipped that way, and
    the golden had recorded it as expected output — the defect is invisible in a
    diff because the markup is exactly what the builder produced. So this gate
    checks the structural invariant instead of the bytes.

    Colspan counts as the columns it spans, so a group header row spanning the
    whole table is not a short row.
    """

    @staticmethod
    def _ragged(html: str) -> list[tuple[str, list[int]]]:
        from html.parser import HTMLParser

        class Scan(HTMLParser):
            def __init__(self) -> None:
                super().__init__(convert_charrefs=True)
                self.open: list[dict] = []
                self.bad: list[tuple[str, list[int]]] = []

            def handle_starttag(self, tag, attrs):
                if tag == "table":
                    self.open.append({"rows": [], "label": None})
                elif tag == "tr" and self.open:
                    self.open[-1]["rows"].append(0)
                elif tag in ("td", "th") and self.open and self.open[-1]["rows"]:
                    span = 1
                    for key, value in attrs:
                        if key == "colspan":
                            try:
                                span = max(1, int(value))
                            except (TypeError, ValueError):
                                span = 1
                    self.open[-1]["rows"][-1] += span

            def handle_endtag(self, tag):
                if tag == "table" and self.open:
                    t = self.open.pop()
                    widths = [r for r in t["rows"] if r]
                    if len(widths) > 1 and len(set(widths)) > 1:
                        self.bad.append((t["label"] or "?", widths))

            def handle_data(self, data):
                text = data.strip()
                if (text and len(text) > 1 and self.open
                        and self.open[-1]["label"] is None):
                    self.open[-1]["label"] = text[:40]

        scan = Scan()
        scan.feed(html)
        return scan.bad

    def test_no_table_has_rows_of_differing_width(self, rendered):
        html, _m, _c = rendered
        ragged = self._ragged(html)
        assert not ragged, "\n".join(
            f"{label}: row widths {widths}" for label, widths in ragged)

    def test_the_check_can_detect_a_short_row(self):
        """Guard against a vacuous gate."""
        short = ("<table><tr><td>a</td><td>b</td></tr>"
                 "<tr><td>c</td></tr></table>")
        assert self._ragged(short), "the detector must flag a short row"

    def test_a_colspan_group_header_is_not_a_short_row(self):
        grouped = ('<table><tr><td>a</td><td>b</td></tr>'
                   '<tr><td colspan="2">group</td></tr>'
                   '<tr><td>c</td><td>d</td></tr></table>')
        assert not self._ragged(grouped)


class TestTheDocumentIsWellFormed:
    """No ampersand reaches the document unescaped, anywhere.

    The digest template is ``.html.j2``, an extension ``select_autoescape()``
    does not match, so Jinja escapes nothing on the way out and every builder
    has to escape at its own markup boundary. Fixing them one at a time is
    whack-a-mole — this asserts the property over the WHOLE rendered document,
    so a new unescaped interpolation fails here wherever it is added.

    It has teeth on the synthetic fixture: asset-class names ("Cash & Cash
    Equivalents") and the P&L tile labels both carry an ampersand, and four
    separate builders were emitting them raw.
    """

    _ENTITY = re.compile(
        r"&(?!(?:amp|lt|gt|quot|apos|nbsp|middot|rsquo|lsquo|ndash|mdash|"
        r"minus|times|hellip|bull|deg|euro|copy|#\d+|#x[0-9a-fA-F]+);)"
    )

    def test_no_unescaped_ampersand_anywhere(self, rendered):
        html, _metrics, _config = rendered
        offenders = [
            html[max(0, m.start() - 60):m.start() + 30]
            for m in self._ENTITY.finditer(html)
        ]
        assert not offenders, (
            f"{len(offenders)} unescaped ampersand(s) in the rendered digest; "
            "escape at the builder that interpolates the text:\n  "
            + "\n  ".join(repr(o) for o in offenders[:8])
        )

    def test_the_guard_would_catch_a_raw_ampersand(self):
        """The regex must not be so permissive that it never fires."""
        assert self._ENTITY.search('<td>Cash & Cash Equivalents</td>')
        assert not self._ENTITY.search('<td>Cash &amp; Cash Equivalents</td>')
        assert not self._ENTITY.search('<td>&nbsp;&#8364;&minus;1</td>')
