"""Every row of a diversification table must have as many cells as the header.

The total row had no "vs target" cell and the cash row had neither that nor a
trend cell, so on both of them the drift figure rendered under the TREND header
and the last column stayed empty. Nothing failed: HTML happily renders a short
row, and the golden snapshot recorded the misalignment as expected output. Only
looking at the rendered page showed it.

Network-free: the table builder takes plain row dicts.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from tarzan.export.newsletter._sections_alloc import _div_table


class _Grid(HTMLParser):
    """Cells per row of the OUTERMOST table only.

    A regex cannot do this: the Now/Target cells each contain a nested table
    that lines the % up with the EUR amount, so pattern-matching ``<td>`` counts
    those too and every row comes out a different width.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.rows: list[list[str]] = []
        self._depth = 0          # table nesting depth
        self._cell: list[str] = []
        self._in_cell = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._depth += 1
        elif tag == "tr" and self._depth == 1:
            self.rows.append([])
        elif tag == "td":
            if self._depth == 1 and self._in_cell == 0 and self.rows:
                self.rows[-1].append("")
                self._cell = []
            self._in_cell += 1

    def handle_endtag(self, tag):
        if tag == "table":
            self._depth -= 1
        elif tag == "td":
            self._in_cell -= 1
            if self._in_cell == 0 and self.rows and self.rows[-1]:
                self.rows[-1][-1] = "".join(self._cell)

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)

    def handle_entityref(self, name):
        if self._in_cell:
            self._cell.append(f"&{name};")


def _rows(html: str) -> list[list[str]]:
    """Each outer-table row's cell texts, header row first."""
    g = _Grid()
    g.feed(html)
    return [r for r in g.rows if r]


def _value_row(**over) -> dict:
    row = {"label_html": "Equities", "now": 77.6, "target": 75.0,
           "color": "#6E9BFF", "spark_vals": [70.0, 74.0, 77.6]}
    row.update(over)
    return row


class TestColumnCount:
    def test_value_row_matches_the_header(self):
        html = _div_table([_value_row()], tol=2.0, base=300_000.0)
        rows = _rows(html)
        assert len(rows) >= 2
        assert len(rows[1]) == len(rows[0]), "value row vs header"

    def test_total_row_matches_the_header(self):
        html = _div_table(
            [_value_row(),
             _value_row(label_html="Invested Portfolio", is_total=True,
                        now=102.2, target=125.0, leverage=1.02)],
            tol=2.0, base=300_000.0, show_leverage=True,
            first_label="Asset class")
        rows = _rows(html)
        for i, row in enumerate(rows[1:], start=1):
            assert len(row) == len(rows[0]), f"row {i} has {len(row)} cells"

    def test_cash_row_matches_the_header(self):
        html = _div_table(
            [_value_row(),
             {"label_html": "Cash &amp; Cash Eq.", "eur_row": True,
              "now_eur": 9_700.0, "target_eur": 9_700.0, "delta_eur": -18.0,
              "delta_color": "#E55B5B"}],
            tol=2.0, base=300_000.0, first_label="Asset class")
        rows = _rows(html)
        for i, row in enumerate(rows[1:], start=1):
            assert len(row) == len(rows[0]), f"row {i} has {len(row)} cells"

    def test_every_row_type_together(self):
        html = _div_table(
            [_value_row(),
             _value_row(label_html="Fixed Income", now=9.7, target=27.0,
                        leverage=1.38),
             _value_row(label_html="Invested Portfolio", is_total=True,
                        now=102.2, target=125.0, leverage=1.02),
             {"label_html": "Cash", "eur_row": True, "now_eur": 9_700.0,
              "target_eur": 9_700.0, "delta_eur": -18.0,
              "delta_color": "#E55B5B"}],
            tol=2.0, base=300_000.0, show_leverage=True,
            first_label="Asset class")
        rows = _rows(html)
        widths = {len(r) for r in rows}
        assert len(widths) == 1, f"ragged table: row widths {sorted(widths)}"


class TestHeaderContract:
    def test_first_column_is_named_by_the_caller(self):
        html = _div_table([_value_row()], tol=2.0, first_label="Equity geography")
        assert "Equity geography" in html
        assert ">Name<" not in html

    def test_leverage_rides_in_the_now_and_target_cells(self):
        """Actual leverage beside the actual weight, the plan's beside the plan's.

        It used to sit under the drift figure, which put both factors in the column
        about the GAP between them — the one place neither belongs.
        """
        html = _div_table([_value_row(leverage=1.38, target_leverage=1.20)],
                          tol=2.0, show_leverage=True)
        cells = _rows(html)[1]
        assert "1.38\u00d7" in cells[2], "actual leverage belongs in the Now cell"
        assert "1.20\u00d7" in cells[3], "target leverage belongs in the Target cell"
        assert ">Lev<" not in html and "Drift \u00b7 lev" not in html

    def test_a_class_with_no_physical_capital_reads_synth(self):
        """A target leverage of None is not a missing number: it says the plan holds no
        physical capital in the class, so the exposure is entirely synthetic. Dividing
        by that zero would have invented a ratio."""
        html = _div_table([_value_row(leverage=2.17, target_leverage=None)],
                          tol=2.0, show_leverage=True)
        assert "synth" in _rows(html)[1][3]

    def test_without_the_flag_neither_factor_is_printed(self):
        html = _div_table([_value_row(leverage=1.38, target_leverage=1.2)], tol=2.0)
        assert "1.38\u00d7" not in html and "1.20\u00d7" not in html

    def test_leverage_is_printed_at_every_value(self):
        """Not suppressed at ~1.00x: a sleeve carrying exactly its own capital is worth
        stating next to one that does not."""
        html = _div_table([_value_row(leverage=1.0, target_leverage=1.0)], tol=2.0,
                          show_leverage=True)
        assert "1.00\u00d7" in html

    def test_trend_sits_before_drift(self):
        html = _div_table([_value_row()], tol=2.0)
        header = _rows(html)[0]
        labels = header
        assert labels.index("Trend") < labels.index("Drift")


class TestOneLinePerRow:
    """The whole point of the rewrite: a row is one line.

    The old table stacked the percentage over its euro amount and drew a 40px
    sparkline, which is where 1150px of section height came from. ``subs`` and
    ``value_subs`` survive as accepted arguments so the call sites did not all have to
    change at once, but they no longer gate a second line, because there is not one.
    """

    def test_the_euro_amount_rides_beside_the_percentage(self):
        html = _div_table([_value_row()], tol=2.0, base=300_000.0)
        cells = _rows(html)[1]
        assert "%" in cells[2] and "\u20ac" in cells[2], cells[2]

    def test_no_base_means_no_euro_anywhere(self):
        html = _div_table([_value_row()], tol=2.0, base=None)
        assert "\u20ac" not in html

    def test_the_euro_appears_whatever_the_legacy_toggles_say(self):
        for subs, value_subs in ((True, None), (False, None), (False, True),
                                 (True, False)):
            html = _div_table([_value_row()], tol=2.0, base=300_000.0,
                              subs=subs, value_subs=value_subs)
            assert "\u20ac" in html, (subs, value_subs)

    def test_the_trend_move_and_the_drift_are_both_present_and_different(self):
        """Two figures a reader could confuse: the trend is how far the weight moved
        over the month, the drift is how far it sits from target. 70.0 -> 77.6 is +7.6
        of movement; 77.6 against a 75.0 target is +2.6pp of drift.

        Only the drift carries the unit. The trend's sits immediately to its left and
        the two columns are adjacent, so repeating "pp" cost 12px the sparkline needed
        more than the reader needed the letters.
        """
        html = _div_table([_value_row()], tol=2.0, base=300_000.0)
        assert ">+7.6<" in html or "+7.6</span>" in html, "trend move, unitless"
        assert "+2.6pp" in html, "drift, with its unit"
