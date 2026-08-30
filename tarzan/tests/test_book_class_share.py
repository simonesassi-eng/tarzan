"""The Book's "% Class" column must be a share, and every holding must be counted.

A holding whose ``asset_class`` is unset used to divide its value by a ONE EURO
default and print €3,013.09 as 301309.3%: the per-class totals were keyed on the
raw column (which ``groupby`` drops None/NaN from) while each row looked its total
up under a normalised key, so no "Other" entry could ever exist. The same split
made the summary chips count only the classed holdings.

These tests assert both halves — the share and the count — because fixing the
percentage while leaving a rendered group with no chip is not fixing it.
"""

from __future__ import annotations

import re

import pandas as pd

from tarzan.export.newsletter._sections_alloc import _build_holdings
from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics

#: The "% Class" cell is the only one rendered at this width in a Book row.
_CLASS_CELL = re.compile(r'width:52px;">([^<]+)</td>')


def _book(*classes) -> dict:
    """Render The Book for one holding per ``classes`` entry, €1k…€Nk each."""
    rows = []
    for i, klass in enumerate(classes):
        value = 1000.0 * (i + 1)
        rows.append({
            "isin": f"US000000000{i}", "ticker": f"AA{i}", "name": f"Alpha {i}",
            "asset_class": klass, "current_value": value,
            "cost_basis_eur": value, "weight_pct": 0.0, "gain_pct": 0.0,
            "gain_eur": 0.0, "quantity": 10.0, "avg_purchase_price": value / 10,
            "pct_of_class": 0.0, "currency": "EUR",
        })
    df = pd.DataFrame(rows)
    total = float(df["current_value"].sum())
    m = PortfolioMetrics(total_value=total, invested_value=total, cash_value=0.0,
                         holdings_df=df)
    return _build_holdings(_NewsletterContext(metrics=m, config=InvestorConfig()))


def _class_shares(table_html: str) -> list:
    """Every rendered "% Class" FIGURE, as floats.

    Anything that is not a number is skipped: the column header ("% Class") and
    the em-dash a withheld share renders as are both matched by the width, and
    neither is a figure.
    """
    out = []
    for cell in _CLASS_CELL.findall(table_html):
        text = cell.replace("&nbsp;", " ").strip().rstrip("%")
        try:
            out.append(float(text))
        except ValueError:
            continue
    return out


class TestBookClassShare:
    def test_an_unclassified_holding_gets_a_share_not_its_euro_value(self):
        book = _book("Equities", None, None)
        shares = _class_shares(book["table_html"])
        assert shares, "no % Class figures rendered at all"
        assert all(s <= 100.5 for s in shares), \
            f"% Class is not a share: {shares}"
        # The two unclassified holdings are €2k and €3k of one €5k residual
        # class, so their shares are each other's complement.
        others = sorted(s for s in shares if s < 99.9)
        assert others == [40.0, 60.0], f"residual class shares are wrong: {others}"

    def test_a_nan_class_behaves_like_a_missing_one(self):
        book = _book("Equities", float("nan"), float("nan"))
        shares = _class_shares(book["table_html"])
        assert all(s <= 100.5 for s in shares), \
            f"% Class is not a share: {shares}"
        # ``str(nan or "") or "Other"`` returns the STRING "nan" (NaN is truthy),
        # which named a whole asset class "nan" in the group header and the chip.
        assert ">nan<" not in book["table_html"], "a group header is named 'nan'"
        assert "nan" not in [c["name"] for c in book["summary"]], \
            "a chip is named 'nan'"

    def test_every_holding_is_counted_in_the_class_chips(self):
        book = _book("Equities", None, "Gold", None)
        counted = sum(int(c["count"]) for c in book["summary"])
        assert counted == book["total_count"] == 4, \
            f"chips count {counted} of {book['total_count']} holdings"
        # Chips and group headers must come from one key set: a class with a
        # rendered group and no chip is the same divergence in another place.
        for chip in book["summary"]:
            assert f'>{chip["name"]}</span>' in book["table_html"], \
                f"chip {chip['name']!r} has no group header"

    def test_a_class_summing_to_zero_states_no_share(self):
        """The one case normalising cannot fix: a real class total of zero."""
        df = pd.DataFrame([
            {"isin": "US0000000001", "ticker": "AAA", "name": "Alpha",
             "asset_class": "Gold", "current_value": 500.0, "cost_basis_eur": 500.0,
             "weight_pct": 0.0, "gain_pct": 0.0, "gain_eur": 0.0, "quantity": 1.0,
             "avg_purchase_price": 500.0, "pct_of_class": 0.0, "currency": "EUR"},
            {"isin": "US0000000002", "ticker": "BBB", "name": "Beta",
             "asset_class": "Gold", "current_value": -500.0, "cost_basis_eur": 0.0,
             "weight_pct": 0.0, "gain_pct": 0.0, "gain_eur": 0.0, "quantity": -1.0,
             "avg_purchase_price": 500.0, "pct_of_class": 0.0, "currency": "EUR"},
        ])
        m = PortfolioMetrics(total_value=0.0, invested_value=0.0, cash_value=0.0,
                             holdings_df=df)
        book = _build_holdings(_NewsletterContext(metrics=m, config=InvestorConfig()))
        assert _class_shares(book["table_html"]) == [], \
            "a class whose holdings net to zero must state no share"
