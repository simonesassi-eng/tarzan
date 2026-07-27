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
    # environment. backtest_portfolios stays None: the backtest section is
    # skipped in a deterministic run anyway.
    return render_newsletter(
        metrics=metrics,
        config=config,
        issue_number=31,
        ai_summary="",
        backtest_portfolios=None,
    )


@pytest.fixture
def rendered(tmp_path, monkeypatch):
    import pandas as pd

    from tarzan import orchestrator

    monkeypatch.setattr("tarzan.data.enricher.enrich_holdings", _stub_enrich)
    empty = pd.Series(dtype=float)
    monkeypatch.setattr("tarzan.engine.metrics._fetch_benchmark_history",
                        lambda *a, **k: empty)
    monkeypatch.setattr("tarzan.engine.metrics._build_benchmark_series",
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
        cannot quietly pass. These are the section kickers the template emits.
        """
        html, _m, _c = rendered
        for kicker in ("Portfolio Digest", "Returns snapshot", "Legend"):
            assert kicker in html, f"section {kicker!r} vanished from the render"

    def test_email_safety_floor(self, rendered):
        """No script, no remote asset, no webfont — the properties that make
        this HTML safe to put in a mail client. Never checked before.
        """
        html, _m, _c = rendered
        assert "<script" not in html.lower()
        assert not re.search(r'src\s*=\s*["\']https?://', html, re.I)
        assert not re.search(r'@import|<link[^>]+stylesheet', html, re.I)


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
            ai_summary="A market context note.", backtest_portfolios=None,
        )
        ordinals = self._ordinals(with_ai)
        assert ordinals == list(range(1, len(ordinals) + 1)), ordinals
        assert len(ordinals) == len(self._ordinals(_html)) + 1
