"""Untrusted display text cannot become markup in the actual .html.j2 files."""

from dataclasses import replace
from html import escape, unescape
from html.parser import HTMLParser

from markupsafe import Markup
import pytest

from tarzan.contracts.report import ReportView
from tarzan.export import newsletter
from tarzan.export.newsletter._format import _escaped_text
from tarzan.export.newsletter._semantic import validate_newsletter_semantics
from tarzan.tests.test_newsletter_golden_html import GOLDEN_PATH, rendered  # noqa: F401 — fixture


PAYLOAD = '<img src="https://invalid.example/x" onerror="alert(1)"> & Smith\'s'


class Document(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.events = []

    def handle_starttag(self, tag, attrs):
        self.events.append(("start", tag, tuple(attrs)))

    def handle_startendtag(self, tag, attrs):
        self.events.append(("empty", tag, tuple(attrs)))

    def handle_endtag(self, tag):
        self.events.append(("end", tag))

    def handle_data(self, data):
        if data.strip():
            # Raw "<" in the old legend makes HTMLParser split a text node.
            # Combine adjacent callbacks before comparing decoded documents.
            if self.events and self.events[-1][0] == "text":
                self.events[-1] = ("text", self.events[-1][1] + data)
            else:
                self.events.append(("text", data))


def assert_no_active_payload(html):
    parsed = Document()
    parsed.feed(html)
    for event in parsed.events:
        if event[0] in {"start", "empty"}:
            assert event[1] not in {"img", "script", "iframe", "object", "embed"}
            assert not any(name.lower().startswith("on") for name, _ in event[2])
    assert '&lt;img' in html


@pytest.mark.parametrize("value", [PAYLOAD, Markup(PAYLOAD)])
def test_typed_escaped_text_rejects_already_marked_untrusted_html(value):
    escaped = _escaped_text(value)
    assert isinstance(escaped, Markup)
    assert str(escaped) == escape(PAYLOAD)
    assert unescape(str(escaped)) == PAYLOAD


def isolated_cube(rendered):
    _, original, investor = rendered
    return ReportView.build(original, investor).project()


@pytest.mark.parametrize("field", ["history_ticker", "intraday_effective_ticker", "portfolio_presence", "name"])
def test_provider_broker_and_feed_text_is_escaped_in_the_real_template(rendered, field):
    metrics, investor = isolated_cube(rendered)
    record = {
        "canonical_ticker": "UNLISTED.X", "isin": "XX0000000001", "name": "Unknown",
        "history_ticker": "UNLISTED.X", "intraday_effective_ticker": "",
        "portfolio_presence": "Current",
    }
    record[field] = PAYLOAD
    metrics.ticker_resolutions = (record,)
    html = newsletter.render_newsletter(metrics, investor, ai_summary="")
    assert_no_active_payload(html)
    assert "&amp;amp; Smith" not in html


def test_plain_risk_note_and_benchmark_name_are_not_trusted_markup(rendered):
    metrics, investor = isolated_cube(rendered)
    metrics.report_reference = replace(metrics.report_reference, benchmark_alpha_beta=PAYLOAD)
    metrics.historical_risk = {
        "available": True,
        "portfolio": {"span_label": PAYLOAD, "note": PAYLOAD, "metrics": {"beta": 1.0}},
    }
    html = newsletter.render_newsletter(metrics, investor, ai_summary="")
    assert_no_active_payload(html)
    assert PAYLOAD in unescape(html)


def test_ai_text_is_escaped_before_trusted_colored_html_and_charts_still_render(rendered):
    _, metrics, investor = rendered
    html = newsletter.render_newsletter(metrics, investor, ai_summary=f"[Today] {PAYLOAD} +1.23%")
    assert_no_active_payload(html)
    assert "+1.23%" in html and "<svg" in html
    assert "&lt;svg" not in html


def test_market_metadata_category_is_escaped_inside_prebuilt_html(rendered):
    metrics, investor = isolated_cube(rendered)
    metrics.market_quotes = ({
        "name": PAYLOAD, "symbol": "UNKNOWN.X", "category": PAYLOAD,
        "value": 101.0, "pct": 1.0, "baseline": 100.0, "spark": [100.0, 101.0],
    },)
    html = newsletter.render_newsletter(metrics, investor, ai_summary="")
    assert_no_active_payload(html)


def test_section_subtitle_and_plan_label_are_text_in_the_actual_template(rendered, monkeypatch):
    metrics, investor = isolated_cube(rendered)
    monkeypatch.setattr(newsletter, "_build_optimizer", lambda ctx: {
        "available": True, "subtitle": PAYLOAD, "plans": [{
            "label": PAYLOAD, "active": True, "n_buy": 0, "n_sell": 0,
            "total_buy": "0", "total_sell": "0", "actions": [],
        }],
    })
    html = newsletter.render_newsletter(metrics, investor, ai_summary="")
    assert_no_active_payload(html)
    assert PAYLOAD in unescape(html)


def test_ampersands_remain_single_escaped_and_the_css_font_literal_remains_css(rendered):
    html, _, _ = rendered
    assert ">Total P&amp;L<" in html
    assert "Total P&amp;amp;L" not in html
    style = html.split("<style>", 1)[1].split("</style>", 1)[0]
    assert "'Liberation Mono'" in style
    assert "&#39;" not in style and "&#x27;" not in style


def test_escaping_change_preserves_the_decoded_golden_document(rendered):
    """Review serialized escapes without blindly blessing a new layout/number."""
    html, metrics, _ = rendered
    before, after = Document(), Document()
    original = GOLDEN_PATH.read_text()
    before.feed(original)
    after.feed(html)
    assert unescape(original) == unescape(html)
    assert before.events == after.events
    # This fixture deliberately has no critical benchmark history. Escaping
    # must preserve its rejection, not pretend this synthetic issue can send.
    audit = {}
    newsletter.build_context(metrics, rendered[2], ai_summary="", semantic_audit=audit)
    baseline_errors = validate_newsletter_semantics(metrics, audit, original)
    assert any("critical benchmark(s) did not resolve" in error for error in baseline_errors)
    assert validate_newsletter_semantics(metrics, audit, html) == baseline_errors
