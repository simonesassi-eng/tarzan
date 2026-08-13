"""The appendix feed audit: one row per instrument, banded by presence.

The table lists holdings AND watchlist instruments, which live in two different
places (resolution records / the benchmark catalog) and can be the same
instrument. These assert the merge: held beats tracked beats held-once, an
instrument in both sources appears once, and a feedless holding stays visible.

Network-free: the builder only reads attributes off metrics.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.export.newsletter._sections_alloc import _build_ticker_sources


def _record(ticker, isin, presence, *, name="", hist=None, intraday=""):
    return {
        "canonical_ticker": ticker,
        "isin": isin,
        "name": name or ticker,
        "portfolio_presence": presence,
        "history_ticker": ticker if hist is None else hist,
        "current_ticker": "",
        "intraday_effective_ticker": intraday,
    }


def _rows(monkeypatch, records, benchmarks=(), *, resolved=None, quotes=None):
    from tarzan import config as cfg

    monkeypatch.setattr(cfg, "benchmark_identities", lambda: tuple(benchmarks))
    # Names/ISINs below are deliberately absent from the real taxonomy, so the
    # curated-name lookup falls through to the record's own name.
    metrics = SimpleNamespace(
        ticker_resolutions=tuple(records),
        benchmark_tickers=dict(resolved or {}),
        intraday_quotes=dict(quotes or {}),
    )
    out = _build_ticker_sources(_NewsletterContext(metrics=metrics, config=None))
    return out["available"], out["rows"]


class TestPresenceBands:

    def test_bands_come_first_then_alphabetical_by_ticker(self, monkeypatch):
        available, rows = _rows(
            monkeypatch,
            [_record("ZED.MI", "XX0000000001", "Current + Historical"),
             _record("ABLE.MI", "XX0000000002", "Historical only"),
             _record("MID.MI", "XX0000000003", "Current")],
            [("Watch Two", "WTWO", "XX0000000004"),
             ("Watch One", "WONE", "XX0000000005")],
            resolved={"Watch One": "WONE.MI", "Watch Two": "WTWO.DE"},
        )
        assert available
        assert [(r["ticker"], r["presence"]) for r in rows] == [
            ("MID", "Portfolio"), ("ZED", "Portfolio"),
            ("WONE", "Watchlist"), ("WTWO", "Watchlist"),
            ("ABLE", "Hist. Portfolio only"),
        ]

    def test_a_held_benchmark_is_one_row_and_stays_portfolio(self, monkeypatch):
        # Same instrument in both sources, met on ISIN. The Watchlist table drops
        # a benchmark the portfolio owns; this must agree with it.
        _, rows = _rows(
            monkeypatch,
            [_record("SGLD.MI", "XX0000000001", "Current + Historical",
                     intraday="SGLD.MI")],
            [("Gold ETC", "SGLD", "XX0000000001")],
            resolved={"Gold ETC": "SGLD.MI"},
        )
        assert [(r["ticker"], r["presence"]) for r in rows] == [
            ("SGLD", "Portfolio")]

    def test_a_watchlisted_rebalance_target_is_upgraded_to_watchlist(self, monkeypatch):
        # A seeded target carries a symbol but no ISIN, so the two sources meet
        # on the bare ticker; "Rebalance target" is not a band, Watchlist is.
        _, rows = _rows(
            monkeypatch,
            [_record("X25E.MI", "", "Rebalance target")],
            [("Euro Govt 25+", "X25E", "XX0000000009")],
            resolved={"Euro Govt 25+": "X25E.MI"},
        )
        assert [(r["ticker"], r["presence"], r["isin"]) for r in rows] == [
            ("X25E", "Watchlist", "XX0000000009")]

    def test_an_unwatched_rebalance_target_keeps_its_label_and_sorts_last(self, monkeypatch):
        _, rows = _rows(
            monkeypatch,
            [_record("AAA.MI", "XX0000000001", "Rebalance target"),
             _record("ZZZ.MI", "XX0000000002", "Historical only")])
        assert [(r["ticker"], r["presence"]) for r in rows] == [
            ("ZZZ", "Hist. Portfolio only"), ("AAA", "Rebalance target")]


class TestFeeds:

    def test_a_feedless_holding_keeps_its_row_with_empty_rics(self, monkeypatch):
        # A BTP has no market feed at all; the performance frame drops it, so the
        # audit is the only place it is visible. It must not vanish.
        _, rows = _rows(
            monkeypatch,
            [_record("", "IT0005542359", "Historical only", name="BTP-30OT31 4%")])
        assert [(r["ticker"], r["name"], r["hist_ric"], r["intr_ric"]) for r in rows] == [
            ("", "BTP-30OT31 4%", "", "")]

    def test_a_watchlist_intraday_ric_comes_from_the_quote_catalog(self, monkeypatch):
        # A sibling-venue intraday fallback is recorded by preprocessing, not by
        # the request: the row must print the venue that produced the series.
        _, rows = _rows(
            monkeypatch, [],
            [("World Efficient Core", "NTSG", "XX0000000001")],
            resolved={"World Efficient Core": "NTSG.MI"},
            quotes={"NTSG.MI": {"intraday_source_ticker": "NTSG.DE"}},
        )
        assert [(r["hist_ric"], r["intr_ric"]) for r in rows] == [("NTSG.MI", "NTSG.DE")]

    def test_no_instruments_means_the_section_is_hidden(self, monkeypatch):
        available, rows = _rows(monkeypatch, [])
        assert (available, rows) == (False, ())


def test_the_template_reads_exactly_the_keys_the_builder_emits(monkeypatch):
    """The section renders from the context dict, so a renamed key fails silently
    (Jinja prints "—" for an undefined attribute)."""
    import re
    from pathlib import Path

    _, rows = _rows(monkeypatch,
                    [_record("CL2.MI", "XX0000000001", "Current + Historical")])
    template = Path("tarzan/export/templates/portfolio_digest.html.j2").read_text()
    block = template[template.index("INSTRUMENT DATA SOURCES"):]
    block = block[:block.index("</table>")]
    used = set(re.findall(r"row\.([a-z_]+)", block))
    assert used, "the section stopped reading any row field"
    assert used <= set(rows[0]), f"template reads unknown row keys: {used - set(rows[0])}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
