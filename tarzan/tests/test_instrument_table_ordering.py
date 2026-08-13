"""Inside one asset class + role, instruments are listed alphabetically.

Every instrument table shares one grouping engine (group_by_class_role) but each
caller used to hand it a different within-role order: Holdings came out
value-descending, the Returns snapshot in holdings-file order, the Watchlist in
benchmark-config order. The same fund therefore sat in a different place in each
table. The order now comes from the engine, keyed on the ticker -- the first
thing every row prints.

Network-free: the engine takes plain dicts.
"""

from __future__ import annotations

import re

from tarzan.export.newsletter._constants import (
    _PERF_CLASS_ORDER,
    group_by_class_role,
)


def _rows(*specs):
    return [{"ac": ac, "role": role, "tk": tk} for ac, role, tk in specs]


def _grouped(rows, **kw):
    """(class, [(role, [ticker, ...]), ...]) — the shape without the colours."""
    return [(ac, [(role, [it["tk"] for it in items]) for role, items in roles])
            for ac, _col, roles in group_by_class_role(
                rows, asset_class=lambda r: r["ac"],
                role=lambda r: r["role"], **kw)]


class TestAlphabeticalWithinRole:

    def test_a_roles_items_are_sorted_by_ticker(self):
        rows = _rows(("Equities", "Efficient Core", "RSST"),
                     ("Equities", "Efficient Core", "CTAP"),
                     ("Equities", "Efficient Core", "NTSX"),
                     ("Equities", "Efficient Core", "RSSB"))
        assert _grouped(rows, ticker=lambda r: r["tk"]) == [
            ("Equities", [("Efficient Core", ["CTAP", "NTSX", "RSSB", "RSST"])]),
        ]

    def test_sorting_is_case_insensitive(self):
        rows = _rows(("Gold", "Gold", "sgld"), ("Gold", "Gold", "PPFB"),
                     ("Gold", "Gold", "GDE"))
        assert _grouped(rows, ticker=lambda r: r["tk"]) == [
            ("Gold", [("Gold", ["GDE", "PPFB", "sgld"])]),
        ]

    def test_a_missing_ticker_sorts_first_and_never_raises(self):
        rows = _rows(("Crypto", "Crypto", "BTC"), ("Crypto", "Crypto", None))
        assert _grouped(rows, ticker=lambda r: r["tk"]) == [
            ("Crypto", [("Crypto", [None, "BTC"])]),
        ]

    def test_class_and_role_order_stay_canonical(self):
        # Alphabetical applies WITHIN a role only: the class and role sequence is
        # still the curated risk-ordering, not A-Z.
        rows = _rows(("Fixed Income", "Long Duration", "ZROZ"),
                     ("Equities", "Equity Factor", "AVWS"),
                     ("Equities", "Equity Broad", "VWCE"),
                     ("Fixed Income", "Govt Nominal", "IBTA"))
        out = _grouped(rows, ticker=lambda r: r["tk"])
        assert [ac for ac, _ in out] == ["Equities", "Fixed Income"]
        assert [r for _, roles in out for r, _ in roles] == [
            "Equity Broad", "Equity Factor", "Govt Nominal", "Long Duration"]
        assert _PERF_CLASS_ORDER.index("Equities") < _PERF_CLASS_ORDER.index("Fixed Income")

    def test_without_a_ticker_accessor_the_callers_order_is_kept(self):
        # The Optimizer's action list is deliberately largest-trade-first.
        rows = _rows(("Equities", "Equity Broad", "VWCE"),
                     ("Equities", "Equity Broad", "AVWS"))
        assert _grouped(rows) == [
            ("Equities", [("Equity Broad", ["VWCE", "AVWS"])]),
        ]


class TestEveryInstrumentTableAsksForTheOrdering:
    """The engine can only sort what the call site tells it about.

    The Watchlist passed ``role=`` but no ``ticker=``, so it silently kept
    benchmark-config order. Assert every real call site passes it, or the next
    table added drifts the same way.
    """

    def test_all_call_sites_pass_a_ticker_accessor(self):
        import inspect

        from tarzan.export.newsletter import _sections_alloc, _sections_perf

        calls = 0
        for mod in (_sections_alloc, _sections_perf):
            src = inspect.getsource(mod)
            for m in re.finditer(r"group_by_class_role\(", src):
                calls += 1
                # The argument list ends at the call's own ")" -- which is at end
                # of line, or followed by ":" when the call is a for-header.
                args = re.split(r"\)[:\n]", src[m.end():m.end() + 400])[0]
                assert "ticker=" in args, (
                    f"{mod.__name__}: group_by_class_role call without "
                    f"ticker= -> its table will not be alphabetical\n{args[:200]}")
        # Holdings, Returns snapshot, Watchlist.
        assert calls >= 3, f"expected the 3 instrument tables, found {calls}"
