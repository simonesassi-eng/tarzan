"""Direct structural check on input/instrument_taxonomy.csv.

A malformed row (one extra or missing delimiter) makes pandas raise a
ParserError inside ``_load_indexes_csv``, which is caught and turned into a
silent empty DataFrame plus a data-quality WARNING — intentional, since the
pipeline must not crash on a bad taxonomy row (see tarzan/config/__init__.py).

That degradation is the right call at runtime, but it means a malformed row
previously only surfaced as a handful of unrelated-looking failures in other
suites (ISIN resolution, referential integrity, order seeding) rather than as
a direct, actionable signal — which is what made the AGGH row slow to
diagnose. These tests check the CSV's structure directly, so a malformed row
fails here first, with the row number and field count.
"""

from __future__ import annotations

import csv
from pathlib import Path

from tarzan import config as cfg

_CSV_PATH = Path(__file__).resolve().parents[2] / "input" / "instrument_taxonomy.csv"


def _rows():
    with _CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_every_row_matches_header_field_count():
    rows = _rows()
    header = rows[0]
    bad = [
        (line_no, len(row))
        for line_no, row in enumerate(rows[1:], start=2)
        if len(row) != len(header)
    ]
    assert not bad, (
        f"instrument_taxonomy.csv has rows with the wrong field count "
        f"(expected {len(header)}), (line, field_count): {bad}"
    )


def test_required_columns_present():
    header = _rows()[0]
    required = {"name", "ticker", "isin", "kind", "asset_class"}
    missing = required.difference(c.strip().lower() for c in header)
    assert not missing, f"instrument_taxonomy.csv is missing columns: {missing}"


def test_loader_resolves_the_real_file():
    """End-to-end: the real loader, against the real file, must not silently
    degrade to empty. This also exercises the path-resolution logic that
    _load_indexes_csv uses to locate the CSV from an arbitrary cwd."""
    cfg._load_indexes_csv.cache_clear()
    try:
        df = cfg._load_indexes_csv()
    finally:
        cfg._load_indexes_csv.cache_clear()
    assert not df.empty, (
        "_load_indexes_csv() returned an empty DataFrame against the real "
        "instrument_taxonomy.csv — check for a parser error or a "
        "path-resolution regression"
    )
    # Floor, not an exact count: the curated list has ~60 rows today and
    # will grow. This only guards against "empty" or "drastically truncated".
    assert len(df) >= 50
