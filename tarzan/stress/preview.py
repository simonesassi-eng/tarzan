"""Render one newsletter per synthetic book, for looking at with your eyes.

The matrix asserts; this exists to be READ. Thirteen issues — the ten seeded books
at a live instant plus three re-run point-in-time for contrast — land in
``output/stress_previews/`` with an index, so a defect that no oracle thought to
assert still has a chance of being noticed by someone scrolling.

    python -m tarzan.stress.preview

Same isolation as the matrix: a per-session cache copy, no network, no writes to
``input/``. It does write under ``output/`` — that directory is gitignored and this
is the one part of the bench meant for a human rather than for a check.
"""

from __future__ import annotations

import html
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "output" / "stress_previews"

#: (cell, book, instant, mode) — ten books live, then three point-in-time so the
#: same book can be compared across run modes.
PREVIEWS = [
    ("01-eu-only", "P01", "C4", "LIVE"),
    ("02-us-usd-fx", "P02", "C4", "LIVE"),
    ("03-eu-us-lse-gbp", "P03", "C4", "LIVE"),
    ("04-fixed-income-coupons", "P04", "C4", "LIVE"),
    ("05-single-stock-etc", "P05", "C4", "LIVE"),
    ("06-distributing-transfers", "P06", "C4", "LIVE"),
    ("07-liquidated-then-rebought", "P07", "C4", "LIVE"),
    ("08-single-order", "P08", "C4", "LIVE"),
    ("09-fully-liquidated", "P09", "C4", "LIVE"),
    ("10-pathological", "P10", "C4", "LIVE"),
    ("11-us-usd-fx-pit", "P02", "C1", "PIT"),
    ("12-fixed-income-pit", "P04", "C1", "PIT"),
    ("13-pathological-pit", "P10", "C1", "PIT"),
]

NOTES = {
    "P01": "EU-only accumulating ETFs, 3 weeks, 4 orders",
    "P02": "US-only USD, 2 years — native tape plus FX",
    "P03": "EU + US + LSE, 4 years, includes a GBp order",
    "P04": "fixed income, 6 years, 3 tax years, coupons and a per-100 BTP",
    "P05": "single stock plus commodity ETCs, 5 years",
    "P06": "distributing, dividends and transfers, 3 years",
    "P07": "liquidated to zero, then re-bought",
    "P08": "a single order",
    "P09": "entirely liquidated — zero holdings today",
    "P10": "pathological: off-taxonomy, unresolvable ISIN, fractional, weekend, same-day",
}


def main() -> int:
    sys.path.insert(0, str(REPO))
    from tarzan.stress import clock, driver, net

    session = Path(tempfile.mkdtemp(prefix="tarzan_preview_"))
    cache = driver.prepare_cache(Path.home() / ".cache" / "tarzan", session / "cache")
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    net.install_socket_guard()
    rows = []
    for name, book, instant, mode in PREVIEWS:
        res = driver.run_one(portfolio=book, instant=instant, mode=mode, strict=False,
                             outroot=session / "out", cache_dir=cache, run_id=name,
                             quotes=None)
        clock.uninstall()
        src = res.artifact("newsletter.html")
        if src is None:
            rows.append((name, book, mode, None, res.error or f"exit {res.exit_code}"))
            print(f"  {name:<28} NO NEWSLETTER  ({res.error or res.exit_code})")
            continue
        out = DEST / f"{name}.html"
        shutil.copyfile(src, out)
        # summary.json nests the figures under "metrics"; reading them off the top
        # level printed "None holdings" for every issue in the index.
        m = (res.summary or {}).get("metrics") or {}
        note = (f"{m.get('num_holdings')} holdings · "
                f"EUR {m.get('total_value_eur')} · {m.get('valuation_availability')}")
        rows.append((name, book, mode, out.name, note))
        print(f"  {name:<28} {out.name:<34} {note}")

    _write_index(rows)
    print(f"\n{len(rows)} previews under {DEST}")
    print(f"open {DEST / 'index.html'}")
    print(f"network attempts: {len(net.attempts())}")
    return 0


def _write_index(rows: list) -> None:
    li = []
    for name, book, mode, href, note in rows:
        label = f"{book} · {mode} — {NOTES.get(book, '')}"
        link = (f'<a href="{html.escape(href)}">{html.escape(name)}</a>'
                if href else f'<s>{html.escape(name)}</s>')
        li.append(f"<li>{link}<br><small>{html.escape(label)}</small>"
                  f"<br><code>{html.escape(str(note))}</code></li>")
    (DEST / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Tarzan — synthetic book previews</title>"
        "<style>body{font:14px/1.6 system-ui;max-width:56em;margin:3em auto;padding:0 1em}"
        "li{margin-bottom:1.1em}code{color:#666;font-size:12px}"
        "small{color:#444}</style>"
        "<h1>Synthetic book previews</h1>"
        "<p>Thirteen newsletters rendered from the ten seeded stress books. No real "
        "position, quantity or size appears in any of them.</p>"
        f"<ol>{''.join(li)}</ol>\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
