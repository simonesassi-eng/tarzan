"""Block E: the only part allowed to touch the network.

Everything else in the bench is internal consistency. This compares Tarzan's own
1D/5D/1M against the venue's tape, computed independently here — 1D from the
published quote pair, 5D and 1M from closes N SESSIONS back, never N calendar
days. A window that anchors on a calendar day fails here and cannot be explained
away as a rounding difference.

    STRESS_ALLOW_NETWORK=1 python -m tarzan.stress.external
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

SAMPLES = ["RSSB", "RSSY", "CTAP", "WTIP", "XDEQ.MI"]


def tape_expectations(tickers: list) -> dict:
    """(1D, 5D, 1M) per ticker from the real tape, in the instrument's OWN units."""
    import pandas as pd
    import yfinance as yf

    from tarzan.data.market_quotes import official_quotes

    quotes = official_quotes(tickers)
    out = {}
    for tk in tickers:
        q = quotes.get(tk) or {}
        price, prev = q.get("price"), q.get("prev_close")
        try:
            hist = yf.Ticker(tk).history(period="3mo", auto_adjust=False)["Close"].dropna()
        except Exception as exc:                            # noqa: BLE001
            out[tk] = {"error": str(exc)[:120]}
            continue
        hist.index = pd.DatetimeIndex(hist.index).tz_localize(None).normalize()
        closes = hist[~hist.index.duplicated(keep="last")]
        d1 = ((float(price) / float(prev) - 1) * 100) if price and prev else None

        def back(n):
            """Return over ``n`` SESSIONS ending at the live quote.

            ``closes.iloc[-1]`` is the last close and the quote sits on that same
            session (on a Saturday it IS that close), so the anchor n sessions
            earlier is ``iloc[-1 - n]``. Indexing ``iloc[-n]`` spans n-1 sessions
            and is why this oracle first reported RSSB's 5D as +0.48 against a
            rendered +0.19: an off-by-one in the harness, matching the product's
            own 1D exactly while disagreeing on every longer window.
            """
            if len(closes) <= n + 1 or not price:
                return None
            return (float(price) / float(closes.iloc[-1 - n]) - 1) * 100

        def months_back(n):
            """Return since the last close at or before ``n`` CALENDAR months ago.

            ``PERIOD_WINDOWS`` declares ``"1m": ("months", 1)`` — a calendar month,
            not 21 sessions — and the two differ by whole percentage points
            (CTAP: +8.68 vs +5.87). Implemented here from the stated semantics
            rather than by calling ``window_anchor``, so this checks the product
            against its specification instead of against itself.
            """
            if not price or closes.empty:
                return None
            anchor = closes.index[-1] - pd.DateOffset(months=n)
            prior = closes[closes.index <= anchor]
            if prior.empty:
                return None
            return (float(price) / float(prior.iloc[-1]) - 1) * 100

        out[tk] = {"1d": d1, "5d": back(5), "1m": months_back(1),
                   "quote": price, "prev_close": prev,
                   "last_close_date": str(closes.index[-1].date())}
    return out


def main() -> int:
    if os.environ.get("STRESS_ALLOW_NETWORK") != "1":
        print("refusing: set STRESS_ALLOW_NETWORK=1 to allow the one networked block")
        return 2
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    logging.disable(logging.WARNING)
    from tarzan.stress import checks, driver, net

    net.install_socket_guard()
    net.allow_network(True)
    exp = tape_expectations(SAMPLES)
    print("tape expectations (native units):")
    for tk, v in exp.items():
        print(f"  {tk:9s} {json.dumps(v, default=str)}")

    root = Path(tempfile.mkdtemp(prefix="stress_ext_"))
    cache = driver.prepare_cache(Path(os.environ.get(
        "STRESS_CACHE_SNAPSHOT", Path.home() / ".cache" / "tarzan")), root / "cache")
    # No pinned instant: both sides of the comparison have to be "now". Pinning C4
    # moved the run to another date while the expectations came from today's tape,
    # which is half of why the first networked block reported four false mismatches.
    res = driver.run_one(portfolio="P02", instant=None, mode="LIVE", strict=False,
                        outroot=root / "out", cache_dir=cache, run_id="E01",
                        allow_network=True)
    samples = [(tk, v.get("1d"), v.get("5d"), v.get("1m"))
               for tk, v in exp.items() if "error" not in v]
    ledger = Path(__file__).parent / "ledger.jsonl"
    with ledger.open("a") as fh:
        fh.write(json.dumps({"kind": "run", "cell": "E01", "mode": "LIVE",
                             "network_attempts": len(res.network_attempts),
                             "exit_code": res.exit_code}, default=str) + "\n")
        for v in checks.e9_windows_against_the_tape(res, samples):
            print(" ", v.line())
            fh.write(json.dumps({"kind": "check", "cell": "E01", "check": v.check,
                                 "type": v.kind,
                                 "verdict": "SKIP" if v.passed is None else
                                            ("PASS" if v.passed else "FAIL"),
                                 "detail": v.detail}, default=str) + "\n")
    print(f"\nnetwork calls used: {len(res.network_attempts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
