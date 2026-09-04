"""Compare the newsletter's own return figures against Yahoo, in ONE run.

Why this exists: the return columns have been argued about repeatedly from separate
measurements taken minutes apart, and a live tape moves between them — so a
disagreement could always be explained away as timing. Everything here comes from a
SINGLE orchestrator run and a single set of fetches, so a difference is a difference.

Two independent checks, both against something outside our own math:

1.  RENDER vs ENGINE. Every instrument row's printed figures are read back out of the
    generated HTML and compared with the engine's ``holding_performance`` columns for
    the same ticker. This is what catches a display reading a stale or wrong series
    while the computation is right (or the reverse).

2.  ENGINE vs YAHOO. For a sample of instruments and timeframes, the period return is
    recomputed from a raw ``yfinance`` daily pull with no Tarzan code in the path
    except the date arithmetic, and compared with the engine's figure. This is what
    catches our math being self-consistently wrong.

Usage:
    python3 scripts/verify_returns_vs_yahoo.py                 # 3 instruments, 3 windows
    python3 scripts/verify_returns_vs_yahoo.py --all           # every holding
    python3 scripts/verify_returns_vs_yahoo.py --seed 7        # a different sample
    python3 scripts/verify_returns_vs_yahoo.py --windows 1d,1m,1y

Exit code is 1 when any comparison exceeds its tolerance, so it can gate a release.

It hits the network and reads the real book, so it is a script rather than a unit
test: nothing here belongs in CI, where Yahoo's availability would decide whether the
suite is green.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ.setdefault("TARZAN_DISABLE_AI", "1")

import pandas as pd  # noqa: E402

#: engine vs yahoo — both compute from the same closes; the slack is for the last bar
#: moving between two fetches inside one run.
TOL_YAHOO_PP = 0.05

#: render vs engine needs NO tolerance. A fixed one was wrong in both directions: the
#: returns grid tapers precision by magnitude (``_pct_compact``: 2 decimals under 100%,
#: 1 decimal under 1000%), so 0.005pp flagged five rows whose only sin was printing
#: +130.4% for 130.4443 — and a tolerance loose enough for those would hide a real
#: 0.04pp error on a small figure. So the check formats the engine's own number through
#: the render's own formatter and compares the STRINGS. Then a disagreement is a
#: disagreement at every magnitude, and rounding is not one.

WINDOWS = ("1d", "5d", "1m", "3m", "ytd", "1y")


def _pct(a, b):
    if a is None or b in (None, 0):
        return None
    return (float(a) / float(b) - 1.0) * 100.0


def _yahoo_period_return(closes: pd.Series, bucket: str, quote: dict | None = None):
    """The period return a reader would compute off Yahoo's own data.

    Deliberately NOT ``stats.compute_period_return``: this is the outside opinion, so
    it reimplements the window edge rather than sharing our code. Returns
    ``(pct, anchor, verdict)`` where ``verdict`` is "ok" or a reason the comparison
    cannot decide anything — an oracle that cannot tell "we are wrong" from "I cannot
    judge this" is worse than no oracle, because every one of its cries is a wolf.

    Two rules were naive in the first draft and produced four false findings:

    * **1D** is not "the previous row of the daily frame". It is the pair Yahoo's own
      page prints in its big number, ``regularMarketPrice`` over
      ``regularMarketPreviousClose`` — and those disagree with row[-2] exactly when a
      session is missing from the frame, which is when the check mattered. Measured:
      three sampled instruments all "failed" 1D by 0.63-2.09pp against row[-2] while
      the frame was missing 3 Sep.
    * **5D** is five SESSIONS back, and counting five rows only equals that when the
      frame has every session in the span. With a hole, five rows reach six sessions
      and the disagreement is the oracle's, not the engine's. So the span is checked
      for missing business days first and the verdict says so.

    The long windows need neither correction: the last close at or before the cutoff
    is unambiguous whatever is missing in between, which is why 1m/3m/ytd/1y matched
    to the last decimal on the instruments whose listing is current.
    """
    s = closes.dropna()
    if len(s) < 2:
        return None, None, "fewer than two closes"
    end = s.index[-1]
    if bucket == "1d":
        price = (quote or {}).get("price")
        prev = (quote or {}).get("prev_close")
        if not (price and prev):
            return None, None, "no published quote pair"
        return _pct(price, prev), "published pair", "ok"
    if bucket == "5d":
        if len(s) <= 5:
            return None, None, "fewer than six closes"
        anchor = s.index[-6]
        span = pd.bdate_range(anchor, end)
        missing = [d for d in span if d.normalize() not in
                   set(pd.DatetimeIndex(s.index).normalize())]
        if missing:
            return None, str(pd.Timestamp(anchor).date()), (
                f"frame missing {len(missing)} business day(s) in the span "
                f"({', '.join(str(d.date()) for d in missing[:3])})")
        return _pct(s.iloc[-1], s.loc[anchor]), str(pd.Timestamp(anchor).date()), "ok"
    if bucket == "ytd":
        prior = s[s.index.year < end.year]
        if prior.empty:
            return None, None, "no prior-year close"
        anchor = prior.index[-1]
    else:
        n, unit = int(bucket[:-1]), bucket[-1]
        if unit == "m":
            cutoff = end - pd.DateOffset(months=n)
        elif unit == "y":
            cutoff = end - pd.DateOffset(years=n)
        else:
            return None, None, f"unsupported window {bucket}"
        prior = s[s.index <= cutoff]
        if prior.empty:
            return None, None, "history does not reach back that far"
        anchor = prior.index[-1]
    return (_pct(s.iloc[-1], s.loc[anchor]),
            str(pd.Timestamp(anchor).date()), "ok")


#: Header label -> the engine's column name. The table's own header is the authority
#: on which column is which, so the mapping only has to translate the labels.
_HEADER_TO_KEY = {"intraday": "1d", "1d": "1d", "5d": "5d", "1m": "1m", "3m": "3m",
                  "6m": "6m", "ytd": "ytd", "1y": "1y", "1a": "1y",
                  "3y": "3y", "5y": "5y"}


def _header_keys(html: str) -> list:
    """The RETURNS table's period columns, in the order it prints them."""
    if "[06]" not in html:
        return []
    sec = html.split("[06]", 1)[1].split("[07]", 1)[0]
    for tr in re.findall(r"<tr>.*?</tr>", sec, re.S):
        cells = [re.sub(r"<[^>]+>", "", c).strip().lower()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if cells and cells[0].startswith("instrument"):
            return [_HEADER_TO_KEY.get(c, c) for c in cells[1:]]
    return []


def _rendered_rows(html: str) -> dict:
    """``{displayed ticker: [figures]}`` from the RETURNS section's own markup."""
    if "[06]" not in html:
        return {}
    sec = html.split("[06]", 1)[1].split("[07]", 1)[0]
    out = {}
    for tr in re.findall(r"<tr>.*?</tr>", sec, re.S):
        cells = [re.sub(r"<[^>]+>", "", c).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        cells = [c for c in cells if c]
        if len(cells) < 3:
            continue
        name = cells[0]
        if name.lower().startswith("instrument"):
            continue
        # Keep every cell in position. Filtering to the ones that look like a
        # percentage drops an em-dash and shifts every column after it, which is a
        # second way to compare the wrong pair.
        out[name] = cells[1:]
    return out


def _displayed(fig: str):
    try:
        return float(fig.replace("−", "-").replace("%", "").replace("+", ""))
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="check every holding instead of a sample of three")
    ap.add_argument("--seed", type=int, default=None,
                    help="sample seed; omit for a fresh random sample each run")
    ap.add_argument("--windows", default="",
                    help="comma-separated subset of " + ",".join(WINDOWS))
    args = ap.parse_args()
    windows = ([w.strip() for w in args.windows.split(",") if w.strip()]
               or list(WINDOWS))

    import tarzan.runtime as rt
    rt.allows_live_transport = lambda: True

    from tarzan import config as cfg
    from tarzan import orchestrator
    global _pct_compact
    from tarzan.export.newsletter._format import _pct_compact
    from tarzan.data.enricher import _fetch_history
    from tarzan.export.newsletter import render_newsletter

    print("running the pipeline (one run; every figure below comes from it)...")
    metrics, config = orchestrator.run(
        config_source="input/targets.csv",
        orders_source="input/order_list.csv",
        targets_per_holding_source="input/targets_per_holding.csv")

    html = render_newsletter(
        metrics, config,
        benchmark_alpha_beta=cfg.benchmark_beta_name(),
        benchmark_geo=cfg.benchmark_geo_allocation())

    hp = getattr(metrics, "holding_performance", None)
    if hp is None or getattr(hp, "empty", True):
        print("no holding_performance: nothing to verify")
        return 1

    findings = []

    # Write the HTML this run produced. Reading a file from an earlier render is how
    # a staler tape gets mistaken for a display bug -- a 1M of -0.69% against the
    # engine's -1.61% turned out to be the same arithmetic one session behind.
    Path("output").mkdir(exist_ok=True)
    Path("output/verify_returns.html").write_text(html)

    # ── 1. render vs engine ──────────────────────────────────────────────────
    header_keys = _header_keys(html)
    print(f"    columns read from the table header: {header_keys}")
    rendered = _rendered_rows(html)
    print(f"\n[1] RENDER vs ENGINE  ({len(rendered)} rows read back from the HTML)")
    # Keyed by BARE ticker, but a bare ticker is not unique: an instrument that is
    # both held and tracked appears twice, once per listing (NTSG.MI as the holding,
    # NTSG.DE as the benchmark). Keeping whichever came last made this compare the
    # render's holding row against the engine's benchmark row and report five
    # differences that were the two listings disagreeing, not the render.
    #
    # The RETURNS table prints HOLDINGS, so the holdings' own tickers win.
    held = {str(t) for t in (metrics.holdings_df["ticker"]
                             if getattr(metrics, "holdings_df", None) is not None
                             and "ticker" in metrics.holdings_df else [])}
    engine_by_bare = {}
    ambiguous = {}
    for _i, row in hp.iterrows():
        tk = str(row.get("ticker") or "")
        if not tk:
            continue
        bare = tk.split(".")[0].upper()
        ambiguous.setdefault(bare, []).append(tk)
        if bare in engine_by_bare and tk not in held:
            continue                      # keep the held listing, not the tracked one
        engine_by_bare[bare] = row
    dupes = {b: v for b, v in ambiguous.items() if len(v) > 1}
    if dupes:
        print(f"    two listings for: {dupes} (the held one is compared)")

    # The row's first cell is the ticker glued to the abbreviated name
    # ("CL2Amu. MSCI USA (2x) Lev. [EUR]"), so a greedy [A-Z0-9]+ swallowed the
    # name's first letters and matched nothing. Resolve by LONGEST prefix against
    # the engine's own ticker set instead of guessing where the boundary is.
    def _ticker_of(display: str):
        up = display.upper()
        cands = [b for b in engine_by_bare if up.startswith(b)]
        return max(cands, key=len) if cands else None

    checked = 0
    unmatched = []
    for name, figs in rendered.items():
        bare_key = _ticker_of(name)
        if bare_key is None:
            unmatched.append(name[:28])
            continue
        row = engine_by_bare.get(bare_key)
        if row is None:
            continue
        # Column order in the row, read from the table's OWN header rather than
        # hardcoded: the first draft omitted 3Y, so every row compared its printed
        # 3Y against the engine's 5Y and reported twelve findings that were mine.
        order = header_keys
        for fig, key in zip(figs, order):
            if key not in hp.columns:
                continue
            eng = row.get(key)
            if eng is None or (isinstance(eng, float) and eng != eng):
                continue
            if _displayed(fig) is None:      # an em-dash cell: nothing was claimed
                continue
            expected = _pct_compact(float(eng))
            checked += 1
            if fig.strip() != expected:
                findings.append(
                    f"RENDER {bare_key} {key}: printed {fig!r} but the engine's "
                    f"{float(eng):+.4f}% formats as {expected!r}")
    print(f"    {checked} figures compared, "
          f"{len([f for f in findings if f.startswith('RENDER')])} disagreeing")
    if unmatched:
        print(f"    unmatched rows (not checked): {unmatched}")

    # ── 2. engine vs yahoo ──────────────────────────────────────────────────
    tickers = sorted({str(r.get("ticker") or "") for _i, r in hp.iterrows()
                      if str(r.get("ticker") or "")})
    holdings = sorted({str(t) for t in (metrics.holdings_df["ticker"]
                                        if getattr(metrics, "holdings_df", None)
                                        is not None
                                        and "ticker" in metrics.holdings_df else [])})
    pool = [t for t in tickers if t in holdings] or tickers
    if args.all:
        sample = pool
    else:
        rnd = random.Random(args.seed)
        sample = rnd.sample(pool, min(3, len(pool)))

    print(f"\n[2] ENGINE vs YAHOO  ({len(sample)} instruments x {len(windows)} windows)")
    print(f"    sample: {', '.join(sample)}")
    from tarzan.data.market_quotes import _fetch_official_quotes
    inconclusive = []
    # How current is a source frame? Not "days from today" — a Monday run is three days
    # from Friday's close and perfectly current. What matters is whether THIS listing's
    # frame is behind the others in the same run: our tape is patched to the published
    # price, so a frame that stopped earlier than its peers cannot referee it, and
    # every window would "disagree" by whatever happened after it went quiet.
    frames = {}
    for tk in sample:
        f = _fetch_history(tk)
        cl = (f["Close"].dropna() if f is not None and "Close" in f else None)
        if cl is not None and not cl.empty:
            frames[tk] = cl
    if not frames:
        print("    no source frames returned")
        return 1
    # Compare DATES, not timestamps: a ``.PA`` frame is stamped Europe/Paris and a
    # ``.MI`` one Europe/Rome, and ``bdate_range`` refuses two tz-aware endpoints in
    # different zones. A session date has no zone to disagree about.
    newest = max(c.index[-1].date() for c in frames.values())
    print(f"    newest source close in this run: {newest}")

    for tk in sample:
        closes = frames.get(tk)
        if closes is None or closes.dropna().empty:
            findings.append(f"YAHOO {tk}: no closes returned")
            continue
        quote = _fetch_official_quotes([tk]).get(tk) or {}
        row = next((r for _i, r in hp.iterrows()
                    if str(r.get("ticker") or "") == tk), None)
        last_day = closes.dropna().index[-1].date()
        behind = max(0, len(pd.bdate_range(last_day, newest)) - 1)
        note = "" if behind <= 0 else f"   [frame {behind} session(s) behind its peers]"
        # A source whose own frame has stopped cannot referee our figure. Our tape is
        # patched to today from the published pair (and from a sibling venue when the
        # canonical listing is dormant), so every window would "disagree" by whatever
        # happened after the source went quiet. Measured on the dormant listing: five
        # windows off by 0.37-1.56pp, all of it the source being days behind.
        source_stale = behind > 0
        print(f"\n    {tk}   last close {last_day} "
              f"= {float(closes.dropna().iloc[-1]):.4f}{note}")
        print(f"      {'window':<6}{'engine':>11}{'yahoo':>11}{'gap':>9}  anchor / why")
        for w in windows:
            theirs, anchor, verdict = _yahoo_period_return(closes, w, quote)
            ours = None if row is None else row.get(w)
            if ours is not None and isinstance(ours, float) and ours != ours:
                ours = None
            gap = (None if None in (ours, theirs) else float(ours) - theirs)
            tail = anchor or "—" if verdict == "ok" else f"INDECIDIBILE: {verdict}"
            print(f"      {w:<6}"
                  f"{('—' if ours is None else f'{float(ours):+.4f}%'):>11}"
                  f"{('—' if theirs is None else f'{theirs:+.4f}%'):>11}"
                  f"{('—' if gap is None else f'{gap:+.4f}'):>9}  {tail}")
            if source_stale:
                inconclusive.append(
                    f"{tk} {w}: the source's frame is {behind} session(s) behind its "
                    f"peers (stops {last_day}), so it cannot referee a tape patched "
                    f"to the published price")
                continue
            if verdict != "ok":
                inconclusive.append(f"{tk} {w}: {verdict}")
                continue
            if gap is not None and abs(gap) > TOL_YAHOO_PP:
                findings.append(
                    f"YAHOO {tk} {w}: engine {float(ours):+.4f}% vs Yahoo "
                    f"{theirs:+.4f}% ({gap:+.4f}pp, anchor {anchor})")

    print("\n" + "=" * 70)
    if inconclusive:
        print(f"{len(inconclusive)} comparison(s) the oracle cannot decide "
              f"(the SOURCE's own frame is incomplete, not our figure):")
        for i in inconclusive:
            print(f"  ? {i}")
        print()
    if findings:
        print(f"{len(findings)} FINDING(S)")
        for f in findings:
            print(f"  - {f}")
    else:
        print("no disagreement beyond tolerance")
    Path("output").mkdir(exist_ok=True)
    Path("output/verify_returns.json").write_text(json.dumps(
        {"now": str(pd.Timestamp.now(tz="Europe/Rome")),
         "sample": sample, "windows": windows,
         "findings": findings, "inconclusive": inconclusive}, indent=1))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
