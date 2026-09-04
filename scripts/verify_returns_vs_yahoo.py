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

Exit code is 1 when any comparison exceeds its tolerance, so it IS the assertion.

It hits the network and reads the real book, so it is a script rather than a unit
test: in the suite, Yahoo's availability would decide whether the tests are green.
It runs instead as its own scheduled workflow
(``.github/workflows/verify-returns.yml``) — NOT as a step in the newsletter, whose
render is already throttle-bound, and which a second full pipeline run per issue
would slow for every send to catch a rare fault.

This repo is PUBLIC and a run's log is public with it, so under ``$CI`` the script
redacts: counts and findings print, the per-instrument table does not. A return is
public information; the list of instruments someone holds is not, and printing the
whole book daily would republish what the repo was scrubbed to remove.
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


def source_can_referee(source_last, stamped_end, bucket: str) -> bool:
    """Can a source frame ending on ``source_last`` judge our ``bucket`` figure,
    given that the tape it is judging was stamped to ``stamped_end``?

    Extracted and tested because this one rule decides whether a difference is
    reported or waved through, and it has been wrong twice in opposite directions:

    * **Relative to the sample.** "Is this frame behind the OTHERS in this run?" reads
      a uniformly stale sample as current. Measured on a Saturday: three Milan frames
      all stopped on the same earlier session, all agreed with each other, nothing
      abstained, and nine endpoint mismatches of 0.10-4.25pp were reported as findings.
    * **The venue's last session.** Absolute, but the wrong absolute: before the open,
      the last session is TODAY and nothing has traded, so every frame looks a session
      behind and every window abstains — a check that runs daily and decides nothing.

    The tape's own stamped end is the reference, because that is the endpoint our figure
    was measured to. Equal dates: both sides measured to the same close, so a difference
    is real. Frame behind: different endpoints, and the gap is the sessions in between.

    1D is exempt. It is compared against the published quote pair, which is current
    whatever the daily frame is missing — the very reason it reads the pair and not the
    frame's second-to-last row.
    """
    if bucket == "1d":
        return True
    return not source_last < stamped_end


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
    # On by default in CI because this repo is PUBLIC and a run's log is public with
    # it. Printing every held ticker and its returns daily publishes the book's
    # composition, which is the thing the repo was scrubbed to remove — a return is
    # public information, the list of instruments someone holds is not. Redacted mode
    # keeps the counts and the findings (a finding has to name its instrument to be
    # worth anything) and drops the per-instrument table.
    ap.add_argument("--redact", action="store_true",
                    default=bool(os.environ.get("CI")),
                    help="suppress the per-instrument table (default: on under CI)")
    ap.add_argument("--no-redact", dest="redact", action="store_false",
                    help="print the full table even under CI")
    args = ap.parse_args()
    os.chdir(REPO)
    os.environ.setdefault("TARZAN_DISABLE_AI", "1")
    windows = ([w.strip() for w in args.windows.split(",") if w.strip()]
               or list(WINDOWS))

    def say(line: str) -> None:
        """Print unless redacted. Every line that NAMES an instrument goes
        through here; counts and findings print unconditionally."""
        if not args.redact:
            print(line)

    if args.redact:
        print("redacted mode: per-instrument detail is suppressed "
              "(public repo, public log)")

    import tarzan.runtime as rt
    rt.allows_live_transport = lambda: True

    from tarzan import config as cfg
    from tarzan import orchestrator
    global _pct_compact
    from tarzan.export.newsletter._format import _pct_compact
    from tarzan.data.enricher import _fetch_history
    from tarzan.export.newsletter import render_newsletter

    # The SAME input resolution the send uses, not a hardcoded path: in CI the book
    # is not in the repo at all (only the taxonomy is tracked) and comes from the
    # private Drive folder, so hardcoding ``input/`` verified nothing there. Sharing
    # ``resolve_inputs`` also means this checks the book the newsletter actually
    # renders rather than a second, possibly staler copy of it.
    from tarzan.delivery import _seed_manual_proxies, resolve_inputs

    # Its local defaults are ``.private/``; this tree keeps the book in ``input/``.
    # Fill in the documented overrides only when unset and the file is there, so
    # Drive mode and an explicit override both still win.
    if not (os.environ.get("DRIVE_FOLDER_ID")
            and os.environ.get("GOOGLE_DRIVE_CREDENTIALS_JSON")):
        for var, path in (("ORDERS_PATH", "input/order_list.csv"),
                          ("TARGETS_PATH", "input/targets.csv"),
                          ("TARGETS_PER_HOLDING_PATH",
                           "input/targets_per_holding.csv")):
            if not os.environ.get(var) and Path(path).exists():
                os.environ[var] = path
    inputs = resolve_inputs()
    say(f"    inputs: orders={inputs['orders']}")

    print("running the pipeline (one run; every figure below comes from it)...")
    metrics, config = orchestrator.run(
        config_source=inputs["config"],
        orders_source=inputs["orders"],
        targets_per_holding_source=inputs["targets_per_holding"])

    # Mirrors the send: the carry/CTA sleeves need their manual index levels before
    # render. Never fatal here — a missing proxy is not a wrong return figure, and
    # failing the oracle over one would teach us to ignore it.
    try:
        _seed_manual_proxies()
    except Exception as exc:                                        # noqa: BLE001
        print(f"    (manual proxies unavailable: {exc})")

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
        say(f"    two listings for: {dupes} (the held one is compared)")

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
        say(f"    unmatched rows (not checked): {unmatched}")

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
    say(f"    sample: {', '.join(sample)}")
    from tarzan.data.current_session import pick_quote, stamp_date
    from tarzan.data.market_quotes import _sibling_symbols, official_quotes
    inconclusive = []
    # How current is a source frame? Not "days from today" — a Monday run is three days
    # from Friday's close and perfectly current. And NOT "behind its peers in this run"
    # either, which is what this asked first and got wrong: that is a RELATIVE test, so
    # a sample whose frames are all equally stale reads as all current. Measured on a
    # Saturday run: Yahoo's daily frames for three Milan listings all stopped on Wed
    # 2 Sep while the last session was Fri 4 Sep, every frame agreed with every other,
    # nothing was flagged inconclusive, and all nine comparisons were reported as
    # findings — 0.10 to 4.25pp of pure endpoint mismatch. A direct yfinance pull
    # confirmed the frames, so this was the oracle, not the cache and not the engine.
    #
    # The reference is the date the engine's tape actually ENDS on, which is neither
    # the frame's own end nor the calendar's last session. The calendar is absolute but
    # answers the wrong question: at 08:30, before Milan opens, "the venue's last
    # session" is today while nothing has traded, so every frame would look a session
    # behind and every window would abstain every day -- the check would run daily and
    # decide nothing.
    #
    # ``current_session.stamp_date`` is the production function that decides which
    # session a published quote belongs to, from the observation on the venue's own
    # clock. Handed the same quote, it returns the very date the engine stamped. So:
    # equal dates means both sides measured to the same close and the comparison is
    # real; a frame behind it means the engine measured to a later point and the source
    # cannot referee. Pre-open that date IS yesterday, so the morning run decides.
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
    # The run's own clock, not the system's: a pinned run must judge staleness against
    # the date it is pretending to be, or it decides differently on every replay.
    today = pd.Timestamp(rt.today()).date()
    print(f"    newest source close in this run: {newest}  (run date {today})")

    for tk in sample:
        closes = frames.get(tk)
        if closes is None or closes.dropna().empty:
            findings.append(f"YAHOO {tk}: no closes returned")
            continue
        last_day = closes.dropna().index[-1].date()
        reference = float(closes.dropna().iloc[-1])
        # Resolve the quote the way PRODUCTION does, not by asking the canonical symbol.
        # ``pick_quote`` takes the first candidate whose level agrees with the
        # instrument's own last close, which is what rejects a corrupt feed: this
        # book holds a listing whose quote endpoint returns 25.515 against its own
        # 28.82 frame — 11.5% out, a split reflected in one feed and not the other —
        # and the engine prices it from the sibling venue instead.
        #
        # Asking the canonical symbol made the oracle quote the very feed production
        # refuses, and produced six findings on that one instrument: its "Yahoo" 1D was
        # 25.515/25.805 = -1.12% while the engine's coherent pair gave -0.76%. The
        # engine was right and the oracle was reading garbage.
        candidates = [tk, *_sibling_symbols(tk)]
        cand_quotes = official_quotes(candidates)
        quote = pick_quote(candidates, cand_quotes, reference)
        source_symbol = next(
            (c for c in candidates
             if (cand_quotes.get(c) or {}).get("price")
             and quote and (cand_quotes.get(c) or {}).get("price") == quote.get("price")),
            tk)
        row = next((r for _i, r in hp.iterrows()
                    if str(r.get("ticker") or "") == tk), None)
        # None when the quote belongs on no session, which means the engine stamped
        # nothing and its tape ends where this frame does — comparable.
        stamped = stamp_date(quote, today, tk)
        engine_end = stamped.date() if stamped is not None else last_day
        # When production had to leave the canonical venue, no single frame reproduces
        # our tape: it is this listing's own history with a SIBLING's close on the end.
        # Comparing it against either venue's frame is apples to oranges at the
        # endpoint (0.17% apart here), so the multi-day windows abstain and say why.
        # This is worth reading, not hiding: it means the canonical quote is unusable.
        cross_venue = source_symbol != tk
        source_stale = not source_can_referee(last_day, engine_end, "3m")
        note = "".join([
            "" if not source_stale else
            f"   [frame stops {last_day}; the tape is stamped to {engine_end}]",
            "" if not cross_venue else
            f"   [canonical quote rejected; priced from {source_symbol}]"])
        say(f"\n    {tk}   last close {last_day} "
              f"= {float(closes.dropna().iloc[-1]):.4f}{note}")
        say(f"      {'window':<6}{'engine':>11}{'yahoo':>11}{'gap':>9}  anchor / why")
        for w in windows:
            theirs, anchor, verdict = _yahoo_period_return(closes, w, quote)
            ours = None if row is None else row.get(w)
            if ours is not None and isinstance(ours, float) and ours != ours:
                ours = None
            gap = (None if None in (ours, theirs) else float(ours) - theirs)
            tail = anchor or "—" if verdict == "ok" else f"INDECIDIBILE: {verdict}"
            say(f"      {w:<6}"
                  f"{('—' if ours is None else f'{float(ours):+.4f}%'):>11}"
                  f"{('—' if theirs is None else f'{theirs:+.4f}%'):>11}"
                  f"{('—' if gap is None else f'{gap:+.4f}'):>9}  {tail}")
            if not source_can_referee(last_day, engine_end, w):
                inconclusive.append(
                    f"{tk} {w}: the source's frame stops {last_day} while the tape is "
                    f"stamped to {engine_end}, so the two measure to different "
                    f"closes")
                continue
            # 1D stays decidable: it is compared against the coherent published pair,
            # which is the very pair the engine used.
            if cross_venue and w != "1d":
                inconclusive.append(
                    f"{tk} {w}: its canonical quote failed the coherence gate and the "
                    f"price came from {source_symbol}, so our tape is this venue's "
                    f"history ending on another venue's close — no single frame "
                    f"reproduces it")
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
            say(f"  ? {i}")
        if args.redact:
            # The reasons are not private — dates and vendor gaps are not positions —
            # and without them a fully-abstaining run is indistinguishable from a
            # broken one. So the REASONS print, grouped, with the instruments stripped.
            reasons: dict[str, int] = {}
            for i in inconclusive:
                reasons[i.split(": ", 1)[-1]] = reasons.get(i.split(": ", 1)[-1], 0) + 1
            for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
                print(f"  ? x{n}: {reason}")
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
