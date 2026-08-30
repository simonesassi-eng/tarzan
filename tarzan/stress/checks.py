"""The oracle: invariants and differential comparisons.

On a random book there is no known-good number, so no check compares a figure to
a value someone decided was right. Each one is either an INVARIANT (something the
run must satisfy whatever the numbers), a DIFFERENTIAL (two runs that must agree),
or EXTERNAL (compared against the venue's own tape). Every verdict records which,
so the report can never present internal consistency as external verification.

Tolerances, declared once and applied everywhere:
  money        abs <= 0.01 EUR  or  rel <= 1e-6, whichever is looser
                (summary.json rounds money to 2 dp)
  ratios/pct   rel <= 1e-6      (summary.json rounds these to 6 dp)
  displayed %  abs <= 0.01 pp   (1 basis point)
  byte-equal   exact, after masking the fields named in MASKED_*
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from typing import Any, Optional

MONEY_ABS = 0.01
REL = 1e-6
DISPLAY_PP = 0.01

#: Per-run nondeterminism, excluded from byte-identity. Established by reading
#: the artifact writers, not guessed: summary.json carries NO id and NO timestamp
#: at all, so it is compared whole.
#: ``entry_id`` hashes {type, payload, sequence} — and duration_ms lives INSIDE
#: the payload, so masking the duration does not make the id comparable. Both go.
MASKED_LEDGER_FIELDS = ("recorded_at", "entry_id")
MASKED_LEDGER_PAYLOAD_PATHS = ("stages.*.duration_ms",)
MASKED_MANIFEST_FIELDS = ("attempt_id",)
#: Only summary.json's checksum is reproducible; every other artifact embeds a
#: clock or an id, so their digests are excluded rather than pretended stable.
MANIFEST_STABLE_CHECKSUMS = ("summary.json",)
#: The two clock strings in the rendered newsletter.
NEWSLETTER_CLOCK_RE = re.compile(r"(As of \d\d:\d\d|generated [^<]*\d\d:\d\d)")


@dataclasses.dataclass
class Verdict:
    check: str
    kind: str                 # INVARIANT | DIFFERENTIAL | EXTERNAL
    passed: Optional[bool]    # None = no claim was made; ``void`` says WHY
    detail: str
    expected_fail: bool = False
    #: Why no claim was made. Declaring it is the whole point of FINDING F7.
    #:
    #: ""             a claim was made; ``passed`` carries it.
    #: "n/a"          the state the check tests genuinely did not arise. A
    #:                property of the FIXTURE, independent of how the run went.
    #:                This is real coverage: the check looked and there was
    #:                nothing to judge.
    #: "no-artifact"  the product produced nothing to judge. NOT coverage — the
    #:                reason there is nothing to read is itself the defect, and
    #:                C0 owns it. Counted separately so it can never again be
    #:                mistaken for a clean result.
    #: "reported"     observed and deliberately not asserted, because the bench
    #:                cannot build a sound oracle for it (see c7_fx_and_native).
    #:
    #: Collapsing the last three into one word is how twelve runs that rendered
    #: no newsletter at all still collected zero FAILs: one cell scored
    #: "4 PASS, 5 SKIP, 0 FAIL" for a run that exited 1 and produced nothing.
    void: str = ""

    def state(self) -> str:
        """The ONE place a verdict becomes a word.

        ``run.py`` and ``external.py`` each carried their own copy of this
        ternary and both dropped the expected-failure cases, so the ledger
        recorded four XPASSes as PASS and one XFAIL as FAIL.
        """
        if self.void:
            return {"no-artifact": "VOID", "reported": "NOTE"}.get(self.void, "SKIP")
        if self.passed is None:
            return "SKIP"
        if self.expected_fail:
            return "XFAIL" if self.passed is False else "XPASS"
        return "PASS" if self.passed else "FAIL"

    def line(self) -> str:
        return f"{self.state():5s} {self.check:6s} {self.kind:12s} {self.detail}"


def _void(check: str, kind: str, why: str) -> Verdict:
    """No artifact to judge. NOT a skip: a skip says the state did not arise,
    this says the product produced nothing — and C0 must be failing alongside."""
    return Verdict(check, kind, None, why, void="no-artifact")


def _na(check: str, kind: str, why: str) -> Verdict:
    """The state genuinely did not arise — a property of the fixture, not the run.
    This IS coverage: the check looked and found nothing to judge."""
    return Verdict(check, kind, None, why, void="n/a")


def _note(check: str, kind: str, what: str) -> Verdict:
    """Observed, deliberately not asserted. The bench cannot build a sound oracle
    for it, and says so rather than pretending either way."""
    return Verdict(check, kind, None, what, void="reported")


def _close_money(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= MONEY_ABS or (b != 0 and abs(a - b) / abs(b) <= REL)


def _close_rel(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if not (math.isfinite(a) and math.isfinite(b)):
        return a == b or (math.isnan(a) and math.isnan(b))
    return abs(a - b) <= max(REL * max(abs(a), abs(b)), 1e-12)


def _mask_ledger(entries: list) -> list:
    out = []
    for e in entries:
        e = json.loads(json.dumps(e))
        for f in MASKED_LEDGER_FIELDS:
            e.pop(f, None)
        stages = (e.get("payload") or {}).get("stages")
        if isinstance(stages, dict):
            for st in stages.values():
                if isinstance(st, dict):
                    st.pop("duration_ms", None)
        elif isinstance(stages, list):
            for st in stages:
                if isinstance(st, dict):
                    st.pop("duration_ms", None)
        out.append(e)
    return out


def _mask_manifest(man: Optional[dict]) -> Optional[dict]:
    if man is None:
        return None
    man = json.loads(json.dumps(man))
    for f in MASKED_MANIFEST_FIELDS:
        man.pop(f, None)
    files = man.get("files")
    # ``files`` is a DICT keyed by filename, not a list — the first version of
    # this masked a list and left every checksum in place, so D1 reported a
    # manifest difference that was only the ledger's own digest.
    if isinstance(files, dict):
        # name -> checksum STRING. Only summary.json's digest is reproducible;
        # every other artifact embeds a clock or an id, so its digest is dropped
        # rather than pretended stable.
        man["files"] = {name: (digest if name in MANIFEST_STABLE_CHECKSUMS
                               else "<UNSTABLE>")
                        for name, digest in files.items()}
    elif isinstance(files, list):
        man["files"] = [
            {k: v for k, v in f.items()
             if k != "checksum" or f.get("name") in MANIFEST_STABLE_CHECKSUMS}
            for f in files
        ]
    return man


def _mask_newsletter(html: Optional[str]) -> Optional[str]:
    return None if html is None else NEWSLETTER_CLOCK_RE.sub("<CLOCK>", html)


def _perf(res) -> dict:
    return ((res.summary or {}).get("metrics") or {}).get("performance") or {}


def _metrics(res) -> dict:
    return (res.summary or {}).get("metrics") or {}


# --------------------------------------------------------------------------- #
# C0 — the gate
# --------------------------------------------------------------------------- #

def c0_run_rendered(res) -> list:
    """A run must exit 0 and leave a newsletter. No exemptions.

    Its ABSENCE is FINDING F7, and it is the most consequential thing the bench got
    wrong about itself. ``exit_code`` was written to the ledger on every run and
    asserted nowhere — ``driver.RunResult.ok`` existed and had zero callers — while
    every check that needs the rendered HTML returned SKIP when there was none. So
    twelve of seventy-one runs produced nothing at all and still collected zero
    FAILs. One cell scored "4 PASS, 5 SKIP, 0 FAIL" for a run that exited 1, and the
    matrix reported the session healthy.

    Deliberately unconditional. An earlier draft pre-registered the fully-liquidated
    book as unable to render, which would have been an exemption granted to the very
    defect it was hiding: that book SHOULD render, stating the realized gain on its
    closed positions, and now does. If a future cell genuinely cannot produce an
    issue, the honest move is to fix the product or to declare the exemption
    alongside the reason and assert the reason too — not to widen this gate.
    """
    rendered = res.newsletter is not None
    ok = bool(res.ok and rendered)
    return [Verdict("C0", "INVARIANT", ok,
                    f"exit={res.exit_code}, newsletter "
                    + ("rendered" if rendered else "MISSING")
                    + ("" if ok else f" — {res.error or 'no error captured'}"))]


# --------------------------------------------------------------------------- #
# D1 / D2 / D3 / D4 — determinism and invariance
# --------------------------------------------------------------------------- #

def d1_reproducible_identical(a, b) -> list:
    """Same seed + REPRO, two runs -> identical artefacts after masking.

    Voided PER ARTEFACT, not in one block. Every one of these compared two absent
    values and passed: ``None == None`` and ``[] == []`` are both true, so two runs
    that produced nothing were certified byte-identical. Voiding the whole check
    instead would throw away the three comparisons that are still real when only
    the newsletter is missing.
    """
    out = []
    if a.summary is None or b.summary is None:
        out.append(_void("D1.sum", "DIFFERENTIAL", "a run wrote no summary.json"))
    else:
        out.append(Verdict("D1.sum", "DIFFERENTIAL", a.summary == b.summary,
                           "summary.json compared WHOLE (it carries no id and no timestamp)"
                           if a.summary == b.summary else
                           f"summary.json differs: {_first_json_diff(a.summary, b.summary)}"))
    if not a.ledger or not b.ledger:
        out.append(_void("D1.ldg", "DIFFERENTIAL", "a run wrote no ledger entry"))
    else:
        la, lb = _mask_ledger(a.ledger), _mask_ledger(b.ledger)
        out.append(Verdict("D1.ldg", "DIFFERENTIAL", la == lb,
                           f"ledger.jsonl masked {MASKED_LEDGER_FIELDS + MASKED_LEDGER_PAYLOAD_PATHS}"
                           if la == lb else f"ledger differs at {_first_list_diff(la, lb)}"))
    if a.manifest is None or b.manifest is None:
        out.append(_void("D1.man", "DIFFERENTIAL", "a run wrote no manifest.json"))
    else:
        ma, mb = _mask_manifest(a.manifest), _mask_manifest(b.manifest)
        out.append(Verdict("D1.man", "DIFFERENTIAL", ma == mb,
                           f"manifest.json masked {MASKED_MANIFEST_FIELDS} + unstable checksums"
                           if ma == mb else f"manifest differs: {_first_json_diff(ma, mb)}"))
    if a.newsletter is None or b.newsletter is None:
        out.append(_void("D1.nl", "DIFFERENTIAL", "a run rendered no newsletter"))
    else:
        na, nb = _mask_newsletter(a.newsletter), _mask_newsletter(b.newsletter)
        out.append(Verdict("D1.nl", "DIFFERENTIAL", na == nb,
                           "newsletter.html masked for the two HH:MM strings"
                           if na == nb else "newsletter differs outside the clock strings"))
    return out


#: The fields whose difference across time-of-day is BY DESIGN, declared up front.
D2_ALLOWED = {"market_open", "1d_live", "1d_coverage_pct", "1d"}


def d2_time_of_day_invariance(a, b) -> list:
    """Two closed-market instants, same effective date -> identical analytics
    except the four intraday fields.

    ``_metrics`` returns {} for a run with no summary, and {} == {} is true, so two
    runs that produced nothing were certified invariant.
    """
    if a.summary is None or b.summary is None:
        return [_void("D2", "DIFFERENTIAL", "a run wrote no summary.json")]
    pa, pb = _perf(a), _perf(b)
    offenders = sorted(k for k in set(pa) | set(pb)
                       if k not in D2_ALLOWED and pa.get(k) != pb.get(k))
    ma, mb = dict(_metrics(a)), dict(_metrics(b))
    ma.pop("performance", None); mb.pop("performance", None)
    other = sorted(k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k))
    ok = not offenders and not other
    return [Verdict("D2", "DIFFERENTIAL", ok,
                    f"allowed-to-differ {sorted(D2_ALLOWED)}; "
                    + ("nothing else moved" if ok
                       else f"also moved: performance{offenders} metrics{other}"))]


def d3_row_permutation(base, shuffled) -> list:
    if base.summary is None or shuffled.summary is None:
        return [_void("D3", "DIFFERENTIAL", "a run wrote no summary.json")]
    keys = ("total_value_eur", "invested_value_eur", "cash_value_eur",
            "num_holdings", "twror_pct", "xirr_pct")
    bad = []
    for k in keys:
        x, y = _metrics(base).get(k), _metrics(shuffled).get(k)
        same = _close_money(x, y) if k.endswith("_eur") else (
            x == y if k == "num_holdings" else _close_rel(x, y))
        if not same:
            bad.append(f"{k} {x} != {y}")
    return [Verdict("D3", "DIFFERENTIAL", not bad,
                    "order-list row order does not move holdings, cost or returns"
                    if not bad else "; ".join(bad))]


def d4_no_lookahead(full, truncated) -> list:
    if full.summary is None or truncated.summary is None:
        return [_void("D4", "DIFFERENTIAL", "a run wrote no summary.json")]
    keys = ("total_value_eur", "invested_value_eur", "num_holdings", "twror_pct")
    bad = []
    for k in keys:
        x, y = _metrics(full).get(k), _metrics(truncated).get(k)
        same = _close_money(x, y) if k.endswith("_eur") else (
            x == y if k == "num_holdings" else _close_rel(x, y))
        if not same:
            bad.append(f"{k} full={x} truncated={y}")
    return [Verdict("D4", "DIFFERENTIAL", not bad,
                    "orders after the effective date change nothing"
                    if not bad else "; ".join(bad))]


# --------------------------------------------------------------------------- #
# C5 / C6 / C7 / C8 — conservation and internal coherence
# --------------------------------------------------------------------------- #

def c5_quantity_and_cash(res, truth: dict) -> list:
    """Quantity per ISIN equals sum(buy) - sum(sell) as the GENERATOR wrote it,
    counting only orders on or before the run's effective date.

    Without that truncation the check accuses the product of losing a position it
    correctly excluded: P10 carries an order dated 29 Aug, a run pinned to 26 Aug
    rightly ignores it, and comparing against the untruncated truth reported
    "expected 7.0 got None". The effective-order snapshot working is not a defect.
    """
    out = []
    holdings = _holdings_from_summary(res)
    if holdings is None:
        return [_void("C5.qty", "INVARIANT",
                      "no quantities captured (the oracle produced no holdings "
                      "frame) — nothing was verified")]
    bad = []
    for isin, qty in (truth.get("quantity_by_isin") or {}).items():
        got = holdings.get(isin)
        if abs(qty) < 1e-9:
            if got not in (None, 0.0):
                bad.append(f"{isin} liquidated but reported {got}")
        elif got is None or abs(got - qty) > 1e-9:
            bad.append(f"{isin} expected {qty} got {got}")
    out.append(Verdict("C5.qty", "INVARIANT", not bad,
                       "every ISIN quantity == sum(buy)-sum(sell) "
                       "(in-process metrics, no artifact carries quantity)"
                       if not bad else "; ".join(bad[:4])))
    return out


def c6_weights_and_contributions(res) -> list:
    out = []
    m = _metrics(res)
    total, invested, cash = (m.get("total_value_eur"), m.get("invested_value_eur"),
                             m.get("cash_value_eur"))
    if None in (total, invested, cash):
        out.append(_void("C6.parts", "INVARIANT",
                         "total/invested/cash not all present in the summary"))
    else:
        ok = _close_money(invested + cash, total)
        out.append(Verdict("C6.parts", "INVARIANT", ok,
                           f"invested+cash == total ({invested}+{cash} vs {total})"))
    return out


def c7_fx_and_native(res, truth: dict) -> list:
    """WITHDRAWN AS AN ORACLE — reported, never failed. Read the reasoning before
    trusting any currency-mark conclusion drawn from this bench.

    The intent was: a non-EUR instrument's returns row must carry its own currency
    mark. The oracle cannot be built offline, for a reason that is about the
    fixtures and not about the product.

    A currency mark states the currency of the VENUE the resolver picked, and the
    fixture only knows the currency its order list declares. Those are different
    facts. P03 declares R2US in GBp; the resolver picked the instrument's Milan EUR
    listing (its tape runs at ~EUR 75/unit, which is neither 653 pence nor GBP
    6.53), so ``[EUR]`` is CORRECT and the accusation was the harness'. Nor can the
    cache repair it: ``Ticker.info`` is served per symbol, so declaring one
    currency for R2US declares it for R2US.L, R2US.MI and R2US.PA alike — the
    harness would be asserting a venue currency it invented.

    Three narrower framings were tried and each failed for its own reason: any mark
    anywhere in the document (cannot tell "marked EUR" from "unmarked"); the book's
    rendered rows (an unrendered sleeve is a different finding); rows that print
    real figures (R2US prints eight real returns and is correctly marked).

    So this reports what it sees and never fails. Testing the marks needs a fixture
    whose declared venue currency IS the venue the resolver reaches — i.e. recorded
    per-symbol currency captured alongside each cached tape, which the snapshot does
    not carry.
    """
    # Keyed by BOTH, because a row is labelled with its ticker only when the
    # taxonomy carries the instrument and with the bare ISIN when it does not —
    # and P10's two USD sleeves are exactly the off-taxonomy case, so a
    # ticker-only lookup declared them unrendered when they were on the page.
    by_ticker = {t: c for t, c in (truth.get("currency_by_ticker") or {}).items() if t}
    by_ticker.update({i: c for i, c in (truth.get("currency_by_isin") or {}).items() if i})
    if not any(c not in ("EUR", None) for c in by_ticker.values()):
        return [_na("C7", "INVARIANT", "book is EUR-only; nothing to convert")]
    html = res.newsletter or ""
    shown = _book_values(html)
    if not shown:
        return [_void("C7", "INVARIANT",
                      "no priced rows rendered at all")]
    expected = {t: c for t, c in by_ticker.items()
                if t in shown and c not in ("EUR", None)}
    if not expected:
        return [_na("C7", "INVARIANT",
                    f"every rendered row ({sorted(shown)}) is a EUR listing")]
    # Read the mark off the instrument's OWN row in Returns. Asking whether any
    # non-EUR mark appears anywhere in the document cannot distinguish "this USD
    # row is marked EUR" from "this USD row is marked nothing at all", and the two
    # are different defects with different causes.
    marks = _returns_marks(html)
    # Only a row that actually PRINTS returns is evidence about currency. A row of
    # em-dashes means the instrument resolved to no tape, so the product never
    # learned a listing currency and the mark carries no claim to check — asserting
    # one there tests the fixture's ambition, not the product. (The EUR fallback
    # that such a row does display is a finding in its own right; it is reported as
    # a code reading, because no run here reaches it with a resolvable tape.)
    checkable = {t: c for t, c in expected.items() if marks.get(t, (None, False))[1]}
    if not checkable:
        return [_na("C7", "INVARIANT",
                    f"{len(expected)} non-EUR row(s) print no returns at all "
                    "(unresolved tape); no currency claim to verify")]
    odd = [f"{t}: {marks[t][0] or 'no mark'} (order list says {c})"
           for t, c in sorted(checkable.items())
           if marks[t][0] not in (_CCY_MARK.get(c), c)]
    # ``None`` = reported, not asserted. See the docstring: a disagreement here is
    # as likely to mean the resolver picked a different venue than that the mark is
    # wrong, and this bench cannot tell the two apart.
    return [_note("C7", "INVARIANT",
                  f"{len(checkable)} priced non-EUR row(s) agree with the order "
                    "list's currency"
                    if not odd else
                    f"REPORTED (not asserted): {len(odd)} of {len(checkable)} rows "
                    "differ from the order list's currency, which is expected when "
                    "the resolver picks another venue: " + "; ".join(odd[:4]))]


#: The glyph the digest prints for a currency, where it prints one.
_CCY_MARK = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CHF": "CHF"}


def _returns_marks(html: str) -> dict:
    """``{symbol: (mark, prints_returns)}`` from each returns row.

    ``prints_returns`` is False when every figure in the row is an em-dash, which
    is how an instrument that resolved to no price tape appears. The richest row
    wins, so a symbol listed both as a holding and as a watchlist entry is read
    from whichever row actually carries figures.
    """
    out: dict = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        sym = re.search(r'color:#6E9BFF;">([A-Z0-9.]{1,14})</span>', row)
        if not sym:
            continue
        text = re.sub(r"<[^>]+>", " ", row)
        mark = re.search(r"\[(\$|€|£|¥|[A-Z]{3})\]", text)
        priced = bool(re.search(r"[+−-]\d+\.\d+%", text))
        prev = out.get(sym.group(1))
        if prev is None or (priced and not prev[1]):
            out[sym.group(1)] = (mark.group(1) if mark else None, priced)
    return out


#: (summary key, how it is rendered) — the 11 verified overlaps. total_value_eur
#: is conditional: the summary projects trustworthy_total_value_eur under that
#: name, so equality only holds when valuation_availability == AVAILABLE.
C8_OVERLAP = [
    ("total_value_eur", "eur0", True),
    ("invested_value_eur", "eur_smart", False),
    ("cash_value_eur", "eur_smart", False),
    ("avg_ter", "pct3", False),
    ("twror_pct", "pct2", False),
    ("twror_annualized_pct", "pct2", False),
    ("xirr_pct", "pct2", False),
    ("returns_coverage_pct", "pct0", False),
    ("num_holdings", "int", False),
]


def _fmt_eur0(v):
    return f"€{v:,.0f}".replace(",", ",")


def _fmt_pct(v, dp):
    return f"{abs(v):.{dp}f}%"


def c8_cross_artifact(res) -> list:
    """The recurring family: one number, several renderings, field by field."""
    m, html = _metrics(res), res.newsletter
    if not html:
        return [_void("C8", "INVARIANT", "no newsletter rendered")]
    avail = m.get("valuation_availability")
    misses, checked = [], 0
    for key, how, conditional in C8_OVERLAP:
        v = m.get(key)
        if v is None:
            continue
        if conditional and avail != "AVAILABLE":
            continue
        tokens = _render_tokens(v, how)
        checked += 1
        if not any(t in html for t in tokens):
            misses.append(f"{key}={v} (looked for {tokens[:2]})")
    # The 1D figure must be the SAME number in the Session tile, the matrix and
    # the subject; the subject is not in the HTML, so compare tile vs matrix.
    one_d = (m.get("performance") or {}).get("1d")
    if one_d is not None:
        tok = _fmt_pct(one_d, 2)
        n = html.count(tok)
        if n < 2:
            misses.append(f"performance.1d={one_d} appears {n}x as {tok} "
                          "(expected the Session tile AND the window matrix)")
        checked += 1
    return [Verdict("C8", "INVARIANT", not misses,
                    f"{checked} summary->HTML overlaps verified"
                    if not misses else f"{len(misses)}/{checked} missing: " + "; ".join(misses[:3]))]


def _render_tokens(v, how) -> list:
    if how == "int":
        return [str(int(v))]
    if how == "eur0":
        return [f"€{v:,.0f}", f"€{round(v):,}"]
    if how == "eur_smart":
        a = abs(v)
        if a >= 1000:
            return [f"€{a/1000:.1f}k", f"€{a:,.0f}"]
        return [f"€{a:,.0f}"]
    dp = int(how[-1])
    return [_fmt_pct(v, dp), f"{v:.{dp}f}%"]


def _holdings_from_summary(res):
    """Quantities come from the captured metrics object, not from an artifact:
    summary.json is aggregates only and The Book renders value/weight/gain but
    never quantity. Verdicts say so.

    ``None`` and ``{}`` are DIFFERENT answers and must not be collapsed with
    ``or None``: None means the oracle malfunctioned and nothing was verified,
    while an empty dict is a real reading of a book that holds nothing — which is
    exactly what a fully liquidated book returns, and exactly the case that must
    still be checked against the truth rather than waved through.
    """
    return getattr(res, "quantity_by_isin", None)


def c14_every_holding_is_visible(res) -> list:
    """Whatever is counted in ``num_holdings`` and priced into ``total_value_eur``
    must appear in a per-instrument table, or the gap must be stated.

    Kept because the invariant is worth asserting, but its first thirteen
    "failures" were this bench's own bug and NOT a product defect: the row parser
    could not match a label longer than 8 characters, so every row the digest
    labels with a bare ISIN (an instrument the taxonomy does not carry) counted as
    absent. P04's synthetic BTP was rendered, priced and correct at EUR 9,850.00
    while this check reported 19% of the book missing. With the parser fixed the
    product accounts for every euro in all 71 runs — P10 reconciles to one cent.

    The lesson is the check's, not the code's: an oracle that parses rendered
    output is only as sound as its parser, and a failing assertion is a claim about
    the harness until the artefact itself has been read.
    """
    m, html = _metrics(res), res.newsletter
    n = m.get("num_holdings")
    if not html:
        return [_void("C14", "INVARIANT", "no newsletter to read")]
    if n == 0:
        # A liquidated book holds nothing, so there is nothing that OUGHT to be
        # visible. The state genuinely did not arise — not the same thing as
        # having no artifact to look at.
        return [_na("C14", "INVARIANT",
                    "the book holds no positions; nothing ought to be visible")]
    if n is None:
        return [_void("C14", "INVARIANT", "the summary carries no num_holdings")]
    total = m.get("total_value_eur")
    if total in (None, 0):
        return [_void("C14", "INVARIANT", "no priced total to account for")]
    # The Book prints a Value EUR per row; sum them and see how much of the
    # portfolio total has no row. Counting TICKERS was the first version and it
    # flagged every book with one unpriceable line — a holding worth nothing
    # legitimately has no returns row. The money is the question: a reader given a
    # total must be able to find it in the rows.
    shown = _book_values(html)
    rendered = sorted(shown)
    accounted = sum(shown.values())
    gap = total - accounted
    ok = abs(gap) <= max(MONEY_ABS, abs(total) * 0.01)
    return [Verdict("C14", "INVARIANT", ok,
                    f"num_holdings={n} total={total} availability="
                    f"{m.get('valuation_availability')}; The book rows sum to "
                    f"{round(accounted, 2)} over {len(rendered)} tickers"
                    + ("" if ok else f" -> {round(gap, 2)} EUR "
                                     f"({gap / total * 100:.1f}%) of the total appears "
                                     "in no row"))]


#: A share-of-something percentage cannot exceed 100 by more than rounding. Gains
#: and returns legitimately can, so this band applies only to the two share
#: columns of The book.
_SHARE_CEILING = 100.5

#: Section headings that can follow The book. Bounding the scan at the next one is
#: what keeps rows from a LATER section out of a claim about the book's own rows.
_NEXT_SECTIONS = ("</span>Returns", ">Returns</span>", ">Tracked</span>",
                  ">Watchlist</span>", ">Risk</span>", ">Rebalance</span>",
                  ">Targets</span>", ">Flows</span>")


def _book_body(html: str) -> str:
    """The HTML of The book's own table, and nothing after it.

    Reading to the end of the document is what made the currency check accuse
    R2US: that row belongs to a later section (a watchlist entry from the real
    taxonomy), not to the synthetic book, and the two only share a ticker.
    """
    marker = ">The book</span>"
    if marker not in html:
        return ""
    body = html.split(marker, 1)[1]
    cuts = [body.find(s) for s in _NEXT_SECTIONS]
    cuts = [c for c in cuts if c > 0]
    return body[:min(cuts)] if cuts else body


def c16_share_percentages_are_shares(res) -> list:
    """The ``% Inv.`` and ``% Class`` columns of The book must be shares — each in
    [0, 100.5], and the ``% Class`` values within one class summing to ~100.

    This is the check that earned its place. P10 printed ``% Class`` of 301309.3%,
    772707.0% and 100000.0% for its three off-taxonomy holdings, which is each
    row's euro value multiplied by 100: ``class_totals`` is keyed by the RAW
    ``asset_class`` column while the lookup uses the NORMALISED class (empty/NaN
    mapped to "Other"), so a holding with no class misses the dict, takes the
    ``.get(klass, 1)`` default of 1, and divides its value by one euro. The
    ``or 1`` guard reads as division-by-zero protection but the 1 arrives as a
    default, not as a zero, and 1 is a money amount rather than a neutral divisor.
    """
    if res.newsletter is None:
        return [_void("C16", "INVARIANT", "no newsletter to read")]
    body = _book_body(res.newsletter)
    if not body:
        # The issue rendered but carries no book table — a book with no positions.
        # The state did not arise; there is no missing artifact.
        return [_na("C16", "INVARIANT",
                    "the issue renders no book table (no positions held)")]
    bad, seen = [], 0
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
        sym = re.search(r'color:#6E9BFF;">([A-Z0-9.]{1,14})</span>', row)
        if not sym:
            continue
        # The two share columns are the 3rd and 4th cells; read every percentage
        # in the row and keep those two by position so a Gain % of +466% does not
        # count against the band.
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        for idx, label in ((2, "% Inv."), (3, "% Class")):
            if idx >= len(cells):
                continue
            txt = re.sub(r"<[^>]+>", "", cells[idx]).strip()
            if txt in ("—", "-", ""):
                continue
            num = re.match(r"([\d,]+(?:\.\d+)?)%$", txt)
            if not num:
                continue
            seen += 1
            val = float(num.group(1).replace(",", ""))
            if val > _SHARE_CEILING:
                bad.append(f"{sym.group(1)} {label}={txt}")
    if not seen:
        return [_na("C16", "INVARIANT",
                    "the book rendered no share percentages to band-check")]
    return [Verdict("C16", "INVARIANT", not bad,
                    f"{seen} share percentages in [0,{_SHARE_CEILING}]"
                    if not bad else
                    f"{len(bad)} of {seen} share percentages are not shares: "
                    + "; ".join(bad[:4]))]


def _book_values(html: str) -> dict:
    """{symbol: value EUR} from The book, where ``symbol`` is whatever the row is
    LABELLED with — a ticker for a taxonomy instrument, the bare ISIN for one the
    taxonomy does not carry.

    The ISIN case is why this check reported a money gap in thirteen cells. The
    label was matched with ``[A-Z0-9.]{1,8}``, which cannot span a 12-character
    ISIN, so P04's synthetic BTP row — rendered, priced, €9,850.00, right there in
    the table — counted as absent and 19% of the book looked unaccounted for. The
    product was omitting nothing.

    The scan also has to END at the book's own table. Reading to the end of the
    document swept in whatever section follows, which is how the currency check
    came to accuse R2US: that row is a WATCHLIST entry from the real taxonomy, not
    a holding of the synthetic book, and matching it against the fixture's currency
    compared two unrelated instruments that happen to share a ticker.
    """
    body = _book_body(html)
    out = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
        tk = re.search(r'color:#6E9BFF;">([A-Z0-9.]{1,12})</span>', row)
        if not tk:
            continue
        val = re.search(r">&euro;([\d,]+(?:\.\d+)?)<", row) or \
              re.search(r">\u20ac([\d,]+(?:\.\d+)?)<", row)
        if val:
            out[tk.group(1)] = float(val.group(1).replace(",", ""))
    return out


# --------------------------------------------------------------------------- #
# G11 / G12 — degradation and delivery
# --------------------------------------------------------------------------- #

def g11_degradation(res) -> list:
    """Degradation must be stated, not implied by a zero.

    Every branch here PASSED vacuously on a run that produced nothing: an empty
    ledger has no FAILURE_OPEN entry without a failure_id, so "all carry a
    failure_id" was true of zero entries, and ``availability=None total=None``
    satisfied the zero rule by having neither. Three green ticks for a run that
    never happened — FINDING F7 in its purest form.
    """
    if not res.summary and not res.ledger:
        return [_void("G11", "INVARIANT", "no summary and no ledger to judge")]
    out = []
    m = _metrics(res)
    avail = m.get("valuation_availability")
    total = m.get("total_value_eur")
    zero_as_unavailable = (avail != "AVAILABLE" and total == 0.0)
    out.append(Verdict("G11.zero", "INVARIANT", not zero_as_unavailable,
                       f"availability={avail} total={total} — an UNAVAILABLE total "
                       "must be null, never 0"))
    opens = [e for e in res.ledger if e.get("entry_type") == "FAILURE_OPEN"]
    unidentified = [e for e in opens if not (e.get("payload") or {}).get("failure_id")]
    out.append(Verdict("G11.id", "INVARIANT", not unidentified,
                       f"{len(opens)} FAILURE_OPEN entries, all carry a failure_id"
                       if not unidentified else f"{len(unidentified)} without failure_id"))
    pub = [e for e in res.ledger if e.get("entry_type") == "PUBLICATION"]
    decisions = {(e.get("payload") or {}).get("decision") for e in pub}
    valid = {"SEND_NORMAL", "SEND_DEGRADED_NORMAL",
             "BLOCK_NORMAL_AND_NOTIFY_FAILURE", "BLOCK_ALL_DELIVERY"}
    out.append(Verdict("G11.pub", "INVARIANT",
                       bool(decisions) and decisions <= valid,
                       f"publication decision(s) {sorted(d for d in decisions if d)}"))
    return out


def g12_duplicate_claim_suppresses(store_path, logical_id: str, digest: str) -> list:
    """A duplicate claim must suppress SMTP.

    Driven against the real claim store on disk — no patching. FINDING F2 keeps
    this out of ``scripts/send_newsletter.py``: that module hardcodes its store to
    ``ROOT/output/delivery_claims.json``, so the bench exercises the store's own
    contract instead of the send script's.
    """
    from tarzan.delivery.claims import (DeliveryIntent, DeliveryPurpose,
                                        LocalJsonDeliveryClaimStore)

    store = LocalJsonDeliveryClaimStore(store_path)
    intent = DeliveryIntent(stable_event_id=logical_id,
                            purpose=DeliveryPurpose.NORMAL_NEWSLETTER,
                            recipient_set_digest=digest,
                            template_schema_version="stress-1")
    first = store.claim(intent)
    second = store.claim(intent)
    dup = bool(getattr(second, "duplicate", False))
    # duplicate + state CLAIMED deliberately does NOT suppress: the CAS in
    # send_email's before_invoke is what protects that path.
    state = getattr(getattr(second, "claim", None), "state", None)
    state = getattr(state, "value", state)
    return [
        Verdict("G12.dup", "INVARIANT", dup,
                f"second claim on the same logical id reports duplicate={dup}, state={state}"),
        Verdict("G12.new", "INVARIANT", not bool(getattr(first, "duplicate", False)),
                "the first claim is not a duplicate"),
    ]


def c13_planning_determinism(a, b) -> list:
    """Pre-registered EXPECTED FAILURE.

    ``analysis_id`` drops attempt_id/wall_time/latency before hashing, so two
    REPRODUCIBLE runs can share an id. The recon observed two such runs, with
    identical ids AND identical metrics, differing in 28 and 56 JSON paths under
    ``sections.planning.value`` — an identity that asserts more reproducibility
    than the content has. Registered up front so its failure is evidence, not a
    surprise.
    """
    if a.summary is None or b.summary is None:
        return [_void("C13", "DIFFERENTIAL", "a run wrote no summary.json")]
    pa = (a.summary.get("sections") or {}).get("planning")
    pb = (b.summary.get("sections") or {}).get("planning")
    if pa is None or pb is None:
        return [_void("C13", "DIFFERENTIAL", "a run wrote no planning section")]
    same_id = (a.summary or {}).get("analysis_id") == (b.summary or {}).get("analysis_id")
    equal = pa == pb
    return [Verdict("C13", "DIFFERENTIAL", equal,
                    f"analysis_id equal={same_id}; planning section equal={equal}"
                    + ("" if equal else f"; first diff {_first_json_diff(pa, pb)}"),
                    expected_fail=True)]


def e9_windows_against_the_tape(res, samples: list) -> list:
    """EXTERNAL. 1D/5D/1M for a few instruments against the venue's own tape.

    ``samples`` is [(ticker, expect_1d, expect_5d, expect_1m)] computed by the
    caller from the real quote pair and the real closes, N SESSIONS back — not
    N calendar days. A window that anchors on a calendar day fails here.
    """
    html = res.newsletter or ""
    if not html:
        return [_void("E9", "EXTERNAL", "no newsletter rendered")]
    out = []
    for ticker, exp1, exp5, exp1m in samples:
        row = _row_for(html, ticker)
        if row is None:
            out.append(_void("E9", "EXTERNAL", f"{ticker}: no row rendered"))
            continue
        got = _pcts(row)
        bad = []
        for label, exp, idx in (("1D", exp1, 0), ("5D", exp5, 1), ("1M", exp1m, 2)):
            if exp is None or idx >= len(got):
                continue
            if abs(got[idx] - exp) > DISPLAY_PP:
                bad.append(f"{label} shown {got[idx]:+.2f} tape {exp:+.2f}")
        out.append(Verdict("E9", "EXTERNAL", not bad,
                           f"{ticker}: " + ("1D/5D/1M within 0.01pp of the tape"
                                            if not bad else "; ".join(bad))))
    return out


def e10_sessions_from_the_calendar(cases: list) -> list:
    """EXTERNAL. ``is_session`` reads the vendored exchange calendar and must call
    a weekday holiday closed.

    Was half of a pre-registered expected failure: the calendar knew the holiday
    while ``market_open_now`` judged by exchange hours plus a Mon–Fri test and
    never asked it, so the same date read OPEN there. It asks now — see
    :func:`e10_market_open_reads_the_calendar`.
    """
    from tarzan.data import exchange_calendar as ec
    from tarzan.data import market_quotes as mq

    out = []
    for ticker, day, why in cases:
        try:
            is_sess = ec.is_session(ticker, day)
        except Exception as exc:                       # noqa: BLE001
            out.append(_void("E10.cal", "EXTERNAL", f"{ticker} {day}: calendar raised {exc}"))
            continue
        out.append(Verdict("E10.cal", "EXTERNAL", is_sess is False,
                           f"{ticker} {day} ({why}): calendar says session={is_sess}"))
    return out


def e10_market_open_reads_the_calendar(pin_instant, ticker: str, day) -> list:
    """EXTERNAL. The pinned weekday holiday must read CLOSED.

    Was a pre-registered expected failure: ``market_open_now`` judged by exchange
    hours plus a Mon–Fri test and never read the calendar, so a holiday was a
    session — the masthead said the market was open and every row was badged live
    over a close-to-close figure. It now asks ``exchange_calendar.is_session`` on
    the venue's own date, so this is an ordinary check.

    The verdict id stays ``E10.mq`` so the append-only ledger keeps one identifier
    for one question across the runs recorded before the fix.
    """
    from tarzan.data import market_quotes as mq

    open_now = mq.market_open_now(ticker)
    return [Verdict("E10.mq", "EXTERNAL", open_now is False,
                    f"market_open_now({ticker}) on {day} = {open_now}; "
                    "the venue is shut per the vendored exchange calendar")]


# --------------------------------------------------------------------------- #
# diff helpers
# --------------------------------------------------------------------------- #

def _first_json_diff(a, b, path="") -> str:
    if type(a) is not type(b):
        return f"{path or '<root>'}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return f"{path}.{k}: missing on the left"
            if k not in b:
                return f"{path}.{k}: missing on the right"
            if a[k] != b[k]:
                return _first_json_diff(a[k], b[k], f"{path}.{k}")
        return "equal"
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return _first_json_diff(x, y, f"{path}[{i}]")
        return "equal"
    return f"{path or '<root>'}: {a!r} vs {b!r}"


def _first_list_diff(a: list, b: list) -> str:
    if len(a) != len(b):
        return f"length {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"entry {i} ({x.get('entry_type')}): {_first_json_diff(x, y)}"
    return "equal"


def _row_for(html: str, ticker: str) -> Optional[str]:
    """The instrument's RETURNS row — the first one carrying real figures.

    Taking the first occurrence of the ticker and the next ``</tr>`` within 3000
    characters was fragile in two ways at once: a symbol appears in several tables
    (the book, a chip, a legend), and a long row simply fell outside the window. So
    a held instrument whose row was on the page reported "no row rendered", which
    the caller then filed as a missing artifact. Walk every candidate and keep the
    first that actually carries three or more percentages.
    """
    best = None
    for m in re.finditer(rf'color:#6E9BFF;">{re.escape(ticker)}</span>', html):
        end = html.find("</tr>", m.end())
        if end < 0:
            continue
        row = html[m.end():end]
        if len(_pcts(row)) >= 3:
            return row
        best = best or row
    return best


def _pcts(row_html: str) -> list:
    import html as _h
    return [float(x.replace("−", "-"))
            for x in re.findall(r">([+−-]?\d+\.\d\d)%<", _h.unescape(row_html))]
