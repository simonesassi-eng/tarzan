"""One pipeline run, isolated, with a pinned instant — and the evidence it left.

Every run gets its own temporary output root. Nothing reads ``input/`` and
nothing writes to ``output/``: the fixtures come from ``tarzan/stress/fixtures``,
the market cache is a snapshot pointed at by ``TARZAN_CACHE_DIR``, and artifacts
land under a caller-supplied temporary directory.

FINDING F2: the delivery layer cannot be isolated this way — ``tarzan/delivery``
hardcodes ``ROOT/output/<date>`` and ``ROOT/output/delivery_claims.json`` with
neither a flag nor an env var to move them. So the publication and claim checks
drive ``PublicationEvaluator`` and a seeded claim store directly and never go
through ``scripts/send_newsletter.py``.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import glob
import io
import json
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"


@dataclasses.dataclass
class RunResult:
    run_id: str
    portfolio: str
    instant: str
    mode: str
    strict: bool
    exit_code: Optional[int]
    outdir: Optional[str]
    summary: Optional[dict]
    ledger: list
    manifest: Optional[dict]
    newsletter: Optional[str]
    #: Per-holding quantities, captured from the metrics object. The Book renders
    #: value/weight/gain but never QUANTITY, so no artifact carries it — the
    #: conservation check is executed against the returned metrics instead, and
    #: says so.
    quantity_by_isin: dict
    network_attempts: list
    error: Optional[str]
    argv: list

    @property
    def ok(self) -> bool:
        return self.error is None and self.exit_code == 0

    def artifact(self, name: str) -> Optional[Path]:
        if not self.outdir:
            return None
        hits = sorted(Path(self.outdir).glob(f"*/*/{name}"))
        return hits[0] if hits else None


def listing_currencies(orders_csv: str) -> dict:
    """{ticker: ISO code} as the order list declares it.

    Feeds ``Ticker.info["currency"]``, so a synthetic listing arrives at the
    enricher quoted in the currency its own fixture says it trades in. Without
    this the currency is empty for every instrument and ``_own_tape`` falls back
    to EUR, which silently made every native-currency assertion vacuous.
    """
    out: dict = {}
    try:
        with open(orders_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                tk, ccy = (row.get("ticker") or "").strip(), (row.get("currency") or "").strip()
                if tk and ccy:
                    out.setdefault(tk, ccy)
    except (OSError, csv.Error):
        return {}
    return out


def prepare_cache(snapshot: Path, dest: Path) -> Path:
    """A per-session WRITABLE copy of the cache snapshot, so a run that stores a
    row cannot contaminate the next session's baseline."""
    dest.mkdir(parents=True, exist_ok=True)
    if snapshot.exists() and not any(dest.iterdir()):
        shutil.copytree(snapshot, dest, dirs_exist_ok=True)
    return dest


def run_one(*, portfolio: str, instant: Optional[str], mode: str, strict: bool,
            outroot: Path, cache_dir: Path, run_id: str,
            quotes: Optional[dict] = None,
            orders_override: Optional[str] = None,
            as_of_override: Optional[dt.date] = None,
            allow_network: bool = False,
            extra_patches=None) -> RunResult:
    """Drive one full pipeline run.

    ``mode`` is one of LIVE / PIT / REPRO. LIVE is the only mode that honours a
    pinned hour (F0-bis), so instants C1-C7 are meaningful only there; PIT and
    REPRO vary the DATE instead and their market state is forced closed.
    """
    from tarzan.stress import clock, net

    os.environ["TARZAN_CACHE_DIR"] = str(cache_dir)
    os.environ["TARZAN_DISABLE_AI"] = "1"
    os.environ.pop("TARZAN_STRICT_INPUT", None)

    fixture = FIXTURES / portfolio
    orders = orders_override or str(fixture / "order_list.csv")
    outdir = outroot / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    argv = ["--input_orders", orders,
            "--input_config", str(fixture / "targets.csv"),
            "--input_targets_per_holding", str(fixture / "targets_per_holding.csv"),
            "--output", str(outdir)]

    pin = clock.pin_for(instant) if instant else None
    if mode in ("PIT", "REPRO"):
        eff = as_of_override or (pin.effective_date if pin else dt.date(2026, 8, 26))
        argv += ["--as_of", eff.isoformat()]
        if mode == "REPRO":
            argv += ["--deterministic"]
    if strict:
        argv += ["--strict"]

    net.allow_network(allow_network)
    net.reset_attempts()
    # Serving the cache while ALSO allowing the network made the one networked
    # block meaningless: block E compared Yahoo's live tape against a run whose
    # every fetch boundary was still patched to the snapshot, so it reported four
    # window mismatches that were really a comparison of two different days.
    # Allowing the network means using it.
    if not allow_network:
        net.serve_from_cache(quotes=quotes,
                             currencies=listing_currencies(orders))
    net.skip_whatif()
    if pin is not None and mode == "LIVE":
        clock.install(pin)
    else:
        # A sticky pin would override this run's own --as_of and mode.
        clock.uninstall()
    if extra_patches:
        extra_patches()

    from tarzan.data import enricher
    enricher.reset_run_caches()

    exit_code, error = None, None
    try:
        from tarzan import main as tmain
        exit_code = tmain.main(argv)
    except BaseException as exc:                       # noqa: BLE001 — recorded
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"

    res = RunResult(run_id=run_id, portfolio=portfolio, instant=instant or "-",
                    mode=mode, strict=strict, exit_code=exit_code,
                    outdir=str(outdir), summary=None, ledger=[], manifest=None,
                    newsletter=None, quantity_by_isin={},
                    network_attempts=net.attempts(), error=error, argv=argv)
    _collect(res)
    return res


def quantities_in_process(*, portfolio: str, as_of, cache_dir: Path,
                          quotes=None) -> dict:
    """Per-holding quantities, obtained by calling orchestrator.run DIRECTLY.

    No artifact carries quantity: summary.json is aggregates only and The Book
    renders value, weight and gain but not size. The first attempt at this wrapped
    orchestrator.run and stashed the frame in a module global — and that global
    went stale between cells, which made the bench accuse a correctly liquidated
    book (P09, every position closed by Feb 2026, run at 29 Aug 2026) of still
    holding 31/67/30 units. Re-running it standalone showed num_holdings 0 and a
    zero total, i.e. the product was right and the harness was lying.

    So the metrics object is held in hand here instead of reached for through a
    global. It costs one extra run per checked cell and cannot go stale.
    """
    os.environ["TARZAN_CACHE_DIR"] = str(cache_dir)
    os.environ["TARZAN_DISABLE_AI"] = "1"
    from tarzan.data import enricher
    from tarzan.stress import net

    from tarzan.stress import clock
    clock.uninstall()                      # this call supplies its own as_of
    fixture = FIXTURES / portfolio
    net.serve_from_cache(
        quotes=quotes,
        currencies=listing_currencies(str(fixture / "order_list.csv")))
    net.skip_whatif()
    enricher.reset_run_caches()

    from tarzan import orchestrator
    metrics, _config = orchestrator.run(
        config_source=str(fixture / "targets.csv"),
        orders_source=str(fixture / "order_list.csv"),
        targets_per_holding_source=str(fixture / "targets_per_holding.csv"),
        as_of=as_of)
    df = getattr(metrics, "holdings_df", None)
    # Three outcomes, and they must not be collapsed:
    #   None  the oracle could not read a frame at all -> nothing was verified.
    #   {}    the frame is EMPTY -> a real reading of a book that holds nothing.
    #         A liquidated book's frame has no columns either, so the column check
    #         must come after the emptiness one or a legitimate empty book reads as
    #         a malfunction and C5 certifies the emptiness it failed to read.
    #   {..}  the quantities.
    if df is None:
        return None
    if df.empty:
        return {}
    if "isin" not in df or "quantity" not in df:
        return None
    return {str(r["isin"]): float(r["quantity"]) for _, r in df.iterrows()}


def _collect(res: RunResult) -> None:
    s = res.artifact("summary.json")
    if s:
        res.summary = json.loads(s.read_text())
    l = res.artifact("ledger.jsonl")
    if l:
        res.ledger = [json.loads(x) for x in l.read_text().splitlines() if x.strip()]
    m = res.artifact("manifest.json")
    if m:
        res.manifest = json.loads(m.read_text())
    n = res.artifact("newsletter.html")
    if n:
        res.newsletter = n.read_text()
