"""Seeded fixture generator: 10 synthetic books, each with its own order list.

Every book is reproducible from its seed, and the generator RETURNS the ground
truth it wrote (quantity per ISIN, cash from net flows, per-instrument currency)
so the conservation checks have an oracle instead of comparing Tarzan to itself.

Privacy: no position of the operator's is used. The ISINs are public identifiers
taken from the committed ``input/instrument_taxonomy.csv``; every quantity, price
and date is generated. Two books deliberately carry instruments that are NOT in
the taxonomy, and one carries an ISIN that resolves to nothing.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

#: (isin, ticker, kind, currency) — public identifiers with a tape in the cache
#: snapshot, so the bench runs offline. Currency is the ORDER currency, which is
#: what drives the minor-unit and FX paths.
EU_ETF = [
    ("IE00B4L5Y983", "SWDA", "ETF", "EUR"),
    ("IE00B6R52259", "ISAC", "ETF", "EUR"),
    ("IE00BTJRMP35", "XMME", "ETF", "EUR"),
    ("LU0274209740", "XMJP", "ETF", "EUR"),
    ("LU0380865021", "XESC", "ETF", "EUR"),
    ("LU0328475792", "XSX6", "ETF", "EUR"),
    ("IE00BL25JL35", "XDEQ", "ETF", "EUR"),
    ("IE00BL25JM42", "XDEV", "ETF", "EUR"),
    ("IE00BL25JP72", "XDEM", "ETF", "EUR"),
    ("IE00B3YLTY66", "IMIE", "ETF", "EUR"),
]
DE_ETF = [
    ("IE00077IIPQ8", "NTSG", "ETF", "EUR"),
    ("IE000RJECXS5", "AVWC", "ETF", "EUR"),
    ("IE0003R87OG3", "AVWS", "ETF", "EUR"),
    ("IE000OW54ZX1", "AVUS", "ETF", "EUR"),
    ("IE000SDFJUU0", "AVEU", "ETF", "EUR"),
    ("FR0014010HV4", "LVWC", "ETF", "EUR"),
]
#: US-listed, USD-quoted: the native-tape / FX path.
US_ETF = [
    ("US88636J2042", "RSSB", "ETF", "USD"),
    ("US88636J3453", "RSSY", "ETF", "USD"),
    ("US82889N2282", "CTAP", "ETF", "USD"),
    ("US97717Y3523", "WTIP", "ETF", "USD"),
]
#: LSE. R2US.L is a real cached listing; the GBp order currency exercises the
#: minor-unit /100 rescale, which is decided by the ORDER currency.
LSE_ETF = [("IE00BJ38QD84", "R2US", "ETF", "GBp")]
BOND_ETF = [
    ("LU0290357507", "X15E", "ETF", "EUR"),
    ("LU0290357846", "X25E", "ETF", "EUR"),
    ("LU0290357929", "XGIN", "ETF", "EUR"),
]
COMMODITY = [
    ("IE00B579F325", "SGLD", "ETF", "EUR"),
    ("IE00BKFB6L02", "UEQC", "ETF", "EUR"),
    ("IE00BFXR6159", "ENCO", "ETF", "EUR"),
]
CASHLIKE = [("IE0005PGDJ30", "MONEY", "ETF", "EUR")]
ALT = [("LU2951555403", "DBMFE", "ETF", "EUR"), ("IE000UWJUW87", "CATB", "ETF", "EUR")]

#: Deliberately OUTSIDE the curated taxonomy — a real instrument the taxonomy
#: does not describe. Both have a cached tape, so they price but are unclassified.
OFF_TAXONOMY = [
    ("US9219097683", "VEU", "ETF", "USD"),
    ("US4642872349", "EEM", "ETF", "USD"),
]
#: Invented, resolves to nothing. Exercises the unresolvable-identity path.
UNRESOLVABLE = [("XX0000000000", "ZZZZ", "ETF", "EUR")]
#: Invented BTP. A coupon-paying bond quoted per 100 has no cached tape offline,
#: so this exercises the coupon + per-100 + no-tape paths, not a priced sleeve.
BTP = [("IT0009999999", "BTP99", "BOND", "EUR")]
#: A single stock with no cached tape.
STOCK = [("NL0000009355", "UNA", "STOCK", "EUR")]

# --- Single stocks, by venue -------------------------------------------------
# The corpus was 46 ETF orders, 6 BOND and 2 STOCK, and its one stock was
# EUR-quoted on Euronext Amsterdam. So nothing exercised a US single stock, a
# suffixless US ticker, a stock in a minor unit, or a stock with no TER — which is
# how a whole class of venue-resolution defect went unnoticed for 71 runs.
#
# Real, public, large-cap instruments with their real ISINs; every SIZE below is
# invented from the book's seed. None has a cached tape, so each prices carry-flat
# from its own order ladder — the same footing as every other synthetic sleeve
# here, which is enough to exercise kind=STOCK, the currency and venue paths, the
# no-TER path and the dividend path.
US_STOCK = [
    ("US0378331005", "AAPL", "STOCK", "USD"),
    ("US5949181045", "MSFT", "STOCK", "USD"),
    ("US4781601046", "JNJ", "STOCK", "USD"),     # a reliable dividend payer
]
#: LSE, quoted in PENCE — the minor-unit rescale on a STOCK rather than an ETF.
UK_STOCK = [("GB0009895292", "AZN", "STOCK", "GBp")]
#: SIX Swiss. CHF has no FX series in the offline cache, so this is the
#: currency-known-but-unconvertible path on a stock.
CH_STOCK = [("CH0038863350", "NESN", "STOCK", "CHF")]

ORDER_COLUMNS = [
    "date", "trade_date", "type", "isin", "name", "ticker", "quantity",
    "currency", "price_native", "fx_rate", "gross_eur", "fees_eur", "net_eur",
    "instrument_kind", "source",
]
FEE = 19.0


@dataclass
class Truth:
    """What the generator KNOWS it wrote — the oracle for conservation."""
    quantity_by_isin: dict = field(default_factory=dict)
    cash_from_flows: float = 0.0
    currency_by_isin: dict = field(default_factory=dict)
    #: Same currencies keyed by TICKER, because the newsletter's per-instrument
    #: rows are keyed by ticker and a check that wants to read a rendered row's
    #: currency mark has no ISIN to look up.
    currency_by_ticker: dict = field(default_factory=dict)
    kind_by_isin: dict = field(default_factory=dict)
    order_count: int = 0
    first_date: Optional[dt.date] = None
    last_date: Optional[dt.date] = None
    liquidated_isins: list = field(default_factory=list)


@dataclass
class Book:
    pid: str
    seed: int
    note: str
    orders: list
    truth: Truth
    targets: dict
    per_holding: list


#: Currencies quoted in a minor unit (a hundredth of the major one), mirroring the
#: enricher's own ``_MINOR_TO_MAJOR_CURRENCY`` keys.
_MINOR_UNITS = frozenset({"GBp", "GBX", "ZAc", "ZAC", "ILa", "ILA"})


def _price(rng: random.Random, base: float) -> float:
    return round(base * rng.uniform(0.75, 1.35), 4)


def _row(date, trade, typ, inst, qty, price, *, fx=1.0, fee=FEE, name=None):
    isin, ticker, kind, ccy = inst
    # A bond is quoted per 100 of nominal, and the CASH leg has to honour that or
    # the fixture is internally inconsistent. Writing gross = qty x price for
    # 10,000 nominal at 98.5 claimed the book paid EUR 985,000 for a position the
    # product then correctly valued at EUR 9,850 — and the digest dutifully printed
    # a truthful -99.00% gain and -EUR 963k of P&L on a EUR 52k portfolio. The
    # arithmetic was the product's; the impossible trade was this generator's.
    # Bonds quote per 100 of nominal; GBp/ZAc/ILa quote in a minor unit worth a
    # hundredth of the major one. Either way the cash leg is price/100 per unit, and
    # writing qty x price instead made the fixture claim a cost the product could
    # never reconcile — P03's R2US showed a truthful -90.09% gain because the order
    # said GBp 653 was EUR 653.
    per_unit = price / 100.0 if (kind == "BOND" or ccy in _MINOR_UNITS) else price
    gross = round(abs(qty) * per_unit * (1.0 / fx if fx else 1.0), 2)
    if typ in ("buy", "transfer_in"):
        net = round(-(gross + fee), 2)
        signed = abs(qty)
    elif typ in ("sell", "transfer_out"):
        net = round(gross - fee, 2)
        signed = -abs(qty)
    else:                                    # coupon / dividend
        gross = round(abs(qty), 2)
        net = gross
        signed = 0.0
        price = ""
    return {
        "date": date.isoformat(), "trade_date": trade.isoformat(), "type": typ,
        "isin": isin, "name": name or f"Synthetic {ticker}", "ticker": ticker,
        "quantity": signed, "currency": ccy, "price_native": price,
        "fx_rate": fx, "gross_eur": gross, "fees_eur": fee if price != "" else 0.0,
        "net_eur": net, "instrument_kind": kind, "source": "stress",
    }


def _apply(truth: Truth, row: dict) -> None:
    isin = row["isin"]
    truth.quantity_by_isin[isin] = truth.quantity_by_isin.get(isin, 0.0) + float(row["quantity"])
    truth.cash_from_flows = round(truth.cash_from_flows + float(row["net_eur"]), 2)
    truth.currency_by_isin[isin] = row["currency"]
    truth.currency_by_ticker[str(row.get("ticker") or "")] = row["currency"]
    truth.kind_by_isin[isin] = row["instrument_kind"]
    truth.order_count += 1
    d = dt.date.fromisoformat(row["date"])
    truth.first_date = d if truth.first_date is None else min(truth.first_date, d)
    truth.last_date = d if truth.last_date is None else max(truth.last_date, d)


def _build(pid, seed, note, plan, *, targets=None, per_holding=None) -> Book:
    """``plan`` is a callable (rng, emit) that emits rows in date order."""
    rng = random.Random(seed)
    truth = Truth()
    rows: list = []

    def emit(row):
        rows.append(row)
        _apply(truth, row)

    plan(rng, emit)
    rows.sort(key=lambda r: (r["date"], r["isin"]))
    return Book(pid=pid, seed=seed, note=note, orders=rows, truth=truth,
                targets=targets or {"target_cash_buffer_eur": 1000},
                per_holding=per_holding or [])


# --------------------------------------------------------------------------- #
# The ten books
# --------------------------------------------------------------------------- #

def _p1(rng, emit):
    """EU-only accumulating ETFs, three weeks, a handful of orders."""
    d = dt.date(2026, 8, 3)
    for inst, base in zip(EU_ETF[:4], (95, 105, 78, 100)):
        emit(_row(d, d - dt.timedelta(days=2), "buy", inst,
                  rng.randint(10, 60), _price(rng, base)))
        d += dt.timedelta(days=4)


def _p2(rng, emit):
    """US-only, USD, two years — the native-tape and FX path."""
    d = dt.date(2024, 9, 2)
    for inst, base in zip(US_ETF, (30, 25, 28, 37)):
        emit(_row(d, d, "buy", inst, rng.randint(50, 200), _price(rng, base),
                  fx=round(rng.uniform(1.05, 1.15), 4)))
        d += dt.timedelta(days=140)
    # a partial sell late in the window
    emit(_row(dt.date(2026, 6, 15), dt.date(2026, 6, 11), "sell", US_ETF[0], 20,
              _price(rng, 31), fx=1.09))


def _p3(rng, emit):
    """EU + US + LSE, four years, including a GBp (pence) order."""
    d = dt.date(2022, 9, 5)
    for inst, base in zip(EU_ETF[:3] + US_ETF[:2] + LSE_ETF, (95, 105, 78, 30, 25, 640)):
        fx = 1.0 if inst[3] == "EUR" else (1.10 if inst[3] == "USD" else 0.86)
        emit(_row(d, d, "buy", inst, rng.randint(15, 120), _price(rng, base), fx=fx))
        d += dt.timedelta(days=210)


def _p4(rng, emit):
    """Fixed income across three tax years: bond ETFs with a tape, plus an
    invented coupon-paying BTP that has none (per-100 + coupon + no-tape)."""
    d = dt.date(2020, 10, 1)
    for inst, base in zip(BOND_ETF, (170, 120, 210)):
        emit(_row(d, d, "buy", inst, rng.randint(20, 80), _price(rng, base)))
        d += dt.timedelta(days=430)
    emit(_row(dt.date(2021, 3, 1), dt.date(2021, 2, 25), "buy", BTP[0], 10000, 98.5))
    for year in (2022, 2023, 2024, 2025, 2026):
        emit(_row(dt.date(year, 3, 1), dt.date(year, 3, 1), "coupon", BTP[0], 175.0, 0))


def _p5(rng, emit):
    """Single stock + commodity ETCs, five years."""
    d = dt.date(2021, 9, 6)
    emit(_row(d, d, "buy", STOCK[0], 300, 48.0))
    for inst, base in zip(COMMODITY, (380, 120, 32)):
        d += dt.timedelta(days=300)
        emit(_row(d, d, "buy", inst, rng.randint(10, 90), _price(rng, base)))
    emit(_row(dt.date(2025, 5, 6), dt.date(2025, 5, 2), "sell", STOCK[0], 100, 55.0))


def _p6(rng, emit):
    """Distributing ETFs with dividends and transfers, three years."""
    d = dt.date(2023, 9, 4)
    for inst, base in zip(ALT + CASHLIKE, (120, 105, 10)):
        emit(_row(d, d, "buy", inst, rng.randint(30, 150), _price(rng, base)))
        d += dt.timedelta(days=200)
    emit(_row(dt.date(2024, 1, 15), dt.date(2024, 1, 15), "transfer_in", EU_ETF[0], 25, 92.0))
    emit(_row(dt.date(2025, 2, 20), dt.date(2025, 2, 20), "transfer_out", EU_ETF[0], 5, 99.0))
    for year in (2024, 2025, 2026):
        emit(_row(dt.date(year, 6, 10), dt.date(year, 6, 10), "dividend", ALT[0], 88.0, 0))


def _p7(rng, emit):
    """Total liquidation to zero, then a re-purchase of the same ISIN."""
    inst = EU_ETF[1]
    emit(_row(dt.date(2024, 10, 1), dt.date(2024, 9, 27), "buy", inst, 100, 96.0))
    emit(_row(dt.date(2025, 4, 14), dt.date(2025, 4, 10), "sell", inst, 100, 103.0))
    emit(_row(dt.date(2025, 11, 3), dt.date(2025, 10, 30), "buy", inst, 60, 99.5))
    emit(_row(dt.date(2026, 5, 18), dt.date(2026, 5, 14), "buy", inst, 40, 108.0))


def _p8(rng, emit):
    """A single order in the whole book."""
    emit(_row(dt.date(2026, 8, 25), dt.date(2026, 8, 21), "buy", EU_ETF[0], 12, 101.0))


def _p9(rng, emit):
    """Entirely liquidated: every position closed, book at zero today."""
    for inst, base, day in ((EU_ETF[2], 78, 1), (EU_ETF[3], 100, 40), (COMMODITY[0], 380, 80)):
        b = dt.date(2025, 3, 3) + dt.timedelta(days=day)
        qty = rng.randint(20, 70)
        emit(_row(b, b, "buy", inst, qty, _price(rng, base)))
        s = b + dt.timedelta(days=260)
        emit(_row(s, s, "sell", inst, qty, _price(rng, base)))


def _p10(rng, emit):
    """The pathological book: off-taxonomy instruments, an unresolvable ISIN,
    fractional quantities, a weekend order date, and an order dated on the run
    day itself."""
    d = dt.date(2020, 9, 7)
    for inst, base in zip(OFF_TAXONOMY, (58, 44)):
        emit(_row(d, d, "buy", inst, rng.randint(40, 120), _price(rng, base), fx=1.11))
        d += dt.timedelta(days=700)
    emit(_row(dt.date(2023, 6, 12), dt.date(2023, 6, 8), "buy", UNRESOLVABLE[0], 50, 20.0))
    # Fractional quantity.
    emit(_row(dt.date(2024, 2, 5), dt.date(2024, 2, 1), "buy", EU_ETF[0], 3.4567, 97.25))
    # Order DATED ON A SATURDAY (2026-08-29 is a Saturday).
    emit(_row(dt.date(2026, 8, 29), dt.date(2026, 8, 29), "buy", EU_ETF[4], 7, 101.5))
    # Order dated on the run day used by the pinned instants C1-C5.
    emit(_row(dt.date(2026, 8, 26), dt.date(2026, 8, 26), "buy", EU_ETF[5], 9, 172.0))


def _p11(rng, emit):
    """US only, stocks AND ETFs, every leg USD on a suffixless ticker.

    The book the corpus was missing. A US stock differs from a US ETF in ways the
    pipeline cares about: it carries no TER (so it enters the avg_ter weighted
    average with nothing to contribute), the curated taxonomy does not describe it
    (so its asset class comes from nowhere), and it pays a dividend rather than a
    distribution.
    """
    d = dt.date(2024, 3, 4)
    for inst, base in zip(US_STOCK[:2] + US_ETF[:2], (170, 400, 30, 28)):
        emit(_row(d, d, "buy", inst, rng.randint(8, 60), _price(rng, base),
                  fx=round(rng.uniform(1.05, 1.15), 4)))
        d += dt.timedelta(days=120)
    # A dividend on the STOCK, once a year. Stocks pay dividends; ETFs distribute.
    for year in (2025, 2026):
        emit(_row(dt.date(year, 5, 12), dt.date(year, 5, 12), "dividend",
                  US_STOCK[0], 61.0, 0))
    # A partial sell, so the book has a realized leg as well as open positions.
    emit(_row(dt.date(2026, 7, 20), dt.date(2026, 7, 16), "sell", US_STOCK[1], 5,
              _price(rng, 430), fx=1.09))


def _p12(rng, emit):
    """FOUR venues in one book, so four exchange calendars govern one portfolio.

    A US stock, a Milan ETF, a London stock in pence and a Swiss stock in francs.
    Every US holiday is a Milan trading day and the reverse, so a book like this is
    where a venue resolved to the wrong calendar — or to none — shows up as one
    sleeve's window sliding while its neighbours' hold.
    """
    d = dt.date(2023, 6, 6)
    plan = [(US_STOCK[0], 165.0, 1.09), (EU_ETF[0], 96.0, 1.0),
            (UK_STOCK[0], 11200.0, 0.86), (CH_STOCK[0], 104.0, 0.96)]
    for inst, base, fx in plan:
        emit(_row(d, d, "buy", inst, rng.randint(6, 45), _price(rng, base), fx=fx))
        d += dt.timedelta(days=190)
    # Top up the Swiss leg: CHF has no FX series offline, so a second order proves
    # the unconvertible-currency path survives more than one visit.
    emit(_row(dt.date(2026, 4, 14), dt.date(2026, 4, 10), "buy", CH_STOCK[0],
              rng.randint(4, 20), _price(rng, 112), fx=0.94))


def _p13(rng, emit):
    """Every income type in one book: a stock DIVIDEND, an ETF DISTRIBUTION and a
    bond COUPON, so the three cannot be conflated by a check that only ever saw one.
    """
    d = dt.date(2022, 4, 5)
    emit(_row(d, d, "buy", US_STOCK[2], rng.randint(20, 90), _price(rng, 165),
              fx=1.08))
    emit(_row(d + dt.timedelta(days=60), d + dt.timedelta(days=60), "buy",
              BOND_ETF[2], rng.randint(30, 140), _price(rng, 100)))
    emit(_row(d + dt.timedelta(days=150), d + dt.timedelta(days=150), "buy",
              BTP[0], 8000, 99.0))
    for year in (2023, 2024, 2025, 2026):
        emit(_row(dt.date(year, 3, 15), dt.date(year, 3, 15), "dividend",
                  US_STOCK[2], 74.0, 0))
        emit(_row(dt.date(year, 9, 20), dt.date(year, 9, 20), "dividend",
                  BOND_ETF[2], 52.0, 0))
        emit(_row(dt.date(year, 11, 8), dt.date(year, 11, 8), "coupon",
                  BTP[0], 140.0, 0))


def _p14(rng, emit):
    """Stocks ONLY, three venues, no fund anywhere in the book.

    Nothing here carries a TER, so the weighted average has no contributor at all,
    and no leg is described by the curated taxonomy, so every asset class is the
    residual one. Both are paths a book with a single ETF in it silently covers up.
    """
    d = dt.date(2021, 11, 9)
    for inst, base, fx in ((US_STOCK[0], 148.0, 1.13), (UK_STOCK[0], 8600.0, 0.85),
                           (STOCK[0], 47.0, 1.0)):
        emit(_row(d, d, "buy", inst, rng.randint(12, 70), _price(rng, base), fx=fx))
        d += dt.timedelta(days=260)


BOOKS = [
    ("P01", 1001, "EU-only accumulating ETFs, 3 weeks, 4 orders", _p1),
    ("P02", 1002, "US-only USD, 2 years, native tape + FX", _p2),
    ("P03", 1003, "EU+US+LSE, 4 years, includes a GBp order", _p3),
    ("P04", 1004, "fixed income, 6 years, 3 tax years, coupons + per-100 BTP", _p4),
    ("P05", 1005, "single stock + commodity ETCs, 5 years", _p5),
    ("P06", 1006, "distributing + dividends + transfers, 3 years", _p6),
    ("P07", 1007, "liquidated to zero then re-bought", _p7),
    ("P08", 1008, "a single order", _p8),
    ("P09", 1009, "entirely liquidated, zero today", _p9),
    ("P10", 1010, "pathological: off-taxonomy, unresolvable ISIN, fractional, weekend, same-day", _p10),
    ("P11", 1011, "hybrid US: single stocks + ETFs, all USD, suffixless tickers", _p11),
    ("P12", 1012, "hybrid four venues: US stock + Milan ETF + LSE stock (GBp) + Swiss stock (CHF)", _p12),
    ("P13", 1013, "hybrid income: stock dividend + ETF distribution + bond coupon", _p13),
    ("P14", 1014, "stocks only, three venues, no fund and therefore no TER", _p14),
]


def build_all() -> list:
    out = []
    for pid, seed, note, plan in BOOKS:
        book = _build(pid, seed, note, plan)
        if pid == "P09":
            book.truth.liquidated_isins = [i for i, q in book.truth.quantity_by_isin.items()
                                           if abs(q) < 1e-9]
        out.append(book)
    return out


def write(book: Book, root: Path) -> Path:
    d = root / book.pid
    d.mkdir(parents=True, exist_ok=True)
    with (d / "order_list.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ORDER_COLUMNS)
        w.writeheader()
        w.writerows(book.orders)
    with (d / "targets.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "value", "description"])
        for k, v in book.targets.items():
            w.writerow([k, v, "stress fixture"])
    with (d / "targets_per_holding.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["isin", "ticker", "name", "target_portfolio", "no_buy_no_sell"])
        for row in book.per_holding:
            w.writerow(row)
    (d / "seed.json").write_text(json.dumps({
        "portfolio_id": book.pid, "seed": book.seed, "note": book.note,
        "generator": "tarzan.stress.generate", "order_count": book.truth.order_count,
        "first_order": book.truth.first_date.isoformat() if book.truth.first_date else None,
        "last_order": book.truth.last_date.isoformat() if book.truth.last_date else None,
        "truth": {
            "quantity_by_isin": book.truth.quantity_by_isin,
            "cash_from_flows": book.truth.cash_from_flows,
            "currency_by_isin": book.truth.currency_by_isin,
            "currency_by_ticker": book.truth.currency_by_ticker,
            "kind_by_isin": book.truth.kind_by_isin,
            "liquidated_isins": book.truth.liquidated_isins,
        },
    }, indent=2, sort_keys=True) + "\n")
    return d


def main() -> None:
    root = Path(__file__).parent / "fixtures"
    for book in build_all():
        d = write(book, root)
        print(f"{book.pid}  {book.truth.order_count:3d} orders  "
              f"{book.truth.first_date} -> {book.truth.last_date}  {book.note}")
    print(f"\nwritten under {root}")


if __name__ == "__main__":
    main()
