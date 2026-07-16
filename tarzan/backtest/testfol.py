"""testfol.io proxy expansion — shareable lines to paste on the site.

Maps each instrument to a testfol.io proxy expression (leveraged tickers,
efficient-core expansions, managed-futures blends, carry stand-ins), baking the
fund TER as an ``?E=`` drag. Used only by the Excel export.
"""

from __future__ import annotations

from tarzan.export._format import short_instrument_name

from tarzan.backtest.model import EQ, FI, GOLD, COMM, ALT, CRYPTO, CASH, Portfolio
from tarzan.backtest.ter import instrument_ter

# ("one", TICKER): 1:1 proxy. ("lev2", EXPR): native daily-leverage ticker.
# ("ec_world"/"ec_eur",): efficient-core 90/60 expanded into equity+bond−CASHX.
_TESTFOL_PROXY = {
    "CL2": ("lev2", "SPYSIM?L=2&E=0.5&SW=1.1&SP=0.4"),   # Amundi MSCI USA 2x
    "LVWC": ("lev2", "VTSIM?L=2&E=1.60&SW=1.1&SP=0.4"),  # Amundi MSCI World 2x
    "NTSG": ("ec_world",),
    "NTSZ": ("ec_eur",),
    "NTSX": ("ec_world",),
    "EXUS": ("one", "EFASIM"),
    "SWDA": ("one", "VTSIM"), "VWCE": ("one", "VTSIM"), "FWRA": ("one", "VTSIM"),
    "XDEM": ("one", "MTUM"), "IWMO": ("one", "MTUM"),
    "XDEQ": ("one", "QUAL"), "IWQU": ("one", "QUAL"),
    "XDEV": ("one", "VLUE"), "IWVL": ("one", "VLUE"),
    "ZPRV": ("one", "AVUV"), "ZPRX": ("one", "AVDV"),
    "SGLD": ("one", "GLDSIM"), "WGLD": ("one", "GLDSIM"),
    "MFEH": ("blend", [("DBMFSIM", 0.5), ("KMLMSIM", 0.5)]),
    "DBMFE": ("blend", [("DBMFSIM", 0.5), ("KMLMSIM", 0.5)]),
    "DBMF": ("blend", [("DBMFSIM", 0.5), ("KMLMSIM", 0.5)]),
    "UEQC": ("carry", "DBMFSIM"), "CRRY": ("carry", "DBMFSIM"), "CRRE": ("carry", "DBMFSIM"),
    "XMME": ("one", "EEM"), "AVEM": ("one", "EEM"), "EM710": ("one", "EEM"),
    "XMJP": ("one", "EWJ"),
    "XESC": ("one", "EZU"),
    "XSX6": ("one", "IEUR"),
    "AGGH": ("one", "BNDW"),
    "XGIN": ("one", "TIP"),
    "XS5E": ("one", "SPYSIM"),
    "LMTH": ("one", "TLT"), "X710": ("one", "IEFSIM"), "X15E": ("one", "IEFSIM"),
    "CATB": ("one", "CASHX"),
}

# Fallback testfol proxy by dominant asset class when a ticker is unmapped.
_TESTFOL_FALLBACK = {EQ: "VTSIM", FI: "TLT", GOLD: "GLDSIM", COMM: "DBC",
                     ALT: "KMLMSIM", CRYPTO: "BTC-USD", CASH: "CASHX"}


def testfol_lines(p: "Portfolio") -> list[tuple[str, float]]:
    """testfol.io (ticker, weight%) lines for a portfolio, using the KB proxies."""
    agg_w: dict[str, float] = {}
    agg_wter: dict[str, float] = {}
    baked: set[str] = set()

    def add(t, w, ter=None):
        agg_w[t] = agg_w.get(t, 0.0) + w
        if ter is None:
            baked.add(t)
        else:
            agg_wter[t] = agg_wter.get(t, 0.0) + w * ter

    for it in p.items:
        w = it.weight
        ter = instrument_ter(it)
        rule = _TESTFOL_PROXY.get(it.bare.upper())
        if rule is None:
            dom = max(it.comp_notional, key=it.comp_notional.get) if it.comp_notional else EQ
            add(_TESTFOL_FALLBACK.get(dom, it.bare), w, ter)
        elif rule[0] == "lev2":
            add(rule[1], w)
        elif rule[0] == "one" or rule[0] == "carry":
            add(rule[1], w, ter)
        elif rule[0] == "blend":
            for t, f in rule[1]:
                add(t, w * f, ter)
        elif rule[0] == "ec_world":
            for t, f in (("SPYSIM", 0.63), ("EFASIM", 0.27), ("IEFSIM", 0.60)):
                add(t, w * f, ter)
            add("CASHX", w * -0.50)
        elif rule[0] == "ec_eur":
            for t, f in (("EZU", 0.90), ("IEFSIM", 0.60)):
                add(t, w * f, ter)
            add("CASHX", w * -0.50)

    out = []
    for t, w in agg_w.items():
        if abs(w) <= 0.05:
            continue
        if t in baked or t == "CASHX" or t not in agg_wter or w <= 0:
            expr = t
        else:
            e = round(agg_wter[t] / w, 2)
            expr = f"{t}?E={e}" if "?" not in t else f"{t}&E={e}"
        out.append((expr, round(w, 1)))
    return sorted(out, key=lambda kv: -kv[1])


def testfol_expand_item(it: "WhatIfItem") -> list[tuple[str, float]]:
    """testfol code line(s) for ONE instrument. TER baked as ?E=."""
    w = it.weight
    ter = instrument_ter(it)

    def e(t):
        return f"{t}?E={ter}"

    rule = _TESTFOL_PROXY.get(it.bare.upper())
    if rule is None:
        dom = max(it.comp_notional, key=it.comp_notional.get) if it.comp_notional else EQ
        base = _TESTFOL_FALLBACK.get(dom, it.bare)
        lines = [(e(base) if base != "CASHX" else base, w)]
    elif rule[0] == "lev2":
        lines = [(rule[1], w)]
    elif rule[0] == "one":
        lines = [(e(rule[1]), w)]
    elif rule[0] == "carry":
        lines = [(f"{e(rule[1])}  ⟨rough: carry≠trend; custom CSV → "
                  f"BNPIF73P + CASHX − 0.66%⟩", w)]
    elif rule[0] == "blend":
        lines = [(e(t), f * w) for t, f in rule[1]]
    elif rule[0] == "ec_world":
        lines = [(e("SPYSIM"), 0.63 * w), (e("EFASIM"), 0.27 * w),
                 (e("IEFSIM"), 0.60 * w), ("CASHX", -0.50 * w)]
    elif rule[0] == "ec_eur":
        lines = [(e("EZU"), 0.90 * w), (e("IEFSIM"), 0.60 * w), ("CASHX", -0.50 * w)]
    else:
        lines = [(e(it.bare), w)]
    return [(t, round(x, 1)) for t, x in lines]


def testfol_instrument_map(p: "Portfolio") -> list[dict]:
    """Per-instrument testfol breakdown: {ticker, name, weight, codes:[(expr,w)]}."""
    rows = []
    for it in p.items:
        rows.append({
            "ticker": it.bare,
            "name": short_instrument_name(it.holding.name or it.symbol, 40),
            "weight": it.weight,
            "codes": testfol_expand_item(it),
        })
    return rows
