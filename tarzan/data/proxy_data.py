"""Long-history proxy indices for synthetic robustness backtests.

Maps asset-class / equity-geography buckets to long-history yfinance
proxies and fetches their full ("max") daily total-return-ish series, so a
portfolio's *exposure* can be replicated over multiple decades (covering
GFC 2008 / COVID 2020 / 2022) even when the actual instruments only have a
few years of history.

The financing leg uses ^IRX (13-week T-bill). All series are cached on
disk under a ``PROXYMAX_`` prefix so they never collide with the enricher's
5-year holding/benchmark histories.

These are deliberately US-listed USD proxies: the output is *modeled*
history for stress/rolling analysis, not an exact EUR replication.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from tarzan.data import manual_proxies, price_cache
from tarzan.engine.stats import TRADING_DAYS

# Approx all-in fee drag (%/yr) for the commodity-carry ETFs (UEQC ~0.34 TER,
# CRRY ~0.66 TER+swap) applied to the BNP carry index for a net-of-fees proxy.
_CARRY_FEE_ANNUAL = 0.006

# EUR risk-free: the ECB SDMX API's AAA euro-area government yield curve,
# 3-month spot rate — a real daily rate LEVEL (%), the direct EUR analogue of
# ^IRX (US 13-week T-bill). AAA-government = the truest euro risk-free (no bank
# credit premium as in 3M EURIBOR, no ETF fee drag as in a €STR ETF). Fetched
# from the ECB SDMX REST endpoint (CSV, no extra package) and disk-cached with
# staleness refresh like the yfinance proxies — no manual input file. Real
# daily data from 2004-09-06, updated every TARGET business day.
_ECB_YC_3M_URL = ("https://data-api.ecb.europa.eu/service/data/YC/"
                  "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_3M?format=csvdata")
_ECB_YC_3M_KEY = "ECB_YC_SR_3M"

# EONIA (euro overnight index average) — the real ECB overnight rate, used to
# fill the 1999→2004-09 tail that predates the AAA yield curve above. EONIA was
# discontinued on 2022-01-03, so it is a CLOSED series: fetched once, then read
# from cache forever (no staleness re-fetch). Overnight vs the 3M spot differ by
# only ~10-15bps in normal times — a negligible splice for 2000-2004.
_ECB_EONIA_URL = ("https://data-api.ecb.europa.eu/service/data/EON/"
                  "D.EONIA_TO.RATE?format=csvdata")
_ECB_EONIA_KEY = "ECB_EONIA"

# Pre-1999 FALLBACK only: 3-month EURIBOR annual averages (%), forward-filled to
# a daily path. This predates our ~2000 backtest window, so within any real
# window it is never actually reached — kept only as an ultimate safety net if
# BOTH ECB series are unreachable and uncached.
_EUR_RATE_ANNUAL: dict[int, float] = {
    2000: 4.40, 2001: 4.26, 2002: 3.32, 2003: 2.33, 2004: 2.11, 2005: 2.19,
    2006: 3.08, 2007: 4.28, 2008: 4.64, 2009: 1.22, 2010: 0.81, 2011: 1.39,
    2012: 0.57, 2013: 0.22, 2014: 0.21, 2015: -0.02, 2016: -0.26, 2017: -0.33,
    2018: -0.32, 2019: -0.36, 2020: -0.43, 2021: -0.55, 2022: 0.35, 2023: 3.43,
    2024: 3.58, 2025: 2.20,
}


def _eur_rate_for_year(y: int) -> float:
    """EURIBOR-3M annual average (%) for year ``y``, clamped to the table
    range (nearest edge year) for out-of-range years like 2026."""
    if y in _EUR_RATE_ANNUAL:
        return _EUR_RATE_ANNUAL[y]
    ys = sorted(_EUR_RATE_ANNUAL)
    return _EUR_RATE_ANNUAL[ys[0]] if y < ys[0] else _EUR_RATE_ANNUAL[ys[-1]]

logger = logging.getLogger(__name__)

# Each bucket resolves to the FIRST candidate that yfinance prices, ordered
# longest-history first (mutual funds reach further back than ETFs, à la
# testfol.io's SIM backfills) → enables a ~2000 start.
EQUITY_GEO_PROXY = {
    # ^SP500TR is the S&P 500 TOTAL-return index (dividends reinvested, 1988+);
    # ^GSPC is price-only and silently omits ~1.9%/yr of dividends, which would
    # understate the dominant USA sleeve and the beta/alpha benchmark. TR first,
    # price index only as a last-resort fallback.
    "USA": ["^SP500TR", "^GSPC"],             # S&P 500 total return, 1988+
    "Japan": ["EWJ"],                         # iShares Japan, 1996+
    "Eurozone EMU": ["EZU"],                  # iShares EMU, 2000+
    # Vanguard Developed Markets index (1999-08) → Fidelity Diversified Intl
    # (1991) → iShares EAFE (2001). Longest-history first, à la testfol SIMs,
    # so the developed-ex-US sleeve no longer caps the window at EFA's 2001.
    "Dev ex-USA ex-EMU ex-JP": ["VTMGX", "FDIVX", "EFA"],
    "Emerging Markets": ["VEIEX", "EEM"],     # Vanguard EM (1994) → EEM (2003)
    "Other": ["VTWSX", "ACWI"],               # Vanguard World (1994) → ACWI (2008)
}

# Non-equity asset class → long-history proxy candidates. "Alternative"/"Cash"
# are handled as the cash (financing) leg, not a ticker.
ASSET_PROXY = {
    "Fixed Income": ["VUSTX"],                # Vanguard Long-Term Treasury, 1986+
    "Gold": ["GC=F", "GLD"],                  # Gold futures (2000) → GLD (2004)
    "Commodities": ["^BCOM", "DBC"],          # Bloomberg Commodity index → DBC (2006)
    "Crypto": ["BTC-USD"],                    # Bitcoin, 2014+
}

CASH_KEYS = ("Alternative", "Cash & Cash Equivalents")
_FINANCING_SYMBOL = "^IRX"
# Managed-futures / trend proxy for the "Alternative" bucket. Real trend funds
# spliced longest-history first: RYMFX (Rydex/Guggenheim MF, ~2007) as the deep
# base — so the GFC 2008 crisis-alpha is captured, not flat cash — with AQMNX
# (AQR MF, ~2010, a cleaner trend model) taking precedence where they overlap.
# Cash fills the pre-2007 tail. For an exact long series (SG CTA Index, the
# index MFEH/DBMF replicate, back to 2000) ingest it once into the cache DB:
# `python -m tarzan.data.manual_proxies ingest MFSIM <path>` (read at runtime
# from the cache only — never parsed from a file at launch).
_MF_FUND = "AQMNX"
_MF_FUND_DEEP = "RYMFX"

_returns_memo: dict[tuple[str, str], pd.Series] = {}
# Which concrete ticker each bucket resolved to, and from when.
USED_PROXY: dict[str, tuple[str, pd.Timestamp]] = {}


_STALE_DAYS = 7   # re-fetch a cached proxy only once its tail is this old


def _clip_cached_to_as_of(data):
    """Return only cached observations visible at the run boundary."""
    if data is None or data.empty:
        return data
    from tarzan import runtime

    boundary = runtime.as_of()
    if boundary is None:
        return data
    try:
        cutoff = pd.Timestamp(boundary)
        if getattr(data.index, "tz", None) is not None:
            cutoff = cutoff.tz_localize(data.index.tz)
        return data.loc[data.index <= cutoff]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not clip cached provider data to as_of: %s", exc)
        return data.iloc[0:0]


def _fetch_max(symbol: str) -> pd.Series:
    """Full-history daily close for ``symbol`` (disk-cached, namespaced).

    The cached series is reused while its last date is fresh (< _STALE_DAYS old)
    and only re-fetched when stale, so historical closes are reused across runs
    but recent sessions stay current. Pinned runs return only cache rows visible
    at ``as_of`` and never attempt Yahoo transport.
    """
    key = f"PROXYMAX_{symbol}"
    cached = price_cache.load_history(key)
    from tarzan import runtime

    if not runtime.allows_live_transport():
        visible = _clip_cached_to_as_of(cached)
        return visible if visible is not None else pd.Series(dtype=float)
    if cached is not None and not cached.empty:
        last = pd.Timestamp(cached.index.max())
        if (pd.Timestamp.now().normalize() - last).days <= _STALE_DAYS:
            return cached
    try:
        import yfinance as yf
        from tarzan.data import _yf_net
        # auto_adjust=True is set EXPLICITLY: it makes fund/ETF closes
        # dividend- and split-adjusted (total return). The yfinance default has
        # flip-flopped across versions, so pinning it here prevents a silent
        # regression to price-only closes for the income-paying proxies.
        # Go through the shared spacing+retry so a proxy fetch survives a 429
        # burst the same way the enricher does (was raw before → flaky).
        h = _yf_net.fetch_yf(
            lambda: yf.Ticker(symbol).history(period="max", auto_adjust=True),
            what=f"proxy {symbol}", log=logger)
    except Exception as e:  # noqa: BLE001
        logger.warning("Proxy fetch failed for %s: %s", symbol, e)
        return cached if cached is not None else pd.Series(dtype=float)
    if h is None or h.empty or "Close" not in h:
        return cached if cached is not None else pd.Series(dtype=float)
    s = h["Close"].dropna()
    idx = s.index
    s.index = (idx.tz_convert("UTC").tz_localize(None).normalize()
               if getattr(idx, "tz", None) is not None else idx.normalize())
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if not s.empty:
        price_cache.store_history(key, s)
    return s


def _returns(symbol: str) -> pd.Series:
    from tarzan import runtime

    context = runtime.context()
    run_identity = "|".join((
        context.attempt_id,
        context.mode.value,
        str(context.effective_date or ""),
        context.captured_at.isoformat(),
    ))
    memo_key = (run_identity, symbol)
    if memo_key in _returns_memo:
        return _returns_memo[memo_key]
    px = _fetch_max(symbol)
    r = px.pct_change().dropna() if not px.empty else pd.Series(dtype=float)
    _returns_memo[memo_key] = r
    return r


def financing_daily() -> Optional[pd.Series]:
    """Daily financing rate (fraction) from ^IRX (annualised % → /100 /252).

    ^IRX is consumed as a yield LEVEL (not a price change). Dividing the annual
    rate by TRADING_DAYS (252) is consistent because the resulting series is
    applied on a trading-day grid (252 × daily ≈ annual). It would need /365 if
    ever laid on a calendar-day index.
    """
    px = _fetch_max(_FINANCING_SYMBOL)
    if px.empty:
        return None
    return (px / 100.0) / TRADING_DAYS


def _resolve_bucket(candidates: list[str]) -> tuple[pd.Series, Optional[str]]:
    """First candidate that yfinance prices → (returns, ticker used)."""
    for sym in candidates:
        r = _returns(sym)
        if r is not None and not r.empty:
            return r, sym
    return pd.Series(dtype=float), None


# ---------------------------------------------------------------------------
# Currency: the proxies are USD, but the real holdings' price_history is already
# EUR-converted by the enricher. Splicing a USD proxy tail onto EUR live returns
# would mix currencies inside one instrument, so by default we convert proxy
# returns to a EUR investor's (unhedged) returns — which also makes the whole
# backtest EUR-native (testfol.io is USD-only). Set to "USD" to keep raw USD.
# ---------------------------------------------------------------------------
_TARGET_CCY = "EUR"


def set_target_currency(ccy: Optional[str]) -> None:
    """Set the reporting currency for proxy returns ('EUR' default, or 'USD')."""
    global _TARGET_CCY
    _TARGET_CCY = (ccy or "EUR").strip().upper()


def _eur_per_usd() -> Optional[pd.Series]:
    """Daily EUR-per-USD level = 1 / EURUSD=X (which quotes USD per EUR)."""
    px = _fetch_max("EURUSD=X")
    if px is None or px.empty:
        return None
    return (1.0 / px).sort_index()


def _usd_returns_to_eur(usd_ret: pd.Series) -> pd.Series:
    """Convert a USD daily-return series to a EUR investor's UNHEDGED return.

    r_eur = (1+r_usd)·(1+r_fx) − 1, where r_fx is the change in EUR-per-USD
    (USD appreciating vs EUR ⇒ the EUR holder of a USD asset gains). Returns
    the input unchanged when the target is USD or FX data is unavailable.
    """
    if _TARGET_CCY == "USD" or usd_ret is None or usd_ret.empty:
        return usd_ret
    eu = _eur_per_usd()
    if eu is None or eu.empty:
        return usd_ret
    fx_ret = (eu.reindex(eu.index.union(usd_ret.index)).ffill()
              .pct_change().reindex(usd_ret.index).fillna(0.0))
    out = (1.0 + usd_ret) * (1.0 + fx_ret) - 1.0
    return out.dropna()


def _carry_returns(fin: Optional[pd.Series]) -> pd.Series:
    """Commodity-carry proxy. PREFER the vendored UEQC index (the actual UBS
    CMCI Commodity Carry 2.5x strategy, total-return, base 100 from 2001) — the
    correct benchmark for the UEQC ETF, better than the mismatched BNP x3
    CRRYSIM. It is a leveraged long/short commodity SPREAD (~currency-neutral,
    no FX conversion, like the FF legs) and gross of the ETF TER (charged by the
    engine on the modeled base), so nothing is added/subtracted here beyond
    splicing cash before its inception. Falls back to the manually-ingested BNP
    CRRYSIM excess-return index (2008+) when the UEQC bundle is absent. Empty if
    neither is available (the caller then leaves the carry sleeve unmodelled)."""
    lvl = _clip_cached_to_as_of(_ueqc_bundled())
    if lvl is not None and not lvl.empty:
        r = lvl.pct_change().dropna()
        if fin is not None and not fin.empty:      # carry = cash before 2001
            r = pd.concat([fin.loc[fin.index < r.index.min()], r]).sort_index()
        return r
    lvl = _clip_cached_to_as_of(manual_proxies.get_series("CRRYSIM"))
    if lvl is None or lvl.empty:
        return pd.Series(dtype=float)
    r = lvl.pct_change().dropna()
    r = _apply_collateral(r, fin)                     # excess → total return
    r = r - _CARRY_FEE_ANNUAL / TRADING_DAYS          # net-of-fees fund proxy
    # Splice onto cash before the index's 2008 inception so carry-holding
    # portfolios still reach the ~2000 window (carry sleeve = cash pre-2008),
    # mirroring the managed-futures pre-inception handling.
    if fin is not None and not fin.empty:
        r = pd.concat([fin.loc[fin.index < r.index.min()], r]).sort_index()
    return _usd_returns_to_eur(r)


def _ecb_fetch(url: str, key: str, static: bool = False) -> Optional[pd.Series]:
    """Daily rate LEVEL (%) from an ECB SDMX CSV REST endpoint, disk-cached.

    No extra package and no manual file — a plain CSV call. Reused from cache
    while fresh (or FOREVER when ``static`` marks a discontinued series), re-
    fetched when stale, and degrading gracefully to the cached copy on any
    network/parse failure.
    """
    cached = price_cache.load_history(key)
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return _clip_cached_to_as_of(cached)
    if cached is not None and not cached.empty:
        if static:
            return cached           # closed series (e.g. EONIA): never re-fetch
        last = pd.Timestamp(cached.index.max())
        if (pd.Timestamp.now().normalize() - last).days <= _STALE_DAYS:
            return cached
    try:
        import io
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=30).read().decode()
        df = pd.read_csv(io.StringIO(raw))
        s = pd.Series(df["OBS_VALUE"].values,
                      index=pd.to_datetime(df["TIME_PERIOD"])).dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s.index = s.index.normalize()
        if not s.empty:
            price_cache.store_history(key, s)
            return s
    except Exception as e:  # noqa: BLE001
        logger.warning("ECB SDMX fetch failed for %s: %s", key, e)
    return cached  # cached copy if we have one, else None → table fallback


def _ecb_eur_rate_level() -> Optional[pd.Series]:
    """Daily EUR risk-free LEVEL (%) from real ECB series, spliced longest-first:
    the AAA-govt 3-month spot rate (2004-09→, the ^IRX analogue) takes priority,
    and EONIA (the ECB overnight rate, 1999→2021) fills the earlier tail."""
    sr3m = _ecb_fetch(_ECB_YC_3M_URL, _ECB_YC_3M_KEY)
    eonia = _ecb_fetch(_ECB_EONIA_URL, _ECB_EONIA_KEY, static=True)
    level = sr3m if (sr3m is not None and not sr3m.empty) else None
    if eonia is not None and not eonia.empty:
        # combine_first: SR_3M wins where present; EONIA fills the pre-2004 gap.
        level = eonia if level is None else level.combine_first(eonia)
    return level


def risk_free_daily(start=None, end=None) -> Optional[pd.Series]:
    """Daily risk-free path (fraction, per trading day) in the TARGET currency,
    on the ^IRX trading-day grid — the time-varying input for an excess-return
    Sharpe/Sortino (r_t − rf_t), so ZIRP years and rate-hike years are each
    charged their OWN rate instead of a single window average.

    USD: the ^IRX 13-week T-bill path itself.
    EUR: REAL ECB data (``_ecb_eur_rate_level``) — AAA-govt 3M spot from 2004-09
    spliced onto EONIA back to 1999 — an independent EUR rate series, so 2011
    (EUR above the Fed) and 2015-2021 (negative EUR rates) are faithful. Within
    any ~2000+ window this is entirely real; the EURIBOR table is only an
    ultimate fallback if BOTH ECB series are unreachable and uncached.
    """
    fin = financing_daily()          # USD daily fraction on the ^IRX grid
    if fin is None or fin.empty:
        return None
    if _TARGET_CCY == "USD":
        s = fin
    else:
        # Ultimate fallback: annual EURIBOR table on the ^IRX grid (no NaN).
        tbl = pd.Series([_eur_rate_for_year(int(y)) for y in fin.index.year],
                        index=fin.index)
        tbl = (tbl / 100.0) / TRADING_DAYS
        lvl = _ecb_eur_rate_level()  # ECB rate LEVEL in %, from 1999
        if lvl is None or lvl.empty:
            s = tbl
        else:
            # A rate LEVEL persists between observations, so reindexing onto the
            # ^IRX grid with ffill is robust to the TARGET-vs-US calendar
            # mismatch. Convert %→daily fraction; table only fills any pre-1999
            # gap (outside our window).
            lvl_grid = lvl.reindex(fin.index, method="ffill")
            eur_daily = (lvl_grid / 100.0) / TRADING_DAYS
            s = eur_daily.where(eur_daily.notna(), tbl)
    return s.loc[start:end] if (start or end) else s


def risk_free_annual(start=None, end=None) -> Optional[float]:
    """Window-average annualised risk-free (%) in the TARGET currency — the
    mean of ``risk_free_daily`` over the window. Used for DISPLAY (the header
    figure) and as a scalar fallback; the newsletter Sharpe/Sortino themselves
    consume the full time-varying daily path via ``risk_free_daily`` (wired into
    ``MetricsEngine._rf_daily`` → ``risk_metric_row``), falling back to the
    scalar ``RISK_FREE_RATE`` only when the historical series is unavailable
    (e.g. a pinned run with no cached ^IRX/ECB rows).
    """
    s = risk_free_daily(start, end)
    if s is None or s.empty:
        return None
    return float(s.mean()) * TRADING_DAYS * 100.0


# ---------------------------------------------------------------------------
# Fama-French-Carhart factor legs (for the factor-aware backfill)
# ---------------------------------------------------------------------------
# The Ken French Data Library publishes the long-SHORT research factors as
# daily returns (US research factors, back to 1926/1927). SMB (size), HML
# (value) and MOM (momentum) are dollar-NEUTRAL long-short legs, so their EUR
# value ≈ their USD value (the FX on the long and short sides cancels) — no
# currency conversion is needed. Fetched from the public CSV zips and disk-
# cached like the ECB series (no manual file). Used to give factor ETFs
# (value/momentum/quality/size) a factor-AWARE pre-inception backfill instead
# of a plain market-cap proxy that discards the tilt.
# DEVELOPED-markets factor set (not the US one): our factor ETFs track MSCI
# WORLD (developed) indices, so the Developed research factors are the correct
# regressors. Using US factors badly understated the momentum loading (a World
# momentum fund on the US UMD factor); Developed factors recover it. Daily from
# 1990-07 (covers the ~2000 backtest window). Long-short legs are dollar-neutral
# so no currency conversion is needed.
_FF_DEV5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                "Developed_5_Factors_Daily_CSV.zip")
_FF_DEVMOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                  "Developed_Mom_Factor_Daily_CSV.zip")
_FF_CACHE_KEY = "FF_DEV_FACTORS_DAILY"
# Vendored snapshot of the Developed factor series, shipped WITH the script so
# the backtest is self-contained and reproducible offline (no live Ken French
# fetch required). The live fetch, when allowed, only EXTENDS this base with
# newer rows. Path is package-relative.
_FF_BUNDLED = Path(__file__).resolve().parent / "ff_developed_factors.csv.gz"
_FF_BUNDLED_CACHE: Optional[pd.DataFrame] = None

# EMERGING-markets factor set. Ken French publishes the EM research factors only
# MONTHLY (no daily), from 1991-07 (RMW-complete). The Developed legs are the
# WRONG regressors for an EM fund, so EM factor ETFs (e.g. Avantis AVEM) get
# their tilt from THESE legs instead — reconstructed at monthly resolution and
# spread over each month's trading days (see synthetic.factor_splice_monthly).
_FF_EM5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
               "Emerging_5_Factors_CSV.zip")
_FF_EM_CACHE_KEY = "FF_EM_FACTORS_MONTHLY"
_FF_EM_BUNDLED = Path(__file__).resolve().parent / "ff_emerging_factors.csv.gz"
_FF_EM_BUNDLED_CACHE: Optional[pd.DataFrame] = None

# Vendored Bloomberg Commodity Index (BCOM) daily LEVELS, 1991+. BCOM is an
# EXCESS-return index (^BCOM); the broad-commodity total-return proxy adds T-bill
# collateral on top. Shipped with the script so the commodity sleeve reconstructs
# to 1991 offline — the live ^BCOM feed is flaky (why broad commodities failed to
# backfill before). Same self-contained pattern as the FF factor bundle.
_BCOM_BUNDLED = Path(__file__).resolve().parent / "bcom_commodity.csv.gz"
_BCOM_BUNDLED_CACHE: Optional[pd.Series] = None

# Vendored UEQC (UBS CMCI Commodity Carry, 2.5x) index — daily TOTAL-return
# LEVELS, base 100 from 2001-08. The ACTUAL fund's strategy index, so it is the
# PREFERRED commodity-carry source over the mismatched BNP x3 CRRYSIM proxy.
# Self-contained/offline. Gross of the ETF TER (the engine charges UEQC's TER on
# the modeled base), so no fee is subtracted here. Verified against the real
# UEQC.DE NAV (2021+, EUR, net): total return matches (CAGR ~equal in EUR) and
# the EUR/USD beta is ~-0.2 (not ~1), so the series is already ~EUR/net-
# equivalent and is NOT FX-converted (like the FF long-short legs). BUT it
# correlates only ~0.39 with the real fund and shows ~HALF its volatility (7.3%
# vs 15.4%): the simulation is too smooth, its Sharpe (~1.4) is a mirage, and
# carry's real model/tail risk is NOT captured — use for long-run total return
# only and DISCOUNT the carry weight (target caps it near 5%).
_UEQC_BUNDLED = Path(__file__).resolve().parent / "ueqc_carry.csv.gz"
_UEQC_BUNDLED_CACHE: Optional[pd.Series] = None


def _bcom_bundled() -> Optional[pd.Series]:
    """Vendored BCOM daily index levels (read once)."""
    global _BCOM_BUNDLED_CACHE
    if _BCOM_BUNDLED_CACHE is None:
        if not _BCOM_BUNDLED.exists():
            return None
        df = pd.read_csv(_BCOM_BUNDLED, index_col="date", parse_dates=True)
        df.index = df.index.normalize()
        _BCOM_BUNDLED_CACHE = df["level"].sort_index()
    return _BCOM_BUNDLED_CACHE


def _ueqc_bundled() -> Optional[pd.Series]:
    """Vendored UEQC (UBS CMCI Commodity Carry 2.5x) daily total-return index
    levels (read once)."""
    global _UEQC_BUNDLED_CACHE
    if _UEQC_BUNDLED_CACHE is None:
        if not _UEQC_BUNDLED.exists():
            return None
        df = pd.read_csv(_UEQC_BUNDLED, index_col="date", parse_dates=True)
        df.index = df.index.normalize()
        _UEQC_BUNDLED_CACHE = df["level"].sort_index()
    return _UEQC_BUNDLED_CACHE


def _bcom_returns(fin: Optional[pd.Series]) -> pd.Series:
    """Broad-commodity TOTAL return from the vendored BCOM excess-return index
    (1991+) plus T-bill collateral. Self-contained/offline and pinned-run safe
    (clipped to as_of). Empty only if the bundle is missing."""
    lvl = _clip_cached_to_as_of(_bcom_bundled())
    if lvl is None or lvl.empty:
        return pd.Series(dtype=float)
    r = lvl.pct_change().dropna()
    return _apply_collateral(r, fin)            # excess → total return


def _ff_bundled() -> Optional[pd.DataFrame]:
    """The vendored Developed-factor series (already fractions), read once."""
    global _FF_BUNDLED_CACHE
    if _FF_BUNDLED_CACHE is None:
        if not _FF_BUNDLED.exists():
            return None
        df = pd.read_csv(_FF_BUNDLED, index_col="date", parse_dates=True)
        df.index = df.index.normalize()
        _FF_BUNDLED_CACHE = df.sort_index()
    return _FF_BUNDLED_CACHE


def _ff_merge(base: Optional[pd.DataFrame],
              extra: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Union two factor frames, preferring ``extra`` rows on overlap."""
    frames = [f for f in (base, extra) if f is not None and not f.empty]
    if not frames:
        return None
    merged = pd.concat(frames)
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def _ff_download(url: str, colnames: list[str]) -> Optional[pd.DataFrame]:
    """Download and parse one Ken French daily-factor CSV archive.

    This low-level entry point is fail-closed outside live mode; callers in a
    pinned run must consume the combined, as-of-clipped factor cache instead.
    """
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return None
    import io
    import re
    import urllib.request
    import zipfile
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=40).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    text = z.read(z.namelist()[0]).decode("latin-1")
    idx, rows = [], []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not re.fullmatch(r"\d{8}", parts[0] or ""):
            continue
        vals = parts[1:1 + len(colnames)]
        try:
            fv = [float(v) for v in vals]
        except ValueError:
            continue
        if len(fv) == len(colnames):
            idx.append(pd.Timestamp(parts[0]))
            rows.append(fv)
    if not idx:
        return None
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=colnames)


def factor_daily() -> Optional[pd.DataFrame]:
    """Daily DEVELOPED-markets factor-leg returns as FRACTIONS: columns ``SMB``
    (size), ``HML`` (value), ``RMW`` (profitability/quality) from the Developed
    5-factor set, plus ``MOM`` (momentum) — the dollar-neutral long-short legs
    spanning the common ETF tilts. Developed (not US) factors match the MSCI
    World indices our factor ETFs track, which is what recovers the momentum
    loading. The CMA (investment) leg is intentionally excluded: it is highly
    collinear with HML/RMW and not a standard ETF factor, so it only
    destabilises the fitted loadings. Shipped as a VENDORED snapshot in the
    package (self-contained, reproducible offline); a live Ken French fetch,
    when allowed and stale, only EXTENDS it with newer rows. Pinned runs use the
    bundled+cached rows at or before ``as_of``. None only if the bundle is
    missing AND nothing is cached (caller falls back to calibrated/naive)."""
    bundled = _ff_bundled()
    cached = _ff_merge(bundled, price_cache.load_history(_FF_CACHE_KEY))
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return _clip_cached_to_as_of(cached)
    if cached is not None and not cached.empty:
        last = pd.Timestamp(cached.index.max())
        if (pd.Timestamp.now().normalize() - last).days <= _STALE_DAYS:
            return cached
    try:
        ff = _ff_download(_FF_DEV5_URL, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
        mom = _ff_download(_FF_DEVMOM_URL, ["MOM"])
        if ff is None or ff.empty:
            return cached
        # SMB/HML/RMW/MOM are the tilt legs; MKT (Mkt-RF) and RF are kept as
        # clean market/riskfree CONTROLS for the loading regression (they let
        # the tilt loadings come out clean; they are NOT applied as a tilt).
        df = ff[["SMB", "HML", "RMW"]].copy()
        df["MKT"] = ff["Mkt-RF"]
        df["RF"] = ff["RF"]
        if mom is not None and not mom.empty:
            df = df.join(mom["MOM"], how="outer")
        df = (df / 100.0).sort_index()          # percent → fraction
        df.index = df.index.normalize()
        df = df[~df.index.duplicated(keep="last")].dropna(how="all")
        # Merge the fresh pull over the vendored base so the series only ever
        # grows, then persist the union.
        df = _ff_merge(bundled, df)
        if df is not None and not df.empty:
            price_cache.store_history(_FF_CACHE_KEY, df)
            return df
    except Exception as e:  # noqa: BLE001
        logger.warning("Ken French factor fetch failed: %s", e)
    return cached


def _ff_em_bundled() -> Optional[pd.DataFrame]:
    """The vendored Emerging-markets factor series (monthly, already fractions)."""
    global _FF_EM_BUNDLED_CACHE
    if _FF_EM_BUNDLED_CACHE is None:
        if not _FF_EM_BUNDLED.exists():
            return None
        df = pd.read_csv(_FF_EM_BUNDLED, index_col="date", parse_dates=True)
        df.index = df.index.normalize()
        _FF_EM_BUNDLED_CACHE = df.sort_index()
    return _FF_EM_BUNDLED_CACHE


def _ff_download_monthly(url: str, colnames: list[str]) -> Optional[pd.DataFrame]:
    """Download and parse one Ken French MONTHLY-factor CSV archive (YYYYMM
    dated rows), stamping each row at its month-end. Missing values (French's
    -99.99/-999 codes) become NaN. Fail-closed outside live mode."""
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return None
    import io
    import re
    import urllib.request
    import zipfile
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=40).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    text = z.read(z.namelist()[0]).decode("latin-1")
    idx, rows = [], []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not re.fullmatch(r"\d{6}", parts[0] or ""):
            continue
        vals = parts[1:1 + len(colnames)]
        try:
            fv = [float(v) for v in vals]
        except ValueError:
            continue
        if len(fv) == len(colnames):
            idx.append(pd.Period(parts[0], freq="M").to_timestamp("M"))
            rows.append(fv)
    if not idx:
        return None
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=colnames)
    return df.replace(-99.99, float("nan")).replace(-999.0, float("nan"))


def em_factor_monthly() -> Optional[pd.DataFrame]:
    """Monthly EMERGING-markets factor-leg returns as FRACTIONS: ``SMB`` (size),
    ``HML`` (value), ``RMW`` (profitability) plus ``MKT``/``RF`` controls, from
    the Ken French Emerging Markets 5-factor set (monthly-only; RMW-complete from
    1991-07). These are the correct regressors for an EM factor ETF; the Developed
    legs are not. Vendored snapshot (offline-reproducible); a live fetch, when
    allowed and stale, only EXTENDS it with newer month-ends. None if the bundle
    is missing and nothing is cached."""
    bundled = _ff_em_bundled()
    cached = _ff_merge(bundled, price_cache.load_history(_FF_EM_CACHE_KEY))
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return _clip_cached_to_as_of(cached)
    if cached is not None and not cached.empty:
        last = pd.Timestamp(cached.index.max())
        # Monthly series: only refetch when more than a couple of months stale.
        if (pd.Timestamp.now().normalize() - last).days <= max(_STALE_DAYS, 45):
            return cached
    try:
        em = _ff_download_monthly(_FF_EM5_URL,
                                  ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
        if em is None or em.empty:
            return cached
        df = em[["SMB", "HML", "RMW"]].copy()
        df["MKT"] = em["Mkt-RF"]
        df["RF"] = em["RF"]
        df = (df / 100.0).sort_index()
        df.index = df.index.normalize()
        df = df[~df.index.duplicated(keep="last")].dropna(subset=["SMB", "HML", "RMW"])
        df = _ff_merge(bundled, df)
        if df is not None and not df.empty:
            price_cache.store_history(_FF_EM_CACHE_KEY, df)
            return df
    except Exception as e:  # noqa: BLE001
        logger.warning("Ken French EM factor fetch failed: %s", e)
    return cached


def _apply_collateral(excess_ret: pd.Series, fin: Optional[pd.Series]) -> pd.Series:
    """Turn an EXCESS-return series (e.g. ^BCOM) into TOTAL return by adding the
    T-bill collateral yield: (1+r_tr) = (1+r_excess)·(1+r_cash)."""
    if fin is None or fin.empty or excess_ret is None or excess_ret.empty:
        return excess_ret
    coll = fin.reindex(excess_ret.index).ffill().fillna(0.0)
    return (1.0 + excess_ret) * (1.0 + coll) - 1.0


def proxy_returns_for(keys: set[str]) -> tuple[dict, Optional[pd.Series]]:
    """Return ({key: daily-return Series} for the requested buckets, financing).

    Records the concrete ticker + start date each bucket resolved to in
    ``USED_PROXY`` (for the per-instrument simulation report). Cash-like
    buckets map to the financing (T-bill) return.
    """
    fin = financing_daily()          # USD T-bill rate (kept USD as a cost proxy)
    ccy = "" if _TARGET_CCY == "USD" else " → EUR"
    out: dict[str, pd.Series] = {}
    for k in keys:
        if k == "Commodities":
            # Broad commodities from the VENDORED BCOM excess-return index + T-bill
            # collateral — robust to 1991 offline, unlike the flaky live ^BCOM feed.
            r = _bcom_returns(fin)
            if not r.empty:
                USED_PROXY[k] = (f"BCOM 1991+ vendored+coll{ccy}", r.index.min())
            out[k] = _usd_returns_to_eur(r)
        elif k in EQUITY_GEO_PROXY or k in ASSET_PROXY:
            candidates = EQUITY_GEO_PROXY.get(k) or ASSET_PROXY.get(k)
            r, sym = _resolve_bucket(candidates)
            if sym is not None and not r.empty:
                USED_PROXY[k] = (f"{sym}{ccy}", r.index.min())
            out[k] = _usd_returns_to_eur(r)
        elif k == "Alternative":
            s, label, start = _alt_series(fin)
            if s is not None and not s.empty:
                USED_PROXY[k] = (f"{label}{ccy}", start)
            out[k] = _usd_returns_to_eur(s)
        elif k == "Carry":
            out[k] = _carry_returns(fin)
            if not out[k].empty:
                src = ("UEQC 2.5x carry 2001+/cash" if _ueqc_bundled() is not None
                       else f"BNPIF73P carry 2008+/cash{ccy}")
                USED_PROXY[k] = (src, out[k].index.min())
        elif k in CASH_KEYS:
            out[k] = _usd_returns_to_eur(fin) if fin is not None else pd.Series(dtype=float)
            if fin is not None and not fin.empty:
                USED_PROXY[k] = (f"{_FINANCING_SYMBOL} (cash){ccy}", fin.index.min())
    # ``fin`` (the financing COST rate charged on leverage) is intentionally
    # returned in USD terms — the borrow is a USD short rate; EUR/USD short-rate
    # differences are a small second-order effect on the financed sleeve.
    return out, fin


def _alt_series(fin: Optional[pd.Series]):
    """Managed-futures returns for the Alternative bucket.

    Priority: an ad-hoc series in the cache DB (``MANUAL_MFSIM``, e.g. an SG
    CTA/Trend index ingested once) -> real trend funds spliced longest-first
    (RYMFX ~2007, overridden by AQMNX ~2010 where they overlap) -> cash for the
    pre-2007 tail -> cash-only fallback. Using RYMFX for 2007-2010 means the GFC
    crisis-alpha of managed futures is modeled instead of a flat cash line.

    The ad-hoc series is READ FROM THE CACHE DB only, never parsed from a file
    at launch. Populate it with:
    ``python -m tarzan.data.manual_proxies ingest MFSIM <path>``.
    """
    custom = _clip_cached_to_as_of(manual_proxies.get_series("MFSIM"))
    if custom is not None and not custom.empty:
        r = custom.pct_change().dropna()
        return r, "custom managed-futures series (cache)", r.index.min()

    # Real trend funds, deepest history first; later funds override in overlap.
    parts = [(sym, _returns(sym)) for sym in (_MF_FUND_DEEP, _MF_FUND)]
    parts = [(sym, r) for sym, r in parts if r is not None and not r.empty]
    if not parts:
        start = fin.index.min() if fin is not None and not fin.empty else None
        return fin, f"{_FINANCING_SYMBOL} (cash)", start

    trend = parts[0][1]
    used = [parts[0][0]]
    for sym, r in parts[1:]:
        trend = pd.concat([trend.loc[trend.index < r.index.min()], r]).sort_index()
        used.append(sym)
    tstart = trend.index.min()
    label = "/".join(used)
    if fin is None or fin.empty:
        return trend, f"{label} (managed futures)", tstart
    spliced = pd.concat([fin.loc[fin.index < tstart], trend]).sort_index()
    return spliced, f"{label} {tstart:%Y}+/cash", tstart


# ---------------------------------------------------------------------------
# Market state: today's starting point, for conditioning a forward simulation
# ---------------------------------------------------------------------------

_CAPE_BUNDLE = Path(__file__).with_name("shiller_cape.csv.gz")


def shiller_cape() -> Optional[pd.Series]:
    """Monthly US CAPE (Shiller's cyclically-adjusted P/E), vendored from 1900.

    Used only to price the STARTING POINT of a forward simulation, never to
    forecast a path. Snapshot, not live: Shiller publishes with a lag of a month
    or two, so callers should read ``market_state()["cape_asof"]`` rather than
    assume the last row is today.

    TO REFRESH: take the ``ie_data.xls`` link from the page source of
    https://shillerdata.com/ — it carries a per-release path segment and a
    ``?ver=`` stamp, and the directory-level URL without them serves a STALE
    file. That trap cost a run once: the older blob ended 23 months early at a
    CAPE of 35.2 against the then-current 41.2, which alone halved the equity
    valuation adjustment. Parse the ``Data`` sheet with ``skiprows=7``, keep
    ``Date`` and ``CAPE``, and store the trailing rows where CAPE is blank as
    absent rather than carrying the last value forward."""
    if not _CAPE_BUNDLE.exists():
        return None
    try:
        df = pd.read_csv(_CAPE_BUNDLE, index_col=0, parse_dates=True)
    except Exception:  # noqa: BLE001
        return None
    s = df.iloc[:, 0].astype(float).dropna()
    return s if not s.empty else None


def market_state(sample_start=None, sample_end=None) -> dict:
    """Today's state variables next to their average over the backtest sample.

    A backtest treats its own window as the future's distribution; that is only
    right if today looks like an average day in it. These are the three inputs
    where "today" is measurable and clearly does NOT: the short rate that prices
    leverage, the long yield that anchors bond returns, and the equity valuation
    that anchors equity returns.

    Returns raw levels in percent (plus the CAPE as a ratio), with no judgement
    applied — turning them into return assumptions is
    :func:`tarzan.engine.robustness.conditional_drift`'s job, and needs anchors
    a caller has to own. Missing inputs come back as ``None`` rather than a
    guess, so a partial state degrades a scenario instead of inventing one.
    """
    out: dict = {"short_rate_now": None, "short_rate_avg": None,
                 "long_yield_now": None, "long_yield_avg": None,
                 "cape_now": None, "cape_avg": None, "cape_asof": None}

    def _level(ticker: str):
        try:
            s = _fetch_max(ticker)
        except Exception:  # noqa: BLE001
            return None
        if s is None or getattr(s, "empty", True):
            return None
        s = s.dropna()
        return s if not s.empty else None

    for key, ticker in (("short_rate", "^IRX"), ("long_yield", "^TNX")):
        s = _level(ticker)
        if s is None:
            continue
        out[f"{key}_now"] = float(s.iloc[-1])
        win = s.loc[sample_start:sample_end] if (sample_start or sample_end) else s
        if not win.empty:
            out[f"{key}_avg"] = float(win.mean())

    cape = shiller_cape()
    if cape is not None:
        out["cape_now"] = float(cape.iloc[-1])
        out["cape_asof"] = cape.index[-1].date().isoformat()
        win = cape.loc[sample_start:sample_end] if (sample_start or sample_end) else cape
        if not win.empty:
            out["cape_avg"] = float(win.mean())
            out["cape_at_sample_start"] = float(win.iloc[0])
    return out
