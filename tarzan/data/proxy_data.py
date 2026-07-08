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
from typing import Optional

import pandas as pd

from tarzan.data import price_cache
from tarzan.engine.stats import TRADING_DAYS

logger = logging.getLogger(__name__)

# Each bucket resolves to the FIRST candidate that yfinance prices, ordered
# longest-history first (mutual funds reach further back than ETFs, à la
# testfol.io's SIM backfills) → enables a ~2000 start.
EQUITY_GEO_PROXY = {
    "USA": ["^GSPC"],                         # S&P 500, 1927+
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
# Managed-futures / trend proxy for the "Alternative" bucket: real trend
# fund (AQR Managed Futures, daily) from its 2010 inception, spliced onto
# cash before then (so it captures 2022 crisis-alpha without truncating the
# window). Replace/extend by dropping input/managed_futures.csv (see loader).
_MF_FUND = "AQMNX"

_returns_memo: dict[str, pd.Series] = {}
# Which concrete ticker each bucket resolved to, and from when.
USED_PROXY: dict[str, tuple[str, pd.Timestamp]] = {}


def _fetch_max(symbol: str) -> pd.Series:
    """Full-history daily close for ``symbol`` (disk-cached, namespaced)."""
    key = f"PROXYMAX_{symbol}"
    cached = price_cache.load_history(key)
    if cached is not None and not cached.empty:
        return cached
    try:
        import yfinance as yf
        h = yf.Ticker(symbol).history(period="max")
    except Exception as e:  # noqa: BLE001
        logger.warning("Proxy fetch failed for %s: %s", symbol, e)
        return pd.Series(dtype=float)
    if h is None or h.empty or "Close" not in h:
        return pd.Series(dtype=float)
    s = h["Close"].dropna()
    idx = s.index
    s.index = (idx.tz_convert("UTC").tz_localize(None).normalize()
               if getattr(idx, "tz", None) is not None else idx.normalize())
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if not s.empty:
        price_cache.store_history(key, s)
    return s


def _returns(symbol: str) -> pd.Series:
    if symbol in _returns_memo:
        return _returns_memo[symbol]
    px = _fetch_max(symbol)
    r = px.pct_change().dropna() if not px.empty else pd.Series(dtype=float)
    _returns_memo[symbol] = r
    return r


def financing_daily() -> Optional[pd.Series]:
    """Daily financing rate (fraction) from ^IRX (annualised % → /100 /252)."""
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


def proxy_returns_for(keys: set[str]) -> tuple[dict, Optional[pd.Series]]:
    """Return ({key: daily-return Series} for the requested buckets, financing).

    Records the concrete ticker + start date each bucket resolved to in
    ``USED_PROXY`` (for the per-instrument simulation report). Cash-like
    buckets map to the financing (T-bill) return.
    """
    fin = financing_daily()
    out: dict[str, pd.Series] = {}
    for k in keys:
        if k in EQUITY_GEO_PROXY or k in ASSET_PROXY:
            candidates = EQUITY_GEO_PROXY.get(k) or ASSET_PROXY.get(k)
            r, sym = _resolve_bucket(candidates)
            out[k] = r
            if sym is not None and not r.empty:
                USED_PROXY[k] = (sym, r.index.min())
        elif k == "Alternative":
            s, label, start = _alt_series(fin)
            out[k] = s
            if s is not None and not s.empty:
                USED_PROXY[k] = (label, start)
        elif k in CASH_KEYS:
            out[k] = fin if fin is not None else pd.Series(dtype=float)
            if fin is not None and not fin.empty:
                USED_PROXY[k] = (f"{_FINANCING_SYMBOL} (cash)", fin.index.min())
    return out, fin


def _custom_mf_returns() -> Optional[pd.Series]:
    """Optional user-supplied managed-futures/trend series from
    ``input/managed_futures.csv`` (columns: date + level/close/value or
    return/ret). Lets you plug in a full-history series (AQR TSMOM, SG Trend)
    à la testfol's custom tickers. Returns daily-ish returns, or None."""
    import os
    path = os.path.join("input", "managed_futures.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c), df.columns[0])
        idx = pd.to_datetime(df[date_col]).dt.normalize()
        lvl = next((c for c in df.columns if c in ("level", "close", "value", "nav", "price")), None)
        ret = next((c for c in df.columns if c in ("return", "ret", "monthly_return")), None)
        if lvl is not None:
            s = pd.Series(pd.to_numeric(df[lvl], errors="coerce").values, index=idx).dropna()
            return s.pct_change().dropna()
        if ret is not None:
            r = pd.to_numeric(df[ret], errors="coerce")
            r = r / 100.0 if r.abs().median() > 1 else r  # percent → fraction
            return pd.Series(r.values, index=idx).dropna()
    except Exception as e:  # noqa: BLE001
        logger.warning("Custom managed_futures.csv unreadable: %s", e)
    return None


def _alt_series(fin: Optional[pd.Series]):
    """Managed-futures returns for the Alternative bucket: custom CSV if
    provided, else AQMNX (2010+) spliced onto cash, else cash."""
    custom = _custom_mf_returns()
    if custom is not None and not custom.empty:
        return custom, "custom managed-futures series", custom.index.min()
    aq = _returns(_MF_FUND)
    if aq is None or aq.empty:
        start = fin.index.min() if fin is not None and not fin.empty else None
        return fin, f"{_FINANCING_SYMBOL} (cash)", start
    if fin is None or fin.empty:
        return aq, f"{_MF_FUND} (managed futures)", aq.index.min()
    start = aq.index.min()
    spliced = pd.concat([fin.loc[fin.index < start], aq]).sort_index()
    return spliced, f"{_MF_FUND} {start:%Y}+/cash", start


def reset_memo() -> None:
    _returns_memo.clear()
