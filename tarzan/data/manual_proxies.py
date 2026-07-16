"""Ad-hoc historical-series inputs, stored in Tarzan's own cache DB.

Some exposures (commodity carry, certain CTA/trend indices) have NO yfinance
ticker and are only published as downloadable files behind a login/licence, so
they cannot be fetched live. Rather than read such files on every run, Tarzan
keeps them in its on-disk cache DB (``price_cache``, namespaced ``MANUAL_<key>``)
and the script reads ONLY from that cache at launch.

Two clearly separated paths:

  * ``get_series(key)`` — the RUNTIME read. Cache-only. It never opens a source
    file, so a normal ``scripts.whatif`` run has no file-ingestion routine at
    all. A missing key simply returns None and the caller falls back to its
    generic proxy.

  * ``ingest(key, path)`` — the AD-HOC write, run by hand (CLI below), NOT at
    launch. It parses a source file once and stores the daily LEVEL series in
    the cache DB. Historical index data is immutable, so this is a one-off per
    series; the recent period is taken over by the real fund's own returns
    through the per-instrument splice downstream.

Populate the DB ad-hoc, e.g.::

    python -m tarzan.data.manual_proxies ingest CRRYSIM path/to/US_BNPIF73P.xlsx
    python -m tarzan.data.manual_proxies ingest MFSIM   path/to/sg_cta_index.csv
    python -m tarzan.data.manual_proxies list

The raw index level is cached; excess→total-return collateral, fees and
currency conversion are applied downstream in ``proxy_data`` so the cache stays
a faithful copy of the source.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tarzan.data import price_cache

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "MANUAL_"

# Convenience presets: key → (default filename, parser). Only used by the
# ad-hoc ``ingest`` step when no explicit path is given — NEVER at launch.
# CRRYSIM = BNP Paribas Enhanced Commodity Carry ER index (CRRY's benchmark).
# NHCTA   = NilssonHedge CTA index (monthly). MFSIM = a managed-futures/trend
# index (e.g. SG CTA/Trend) supplied ad-hoc via an explicit path.
_ROOT = Path(__file__).resolve().parents[2]
_DIR = _ROOT / "tbtf-analisi"
_SOURCES = {
    "CRRYSIM": ("US_BNPIF73P.xlsx", "bnp"),
    "NHCTA": ("NHIndexMonthly.csv", "nhcta"),
}


# ---------------------------------------------------------------------------
# Runtime read (cache-only — no file access)
# ---------------------------------------------------------------------------

def get_series(key: str) -> Optional[pd.Series]:
    """Daily LEVEL series for an ad-hoc-ingested proxy, read ONLY from the
    cache DB (``MANUAL_<key>``). Never parses a source file, so it adds no
    file-ingestion routine to a normal run. Returns None if the key has not
    been ingested (the caller then falls back to its generic proxy)."""
    cached = price_cache.load_history(f"{_CACHE_PREFIX}{key}")
    return cached if (cached is not None and not cached.empty) else None


def list_cached() -> list[str]:
    """Keys currently present in the cache DB (for the CLI ``list`` command)."""
    hist = price_cache.cache_dir() / "history"
    if not hist.exists():
        return []
    out = []
    for p in sorted(hist.glob(f"{_CACHE_PREFIX}*.pkl")):
        out.append(p.stem[len(_CACHE_PREFIX):])
    return out


# ---------------------------------------------------------------------------
# Parsers (used only by the ad-hoc ingest step)
# ---------------------------------------------------------------------------

def _monthly_ret_to_daily_levels(ret: pd.Series) -> pd.Series:
    """Spread monthly returns onto a business-day grid (log-linear on the NAV)
    and return a daily LEVEL series (start 100)."""
    nav = (1.0 + ret).cumprod()
    bidx = pd.bdate_range(nav.index.min(), nav.index.max())
    lognav = (np.log(nav).reindex(nav.index.union(bidx))
              .interpolate("time").reindex(bidx))
    daily_ret = np.exp(lognav).pct_change().dropna()
    return 100.0 * (1.0 + daily_ret).cumprod()


def _looks_monthly(idx: pd.DatetimeIndex) -> bool:
    """True if observations are ~monthly (median gap > 20 days)."""
    if len(idx) < 3:
        return False
    gaps = pd.Series(idx.sort_values()).diff().dropna().dt.days
    return bool(gaps.median() > 20)


def _parse_nhcta(path: Path) -> Optional[pd.Series]:
    """NilssonHedge monthly index CSV → daily CTA level series."""
    df = pd.read_csv(path)
    if "type" not in df.columns or "ror" not in df.columns:
        return None
    cta = df[df["type"] == "CTA"].copy()
    if cta.empty:
        return None
    cta["date"] = pd.to_datetime(cta["date"], errors="coerce").dt.normalize()
    s = pd.Series(pd.to_numeric(cta["ror"], errors="coerce").values,
                  index=cta["date"]).dropna().sort_index()
    return _monthly_ret_to_daily_levels(s)


def _parse_bnp(path: Path) -> Optional[pd.Series]:
    """BNP index workbook → daily excess-return LEVEL series (raw, USD)."""
    raw = pd.read_excel(path, sheet_name=0)
    key_col = raw.columns[0]
    val_col = raw.columns[1]
    mask = raw[key_col].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")
    dat = raw.loc[mask, [key_col, val_col]].copy()
    idx = pd.to_datetime(dat[key_col], errors="coerce").dt.normalize()
    lvl = pd.Series(pd.to_numeric(dat[val_col], errors="coerce").values,
                    index=idx).dropna().sort_index()
    lvl = lvl[~lvl.index.duplicated(keep="last")]
    return lvl if not lvl.empty else None


def _parse_generic(path: Path) -> Optional[pd.Series]:
    """Auto-detect a date + level/return column from a CSV/XLSX and return a
    daily LEVEL series. Level columns are used as-is; return columns are
    cumulated (monthly returns are first spread onto a daily grid)."""
    df = (pd.read_excel(path, sheet_name=0)
          if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path))
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c), df.columns[0])
    idx = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    lvl_c = next((c for c in df.columns
                  if c in ("level", "close", "value", "nav", "price", "index")), None)
    ret_c = next((c for c in df.columns
                  if c in ("return", "ret", "monthly_return", "ror")), None)
    if lvl_c is not None:
        s = pd.Series(pd.to_numeric(df[lvl_c], errors="coerce").values,
                      index=idx).dropna().sort_index()
        s = s[~s.index.duplicated(keep="last")]
        return s if not s.empty else None
    if ret_c is not None:
        r = pd.to_numeric(df[ret_c], errors="coerce")
        r = r / 100.0 if r.abs().median() > 1 else r      # percent → fraction
        r = pd.Series(r.values, index=idx).dropna().sort_index()
        r = r[~r.index.duplicated(keep="last")]
        if r.empty:
            return None
        if _looks_monthly(r.index):
            return _monthly_ret_to_daily_levels(r)
        return 100.0 * (1.0 + r).cumprod()
    logger.error("No date+level/return columns found in %s", path)
    return None


_PARSERS = {"bnp": _parse_bnp, "nhcta": _parse_nhcta, "generic": _parse_generic}


# ---------------------------------------------------------------------------
# Ad-hoc write (run by hand, never at launch)
# ---------------------------------------------------------------------------

def ingest(key: str, path: Optional[str] = None) -> Optional[pd.Series]:
    """Parse a source file into a daily LEVEL series and STORE it in the cache
    DB under ``MANUAL_<key>``. Ad-hoc tool — run manually, NOT at script launch.

    ``path`` points at the source file (CSV/XLSX, auto-detected). If omitted,
    the preset filename for ``key`` in ``_SOURCES`` (under ``tbtf-analisi/``) is
    used with its dedicated parser. Returns the stored series (or None)."""
    if path is not None:
        p = Path(path).expanduser()
        # Use the dedicated parser for a known key (e.g. the BNP xlsx layout),
        # even when the file comes from an explicit path (e.g. downloaded from
        # the private Drive in CI); fall back to the generic parser otherwise.
        src = _SOURCES.get(key)
        parser = _PARSERS[src[1]] if src else _PARSERS["generic"]
    else:
        src = _SOURCES.get(key)
        if not src:
            logger.error("No preset source for key '%s'; pass an explicit path.", key)
            return None
        p = _DIR / src[0]
        parser = _PARSERS[src[1]]
    if not p.exists():
        logger.error("Source file not found: %s", p)
        return None
    try:
        s = parser(p)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to parse '%s' from %s: %s", key, p, e)
        return None
    if s is None or s.empty:
        logger.error("Parsed no usable data from %s", p)
        return None
    price_cache.store_history(f"{_CACHE_PREFIX}{key}", s)
    logger.info("Ingested '%s': %d rows %s→%s into the cache DB",
                key, len(s), s.index.min().date(), s.index.max().date())
    return s


def _main(argv=None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description="Ad-hoc: ingest a historical series into Tarzan's cache DB "
                    "(run by hand; the analysis script never reads source files).")
    sub = ap.add_subparsers(dest="cmd")
    ing = sub.add_parser("ingest", help="parse a source file into the cache DB")
    ing.add_argument("key", help="cache key, e.g. CRRYSIM, NHCTA, MFSIM")
    ing.add_argument("path", nargs="?", help="source file (csv/xlsx); omit to use a preset")
    sub.add_parser("list", help="list ingested series in the cache DB")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        keys = list_cached()
        if not keys:
            print("No manual series in the cache DB yet.")
        else:
            print("Ingested series (cache DB):")
            for k in keys:
                s = get_series(k)
                span = (f"{s.index.min().date()}→{s.index.max().date()} ({len(s)} rows)"
                        if s is not None else "unreadable")
                print(f"  {k:12s} {span}")
        return 0
    if args.cmd == "ingest":
        s = ingest(args.key, args.path)
        return 0 if s is not None else 1
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
