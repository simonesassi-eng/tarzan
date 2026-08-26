"""A fitted factor loading drives 20+ years of synthetic history, so it must be
earned: significant AND reproducible across both halves of the sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tarzan.engine.synthetic import factor_loadings


def _factors(n: int, seed: int = 0) -> pd.DataFrame:
    """Daily factor legs (business days) with a MOM leg and MKT/RF controls."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame(
        {
            "MKT": rng.normal(0.0004, 0.010, n),
            "SMB": rng.normal(0.0, 0.004, n),
            "MOM": rng.normal(0.0, 0.005, n),
            "RF": np.full(n, 0.00004),
        },
        index=idx,
    )


def _fund(fac: pd.DataFrame, loads: dict, noise: float, seed: int = 1) -> pd.Series:
    """Fund daily returns built as MKT + Σ load·leg + idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    r = fac["RF"] + fac["MKT"]
    for leg, load in loads.items():
        r = r + load * fac[leg]
    return r + rng.normal(0.0, noise, len(fac))


def test_stable_leg_survives_and_absent_leg_is_dropped():
    fac = _factors(2000)
    real = _fund(fac, {"MOM": 0.30}, noise=0.0015)

    out = factor_loadings(real, real, fac)

    assert "MOM" in out, "a real, sample-wide tilt must survive both gates"
    assert abs(out["MOM"] - 0.30) < 0.10
    assert "SMB" not in out, "a leg the fund has no exposure to must be dropped"


def test_regime_switching_leg_is_dropped_not_extrapolated():
    """A loading present in only one half is noise for backfill purposes."""
    fac = _factors(2000)
    half = len(fac) // 2
    # SMB exposure exists ONLY in the second half; MOM is steady throughout.
    first = _fund(fac.iloc[:half], {"MOM": 0.30}, noise=0.0010, seed=2)
    second = _fund(fac.iloc[half:], {"MOM": 0.30, "SMB": 1.20}, noise=0.0010, seed=3)
    real = pd.concat([first, second])

    out = factor_loadings(real, real, fac)

    assert "MOM" in out, "the steady leg still survives"
    assert "SMB" not in out, "a leg that only shows up in one half must not backfill 20y"


def test_short_overlap_returns_nothing():
    fac = _factors(200)
    real = _fund(fac, {"MOM": 0.30}, noise=0.0015)

    assert factor_loadings(real, real, fac, min_overlap_months=24) == {}
