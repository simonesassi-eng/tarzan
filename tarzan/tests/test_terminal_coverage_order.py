"""Coverage is a deterministic share, not a hash-seed-dependent disclosure."""

from datetime import date
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from tarzan.engine.returns_builder import build_order_derived_series
from tarzan.instruments.registry import InstrumentKind
from tarzan.models.holding import AssetClass, Holding
from tarzan.models.order import Order, OrderType
from tarzan.runtime import data_quality as dq


def synthetic_book(*, partial=False):
    start, end = date(2026, 10, 20), date(2026, 10, 23)
    orders, holdings = [], {}
    # The large constituent is deliberately last in the actual valuation
    # order. A provenance SET may place it first and lose the smaller values.
    values = [1.0, 1.0, 1.0, 1e16] if not partial else [100.0, 100.0]
    for index, price in enumerate(values):
        isin = f"XX{index:010d}"
        orders.append(Order(
            date=start, trade_date=start, type=OrderType.BUY, isin=isin,
            name=f"Synthetic stock {index}", ticker=f"SYNTHETIC-{index}",
            quantity=1.0, currency="EUR", price_native=price,
            fx_rate=1.0, gross_eur=price, fees_eur=0.0, net_eur=-price,
            source="synthetic-fixture", instrument_kind=InstrumentKind.STOCK,
        ))
        history = pd.Series(
            price, index=pd.date_range(start, end), name=f"SYNTHETIC-{index}",
        )
        holdings[isin] = Holding(
            isin=isin, ticker=f"SYNTHETIC-{index}", quantity=1.0,
            cost_basis_eur=price, market_value_eur=price, current_value=price,
            current_price=price, currency="EUR", security_type="STOCK",
            asset_class=AssetClass.EQUITIES,
            price_history=None if partial and index == 1 else history,
            price_history_price_only=None if partial and index == 1 else history.copy(),
        )
    return build_order_derived_series(orders, holdings, today=end)


def test_full_market_coverage_is_exact_and_does_not_invent_fallback_disclosure():
    dq.reset()
    result = synthetic_book()
    assert result.coverage_pct == 100.0
    assert not any("remainder used" in issue.message for issue in dq.issues())


def test_partial_coverage_is_not_promoted_by_rounding_or_epsilon():
    result = synthetic_book(partial=True)
    assert result.coverage_pct == 50.0
    assert result.provenance["synthetic"] or result.provenance["carry_flat"]


@pytest.mark.parametrize("seed", ["0", "1", "2", "7", "42", "99"])
def test_real_builder_coverage_is_identical_across_interpreter_hash_seeds(seed):
    root = Path(__file__).resolve().parents[2]
    environment = {
        name: value for name, value in os.environ.items()
        if name in {
            "PATH", "SYSTEMROOT", "WINDIR", "LANG", "LC_ALL", "TZ",
            "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
        }
    }
    environment.update(
        PYTHONHASHSEED=seed, TARZAN_DISABLE_CACHE="1", TARZAN_DISABLE_AI="1",
        PYTHONPATH=str(root), PYTHONNOUSERSITE="1",
    )
    script = (
        "import json;"
        "from tarzan.tests.test_terminal_coverage_order import synthetic_book;"
        "from tarzan.runtime import data_quality as dq;"
        "dq.reset();s=synthetic_book();"
        "print(json.dumps({'coverage':s.coverage_pct,"
        "'false_disclosure':any('remainder used' in i.message for i in dq.issues())}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root, env=environment,
        check=True, capture_output=True, text=True, timeout=15,
    )
    assert json.loads(result.stdout) == {"coverage": 100.0, "false_disclosure": False}
