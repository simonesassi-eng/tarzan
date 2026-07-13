"""Delivery service: subject line + input resolution (the pure, testable core).

The SMTP send and full-pipeline run are exercised end-to-end elsewhere; here we
pin the two pieces with real branching logic — subject math and the
Drive-vs-local input resolution.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tarzan import delivery


def _metrics(total_value: float, cost_basis: float):
    """Minimal stand-in for PortfolioMetrics: only the fields build_subject reads."""
    df = pd.DataFrame({"cost_basis_eur": [cost_basis]}) if cost_basis else pd.DataFrame()
    return SimpleNamespace(total_value=total_value, holdings_df=df)


def test_build_subject_gain_and_loss(monkeypatch):
    # Pin the clock so the HH:MM segment is deterministic.
    fixed = delivery.datetime(2026, 7, 13, 19, 35, tzinfo=delivery.ZoneInfo("Europe/Rome"))
    monkeypatch.setattr(delivery, "now_local", lambda: fixed)

    up = delivery.build_subject(_metrics(106.68, 100.0), "Portfolio Digest")
    assert up == "Portfolio Digest - 19:35 - uP&L +6.68%"

    down = delivery.build_subject(_metrics(90.0, 100.0), "Portfolio Digest")
    assert down == "Portfolio Digest - 19:35 - uP&L −10.00%"  # note: minus sign U+2212


def test_build_subject_no_cost_basis_is_zero_pct(monkeypatch):
    fixed = delivery.datetime(2026, 7, 13, 9, 5, tzinfo=delivery.ZoneInfo("Europe/Rome"))
    monkeypatch.setattr(delivery, "now_local", lambda: fixed)
    # Empty holdings → cost 0 → guard yields 0.00% rather than dividing by zero.
    assert delivery.build_subject(_metrics(0.0, 0.0), "") == "Portfolio Digest - 09:05 - uP&L +0.00%"


def test_resolve_inputs_local(monkeypatch, tmp_path):
    orders = tmp_path / "order_list.csv"
    targets = tmp_path / "targets.csv"
    orders.write_text("date,type,isin,quantity,gross_eur,net_eur\n", encoding="utf-8")
    targets.write_text("key,value\n", encoding="utf-8")

    # No Drive creds → local mode; per-holding absent → None (not a failure).
    monkeypatch.delenv("DRIVE_FOLDER_ID", raising=False)
    monkeypatch.delenv("GOOGLE_DRIVE_CREDENTIALS_JSON", raising=False)
    monkeypatch.setenv("ORDERS_PATH", str(orders))
    monkeypatch.setenv("TARGETS_PATH", str(targets))
    monkeypatch.setenv("TARGETS_PER_HOLDING_PATH", str(tmp_path / "absent.csv"))

    resolved = delivery.resolve_inputs()
    assert resolved == {
        "config": str(targets),
        "orders": str(orders),
        "targets_per_holding": None,
    }


def test_resolve_inputs_local_missing_orders_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("DRIVE_FOLDER_ID", raising=False)
    monkeypatch.delenv("GOOGLE_DRIVE_CREDENTIALS_JSON", raising=False)
    monkeypatch.setenv("ORDERS_PATH", str(tmp_path / "nope.csv"))
    monkeypatch.setenv("TARGETS_PATH", str(tmp_path / "also_nope.csv"))
    with pytest.raises(FileNotFoundError):
        delivery.resolve_inputs()
