"""The digest carries the verdict of the independent return check.

``verify-returns.yml`` recomputes every printed return against a raw Yahoo pull, daily
and out of band. That closes the diagnosis gap but not the delivery one: it fails into
a workflow log, and the reader of the digest never opens one. So the send passes the
last verdict in through the environment and the header states it — a pass folded into
the data stamp, a failure as its own strip.

The rules that matter, and are checked here:

* absent verdict renders NOTHING, because most runs (local, pinned, the golden) are
  handed none, and a missing annotation must never become a claim in either direction;
* a pass never gets a banner — a badge that is always on is furniture;
* a failure always does, and says not to act on the figures;
* a green verdict that has gone stale says "last verified", because the scheduled check
  runs every weekday and GitHub disables schedules on an inactive repo, which would
  otherwise freeze a reassuring badge in place for good.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tarzan.export.newsletter import _sections_alloc as alloc


@pytest.fixture(autouse=True)
def _no_ambient_verdict(monkeypatch):
    """No test may inherit a verdict from the shell that started pytest."""
    for var in ("VERIFY_CONCLUSION", "VERIFY_AT", "VERIFY_URL"):
        monkeypatch.delenv(var, raising=False)


def _pin_today(monkeypatch, day: dt.date):
    from tarzan import runtime
    monkeypatch.setattr(runtime, "today", lambda: day)


class TestTheVerdictTheHeaderReports:
    def test_no_verdict_no_annotation(self):
        assert alloc._verify_verdict() is None

    def test_a_pass_is_stated_with_when(self, monkeypatch):
        _pin_today(monkeypatch, dt.date(2026, 9, 5))
        monkeypatch.setenv("VERIFY_CONCLUSION", "success")
        monkeypatch.setenv("VERIFY_AT", "2026-09-04T22:51:11Z")
        verdict = alloc._verify_verdict()
        assert verdict["ok"] is True
        assert verdict["text"].startswith("verified 04 Sep")

    def test_a_failure_says_so_loudly(self, monkeypatch):
        _pin_today(monkeypatch, dt.date(2026, 9, 5))
        monkeypatch.setenv("VERIFY_CONCLUSION", "failure")
        monkeypatch.setenv("VERIFY_AT", "2026-09-04T22:51:11Z")
        verdict = alloc._verify_verdict()
        assert verdict["ok"] is False
        assert "FAILED" in verdict["text"]

    def test_anything_that_is_not_success_is_a_failure(self, monkeypatch):
        """``cancelled``/``timed_out``/``startup_failure`` are not reassurance.

        The check did not pass, so the digest may not imply it did.
        """
        _pin_today(monkeypatch, dt.date(2026, 9, 5))
        for conclusion in ("cancelled", "timed_out", "startup_failure", "neutral"):
            monkeypatch.setenv("VERIFY_CONCLUSION", conclusion)
            assert alloc._verify_verdict()["ok"] is False, conclusion

    def test_a_stale_pass_does_not_claim_to_be_current(self, monkeypatch):
        # Nine days on from a weekday schedule means the schedule stopped running.
        _pin_today(monkeypatch, dt.date(2026, 9, 14))
        monkeypatch.setenv("VERIFY_CONCLUSION", "success")
        monkeypatch.setenv("VERIFY_AT", "2026-09-05T06:31:00Z")
        assert alloc._verify_verdict()["text"].startswith("last verified")

    def test_a_fresh_pass_is_not_hedged(self, monkeypatch):
        _pin_today(monkeypatch, dt.date(2026, 9, 7))
        monkeypatch.setenv("VERIFY_CONCLUSION", "success")
        monkeypatch.setenv("VERIFY_AT", "2026-09-05T06:31:00Z")
        assert alloc._verify_verdict()["text"].startswith("verified")

    def test_an_unparseable_timestamp_still_reports_the_verdict(self, monkeypatch):
        """The pass/fail is the signal; the time is decoration.

        GitHub changing its timestamp format must not silence the failure banner.
        """
        monkeypatch.setenv("VERIFY_CONCLUSION", "failure")
        monkeypatch.setenv("VERIFY_AT", "not a timestamp")
        verdict = alloc._verify_verdict()
        assert verdict["ok"] is False and "FAILED" in verdict["text"]


class TestWhatReachesTheReader:
    """End to end through the real template, on the synthetic golden portfolio.

    Reuses the numeric golden's dataset (the same one the markup golden renders) so
    this exercises the shipped template rather than a stand-in, and so it cannot pass
    against markup the digest does not actually use.
    """

    @pytest.fixture
    def analysis(self, tmp_path, monkeypatch):
        import pandas as pd

        from tarzan import orchestrator
        from tarzan.tests.test_golden_master import (
            _AS_OF, _ORDERS_CSV, _stub_enrich)

        monkeypatch.setattr("tarzan.data.enricher.enrich_holdings", _stub_enrich)
        monkeypatch.setattr("tarzan.engine.metrics._fetch_benchmark_history",
                            lambda *a, **k: pd.Series(dtype=float))
        orders = tmp_path / "order_list.csv"
        orders.write_text(_ORDERS_CSV)
        return orchestrator.run(
            config_source=None, orders_source=str(orders),
            targets_per_holding_source=None, deterministic=True, as_of=_AS_OF)

    def _render(self, analysis, monkeypatch, conclusion, at="2025-06-28T06:31:00Z"):
        from tarzan.export.newsletter import render_newsletter

        if conclusion:
            monkeypatch.setenv("VERIFY_CONCLUSION", conclusion)
            monkeypatch.setenv("VERIFY_AT", at)
            monkeypatch.setenv(
                "VERIFY_URL", "https://github.com/example/tarzan/actions/runs/1")
        metrics, config = analysis
        return render_newsletter(metrics=metrics, config=config,
                                 issue_number=31, ai_summary="")

    def test_a_failure_reaches_the_top_of_the_issue(self, analysis, monkeypatch):
        html = self._render(analysis, monkeypatch, "failure")
        assert "RETURN CHECK FAILED" in html
        assert "do not act on them" in html
        # Above the first section, not buried next to the version stamp.
        assert html.index("RETURN CHECK FAILED") < html.index("[01]")

    def test_a_pass_annotates_but_does_not_shout(self, analysis, monkeypatch):
        html = self._render(analysis, monkeypatch, "success")
        assert "verified 28 Jun" in html
        assert "RETURN CHECK FAILED" not in html

    def test_without_a_verdict_the_issue_says_nothing_about_it(self, analysis,
                                                               monkeypatch):
        """The case every local run, replay and golden render takes."""
        html = self._render(analysis, monkeypatch, None)
        assert "RETURN CHECK FAILED" not in html
        assert "verified" not in html
