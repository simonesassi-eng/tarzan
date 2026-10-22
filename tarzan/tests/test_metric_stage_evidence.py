"""Metric-stage timing is observational and never converts an abort to success."""

from datetime import datetime, timezone

import pytest

from tarzan.engine.metrics import MetricsEngine
from tarzan.models.investor_config import InvestorConfig
from tarzan.runtime.ledger import LedgerEntryType, RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session


@pytest.mark.parametrize("outcome", ["SUCCEEDED", "FAILED", "INTERRUPTED"])
def test_stage_outcome_matches_completion_failure_or_unswallowed_cancellation(outcome):
    class Cancelled(BaseException):
        pass

    def stage(context):
        if outcome == "FAILED":
            raise ValueError("synthetic metric-stage failure")
        if outcome == "INTERRUPTED":
            raise Cancelled()

    context = RunContext("synthetic-stage", RunMode.LIVE, None, datetime.now(timezone.utc))
    ledger = RunLedger(context.attempt_id)
    session = RunSession(context, {}, ledger)
    engine = MetricsEngine([], InvestorConfig())
    engine._computers = [stage]
    with activate_session(session):
        if outcome == "INTERRUPTED":
            with pytest.raises(Cancelled):
                engine.compute_all()
        else:
            engine.compute_all()
    evidence = [
        entry for entry in ledger.entries
        if entry.entry_type is LedgerEntryType.STAGE and entry.payload.get("stage") == "metrics:stage"
    ]
    assert len(evidence) == 1
    assert evidence[0].payload["outcome"] == outcome
    assert evidence[0].payload["duration_ms"] >= 0
