"""Tarzan — CLI entry point.

Usage:
    python -m tarzan.main --input_orders input/order_list.csv
    python -m tarzan.main --input_orders input/order_list.csv --input_config input/targets.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import traceback
from datetime import date as _dtdate
from pathlib import Path

from tarzan import __version__
from tarzan.contracts.exceptions import DataIngestionError, TarzanError

logger = logging.getLogger("tarzan")


class _RecordCaptureHandler(logging.Handler):
    """Capture each log record's structured fields (level, time, logger,
    message) for the run, so the single report.html can render them as a
    color-coded table (like the reference DetailedRunLog). No file is written;
    this replaces analyzer.log entirely."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[dict] = []
        # A Formatter owns asctime formatting (Handler does not); use one to
        # turn each record's created-time into an HH:MM:SS string.
        self._fmt = logging.Formatter(datefmt="%H:%M:%S")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append({
                "level": record.levelname,
                "time": self._fmt.formatTime(record, "%H:%M:%S"),
                "origin": record.name,
                "message": record.getMessage(),
            })
        except Exception:  # noqa: BLE001 — logging must never break the run
            pass


# Process-global capture handler, (re)installed by setup_logging each run.
# The whole run's log is captured here (no separate analyzer.log file) and
# rendered as the color-coded table in the single output/report.html.
_LOG_CAPTURE = _RecordCaptureHandler()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tarzan CLI")
    parser.add_argument(
        "--input_orders", default="input/order_list.csv",
        help="Order-list CSV. The single source of truth: the snapshot is "
             "derived from it and it drives the historical series + XIRR/TWROR.",
    )
    parser.add_argument("--input_config", default="input/targets.csv")
    parser.add_argument(
        "--input_targets_per_holding", default="input/targets_per_holding.csv",
        help="Optional per-holding rebalancing targets (by ISIN), attached "
             "to the order-derived snapshot.",
    )
    parser.add_argument("--output", default="output/")
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Reproducible run: pin the clock, skip live intraday quotes and "
             "the AI summary, so the same inputs produce the same output "
             "(golden-testable / offline-reproducible on a warm cache).",
    )
    parser.add_argument(
        "--as_of", default=None, metavar="YYYY-MM-DD",
        help="Value the portfolio as of this date instead of today (pins the "
             "terminal valuation date for XIRR/TWROR and the daily series). "
             "Implies a pinned clock.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Strict input validation: reject an order list with unrecognized "
             "columns (with an actionable error) instead of tolerating them. "
             "Default off keeps the lenient behavior. Also settable via "
             "TARZAN_STRICT_INPUT=1.",
    )
    return parser.parse_args(argv)


def _dated_output_dir(base: str, as_of=None) -> str:
    """Return ``base/<YYYY-MM-DD>`` so each run's artifacts are grouped by date
    instead of piling up in one flat directory.

    The date is the pinned ``as_of`` when given (so an as-of/deterministic run
    is reproducible into a stable folder), else today's date. Idempotent: if
    ``base`` already ends in a YYYY-MM-DD segment (a caller that pre-namespaced,
    or a re-run), it is returned unchanged so we never nest date/date.
    """
    import datetime as _dt
    import re as _re
    base = base.rstrip("/") or "output"
    if _re.search(r"\d{4}-\d{2}-\d{2}$", os.path.basename(base)):
        return base
    day = (as_of or _dt.date.today()).strftime("%Y-%m-%d")
    return os.path.join(base, day)


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Live console output (as before).
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(ch)
    # Full DEBUG trace captured as structured records (no separate
    # analyzer.log file) and rendered as the color-coded table in report.html.
    _LOG_CAPTURE.records.clear()
    if _LOG_CAPTURE not in root.handlers:
        root.addHandler(_LOG_CAPTURE)


def _write_run_reports(
    output_dir: str,
    metrics=None,
    config=None,
    *,
    newsletter_html: str | None = None,
    what_if=None,
) -> None:
    """Finalize one correlated local artifact set for every initialized run."""
    from tarzan.runtime import data_quality as dq
    from tarzan.runtime import report_html
    from tarzan import runtime
    from tarzan.runtime.artifacts import LocalArtifactWriter, StorageDescriptor
    from tarzan.runtime.ledger import LedgerEntryType
    from tarzan.runtime.publication import PublicationEvaluator
    from tarzan.runtime.session import last_run_result
    from tarzan.runtime.summary import SummaryProjector

    try:
        logger.info(dq.summary_line())
        result = last_run_result()
        stamp = runtime.now_stamp("%Y-%m-%d %H:%M")
        if result is None:
            # Argument parsing can fail before a RunSession exists. Keep the
            # legacy atomic report fallback for that pre-initialization case.
            path = report_html.write_report(
                output_dir,
                generated_at=stamp,
                log_records=_LOG_CAPTURE.records,
            )
            if path:
                logger.info("Pre-run report: %s", path)
            return

        publication = PublicationEvaluator.evaluate(result.ledger.failure_records())
        summary = SummaryProjector.project(result, publication)
        storage = StorageDescriptor(
            storage_scope="local",
            automation_local_ephemeral=False,
            retention_guarantee="none",
            execution_environment="cli",
        )
        rendered_report = report_html.render(
            generated_at=stamp,
            log_records=_LOG_CAPTURE.records,
            ledger=result.ledger,
            publication_state=publication.decision.value,
            storage_scope=storage.storage_scope,
            automation_local_ephemeral=storage.automation_local_ephemeral,
            retention_guarantee=storage.retention_guarantee,
        )
        result.ledger.append(LedgerEntryType.ARTIFACT, {
            "artifact_set": "local",
            "state": "FINALIZATION_REQUESTED",
            "storage_scope": storage.storage_scope,
            "automation_local_ephemeral": storage.automation_local_ephemeral,
            "retention_guarantee": storage.retention_guarantee,
        })
        writer = LocalArtifactWriter(
            Path(output_dir),
            result.attempt_id,
            storage=storage,
        )
        manifest = writer.finalize(
            analysis_id=result.analysis_id,
            summary=summary.to_dict(),
            ledger_entries=(entry.to_dict() for entry in result.ledger.entries),
            report_html=rendered_report,
            publication_state=publication.decision.value,
            newsletter_html=newsletter_html,
            what_if=what_if,
        )
        logger.info(
            "Local artifact manifest: %s (%d data-quality issue(s))",
            manifest,
            len(dq.issues()),
        )
    except Exception as error:  # noqa: BLE001
        result = last_run_result()
        if result is not None:
            result.ledger.open_failure(
                stage="local_artifacts",
                stable_code="LOCAL_ARTIFACT_FINALIZATION_FAILED",
                severity="ERROR",
                error=error,
                affected_outputs=["manifest", "summary", "ledger", "report"],
                analytical_impact="analysis remains in memory but local evidence is incomplete",
                publication_impact="DEGRADE",
            )
        logger.error(
            "Local artifact finalization failed (%s); normalized evidence remains in memory.",
            type(error).__name__,
        )


def _export_whatif(portfolios, config):
    """Render What-If to bytes; only LocalArtifactWriter can publish it."""
    try:
        from tarzan.backtest import (
            simulation_rows, testfol_instrument_map, testfol_lines,
        )
        from tarzan.export.whatif_excel import export_whatif_excel
        from tarzan.runtime.artifacts import LocalOnlyWorkbook

        with tempfile.TemporaryDirectory(prefix="tarzan-what-if-") as directory:
            out = Path(directory) / "what_if.xlsx"
            export_whatif_excel(
                str(out), portfolios,
                config.invested_allocation_targets_pctg or {},
                config.equity_geo_targets_pctg or {},
                100_000.0,
                tolerance=config.rebalancing_target_tolerance_pctg,
                sim_rows=simulation_rows(portfolios),
                testfol={p.name: testfol_lines(p) for p in portfolios},
                testfol_byinst={p.name: testfol_instrument_map(p) for p in portfolios},
            )
            workbook = LocalOnlyWorkbook(content=out.read_bytes())
        logger.info("What-If workbook rendered for local-only finalization.")
        return workbook
    except Exception as error:  # noqa: BLE001
        logger.warning("What-If Excel skipped (%s): %s", type(error).__name__, error)
        return None


def main(argv=None) -> int:
    args = parse_args(argv)

    from tarzan.runtime import data_quality as dq
    # Parse the optional as-of date FIRST (before logging is set up) — it is a
    # hard, actionable error if malformed, and it also names the output folder.
    as_of = None
    if args.as_of:
        try:
            as_of = _dtdate.fromisoformat(args.as_of)
        except ValueError:
            # Logging isn't configured yet here; set up on the base dir so the
            # error is still captured in a report.
            setup_logging(args.output)
            logger.error("Invalid --as_of %r (expected YYYY-MM-DD)", args.as_of)
            return 1

    # Organize outputs per run date: everything for a run lands in
    # output/<YYYY-MM-DD>/ (the as-of date when pinned, else today), so runs
    # don't pile up in one flat directory. If --output already ends in a
    # date-looking folder it's used as-is (idempotent for re-runs / callers
    # that pre-namespace).
    output_dir = _dated_output_dir(args.output, as_of)
    setup_logging(output_dir)
    logger.info("Tarzan v%s starting...", __version__)
    logger.info("Output directory: %s", output_dir)

    if args.deterministic or as_of is not None:
        logger.info("Deterministic run%s (live quotes + AI summary skipped).",
                    f" as of {as_of}" if as_of else "")
    strict = args.strict or os.environ.get("TARZAN_STRICT_INPUT", "").strip() in ("1", "true", "yes")
    if strict:
        logger.info("Strict input validation ON (unrecognized columns rejected).")
    metrics = None
    config = None
    newsletter_html = None
    what_if = None
    try:
        from tarzan.orchestrator import run
        metrics, config = run(
            config_source=args.input_config,
            orders_source=args.input_orders,
            targets_per_holding_source=args.input_targets_per_holding,
            deterministic=args.deterministic,
            as_of=as_of,
            strict=strict,
        )
        from tarzan.runtime.ledger import LedgerEntryType
        from tarzan.runtime.publication import PublicationDecision, PublicationEvaluator
        from tarzan.runtime.session import last_run_result

        result = last_run_result()
        if result is None:
            raise RuntimeError("orchestrator completed without a RunResult")
        publication = PublicationEvaluator.evaluate(result.ledger.failure_records())
        result.ledger.append(LedgerEntryType.PUBLICATION, {
            "decision": publication.decision.value,
            "delivery_purpose": publication.delivery_purpose.value,
            "critical_failure_refs": list(publication.critical_failure_ids),
        })
        # The publication decision is the ONLY thing that may set a failing exit
        # code. A zero total is not evidence of failure — a fully liquidated book's
        # true total IS zero — and the run's real failures are on the ledger, which
        # is what the decision reads. The deleted proxy manufactured a
        # disagreement: it exited 1 while the decision beside it said SEND_NORMAL
        # and no issue had been written, so three parts of one run each reported a
        # different outcome for the same state.
        if publication.decision is PublicationDecision.BLOCK_NORMAL_AND_NOTIFY_FAILURE:
            logger.critical(
                "Normal newsletter blocked by critical run evidence: %s",
                ", ".join(publication.critical_failure_ids),
            )
            return 1

        # Long-history backtest of the candidate portfolios — for the local-only
        # What-If workbook. The newsletter does not carry a backtest section:
        # comparing portfolios the reader does not hold took 28% of the issue,
        # and a reader who wants that comparison opens the workbook.
        from tarzan.backtest import newsletter_portfolios
        backtest_portfolios = newsletter_portfolios(
            deterministic=(args.deterministic or as_of is not None))

        # Render in memory. LocalArtifactWriter is the only file publisher for
        # newsletter, report, summary, ledger, manifest, and What-If evidence.
        from tarzan.export.newsletter import render_newsletter
        newsletter_html = render_newsletter(
            metrics=metrics,
            config=config,
        )

        if backtest_portfolios:
            what_if = _export_whatif(backtest_portfolios, config)

        logger.info("Completed successfully.")
        return 0

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except (DataIngestionError, ValueError) as e:
        logger.error("Validation error: %s", e)
        return 1
    except TarzanError as e:
        logger.error("Analysis error: %s", e)
        return 1
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        logger.debug(traceback.format_exc())
        return 1
    finally:
        # Always write the unified run report (run summary + data quality) —
        # on success, on the no-value early exit, and on any failure — so the
        # user always has one readable record of what the run produced and
        # what it skipped, regardless of how the run ended.
        _write_run_reports(
            output_dir,
            metrics=metrics,
            config=config,
            newsletter_html=newsletter_html,
            what_if=what_if,
        )


if __name__ == "__main__":
    sys.exit(main())
