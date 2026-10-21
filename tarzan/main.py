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
from tarzan.runtime.artifacts import ArtifactFinalizationOutcome
from tarzan.runtime.logging import LOG_CAPTURE as _LOG_CAPTURE

logger = logging.getLogger("tarzan")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tarzan CLI")
    parser.add_argument(
        "--input_orders", default="input/order_list.csv",
        help="Order-list CSV. The single source of truth: the snapshot is "
             "derived from it and it drives the historical series + XIRR/TWR.",
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
             "terminal valuation date for XIRR/TWR and the daily series). "
             "Implies a pinned clock.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Strict input validation: reject an order list with unrecognized "
             "columns (with an actionable error) instead of tolerating them. "
             "Default off keeps the lenient behavior. Also settable via "
             "TARZAN_STRICT_INPUT=1.",
    )
    parser.add_argument(
        "--what-if", action="store_true",
        help="After committing the core newsletter, build the optional local "
             "candidate-portfolio workbook in a separate artifact set.",
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
    from tarzan.runtime.logging import setup_logging as setup

    os.makedirs(output_dir, exist_ok=True, mode=0o700)
    setup()


def _write_run_reports(
    output_dir: str,
    metrics=None,
    config=None,
    *,
    newsletter_html: str | None = None,
    what_if=None,
    result=None,
) -> ArtifactFinalizationOutcome:
    """Finalize one correlated local artifact set for every initialized run."""
    from contextlib import nullcontext
    from tarzan.runtime.session import completed_run_scope

    with completed_run_scope(result) if result is not None else nullcontext():
        return _write_run_reports_owned(
            output_dir, newsletter_html=newsletter_html, what_if=what_if, result=result,
        )


def _write_run_reports_owned(output_dir, *, newsletter_html, what_if, result):
    from tarzan.runtime import data_quality as dq
    from tarzan.runtime import report_html
    from tarzan import runtime
    from tarzan.runtime.artifacts import LocalArtifactWriter, StorageDescriptor
    from tarzan.runtime.ledger import LedgerEntryType
    from tarzan.runtime.publication import PublicationEvaluator
    from tarzan.runtime.summary import SummaryProjector

    try:
        logger.info(dq.summary_line())
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
            return ArtifactFinalizationOutcome(Path(path) if path else None)

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
        return ArtifactFinalizationOutcome(manifest)
    except Exception as error:  # noqa: BLE001
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
        return ArtifactFinalizationOutcome(error_type=type(error).__name__)


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


def _optional_what_if(args, output_dir, result, config, newsletter_html):
    """Optional work cannot overwrite or invalidate the committed core set."""
    from tarzan.runtime.session import completed_run_scope
    from tarzan.runtime.ledger import LedgerEntryType

    with completed_run_scope(result):
        try:
            from tarzan.backtest import newsletter_portfolios

            portfolios = newsletter_portfolios(
                deterministic=(args.deterministic or bool(args.as_of)),
            )
            workbook = _export_whatif(portfolios, config) if portfolios else None
            if workbook is None:
                raise RuntimeError("optional candidate workbook is unavailable")
            result.ledger.append(LedgerEntryType.STAGE, {
                "stage": "what_if", "outcome": "SUCCEEDED", "core_already_committed": True,
            })
            outcome = _write_run_reports(
                str(Path(output_dir) / "what-if"), result=result,
                newsletter_html=newsletter_html, what_if=workbook,
            )
            if not outcome.succeeded:
                raise OSError("optional artifact set could not be finalized")
        except Exception as error:
            result.ledger.open_failure(
                stage="what_if", stable_code="OPTIONAL_WHAT_IF_FAILED", severity="WARNING",
                error=error, affected_outputs=["what_if"],
                analytical_impact="core financial results and committed newsletter remain unchanged",
                publication_impact="NONE",
            )
            logger.warning("Optional What-If failed; the core artifact set is already committed.")
            _write_run_reports(
                str(Path(output_dir) / "what-if"), result=result,
                newsletter_html=None, what_if=None,
            )


def main(argv=None) -> int:
    args = parse_args(argv)

    from tarzan import runtime
    from tarzan.runtime.session import RunAttemptEnvelope, completed_run_scope, last_run_result

    previous_result = last_run_result()
    bootstrap = RunAttemptEnvelope.create("cli-bootstrap")
    runtime.reset(attempt_id=bootstrap.attempt_id, invocation_source=bootstrap.invocation_source)
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
    result = None
    status = 1
    output_error = None
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
        result = last_run_result()
        if result is None or result is previous_result:
            raise RuntimeError("orchestrator completed without a RunResult")
        with completed_run_scope(result):
            if result.report_view is not None:
                metrics, config = result.report_view.project()
            publication = PublicationEvaluator.evaluate(result.ledger.failure_records())
            if publication.decision in (
                PublicationDecision.SEND_NORMAL, PublicationDecision.SEND_DEGRADED_NORMAL,
            ):
                from tarzan.export.newsletter import render_newsletter
                from tarzan.export.newsletter._semantic import validate_newsletter_semantics

                semantic_audit = {}
                newsletter_html = render_newsletter(
                    metrics=metrics, config=config, semantic_audit=semantic_audit,
                    run_result=result,
                )
                violations = validate_newsletter_semantics(metrics, semantic_audit, newsletter_html)
                if violations:
                    result.ledger.open_failure(
                        stage="newsletter_semantics",
                        stable_code="NEWSLETTER_SEMANTIC_INVARIANT_FAILED",
                        severity="CRITICAL", error="; ".join(violations),
                        affected_outputs=["newsletter", "publication"],
                        analytical_impact="rendered facts do not match the computed financial authority",
                        publication_impact="BLOCK_NORMAL_AND_DO_NOT_SEND",
                    )
                    newsletter_html = None
                else:
                    status = 0
            else:
                logger.critical("Normal newsletter blocked by critical run evidence.")
            publication = PublicationEvaluator.evaluate(result.ledger.failure_records())
            result.ledger.append(LedgerEntryType.PUBLICATION, {
                "decision": publication.decision.value,
                "delivery_purpose": publication.delivery_purpose.value,
                "critical_failure_refs": list(publication.critical_failure_ids),
            })

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        output_error = e
    except (DataIngestionError, ValueError) as e:
        logger.error("Validation error: %s", e)
        output_error = e
    except TarzanError as e:
        logger.error("Analysis error: %s", e)
        output_error = e
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        logger.debug(traceback.format_exc())
        output_error = e
    finally:
        if result is None:
            fresh = last_run_result()
            result = fresh if fresh is not previous_result else None
        if output_error is not None and result is not None and metrics is not None:
            result.ledger.open_failure(
                stage="cli_output", stable_code="NEWSLETTER_RENDER_FAILED", severity="CRITICAL",
                error=output_error, affected_outputs=["newsletter", "publication"],
                analytical_impact="financial analysis succeeded but newsletter rendering did not",
                publication_impact="BLOCK_NORMAL_AND_DO_NOT_SEND",
            )
        # Always write the unified run report (run summary + data quality) —
        # on success, on the no-value early exit, and on any failure — so the
        # user always has one readable record of what the run produced and
        # what it skipped, regardless of how the run ended.
        outcome = _write_run_reports(
            output_dir,
            metrics=metrics,
            config=config,
            newsletter_html=newsletter_html,
            result=result,
        )
    if not isinstance(outcome, ArtifactFinalizationOutcome) or not outcome.succeeded:
        return 1
    if status == 0:
        logger.info("Core newsletter and evidence committed successfully.")
        if args.what_if:
            _optional_what_if(args, output_dir, result, config, newsletter_html)
    return status


if __name__ == "__main__":
    sys.exit(main())
