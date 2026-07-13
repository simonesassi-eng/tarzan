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
import traceback
from datetime import date as _dtdate

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


def _write_run_reports(output_dir: str, metrics=None) -> None:
    """Write THE single run report — ``output/report.html`` — a color-coded
    table of the whole run's log (one row per entry, colored by level).

    This is the one and only log for a run: there is no separate analyzer.log.
    Best-effort: a diagnostic must never turn a good run into a failure.
    """
    from tarzan.runtime import data_quality as dq
    from tarzan.runtime import report_html
    from tarzan import runtime
    try:
        logger.info(dq.summary_line())
        stamp = runtime.now_stamp("%Y-%m-%d %H:%M")
        path = report_html.write_report(output_dir, generated_at=stamp,
                                        log_records=_LOG_CAPTURE.records)
        if path:
            logger.info("Run report: %s (%d data-quality issue(s))",
                        path, len(dq.issues()))
    except Exception as e:  # noqa: BLE001
        logger.debug("Run report step failed: %s", e)


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
    logger.info("Tarzan v3.0 starting...")
    logger.info("Output directory: %s", output_dir)

    if args.deterministic or as_of is not None:
        logger.info("Deterministic run%s (live quotes + AI summary skipped).",
                    f" as of {as_of}" if as_of else "")
    strict = args.strict or os.environ.get("TARZAN_STRICT_INPUT", "").strip() in ("1", "true", "yes")
    if strict:
        logger.info("Strict input validation ON (unrecognized columns rejected).")
    metrics = None
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
        if metrics.total_value == 0:
            logger.error("No portfolio value computed. Check input data.")
            return 1

        # Render the newsletter — Tarzan's primary artifact. (Benchmark names
        # are resolved from config inside generate_newsletter when omitted.)
        from tarzan.export.newsletter import generate_newsletter
        output_path = generate_newsletter(metrics, config, output_dir)
        logger.info("Newsletter saved to: %s", output_path)
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
        _write_run_reports(output_dir, metrics=metrics)


if __name__ == "__main__":
    sys.exit(main())
