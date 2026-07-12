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

from tarzan.exceptions import DataIngestionError, TarzanError

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
    from tarzan import data_quality as dq
    from tarzan import report_html
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
    setup_logging(args.output)
    logger.info("Tarzan v3.0 starting...")

    from tarzan import data_quality as dq
    # Parse the optional as-of date; an invalid value is a hard, actionable
    # error (not a silent fallback to today, which would mislead).
    as_of = None
    if args.as_of:
        try:
            as_of = _dtdate.fromisoformat(args.as_of)
        except ValueError:
            logger.error("Invalid --as_of %r (expected YYYY-MM-DD)", args.as_of)
            return 1
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

        # Generate Excel (keep legacy export)
        from tarzan.export.excel import generate_excel
        output_path = generate_excel(metrics, [], config, args.output)
        logger.info("Dashboard saved to: %s", output_path)
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
        _write_run_reports(args.output, metrics=metrics)


if __name__ == "__main__":
    sys.exit(main())
