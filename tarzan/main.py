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
    return parser.parse_args(argv)


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(ch)
    fh = logging.FileHandler(os.path.join(output_dir, "analyzer.log"), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(fh)


def _write_run_reports(output_dir: str) -> None:
    """Write the single unified, human-readable run report (data-quality +
    rebalancing audit) as ``output/report.html``.

    The verbose ``analyzer.log`` stays separate (raw debug trace, rewritten
    each run). Best-effort: a diagnostic/audit must never turn a good run into
    a failure.
    """
    from tarzan import data_quality as dq
    from tarzan import audit
    from tarzan import report_html
    from tarzan import runtime
    try:
        logger.info(dq.summary_line())
        stamp = runtime.now_stamp("%Y-%m-%d %H:%M")
        path = report_html.write_report(output_dir, generated_at=stamp)
        if path:
            logger.info("Run report: %s (%d data-quality issue(s), %d "
                        "rebalancing plan(s))",
                        path, len(dq.issues()), len(audit.records()))
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
    try:
        from tarzan.orchestrator import run
        metrics, config = run(
            config_source=args.input_config,
            orders_source=args.input_orders,
            targets_per_holding_source=args.input_targets_per_holding,
            deterministic=args.deterministic,
            as_of=as_of,
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
        # Always write the per-run side reports (data-quality + rebalancing
        # audit) — on success, on the no-value early exit, and on any failure —
        # so the user can see what was skipped/coerced/failed and why each
        # rebalancing trade was suggested, regardless of how the run ended.
        _write_run_reports(args.output)


if __name__ == "__main__":
    sys.exit(main())
