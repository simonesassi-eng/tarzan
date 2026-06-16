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


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.output)
    logger.info("Tarzan v3.0 starting...")

    try:
        from tarzan.orchestrator import run
        metrics, config = run(
            config_source=args.input_config,
            orders_source=args.input_orders,
            targets_per_holding_source=args.input_targets_per_holding,
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


if __name__ == "__main__":
    sys.exit(main())
