"""Durable, encrypted run evidence; never archive raw SDK logs or credentials.

The public repository may retain only ``report.sealed``. A recovery key can
decrypt it locally into a caller-selected private JSON file. This observer
does not choose prices, relax publication gates or authorize delivery.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re

from tarzan.runtime import sealed
from tarzan.runtime.io_utils import atomic_write_bytes, canonical_json_bytes
from tarzan.runtime.ledger import ErrorNormalizer, LedgerEntryType
from tarzan.runtime.publication import PublicationEvaluator


SCHEMA_VERSION = "1"
PURPOSE = "tarzan-diagnostics-v1"
FILENAME = "report.sealed"
DIRECTORY_ENV = "TARZAN_DIAGNOSTIC_DIR"
_INPUT_NAMES = frozenset({"order_list.csv", "targets.csv", "targets_per_holding.csv"})
_FIELDS = frozenset({
    "schema_version", "operation", "exit_code", "duration_seconds", "captured_at",
    "run", "input_sha256", "entrypoint_error_type", "provider_counts", "quality",
    "summary", "kpis", "ledger", "projection_error_type",
})
_CREDENTIAL_ENV = (
    sealed.KEY_ENV, "SMTP_USER", "SMTP_PASS", "RECIPIENT_EMAIL", "GEMINI_API_KEY", "GOOGLE_DRIVE_CREDENTIALS_JSON",
    "DELIVERY_CLAIM_TOKEN", "GITHUB_TOKEN", "GH_TOKEN",
)
logger = logging.getLogger(__name__)


def _redact_credentials(value):
    """Exclude native credential values even if an upstream exception echoed one."""
    credentials = {os.environ[name] for name in _CREDENTIAL_ENV if os.environ.get(name)}
    encoded = os.environ.get("GOOGLE_DRIVE_CREDENTIALS_JSON")
    if encoded:
        try:
            parsed = json.loads(encoded)
            if isinstance(parsed, dict):
                credentials.update(
                    item for name, item in parsed.items()
                    if name in {"private_key", "client_secret", "access_token", "refresh_token"}
                    and isinstance(item, str) and item
                )
        except (TypeError, ValueError):
            pass
    credential_values = sorted(credentials, key=len, reverse=True)

    def clean(item):
        if isinstance(item, dict):
            return {key: clean(child) for key, child in item.items()}
        if isinstance(item, list):
            return [clean(child) for child in item]
        if isinstance(item, str):
            item = ErrorNormalizer.normalize(item)
            for secret in credential_values:
                item = item.replace(secret, "[REDACTED]")
            return item
        return item

    # Ledger payloads were recursively normalized by RunLedger on append.
    # Normalizing the envelope again would redact the entire field named
    # "payload", destroying the exact source/semantic failure we must retain.
    return clean(value)


def _quality(metrics):
    if metrics is None:
        return {}
    from tarzan.contracts.reference import reference_for
    from tarzan.data.market_quotes import MARKETS

    reference = reference_for(metrics)
    held = set(metrics.holdings_df.get("ticker", ())) & set(metrics.intraday_requested_tickers)
    return {
        "valuation_availability": metrics.valuation_availability,
        "history_availability": metrics.history_availability,
        "history_unavailable_instruments": list(metrics.history_unavailable_instruments),
        "returns_coverage_pct": metrics.returns_coverage_pct,
        "required_benchmarks": {
            "alpha_beta": reference.benchmark_alpha_beta,
            "geography": reference.benchmark_geo,
        },
        "benchmark_tickers": dict(metrics.benchmark_tickers),
        "benchmarks_requested": len(reference.benchmarks),
        "benchmarks_resolved": len(metrics.benchmark_tickers),
        "benchmark_resolution_errors": list(metrics.benchmark_resolution_errors),
        "intraday_requested": len(metrics.intraday_requested_tickers),
        "intraday_with_bars": len(metrics.intraday_quotes),
        "intraday_missing": [
            ticker for ticker in metrics.intraday_requested_tickers
            if ticker not in metrics.intraday_quotes
        ],
        "intraday_lagged": [
            ticker for ticker, fact in metrics.intraday_evidence.items()
            if fact.get("state") == "LAGGED_CURRENT_SESSION"
        ],
        "intraday_single_observation": [
            ticker for ticker, fact in metrics.intraday_evidence.items()
            if fact.get("native_bar_count") == 1
        ],
        "intraday_evidence": dict(metrics.intraday_evidence),
        "intraday_current_holdings_missing": sorted(held - set(metrics.intraday_quotes)),
        "intraday_current_holdings_lagged": sorted(
            ticker for ticker in held
            if metrics.intraday_evidence.get(ticker, {}).get("state") == "LAGGED_CURRENT_SESSION"
        ),
        "macro_requested": len(MARKETS),
        "macro_available": len(metrics.market_quotes) if metrics.market_quotes is not None else None,
        "degraded_computers": list(metrics.degraded_computers),
    }


def build_report(
    result, *, operation, exit_code, duration_seconds, input_sha256=None,
    entrypoint_error_type=None,
):
    """Project exact owned evidence, including source/semantic failures."""
    entries = [entry.to_dict() for entry in result.ledger.entries] if result is not None else []
    native, adapters = defaultdict(Counter), defaultdict(Counter)
    if result is not None:
        for entry in result.ledger.entries:
            if entry.entry_type is LedgerEntryType.PROVIDER_ATTEMPT:
                counts = native if entry.payload.get("operation") == "native_http" else adapters
                counts[str(entry.payload.get("source", "UNKNOWN"))][
                    str(entry.payload.get("outcome", "UNKNOWN"))
                ] += 1
    report = {
        "schema_version": SCHEMA_VERSION,
        "operation": operation,
        "exit_code": exit_code,
        "duration_seconds": duration_seconds,
        "captured_at": (
            result.session.context.captured_at.isoformat()
            if result is not None and result.session is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        "run": {
            "attempt_id": result.attempt_id if result is not None else None,
            "analysis_id": result.analysis_id if result is not None else None,
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "revision": os.environ.get("GITHUB_SHA"),
        },
        "input_sha256": dict(input_sha256 or {}),
        "entrypoint_error_type": entrypoint_error_type,
        "provider_counts": {"native": dict(native), "adapters": dict(adapters)},
        "quality": {},
        "summary": None,
        "kpis": {},
        "ledger": entries,
        "projection_error_type": None,
    }
    if result is not None:
        try:
            from tarzan.runtime.summary import SummaryProjector
            from tarzan.verification.portfolio import portfolio_kpis

            metrics = result.report_view.project()[0] if result.report_view is not None else result.metrics
            report["quality"] = _quality(metrics)
            report["summary"] = SummaryProjector.project(
                result, PublicationEvaluator.evaluate(result.ledger.failure_records()),
            ).to_dict()
            if metrics is not None:
                report["kpis"] = portfolio_kpis(metrics)
        except Exception as error:
            # A projection defect must not discard the primary failure ledger.
            report["projection_error_type"] = type(error).__name__
    return _redact_credentials(report)


def _validate(report):
    if not isinstance(report, dict) or set(report) != _FIELDS or report["schema_version"] != SCHEMA_VERSION:
        raise ValueError("incompatible private diagnostic report")
    if report["operation"] not in {"send", "verify"} or type(report["exit_code"]) is not int \
            or report["exit_code"] not in {0, 1}:
        raise ValueError("invalid private diagnostic operation")
    duration = report["duration_seconds"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) \
            or not math.isfinite(duration) or duration < 0:
        raise ValueError("invalid private diagnostic duration")
    fingerprints = report["input_sha256"]
    if not isinstance(fingerprints, dict) or not set(fingerprints).issubset(_INPUT_NAMES) or any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in fingerprints.values()
    ):
        raise ValueError("invalid private diagnostic input fingerprints")
    if not isinstance(report["ledger"], list) or any(
        not isinstance(entry, dict) or set(entry) != {
            "sequence", "entry_id", "entry_type", "recorded_at", "payload",
        } or not isinstance(entry["payload"], dict) for entry in report["ledger"]
    ):
        raise ValueError("invalid private diagnostic ledger")
    for name in ("run", "provider_counts", "quality", "kpis"):
        if not isinstance(report[name], dict):
            raise ValueError("invalid private diagnostic projection")
    if report["summary"] is not None and not isinstance(report["summary"], dict):
        raise ValueError("invalid private diagnostic summary")
    return report


def write_report(directory, report, key):
    directory = sealed.owned_directory(Path(directory))
    if any(path.name != FILENAME or path.is_symlink() or not path.is_file() for path in directory.iterdir()):
        raise ValueError("private diagnostic staging contains an unrelated file")
    sealed.write_snapshot(
        directory / FILENAME, canonical_json_bytes(_validate(report)), key, purpose=PURPOSE,
    )


def capture_configured(result, *, key_file=None, **values):
    """Best-effort observer: capture failures must never replace the send result."""
    directory = os.environ.get(DIRECTORY_ENV)
    if not directory:
        return False
    try:
        report = build_report(result, **values)
        write_report(directory, report, sealed.configured_key(key_file=key_file))
    except Exception:
        logger.warning(
            "Encrypted diagnostic capture unavailable; primary outcome retained.",
            extra={"public_code": "DIAGNOSTIC_CAPTURE_FAILED"},
        )
        return False
    logger.info(
        "Owned diagnostic evidence encrypted.",
        extra={"public_code": "DIAGNOSTIC_CAPTURE_COMPLETE"},
    )
    return True


def read_report(path, key):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate private diagnostic field")
            result[name] = value
        return result

    def invalid_constant(_):
        raise ValueError("nonfinite private diagnostic JSON")

    content = sealed.unseal(
        sealed.read_bounded_file(Path(path), sealed.MAX_SEALED_BYTES), key, purpose=PURPOSE,
    )
    return _validate(json.loads(
        content, object_pairs_hook=unique, parse_constant=invalid_constant,
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("decrypt",))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("private diagnostic output already exists")
        report = read_report(args.snapshot, sealed.configured_key(key_file=args.key_file))
        directory = sealed.owned_directory(args.output.parent)
        atomic_write_bytes(directory / args.output.name, canonical_json_bytes(report), fsync_dir=True)
    except Exception:
        print("Private diagnostic decryption failed; no unverified report was written.")
        return 1
    print("Private diagnostic report decrypted into the requested local file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
