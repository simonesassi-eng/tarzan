"""Analyze and render the real inputs with no claim or SMTP invocation.

Unlike a delivery dry-run, this entry point needs no mail credentials. GitHub
retains only encrypted diagnostics; explicitly requested local output is private.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import logging
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tarzan import log_redaction, orchestrator  # noqa: E402
from tarzan.delivery import ResolvedInputSet, _close_delivery_inputs, resolve_inputs  # noqa: E402
from tarzan.export.newsletter import render_newsletter  # noqa: E402
from tarzan.export.newsletter._semantic import validate_newsletter_semantics  # noqa: E402
from tarzan.runtime import diagnostics, sealed  # noqa: E402
from tarzan.runtime.io_utils import atomic_write_bytes, canonical_json_bytes  # noqa: E402
from tarzan.runtime.ledger import LedgerEntryType  # noqa: E402
from tarzan.runtime.logging import setup_logging  # noqa: E402
from tarzan.runtime.publication import PublicationDecision, PublicationEvaluator  # noqa: E402
from tarzan.runtime.session import completed_run_scope, last_run_result  # noqa: E402


logger = logging.getLogger("tarzan.newsletter")


@contextmanager
def _without_ai():
    previous = os.environ.get("TARZAN_DISABLE_AI")
    os.environ["TARZAN_DISABLE_AI"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TARZAN_DISABLE_AI", None)
        else:
            os.environ["TARZAN_DISABLE_AI"] = previous


def _inputs(directory):
    if directory is None:
        return resolve_inputs()
    directory = directory.expanduser().resolve()
    paths = {
        "orders": directory / "order_list.csv",
        "config": directory / "targets.csv",
        "targets_per_holding": directory / "targets_per_holding.csv",
    }
    for name in ("orders", "config"):
        if not paths[name].is_file():
            raise FileNotFoundError("required verification input is unavailable")
    return ResolvedInputSet({
        name: str(path) if path.is_file() else None for name, path in paths.items()
    })


def _fingerprints(inputs):
    names = {
        "orders": "order_list.csv", "config": "targets.csv",
        "targets_per_holding": "targets_per_holding.csv",
    }
    fingerprints = {}
    for key, name in names.items():
        if inputs.get(key) is not None:
            with open(inputs[key], "rb") as handle:
                fingerprints[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return fingerprints


def verify(inputs, previous_result):
    orchestrator.run(
        config_source=inputs["config"], orders_source=inputs["orders"],
        targets_per_holding_source=inputs["targets_per_holding"],
    )
    result = last_run_result()
    if result is None or result is previous_result:
        raise RuntimeError("verification did not record a fresh owned run result")
    with completed_run_scope(result):
        metrics, config = result.report_view.project() if result.report_view is not None \
            else result.compatibility_tuple()
        publication = PublicationEvaluator.evaluate(result.ledger.failure_records())
        html = None
        if publication.decision in {
            PublicationDecision.SEND_NORMAL, PublicationDecision.SEND_DEGRADED_NORMAL,
        }:
            audit = {}
            html = render_newsletter(
                metrics=metrics, config=config, semantic_audit=audit, run_result=result,
            )
            errors = validate_newsletter_semantics(metrics, audit, html)
            if errors:
                result.ledger.open_failure(
                    stage="newsletter_semantics", stable_code="NEWSLETTER_SEMANTIC_INVARIANT_FAILED",
                    severity="CRITICAL", error={"violations": list(errors)},
                    affected_outputs=["newsletter", "benchmark_tables", "performance_charts"],
                    analytical_impact="rendered identities or labels violate the analytical contract",
                    publication_impact="BLOCK_NORMAL_AND_DO_NOT_SEND",
                )
                logger.error(
                    "Newsletter verification failed the semantic gate.",
                    extra={"public_code": "SEMANTIC_VALIDATION_FAILED"},
                )
        publication = PublicationEvaluator.evaluate(result.ledger.failure_records())
        result.ledger.append(LedgerEntryType.PUBLICATION, {
            "decision": publication.decision.value,
            "delivery_purpose": publication.delivery_purpose.value,
            "critical_failure_refs": list(publication.critical_failure_ids),
        })
        result.ledger.append(LedgerEntryType.DELIVERY, {
            "state": "VERIFICATION_ONLY", "smtp_invoked": False, "claim_invoked": False,
            "purpose": publication.delivery_purpose.value,
        })
        success = publication.decision in {
            PublicationDecision.SEND_NORMAL, PublicationDecision.SEND_DEGRADED_NORMAL,
        }
        return result, html, 0 if success else 1


def _write_private_output(directory, report, html):
    directory = sealed.owned_directory(directory)
    files = {"report.json": canonical_json_bytes(report)}
    if html is not None:
        files["newsletter.html"] = html.encode("utf-8")
    # Never overwrite a previous comparison or a caller's report.
    if any((directory / name).exists() or (directory / name).is_symlink() for name in files):
        raise FileExistsError("private verification output already exists")
    for name, content in files.items():
        atomic_write_bytes(directory / name, content, fsync_dir=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args(argv)
    setup_logging()
    log_redaction.install(enabled=True)
    previous_result = last_run_result()
    started = time.monotonic()
    result, inputs, html = None, None, None
    fingerprints = {}
    exit_code = 1
    error_type = None
    with _without_ai():
        try:
            inputs = _inputs(args.input_dir)
            fingerprints = _fingerprints(inputs)
            result, html, exit_code = verify(inputs, previous_result)
        except Exception as error:
            fresh = last_run_result()
            result = fresh if fresh is not previous_result else None
            error_type = type(error).__name__
            logger.error(
                "Newsletter verification failed; no mail or delivery claim was requested.",
                extra={"public_code": "VERIFICATION_FAILED"},
            )
        finally:
            if inputs is not None:
                _close_delivery_inputs(inputs, result=result)
            values = {
                "operation": "verify", "exit_code": exit_code,
                "duration_seconds": time.monotonic() - started,
                "input_sha256": fingerprints, "entrypoint_error_type": error_type,
            }
            if args.private_output is not None:
                try:
                    report = diagnostics.build_report(result, **values)
                    _write_private_output(args.private_output, report, html)
                except Exception as error:
                    exit_code = 1
                    values.update(exit_code=1, entrypoint_error_type=type(error).__name__)
                    logger.error(
                        "Requested local verification output could not be retained.",
                        extra={"public_code": "VERIFICATION_FAILED"},
                    )
            captured = diagnostics.capture_configured(result, key_file=args.key_file, **values)
            if os.environ.get(diagnostics.DIRECTORY_ENV) and not captured:
                # A verification cannot attest to durable remote evidence
                # when its explicitly requested encrypted report is missing.
                # The real send shim keeps diagnostics best-effort instead.
                exit_code = 1
    logger.info(
        "Newsletter verification finished without delivery.",
        extra={"public_code": "VERIFICATION_COMPLETE" if exit_code == 0 else "VERIFICATION_FAILED"},
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
