"""Capture and replay a private portfolio with identical provider inputs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from tarzan.runtime.io_utils import atomic_write_bytes, canonical_json_bytes
from tarzan.verification.tape import ProviderTape, decode


def _private_json(path: Path, value) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value, default=str), fsync_dir=True)


@contextmanager
def _verification_failure(snapshot, label, tape):
    try:
        yield
    except BaseException as error:
        _private_json(snapshot / f"{label}.failure.json", {
            "error_type": type(error).__name__,
            "error": str(error),
            "missing_observations": tape.misses,
            "network_attempts": tape.network_attempts,
        })
        raise


def _financial_value(value):
    if isinstance(value, Mapping):
        return {str(getattr(key, "value", key)): _financial_value(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return _financial_value(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return {"index": [str(item) for item in value.index],
                "values": _financial_value(value.tolist())}
    if isinstance(value, (tuple, list)):
        return [_financial_value(item) for item in value]
    if isinstance(value, np.generic):
        return _financial_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(getattr(value, "value", value))


def portfolio_kpis(metrics):
    """Financial outputs, separated from operation IDs, wall time and AI prose."""
    scalar_names = (
        "total_value", "trustworthy_total_value_eur", "known_valuation_subtotal_eur",
        "invested_value", "cash_value", "cash_target_eur",
        "pnl_eur", "pnl_pct", "invested_capital_eur",
        "estimated_cgt_eur", "pnl_eur_net_tax", "pnl_pct_net_tax",
        "xirr_pct", "xirr_net_tax_pct", "twr_pct", "twr_annualized_pct",
        "returns_coverage_pct", "weighted_yield", "avg_ter",
        "valuation_availability", "history_availability",
        "unrealized_pnl_eur", "unrealized_pnl_pct",
    )
    result = {name: _financial_value(getattr(metrics, name, None)) for name in scalar_names}
    for name in (
        "performance", "performance_full", "risk", "historical_risk",
        "allocation_by_class", "allocation_by_geo", "allocation_by_sector",
        "holding_performance", "goal_deltas", "target_weights",
        "rebalancing_plans", "degraded_computers", "benchmark_comparison",
        "portfolio_history", "actual_value_series", "pnl_series", "unrealized_series",
        "target_history", "holding_histories", "benchmark_histories",
        "allocation_timeline", "external_flows", "xirr_cashflows",
    ):
        result[name] = _financial_value(getattr(metrics, name, None))
    columns = (
        "isin", "ticker", "quantity", "current_price", "current_value",
        "cost_basis_eur", "weight_pct", "gain_eur", "gain_pct",
        "value_eur", "portfolio_weight_pct",
    )
    holdings = metrics.holdings_df
    result["holdings"] = _financial_value(holdings[[name for name in columns if name in holdings]])
    return result


def differences(left, right, prefix=""):
    """Recursive changed values; tiny numerical roundoff is not a KPI change."""
    if isinstance(left, bool) != isinstance(right, bool):
        return [{"kpi": prefix, "before": left, "after": right}]
    if isinstance(left, dict) and isinstance(right, dict):
        found = []
        for name in sorted(set(left) | set(right)):
            found.extend(differences(left.get(name), right.get(name), f"{prefix}.{name}".strip(".")))
        return found
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [change for i, (a, b) in enumerate(zip(left, right))
                for change in differences(a, b, f"{prefix}[{i}]")]
    if (isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float)) and not isinstance(right, bool)):
        if math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9):
            return []
        return [{"kpi": prefix, "before": left, "after": right, "delta": right - left}]
    return [] if left == right else [{"kpi": prefix, "before": left, "after": right}]


def compare_checkpoints(left, right):
    for name in ("input_sha256", "captured_at", "observation_digest", "identity_cache_digest"):
        if not left.get(name) or left.get(name) != right.get(name):
            raise ValueError(f"different or missing {name}; not a valid code-only comparison")
    if left.get("semantic_errors") or right.get("semantic_errors"):
        raise ValueError("semantic validation failed; inspect private checkpoints")
    if left.get("offline_network_attempts") or right.get("offline_network_attempts"):
        raise ValueError("unrecorded network was attempted")
    return differences(left["kpis"], right["kpis"])


def run_snapshot(snapshot: Path, *, label: str, capture: bool, input_dir: Path, seed_tape: Path | None = None):
    from tarzan import orchestrator, runtime
    from tarzan.data import price_cache
    from tarzan.export.newsletter import render_newsletter
    from tarzan.runtime.session import last_run_result

    snapshot = snapshot.resolve()
    if not label or Path(label).name != label or label.startswith("."):
        raise ValueError("label must be one non-hidden filename component")
    if capture:
        seed = json.loads(seed_tape.read_text()) if seed_tape is not None else None
        snapshot.mkdir(parents=True, exist_ok=True, mode=0o700)
        if (snapshot / "snapshot.json").exists():
            raise FileExistsError("snapshot already exists; choose a new directory")
        snapshot.chmod(0o700)
        inputs = snapshot / "inputs"
        inputs.mkdir(mode=0o700)
        fingerprints = {}
        for name in ("order_list.csv", "targets.csv", "targets_per_holding.csv"):
            source = input_dir / name
            if not source.exists():
                if name == "order_list.csv":
                    raise FileNotFoundError(source)
                continue
            data = source.read_bytes()
            atomic_write_bytes(inputs / name, data)
            fingerprints[name] = hashlib.sha256(data).hexdigest()
        # Identity evidence is retained, but price tapes are cold so the capture
        # includes raw full histories, not previously repaired cached heads.
        cache_source = price_cache.cache_dir()
        if seed is not None:
            if seed["input_sha256"] != fingerprints:
                raise ValueError("seed tape belongs to different portfolio inputs")
            cache_source = seed_tape.parent / "initial-cache"
        initial = snapshot / "initial-cache"
        shutil.copytree(cache_source, initial, ignore=shutil.ignore_patterns("history", "*.lock"))
        meta = {
            "schema_version": ProviderTape.SCHEMA_VERSION,
            "captured_at": seed["captured_at"] if seed is not None else dt.datetime.now(dt.timezone.utc).isoformat(),
            "input_sha256": fingerprints,
            "cache_history": "cold-raw-provider-capture",
        }
        seed_records = seed["records"] if seed is not None else None
        if seed_records is not None:
            # Pre-protocol captures lacked the public content-type metadata.
            # Do not guess it: re-observe those responses before sealing the
            # valid baseline, keeping all already recorded price tapes.
            seed_records = {
                key: record for key, record in seed_records.items()
                if record.get("operation") != "requests"
                or "value" not in record or "headers" in decode(record["value"])
            }
        tape = ProviderTape(records=seed_records)
    else:
        meta = json.loads((snapshot / "snapshot.json").read_text())
        if meta.get("schema_version") != ProviderTape.SCHEMA_VERSION:
            raise ValueError("unsupported snapshot schema")
        expected_tape = meta.get("tape_sha256")
        if expected_tape and hashlib.sha256(canonical_json_bytes(meta["records"])).hexdigest() != expected_tape:
            raise ValueError("snapshot tape checksum mismatch")
        inputs = snapshot / "inputs"
        initial = snapshot / "initial-cache"
        for name, expected in meta["input_sha256"].items():
            if hashlib.sha256((inputs / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"snapshot input checksum mismatch: {name}")
        tape = ProviderTape(records=meta["records"], replay=True)

    captured_at = dt.datetime.fromisoformat(meta["captured_at"])
    identity_cache_digest = hashlib.sha256(canonical_json_bytes({
        str(path.relative_to(initial)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(initial.rglob("*")) if path.is_file()
    })).hexdigest()
    original_build = runtime._build_context

    def frozen_context(**kwargs):
        return replace(original_build(**kwargs), captured_at=captured_at)

    with _verification_failure(snapshot, label, tape), \
            tempfile.TemporaryDirectory(prefix="working-cache-", dir=snapshot) as directory:
        working_cache = Path(directory) / "cache"
        shutil.copytree(initial, working_cache)
        with patch.dict(os.environ, {"TARZAN_CACHE_DIR": str(working_cache), "TARZAN_DISABLE_AI": "1"}), \
                patch.object(runtime, "_build_context", frozen_context), tape:
            metrics, config = orchestrator.run(
                orders_source=str(inputs / "order_list.csv"),
                config_source=str(inputs / "targets.csv") if (inputs / "targets.csv").exists() else None,
                targets_per_holding_source=str(inputs / "targets_per_holding.csv")
                if (inputs / "targets_per_holding.csv").exists() else None,
            )
            semantic_audit = {}
            html = render_newsletter(metrics=metrics, config=config, semantic_audit=semantic_audit)
            from tarzan.export.newsletter._semantic import validate_newsletter_semantics
            semantic_errors = validate_newsletter_semantics(metrics, semantic_audit, html)
            result = last_run_result()
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
            ).stdout.strip()
            output = {
                "revision": revision,
                "dirty_code": subprocess.run(
                    ["git", "status", "--porcelain", "--untracked-files=normal"],
                    check=True, capture_output=True, text=True,
                ).stdout.splitlines(),
                "label": label,
                "captured_at": meta["captured_at"],
                "input_sha256": meta["input_sha256"],
                "identity_cache_digest": identity_cache_digest,
                "observation_digest": hashlib.sha256(canonical_json_bytes(tape.records)).hexdigest(),
                "kpis": portfolio_kpis(metrics),
                "analysis_id": result.analysis_id,
                "provider_requests": tape.calls,
                "recorded_requests": len(tape.records),
                "offline_network_attempts": len(tape.network_attempts),
                "semantic_errors": semantic_errors,
                "failure_codes": [failure.stable_code for failure in result.ledger.failure_records()],
            }
            _private_json(snapshot / f"{label}.json", output)
            atomic_write_bytes(snapshot / f"{label}.html", html.encode())
    if capture:
        meta["records"] = tape.records
        meta["tape_sha256"] = hashlib.sha256(canonical_json_bytes(tape.records)).hexdigest()
        _private_json(snapshot / "snapshot.json", meta)
    print(f"Private KPI checkpoint: {snapshot / (label + '.json')}")
    print(f"Recorded requests: {len(tape.records)}; replay network attempts: {len(tape.network_attempts)}")
    if semantic_errors:
        raise ValueError(f"newsletter semantic gate: {len(semantic_errors)} error(s); see private checkpoint")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("capture", "replay"):
        sub = commands.add_parser(command)
        sub.add_argument("--snapshot", type=Path, required=True)
        sub.add_argument("--label", required=True)
        sub.add_argument("--input-dir", type=Path, default=Path("input"))
        if command == "capture":
            sub.add_argument("--seed-tape", type=Path, help="Retain recorded observations while completing a capture")
    compare = commands.add_parser("compare")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.command == "compare":
        left, right = (json.loads(path.read_text()) for path in (args.before, args.after))
        delta = compare_checkpoints(left, right)
        _private_json(args.after.with_suffix(".delta.json"), {
            "before_revision": left["revision"], "after_revision": right["revision"],
            "changes": delta, "changed_kpis": len(delta),
        })
        print(f"Changed KPI fields: {len(delta)}; private delta: {args.after.with_suffix('.delta.json')}")
        return 0
    run_snapshot(args.snapshot, label=args.label, capture=args.command == "capture", input_dir=args.input_dir,
                 seed_tape=getattr(args, "seed_tape", None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
