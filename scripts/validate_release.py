"""Credential-free validation of Tarzan's immutable release declarations.

The checker reads only the manifest and explicitly named positive Tarzan paths.
It never discovers repository contents recursively.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_HASH = re.compile(r"--hash=sha256:[0-9a-f]{64}(?:\s|$)")
_EXACT_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)(?:\s|$)")
_ALLOWED_EXACT = {
    ".github/workflows/newsletter.yml",
    "README.md",
    "requirements.in",
    "requirements.txt",
    "tarzan/README.md",
    "tarzan/release_manifest.json",
}
_ALLOWED_PREFIXES = ("tarzan/", "scripts/")
_EXPECTED_COMMANDS = {
    "python scripts/validate_release.py --manifest tarzan/release_manifest.json",
    "python -m compileall -q tarzan scripts",
    "python -m pytest tarzan/tests -q",
}


class ReleaseValidationError(ValueError):
    """A release declaration violates the positive-scope contract."""


def _read(root: Path, relative: str) -> str:
    _validate_positive_path(relative)
    return (root / relative).read_text(encoding="utf-8")


def _validate_positive_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or any(char in value for char in "*?[]{}"):
        raise ReleaseValidationError(f"non-literal release path: {value!r}")
    if value not in _ALLOWED_EXACT and not value.startswith(_ALLOWED_PREFIXES):
        raise ReleaseValidationError(f"path is outside the positive Tarzan scope: {value!r}")


def _constant(source: str, name: str) -> str:
    match = re.search(
        rf"(?m)^{re.escape(name)}\s*=\s*(?:[\"']([^\"']+)[\"']|(\d+))\s*$",
        source,
    )
    if not match:
        raise ReleaseValidationError(f"missing version authority {name}")
    return match.group(1) or match.group(2)


def _validate_requirements(root: Path) -> None:
    direct_lines = [
        line.strip()
        for line in _read(root, "requirements.in").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    lock_lines = [
        line.strip()
        for line in _read(root, "requirements.txt").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not direct_lines or not lock_lines:
        raise ReleaseValidationError("dependency manifests cannot be empty")

    direct_names: set[str] = set()
    for line in direct_lines:
        match = _EXACT_REQUIREMENT.match(line)
        if not match or match.group(0).strip() != line:
            raise ReleaseValidationError(f"direct dependency is not exact: {line}")
        direct_names.add(re.sub(r"[-_.]+", "-", match.group(1)).lower())

    locked_names: set[str] = set()
    for line in lock_lines:
        match = _EXACT_REQUIREMENT.match(line)
        if not match or not _HASH.search(line):
            raise ReleaseValidationError(f"lock entry is not exact and hashed: {line[:120]}")
        hashes = re.findall(r"--hash=sha256:([0-9a-f]{64})(?:\s|$)", line)
        if not hashes or len(hashes) != len(set(hashes)):
            raise ReleaseValidationError(f"lock entry has missing or duplicate hashes: {match.group(1)}")
        locked_names.add(re.sub(r"[-_.]+", "-", match.group(1)).lower())

    missing = direct_names - locked_names
    if missing:
        raise ReleaseValidationError(f"direct dependencies missing from lock: {sorted(missing)}")


def _validate_workflow(root: Path) -> None:
    workflow = _read(root, ".github/workflows/newsletter.yml")
    action_refs = re.findall(r"(?m)^\s*uses:\s*[^\s#]+@([^\s#]+)", workflow)
    if not action_refs or any(not _FULL_SHA.fullmatch(ref) for ref in action_refs):
        raise ReleaseValidationError("every workflow action must use a full commit SHA")
    if not re.search(r"(?m)^  validate:\s*$", workflow):
        raise ReleaseValidationError("workflow has no credential-free validate job")
    if not re.search(r"(?m)^    needs:\s*validate\s*$", workflow):
        raise ReleaseValidationError("publication does not depend on validation")
    if re.search(r"(?ms)^    env:\s*\n(?:      .+\n)*?      .+secrets\.", workflow):
        raise ReleaseValidationError("publication credentials cannot be job scoped")
    if "pip install --require-hashes -r requirements.txt" not in workflow:
        raise ReleaseValidationError("workflow installation is not hash enforced")
    if "tarzan/release_manifest.json" not in workflow:
        raise ReleaseValidationError("workflow does not validate the release manifest")

    current_step = ""
    for raw_line in workflow.splitlines():
        step = re.match(r"^      - name:\s*(.+?)\s*$", raw_line)
        if step:
            current_step = step.group(1)
        if "secrets." in raw_line and current_step != "Render & send newsletter":
            raise ReleaseValidationError(
                f"credential is exposed outside the final publication step: {current_step!r}"
            )


def _validate_versions(root: Path, manifest: dict[str, Any]) -> None:
    authorities = {
        "input": ("tarzan/contracts/schema.py", "SCHEMA_VERSION"),
        "summary": ("tarzan/runtime/summary.py", "RUN_SUMMARY_SCHEMA_VERSION"),
        "ledger": ("tarzan/runtime/ledger.py", "LEDGER_SCHEMA_VERSION"),
        "manifest": ("tarzan/runtime/artifacts.py", "MANIFEST_SCHEMA_VERSION"),
        "cache": ("tarzan/data/price_cache.py", "_CACHE_SCHEMA_VERSION"),
        "exposure": ("tarzan/engine/allocations.py", "CANONICAL_EXPOSURE_SCHEMA_VERSION"),
        "capability": ("tarzan/instruments/registry.py", "CAPABILITY_SCHEMA_VERSION"),
        "provider_policy": ("tarzan/runtime/provider.py", "PROVIDER_POLICY_SCHEMA_VERSION"),
        "delivery_identity": ("tarzan/delivery/claims.py", "DELIVERY_IDENTITY_SCHEMA_VERSION"),
        "delivery_state": ("tarzan/delivery/claims.py", "DELIVERY_STATE_SCHEMA_VERSION"),
    }
    declared = manifest.get("schema_versions", {})
    if set(declared) != set(authorities):
        raise ReleaseValidationError("release manifest schema authorities are incomplete")
    for key, (path, constant_name) in authorities.items():
        actual = _constant(_read(root, path), constant_name)
        if str(declared[key]) != actual:
            raise ReleaseValidationError(f"{key} schema drift: {declared[key]} != {actual}")

    version = _constant(_read(root, "tarzan/version.py"), "APPLICATION_VERSION")
    if manifest.get("application_version") != version:
        raise ReleaseValidationError("application version drift")
    if "APPLICATION_VERSION as __version__" not in _read(root, "tarzan/__init__.py"):
        raise ReleaseValidationError("package version is not sourced from tarzan.version")
    if "from tarzan import __version__" not in _read(root, "tarzan/main.py"):
        raise ReleaseValidationError("CLI does not use the package version authority")

    ai_source = _read(root, "tarzan/export/ai_summary.py")
    pins = manifest.get("provider_pins", {})
    if _constant(ai_source, "_PINNED_MODEL") != pins.get("gemini_model"):
        raise ReleaseValidationError("Gemini model pin drift")
    if f"/{pins.get('gemini_api')}/models/" not in ai_source:
        raise ReleaseValidationError("Gemini API pin drift")


def _validate_scope_and_review(manifest: dict[str, Any]) -> None:
    scope = manifest.get("positive_scope", {})
    values: list[str] = []
    for key in ("package_entries", "test_roots", "script_roots", "named_release_files"):
        entries = scope.get(key)
        if not isinstance(entries, list) or not entries:
            raise ReleaseValidationError(f"positive scope declaration is empty: {key}")
        values.extend(str(item) for item in entries)
    for value in values:
        _validate_positive_path(value)

    commands = set(manifest.get("validation_commands", ()))
    if commands != _EXPECTED_COMMANDS:
        raise ReleaseValidationError("validation commands must use the reviewed positive paths")

    review = manifest.get("pin_review", {})
    reviewed = date.fromisoformat(str(review.get("reviewed_on")))
    due = date.fromisoformat(str(review.get("review_due_on")))
    if due < reviewed or (due - reviewed).days > 31:
        raise ReleaseValidationError("pin review interval exceeds one month")
    if date.today() > due:
        raise ReleaseValidationError(f"immutable pins require intentional review; due {due}")


def _validate_documentation(root: Path) -> None:
    documentation = _read(root, "README.md") + "\n" + _read(root, "tarzan/README.md")
    required_claims = (
        "LIVE",
        "POINT_IN_TIME",
        "REPRODUCIBLE",
        "DUPLICATE_TARGET_ROW",
        "notional",
        "Unavailable",
        "ephemeral",
        "local-only",
        "Gemini",
        "BLOCK_NORMAL_AND_NOTIFY_FAILURE",
        "UNCERTAIN",
        "authorized resend",
        "--require-hashes",
        "workload",
    )
    missing = [claim for claim in required_claims if claim not in documentation]
    if missing:
        raise ReleaseValidationError(f"operational documentation is incomplete: {missing}")


def validate(manifest_path: Path) -> None:
    manifest_path = manifest_path.resolve()
    root = manifest_path.parents[1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ReleaseValidationError("unsupported release manifest schema")
    if manifest.get("python_version") != "3.12":
        raise ReleaseValidationError("release Python version drift")
    _validate_scope_and_review(manifest)
    _validate_requirements(root)
    _validate_workflow(root)
    _validate_versions(root, manifest)
    _validate_documentation(root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    validate(args.manifest)
    print("Tarzan release declarations are consistent and positively scoped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
