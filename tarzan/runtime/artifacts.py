"""Atomic, correlated, local-only run artifact lifecycle."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from tarzan.runtime.io_utils import atomic_write_bytes, canonical_json_bytes


MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class StorageDescriptor:
    storage_scope: str = "local"
    automation_local_ephemeral: bool = False
    retention_guarantee: str = "none"
    execution_environment: str = "local"


@dataclass(frozen=True)
class LocalOnlyWorkbook:
    content: bytes
    filename: str = "what_if.xlsx"


class LocalArtifactWriter:
    """Write old-or-new committed files and publish checksum manifest last."""

    def __init__(
        self,
        root: Path,
        attempt_id: str,
        *,
        storage: StorageDescriptor = StorageDescriptor(),
    ) -> None:
        self.directory = Path(root) / attempt_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.attempt_id = attempt_id
        self.storage = storage

    def _atomic_write(self, name: str, content: bytes) -> Path:
        destination = self.directory / name
        atomic_write_bytes(destination, content)
        return destination

    @staticmethod
    def _json_bytes(value: Any) -> bytes:
        return canonical_json_bytes(value, default=str)

    def checkpoint(
        self,
        *,
        analysis_id: str,
        summary: Mapping[str, Any],
        ledger_entries: Iterable[Mapping[str, Any]],
        report_html: str,
        publication_state: str,
        newsletter_html: Optional[str] = None,
        what_if: Optional[LocalOnlyWorkbook] = None,
        delivery_state: str = "CLAIMED",
    ) -> Path:
        """Commit local evidence immediately before irreversible delivery."""
        return self.finalize(
            analysis_id=analysis_id,
            summary=summary,
            ledger_entries=ledger_entries,
            report_html=report_html,
            publication_state=publication_state,
            newsletter_html=newsletter_html,
            what_if=what_if,
            delivery_state=delivery_state,
        )

    def finalize(
        self,
        *,
        analysis_id: str,
        summary: Mapping[str, Any],
        ledger_entries: Iterable[Mapping[str, Any]],
        report_html: str,
        publication_state: str,
        newsletter_html: Optional[str] = None,
        what_if: Optional[LocalOnlyWorkbook] = None,
        delivery_state: str = "NOT_APPLICABLE",
    ) -> Path:
        files: dict[str, bytes] = {
            "summary.json": self._json_bytes(summary),
            "ledger.jsonl": b"\n".join(self._json_bytes(entry) for entry in ledger_entries) + b"\n",
            "report.html": report_html.encode("utf-8"),
        }
        if newsletter_html is not None:
            files["newsletter.html"] = newsletter_html.encode("utf-8")
        if what_if is not None:
            files[what_if.filename] = what_if.content

        checksums: dict[str, str] = {}
        for name, content in files.items():
            self._atomic_write(name, content)
            checksums[name] = hashlib.sha256(content).hexdigest()

        from tarzan.version import APPLICATION_VERSION

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "application_version": APPLICATION_VERSION,
            "attempt_id": self.attempt_id,
            "analysis_id": analysis_id,
            "publication_state": publication_state,
            "delivery_state": delivery_state,
            "storage_scope": self.storage.storage_scope,
            "execution_environment": self.storage.execution_environment,
            "automation_local_ephemeral": self.storage.automation_local_ephemeral,
            "retention_guarantee": self.storage.retention_guarantee,
            "files": checksums,
        }
        # Commit marker is deliberately last: readers ignore a directory whose
        # manifest is absent or whose checksums do not match.
        return self._atomic_write("manifest.json", self._json_bytes(manifest))
