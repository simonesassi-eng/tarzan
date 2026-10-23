"""Persist only authenticated, encrypted market-cache JSON across fresh runners.

Explicit namespaces exclude inputs, claims, output, SDK cookies and manual or
licensed histories. Restored bytes retain their original timestamps/checksums:
restoration does not renew TTLs or make an old price newly observed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re

from tarzan.data import price_cache
from tarzan.runtime.io_utils import atomic_write_bytes, canonical_json_bytes
from tarzan.runtime import sealed


SCHEMA_VERSION = "1"
PURPOSE = "tarzan-market-cache-v1"
FILENAME = "market.sealed"
MAX_FILES = 4096
MAX_FILE_BYTES = 8 * 1024 * 1024
_SYMBOL = re.compile(r"[A-Za-z0-9^][A-Za-z0-9^=._-]{0,63}\Z")
_MAPS = {
    "resolution/isin_to_symbol.json": "resolution",
    "instrument_profiles/profiles_v1.json": price_cache._INSTRUMENT_PROFILE_NAMESPACE,
    "timed/openfigi.json": "openfigi",
    # Compatibility with deployed archives, not an authority to skip probes.
    "timed/no_listing.json": "no_listing",
    "timed/bond_history_v1.json": "bond_history_v1",
    "geo/geo_breakdown.json": "geo",
    "ter/ter_by_isin.json": "ter",
    "xref/ticker_isin.json": "ticker_isin",
}
_PRICE_COLUMNS = frozenset({
    "Open", "High", "Low", "Close", "Adj Close", "Price Close", "Volume",
    "Dividends", "Stock Splits", "Capital Gains", "__value__",
})


def _pairs_unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate private snapshot field")
        result[key] = value
    return result


def _json(content):
    return json.loads(content, object_pairs_hook=_pairs_unique)


def _cache_document(relative: str, document) -> bool:
    """Validate the ordinary cache envelope, plus its explicit archive scope."""
    if not isinstance(document, dict) or set(document) != {
        "schema_version", "namespace", "entries", "checksum",
    }:
        raise ValueError("invalid cached JSON envelope")
    body = {name: value for name, value in document.items() if name != "checksum"}
    if (
        document["schema_version"] != price_cache._CACHE_SCHEMA_VERSION
        or document["checksum"] != hashlib.sha256(price_cache._canonical_json(body)).hexdigest()
        or not price_cache._validate_json_value(document["entries"])
    ):
        raise ValueError("invalid cached JSON integrity")
    entries = document["entries"]
    if relative in _MAPS:
        if document["namespace"] != _MAPS[relative] or not isinstance(entries, dict):
            raise ValueError("market cache namespace mismatch")
        if relative == "timed/bond_history_v1.json":
            from tarzan.data.bond_history import cache_entry_valid

            if any(
                not isinstance(entry, dict) or set(entry) != {"ts", "value"}
                or type(entry["ts"]) not in (int, float)
                or entry["ts"] <= 0 or entry["ts"] > 253402300799
                or not math.isfinite(entry["ts"])
                or not cache_entry_valid(entry["value"], isin)
                for isin, entry in entries.items()
            ):
                raise ValueError("invalid public bond-history cache scope")
        return True
    parts = Path(relative).parts
    if len(parts) != 2 or parts[0] != "history" or not parts[1].endswith(".json"):
        raise ValueError("cache archive path is outside the market allowlist")
    if (
        document["namespace"] != "history" or not isinstance(entries, dict)
        or entries.get("repair_version") != price_cache._HISTORY_REPAIR_VERSION
    ):
        raise ValueError("incompatible market history")
    symbol = entries.get("symbol")
    if (
        not isinstance(symbol, str) or not _SYMBOL.fullmatch(symbol)
        or parts[1] != f"{price_cache._safe(symbol)}.json"
    ):
        return False  # manual/licensed identifiers must never leave the runner
    schema = entries.get("table", {}).get("schema", {})
    fields = schema.get("fields", [])
    primary_key = schema.get("primaryKey", [])
    if (
        not fields or not isinstance(primary_key, list) or len(primary_key) != 1
        or sum(field.get("type") == "datetime" for field in fields) != 1
        or any(
            field.get("name") != primary_key[0] if field.get("type") == "datetime"
            else field.get("name") not in _PRICE_COLUMNS
            or field.get("type") not in {"number", "integer"}
            for field in fields
        )
    ):
        return False  # price tables only, never custom account/input columns
    return True


def _locations(directory):
    staging = Path(directory).expanduser().absolute()
    private = price_cache.cache_dir().absolute()
    for path in (private, staging):
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("cache locations cannot use symlinks")
    private, staging = private.resolve(), staging.resolve()
    if staging == private or staging in private.parents or private in staging.parents:
        raise ValueError("ciphertext staging must be separate from the private cache")
    if staging.exists() and (
        not staging.is_dir()
        or any(child.name != FILENAME or child.is_symlink() or not child.is_file()
               for child in staging.iterdir())
    ):
        raise ValueError("ciphertext staging contains an unrelated file")
    return private, staging


def export_snapshot(directory, key: bytes) -> int:
    private, staging = _locations(directory)
    paths = [private / name for name in _MAPS]
    history = private / "history"
    if history.is_symlink():
        raise ValueError("cache history cannot be a symlink")
    if history.is_dir():
        paths.extend(sorted(history.glob("*.json")))
    if len(paths) > MAX_FILES:
        raise ValueError("cache archive exceeds file count")
    files = {}
    for path in paths:
        if not path.exists():
            continue
        relative = path.relative_to(private).as_posix()
        # A corrupt cache is already a miss for ordinary readers. Do not
        # prevent other validated cache entries from being saved.
        try:
            document = _json(sealed.read_bounded_file(path, MAX_FILE_BYTES))
            if _cache_document(relative, document):
                files[relative] = document
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    document = {
        "schema_version": SCHEMA_VERSION,
        "history_repair_version": price_cache._HISTORY_REPAIR_VERSION,
        "files": files,
    }
    # No unencrypted intermediate archive is written to staging.
    sealed.write_snapshot(
        staging / FILENAME, canonical_json_bytes(document), key, purpose=PURPOSE,
    )
    return len(files)


@dataclass(frozen=True)
class _ImportResult:
    restored_files: int
    public_macro_complete: bool = False


def _authenticated_public_macro_complete(documents, *, reference_time=None) -> bool:
    """Prove a usable recent head for every fixed macro, never a live quote.

    Only the authenticated import may supply ``documents``: each entry was
    validated and either written or found byte-identical at its destination.
    No map TTL is renewed, and every normal history tail/intraday refresh still
    runs. The existing refresh-tail window is a conservative bootstrap guard;
    old, partial, future or disabled cache evidence keeps the public fallback.
    """
    from tarzan.data import public_cache
    import pandas as pd

    if not price_cache.is_enabled():
        return False
    now = reference_time if reference_time is not None else datetime.now(timezone.utc)
    if not isinstance(now, datetime) or now.tzinfo is None:
        return False
    now = now.astimezone(timezone.utc)
    try:
        for symbol in public_cache.PUBLIC_SYMBOLS:
            cached = documents.get(f"history/{price_cache._safe(symbol)}.json")
            if cached is None:
                return False
            entries = cached["entries"]
            if (
                entries.get("kind") != "dataframe"
                or entries.get("symbol") != symbol
                or entries.get("repair_version") != price_cache._HISTORY_REPAIR_VERSION
            ):
                return False
            table = entries["table"]
            rows, fields = table["data"], table["schema"]["fields"]
            if not isinstance(rows, list) or not 1 <= len(rows) <= public_cache.MAX_ROWS:
                return False
            names = {field["name"] for field in fields}
            numeric = {field["name"] for field in fields if field["type"] != "datetime"}
            # pandas can coerce a JSON boolean/string into a number. Such bytes
            # must never become the authority to skip an independent fallback.
            if any(
                not isinstance(row, dict) or set(row) != names
                or any(row[name] is not None and type(row[name]) not in (int, float)
                       for name in numeric)
                for row in rows
            ):
                return False
            frame = pd.read_json(io.StringIO(json.dumps(table)), orient="table")
            public_cache._table(frame)  # same bounded dates/values/columns contract
            closes = frame["Close"].dropna()
            if closes.empty or not math.isfinite(float(closes.iloc[-1])):
                return False
            latest_label = frame.index[-1].to_pydatetime()
            observed = closes.index[-1].to_pydatetime()
            # Yahoo's aligned daily tables can end with a null placeholder.
            # Like the actual macro consumer, date the last REAL close, never
            # that padding; even a null future row keeps the guard conservative.
            if latest_label.tzinfo is None:
                latest_label = latest_label.replace(tzinfo=timezone.utc)
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            if latest_label.astimezone(timezone.utc) > now:
                return False
            # This date guard is not an invented venue or quote freshness rule.
            age = (now - observed.astimezone(timezone.utc)).total_seconds()
            if not 0 <= age <= price_cache.REFRESH_TAIL_DAYS * 86400:
                return False
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return False
    return True


def _import_snapshot(
    directory, key: bytes, *, prove_public_macro=False, reference_time=None,
) -> _ImportResult:
    private, staging = _locations(directory)
    path = staging / FILENAME
    if not path.exists():
        return _ImportResult(0)
    document = _json(sealed.unseal(
        sealed.read_bounded_file(path, sealed.MAX_SEALED_BYTES), key, purpose=PURPOSE,
    ))
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "history_repair_version", "files"}
        or document["schema_version"] != SCHEMA_VERSION
        or document["history_repair_version"] != price_cache._HISTORY_REPAIR_VERSION
        or not isinstance(document["files"], dict) or len(document["files"]) > MAX_FILES
    ):
        raise ValueError("incompatible protected market snapshot")
    validated = []
    for relative, cache_document in document["files"].items():
        if not _cache_document(relative, cache_document):
            raise ValueError("protected snapshot contains a non-market table")
        content = canonical_json_bytes(cache_document)
        if len(content) > MAX_FILE_BYTES:
            raise ValueError("protected cache file exceeds byte limit")
        target = private / relative
        if any(parent.is_symlink() for parent in (target, *target.parents)):
            raise ValueError("protected cache destination cannot use symlinks")
        if target.exists() and not target.is_file():
            raise ValueError("protected cache destination is not a file")
        validated.append((relative, target, content, cache_document))
    # Authenticate and validate ALL files/paths before writing the first one.
    # Existing run-local entries are newer authority and must not be replaced.
    count = 0
    authenticated = {}
    for relative, target, content, cache_document in validated:
        if target.exists():
            if prove_public_macro:
                try:
                    identical = sealed.read_bounded_file(target, MAX_FILE_BYTES) == content
                except (OSError, ValueError):
                    identical = False
                if identical:
                    authenticated[relative] = cache_document
            continue
        sealed.owned_directory(target.parent)
        atomic_write_bytes(target, content, fsync_dir=True)
        count += 1
        if prove_public_macro:
            authenticated[relative] = cache_document
    complete = bool(
        prove_public_macro
        and _authenticated_public_macro_complete(authenticated, reference_time=reference_time)
    )
    return _ImportResult(count, complete)


def import_snapshot(directory, key: bytes) -> int:
    """Import the snapshot without changing the established count-only API."""
    return _import_snapshot(directory, key).restored_files


def _output(name, value):
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("key-id", "generate-key", "import", "export"))
    parser.add_argument("--directory", type=Path, default=Path.home() / ".cache" / "tarzan-sealed")
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    from tarzan import log_redaction
    log_redaction.install()
    try:
        if args.operation == "import":
            # A missing file, invalid key/tag/schema or optional proof failure
            # must keep the fallback enabled even with continue-on-error.
            _output("public_macro_complete", "false")
        if args.operation == "generate-key":
            if args.key_file is None:
                raise ValueError("generate-key requires a local recovery file")
            sealed.generate_key(args.key_file)
            print("Dedicated snapshot key created in the local recovery file.")
            return 0
        key = sealed.configured_key(key_file=args.key_file)
        if args.operation == "key-id":
            _output("key_id", sealed.key_id(key))
            _output("enabled", "true")
            print("Protected cache key configuration verified.")
            return 0
        if args.operation == "import":
            result = _import_snapshot(args.directory, key, prove_public_macro=True)
            count = result.restored_files
            if result.public_macro_complete:
                _output("public_macro_complete", "true")
        else:
            count = export_snapshot(args.directory, key)
            _output("nonempty", "true" if count else "false")
        print("Protected market cache restored." if args.operation == "import" and count
              else "Protected market cache miss; fresh data will be requested."
              if args.operation == "import" else "Protected market cache export verified.")
        return 0
    except Exception:
        # No paths, symbols, key material or provider data in a public console.
        print("Protected market cache unavailable; no unverified snapshot is authorized.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
