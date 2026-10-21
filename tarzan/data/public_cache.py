"""A public Actions cache of fixed macro price tables, never portfolio state.

Export is a projection, not a directory copy. All queried holdings, identity
profiles, manual/licensed series, table names/attrs and unrelated files stay
private. Checksums detect corruption; they do not authenticate market prices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from tarzan.data import price_cache
from tarzan.data.market_quotes import MARKETS
from tarzan.runtime.io_utils import atomic_write_bytes, canonical_json_bytes


POLICY_ID = "fixed-macro-yahoo-history-v1"
SCHEMA_VERSION = "1"
PUBLIC_SYMBOLS = frozenset(symbol for _, symbol, _ in MARKETS)
PUBLIC_COLUMNS = (
    "Open", "High", "Low", "Close", "Adj Close", "Price Close",
    "Volume", "Dividends", "Stock Splits",
)
MAX_ROWS = 5000
MAX_BYTES = 10 * 1024 * 1024
_FILENAME = "snapshot.json"


def _table(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty or len(frame) > MAX_ROWS or "Close" not in frame:
        raise ValueError("public history must be a bounded price DataFrame")
    index = pd.DatetimeIndex(frame.index)
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("invalid public price dates")
    timezone = str(index.tz) if index.tz is not None else None
    if timezone is not None:
        ZoneInfo(timezone)  # Never retain a user-defined timezone description.
    columns = [name for name in PUBLIC_COLUMNS if name in frame.columns]
    if frame.columns.has_duplicates:
        raise ValueError("ambiguous public price columns")
    values = []
    for row in frame[columns].itertuples(index=False, name=None):
        numeric = []
        for item in row:
            if pd.isna(item):
                numeric.append(None)
            elif isinstance(item, bool):
                raise ValueError("boolean is not a public market price")
            else:
                value = float(item)
                if not math.isfinite(value):
                    raise ValueError("nonfinite public price")
                numeric.append(value)
        values.append(numeric)
    return {
        "columns": columns, "dates": [stamp.isoformat() for stamp in index],
        "timezone": timezone, "values": values,
    }


def _frame(table):
    if not isinstance(table, dict) or set(table) != {"columns", "dates", "timezone", "values"}:
        raise ValueError("unknown public table fields")
    columns, dates, values = table["columns"], table["dates"], table["values"]
    if (
        not isinstance(columns, list) or not columns or len(set(columns)) != len(columns)
        or "Close" not in columns or any(name not in PUBLIC_COLUMNS for name in columns)
        or not isinstance(dates, list) or not 1 <= len(dates) <= MAX_ROWS
        or not isinstance(values, list) or len(dates) != len(values)
        or any(not isinstance(row, list) or len(row) != len(columns) for row in values)
        or any(value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        ) for row in values for value in row)
        or any(not isinstance(stamp, str) or len(stamp) > 40 for stamp in dates)
    ):
        raise ValueError("invalid public price table")
    timezone = table["timezone"]
    if timezone is not None:
        if not isinstance(timezone, str) or len(timezone) > 64:
            raise ValueError("invalid public timezone")
        ZoneInfo(timezone)
        index = pd.DatetimeIndex(pd.to_datetime(dates, utc=True, format="ISO8601")).tz_convert(timezone)
    else:
        index = pd.DatetimeIndex(pd.to_datetime(dates, format="ISO8601"))
        if index.tz is not None:
            raise ValueError("unexpected public timezone")
    frame = pd.DataFrame(values, columns=columns, index=index, dtype=float)
    # Reprojection validates dates/order and leaves no table attrs or names.
    if _table(frame) != table:
        raise ValueError("public table is not canonical")
    return frame


def _directory(directory):
    path = Path(directory).expanduser()
    private = price_cache.cache_dir().resolve()
    resolved = path.resolve()
    if resolved == private or resolved in private.parents or private in resolved.parents:
        raise ValueError("public staging must be separate from the private cache")
    if any(parent.is_symlink() for parent in (path, *path.parents)) or (
        path.exists() and not path.is_dir()
    ):
        raise ValueError("public staging must be an owned directory")
    if path.exists() and any(
        child.name != _FILENAME or child.is_symlink() or not child.is_file()
        for child in path.iterdir()
    ):
        raise ValueError("public staging contains an unrelated file; refusing export")
    return path


def _read_snapshot(path):
    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("invalid public snapshot file")
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict) or set(document) != {
        "schema_version", "policy_id", "history_repair_version", "tables", "checksum",
    }:
        raise ValueError("unknown public snapshot fields")
    checksum = document.pop("checksum")
    if (
        document["schema_version"] != SCHEMA_VERSION
        or document["policy_id"] != POLICY_ID
        or document["history_repair_version"] != price_cache._HISTORY_REPAIR_VERSION
        or not isinstance(document["tables"], dict)
        or not set(document["tables"]).issubset(PUBLIC_SYMBOLS)
        or checksum != hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    ):
        raise ValueError("incompatible public snapshot")
    return {symbol: _frame(table) for symbol, table in document["tables"].items()}


def export_snapshot(directory) -> int:
    destination = _directory(directory)
    previous = _read_snapshot(destination / _FILENAME)
    tables = {}
    for symbol in sorted(PUBLIC_SYMBOLS):
        frame = price_cache.load_history(symbol)
        from_private = frame is not None
        if frame is None:
            frame = previous.get(symbol)
        if frame is None:
            continue
        # Require the current original-symbol cache contract, not a legacy
        # filename guess. No private namespace or glob is ever enumerated.
        if from_private:
            payload = price_cache._read_json(price_cache._history_path(symbol), "history", None)
            if not isinstance(payload, dict) or payload.get("symbol") != symbol:
                raise ValueError("public history lacks exact original-symbol evidence")
        tables[symbol] = _table(frame)
    document = {
        "schema_version": SCHEMA_VERSION, "policy_id": POLICY_ID,
        "history_repair_version": price_cache._HISTORY_REPAIR_VERSION, "tables": tables,
    }
    document["checksum"] = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    content = canonical_json_bytes(document)
    if len(content) > MAX_BYTES:
        raise ValueError("public snapshot exceeds byte limit")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination, 0o700)
    atomic_write_bytes(destination / _FILENAME, content, fsync_dir=True)
    return len(tables)


def import_snapshot(directory) -> int:
    source = _directory(directory)
    frames = _read_snapshot(source / _FILENAME)
    # Validate the entire snapshot before any private cache write.
    for symbol, frame in frames.items():
        price_cache.store_history(symbol, frame)
    return len(frames)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("import", "export"))
    parser.add_argument("--directory", default=str(Path.home() / ".cache" / "tarzan-public"))
    args = parser.parse_args(argv)
    # Corrupt private-cache readers may log before they return a miss. Protect
    # their console too; the projection's generic exception message is not enough.
    logging.basicConfig(level=logging.WARNING)
    from tarzan import log_redaction
    log_redaction.install()
    try:
        count = export_snapshot(args.directory) if args.operation == "export" else import_snapshot(args.directory)
        output = os.environ.get("GITHUB_OUTPUT")
        if output and args.operation == "export":
            with open(output, "a", encoding="utf-8") as handle:
                handle.write(f"nonempty={'true' if count else 'false'}\n")
    except Exception:
        # Staging errors must not leak private paths/data to public Actions logs.
        print("Public macro cache unavailable; no snapshot is authorized for save.")
        return 1
    print("Public macro cache projection verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
