"""Download Tarzan input CSVs from a private Google Drive folder.

Used by the GitHub Actions newsletter workflow when the repo is public:
your holdings and targets CSVs live in a Drive folder shared (read-only)
with a service account, instead of being committed to the repo.

Required environment variables:
    DRIVE_FOLDER_ID                  ID portion of the Drive folder URL
                                     (e.g. "1I9BaXVO1R7cpeps-USyrpWB759YQX48a")
    GOOGLE_DRIVE_CREDENTIALS_JSON    Full service-account JSON key, as a
                                     single-line string (newlines escaped).

The folder must contain at least:
    - holdings.csv
    - targets.csv

Anything else in the folder is ignored.
"""

from __future__ import annotations

import io
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REQUIRED_FILES = ("holdings.csv", "targets.csv")
# Files the order-only pipeline can use. Downloaded when present; none is
# individually mandatory (the caller decides which combination is enough).
KNOWN_INPUT_FILES = (
    "order_list.csv", "targets.csv", "targets_per_holding.csv", "holdings.csv",
)
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _drive_service(credentials_json: str):
    """Build an authenticated read-only Drive client from a JSON key."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    try:
        info = json.loads(credentials_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            "GOOGLE_DRIVE_CREDENTIALS_JSON is not valid JSON. "
            "Make sure you pasted the full service-account key, including "
            "all newlines (escape them as \\n inside GitHub Secrets)."
        ) from e
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=DRIVE_SCOPES,
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def download_files(
    folder_id: str,
    credentials_json: str,
    filenames,
    dest_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Download the requested files that exist in the folder.

    Unlike :func:`download_inputs`, this skips files that are absent rather
    than raising, returning only the ones found. Lets the caller support
    several input layouts (order-only vs holdings) from one Drive folder.

    Returns:
        Dict mapping each found filename to its local Path (missing files
        are simply omitted).
    """
    from googleapiclient.http import MediaIoBaseDownload

    if dest_dir is None:
        dest_dir = Path(tempfile.mkdtemp(prefix="tarzan-drive-"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    service = _drive_service(credentials_json)
    query = f"'{folder_id}' in parents and trashed = false"
    response = service.files().list(
        q=query,
        fields="files(id, name, mimeType, size)",
        pageSize=100,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    by_name = {f["name"]: f for f in response.get("files", [])}

    wanted = set(filenames)
    paths: dict[str, Path] = {}
    for name in filenames:
        if name not in by_name:
            continue
        meta = by_name[name]
        local = dest_dir / name
        request = service.files().get_media(fileId=meta["id"])
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        data = buf.getvalue()
        local.write_bytes(data)
        logger.info("Drive: downloaded %s (%d bytes) → %s", name, len(data), local)
        paths[name] = local

    skipped = wanted - set(paths)
    if skipped:
        logger.info("Drive: not present in folder (skipped): %s", ", ".join(sorted(skipped)))
    return paths


def download_inputs(
    folder_id: str,
    credentials_json: str,
    dest_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Download the required holdings + targets CSVs (raises if missing).

    Kept for backward compatibility with the holdings-based flow. New code
    should prefer :func:`download_files`, which tolerates absent files and
    supports the order-only layout.

    Raises:
        FileNotFoundError: If a required CSV is missing from the folder.
    """
    paths = download_files(folder_id, credentials_json, KNOWN_INPUT_FILES, dest_dir)
    missing = [name for name in REQUIRED_FILES if name not in paths]
    if missing:
        raise FileNotFoundError(
            f"Drive folder {folder_id!r} is missing: {', '.join(missing)}. "
            f"Expected at least: {', '.join(REQUIRED_FILES)}."
        )
    return {name: paths[name] for name in REQUIRED_FILES}
