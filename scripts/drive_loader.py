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
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def download_inputs(
    folder_id: str,
    credentials_json: str,
    dest_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Download required CSVs from the given Drive folder.

    Args:
        folder_id: Drive folder ID (the part after `/folders/` in the URL).
        credentials_json: Full service-account JSON key as a string.
        dest_dir: Optional destination directory. If omitted, a fresh
            temporary directory is created.

    Returns:
        Dict mapping each required filename to its local Path.

    Raises:
        FileNotFoundError: If a required CSV is missing from the folder.
        RuntimeError: If credentials are malformed or the API call fails.
    """
    # Imports are local so that local development without Drive integration
    # does not require these dependencies to be installed.
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    if dest_dir is None:
        dest_dir = Path(tempfile.mkdtemp(prefix="tarzan-drive-"))
    dest_dir.mkdir(parents=True, exist_ok=True)

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
    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    # List files directly under the given folder. We do NOT recurse —
    # this is by design: keep the layout simple, fail loudly if a
    # required file is missing.
    query = f"'{folder_id}' in parents and trashed = false"
    response = service.files().list(
        q=query,
        fields="files(id, name, mimeType, size)",
        pageSize=100,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = response.get("files", [])
    by_name = {f["name"]: f for f in files}

    missing = [name for name in REQUIRED_FILES if name not in by_name]
    if missing:
        raise FileNotFoundError(
            f"Drive folder {folder_id!r} is missing: {', '.join(missing)}. "
            f"Expected exactly: {', '.join(REQUIRED_FILES)}. "
            f"Found: {', '.join(sorted(by_name.keys())) or '(empty)'}."
        )

    paths: dict[str, Path] = {}
    for name in REQUIRED_FILES:
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

    return paths
