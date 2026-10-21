"""Download Tarzan input CSVs from a private Google Drive folder.

Used by the GitHub Actions newsletter workflow when the repo is public:
your order list and targets CSVs live in a Drive folder shared (read-only)
with a service account, instead of being committed to the repo.

Required environment variables:
    DRIVE_FOLDER_ID                  ID portion after "/folders/" in its URL
    GOOGLE_DRIVE_CREDENTIALS_JSON    Full service-account JSON key, as a
                                     single-line string (newlines escaped).

The folder must contain at least:
    - order_list.csv
    - targets.csv

Optionally: targets_per_holding.csv. Anything else in the folder is ignored.

Used by :mod:`tarzan.delivery` (the newsletter service); the CI workflow
reaches it through the ``scripts/send_newsletter.py`` shim.
"""

from __future__ import annotations

import io
import hashlib
import inspect
import json
import logging
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Optional

from tarzan.runtime.io_utils import atomic_write_bytes

logger = logging.getLogger(__name__)

# Files the order-only pipeline can use. Downloaded when present; none is
# individually mandatory here (the caller decides which combination is
# enough — typically order_list.csv + targets.csv, with the optional
# per-holding targets file).
KNOWN_INPUT_FILES = (
    "order_list.csv", "targets.csv", "targets_per_holding.csv",
)
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
MAX_LIST_PAGES = 100
MAX_LIST_FILES = 10_000
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 256 * 1024
_DRIVE_ID = re.compile(r"^[A-Za-z0-9_-]{1,200}$")
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,199}$")


class DownloadedInputSet(Mapping[str, Path]):
    """Own only its private staging directory; never own caller files."""

    def __init__(self, directory: tempfile.TemporaryDirectory, paths):
        self._directory = directory
        self.directory = Path(directory.name)
        self._paths = MappingProxyType(dict(paths))
        self._closed = False

    def __getitem__(self, key):
        if self._closed:
            raise RuntimeError("downloaded input set is closed")
        return self._paths[key]

    def __iter__(self):
        return iter(self._paths)

    def __len__(self):
        return len(self._paths)

    def close(self):
        if not self._closed:
            self._closed = True
            self._directory.cleanup()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("downloaded input set is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class _BoundedBuffer(io.BytesIO):
    def write(self, data):
        if self.tell() + len(data) > MAX_DOWNLOAD_BYTES:
            raise RuntimeError("Drive input exceeds the download size limit")
        return super().write(data)


def _drive_service(credentials_json: str):
    """Build an authenticated read-only Drive client from a JSON key."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    import httplib2
    from google_auth_httplib2 import AuthorizedHttp

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
    # Native Google OAuth verification, with a real socket timeout. Credential
    # presence alone never authorizes a download; the listing must succeed.
    http = AuthorizedHttp(creds, http=httplib2.Http(timeout=20))
    return build("drive", "v3", http=http, cache_discovery=False)


def _select_files(service, folder_id, filenames, required):
    wanted = set(filenames)
    query = f"'{folder_id}' in parents and trashed = false"
    by_name = {}
    seen_tokens = set()
    page_token = None
    count = 0
    for _ in range(MAX_LIST_PAGES):
        arguments = dict(
            q=query,
            fields="nextPageToken,files(id,name,mimeType,size,md5Checksum,version)",
            pageSize=100, supportsAllDrives=True, includeItemsFromAllDrives=True,
        )
        if page_token is not None:
            arguments["pageToken"] = page_token
        response = service.files().list(**arguments).execute()
        if not isinstance(response, dict) or not isinstance(response.get("files", []), list):
            raise RuntimeError("Drive returned an invalid file listing")
        batch = response.get("files", [])
        count += len(batch)
        if count > MAX_LIST_FILES:
            raise RuntimeError("Drive folder listing exceeds the file limit")
        for meta in batch:
            if not isinstance(meta, dict):
                raise RuntimeError("Drive returned invalid file metadata")
            name = meta.get("name")
            if name not in wanted:
                continue
            identifier = meta.get("id")
            if not isinstance(identifier, str) or not _DRIVE_ID.fullmatch(identifier):
                raise RuntimeError("Drive returned an invalid input file identifier")
            prior = by_name.get(name)
            if prior is not None and prior != meta:
                raise RuntimeError(f"Drive input name is ambiguous: {name}")
            mime = str(meta.get("mimeType", ""))
            if mime.startswith("application/vnd.google-apps."):
                raise RuntimeError(f"Drive input must be an uploaded binary file: {name}")
            size = meta.get("size")
            if size is not None and (
                not str(size).isdigit() or int(size) > MAX_DOWNLOAD_BYTES
            ):
                raise RuntimeError(f"Drive input has an invalid or excessive size: {name}")
            checksum = meta.get("md5Checksum")
            if checksum is not None and not re.fullmatch(r"[a-fA-F0-9]{32}", str(checksum)):
                raise RuntimeError(f"Drive input has an invalid checksum: {name}")
            by_name[name] = dict(meta)
        next_token = response.get("nextPageToken")
        if not next_token:
            break
        if not isinstance(next_token, str) or next_token in seen_tokens:
            raise RuntimeError("Drive pagination token repeated or malformed")
        seen_tokens.add(next_token)
        page_token = next_token
    else:
        raise RuntimeError("Drive folder listing exceeds the page limit")
    missing = set(required) - by_name.keys()
    if missing:
        raise FileNotFoundError("Drive folder is missing required inputs: " + ", ".join(sorted(missing)))
    return by_name


def download_files(
    folder_id: str,
    credentials_json: str,
    filenames,
    dest_dir: Optional[Path] = None,
    *,
    required_filenames=(),
) -> DownloadedInputSet:
    """Download the requested files that exist in the folder.

    Skips files that are absent rather than raising, returning only the
    ones found. Lets the caller require whichever combination it needs
    (the order-only pipeline wants order_list.csv + targets.csv).

    Returns:
        Owned mapping of found filenames to private staged Paths. Consume it
        in a ``with`` block or call ``close``. Even an explicit dest_dir only
        contains a newly owned subdirectory, not overwritten caller files.
    """
    from googleapiclient.http import MediaIoBaseDownload

    if not isinstance(folder_id, str) or not _DRIVE_ID.fullmatch(folder_id):
        raise ValueError("Drive folder ID must be a nonempty native identifier")
    filenames = tuple(dict.fromkeys(filenames))
    required = tuple(dict.fromkeys(required_filenames))
    if any(not isinstance(name, str) or not _SAFE_FILENAME.fullmatch(name) for name in filenames):
        raise ValueError("Drive filenames must be safe single filename components")
    if not set(required).issubset(filenames):
        raise ValueError("required Drive filenames must be requested")
    service = _drive_service(credentials_json)
    by_name = _select_files(service, folder_id, filenames, required)
    if dest_dir is not None:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory = tempfile.TemporaryDirectory(prefix="tarzan-drive-", dir=dest_dir)
    staging = Path(directory.name)
    paths: dict[str, Path] = {}
    try:
        for name in filenames:
            if name not in by_name:
                continue
            meta = by_name[name]
            get_media = service.files().get_media
            parameters = inspect.signature(get_media).parameters
            media_arguments = {"fileId": meta["id"]}
            if "supportsAllDrives" in parameters or any(
                item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()
            ):
                media_arguments["supportsAllDrives"] = True
            request = get_media(**media_arguments)
            buf = _BoundedBuffer()
            # The real native SDK supports chunksize. Narrow injected ports may
            # omit it; the bounded buffer still enforces their maximum bytes.
            kwargs = (
                {"chunksize": DOWNLOAD_CHUNK_BYTES}
                if "chunksize" in inspect.signature(MediaIoBaseDownload).parameters else {}
            )
            downloader = MediaIoBaseDownload(buf, request, **kwargs)
            done = False
            chunks = 0
            while not done:
                _, done = downloader.next_chunk()
                chunks += 1
                if chunks > MAX_DOWNLOAD_BYTES // DOWNLOAD_CHUNK_BYTES + 2:
                    raise RuntimeError("Drive download did not complete within the chunk limit")
            data = buf.getvalue()
            size = meta.get("size")
            checksum = meta.get("md5Checksum")
            if size is not None and len(data) != int(size):
                raise RuntimeError(f"Drive input changed after listing (size): {name}")
            if checksum is not None and hashlib.md5(data, usedforsecurity=False).hexdigest() != checksum.lower():
                raise RuntimeError(f"Drive input changed after listing (checksum): {name}")
            local = staging / name
            atomic_write_bytes(local, data)
            logger.info("Drive: downloaded required input %s", name)
            paths[name] = local
        return DownloadedInputSet(directory, paths)
    except BaseException:
        directory.cleanup()
        raise
