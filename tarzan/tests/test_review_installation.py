"""Legacy installation resolves the same locked environment without network."""

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from tarzan.data import manual_proxies, price_cache


ROOT = Path(__file__).resolve().parents[2]
EXACT = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)")


def active_lines(path):
    return [
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def test_legacy_manifest_only_includes_the_supported_root_hash_lock():
    legacy = ROOT / "tarzan" / "requirements.txt"
    assert active_lines(legacy) == ["-r ../requirements.txt"]
    assert (legacy.parent / "../requirements.txt").resolve() == ROOT / "requirements.txt"
    locked = active_lines(ROOT / "requirements.txt")
    assert locked and all(EXACT.match(line) and "--hash=sha256:" in line for line in locked)


def test_every_direct_dependency_is_declared_at_its_existing_locked_version():
    locked = {
        canonical(match[1]): match[2]
        for line in active_lines(ROOT / "requirements.txt")
        if (match := EXACT.match(line))
    }
    direct = active_lines(ROOT / "requirements.in")
    assert all(EXACT.fullmatch(line) for line in direct)
    for line in direct:
        match = EXACT.fullmatch(line)
        assert locked[canonical(match[1])] == match[2]
    assert "markupsafe" in {canonical(EXACT.fullmatch(line)[1]) for line in direct}


@pytest.mark.parametrize("working_directory,manifest", [
    (ROOT, "tarzan/requirements.txt"),
    (ROOT / "tarzan", "requirements.txt"),
])
def test_native_pip_accepts_both_legacy_paths_with_hash_enforcement_and_no_network(
    working_directory, manifest,
):
    environment = {
        **os.environ, "PIP_CONFIG_FILE": os.devnull, "PIP_NO_INDEX": "1",
        "PIP_EXTRA_INDEX_URL": "", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    completed = subprocess.run([
        sys.executable, "-m", "pip", "install", "--dry-run", "--require-hashes",
        "--no-index", "--no-cache-dir", "--disable-pip-version-check", "-r", manifest,
    ], cwd=working_directory, env=environment, capture_output=True, text=True, timeout=30)
    # No indexes or installs are authorized. Missing SDKs must fail rather than
    # skip a native boundary test in an unsupported, partly installed host.
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Would install" not in completed.stdout


def test_explicit_local_manual_ingestion_and_list_cli_remain_available(monkeypatch, tmp_path):
    cache = tmp_path / "synthetic-cache"
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(cache))
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    source = tmp_path / "synthetic.csv"
    source.write_text("date,level\n2026-01-02,100\n2026-01-05,105\n")
    ingested = manual_proxies.ingest("SYNTHETIC", str(source))
    assert ingested is not None
    series = manual_proxies.get_series("SYNTHETIC")
    assert series.equals(ingested)
    assert series.tolist() == [100.0, 105.0]
    assert price_cache.list_history_symbols(prefix="MANUAL_") == ["MANUAL_SYNTHETIC"]
    completed = subprocess.run(
        [sys.executable, "-m", "tarzan.data.manual_proxies", "list"],
        cwd=ROOT, capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SYNTHETIC" in completed.stdout
