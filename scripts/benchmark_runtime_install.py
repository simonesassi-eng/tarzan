"""Isolated, hash-locked install plus first-import A/B; no portfolio or delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


PROBE = """
import importlib
import importlib.metadata
import json
import socket
import sys
import time
def denied(*args, **kwargs):
    raise AssertionError("Network is not authorized in the first-import probe")
socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied
start = time.monotonic()
for name in (
    "scripts.send_newsletter", "tarzan.orchestrator",
    "tarzan.engine.metrics", "tarzan.engine.rebalancer",
    "tarzan.data.enricher", "tarzan.export.newsletter",
):
    importlib.import_module(name)
core_seconds = time.monotonic() - start
for name in (
    "google.oauth2.service_account", "googleapiclient.discovery",
    "googleapiclient.http", "google_auth_httplib2",
):
    importlib.import_module(name)
seconds = time.monotonic() - start
inventory = {
    row.metadata["Name"].lower().replace("_", "-"): row.version
    for row in importlib.metadata.distributions()
}
print(json.dumps({
    "import_seconds": seconds, "core_import_seconds": core_seconds,
    "drive_import_seconds": seconds - core_seconds, "inventory": inventory,
}))
"""


def run(command, *, cwd, env):
    result = subprocess.run(
        command, cwd=cwd, env=env, check=True, capture_output=True, text=True,
        timeout=240,
    )
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 4 or args.samples % 4:
        parser.error("samples must be a positive multiple of four, at least four")
    root, wheels = args.root.resolve(), args.wheels.resolve()
    if not wheels.is_dir():
        parser.error("wheel directory must already exist")
    sys.path.insert(0, str(root))
    from scripts.install_runtime_dependencies import runtime_requirements

    lock = (root / "requirements.txt").read_text()
    requirements = runtime_requirements(lock)
    expected = {
        line.split("==", 1)[0].lower().replace("_", "-"):
        line.split("==", 1)[1].split()[0]
        for line in requirements.splitlines()
    }
    env = {
        name: value for name, value in os.environ.items()
        if name in {"PATH", "SYSTEMROOT", "TMPDIR", "LANG", "LC_ALL", "SSL_CERT_FILE"}
    }
    env.update(
        CI="1", TARZAN_DISABLE_AI="1", TARZAN_DISABLE_CACHE="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1", PIP_CONFIG_FILE=os.devnull,
    )
    observations = []
    with tempfile.TemporaryDirectory(prefix="tarzan-install-ab-") as scratch:
        task_dir = Path(scratch)
        task_dir.chmod(0o700)
        projected = task_dir / "runtime.txt"
        projected.write_text(requirements)
        projected.chmod(0o600)
        # Compile source consistently before imports, as the real workflow does.
        # Dependencies are never compiled separately in the --no-compile arm.
        for index in range(args.samples):
            no_compile = (index % 4) in (1, 2)  # balanced ABBA order
            environment = task_dir / f"venv-{index}"
            bootstrap = time.monotonic()
            run([sys.executable, "-m", "venv", str(environment)], cwd=root, env=env)
            bootstrap_seconds = time.monotonic() - bootstrap
            python = str(environment / "bin" / "python")
            command = [
                python, "-m", "pip", "install", "--require-hashes", "--no-index",
                "--find-links", str(wheels), "-r", str(projected),
            ]
            if no_compile:
                command.append("--no-compile")
            install_started = time.monotonic()
            run(command, cwd=root, env=env)
            install_seconds = time.monotonic() - install_started
            run([python, "-m", "pip", "check"], cwd=root, env=env)
            compile_started = time.monotonic()
            run([python, "-m", "compileall", "-q", "tarzan", "scripts"], cwd=root, env=env)
            compile_seconds = time.monotonic() - compile_started
            probe = json.loads(run([python, "-c", PROBE], cwd=root, env=env))
            actual = {name: probe["inventory"].get(name) for name in expected}
            if actual != expected or set(probe["inventory"]) - set(expected) - {"pip", "setuptools"}:
                raise AssertionError("First-import environment differs from the exact runtime lock")
            observations.append({
                "arm": "no_compile" if no_compile else "default",
                "bootstrap_seconds": bootstrap_seconds,
                "install_seconds": install_seconds,
                "tarzan_compile_seconds": compile_seconds,
                "first_import_seconds": probe["import_seconds"],
                "core_import_seconds": probe["core_import_seconds"],
                "drive_import_seconds": probe["drive_import_seconds"],
                "install_compile_import_seconds": (
                    install_seconds + compile_seconds + probe["import_seconds"]
                ),
            })
            print(json.dumps({"observation": observations[-1]}), flush=True)
    summaries = {
        arm: {
            field: statistics.median(row[field] for row in observations if row["arm"] == arm)
            for field in (
                "install_seconds", "tarzan_compile_seconds",
                "first_import_seconds", "install_compile_import_seconds",
            )
        }
        for arm in ("default", "no_compile")
    }
    print(json.dumps({
        "schema": 1,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "lock_sha256": hashlib.sha256(lock.encode()).hexdigest(),
        "runtime_packages": len(expected),
        "order": [row["arm"] for row in observations],
        "medians": summaries,
        "observations": observations,
        "network_authorized": False,
        "email_authorized": False,
        "boundary": "local-wheel install + Tarzan compile + first operational module imports",
        "not_a_delivery_latency": True,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
