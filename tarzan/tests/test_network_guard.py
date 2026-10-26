"""The offline test gate must detect swallowed I/O without exposing requests."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import build_opener

import pytest

from tarzan.tests.network_guard import NetworkGuard


@pytest.mark.parametrize("module,owner,name,channel", [
    ("socket", "socket", "connect", "socket.connect"),
    ("socket", "socket", "connect_ex", "socket.connect_ex"),
    ("socket", None, "create_connection", "socket.create_connection"),
    ("socket", None, "getaddrinfo", "socket.getaddrinfo"),
    ("curl_cffi.requests", "Session", "request", "curl.request"),
    ("requests", "Session", "request", "requests.request"),
    ("urllib.request", None, "urlopen", "urllib.urlopen"),
    ("smtplib", None, "SMTP", "smtp"),
    ("smtplib", None, "SMTP_SSL", "smtp.ssl"),
])
def test_every_native_transport_is_denied_and_recorded(module, owner, name, channel):
    from importlib import import_module

    guard = NetworkGuard()
    with pytest.MonkeyPatch.context() as patch:
        guard.install(patch)
        target = import_module(module)
        if owner:
            target = getattr(target, owner)
        with pytest.raises(AssertionError, match="Unstubbed external transport"):
            getattr(target, name)("synthetic-private-request")
    with pytest.raises(AssertionError, match=channel.replace(".", r"\.")):
        guard.assert_unused()


def test_guard_never_formats_private_arguments():
    class Private:
        def __repr__(self):
            raise RuntimeError("request arguments must not be inspected")

    guard = NetworkGuard()
    with pytest.raises(AssertionError) as denied:
        guard.denied("synthetic")(Private(), secret=Private())
    assert str(denied.value) == "Unstubbed external transport: synthetic"
    with pytest.raises(AssertionError) as recorded:
        guard.assert_unused()
    assert str(recorded.value) == "Unstubbed external transport attempts: {'synthetic': 1}"


def test_caught_worker_attempts_still_fail_the_owner_gate():
    guard = NetworkGuard()

    def worker(_index):
        try:
            guard.denied("synthetic-worker")()
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(32)))
    with pytest.raises(AssertionError) as error:
        guard.assert_unused()
    assert str(error.value).endswith("{'synthetic-worker': 32}")


def test_explicit_http_stub_preserves_native_requests_response_and_json_parser(monkeypatch):
    import requests
    from tarzan.data import justetf

    seen = []

    def request(session, method, url, **kwargs):
        seen.append((method, url))
        response = requests.Response()
        response.status_code = 200
        response.headers["content-type"] = "application/json"
        response._content = json.dumps({"native": True}).encode()
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    assert justetf._get_json("https://synthetic.example.invalid/") == {"native": True}
    assert seen == [("get", "https://synthetic.example.invalid/")]


def test_prebound_stdlib_opener_is_still_blocked_at_the_socket_boundary():
    # Native bound method captured before this guard replaces socket ports.
    # This also works under an independently installed outer urlopen denial.
    imported_open = build_opener().open
    guard = NetworkGuard()
    with pytest.MonkeyPatch.context() as patch:
        guard.install(patch)
        with pytest.raises(AssertionError, match="socket.create_connection"):
            imported_open("https://synthetic.example.invalid/", timeout=1)
    with pytest.raises(AssertionError) as error:
        guard.assert_unused()
    assert str(error.value).endswith("{'socket.create_connection': 1}")


def test_production_credentials_are_absent_unless_the_test_injects_them():
    assert not any(os.environ.get(name) for name in (
        "GEMINI_API_KEY", "SMTP_USER", "SMTP_PASS", "RECIPIENT_EMAIL",
        "DRIVE_FOLDER_ID", "GOOGLE_DRIVE_CREDENTIALS_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS", "TARZAN_CACHE_KEY_B64",
        "DELIVERY_CLAIM_ENDPOINT", "DELIVERY_CLAIM_TOKEN",
        "AUTHORIZED_RESEND_TOKEN", "GITHUB_TOKEN", "GH_TOKEN",
    ))


def test_real_pytest_teardown_rejects_a_swallowed_unstubbed_call(tmp_path):
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "conftest.py").write_text(
        "from tarzan.tests.conftest import _reject_unstubbed_transport\n",
    )
    (tmp_path / "test_swallowed.py").write_text(
        "import requests\n"
        "def test_caught():\n"
        "    try:\n"
        "        requests.get('https://synthetic.example.invalid/')\n"
        "    except Exception:\n"
        "        pass\n",
    )
    child_env = {
        name: os.environ[name]
        for name in ("PATH", "SYSTEMROOT", "TMPDIR", "LANG", "LC_ALL")
        if name in os.environ
    }
    child_env.update(PYTHONPATH=str(root), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(tmp_path), "--tb=short"],
        cwd=tmp_path, env=child_env, text=True, capture_output=True, timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert "1 passed, 1 error" in result.stdout
    assert "Unstubbed external transport attempts: {'requests.request': 1}" in result.stdout
    assert "synthetic.example.invalid" not in result.stdout
