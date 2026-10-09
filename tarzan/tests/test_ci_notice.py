"""An issue built from code that failed its tests says so, and is still sent.

Since d4e0788 the suite no longer gates a send, so a red ``Checks`` run would
otherwise reach the reader unseen. Network-free.
"""

from __future__ import annotations

import io
import json

from tarzan import delivery

_HTML = ('<html><body style="x"><div style="display:none;">preheader</div>'
         '<table role="presentation"><tr><td>issue</td></tr></table></body></html>')


def _api(monkeypatch, runs):
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_SHA", "abc1234def")
    body = json.dumps({"workflow_runs": runs}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: io.BytesIO(body))


def test_a_failed_checks_run_puts_a_line_above_the_issue(monkeypatch):
    _api(monkeypatch, [{"name": "Checks", "conclusion": "failure"}])
    out = delivery._with_ci_notice(_HTML, delivery._failed_checks_sha())
    assert "tests failed" in out and "abc1234" in out
    # Below the hidden preheader, so the inbox preview text is unchanged.
    assert out.index("preheader") < out.index("tests failed") < out.index("<table")


def test_the_newest_checks_run_decides(monkeypatch):
    """A re-run that passed supersedes the failure before it (newest first)."""
    _api(monkeypatch, [{"name": "Checks", "conclusion": "success"},
                       {"name": "Checks", "conclusion": "failure"}])
    assert delivery._failed_checks_sha() == ""


def test_green_running_or_unknown_adds_nothing(monkeypatch):
    for runs in ([{"name": "Checks", "conclusion": "success"}],
                 [{"name": "Checks", "conclusion": None}],          # still running
                 [{"name": "Tarzan Newsletter", "conclusion": "failure"}], []):
        _api(monkeypatch, runs)
        assert delivery._with_ci_notice(_HTML, delivery._failed_checks_sha()) == _HTML


def test_an_unreachable_api_never_fails_the_issue(monkeypatch):
    _api(monkeypatch, [])

    def boom(req, timeout):
        raise OSError("down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert delivery._failed_checks_sha() == ""


def test_outside_actions_nothing_is_asked(monkeypatch):
    for k in ("GITHUB_TOKEN", "GITHUB_REPOSITORY", "GITHUB_SHA"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    assert delivery._failed_checks_sha() == ""
