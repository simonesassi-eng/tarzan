"""The delivery path must only be able to fail for delivery reasons.

The 2026-08-18 morning digest was never rendered because a self-imposed "pins
reviewed on/due on" date in the release manifest had passed the day before.
Nothing about the run was broken; a calendar was.

The first fix moved declaration audits into a push-triggered ``Checks`` workflow.
That reduced the blast radius but kept the apparatus — a 235-line validator and a
61-line manifest that restated constants living in the code, so that a script
could check the copy still matched its original. Most of it could not fail for a
real reason:

  * it asserted that the workflow running it, ran it (``if "validate_release.py"
    not in checks.yml: raise``) — unfailable, and dead the moment the file goes;
  * it compared ``schema_versions`` and ``application_version`` in the manifest
    against the code constants they were copied from;
  * it asserted fourteen magic words appear somewhere in the READMEs, which
    verifies vocabulary rather than accuracy and resists editing the prose;
  * its positive-scope check validated the SHAPE of each declared path, never its
    existence — which is why ``tarzan/backtest.py`` sat in the manifest long after
    the module became the ``tarzan/backtest/`` package, and the audit stayed green.

The manifest and the validator are gone, so the expiring-gate failure is now
structurally impossible rather than guarded against. What was genuinely load
bearing lives here instead, in the suite — which runs on every push AND in the
delivery gate, so these properties are enforced in both places rather than only
where a declaration audit happened to be wired.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PUBLICATION_STEP = "Render & send newsletter"
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_EXACT_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)(?:\s|$)")
_HASH = re.compile(r"--hash=sha256:[0-9a-f]{64}(?:\s|$)")


def _text(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_lines(relative: str) -> list[str]:
    return [
        line.strip()
        for line in _text(relative).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


class TestOnlyDeliveryReasonsCanStopADelivery:
    def test_the_gate_runs_behaviour_not_declarations(self):
        workflow = _text(".github/workflows/newsletter.yml")

        assert "python -m pytest tarzan/tests -q" in workflow, (
            "publication is no longer gated on the test suite"
        )
        assert re.search(r"(?m)^    needs:\s*validate\s*$", workflow), (
            "publication no longer depends on validation"
        )
        assert "validate_release" not in workflow, (
            "a declaration audit is back in the delivery path: a stale manifest, "
            "pin, or README claim can once again withhold a digest"
        )

    def test_no_declaration_manifest_exists_to_expire(self):
        """The strongest form of the 2026-08-18 fix: there is no manifest, so no
        date, pin window or documentation claim in one can gate anything."""
        assert not (_ROOT / "tarzan/release_manifest.json").exists()
        assert not (_ROOT / "scripts/validate_release.py").exists()

    def test_the_delivery_gate_reads_no_clock(self):
        """Every step the gate runs, in order, and none of them consults a date.

        The gate is now three behavioural commands. Pinning the list is what keeps
        a future "check the pins are fresh" step from being added back to the one
        job whose failure costs a digest.
        """
        workflow = _text(".github/workflows/newsletter.yml")
        validate = workflow.split("  validate:")[1].split("\n  send:")[0]
        commands = re.findall(r"(?m)^        run:\s*(.+?)\s*$", validate)

        assert commands == [
            "python -m pip install --require-hashes -r requirements.txt",
            "python -m compileall -q tarzan scripts",
            "python -m pip check",
            "python -m pytest tarzan/tests -q",
        ], commands


class TestCredentialsReachOnlyThePublicationStep:
    """The one invariant in the old validator that nothing else enforced.

    Checkout, Python setup, dependency installation and the cache restore all run
    before the send, and any of them could be handed SMTP, Gemini or Drive
    credentials by a one-line edit. Keeping the secrets in the final step means a
    third-party action compromised upstream sees none of them.
    """

    def test_secrets_appear_in_no_other_step(self):
        workflow = _text(".github/workflows/newsletter.yml")

        step = ""
        offenders = []
        for line in workflow.splitlines():
            named = re.match(r"^      - name:\s*(.+?)\s*$", line)
            if named:
                step = named.group(1)
            if "secrets." in line and step != _PUBLICATION_STEP:
                offenders.append((step, line.strip()))

        assert offenders == [], offenders

    def test_the_publication_step_is_the_one_that_sends(self):
        """Guards the test above from being satisfied by renaming a step."""
        workflow = _text(".github/workflows/newsletter.yml")
        block = workflow.split(f"- name: {_PUBLICATION_STEP}")[1]
        assert "scripts/send_newsletter.py" in block

    def test_credentials_are_not_job_scoped(self):
        """A job-level ``env:`` would hand the secrets to every step in the job,
        which is the same exposure with tidier indentation."""
        workflow = _text(".github/workflows/newsletter.yml")
        assert not re.search(r"(?ms)^    env:\s*\n(?:      .+\n)*?      .+secrets\.",
                             workflow)

    def test_the_validate_job_sees_no_secret_at_all(self):
        workflow = _text(".github/workflows/newsletter.yml")
        validate = workflow.split("  validate:")[1].split("\n  send:")[0]
        assert "secrets." not in validate


class TestTheSupplyChainIsPinned:
    def test_every_action_is_pinned_to_a_full_commit_sha(self):
        """A tag or a branch is mutable, so a compromised upstream release would
        be pulled into the job that holds the credentials.

        Discovered from the workflows on disk rather than a hardcoded pair. A
        fixed list would make deleting a CI file fail the suite — and the suite is
        the delivery gate, so that is the coupling the 2026-08-18 split existed to
        remove. It also means a third workflow is covered the day it is added
        instead of the day someone remembers this test.
        """
        workflows = sorted((_ROOT / ".github/workflows").glob("*.yml"))
        assert workflows, "no workflows found"
        for path in workflows:
            refs = re.findall(r"(?m)^\s*uses:\s*[^\s#]+@([^\s#]+)",
                              path.read_text(encoding="utf-8"))
            unpinned = [r for r in refs if not _FULL_SHA.fullmatch(r)]
            assert unpinned == [], (path.name, unpinned)

    def test_the_delivery_workflow_pins_actions_at_all(self):
        """The loop above is vacuously true for a workflow that uses no actions;
        the one that sends must genuinely pin some."""
        refs = re.findall(r"(?m)^\s*uses:\s*[^\s#]+@([^\s#]+)",
                          _text(".github/workflows/newsletter.yml"))
        assert len(refs) >= 3, refs

    def test_installation_enforces_hashes(self):
        assert "pip install --require-hashes -r requirements.txt" in \
            _text(".github/workflows/newsletter.yml")

    def test_direct_dependencies_are_exactly_pinned(self):
        loose = [line for line in _requirement_lines("requirements.in")
                 if not (_EXACT_REQUIREMENT.match(line)
                         and _EXACT_REQUIREMENT.match(line).group(0).strip() == line)]
        assert loose == [], loose

    def test_every_lock_entry_carries_a_unique_hash(self):
        for line in _requirement_lines("requirements.txt"):
            match = _EXACT_REQUIREMENT.match(line)
            assert match and _HASH.search(line), line[:120]
            hashes = re.findall(r"--hash=sha256:([0-9a-f]{64})(?:\s|$)", line)
            assert hashes and len(hashes) == len(set(hashes)), match.group(1)

    def test_every_direct_dependency_is_in_the_lock(self):
        """``--require-hashes`` proves the lock is internally consistent; it says
        nothing about a direct dependency that never reached the lock at all, and
        that one surfaces as an ImportError at render time."""
        direct = {_canonical(_EXACT_REQUIREMENT.match(l).group(1))
                  for l in _requirement_lines("requirements.in")}
        locked = {_canonical(_EXACT_REQUIREMENT.match(l).group(1))
                  for l in _requirement_lines("requirements.txt")}
        assert direct - locked == set()


class TestOneVersionAuthority:
    """Kept from the old validator, minus the manifest.

    The useful half was never "the manifest agrees with the code" — the manifest
    was the copy. It was that the package and the CLI both read ONE constant.
    """

    def test_the_package_version_comes_from_tarzan_version(self):
        assert "APPLICATION_VERSION as __version__" in _text("tarzan/__init__.py")
        assert re.search(r"(?m)^APPLICATION_VERSION\s*=", _text("tarzan/version.py"))

    def test_the_cli_uses_the_package_authority(self):
        assert "from tarzan import __version__" in _text("tarzan/main.py")

    def test_the_gemini_model_and_api_version_are_pinned(self):
        """A floating model or API version changes the digest's prose without a
        commit. The pin is the constant itself; nothing needs to restate it."""
        source = _text("tarzan/export/ai_summary.py")
        assert re.search(r'(?m)^_PINNED_MODEL\s*=\s*["\'][^"\']+["\']', source)
        assert re.search(r"/v\d[a-z]*/models/", source)
