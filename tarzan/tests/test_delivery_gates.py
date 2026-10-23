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
bearing lives here instead, in the suite, which runs on every push (checks.yml).

On 2026-10-09 the suite itself left the delivery path too: re-running it before each
issue cost ~2 min per send, and a red test on main withheld every digest although the
code had already been tested when it was pushed. The send job is gated only on the
code compiling and its pinned dependencies agreeing.
"""

from __future__ import annotations

import re
from pathlib import Path
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_PUBLICATION_STEP = "Render & send newsletter"
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_EXACT_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)(?:\s|$)")
_HASH = re.compile(r"--hash=sha256:[0-9a-f]{64}(?:\s|$)")


def _text(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


def _workflow():
    # BaseLoader preserves "on" as a key (YAML 1.1 otherwise treats it as True)
    # and cannot construct executable Python objects.
    return yaml.load(_text(".github/workflows/newsletter.yml"), Loader=yaml.BaseLoader)


def _steps():
    steps = _workflow()["jobs"]["send"]["steps"]
    assert len({step["name"] for step in steps}) == len(steps)
    return {step["name"]: step for step in steps}


def _condition(value):
    return value.removeprefix("${{").removesuffix("}}").strip()


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_lines(relative: str) -> list[str]:
    return [
        line.strip()
        for line in _text(relative).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


class TestOnlyDeliveryReasonsCanStopADelivery:
    def test_the_suite_runs_on_every_push_not_before_every_issue(self):
        workflow = _text(".github/workflows/newsletter.yml")
        checks = _text(".github/workflows/checks.yml")

        assert "python -m pytest tarzan/tests -q" in checks, "the suite no longer runs on push"
        assert "pytest" not in workflow, (
            "the test suite is back in the delivery path: one red test withholds every digest"
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

    def test_only_operational_event_freshness_and_dependency_consistency_gate_delivery(self):
        """No declaration/dependency-expiry calendar can withhold an issue.

        Native event age is intentionally a delivery condition: an obsolete queued
        request must not authorize SMTP. It is not an expiring dependency manifest.
        Cache import/export are explicitly best-effort. Queried market data
        leave the runner only through the encrypted allowlist.
        """
        steps = _steps()
        expected = {
            "Resolve native event context": "python scripts/resolve_newsletter_event.py",
            "Compile positive Tarzan scopes": "python -m compileall -q tarzan scripts",
            "Check installed dependency consistency": "python -m pip check",
            _PUBLICATION_STEP: "python scripts/send_newsletter.py",
            "Verify newsletter without delivery": "python scripts/verify_newsletter.py",
        }
        assert all(steps[name]["run"] == command for name, command in expected.items())
        assert list(steps).index("Check installed dependency consistency") < list(steps).index(_PUBLICATION_STEP)
        for step in steps.values():
            if "cache" in step.get("run", "") or "/cache/" in step.get("uses", ""):
                assert step.get("continue-on-error") == "true", step["name"]
        assert _condition(steps[_PUBLICATION_STEP]["if"]) == "!inputs.verify_only"


class TestCredentialsReachOnlyThePublicationStep:
    """The one invariant in the old validator that nothing else enforced.

    Checkout, Python setup, dependency installation and the cache restore all run
    before the send, and any of them could be handed SMTP, Gemini or Drive
    credentials by a one-line edit. Keeping the secrets in the final step means a
    third-party action compromised upstream sees none of them.
    """

    def test_native_credentials_are_scoped_to_their_actual_consumers(self):
        verify = "Verify newsletter without delivery"
        allowed = {
            "SMTP_USER": {_PUBLICATION_STEP},
            "SMTP_PASS": {_PUBLICATION_STEP},
            "RECIPIENT_EMAIL": {_PUBLICATION_STEP},
            "GEMINI_API_KEY": {_PUBLICATION_STEP},
            "DELIVERY_CLAIM_ENDPOINT": {_PUBLICATION_STEP},
            "DELIVERY_CLAIM_TOKEN": {_PUBLICATION_STEP},
            "DRIVE_FOLDER_ID": {_PUBLICATION_STEP, verify},
            "GOOGLE_DRIVE_CREDENTIALS_JSON": {_PUBLICATION_STEP, verify},
            "TARZAN_CACHE_KEY_B64": {
                "Verify protected cache key", "Validate and import encrypted market snapshot",
                "Encrypt updated market snapshot", _PUBLICATION_STEP, verify,
            },
        }
        offenders = []
        for name, step in _steps().items():
            secret_names = re.findall(r"secrets\.([A-Z0-9_]+)", str(step))
            for secret in secret_names:
                if name not in allowed.get(secret, set()) or "uses" in step:
                    offenders.append((name, secret))
        assert offenders == [], offenders

    def test_the_publication_step_is_the_one_that_sends(self):
        """Guards the test above from being satisfied by renaming a step."""
        workflow = _text(".github/workflows/newsletter.yml")
        block = workflow.split(f"- name: {_PUBLICATION_STEP}")[1]
        assert "scripts/send_newsletter.py" in block

    def test_credentials_are_not_job_scoped(self):
        """A job-level ``env:`` would hand the secrets to every step in the job,
        which is the same exposure with tidier indentation."""
        workflow = _workflow()
        assert "secrets." not in str(workflow.get("env", {}))
        assert all("secrets." not in str(job.get("env", {})) for job in workflow["jobs"].values())

    def test_no_email_mode_is_explicit_exclusive_and_has_no_delivery_credential(self):
        workflow = _workflow()
        option = workflow["on"]["workflow_dispatch"]["inputs"]["verify_only"]
        assert option["type"] == "boolean" and option["default"] == "false"
        step = _steps()["Verify newsletter without delivery"]
        assert _condition(step["if"]) == "inputs.verify_only"
        assert not any(name.startswith(("SMTP_", "RECIPIENT_", "GEMINI_", "DELIVERY_CLAIM_"))
                       for name in step["env"])
        assert "tarzan-newsletter-verify" in workflow["concurrency"]["group"]
        assert "inputs.verify_only" in workflow["concurrency"]["group"]


class TestPrivateStateLeavesOnlyAsCiphertext:
    def test_queried_cache_and_diagnostics_use_explicit_ciphertext_files_not_directories(self):
        from tarzan.data.protected_cache import FILENAME as MARKET_FILE, SCHEMA_VERSION
        from tarzan.data.price_cache import _HISTORY_REPAIR_VERSION
        from tarzan.runtime.diagnostics import FILENAME as DIAGNOSTIC_FILE

        steps = _steps()
        restore = steps["Restore encrypted market snapshot"]
        save = steps["Save encrypted market snapshot"]
        for step in (restore, save):
            assert step["with"]["path"] == "${{ runner.temp }}/tarzan-sealed/" + MARKET_FILE
            key = step["with"]["key"]
            assert key.startswith(f"tarzan-sealed-market-v{SCHEMA_VERSION}-r{_HISTORY_REPAIR_VERSION}-")
            assert all(fragment in key for fragment in (
                "steps.cache_key.outputs.key_id", "github.run_id", "github.run_attempt",
            ))
        assert "steps.protected_export.outputs.nonempty == 'true'" in save["if"]
        artifact = steps["Retain encrypted diagnostic evidence"]
        assert artifact["with"]["path"] == "${{ runner.temp }}/tarzan-diagnostics/" + DIAGNOSTIC_FILE
        assert artifact["if"] == "always() && !cancelled()"
        assert artifact["with"]["retention-days"] == "14"
        assert "secrets." not in str(artifact)

    def test_the_public_projection_only_fills_missing_macro_histories(self):
        step = _steps()["Validate and import public macro snapshot"]
        assert "--only-missing" in step["run"]


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
        """``--require-hashes`` is the invariant, not the whole command line. The
        install carries retry flags too (a lost download used to cost a whole issue),
        and pinning the exact string made adding them look like weakening the gate —
        which is the opposite of what they do: every wheel is still hash-pinned."""
        workflow = _text(".github/workflows/newsletter.yml")
        installer = _text("scripts/install_runtime_dependencies.py")
        assert "python scripts/install_runtime_dependencies.py" in workflow
        assert '"--require-hashes"' in installer
        assert '"requirements.txt"' in installer

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
