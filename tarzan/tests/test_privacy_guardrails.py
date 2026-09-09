"""The guardrails that must not depend on a developer's machine.

This repo is PUBLIC and has leaked personal data three separate ways: figures
from live runs quoted in comments and commit messages, the portfolio total
printed into public Actions logs, and the author's own email on every commit.
Each got a fix, and two of those fixes live in ``.githooks`` — which is
``core.hooksPath``, a LOCAL setting. A fresh clone has no hooks, and
``--no-verify`` skips them. So the hooks are the convenience and this file is
the guarantee: it runs in ``checks.yml`` on every push and pull request, where
nobody's local configuration can turn it off.

Deliberately narrow. Each check below is a shape that is essentially never a
legitimate illustration, so it needs no allowlist to maintain and no retro-
marking of the synthetic stand-ins already in the tree. Marking a line
"synthetic" is the escape hatch, same convention as .githooks/pre-commit.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Addresses that must never author a commit here again. A denylist, not a
#: "must be noreply" rule, so an outside contributor is not rejected.
PERSONAL_EMAILS = ("simonesassi4@gmail.com", "simonsa@amazon.it")

#: Fixtures that are synthetic by construction — same exclusions as the hook.
EXCLUDED = ("tarzan/tests/golden/", "tarzan/stress/", ".githooks/")


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"],
                         capture_output=True, text=True).stdout.split()
    keep = []
    for rel in out:
        if rel.startswith(EXCLUDED) or rel.endswith((".gz", ".png", ".xlsx", ".zip")):
            continue
        p = REPO / rel
        try:
            if p.is_file() and b"\0" not in p.read_bytes()[:8000]:
                keep.append(p)
        except OSError:
            pass
    return keep


class TestNoMeasuredAmountInTheTree:
    """A six-figure amount carrying CENTS is a measured value, not an example.

    Nobody illustrates a point with a figure to the cent; that shape only comes
    from reading it off a real run, which is exactly how the portfolio total
    ended up in a docstring four times. (An example here would itself have to be
    a real-looking amount, which is the joke: writing this docstring tripped the
    pre-commit hook.)
    """

    PATTERN = re.compile(r"(?:€|EUR ?)?\b\d{3}[,.]\d{3}\.\d{2}\b|\b\d{6}\.\d{2}\b")

    def test_no_six_figure_amount_with_cents(self):
        offenders = []
        for path in _tracked_text_files():
            for n, line in enumerate(path.read_text(encoding="utf-8",
                                                    errors="replace").splitlines(), 1):
                if "synthetic" in line.lower():
                    continue
                for hit in self.PATTERN.findall(line):
                    offenders.append(f"{path.relative_to(REPO)}:{n}: {hit}")
        assert not offenders, (
            "a measured six-figure amount is in the tree — this repo is public:\n  "
            + "\n  ".join(offenders)
            + "\n\nRound it away, use a synthetic stand-in, or put "
              '"synthetic" on the line.')


class TestTheLogRedactionStaysWired:
    """Every entrypoint a workflow runs must install the filter.

    The send path had it and the daily oracle did not, so the oracle kept
    publishing the whole instrument list to a public log for as long as it ran.
    """

    @staticmethod
    def _workflow_entrypoints() -> set[str]:
        found = set()
        for wf in (REPO / ".github" / "workflows").glob("*.yml"):
            for m in re.finditer(r"run:\s*python3?\s+(scripts/\S+\.py)",
                                 wf.read_text(encoding="utf-8")):
                found.add(m.group(1))
        return found

    def test_every_script_a_workflow_runs_installs_it(self):
        missing = [s for s in self._workflow_entrypoints()
                   if "log_redaction" not in (REPO / s).read_text(encoding="utf-8")]
        assert not missing, (
            f"these run in CI, where the log is public, without the redaction "
            f"filter: {missing}. Add `log_redaction.install()` after logging is "
            f"configured.")

    def test_there_is_at_least_one_entrypoint_to_check(self):
        """Guards the check above from passing because the regex went stale."""
        assert self._workflow_entrypoints(), \
            "found no workflow entrypoints — the regex above no longer matches"


class TestNoWorkflowPublishesAnArtifact:
    """On a public repo a run's artifacts are downloadable by anyone, and a
    rendered issue carries the positions AND their values. Nothing uploads one
    today; this is what keeps that true."""

    def test_no_upload_artifact_step(self):
        offenders = [wf.name for wf in (REPO / ".github" / "workflows").glob("*.yml")
                     if "upload-artifact" in wf.read_text(encoding="utf-8")]
        assert not offenders, (
            f"{offenders} upload an artifact. On a public repo anyone can "
            f"download it. Print what you need into the (redacted) log instead.")


class TestNoPersonalIdentityOnCommits:
    """The history was rewritten to remove the author's personal address, and it
    came straight back on the next two commits because the repo-local
    user.email had not been changed. Depth is whatever CI checked out; on a
    shallow clone this sees the tip, which is the commit that would reintroduce
    it."""

    def test_head_is_not_authored_by_a_personal_address(self):
        out = subprocess.run(
            ["git", "-C", str(REPO), "log", "--format=%H %ae%n%H %ce", "-50"],
            capture_output=True, text=True).stdout
        offenders = [l for l in out.splitlines()
                     if any(e in l for e in PERSONAL_EMAILS)]
        assert not offenders, (
            "commits authored with a personal address:\n  "
            + "\n  ".join(offenders)
            + "\n\nFix the identity, then re-author:\n"
              "  git config user.email 280960750+simonesassi-eng@users.noreply.github.com")
