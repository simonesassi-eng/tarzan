"""The runtime subset preserves every pin/hash and needs no test tooling."""

from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace

from packaging.requirements import Requirement
import pytest
import yaml

from scripts import install_runtime_dependencies as installer


ROOT = Path(__file__).resolve().parents[2]
HASH = " --hash=sha256:" + "a" * 64


def names(text):
    return {
        installer.canonical_name(line.split("==", 1)[0])
        for line in text.splitlines() if line and not line.startswith("#")
    }


def test_runtime_lock_is_exactly_the_existing_lock_minus_test_only_packages():
    original = (ROOT / "requirements.txt").read_text()
    runtime = installer.runtime_requirements(original)
    expected = [
        line for line in original.splitlines() if line and not line.startswith("#")
        and installer.canonical_name(line.split("==", 1)[0]) not in installer.TEST_ONLY
    ]
    assert runtime.splitlines() == expected
    assert names(original) - names(runtime) == installer.TEST_ONLY
    assert len(names(runtime)) == len(names(original)) - len(installer.TEST_ONLY)


def test_runtime_projection_is_a_closed_dependency_set_for_the_pinned_distributions():
    """A dependency upgrade cannot silently turn a test-only tool into runtime."""
    runtime = names(installer.runtime_requirements((ROOT / "requirements.txt").read_text()))
    for name in runtime:
        for row in distribution(name).requires or []:
            requirement = Requirement(row)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            assert installer.canonical_name(requirement.name) in runtime, (name, row)


def test_all_non_test_direct_dependencies_are_preserved():
    direct = names((ROOT / "requirements.in").read_text())
    runtime = names(installer.runtime_requirements((ROOT / "requirements.txt").read_text()))
    assert direct - {"hypothesis", "pytest"} <= runtime


def test_names_are_canonicalized_without_losing_the_original_hash_line():
    source = "Runtime_Lib==1.2" + HASH + "\nPy_Test==7.4" + HASH + "\n"
    # Py_Test is a different distribution, not the test tool pytest.
    assert installer.runtime_requirements(source) == source
    assert installer.canonical_name("Runtime_Lib") == "runtime-lib"


@pytest.mark.parametrize("source", [
    "unpinned>=1" + HASH,
    "runtime==1.0",
    "runtime==1.0 --hash=sha256:bad",
    "-r other.txt",
    "--index-url https://example.invalid",
    "runtime==1.0 --hash=md5:" + "a" * 32,
    "runtime==1.0" + HASH + " --no-deps",
    "runtime==1.0" + HASH + " \\\n",
    "# empty\n",
    "pytest==7.4" + HASH + "\n",
])
def test_malformed_or_empty_runtime_projection_is_rejected(source):
    with pytest.raises(ValueError):
        installer.runtime_requirements(source)


def test_duplicate_canonical_names_are_rejected():
    with pytest.raises(ValueError):
        installer.runtime_requirements("same_name==1" + HASH + "\nsame-name==1" + HASH)


def test_installer_invokes_resolver_and_hash_checks_with_private_transient_input(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text(
        "runtime==1.0" + HASH + "\npytest==7.4" + HASH + "\n",
    )
    monkeypatch.setattr(installer, "_ROOT", tmp_path)
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        path = Path(command[-1])
        assert path.read_text() == "runtime==1.0" + HASH + "\n"
        assert path.stat().st_mode & 0o077 == 0
        assert path.parent.stat().st_mode & 0o077 == 0
        assert kwargs == {"check": False}
        assert "--no-deps" not in command
        return SimpleNamespace(returncode=19)

    monkeypatch.setattr(installer.subprocess, "run", run)
    assert installer.main() == 19
    assert seen[0][:-1] == [
        installer.sys.executable, "-m", "pip", "install", "--require-hashes",
        "--retries", "5", "--timeout", "30", "-r",
    ]
    assert not Path(seen[0][-1]).exists()


def test_invalid_lock_cannot_reach_installation(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("loose>=1\n")
    monkeypatch.setattr(installer, "_ROOT", tmp_path)
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **kw: pytest.fail("unchecked install"))
    assert installer.main() == 1


def test_only_publication_uses_the_subset_and_full_checks_keep_the_original_lock():
    def steps(path):
        workflow = yaml.load((ROOT / path).read_text(), Loader=yaml.BaseLoader)
        return next(iter(workflow["jobs"].values()))["steps"]

    send = next(step for step in steps(".github/workflows/newsletter.yml")
                if step["name"] == "Install immutable dependencies")
    checks = next(step for step in steps(".github/workflows/checks.yml")
                  if step["name"] == "Install immutable dependencies")
    assert send["run"] == "python scripts/install_runtime_dependencies.py"
    assert "env" not in send
    assert "--require-hashes" in checks["run"]
    assert "-r requirements.txt" in checks["run"]
    assert "Install immutable dependencies" in send["name"]
