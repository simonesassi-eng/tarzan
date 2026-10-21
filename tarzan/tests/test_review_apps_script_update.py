"""The production Apps Script Update source executes in Node, without skips."""

import os
from pathlib import Path
import shutil
import subprocess


def test_production_authenticated_update_listener_in_node():
    node = shutil.which(os.environ.get("TARZAN_TEST_NODE", "node"))
    assert node is not None, "Node is required to verify the production Apps Script Update code"
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            node, str(Path(__file__).with_name("apps_script_update_test.js")),
            str(root / "scripts/apps_script/Code.gs"),
        ],
        check=True, capture_output=True, text=True, timeout=30,
    )
    assert "native owner/SENT" in completed.stdout
