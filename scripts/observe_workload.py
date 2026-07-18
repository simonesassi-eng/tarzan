#!/usr/bin/env python3
"""Print deterministic, network-free Tarzan workload evidence as strict JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Keep the stable bare-script invocation usable from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tarzan.runtime.workload import run_network_free_workload_harness  # noqa: E402


def main() -> int:
    print(
        json.dumps(
            run_network_free_workload_harness(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
