"""Regression: assert no v9-deleted tool names linger in production code."""

import subprocess
from pathlib import Path


def test_no_dead_tool_references():
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "verify_v9_no_dead_refs.sh"
    )
    assert script.exists(), f"verification script missing: {script}"
    res = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, (
        f"verify_v9_no_dead_refs.sh failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
