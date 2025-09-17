from __future__ import annotations

import json
from pathlib import Path

import types


def test_ablation_runner_passes_rank_and_scheme(monkeypatch, tmp_path: Path):
    # Capture subprocess.run invocations
    called = []

    def fake_run(cmd, check):
        called.append(list(cmd))
        # simulate successful run
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    from scripts.ablation_runner import main as abl_main

    abl_main(
        [
            "--domains",
            "legal",
            "--ranks",
            "2,4",
            "--schemes",
            "attention,all",
            "--samples",
            "4",
            "--epochs",
            "1",
            "--out",
            str(tmp_path / "abl_out.jsonl"),
        ]
    )

    assert called, "subprocess.run was not invoked"
    # Verify that each invocation contains --rank and --scheme
    for cmd in called:
        # flatten to a single list
        assert "--rank" in cmd, f"--rank missing in command: {cmd}"
        assert "--scheme" in cmd, f"--scheme missing in command: {cmd}"


