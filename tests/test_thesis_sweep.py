from __future__ import annotations

from pathlib import Path
import json

from scripts.sweep import main as sweep_main


def test_thesis_sweep_schema(tmp_path: Path):
    out = tmp_path / "thesis_sweep.jsonl"
    sweep_main(
        [
            "--topos",
            "er",
            "--ns",
            "10",
            "--seeds",
            "7",
            "--p",
            "0.4",
            "--rounds",
            "3",
            "--trojan_rates",
            "0.0",
            "--out",
            str(out),
        ]
    )
    assert out.exists()
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert (
        lines
        and "lambda2" in lines[0]
        and "t_pred" in lines[0]
        and "mi_series" in lines[0]
    )
