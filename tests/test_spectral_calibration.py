from __future__ import annotations

from pathlib import Path
import json

from scripts.calibrate_c import main as calib_main


def test_calibrate_c_schema(tmp_path: Path):
    out = tmp_path / "c_calib_er.json"
    calib_main(
        [
            "--topology",
            "er",
            "--ns",
            "10,12",
            "--p",
            "0.3",
            "--rounds",
            "5",
            "--seed",
            "1",
            "--out",
            str(out),
        ]
    )
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, list) and len(data) >= 1
    rec = data[0]
    # Required fields
    for key in ("topology", "n", "lambda2", "t_all", "C_hat", "params"):
        assert key in rec
    assert rec["topology"] == "er"
