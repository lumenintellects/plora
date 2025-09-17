from __future__ import annotations

from pathlib import Path


def test_sim_v2_entry_nonlocal_prevI(tmp_path: Path):
    # Run a minimal sim to ensure no UnboundLocalError occurs in _on_round
    import sys
    from unittest.mock import patch
    
    # Mock sys.argv to provide command line arguments
    test_args = [
        "sim_v2_entry",
        "--agents", "4",
        "--rounds", "2", 
        "--graph", "er",
        "--graph_p", "0.2",
        "--seed", "3",
        "--security", "off",
        "--report_dir", str(tmp_path),
    ]
    
    with patch.object(sys, 'argv', test_args):
        from swarm.sim_v2_entry import main as sim_main
        sim_main()

    out = list(tmp_path.glob("swarm_v2_report_*.json"))
    assert out, "Expected a v2 report to be written"


def test_sweep_nonlocal_prevI(tmp_path: Path):
    from scripts.sweep import main as sweep_main

    sweep_main([
        "--topos", "er",
        "--ns", "6", 
        "--seeds", "5",
        "--p", "0.2",
        "--rounds", "2",
        "--trojan_rates", "0.0",
        "--out", str(tmp_path / "thesis_sweep.jsonl"),
    ])
    assert (tmp_path / "thesis_sweep.jsonl").exists()


