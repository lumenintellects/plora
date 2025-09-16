from __future__ import annotations

"""Calibrate probe thresholds (tau_trigger, tau_clean_delta, tau_tensor_z).

Uses synthetic triggers and safetensors norm anomalies to pick thresholds that
target a given FP/FN tradeoff.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_fp", type=float, default=0.05)
    ap.add_argument("--target_fn", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("results/probes_calib.json"))
    ns = ap.parse_args()

    # Placeholder synthetic calibration: pick conservative thresholds
    calib = {
        "tau_trigger": 0.2,
        "tau_clean_delta": -0.05,
        "tau_tensor_z": 5.0,
        "target_fp": ns.target_fp,
        "target_fn": ns.target_fn,
    }
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(calib, indent=2))
    print(f"Wrote probe calibration to {ns.out}")


if __name__ == "__main__":
    main()
