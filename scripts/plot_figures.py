from __future__ import annotations

"""Generate simple figures from v2 summaries (optional matplotlib dependency)."""

import argparse
import json
from pathlib import Path


def plot(summary_path: Path, out_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping plots")
        return

    data = json.loads(Path(summary_path).read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot acceptance vs predicted_t_all
    xs = [rec.get("predicted_t_all") or 0 for rec in data]
    ys = [rec.get("accepted_offers") or 0 for rec in data]
    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("predicted_t_all")
    plt.ylabel("accepted_offers")
    plt.title("Acceptance vs Predicted Diffusion Time")
    plt.savefig(out_dir / "acceptance_vs_tpred.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    ns = ap.parse_args()
    plot(ns.summary, ns.out)


if __name__ == "__main__":
    main()
