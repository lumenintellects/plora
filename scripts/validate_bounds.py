from __future__ import annotations

"""Validate diffusion bounds: spectral-gap heuristic, Cheeger bounds, and plots."""

import argparse
import json
from pathlib import Path

from swarm.graph_v2 import erdos_renyi_graph
from swarm.metrics import spectral_gap
from swarm.theory import predicted_rounds_spectral, cheeger_bounds, conductance_estimate
from plora.config import get as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ns", type=lambda s: [int(x) for x in s.split(",")], default=[20, 40, 80, 160]
    )
    ap.add_argument("--p", type=float, default=cfg("graph.p", 0.25))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/bounds_validation.json"))
    ap.add_argument("--plot", action="store_true")
    ns = ap.parse_args()

    recs = []
    for n in ns.ns:
        g = erdos_renyi_graph(n, ns.p, seed=ns.seed + n)
        lam2 = spectral_gap(g)
        t_pred = predicted_rounds_spectral(n, lam2)
        phi = conductance_estimate(g, trials=64)
        lo, hi = cheeger_bounds(g)
        recs.append(
            {
                "n": n,
                "lambda2": lam2,
                "t_pred": t_pred,
                "phi": phi,
                "cheeger": {"lower": lo, "upper": hi},
            }
        )
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(recs, indent=2))
    print(f"Saved {len(recs)} records to {ns.out}")

    if ns.plot:
        try:
            import matplotlib.pyplot as plt

            xs = [r["n"] for r in recs]
            lam2s = [r["lambda2"] for r in recs]
            ts = [r["t_pred"] for r in recs]
            plt.figure(figsize=(6, 4))
            plt.plot(xs, lam2s, marker="o", label="lambda2")
            plt.xlabel("n")
            plt.ylabel("lambda2")
            plt.twinx()
            plt.plot(xs, ts, marker="s", color="orange", label="t_pred")
            plt.ylabel("t_pred")
            outp = ns.out.with_suffix(".png")
            plt.title("Spectral gap and predicted t vs n")
            plt.savefig(outp, dpi=120, bbox_inches="tight")
            print(f"Saved plot to {outp}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
