from __future__ import annotations

"""Summarise Swarm Sim reports (v2) into a concise JSON/Markdown.

Usage:
  python -m scripts.evaluate_v2 --reports results --out results/summary_v2.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from plora.stats import bootstrap_ci_mean
from swarm.theory import predicted_rounds_spectral, cheeger_bounds, epidemic_threshold
from plora.te import transfer_entropy_discrete


def _collect_reports(dir_: Path) -> List[dict]:
    res = []
    # v1 graph engine reports
    for p in dir_.glob("swarm_graph_report_*.json"):
        try:
            res.append(json.loads(p.read_text()))
        except Exception:
            pass
    # v2 in-process reports
    for p in dir_.glob("swarm_v2_report_*.json"):
        try:
            res.append(json.loads(p.read_text()))
        except Exception:
            pass
    return res


def _summarise(rep: dict) -> dict:
    meta = rep.get("meta", {})
    final = rep.get("final", {})
    gate = final.get("gate", {})
    rounds = rep.get("rounds", [])
    # Unify coverage extraction
    coverage = final.get("coverage")
    if coverage is None and rounds:
        coverage = rounds[-1].get("coverage")
    # Unified fields with sensible defaults
    # derive acceptance rate if rounds are present (v1 had fixed offers per round)
    accepted_total = final.get("accepted_offers", 0)
    offers_total = len(rounds) * meta.get("N", 0) if rounds else 0
    acceptance_rate = (accepted_total / offers_total) if offers_total else None

    # MI summary if available in rounds
    mi_vals = [
        r.get("mutual_information", None)
        for r in rounds
        if r.get("mutual_information", None) is not None
    ]
    mi_deltas = [r.get("mi_delta", 0.0) for r in rounds]
    mi_ci = None
    if mi_deltas:
        lo, hi = bootstrap_ci_mean(
            [float(x) for x in mi_deltas], n_resamples=500, ci=0.95, seed=42
        )
        mi_ci = [lo, hi]
    mi_final = mi_vals[-1] if mi_vals else None
    mi_max = max(mi_vals) if mi_vals else None
    mi_min = min(mi_vals) if mi_vals else None

    # Simple TE estimate between first two domains using coverage series
    te_ab = None
    doms = meta.get("domains") or []
    if len(doms) >= 2 and rounds:
        d0, d1 = doms[0], doms[1]
        s0 = [r.get("coverage", {}).get(d0, 0.0) for r in rounds]
        s1 = [r.get("coverage", {}).get(d1, 0.0) for r in rounds]
        try:
            te_ab = transfer_entropy_discrete(s0, s1, k=1, bins=8)
        except Exception:
            te_ab = None

    # Theory predictions and deltas with simple CIs where possible
    lam2 = meta.get("lambda2")
    n_agents = meta.get("N")
    theory = {}
    try:
        if lam2 is not None and n_agents is not None:
            t_pred = predicted_rounds_spectral(int(n_agents), float(lam2))
            t_obs = final.get("observed_t_all")
            if t_obs is not None:
                theory["t_pred"] = t_pred
                theory["t_obs"] = t_obs
                theory["t_delta"] = float(t_obs - t_pred)
        # Placeholders for Cheeger/threshold diagnostics: allow presence if extended inputs exist
        if "cheeger" in meta:
            theory["cheeger_bounds"] = meta.get("cheeger")
        if "epidemic_threshold" in meta:
            theory["epidemic_threshold"] = meta.get("epidemic_threshold")
    except Exception:
        pass

    return {
        "N": meta.get("N"),
        "topology": meta.get("topology"),
        "lambda2": meta.get("lambda2"),
        "observed_t_all": final.get("observed_t_all"),
        "predicted_t_all": final.get("predicted_t_all"),
        "acceptance_rate": acceptance_rate,
        "t_all": final.get("t_all"),
        "bytes_on_wire": final.get("bytes_on_wire", 0),
        "accepted_offers": final.get("accepted_offers", 0),
        "coverage": coverage or {},
        "mi": {
            "final": mi_final,
            "max": mi_max,
            "min": mi_min,
            "delta_sum": sum(mi_deltas) if mi_deltas else None,
            "delta_ci": mi_ci,
        },
        "te_pair": te_ab,
        "theory": theory,
        "gate": {
            "rejected_hash_total": gate.get("rejected_hash_total", 0),
            "rejected_safety_total": gate.get("rejected_safety_total", 0),
            "accepted_clean_total": gate.get("accepted_clean_total", 0),
            "accepted_trojan_total": gate.get("accepted_trojan_total", 0),
            "rejected_clean_total": gate.get("rejected_clean_total", 0),
            "rejected_trojan_total": gate.get("rejected_trojan_total", 0),
            "false_negatives": gate.get("false_negatives", 0),
            "false_positives": gate.get("false_positives", 0),
            "rejection_reasons": gate.get("rejection_reasons", {}),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/summary_v2.json"))
    ns = ap.parse_args()

    reps = _collect_reports(ns.reports)
    summary = [_summarise(r) for r in reps]
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {len(summary)} summaries to {ns.out}")


if __name__ == "__main__":
    main()
