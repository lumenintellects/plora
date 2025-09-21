from __future__ import annotations
"""Summarise Swarm Sim reports (v2) into a concise JSON/Markdown.

Usage:
  python -m scripts.evaluate_v2 --reports results --out results/summary_v2.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import List

from plora.stats import bootstrap_ci_mean
from swarm.theory import predicted_rounds_spectral
from plora.te import transfer_entropy_discrete


def _collect_reports(dir_: Path) -> List[dict]:
    reps: list[dict] = []
    for p in dir_.glob("swarm_graph_report_*.json"):  # legacy v1 (if present)
        try:
            reps.append(json.loads(p.read_text()))
        except Exception:
            pass
    for p in dir_.glob("swarm_v2_report_*.json"):  # v2 reports
        try:
            reps.append(json.loads(p.read_text()))
        except Exception:
            pass
    return reps


def _summarise(rep: dict) -> dict:
    meta = rep.get("meta", {})
    final = rep.get("final", {})
    gate = final.get("gate", {})
    rounds = rep.get("rounds", [])

    coverage = final.get("coverage") or (rounds[-1].get("coverage") if rounds else {})
    N = meta.get("N")
    accepted_total = final.get("accepted_offers", 0)
    offers_total = (len(rounds) * N) if (rounds and N) else 0
    acceptance_rate = (accepted_total / offers_total) if offers_total else None

    # Mutual information metrics per round
    mi_vals = [r.get("mutual_information") for r in rounds if r.get("mutual_information") is not None]
    mi_deltas = [r.get("mi_delta", 0.0) for r in rounds]
    mi_losses = [r.get("mi_loss", 0.0) for r in rounds]
    mi_cum_abs_list = [r.get("mi_cum_abs") for r in rounds if r.get("mi_cum_abs") is not None]
    mi_norm_vals = [r.get("mi_norm") for r in rounds if r.get("mi_norm") is not None]

    mi_ci = None
    if mi_deltas:
        try:
            lo, hi = bootstrap_ci_mean([float(x) for x in mi_deltas], n_resamples=500, ci=0.95, seed=42)
            mi_ci = [lo, hi]
        except Exception:
            mi_ci = None

    mi_final = mi_vals[-1] if mi_vals else None
    mi_max = max(mi_vals) if mi_vals else None
    mi_min = min(mi_vals) if mi_vals else None

    # Aggregated MI activity metrics
    mi_cum_abs = mi_cum_abs_list[-1] if mi_cum_abs_list else (sum(abs(x) for x in mi_deltas) if mi_deltas else None)
    mi_total_loss = sum(mi_losses) if mi_losses else None
    try:
        _D = len(meta.get("domains") or [])
        denom = math.log2(max(2, (N or 0) * _D)) if (N and _D) else 1.0
    except Exception:
        denom = 1.0
    mi_norm_final = (mi_norm_vals[-1] if mi_norm_vals else None) or ((mi_final / denom) if (mi_final is not None and denom > 0) else None)

    # Simple transfer entropy estimate (first two domains if available)
    te_pair = None
    doms = meta.get("domains") or []
    if len(doms) >= 2 and rounds:
        d0, d1 = doms[0], doms[1]
        series0 = [r.get("coverage", {}).get(d0, 0.0) for r in rounds]
        series1 = [r.get("coverage", {}).get(d1, 0.0) for r in rounds]
        try:
            te_pair = transfer_entropy_discrete(series0, series1, k=1, bins=8)
        except Exception:
            te_pair = None

    # Theory prediction
    theory = {}
    lam2 = meta.get("lambda2")
    try:
        if lam2 is not None and N is not None:
            t_pred = predicted_rounds_spectral(int(N), float(lam2))
            t_obs = final.get("observed_t_all")
            if t_obs is not None:
                theory["t_pred"] = t_pred
                theory["t_obs"] = t_obs
                theory["t_delta"] = float(t_obs - t_pred)
    except Exception:
        pass

    return {
        "N": N,
        "topology": meta.get("topology"),
        "lambda2": lam2,
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
            "cum_abs": mi_cum_abs,
            "total_loss": mi_total_loss,
            "norm_final": mi_norm_final,
        },
        "te_pair": te_pair,
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
