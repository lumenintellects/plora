from __future__ import annotations

"""Summarise Swarm Sim reports (v2) into a concise JSON/Markdown.

Usage:
  python -m scripts.evaluate_v2 --reports results --out results/summary_v2.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


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
    return {
        "N": meta.get("N"),
        "topology": meta.get("topology"),
        "t_all": final.get("t_all"),
        "bytes_on_wire": final.get("bytes_on_wire", 0),
        "accepted_offers": final.get("accepted_offers", 0),
        "coverage": coverage or {},
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
