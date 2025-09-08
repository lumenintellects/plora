#!/usr/bin/env python
"""scripts.run_lora_value_add - orchestrate value-add experiment.

This script trains domain-specific LoRA adapters, builds placebo controls, then
evaluates all cells in-domain and cross-domain.  Heavy lifting is delegated to
existing helpers in *plora.* and *scripts.train_task*; new statistical helpers
live in *plora.metrics*.

For now the implementation is a skeleton to be fleshed out incrementally.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

from plora.logging_cfg import setup_logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run value-add experiment for LoRA adapters.")

    p.add_argument("--domains", type=lambda s: s.split(","), required=True,
                   help="Comma-separated list of domains e.g. arithmetic,science,legal")
    p.add_argument("--ranks", type=lambda s: [int(x) for x in s.split(",")], required=True,
                   help="Comma-separated list of LoRA ranks, e.g. 4,8,16")
    p.add_argument("--schemes", type=lambda s: s.split(","), default=["all"],
                   help="Target-module schemes: attention,mlp,all (comma-sep)")
    p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[42],
                   help="Comma-separated RNG seeds")
    p.add_argument("--samples", type=int, default=128,
                   help="Training samples per domain (<=1000 for local runs)")
    p.add_argument("--dev-size", type=int, default=256,
                   help="Dev set size for perplexity/metrics evaluation.")
    p.add_argument("--output", type=Path, default=Path("results/value_add"),
                   help="Directory to place JSONL & Markdown reports.")
    p.add_argument("--base-model", default=os.getenv("PLORA_BASE_MODEL", "sshleifer/tiny-gpt2"))
    p.add_argument("--resume", action="store_true",
                   help="Skip configs with existing results and artifacts; append new runs.")

    return p


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    setup_logging("INFO")

    log.info("Starting value-add experiment: domains=%s ranks=%s schemes=%s seeds=%s",
             args.domains, args.ranks, args.schemes, args.seeds)


    args.output.mkdir(parents=True, exist_ok=True)
    placeholder = {
        "status": "running",
        "domains": args.domains,
        "ranks": args.ranks,
        "schemes": args.schemes,
    }

    # Load existing results if resuming
    existing_records = []
    existing_keys = set()
    jsonl_path = args.output / "value_add.jsonl"
    if args.resume and jsonl_path.exists():
        try:
            for line in jsonl_path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                cfg = rec.get("config", {})
                key = (cfg.get("domain"), cfg.get("rank"), cfg.get("scheme"), cfg.get("seed"))
                existing_records.append(rec)
                existing_keys.add(key)
            log.info("Resume enabled: loaded %d existing records from %s", len(existing_records), jsonl_path)
        except Exception as e:
            log.warning("Failed to parse existing %s: %s (continuing without resume)", jsonl_path, e)
            existing_records = []
            existing_keys = set()

    # We'll accumulate NEW JSONL records in memory then merge on write.
    new_records = []

    import random
    import importlib
    from types import SimpleNamespace

    from plora.dataset_loader import get_dataset
    from plora.metrics import token_nlls, paired_wilcoxon, bootstrap_ci, exact_match, chrf_score
    from plora.loader import random_lora, inject
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from plora.compat import device_dtype
    from peft import PeftModel

    # programmatic access to train() to avoid subprocess overhead
    train_mod = importlib.import_module("scripts.train_task")

    base_model_name = args.base_model

    device, dtype = device_dtype()

    def evaluate_pair(adapter_dir: Path | None):
        """Return per-example NLL list. Loads fresh model if adapter_dir provided."""
        if adapter_dir is None:
            return token_nlls(model, tok, dev_pairs)
        # Load adapter into a fresh base model to avoid structural mutation issues
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype, device_map={"": device}
        )
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        return token_nlls(peft_model, tok, dev_pairs)

    # Pre-load dev sets for all domains once
    dev_sets = {d: get_dataset(d, max_samples=args.dev_size) for d in args.domains}

    for domain in args.domains:
        log.info("Domain %s", domain)

        dev_pairs = get_dataset(domain, max_samples=args.dev_size)

        for rank in args.ranks:
            for scheme in args.schemes:
                for seed in args.seeds:
                    random.seed(seed)

                    # Train adapter
                    out_dir = args.output / f"{domain}_r{rank}_{scheme}_seed{seed}"
                    # Early skip if resuming and everything for this config already exists
                    if args.resume:
                        placebo_a_dir = out_dir / "placebo_random"
                        placebo_b_dir = out_dir / "placebo_shuffle"
                        if out_dir.exists() and placebo_a_dir.exists() and placebo_b_dir.exists():
                            log.info("[resume] Skipping completed config %s r=%d %s seed=%d (artifacts present)", domain, rank, scheme, seed)
                            continue
                    if not out_dir.exists():
                        train_mod.train(
                            domain,
                            epochs=1,
                            samples=args.samples,
                            output_dir=out_dir,
                            base_model=base_model_name,
                            shuffle_labels=False,
                            rank=rank,
                            scheme=scheme,
                        )

                    # Baseline model (shared across evaluations for speed)
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model_name, torch_dtype=dtype, device_map={"": device}
                    )
                    tok = AutoTokenizer.from_pretrained(base_model_name)

                    # Cache baseline per domain & seed to avoid recompute
                    baseline_cache = {}

                    def get_baseline_nlls(dom):
                        key = (dom, seed)
                        if key not in baseline_cache:
                            baseline_cache[key] = evaluate_pair(None) if dom == domain else token_nlls(model, tok, dev_sets[dom])
                        return baseline_cache[key]

                    baseline_nlls = get_baseline_nlls(domain)

                    # Trained adapter evaluation
                    trained_nlls = evaluate_pair(out_dir)

                    # Latency budget check, inject+remove median over 3 runs
                    budget_ms = int(os.getenv("PLORA_LATENCY_BUDGET_MS", "250"))
                    lat_samples = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        with inject(model, out_dir):
                            pass
                        lat_samples.append((time.perf_counter() - t0) * 1e3)
                    inject_median = sorted(lat_samples)[len(lat_samples) // 2]

                    # Placebo A, random weights (rank fixed to 8 as per spec)
                    placebo_a_dir = out_dir / "placebo_random"
                    if not placebo_a_dir.exists():
                        random_lora(model, placebo_a_dir, r=rank)
                    placebo_a_nlls = evaluate_pair(placebo_a_dir)

                    # Placebo B, label-shuffle trained
                    placebo_b_dir = out_dir / "placebo_shuffle"
                    if not placebo_b_dir.exists():
                        train_mod.train(
                            domain,
                            epochs=1,
                            samples=args.samples,
                            output_dir=placebo_b_dir,
                            base_model=base_model_name,
                            shuffle_labels=True,
                            rank=8,
                            scheme=scheme,
                        )
                    placebo_b_nlls = evaluate_pair(placebo_b_dir)

                    def make_stat(baseline, after):
                        delta = [a - b for a, b in zip(after, baseline)]
                        stats = paired_wilcoxon(delta)
                        ci_low, ci_high = bootstrap_ci(baseline, after)
                        return {
                            "delta_mean": sum(delta) / len(delta),
                            "wilcoxon_p": stats["p"],
                            "ci": [ci_low, ci_high],
                        }

                    # Cross-domain negative transfer: apply trained adapter to other domains
                    cross = {}
                    for other_dom in args.domains:
                        if other_dom == domain:
                            continue
                        other_dev = dev_sets[other_dom]
                        other_baseline = get_baseline_nlls(other_dom)
                        with inject(model, out_dir) as peft_model:
                            other_after = token_nlls(peft_model, tok, other_dev)
                        cross[other_dom] = {
                            "delta_mean": sum(o - b for o, b in zip(other_after, other_baseline)) / len(other_after)
                        }

                    rec = {
                        "config": {
                            "domain": domain,
                            "rank": rank,
                            "scheme": scheme,
                            "seed": seed,
                        },
                        "trained": make_stat(baseline_nlls, trained_nlls),
                        "placebo_a": make_stat(baseline_nlls, placebo_a_nlls),
                        "placebo_b": make_stat(baseline_nlls, placebo_b_nlls),
                        "cross_domain": cross,
                        "latency_ms": inject_median,
                    }
                    new_records.append(rec)

                    # Guardrail: fail fast if placebo beats baseline significantly or latency budget exceeded
                    placebo_flag = (
                        rec["placebo_a"]["delta_mean"] < -0.5  # material improvement
                        and rec["placebo_a"]["wilcoxon_p"] < 0.01
                    )
                    latency_flag = inject_median > budget_ms

                    if placebo_flag or latency_flag:
                        log.error(
                            "Guardrail breached – trained Δ=%.3f p=%.4g | placeboA Δ=%.3f p=%.4g | latency=%.0f ms>%d",
                            rec["trained"]["delta_mean"],
                            rec["trained"]["wilcoxon_p"],
                            rec["placebo_a"]["delta_mean"],
                            rec["placebo_a"]["wilcoxon_p"],
                            inject_median,
                            budget_ms,
                        )
                        sys.exit(1)

    # Merge existing and new records by config key and write JSONL
    by_key = {}
    def _key_from_rec(r):
        c = r.get("config", {})
        return (c.get("domain"), c.get("rank"), c.get("scheme"), c.get("seed"))

    for r in existing_records:
        by_key[_key_from_rec(r)] = r
    for r in new_records:
        by_key[_key_from_rec(r)] = r

    all_records = list(by_key.values())
    with jsonl_path.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    log.info("Wrote %d records to %s (%d new, %d existing)", len(all_records), jsonl_path, len(new_records), len(existing_records))

    # Generate Markdown summary
    def _fmt(v):
        return f"{v:+.3f}" if isinstance(v, float) else str(v)

    lines = ["# Value-add experiment - summary", ""]
    for domain in args.domains:
        lines.append(f"## Domain: {domain}\n")
        header = "| Cell | r | scheme | ΔNLL | p | 95% CI |"
        sep = "|---|---|---|---|---|---|"
        lines.extend([header, sep])

        dom_recs = [r for r in all_records if r["config"]["domain"] == domain]
        for rec in dom_recs:
            cfg = rec["config"]
            trained = rec["trained"]

            cell_name = f"trained_seed{cfg['seed']}"
            ci = trained["ci"]
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"

            # Highlight if CI strictly < 0 and p<0.01
            passed = trained["ci"][1] < 0 and trained["wilcoxon_p"] < 0.01
            delta_str = f"**{_fmt(trained['delta_mean'])}**" if passed else _fmt(trained['delta_mean'])

            lines.append(
                f"| {cell_name} | {cfg['rank']} | {cfg['scheme']} | {delta_str} | {trained['wilcoxon_p']:.3e} | {ci_str} |"
            )
        lines.append("")

    md_path = args.output / "value_add.md"
    md_path.write_text("\n".join(lines))
    log.info("Markdown summary at %s", md_path)


if __name__ == "__main__":
    main()
