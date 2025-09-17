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
import sys
import time
from pathlib import Path
from typing import List

from plora.logging_cfg import setup_logging
from plora.config import get as cfg

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run value-add experiment for LoRA adapters."
    )

    p.add_argument(
        "--domains",
        type=lambda s: s.split(","),
        default=cfg("domains", ["arithmetic", "legal", "medical"]),
        help="Comma-separated list of domains e.g. arithmetic,science,legal",
    )
    p.add_argument(
        "--ranks",
        type=lambda s: [int(x) for x in s.split(",")],
        default=cfg("value_add.ranks", [4, 8, 16]),
        help="Comma-separated list of LoRA ranks, e.g. 4,8,16",
    )
    p.add_argument(
        "--schemes",
        type=lambda s: s.split(","),
        default=cfg("value_add.schemes", ["all"]),
        help="Target-module schemes: attention,mlp,all (comma-sep)",
    )
    p.add_argument(
        "--seeds",
        type=lambda s: [int(x) for x in s.split(",")],
        default=cfg("value_add.seeds", [42]),
        help="Comma-separated RNG seeds",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=cfg("samples", 64),
        help="Training samples per domain",
    )
    p.add_argument(
        "--dev-size",
        type=int,
        default=cfg("value_add.dev_size", 256),
        help="Dev set size for perplexity/metrics evaluation.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/value_add"),
        help="Directory to place JSONL & Markdown reports.",
    )
    p.add_argument("--base-model", default=cfg("base_model", "google/gemma-3-1b-it"))
    p.add_argument(
        "--eval-split",
        default=cfg("eval_split", "validation"),
        help="Evaluation split for value-add (validation|test).",
    )
    p.add_argument(
        "--placebo-b-rank",
        type=int,
        default=cfg("value_add.placebo_b_rank", 8),
        help="Rank to use for placebo-B (label-shuffle) training",
    )

    return p


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    setup_logging("INFO")

    log.info(
        "Starting value-add experiment: domains=%s ranks=%s schemes=%s seeds=%s",
        args.domains,
        args.ranks,
        args.schemes,
        args.seeds,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    placeholder = {
        "status": "running",
        "domains": args.domains,
        "ranks": args.ranks,
        "schemes": args.schemes,
    }

    # We'll accumulate JSONL records in memory then write once.
    records = []

    # On-disk cache for NLL lists to speed re-runs
    cache_path = args.output / "nll_cache.json"
    try:
        nll_cache = (
            json.loads(cache_path.read_text())
            if cache_path.exists()
            else {"baseline": {}, "adapter": {}}
        )
    except Exception:
        nll_cache = {"baseline": {}, "adapter": {}}

    def _save_cache():
        try:
            cache_path.write_text(json.dumps(nll_cache))
        except Exception:
            pass

    import random
    import importlib

    from plora.dataset_loader import get_dataset
    from plora.metrics import (
        token_nlls,
        paired_wilcoxon,
        bootstrap_ci,
    )
    from plora.loader import random_lora, inject
    from plora.manifest import Manifest
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from plora.compat import device_dtype
    from peft import PeftModel
    from plora.peft_safety import assert_pristine_base

    # programmatic access to train() to avoid subprocess overhead
    train_mod = importlib.import_module("scripts.train_task")

    base_model_name = args.base_model

    device, dtype = device_dtype()

    def evaluate_pair(adapter_dir: Path | None, domain: str):
        """Return per-example NLL list. Loads fresh model if adapter_dir provided."""
        if adapter_dir is None:
            return token_nlls(model, tok, dev_sets[domain])
        # Adapter cache key by artifact SHA when available + eval split
        try:
            man = Manifest.from_adapter(adapter_dir)
            cache_key = f"{man.artifacts.sha256}:{args.eval_split}"
        except Exception:
            cache_key = f"{str(adapter_dir)}:{args.eval_split}"
        cached = nll_cache["adapter"].get(cache_key)
        if cached is not None:
            return cached
        # Load adapter into a fresh base model to avoid structural mutation issues
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=dtype,
            attn_implementation="eager",
            device_map={"": device},
        )
        peft_model = PeftModel.from_pretrained(
            base, str(adapter_dir), is_trainable=False
        )
        vals = token_nlls(peft_model, tok, dev_sets[domain])
        nll_cache["adapter"][cache_key] = vals
        _save_cache()
        return vals

    # Pre-load dev sets for all domains once (split-aware)
    dev_sets = {
        d: get_dataset(d, max_samples=args.dev_size, split=args.eval_split)
        for d in args.domains
    }

    for domain in args.domains:
        log.info("Domain %s", domain)

        # Baseline model and tokenizer (shared across evaluations within this domain)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=dtype,
            attn_implementation="eager",
            device_map={"": device},
        )
        assert_pristine_base(model, where=f"value_add.baseline.domain={domain}")
        tok = AutoTokenizer.from_pretrained(base_model_name)

        # Cache baseline per domain, split & seed to avoid recompute across inner loops
        baseline_cache = {}

        for rank in args.ranks:
            for scheme in args.schemes:
                for seed in args.seeds:
                    random.seed(seed)

                    # Train adapter on train split
                    rank_root = args.output / f"rank_r{rank}"
                    out_dir = rank_root / f"{domain}_{scheme}_seed{seed}"
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

                    def get_baseline_nlls(dom):
                        key = (dom, seed, args.eval_split)
                        if key not in baseline_cache:
                            bkey = f"{base_model_name}:{dom}:{args.dev_size}:{args.eval_split}"
                            cached_b = nll_cache["baseline"].get(bkey)
                            if cached_b is None:
                                vals = token_nlls(model, tok, dev_sets[dom])
                                nll_cache["baseline"][bkey] = vals
                                _save_cache()
                                baseline_cache[key] = vals
                            else:
                                baseline_cache[key] = cached_b
                        return baseline_cache[key]

                    baseline_nlls = get_baseline_nlls(domain)

                    # Trained adapter evaluation
                    trained_nlls = evaluate_pair(out_dir, domain)

                    # Latency budget check, inject+remove median over 3 runs
                    budget_ms = cfg("latency_budget_ms", 250)
                    lat_samples = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        with inject(model, out_dir):
                            pass
                        lat_samples.append((time.perf_counter() - t0) * 1e3)
                    inject_median = sorted(lat_samples)[len(lat_samples) // 2]

                    # Placebo A, random weights (rank fixed to r)
                    placebo_a_dir = (
                        rank_root / f"{domain}_{scheme}_seed{seed}_placebo_random"
                    )
                    if not placebo_a_dir.exists():
                        random_lora(base_model_name, placebo_a_dir, r=rank)
                    placebo_a_nlls = evaluate_pair(placebo_a_dir, domain)

                    # Placebo B, label-shuffle trained (train split)
                    placebo_b_dir = (
                        rank_root / f"{domain}_{scheme}_seed{seed}_placebo_shuffle"
                    )
                    if not placebo_b_dir.exists():
                        train_mod.train(
                            domain,
                            epochs=1,
                            samples=args.samples,
                            output_dir=placebo_b_dir,
                            base_model=base_model_name,
                            shuffle_labels=True,
                            rank=int(args.placebo_b_rank),
                            scheme=scheme,
                        )
                    placebo_b_nlls = evaluate_pair(placebo_b_dir, domain)

                    def make_stat(baseline, after):
                        delta = [a - b for a, b in zip(after, baseline)]
                        stats = paired_wilcoxon(delta)
                        ci_low, ci_high = bootstrap_ci(baseline, after)
                        return {
                            "delta_mean": sum(delta) / len(delta),
                            "wilcoxon_p": stats["p"],
                            "ci": [ci_low, ci_high],
                        }

                    # Cross-domain negative/positive transfer: apply trained adapter to other domains
                    cross = {}
                    for other_dom in args.domains:
                        if other_dom == domain:
                            continue
                        other_baseline = get_baseline_nlls(other_dom)
                        with inject(model, out_dir) as peft_model:
                            other_after = token_nlls(
                                peft_model, tok, dev_sets[other_dom]
                            )
                        cross[other_dom] = {
                            "delta_mean": sum(
                                o - b for o, b in zip(other_after, other_baseline)
                            )
                            / len(other_after)
                        }

                    rec = {
                        "config": {
                            "domain": domain,
                            "rank": rank,
                            "scheme": scheme,
                            "seed": seed,
                            "eval_split": args.eval_split,
                        },
                        "trained": make_stat(baseline_nlls, trained_nlls),
                        "placebo_a": make_stat(baseline_nlls, placebo_a_nlls),
                        "placebo_b": make_stat(baseline_nlls, placebo_b_nlls),
                        "cross_domain": cross,
                        "latency_ms": inject_median,
                    }
                    records.append(rec)

                    # Guardrail: flag latency only (placebo significance may vary with split)
                    latency_flag = inject_median > budget_ms

                    if latency_flag:
                        log.error(
                            f"Guardrail breached – latency={inject_median} ms>{budget_ms}"
                        )
                        sys.exit(1)

    # Write JSONL
    jsonl_path = args.output / "value_add.jsonl"
    with jsonl_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    log.info("Wrote %d records to %s", len(records), jsonl_path)

    # Generate Markdown summary
    def _fmt(v):
        return f"{v:+.3f}" if isinstance(v, float) else str(v)

    lines = ["# Value-add experiment – summary", ""]
    for domain in args.domains:
        lines.append(f"## Domain: {domain}\n")
        header = "| Cell | r | scheme | ΔNLL | p | 95% CI | split |"
        sep = "|---|---|---|---|---|---|---|"
        lines.extend([header, sep])

        dom_recs = [r for r in records if r["config"]["domain"] == domain]
        for rec in dom_recs:
            rec_cfg = rec["config"]
            trained = rec["trained"]

            cell_name = f"trained_seed{rec_cfg['seed']}"
            ci = trained["ci"]
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"

            # Highlight if CI strictly < 0 and p<0.01
            passed = trained["ci"][1] < 0 and trained["wilcoxon_p"] < 0.01
            delta_str = (
                f"**{_fmt(trained['delta_mean'])}**"
                if passed
                else _fmt(trained["delta_mean"])
            )

            lines.append(
                f"| {cell_name} | {rec_cfg['rank']} | {rec_cfg['scheme']} | {delta_str} | {trained['wilcoxon_p']:.3e} | {ci_str} | {rec_cfg['eval_split']} |"
            )
        lines.append("")

    md_path = args.output / "value_add.md"
    md_path.write_text("\n".join(lines))
    log.info("Markdown summary at %s", md_path)


if __name__ == "__main__":
    main()
