from __future__ import annotations

"""Alternating train-merge schedule runner.

Runs local LoRA fine-tuning for a short cadence, merges adapters with weighted
and trust-region scaling, then repeats for K cycles. Useful for studying
stability and convergence under composition.
"""

import argparse
from pathlib import Path
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

from plora.compat import device_dtype
from plora.config import get as cfg
from plora.loader import merge_plasmids
from scripts.train_task import get_dataset, select_target_modules


def _train_once(
    domain: str, base_model: str, samples: int, rank: int, out_dir: Path
) -> Path:
    device, dtype = device_dtype()
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.pad_token = tok.eos_token
    pairs = get_dataset(domain, max_samples=samples)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map={"": device}
    )
    tmods = select_target_modules(model, scheme="attention") or select_target_modules(
        model, scheme="all"
    )
    cfg = LoraConfig(
        r=rank, lora_alpha=rank * 2, target_modules=tmods, lora_dropout=0.1
    )
    model = get_peft_model(model, cfg)

    # simple one-epoch pass over tiny dataset
    encs = []
    for p, a in pairs:
        text = f"Question: {p}\nAnswer: {a}"
        enc = tok(text, return_tensors="pt", truncation=True)
        enc["labels"] = enc["input_ids"].clone()
        encs.append({k: v.to(device) for k, v in enc.items()})

    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()
    for batch in encs:
        optim.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optim.step()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    return out_dir


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", type=lambda s: s.split(","), required=True)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--samples", type=int, default=cfg("samples", 64))
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--base_model", type=str, default=cfg("base_model", None))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_delta_fro", type=float, default=0.0)
    ap.add_argument(
        "--line_search_scales",
        type=lambda s: [float(x) for x in s.split(",")],
        default=[1.0, 0.5, 0.25],
    )
    ns = ap.parse_args(argv)

    base_model = ns.base_model or Path().joinpath().anchor or "sshleifer/tiny-gpt2"

    # Cycle: train per-domain, then merge into a consolidated model
    losses: List[float] = []
    for c in range(ns.cycles):
        adapters: List[Path] = []
        for d in ns.domains:
            adir = ns.out / f"cycle{c}" / d
            adapters.append(_train_once(d, base_model, ns.samples, ns.rank, adir))

        # Simple line search over global_scale to minimise parameter delta to previous merged
        best_scale = 1.0
        best_score = float("inf")
        cand_scales = ns.line_search_scales or [1.0]
        prev_model_state = None
        if c > 0:
            from transformers import AutoModelForCausalLM

            device, dtype = device_dtype()
            prev = AutoModelForCausalLM.from_pretrained(
                str(ns.out / f"merged_cycle{c-1}"),
                torch_dtype=dtype,
                device_map={"": device},
            )
            prev_model_state = prev.state_dict()
        for scale in cand_scales:
            m = merge_plasmids(
                base_model,
                adapters,
                strategy="weighted_sum",
                reproject_rank=ns.rank,
                max_delta_fro=(ns.max_delta_fro if ns.max_delta_fro > 0 else None),
                global_scale=scale,
            )
            if prev_model_state is None:
                best_scale = scale
                merged = m
                break
            # score: Frobenius of delta to previous merged
            s = 0.0
            with torch.no_grad():
                for k, va in m.state_dict().items():
                    vb = prev_model_state.get(k)
                    if vb is None or vb.shape != va.shape:
                        continue
                    s += float(
                        ((va.to(torch.float32) - vb.to(torch.float32)) ** 2)
                        .sum()
                        .item()
                    )
            if s < best_score:
                best_score = s
                best_scale = scale
                merged = m
        # Save merged backbone (weights merged in-place already)
        save_dir = ns.out / f"merged_cycle{c}"
        save_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(save_dir, safe_serialization=True)

        # Convergence diagnostics: parameter delta norm relative to previous merged
        if c > 0:
            from transformers import AutoModelForCausalLM

            device, dtype = device_dtype()
            prev = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=dtype, device_map={"": device}
            )
            # load previous saved if exists
            try:
                prev = AutoModelForCausalLM.from_pretrained(
                    str(ns.out / f"merged_cycle{c-1}"),
                    torch_dtype=dtype,
                    device_map={"": device},
                )
            except Exception:
                pass
            s = 0.0
            with torch.no_grad():
                sda = merged.state_dict()
                sdb = prev.state_dict()
                for k, va in sda.items():
                    vb = sdb.get(k)
                    if vb is None or vb.shape != va.shape:
                        continue
                    s += float(
                        ((va.to(torch.float32) - vb.to(torch.float32)) ** 2)
                        .sum()
                        .item()
                    )
            losses.append(s**0.5)

    # Write a small diagnostics file
    try:
        diag = ns.out / "convergence.json"
        import json

        diag.write_text(json.dumps({"param_delta_fro": losses}, indent=2))
    except Exception:
        pass


if __name__ == "__main__":
    main()
