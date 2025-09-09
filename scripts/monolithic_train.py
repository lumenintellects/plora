from __future__ import annotations

"""Train a monolithic LoRA adapter over multiple domains (small, fast loop).

Usage:
  python -m scripts.monolithic_train --domains arithmetic,legal,medical \
      --epochs 1 --samples 64 --rank 4 --output out/monolithic_r4
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from plora.compat import device_dtype
from plora.dataset_loader import get_dataset
from plora.manifest import (
    Manifest,
    LoraConfig as ManLoraCfg,
    ArtifactsInfo,
    TrainMeta,
    MetricsInfo,
    SafetyInfo,
    SignerInfo,
    CompatibilityInfo,
)
from plora.metrics import perplexity
from plora.targets import select_target_modules


def build_dataset(pairs: List[Tuple[str, str]], tok, max_len: int = 128):
    encs = []
    for q, a in pairs:
        text = f"Question: {q}\nAnswer: {a}"
        enc = tok(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc["labels"] = enc["input_ids"].clone()
        encs.append(enc)
    return encs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", type=lambda s: s.split(","), required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    ns = ap.parse_args()

    base_model = os.getenv("PLORA_BASE_MODEL", "sshleifer/tiny-gpt2")
    device, dtype = device_dtype()

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.pad_token = tok.eos_token

    pairs: List[Tuple[str, str]] = []
    for d in ns.domains:
        pairs.extend(get_dataset(d, max_samples=ns.samples))

    split = max(1, int(0.8 * len(pairs)))
    train_pairs, dev_pairs = pairs[:split], pairs[split:]

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map={"": device}
    )
    # Dynamically select valid target modules for the base model
    tmods = select_target_modules(model, scheme="attention")
    if not tmods:
        # Fallback to a broader selection
        tmods = select_target_modules(model, scheme="all")
    if not tmods:
        raise ValueError(
            "No compatible LoRA target modules found for base model; check architecture and scheme"
        )
    l_cfg = LoraConfig(
        r=ns.rank, lora_alpha=ns.rank * 2, target_modules=tmods, lora_dropout=0.1
    )
    model = get_peft_model(model, l_cfg)

    train_ds = build_dataset(train_pairs, tok)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()
    for _ in range(ns.epochs):
        for batch in train_ds:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optim.step()

    ns.output.mkdir(parents=True, exist_ok=True)
    adapter_path = ns.output / "adapter_model.safetensors"
    model.save_pretrained(ns.output, safe_serialization=True)

    ppl_before = perplexity(
        AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, device_map={"": device}
        ),
        tok,
        dev_pairs,
    )
    ppl_after = perplexity(model, tok, dev_pairs)
    delta_ppl = ppl_after - ppl_before

    sha_hex = (
        torch.sha256(adapter_path.read_bytes()).hexdigest()
        if hasattr(torch, "sha256")
        else __import__("hashlib").sha256(adapter_path.read_bytes()).hexdigest()
    )
    manifest = Manifest(
        schema_version=0,
        plasmid_id=f"monolithic-{int(time.time())}",
        domain="monolithic",
        base_model=base_model,
        peft_format="lora",
        lora=ManLoraCfg(
            r=l_cfg.r,
            alpha=l_cfg.lora_alpha,
            dropout=l_cfg.lora_dropout,
            target_modules=l_cfg.target_modules,
        ),
        artifacts=ArtifactsInfo(
            filename=adapter_path.name,
            sha256=sha_hex,
            size_bytes=adapter_path.stat().st_size,
        ),
        train_meta=TrainMeta(
            seed=42,
            epochs=ns.epochs,
            dataset_id="+".join(ns.domains),
            sample_count=len(pairs),
            timestamp_unix=int(time.time()),
        ),
        metrics=MetricsInfo(
            val_ppl_before=ppl_before,
            val_ppl_after=ppl_after,
            delta_ppl=delta_ppl,
            val_em=None,
            val_chrf=None,
        ),
        safety=SafetyInfo(licence="research", poisoning_score=0.0),
        signer=SignerInfo(algo="", pubkey_fingerprint="", signature_b64=""),
        compatibility=CompatibilityInfo(peft_min="0.12.0", transformers=">=4.42"),
    )
    manifest.dump(ns.output / "plora.yml")


if __name__ == "__main__":
    main()
