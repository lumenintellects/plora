from __future__ import annotations

"""Train a single LoRA adapter for a given *domain* using tiny data.

Usage (defaults shown)::

    python -m scripts.train_task --domain legal --epochs 1 --samples 128 --output out/legal

Environment variables recognised::

    PLORA_BASE_MODEL     HF model name (default sshleifer/tiny-gpt2)
    PLORA_DEVICE         Force device (cpu|cuda|mps), overrides auto detect.
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from plora.config import get as cfg
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from plora.compat import device_dtype
from plora.dataset_loader import get_dataset
from plora.metrics import perplexity
from plora.logging_cfg import setup_logging
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
from plora.signer import sign_sha256_hex
from plora.targets import select_target_modules

log = logging.getLogger(__name__)
SEED = 42


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


def train(
    domain: str,
    epochs: int,
    samples: int,
    output_dir: Path,
    base_model: str,
    *,
    shuffle_labels: bool = False,
    rank: int = 4,
    scheme: str = "all",
):
    device, dtype = device_dtype()
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.pad_token = tok.eos_token

    pairs = get_dataset(
        domain, max_samples=samples if samples and samples > 0 else None
    )

    # ---------------------------------------------------------------------
    # Optional placebo B: label-shuffled training pairs
    # ---------------------------------------------------------------------
    if shuffle_labels:
        import random

        log.info("Shuffling answers for placebo training (labels only). Seed=%d", SEED)
        random.seed(SEED)
        prompts, answers = zip(*pairs)
        answers = list(answers)
        random.shuffle(answers)
        pairs = list(zip(prompts, answers))

    split = max(1, int(0.8 * len(pairs)))
    train_pairs, dev_pairs = pairs[:split], pairs[split:]

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map={"": device}
    )

    target_modules = select_target_modules(model, scheme)
    l_cfg = LoraConfig(
        r=rank, lora_alpha=rank * 2, target_modules=target_modules, lora_dropout=0.1
    )
    model = get_peft_model(model, l_cfg)

    train_ds = build_dataset(train_pairs, tok)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()

    log.info("Starting training: %d batches * %d epochs", len(train_ds), epochs)
    for epoch in range(epochs):
        for batch in train_ds:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optim.step()
        log.info("Epoch %d loss %.4f", epoch + 1, float(loss))

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "adapter_model.safetensors"
    model.save_pretrained(output_dir, safe_serialization=True)

    # Metrics
    ppl_before = perplexity(
        AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, device_map={"": device}
        ),
        tok,
        dev_pairs,
    )
    ppl_after = perplexity(model, tok, dev_pairs)

    delta_ppl = ppl_after - ppl_before

    log.info("ΔPPL %.3f (before=%.3f, after=%.3f)", delta_ppl, ppl_before, ppl_after)

    # Manifest
    sha_hex = (
        torch.sha256(adapter_path.read_bytes()).hexdigest()
        if hasattr(torch, "sha256")
        else __import__("hashlib").sha256(adapter_path.read_bytes()).hexdigest()
    )
    manifest = Manifest(
        schema_version=0,
        plasmid_id=f"{domain}-{int(time.time())}",
        domain=domain,
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
            seed=SEED,
            epochs=epochs,
            dataset_id=domain,
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
    manifest.dump(output_dir / "plora.yml")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for a domain.")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--samples",
        type=int,
        default=cfg("samples", None),
        help="Number of samples to load (omit for full dataset)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sign", action="store_true")
    parser.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="Permute answers within the batch for placebo B training.",
    )
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank r")
    parser.add_argument("--scheme", choices=["attention", "mlp", "all"], default="all")
    parser.add_argument("--private-key", type=Path)
    args = parser.parse_args()

    setup_logging("INFO")

    base_model = cfg("base_model", os.getenv("PLORA_BASE_MODEL", "sshleifer/tiny-gpt2"))

    manifest = train(
        args.domain,
        args.epochs,
        args.samples,
        args.output,
        base_model,
        shuffle_labels=args.shuffle_labels,
        rank=args.rank,
        scheme=args.scheme,
    )

    if args.sign:
        if not args.private_key:
            raise SystemExit("--private-key required when --sign is used")
        sig = sign_sha256_hex(args.private_key, manifest.artifacts.sha256)
        manifest.signer = manifest.signer.copy(
            update={"algo": "RSA-PSS-SHA256", "signature_b64": sig}
        )
        manifest.dump(args.output / "plora.yml")
        log.info("Signed manifest written.")


if __name__ == "__main__":
    main()
