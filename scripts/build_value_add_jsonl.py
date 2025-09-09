#!/usr/bin/env python
"""Build value_add.jsonl from scratch by evaluating artifacts (memory-safe).

Optimisations to avoid high RAM/VRAM usage:
- Reuse a single baseline model + tokenizer per base model group (no reload per config).
- Avoid inject(), which clones the entire state dict; instead, load a temporary
  base model and wrap with PeftModel for each adapter (trained/placebos), then
  delete and empty cache immediately after use.
- Cap tokenization length with --max-length to bound per-example memory/time.
- Cache baseline per-domain NLLs for each base model to avoid recomputation, with optional on-disk cache.
- Optional skips: --skip-placebos, --skip-cross for faster, lower-resource runs.
- Write JSONL incrementally after each record so progress is never lost.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from plora.dataset_loader import get_dataset
from plora.metrics import paired_wilcoxon, bootstrap_ci
from plora.compat import device_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

CONFIG_RE = re.compile(r"^(?P<domain>[^_]+)_r(?P<rank>\d+?)_(?P<scheme>[^_]+)_seed(?P<seed>\d+)$")


@dataclass(frozen=True)
class Config:
    domain: str
    rank: int
    scheme: str
    seed: int
    dir: Path

    @classmethod
    def from_dir(cls, d: Path) -> Optional["Config"]:
        m = CONFIG_RE.match(d.name)
        if not m:
            return None
        return cls(domain=m.group("domain"), rank=int(m.group("rank")), scheme=m.group("scheme"), seed=int(m.group("seed")), dir=d)

    def key(self) -> Tuple[str, int, str, int]:
        return (self.domain, self.rank, self.scheme, self.seed)


def parse_base_model_from_artifact(dir_: Path) -> Optional[str]:
    # Try plora.yml
    yml = dir_ / "plora.yml"
    if yml.exists():
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(yml.read_text())
            bm = data.get("base_model") or data.get("base-model")
            if bm:
                return str(bm)
        except Exception:
            pass
    # Try README front matter: line starting with 'base_model:'
    readme = dir_ / "README.md"
    if readme.exists():
        try:
            for line in readme.read_text().splitlines():
                if line.strip().startswith("base_model:"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    # Try adapter_config.json (non-standard)
    acfg = dir_ / "adapter_config.json"
    if acfg.exists():
        try:
            data = json.loads(acfg.read_text())
            bm = data.get("base_model_name") or data.get("base_model")
            if bm:
                return str(bm)
        except Exception:
            pass
    return None


def discover_configs(root: Path, allow_domains: Optional[Iterable[str]] = None) -> List[Config]:
    allow = set(allow_domains) if allow_domains else None
    cfgs: List[Config] = []
    for child in sorted([p for p in root.iterdir() if p.is_dir()]):
        cfg = Config.from_dir(child)
        if not cfg:
            continue
        if allow and cfg.domain not in allow:
            continue
        cfgs.append(cfg)
    cfgs.sort(key=lambda c: (c.domain, c.rank, c.scheme, c.seed))
    return cfgs


def _ensure_cache_dir(base_output: Path) -> Path:
    d = base_output.parent / ".cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _baseline_cache_path(cache_dir: Path, base_model: str, domain: str, dev_size: int, max_length: int) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_model)
    return cache_dir / f"baseline_{safe_model}_{domain}_dev{dev_size}_len{max_length}.json"


def _adapter_cache_path(cache_dir: Path, base_model: str, adapter_dir: Path, domain: str, dev_size: int, max_length: int) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_model)
    safe_adapter = re.sub(r"[^A-Za-z0-9_.-]+", "_", adapter_dir.name)
    return cache_dir / f"adapter_{safe_model}_{safe_adapter}_{domain}_dev{dev_size}_len{max_length}.json"


def _model_load_kwargs(device: torch.device) -> dict:
    # Lower peak RAM; allow HF to shard weights smartly
    return {
        "low_cpu_mem_usage": True,
        "device_map": {"": device},
    }


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def build_encodings(tok, dataset: List[tuple[str, str]], max_length: int) -> List[dict]:
    encs: List[dict] = []
    for prompt, answer in dataset:
        text = f"Question: {prompt}\nAnswer: {answer}"
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        # Keep tensors on CPU; we move per-example during eval
        encs.append({"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask")})
    return encs


@torch.inference_mode()
def compute_token_nlls_from_encs(model, encs: List[dict], device: torch.device) -> List[float]:
    model.eval()
    nlls: List[float] = []
    for enc in encs:
        batch = _to_device(enc, device)
        labels = batch["input_ids"].clone()
        loss: torch.Tensor = model(**batch, labels=labels).loss
        nlls.append(float(loss.item()))
        del batch, labels, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return nlls


def make_stat(baseline: List[float], after: List[float]) -> dict:
    delta = [a - b for a, b in zip(after, baseline)]
    stats = paired_wilcoxon(delta)
    ci_low, ci_high = bootstrap_ci(baseline, after)
    return {
        "delta_mean": sum(delta) / len(delta),
        "wilcoxon_p": stats["p"],
        "ci": [ci_low, ci_high],
    }


def load_base(base_model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device, torch.dtype]:
    device, dtype = device_dtype()
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, **_model_load_kwargs(device))
    tok = AutoTokenizer.from_pretrained(base_model_name)
    return model, tok, device, dtype


def load_peft(base_model_name: str, adapter_dir: Path, device: torch.device, dtype: torch.dtype) -> PeftModel:
    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, **_model_load_kwargs(device))
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    return peft_model


def eval_one_config(
    cfg: Config,
    base_model_name: str,
    tok,
    device: torch.device,
    dtype: torch.dtype,
    encs_by_domain: Dict[str, List[dict]],
    baseline_cache_mem: Dict[tuple[str, str], List[float]],
    max_length: int,
    latency_trials: int,
    *,
    skip_placebos: bool = False,
    skip_cross: bool = False,
    cache_dir: Optional[Path] = None,
    use_adapter_cache: bool = True,
    dev_size: int = 0,
) -> dict:
    key_med = (base_model_name, cfg.domain)
    if key_med not in baseline_cache_mem:
        raise RuntimeError(f"Missing baseline cache for {(base_model_name, cfg.domain)}")

    # Trained adapter, check cache, else compute
    trained_cache_path = None
    trained_nlls: List[float]
    if use_adapter_cache and cache_dir is not None:
        trained_cache_path = _adapter_cache_path(cache_dir, base_model_name, cfg.dir, cfg.domain, dev_size, max_length)
        if trained_cache_path.exists():
            try:
                trained_nlls = json.loads(trained_cache_path.read_text())
            except Exception:
                trained_nlls = []
        else:
            trained_nlls = []
    else:
        trained_nlls = []

    if not trained_nlls:
        trained_model = load_peft(base_model_name, cfg.dir, device, dtype)
        trained_nlls = compute_token_nlls_from_encs(trained_model, encs_by_domain[cfg.domain], device)
        # Cleanup trained model
        del trained_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if use_adapter_cache and trained_cache_path is not None:
            try:
                trained_cache_path.write_text(json.dumps(trained_nlls))
            except Exception:
                pass

    # Latency, measure adapter load time into a fresh base
    lat_samples: List[float] = []
    for _ in range(latency_trials):
        t0 = time.perf_counter()
        tmp = load_peft(base_model_name, cfg.dir, device, dtype)
        lat_ms = (time.perf_counter() - t0) * 1e3
        lat_samples.append(lat_ms)
        del tmp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    inject_median = sorted(lat_samples)[len(lat_samples) // 2] if lat_samples else 0.0

    # Placebos
    placebo_a = None
    placebo_b = None
    if not skip_placebos:
        placebo_a_dir = cfg.dir / "placebo_random"
        placebo_b_dir = cfg.dir / "placebo_shuffle"
        if placebo_a_dir.exists():
            pa_cache_path = _adapter_cache_path(cache_dir or Path('.'), base_model_name, placebo_a_dir, cfg.domain, dev_size, max_length) if use_adapter_cache else None
            pa_nlls: List[float] = []
            if pa_cache_path is not None and pa_cache_path.exists():
                try:
                    pa_nlls = json.loads(pa_cache_path.read_text())
                except Exception:
                    pa_nlls = []
            if not pa_nlls:
                pa_model = load_peft(base_model_name, placebo_a_dir, device, dtype)
                pa_nlls = compute_token_nlls_from_encs(pa_model, encs_by_domain[cfg.domain], device)
                del pa_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if pa_cache_path is not None:
                    try:
                        pa_cache_path.write_text(json.dumps(pa_nlls))
                    except Exception:
                        pass
            placebo_a = make_stat(baseline_cache_mem[key_med], pa_nlls)
        if placebo_b_dir.exists():
            pb_cache_path = _adapter_cache_path(cache_dir or Path('.'), base_model_name, placebo_b_dir, cfg.domain, dev_size, max_length) if use_adapter_cache else None
            pb_nlls: List[float] = []
            if pb_cache_path is not None and pb_cache_path.exists():
                try:
                    pb_nlls = json.loads(pb_cache_path.read_text())
                except Exception:
                    pb_nlls = []
            if not pb_nlls:
                pb_model = load_peft(base_model_name, placebo_b_dir, device, dtype)
                pb_nlls = compute_token_nlls_from_encs(pb_model, encs_by_domain[cfg.domain], device)
                del pb_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if pb_cache_path is not None:
                    try:
                        pb_cache_path.write_text(json.dumps(pb_nlls))
                    except Exception:
                        pass
            placebo_b = make_stat(baseline_cache_mem[key_med], pb_nlls)

    # Cross-domain deltas
    cross: Dict[str, dict] = {}
    if not skip_cross:
        # Load trained model once for cross-domain evals if we don't have cached per-domain NLLs
        trained_model_for_cross = None
        for other, encs in encs_by_domain.items():
            if other == cfg.domain:
                continue
            key_other = (base_model_name, other)
            if key_other not in baseline_cache_mem:
                raise RuntimeError(f"Missing baseline cache for {(base_model_name, other)}")
            cached_other_path = _adapter_cache_path(cache_dir or Path('.'), base_model_name, cfg.dir, other, dev_size, max_length) if use_adapter_cache else None
            other_after: List[float] = []
            if cached_other_path is not None and cached_other_path.exists():
                try:
                    other_after = json.loads(cached_other_path.read_text())
                except Exception:
                    other_after = []
            if not other_after:
                if trained_model_for_cross is None:
                    trained_model_for_cross = load_peft(base_model_name, cfg.dir, device, dtype)
                other_after = compute_token_nlls_from_encs(trained_model_for_cross, encs, device)
                if cached_other_path is not None:
                    try:
                        cached_other_path.write_text(json.dumps(other_after))
                    except Exception:
                        pass
            deltas = [a - b for a, b in zip(other_after, baseline_cache_mem[key_other])]
            cross[other] = {"delta_mean": sum(deltas) / len(deltas)}
        if trained_model_for_cross is not None:
            del trained_model_for_cross
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    rec = {
        "config": {
            "domain": cfg.domain,
            "rank": cfg.rank,
            "scheme": cfg.scheme,
            "seed": cfg.seed,
        },
        "trained": make_stat(baseline_cache_mem[key_med], trained_nlls),
        "placebo_a": placebo_a,
        "placebo_b": placebo_b,
        "cross_domain": cross,
        "latency_ms": inject_median,
    }
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build value_add.jsonl from artifacts with full evaluation (memory-safe)")
    ap.add_argument("--artifacts-dir", type=Path, default=Path("results/value_add"))
    ap.add_argument("--output", type=Path, default=Path("results/value_add/value_add.jsonl"))
    ap.add_argument("--domains", type=lambda s: s.split(","), default=["arithmetic", "legal", "medical"], help="Domain list for dev/eval & cross-transfer")
    ap.add_argument("--dev-size", type=int, default=512)
    ap.add_argument("--max-length", type=int, default=512, help="Tokenizer max_length for truncation")
    ap.add_argument("--base-model", type=str, default=os.getenv("PLORA_BASE_MODEL", ""), help="Fallback base model if not found in artifacts")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if exists; otherwise append/update records atomically")
    ap.add_argument("--filter", type=str, default="", help="Optional regex to match artifact dir names")
    ap.add_argument("--latency-trials", type=int, default=1, help="Trials to estimate adapter load latency (kept small to save time/memory)")
    ap.add_argument("--no-disk-cache", action="store_true", help="Disable on-disk baseline NLL cache")
    ap.add_argument("--skip-placebos", action="store_true", help="Skip evaluating placebo adapters to save time/memory")
    ap.add_argument("--skip-cross", action="store_true", help="Skip cross-domain evaluation to save time/memory")
    ap.add_argument("--no-adapter-cache", action="store_true", help="Disable on-disk adapter NLL cache")
    args = ap.parse_args(argv)

    root = args.artifacts_dir
    if not root.exists():
        print(f"Artifacts dir not found: {root}", file=sys.stderr)
        return 2

    cfgs = [c for c in discover_configs(root) if (not args.filter or re.search(args.filter, c.dir.name))]
    if not cfgs:
        print("No artifact configs discovered.", file=sys.stderr)
        return 3

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing (unless overwriting)
    existing: Dict[Tuple[str, int, str, int], dict] = {}
    if out_path.exists() and not args.overwrite:
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            c = rec.get("config", {})
            key = (c.get("domain"), c.get("rank"), c.get("scheme"), c.get("seed"))
            existing[key] = rec

    by_key: Dict[Tuple[str, int, str, int], dict] = dict(existing)

    # Resolve base model per config and group
    groups: Dict[str, List[Config]] = {}
    unresolved: List[Config] = []
    for cfg in cfgs:
        bm = parse_base_model_from_artifact(cfg.dir) or args.base_model
        if not bm:
            unresolved.append(cfg)
            continue
        groups.setdefault(bm, []).append(cfg)

    # Write failures for unresolved base model
    for cfg in unresolved:
        by_key[cfg.key()] = {
            "config": {"domain": cfg.domain, "rank": cfg.rank, "scheme": cfg.scheme, "seed": cfg.seed},
            "status": "eval_failed:base_model_unknown",
            "trained": None,
            "placebo_a": None,
            "placebo_b": None,
            "cross_domain": None,
            "latency_ms": None,
        }

    # Preload dev sets once and pre-encode (shared across groups)
    raw_dev_sets: Dict[str, List[tuple[str, str]]] = {}
    for d in args.domains:
        raw_dev_sets[d] = get_dataset(d, max_samples=args.dev_size)

    cache_dir = _ensure_cache_dir(out_path)
    use_disk_cache = not args.no_disk_cache
    use_adapter_cache = not args.no_adapter_cache

    # For each base-model group: build tokenizer, pre-encode, compute baselines, then eval adapters
    total = sum(len(v) for v in groups.values()) + len(unresolved)
    done = 0
    for bm_name, cfg_list in groups.items():
        # Load tokenizer and device/dtype
        device, dtype = device_dtype()
        tok = AutoTokenizer.from_pretrained(bm_name)
        # Ensure pad token exists for consistent batching/padding behavior
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        # Pre-encode per-domain once per base model
        encs_by_domain: Dict[str, List[dict]] = {dom: build_encodings(tok, raw_dev_sets[dom], args.max_length) for dom in args.domains}
        # Compute/restore baseline caches per domain; free baseline model immediately after
        baseline_model = AutoModelForCausalLM.from_pretrained(bm_name, torch_dtype=dtype, **_model_load_kwargs(device))
        baseline_cache_mem: Dict[tuple[str, str], List[float]] = {}
        try:
            for dom in args.domains:
                key = (bm_name, dom)
                if use_disk_cache:
                    p = _baseline_cache_path(cache_dir, bm_name, dom, args.dev_size, args.max_length)
                    if p.exists():
                        try:
                            baseline_cache_mem[key] = json.loads(p.read_text())
                            continue
                        except Exception:
                            pass
                nlls = compute_token_nlls_from_encs(baseline_model, encs_by_domain[dom], device)
                baseline_cache_mem[key] = nlls
                if use_disk_cache:
                    try:
                        p = _baseline_cache_path(cache_dir, bm_name, dom, args.dev_size, args.max_length)
                        p.write_text(json.dumps(nlls))
                    except Exception:
                        pass
        finally:
            del baseline_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        # Evaluate all configs for this base model
        try:
            for cfg in cfg_list:
                try:
                    rec = eval_one_config(
                        cfg,
                        bm_name,
                        tok,
                        device,
                        dtype,
                        encs_by_domain,
                        baseline_cache_mem,
                        args.max_length,
                        args.latency_trials,
                        skip_placebos=args.skip_placebos,
                        skip_cross=args.skip_cross,
                        cache_dir=cache_dir,
                        use_adapter_cache=use_adapter_cache,
                        dev_size=args.dev_size,
                    )
                except Exception as e:
                    rec = {
                        "config": {"domain": cfg.domain, "rank": cfg.rank, "scheme": cfg.scheme, "seed": cfg.seed},
                        "status": f"eval_failed:{e.__class__.__name__}",
                        "trained": None,
                        "placebo_a": None,
                        "placebo_b": None,
                        "cross_domain": None,
                        "latency_ms": None,
                    }
                by_key[cfg.key()] = rec
                done += 1
                # Write checkpoint atomically
                tmp = out_path.with_suffix(".jsonl.tmp")
                with tmp.open("w") as f:
                    for r in sorted(by_key.values(), key=lambda r: (r["config"]["domain"], r["config"]["rank"], r["config"]["scheme"], r["config"]["seed"])):
                        f.write(json.dumps(r) + "\n")
                tmp.replace(out_path)
                print(f"[{done}/{total}] wrote {cfg.dir.name}")
                # small GC between configs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        finally:
            # Release tokenizer and encodings
            del tok, encs_by_domain
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f"Done. Wrote {len(by_key)} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
