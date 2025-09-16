from __future__ import annotations

"""plora.loader - injection context manager and LoRA merging helpers."""

import contextlib
import logging
import time
from pathlib import Path
from typing import Iterator, List, Sequence, Callable, Dict, Optional

import torch
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from .compat import device_dtype
from .weights import weight_norms_from_safetensors

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context manager, inject & restore
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def inject(model: PreTrainedModel, adapter_dir: Path) -> Iterator[PeftModel]:
    """Temporarily load a LoRA adapter into *model*.

    Example::
        with inject(base_model, Path("adapter_dir")) as peft:
            out = peft.generate(**inputs)
    """
    # Cache pristine weights on CPU to reduce device memory pressure
    pristine_state = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }

    t0 = time.perf_counter()
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    log.debug("Adapter injected in %.3f ms", (time.perf_counter() - t0) * 1e3)
    try:
        yield peft_model
    finally:
        # Robust restore: copy parameter data where shapes match
        with torch.no_grad():
            current_sd = model.state_dict()
            for k, v in pristine_state.items():
                tgt = current_sd.get(k)
                if tgt is not None and tgt.shape == v.shape:
                    tgt.copy_(v)
        del peft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        log.debug("Adapter removed; model restored.")


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge_plasmids(
    base_model_name: str,
    plasmid_dirs: Sequence[Path],
    weights: Sequence[float] | None = None,
    strategy: str = "weighted_sum",
    commit_inplace: bool = False,
    reproject_rank: int | None = None,
    fisher_weighted: bool = False,
    max_delta_fro: float | None = None,
    global_scale: float | None = None,
    *,
    module_caps: Optional[Dict[str, float]] = None,
    line_search_objective: Optional[Callable[[float], float]] = None,
    ls_dataset: Optional[List[tuple[str, str]]] = None,
    ls_tokenizer_name: Optional[str] = None,
) -> PreTrainedModel:
    """Merge multiple LoRA adapters into a single model.

    Parameters
    ----------
    base_model_name : str
        HF model name or path (e.g. ``"sshleifer/tiny-gpt2"``).
    plasmid_dirs : list[Path]
        Directories each containing adapter_model files.
    weights : list[float] | None
        Optional scaling factors for ``weighted_sum`` strategy.  Defaults to
        equal weighting.
    strategy : {"weighted_sum", "sequential"}
    commit_inplace : bool
        If *True* the LoRA weights are merged back into the base *and* PEFT
        wrappers are removed, yielding a plain `PreTrainedModel`.
    """
    device, dtype = device_dtype()

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map={"": device},
    )

    if strategy not in {"sequential", "weighted_sum"}:
        raise ValueError(f"Unknown merge strategy: {strategy}")

    if strategy == "sequential":
        # Fold each adapter into the base model in turn (equal weight 1).
        for dir_ in plasmid_dirs:
            tmp = PeftModel.from_pretrained(model, str(dir_), is_trainable=False)
            model = tmp.merge_and_unload()
    else:
        # Weighted sum of LoRA deltas relative to the pristine base weights.
        if not plasmid_dirs:
            return model
        if weights is None:
            # Optionally derive weights from Fisher diagonals
            if fisher_weighted:
                fisher_vals: List[float] = []
                for d in plasmid_dirs:
                    # Use built-in Fisher diag calculator fallback if no file exists
                    fv = _load_fisher_scalar(d)
                    if fv is None or not (fv > 0):
                        # compute proxy from safetensors (sum of LoRA tensor norms)
                        try:
                            fro_sum, _ = weight_norms_from_safetensors(d)
                            fv = float(fro_sum)
                        except Exception:
                            fv = None
                    if fv is None or not (fv > 0):
                        fisher_vals = []
                        break
                    fisher_vals.append(float(fv))
                if fisher_vals:
                    s = sum(fisher_vals)
                    wts = [fv / s for fv in fisher_vals]
                else:
                    wts = [1.0 / len(plasmid_dirs)] * len(plasmid_dirs)
            else:
                wts = [1.0 / len(plasmid_dirs)] * len(plasmid_dirs)
        else:
            if len(weights) != len(plasmid_dirs):
                raise ValueError("weights length must match plasmid_dirs length")
            wts = list(weights)

        # Snapshot pristine base weights
        with torch.no_grad():
            base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # Initialise accumulator for deltas
        sum_delta = {k: torch.zeros_like(v) for k, v in base_sd.items()}

        for dir_, w in zip(plasmid_dirs, wts):
            # Restore base weights before applying each adapter
            with torch.no_grad():
                current_sd = model.state_dict()
                for k, v_base in base_sd.items():
                    tgt = current_sd.get(k)
                    if tgt is not None and tgt.shape == v_base.shape:
                        tgt.copy_(v_base)
            tmp = PeftModel.from_pretrained(model, str(dir_), is_trainable=False)
            model = tmp.merge_and_unload()
            # Accumulate weighted delta relative to base
            with torch.no_grad():
                current_sd = model.state_dict()
                for k, v_base in base_sd.items():
                    v_after = current_sd.get(k)
                    if v_after is not None and v_after.shape == v_base.shape:
                        sum_delta[k].add_(w * (v_after - v_base))

        # Apply accumulated deltas to base
        # Optional trust-region scaling by global Frobenius norm
        if max_delta_fro is not None and max_delta_fro > 0:
            with torch.no_grad():
                total_sq = 0.0
                for v in sum_delta.values():
                    total_sq += float((v.to(torch.float32) ** 2).sum().item())
                if total_sq > 0.0:
                    total_norm = total_sq**0.5
                    if total_norm > max_delta_fro:
                        scale = max_delta_fro / total_norm
                        for k in sum_delta.keys():
                            sum_delta[k].mul_(scale)

        # Optional per-module trust-region caps (cap Frobenius norm per module key)
        if module_caps:
            with torch.no_grad():
                for key, cap in module_caps.items():
                    # accumulate norm over matching tensors
                    matched = [k for k in sum_delta.keys() if key in k]
                    if not matched:
                        continue
                    sq = 0.0
                    for k in matched:
                        sq += float((sum_delta[k].to(torch.float32) ** 2).sum().item())
                    norm = sq**0.5
                    if norm > 0 and norm > float(cap):
                        scale = float(cap) / norm
                        for k in matched:
                            sum_delta[k].mul_(scale)

        # Optional global scaling (line-search override)
        if global_scale is not None:
            with torch.no_grad():
                for k in sum_delta.keys():
                    sum_delta[k].mul_(float(global_scale))

        # Optional backtracking line-search using objective(scale). If none provided,
        # but a small dataset is given, construct a default objective based on token NLL.
        if (
            line_search_objective is None
            and ls_dataset is not None
            and global_scale is None
        ):
            try:
                from .metrics import token_nlls
            except Exception:
                token_nlls = None  # type: ignore

            if token_nlls is not None:
                tok_name = ls_tokenizer_name or base_model_name
                tok = AutoTokenizer.from_pretrained(tok_name)
                # Capture baseline state to restore between evaluations
                current_sd = model.state_dict()
                base_state = {k: v.detach().clone() for k, v in current_sd.items()}

                def _nll_objective(scale: float) -> float:
                    with torch.no_grad():
                        # apply scaled deltas to a temporary copy of weights
                        for k, v_base in base_state.items():
                            tgt = current_sd.get(k)
                            if tgt is None or tgt.shape != v_base.shape:
                                continue
                            delta = sum_delta[k]
                            tgt.copy_(v_base + delta * float(scale))
                    # compute mean NLL over tiny dataset
                    nll_list = token_nlls(model, tok, list(ls_dataset))  # type: ignore[arg-type]
                    val = float(sum(nll_list) / max(1, len(nll_list)))
                    return val

                line_search_objective = _nll_objective

        # Optional backtracking line-search using objective(scale)
        if line_search_objective is not None and global_scale is None:
            # evaluate at scales 1, 0.5, 0.25, ... until improvement
            best_scale = 1.0
            best_val = line_search_objective(1.0)
            scale = 0.5
            tried = 0
            while tried < 5:
                val = line_search_objective(scale)
                if val <= best_val:
                    best_val = val
                    best_scale = scale
                else:
                    # Armijo-like: stop when no improvement
                    break
                scale *= 0.5
                tried += 1
            with torch.no_grad():
                for k in sum_delta.keys():
                    sum_delta[k].mul_(best_scale)

        # Apply accumulated deltas to base
        with torch.no_grad():
            current_sd = model.state_dict()
            for k, v_base in base_sd.items():
                tgt = current_sd.get(k)
                if tgt is not None and tgt.shape == v_base.shape:
                    tgt.copy_(v_base + sum_delta[k])

    # Optional: project deltas to best rank-k for 2D tensors
    if reproject_rank is not None and reproject_rank > 0:
        with torch.no_grad():
            # We need base weights to form deltas. If not available (sequential),
            # treat current as base + delta and project current itself.
            base_for_proj = None
            try:
                base_for_proj = base_sd  # defined in weighted_sum path
            except NameError:
                pass
            current_sd = model.state_dict()
            for k, v in current_sd.items():
                if v.ndim == 2:
                    if (
                        base_for_proj is not None
                        and k in base_for_proj
                        and base_for_proj[k].shape == v.shape
                    ):
                        base_v = base_for_proj[k]
                    else:
                        base_v = torch.zeros_like(v)
                    delta = (v - base_v).detach().to(torch.float32, copy=True)
                    # Move to CPU for SVD if necessary
                    delta_cpu = delta.cpu()
                    try:
                        U, S, Vh = torch.linalg.svd(delta_cpu, full_matrices=False)
                    except RuntimeError:
                        # Fallback: skip projection on failure
                        continue
                    r = min(reproject_rank, S.shape[0])
                    if r <= 0:
                        continue
                    Ur = U[:, :r]
                    Sr = S[:r]
                    Vhr = Vh[:r, :]
                    delta_k = (Ur * Sr) @ Vhr
                    v.copy_((base_v + delta_k.to(v.device)).to(v.dtype))

    # Note: commit_inplace retained for API compatibility but currently a no-op.
    # The returned model already has deltas applied; callers can save or continue using it directly.
    return model


def _load_fisher_scalar(adapter_dir: Path) -> float | None:
    """Try to load a scalar Fisher signal for weighting from adapter_dir.

    Supports:
    - JSON file "fisher_diag.json" with either {"sum": float} or {"param": value,...}
    - safetensors file "fisher_diag.safetensors" with tensor "diag" or any tensors summed.
    Returns None if nothing usable found.
    """
    # JSON variant
    try:
        j = adapter_dir / "fisher_diag.json"
        if j.exists():
            import json

            obj = json.loads(j.read_text())
            if isinstance(obj, dict):
                if "sum" in obj and isinstance(obj["sum"], (int, float)):
                    return float(obj["sum"])
                # sum numeric values
                vals = [float(v) for v in obj.values() if isinstance(v, (int, float))]
                if vals:
                    return float(sum(vals))
    except Exception:
        pass

    # safetensors variant
    try:
        from safetensors.torch import load_file  # type: ignore
        import torch as _t

        st = adapter_dir / "fisher_diag.safetensors"
        if st.exists():
            tensors = load_file(str(st))
            if "diag" in tensors:
                t = tensors["diag"]
                if isinstance(t, _t.Tensor):
                    return float(_t.sum(_t.abs(t)).item())
            # otherwise sum magnitudes of all tensors
            total = 0.0
            for t in tensors.values():
                if isinstance(t, _t.Tensor):
                    total += float(_t.sum(_t.abs(t)).item())
            return total if total > 0 else None
    except Exception:
        pass
    return None

    # (Deprecated path removed) – commit_inplace is a no-op; function always returns merged model above.


# ---------------------------------------------------------------------------
# Placebo LoRA generator, random weights (for control experiments)
# ---------------------------------------------------------------------------


def random_lora(
    model: PreTrainedModel,
    output_dir: Path,
    *,
    r: int | None = None,
    target_modules: List[str] | None = None,
    like_adapter_dir: Path | None = None,
) -> Path:
    """Create a *random* LoRA adapter compatible with *model* at *output_dir*.

    Parameters
    ----------
    model : PreTrainedModel
        Base model instance to which the adapter will later be applied.
    output_dir : Path
        Directory where the adapter weights & config will be written.  Will be
        created if it does not exist.
    r : int, default 8
        LoRA rank.
    target_modules : list[str] | None
        Which sub-modules to target.  If *None* we try to inspect *model* and
        fall back to *like_adapter_dir*'s config, then to the GPT-style default
        of ["q_proj", "k_proj", "v_proj", "o_proj"].
    like_adapter_dir : Path | None
        Optional existing adapter directory; if provided we replicate its LoRA
        configuration (except the weights, which are random).

    Returns
    -------
    Path
        The *output_dir* where files were written.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Determine LoRA configuration
    # ---------------------------------------------------------------------
    if like_adapter_dir and (like_adapter_dir / "adapter_config.json").exists():
        import json

        cfg = json.loads((like_adapter_dir / "adapter_config.json").read_text())
        r = cfg.get("r", r)
        target_modules = cfg.get("target_modules", target_modules)
        alpha = cfg.get("lora_alpha", r * 2)
        dropout = cfg.get("lora_dropout", 0.0)
    else:
        if r is None:
            r = 2  # minimal default
        alpha = r * 2
        dropout = 0.0

    # Fallback discovery of target modules if still None
    if target_modules is None:
        cand = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"]
        found = {
            suffix
            for name, _ in model.named_modules()
            for suffix in cand
            if name.endswith(suffix)
        }
        target_modules = sorted(found) if found else ["c_attn"]

    l_cfg = LoraConfig(
        r=r, lora_alpha=alpha, target_modules=target_modules, lora_dropout=dropout
    )

    # Create a trainable PEFT wrapper to easily write weights later
    peft_model = get_peft_model(model, l_cfg)

    # ---------------------------------------------------------------------
    # Randomise LoRA weights, use very small scale (1e-4) so placebo is inert
    # ---------------------------------------------------------------------
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                p.copy_(torch.randn_like(p) * 1e-4)

    peft_model.save_pretrained(output_dir, safe_serialization=True)

    # Clean up memory, we created extra weights on model's device
    del peft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_dir
