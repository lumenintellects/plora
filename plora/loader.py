from __future__ import annotations

"""plora.loader - injection context manager and LoRA merging helpers."""

import contextlib
import copy
import logging
import time
from pathlib import Path
from typing import Iterator, List, Sequence

import torch
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from .compat import device_dtype

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

    # Sequential merging: fold each adapter into the base model in turn.
    for dir_ in plasmid_dirs:
        tmp = PeftModel.from_pretrained(model, str(dir_), is_trainable=False)
        model = tmp.merge_and_unload()

    if not commit_inplace:
        return model
    # Otherwise detach and return plain model
    return model


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
