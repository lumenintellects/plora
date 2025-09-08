"""plora.targets - standalone helper to pick LoRA target modules.
This duplicates logic used in experiments.plasmid_swarm but without pulling in
heavy optional dependencies, avoiding import errors for smoke runs.
"""
from typing import List, Set, Tuple

import torch.nn as nn
from transformers.pytorch_utils import Conv1D


def select_target_modules(model: nn.Module, scheme: str) -> List[str]:
    scheme = scheme.lower()
    if scheme not in {"attention", "mlp", "all"}:
        raise ValueError("scheme must be one of attention|mlp|all")

    attn_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"]
    mlp_suffixes = [
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_fc",
        "c_proj",
        "fc_in",
        "fc_out",
    ]

    wanted = {
        "attention": attn_suffixes,
        "mlp": mlp_suffixes,
        "all": attn_suffixes + mlp_suffixes,
    }[scheme]

    found: Set[str] = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Linear, Conv1D)):
            continue
        for suff in wanted:
            if name.endswith(suff):
                found.add(suff)

    return sorted(found)
