"""plora.targets - standalone helper to pick LoRA target modules.
This duplicates logic used in experiments.plasmid_swarm but without pulling in
heavy optional dependencies, avoiding import errors for smoke runs.
"""

from typing import List, Set

import torch.nn as nn
from transformers.pytorch_utils import Conv1D


# Export common suffix lists for policy whitelists
ATTENTION_SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"]
MLP_SUFFIXES = [
    "gate_proj",
    "up_proj",
    "down_proj",
    "c_fc",
    "c_proj",
    "fc_in",
    "fc_out",
]
ALL_SUFFIXES = sorted(set(ATTENTION_SUFFIXES + MLP_SUFFIXES))


def select_target_modules(model: nn.Module, scheme: str) -> List[str]:
    scheme = scheme.lower()
    if scheme not in {"attention", "mlp", "all"}:
        raise ValueError("scheme must be one of attention|mlp|all")

    wanted = {
        "attention": ATTENTION_SUFFIXES,
        "mlp": MLP_SUFFIXES,
        "all": ATTENTION_SUFFIXES + MLP_SUFFIXES,
    }[scheme]

    found: Set[str] = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Linear, Conv1D)):
            continue
        for suff in wanted:
            if name.endswith(suff):
                found.add(suff)

    return sorted(found)
