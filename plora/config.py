from __future__ import annotations

"""Central configuration loader for Plora (YAML-based).

Search order:
1) config/plora.yml at repository root
2) Fallback defaults embedded below

Access helpers:
- load() -> dict
- get("value_add.dev_size", default)

CLI:
  python -m plora.config                # print full JSON config
  python -m plora.config base_model     # print a single value
  python -m plora.config value_add.dev_size
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:
    yaml = None


_DEFAULTS: Dict[str, Any] = {
    "base_model": "google/gemma-3-1b-it",
    "eval_split": "validation",
    "samples": 64,
    "latency_budget_ms": 5000,
    "domains": ["arithmetic", "legal", "medical"],
    "allowed_ranks": [1, 4, 8, 16],
    "allowed_targets": "attention",
    "graph": {"p": 0.25, "ws_k": 4, "ws_beta": 0.2, "ba_m": 2},
    # Value-add experiment grid defaults
    "value_add": {
        "dev_size": 256,
        "ranks": [4, 8, 16],
        "schemes": ["all"],
        "seeds": [42],
        "placebo_b_rank": 8,  # rank used for label-shuffle placebo
    },
    # Training hyperparameters (centralised for reproducibility)
    "train": {
        "lr": 2e-4,
        "dropout": 0.1,
        "max_len": 128,
        "train_split": 0.8,  # fraction of samples used for training (rest is dev)
        "seed": 42,
    },
    # Probe calibration desired error targets
    "probes": {"target_fp": 0.05, "target_fn": 0.1},
    # Swarm / consensus defaults
    "swarm": {"trojan_rate": 0.3, "quorum": 2},
    # Gate threshold defaults (can be overridden in YAML)
    "gate": {"tau_trigger": 0.2, "tau_clean_delta": -0.05, "tau_tensor_z": 5.0},
    # Alternating train-merge defaults
    "alt_train_merge": {"cycles": 3, "samples": 32, "rank": 4},
}


@lru_cache(maxsize=1)
def load() -> Dict[str, Any]:
    override = Path("config/plora.override.yml")
    if override.exists():
        path = override
    else:
        path = Path("config/plora.yml")
    if yaml is None or not path.exists():
        return dict(_DEFAULTS)
    try:
        data = yaml.safe_load(path.read_text()) or {}
        # shallow merge defaults -> data (data wins); nested for value_add/graph/train/probes/swarm/gate
        out = dict(_DEFAULTS)
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                nv = dict(out[k])
                nv.update(v)
                out[k] = nv
            else:
                out[k] = v
        return out
    except Exception:
        return dict(_DEFAULTS)


def get(path: str, default: Any | None = None) -> Any:
    """Get a config value by dot-path, with optional array indexing.
    
    Examples:
        get("value_add.ranks")      -> [1, 4, 8]
        get("value_add.ranks[0]")   -> 1
        get("domains[1]")           -> "legal"
    """
    import re
    parts = path.split(".")
    cur: Any = load()
    for p in parts:
        # Check for array indexing like "ranks[0]"
        match = re.match(r"^(\w+)\[(\d+)\]$", p)
        if match:
            key, idx = match.group(1), int(match.group(2))
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
            if not isinstance(cur, list) or idx >= len(cur):
                return default
            cur = cur[idx]
        else:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
    return cur


def _main() -> None:
    import sys

    if len(sys.argv) == 1:
        print(json.dumps(load(), indent=2))
        return
    if sys.argv[1] == "use":
        # Switch active config by writing config/plora.override.yml
        if len(sys.argv) < 3:
            print("Usage: python -m plora.config use <path-to-yml>")
            return
        p = Path(sys.argv[2])
        if not p.exists():
            print("Config file not found:", p)
            return
        dst = Path("config/plora.override.yml")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(p.read_text())
        # clear cache
        load.cache_clear()
        print("Switched config to", p)
        return
    key = sys.argv[1]
    val = get(key)
    if isinstance(val, (dict, list)):
        print(json.dumps(val))
    elif val is None:
        print("")
    else:
        print(val)


if __name__ == "__main__":
    _main()
