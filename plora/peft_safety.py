from __future__ import annotations
"""Helpers to assert adapter hygiene (avoid accidental multi-adapter stacking).

We treat any presence of a PEFT wrapper on a model that is supposed to act as a
"baseline" (no adapters) as a configuration / programming error.  This helper
is intentionally strict for early detection; callers can opt-in to looser checks
later if multi-adapter baselines become a supported scenario.
"""
from typing import Any

def assert_pristine_base(model: Any, where: str = "") -> None:
    """Raise AssertionError if *model* already carries PEFT adapter config.

    Parameters
    ----------
    model : Any
        HF model instance expected to be a plain (non-PEFT) base.
    where : str
        Context string to improve error messages.
    """
    if hasattr(model, "peft_config"):
        # peft_config is typically a dict mapping adapter names -> config
        try:
            cfg = getattr(model, "peft_config")
            if isinstance(cfg, dict) and len(cfg) > 0:
                raise AssertionError(
                    f"Baseline model not pristine (found {len(cfg)} adapter(s)) in {where}. "
                    "Likely accidental stacking of LoRA wrappers."
                )
            else:
                # Even if empty dict, conservative: raise to prompt review
                raise AssertionError(
                    f"Baseline model carries peft_config attribute in {where} (empty or unknown); expected none."
                )
        except AssertionError:
            raise
        except Exception:
            raise AssertionError(
                f"Baseline model has unexpected PEFT state in {where}; aborting."
            )

