"""
Plora – prototype library for signed LoRA plasmids.
"""

# Suppress noisy third-party warnings in CI output (does not affect behavior)
import warnings as _warnings

# PEFT LoRA Conv1D fan_in_fan_out warning is informational; suppress it in tests
_warnings.filterwarnings(
    "ignore",
    message=r".*fan_in_fan_out is set to False but the target module is `Conv1D`.*",
    category=UserWarning,
)

__all__ = [
    "manifest",
    "signer",
    "loader",
    "metrics",
    "compat",
]
