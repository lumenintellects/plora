from __future__ import annotations

"""plora.grpc – run-time generation of gRPC code from *plora.proto*.

At import time we compile the proto using *grpc_tools.protoc* into a `_generated`
sub-package (ignored by version control).  This avoids committing generated
artifacts and works on CI without pre-steps.
"""

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

from grpc_tools import protoc

_THIS_DIR = Path(__file__).resolve().parent
_PROTO_FILE = _THIS_DIR / "plora.proto"
_GEN_DIR = _THIS_DIR / "_generated"


def _compile_proto():
    _GEN_DIR.mkdir(exist_ok=True)
    cmd = [
        "grpc_tools.protoc",
        f"-I{_THIS_DIR}",
        f"--python_out={_GEN_DIR}",
        f"--grpc_python_out={_GEN_DIR}",
        str(_PROTO_FILE),
    ]
    # Use protoc.main to run the command
    if protoc.main(cmd) != 0:
        raise RuntimeError("Failed to compile plora.proto")


# Compile if not already generated
if not (_GEN_DIR / "plora_pb2.py").exists():
    _compile_proto()

# Ensure generated package is on path and importable
if str(_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(_GEN_DIR))

plora_pb2: ModuleType = importlib.import_module("plora_pb2")  # type: ignore
plora_pb2_grpc: ModuleType = importlib.import_module("plora_pb2_grpc")  # type: ignore

__all__ = [
    "plora_pb2",
    "plora_pb2_grpc",
]
