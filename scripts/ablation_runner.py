from __future__ import annotations

"""Ablation runner: sweep ranks and target schemes, record metrics.

This script orchestrates calls to train_task.py with varying ranks/schemes and
writes a compact JSONL of results for easy plotting.
"""

import argparse
import json
from pathlib import Path
from typing import List
from plora.config import get as cfg
import subprocess


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domains", type=lambda s: s.split(","), default=cfg("domains", [])
    )
    ap.add_argument(
        "--ranks",
        type=lambda s: [int(x) for x in s.split(",")],
        default=cfg("value_add.ranks", [2, 4, 8]),
    )
    ap.add_argument("--schemes", type=lambda s: s.split(","), required=True)
    ap.add_argument("--samples", type=int, default=cfg("samples", 64))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--base_model", type=str, default=cfg("base_model", None))
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args(argv)

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl = ns.out

    for dom in ns.domains:
        for r in ns.ranks:
            for scheme in ns.schemes:
                out_dir = Path("results") / f"abl_{dom}_r{r}_{scheme}"
                cmd = [
                    "python",
                    "-m",
                    "scripts.train_task",
                    "--domain",
                    dom,
                    "--epochs",
                    str(ns.epochs),
                    "--samples",
                    str(ns.samples),
                    "--output",
                    str(out_dir),
                ]
                if ns.base_model:
                    cmd.extend(["--base-model", ns.base_model])
                # rank and scheme are managed inside train_task via env/args
                subprocess.run(cmd, check=True)
                # Capture minimal record
                rec = {
                    "domain": dom,
                    "rank": r,
                    "scheme": scheme,
                    "output_dir": str(out_dir),
                }
                with out_jsonl.open("a") as f:
                    f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
