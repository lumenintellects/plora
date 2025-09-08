# Plora LoRA Swarm – Prototype v0

This repository contains a reference implementation of the “Plasmid LoRA Swarm” idea, LoRA adapters (plasmids) that are signed, shared, and merged between agents.

## Quickstart (local)

```bash
# 1. Create virtualenv & install deps
make setup   # or: python -m venv .venv && source .venv/bin/activate && pip install -e .[dev]

# 2. Run tests (< 10 min on laptop)
make test

# 3. Train a tiny ‘legal’ plasmid
make train-legal

# 4. Sign it (uses temp RSA key)
make sign-legal

# 5. Start gRPC offer server
make offer &

# 6. Fetch and verify from another terminal
make fetch
```

## Repository layout
See `plora-swarm/` tree in project description.  Core modules:

* `plora.manifest` – strict YAML schema & I/O
* `plora.signer`   – RSA-PSS SHA-256 signing utilities
* `plora.loader`   – fast LoRA inject / merge helpers
* `plora.grpc`     – proto, server, client
* `scripts/`       – CLI wrappers for daily tasks
* `tests/`         – unit tests executed on CI

## Environment variables
| Variable | Purpose | Default |
| -------- | ------- | ------- |
| `PLORA_BASE_MODEL` | HF model used for training / metrics | `sshleifer/tiny-gpt2` |
| `PLORA_LATENCY_BUDGET_MS` | Loader latency budget for CI | `250` |

## CI
GitHub Actions workflow in `.github/workflows/ci.yml` installs CPU wheels and executes the full pytest suite in under 10 minutes.

## Value-add experiment

Once you have trained some domain adapters you can run the **value-add experiment** that
benchmarks trained LoRA vs base model and placebo controls, computes paired
statistics and produces a Markdown report.

Example:

```bash
python -m scripts.run_lora_value_add \
  --domains arithmetic,science,legal \
  --ranks 4,8 \
  --schemes all \
  --seeds 41,42,43 \
  --samples 128 \
  --dev-size 256
```

The script writes two artefacts in `results/`:

* `value_add.jsonl` – one JSON record per (domain, cell, seed) with detailed
  metrics, statistics, latency and cross-domain deltas.
* `value_add.md` – human-friendly tables with **bold** highlights for cells that
  meet the acceptance criteria.

Guardrails: the run aborts if any placebo significantly beats the baseline or
if inject+remove latency exceeds `$PLORA_LATENCY_BUDGET_MS` (default 250 ms).
