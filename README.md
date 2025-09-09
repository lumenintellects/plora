## Swarm Sim v2 + Security (push–pull, in-process)

Swarm Sim v2 provides an in-process push–pull gossip over an Erdős–Rényi overlay, optional security gate, and reporting.

Run a small dry-run on CPU:

```
python -m swarm_v2 --agents 6 --rounds 5 --graph_p 0.25 --security on --trojan_rate 0.3
```

Flags:
- `--graph_p`: ER edge probability (default 0.25)
- `--security on|off`: enable policy gate (base model, rank, targets, SHA, optional signatures)
- `--allowed_targets attention|all`: whitelist for LoRA target modules (defaults to attention)
- `--allowed_ranks 4,8,16`: allowed ranks for policy gate
- `--signatures on|off` and `--trusted_pubkeys path1,path2`: enable RSA signature verification on artifact SHA-256
- `--trojan_rate 0..1`: fraction of agents initialised with a marked trojan adapter (for FN/FP evaluation)

Report fields (unified):
- `final.bytes_on_wire`, `final.accepted_offers`
- `final.coverage` per domain
- `final.gate`: totals for accepted/rejected clean/trojan, `false_negatives`, `false_positives`, and `rejection_reasons`

Thresholds and policy tuning:
- Configure via `--policy_file` (JSON) and override `--tau_trigger`, `--tau_norm_z`, `--tau_clean_delta`.
- Dump effective policy: `python -m scripts.dump_policy --policy_file policy.json --allowed_targets_file assets/allowed_targets.txt`.

Reports are written to `results/` by the v1 socket engine; for v2 in-process runs, use the evaluate helper to summarise any v1 graph reports you generated:

```
python -m scripts.evaluate_v2 --reports results --out results/summary_v2.json
```

Monolithic baseline (tiny training loop):

```
python -m scripts.monolithic_train --domains arithmetic,legal,medical --epochs 1 --samples 64 --rank 4 --output out/monolithic_r4
```

Dry-run dataset caps:
- Per-invocation: pass `--samples 64` to training scripts
- Global: set `PLORA_SAMPLES=64` to cap HuggingFace dataset sampling

# Plora LoRA Swarm – Prototype v0

This repository contains a **CPU-friendly** reference implementation of the “Plasmid LoRA Swarm” idea – LoRA adapters (plasmids) that are signed, shared, and merged between agents.

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
make offer &        # background

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

Example (CI-sized run, ~5 min on CPU):

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
