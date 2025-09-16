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
| `PLORA_BASE_MODEL` | HF model used for training / metrics | `google/gemma-3-1b-it` |
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

## Experiment workflows (config-driven)

This repo provides two orchestrated Make targets that exercise the entire stack. 
Both workflows are driven by YAML configs under `config/` and use an override file
`config/plora.override.yml` selected by helper targets.

### Configs

- `config/plora.dry.yml` – fast settings for quick validation (CPU/small GPU)
- `config/plora.full.yml` – full-scale defaults for a thesis run (single GPU)

Switch config (done automatically by the Make targets, or manually):

```bash
make config-use-dry   # selects config/plora.dry.yml
make config-use-full  # selects config/plora.full.yml
```

Key fields (see the YAML comments for full explanations):
- `base_model`, `eval_split`, `samples`, `latency_budget_ms`
- `domains`, `allowed_ranks`, `allowed_targets`, `graph.{p,ws_k,ws_beta,ba_m}`
- `value_add.{dev_size,ranks,schemes,seeds}`

### Minimal dry run (fast)

```bash
make dry-run-lite
```

What it does (fast versions of each):
1) Unit tests
2) Probe calibration → `results/probes_calib.json`
3) Calibrate C (tiny ER) → `results/c_calib_er_lite.json`
4) Validate bounds (tiny) → `results/bounds_validation_lite.json`
5) Train per-domain (tiny) → `out/<domain>/`
6) Sign adapters (temp keys) → manifests updated with signatures
7) Swarm v2 smoke (security on) → v2 reports in `results/`
8) Summarise v2 → `results/summary_v2.json`
9) Monolithic baseline → `out/monolithic_r4/`
10) Value-add rank sweep (small) → artifacts under `results/value_add/`
11) Alternating train–merge (one cycle) → `results/alt_train_merge_lite/`
12) Value-add JSONL build (lowmem) → `results/value_add/value_add.jsonl`
13) Consensus-enabled v2 smoke (short)
14) gRPC demo (offer/fetch with signature verification) → `fetched/`
15) Dump effective security policy (printed JSON)
16) Net IT metrics (tiny history) → `results/net_it_metrics.json`

Artifacts to inspect:
- Swarm v2 raw: `results/swarm_v2_report_*.json`
- Swarm v2 summary: `results/summary_v2.json`
- Value-add: `results/value_add/value_add.jsonl` (and domain outputs)
- Monolithic: `out/monolithic_r4/`
- Calibrations: `results/c_calib_er_lite.json`, `results/bounds_validation_lite.json`
- IT metrics: `results/net_it_metrics.json`

### Full experiment (thesis-grade)

```bash
make full-experiment
```

Sequence (config-driven):
1) Unit tests
2) Calibrate spectral constant C (ER) → `results/c_calib_er.json`
3) Validate spectral/Cheeger bounds → `results/bounds_validation.json`
4) Probe calibration → `results/probes_calib.json`
5) Train per-domain adapters → `out/<domain>/`
6) Sign adapters → manifests updated with signatures
7) Swarm v2 simulation (security on) → `results/swarm_v2_report_*.json`
8) Summarise v2 → `results/summary_v2.json`
9) Monolithic baseline → `out/monolithic_r4/`
10) Value-add rank sweep (small) → `results/value_add/`
11) Thesis sweep (ER/WS/BA) → `results/thesis_sweep.jsonl`
12) Figures → `results/figures/`
13) Ablations (rank/scheme) → `results/abl_*.jsonl`
14) Alternating train–merge → `results/alt_train_merge/`
15) Value-add JSONL build (lowmem) → `results/value_add/value_add.jsonl`
16) Consensus-enabled v2 smoke (short consensus demo)
17) gRPC offer/fetch demo (signed payload)
18) Dump effective security policy (printed JSON)
19) Net IT metrics → `results/net_it_metrics.json`

Hardware notes:
- `config/plora.full.yml` defaults are sized for a single small GPU (e.g., 1000 samples/domain, dev_size 1024).
- Increase/decrease `samples` and `value_add.dev_size` to fit your budget.

### Artefacts check

After either workflow, run:

```bash
make artefacts-check
```

This verifies the presence of key outputs:
- `results/summary_v2.json`, `results/figures/`, `results/value_add/value_add.jsonl`
- `results/thesis_sweep.jsonl`, `results/c_calib_er.json`, `results/bounds_validation.json`
- `results/net_it_metrics.json`, `out/monolithic_r4/`, at least one `results/swarm_v2_report_*.json`
- Per-domain adapter artifacts: `out/<domain>/plora.yml` and `adapter_model.safetensors`

### Reproducibility and policy

- Configuration is centralized in YAML (see `config/`), and the active config is
  copied to `config/plora.override.yml` by the helper targets.
- Dump the effective security policy before a run:

```bash
make dump-policy POLICY=policy.json TARGETS=assets/allowed_targets.txt RANKS=4,8,16 SIG=off
```

### Troubleshooting

- If value-add is memory-constrained, use `value-add-build-lowmem` and consider `--skip-placebos`/`--skip-cross`.
- If the gRPC demo fails signature verification, ensure you passed a private key to `offer_server` and use the matching public key in `fetch_client`.
- If `artefacts-check` fails, re-run the missing steps or consult the logs under `results/`.

## Thesis-grade extensions (math + security + reporting)

- Notebook: `notebooks/math_foundations.ipynb` – full LaTeX-ready derivations (LoRA merge bounds, info theory, graph diffusion, optimization, security). Runs end-to-end on CPU with toy figures.
- Information-theoretic estimators:
  - kNN MI (KSG): `plora/it_estimators.py` with tests `tests/test_mi_knn.py`.
  - MINE MI: `plora/mine.py`, calibration script `scripts/mine_calibrate.py`, tests `tests/test_mine_gaussian.py`.
  - Transfer Entropy: `plora/te.py`, tests `tests/test_te_directionality.py`.
- Graph theory and diffusion:
  - Spectral gap and predictors: `swarm/metrics.py`, `swarm/theory.py`.
  - ER/WS/BA + temporal/weighted graphs: `swarm/graph_v2.py`.
  - Calibration scripts: `scripts/validate_bounds.py`, `scripts/calibrate_c.py`.
- Optimization and merging:
  - Weighted merge + SVD reprojection + trust-region + line search: `plora/loader.py`.
  - Alternating train–merge runner: `scripts/alternating_train_merge.py`.
- Security and consensus:
  - Policy gate with quorum, peer attestations, probes, anomaly checks: `plora/gate.py`, `plora/weights.py`.
  - Threshold multi-sig (RSA aggregate): `plora/threshold_sigs.py` with tests.
  - Consensus (proposal/vote/commit): `swarm/consensus.py`, tests.
  - Audit chain verifier: `scripts/audit_verify.py`.
  - TLS for gRPC server/client: `plora/grpc/server.py`, `plora/grpc/client.py` with CLI flags.
- Reporting:
  - `scripts/evaluate_v2.py` now summarizes spectral gap, MI deltas + CIs, TE pair; `scripts/plot_figures.py` generates figures.

### Helpful Make targets

- `make swarm-v2-smoke && make swarm-v2-eval && make figures` – quick end-to-end run and plots
- `make swarm-v2-eval` – summarise v2 reports into `results/summary_v2.json`.
- `make net-it` – compute network MI/TE metrics with CIs and BH-FDR over a saved history.
- `make calibrate-c` – calibrate diffusion constant C across sizes.
- `make validate-bounds` – validate spectral/cheeger bounds and optionally plot.
- `make math-export` – export `notebooks/math_foundations.ipynb` to PDF via nbconvert/pandoc (optional deps).
- `make mine-calib` – MINE estimator calibration on Gaussian
- `make consensus-smoke` and `make audit-verify` – consensus and audit chain
- `make probes-calib` – write probe thresholds to `results/probes_calib.json`
