# Plora - Plasmid LoRA Swarm

A research prototype implementing **decentralized LoRA adapter sharing** via gossip-based swarm networks, with cryptographic signing, security gates, and information-theoretic analysis.

## Overview

Plora explores the question: *Can lightweight LoRA adapters ("plasmids") diffuse through a peer-to-peer network while maintaining safety guarantees?*

The system implements:
- **Signed LoRA manifests** with SHA-256 artifact hashes and RSA-PSS/Ed25519 signatures
- **Security gates** with multi-layer verification (policy, quorum signatures, reputation, weight-space anomaly detection, behavioral probes)
- **Push-pull gossip** over configurable graph topologies (Erdős-Rényi, Watts-Strogatz, Barabási-Albert)
- **Spectral and information-theoretic analysis** of diffusion dynamics
- **Weighted merging** with trust-region constraints, SVD reprojection, and line search optimization

## Installation

### Option A: Poetry (native)

```bash
git clone <repo-url> && cd plora
make setup   # installs Poetry if missing, then all dependencies
make test
```

### Option B: Docker (any OS, no local Python required)

Install a Docker runtime (free, one-time):

| OS | Command |
|----|---------|
| **Linux** | `curl -fsSL https://get.docker.com \| sh` |
| **macOS** | `brew install colima docker && colima start --vm-type vz --memory 8 --cpu 4 --save-config` |
| **Windows** | `wsl --install`, then run the Linux command inside WSL |

> **Memory:** Training peaks at ~10-12 GB RSS (the model is loaded twice for
> perplexity measurement). Without enough RAM + swap the process is OOM-killed
> (exit 137). Platform-specific setup:
>
> | Platform | What to do |
> |----------|------------|
> | **macOS / Colima** | `colima start --vm-type vz --memory 8 --cpu 4 --save-config`, then `make docker-setup-swap` (adds 8 GB swap inside the VM; re-run after each `colima start`) |
> | **Docker Desktop** | Settings > Resources > set memory to at least 8 GB |
> | **Linux (native)** | Ensure the host has ≥ 8 GB swap (`free -h`). Most distros already do |
> | **Windows / WSL2** | Create `%USERPROFILE%\.wslconfig` with `[wsl2]`, `memory=10GB`, `swap=8GB`, then `wsl --shutdown` and reopen |

Then build and run:

```bash
make docker-build                               # build the image
make docker-setup-swap                          # macOS/Colima only: add 16 GB swap (once per colima start)
make docker-prefetch HF_TOKEN=hf_xxxxx          # download the base model into host cache (once)
make docker-dry-run  HF_TOKEN=hf_xxxxx          # full 19-step dry run (runs offline)
make docker-run      HF_TOKEN=hf_xxxxx          # interactive shell (mounts results/ and out/)
```

All `docker run` targets use **offline mode** (`TRANSFORMERS_OFFLINE=1`,
`HF_HUB_OFFLINE=1`) so the container never makes network requests after the
initial `docker-prefetch`. This prevents DNS-retry memory leaks inside the VM.
The host-side HF cache (`~/.cache/huggingface`) is mounted automatically. To use
a different cache directory:

```bash
make docker-dry-run HF_TOKEN=hf_xxxxx HF_CACHE=/path/to/cache
```

## Quick Start

```bash
# 1. Train a domain adapter (e.g., legal)
make train-legal

# 2. Sign it with an RSA key
make sign-legal

# 3. Start gRPC offer server (background)
make offer &

# 4. Fetch and verify from another terminal
make fetch
```

## Project Structure

```
plora/
├── plora/                  # Core library
│   ├── manifest.py         # Pydantic schema for plora.yml manifests
│   ├── signer.py           # RSA-PSS/Ed25519 signing utilities
│   ├── loader.py           # LoRA injection & merging (weighted, trust-region, SVD)
│   ├── gate.py             # Security gate (policy, quorum, probes, anomaly detection)
│   ├── agent.py            # Lightweight agent abstraction for simulations
│   ├── metrics.py          # PPL, EM, chrF, KL/JS divergence, ECE, Wilcoxon
│   ├── it_estimators.py    # KSG k-NN mutual information estimator
│   ├── mine.py             # MINE neural MI estimator
│   ├── te.py               # Transfer entropy (discrete histogram)
│   ├── weights.py          # Weight-space anomaly detection
│   ├── threshold_sigs.py   # Aggregate signature verification with quorum
│   ├── probes.py           # Behavioral probe scaffolding
│   └── grpc/               # gRPC offer/fetch protocol
│
├── swarm/                  # Swarm simulation engine
│   ├── swarm_v2.py         # Async push–pull gossip driver
│   ├── graph_v2.py         # ER/WS/BA graph overlays + temporal dropout
│   ├── metrics.py          # Coverage, MI, spectral gap, conductance, PID-lite
│   ├── theory.py           # Spectral predictions, Cheeger bounds, epidemic threshold
│   └── consensus.py        # Proposal/vote/commit consensus protocol
│
├── scripts/                # CLI tools
│   ├── train_task.py       # Per-domain adapter training
│   ├── sign_plasmid.py     # Sign adapter with private key
│   ├── run_lora_value_add.py   # Value-add experiments with statistical tests
│   ├── sweep/              # Thesis-scale topology sweeps
│   ├── evaluate_v2.py      # Summarize swarm reports
│   ├── plot_figures.py     # Generate publication figures
│   ├── calibrate_c.py      # Calibrate diffusion constant C
│   ├── validate_bounds.py  # Validate spectral/Cheeger bounds
│   └── ...
│
├── config/                 # YAML experiment configs
│   ├── plora.dry.yml       # Fast validation (CPU/small GPU)
│   └── plora.full.yml      # Thesis-grade experiments
│
├── tests/                  # 40+ unit tests
├── notebooks/              # Analysis & math foundations
└── assets/                 # Probe definitions, reputation maps
```

## Core Concepts

### Manifest Schema (`plora.yml`)

Each LoRA adapter is described by a signed manifest:

```yaml
schema_version: 0
plasmid_id: legal-r4-attention
domain: legal
base_model: google/gemma-3-1b-it
peft_format: lora

lora:
  r: 4
  alpha: 8
  dropout: 0.1
  target_modules: [q_proj, k_proj, v_proj, o_proj]

artifacts:
  filename: adapter_model.safetensors
  sha256: <64-char-hex>
  size_bytes: 8388608

metrics:
  val_ppl_before: 12.5
  val_ppl_after: 8.2
  delta_ppl: -4.3
  val_em: 0.72
  val_chrf: 0.81

safety:
  licence: MIT
  poisoning_score: 0.0

signer:
  algo: RSA-PSS-SHA256
  pubkey_fingerprint: <fingerprint>
  signature_b64: <base64-signature>
```

### Security Gate

Multi-layer verification before accepting an adapter:

| Layer | Description |
|-------|-------------|
| **Policy Check** | Base model match, allowed ranks/targets, artifact size bounds |
| **SHA-256 Verification** | Artifact hash matches manifest |
| **Signature Quorum** | K-of-N signatures from trusted keys (threshold mode) |
| **Reputation Gate** | Minimum peer reputation score |
| **Weight Anomaly** | Frobenius norm z-scores, tensor-level outlier detection |
| **Behavioral Probes** | Trigger compliance rate, clean accuracy regression |
| **Consensus** | Optional proposal/vote/commit protocol |

### Merging Strategies

```python
from plora.loader import merge_plasmids

model = merge_plasmids(
    "google/gemma-3-1b-it",
    [Path("out/legal"), Path("out/medical")],
    weights=[0.6, 0.4],           # Explicit weighting
    strategy="weighted_sum",       # or "sequential"
    fisher_weighted=True,          # Weight by Fisher information proxy
    max_delta_fro=10.0,           # Trust-region constraint
    reproject_rank=8,             # SVD reprojection to rank-k
    ls_dataset=[("q", "a"), ...], # Backtracking line search dataset
)
```

### Swarm Simulation

```bash
# Run push–pull gossip with security enabled
python -m swarm.sim_v2_entry \
    --agents 20 \
    --rounds 15 \
    --graph er \
    --graph_p 0.25 \
    --security on \
    --trojan_rate 0.3 \
    --consensus on \
    --quorum 2
```

Report outputs include:
- Per-domain coverage over time
- Mutual information series and deltas
- Spectral gap (λ₂) of overlay graph
- Gate statistics: accepted/rejected × clean/trojan
- Rejection reason breakdown

## Experiment Workflows

### Configuration

Two YAML configs control all experiments:

| Config | Purpose | Samples | Ranks | Dev Size |
|--------|---------|---------|-------|----------|
| `config/plora.dry.yml` | Fast validation (CPU) | 32 | [1] | 256 |
| `config/plora.full.yml` | Thesis-grade (GPU) | 1000 | [4,8,16] | 1024 |

Switch configs:
```bash
make config-use-dry    # Fast mode
make config-use-full   # Full experiments
```

### Dry Run (19 steps, ~30 min CPU)

```bash
make dry-run-lite
```

Executes:
1. Unit tests
2. Dataset preloading
3. Spectral constant C calibration (ER/WS/BA)
4. Spectral/Cheeger bounds validation
5. Probe calibration
6. Per-domain adapter training
7. Adapter signing
8. Swarm v2 simulation (security on)
9. Report summarization
10. Monolithic baseline training
11. Value-add rank sweep
12. Thesis sweep (ER/WS/BA topologies)
13. Rank/scheme ablations
14. Alternating train–merge stability study
15. Value-add JSONL verification
16. Consensus-enabled smoke test
17. gRPC offer/fetch demo
18. Security policy dump
19. Network IT metrics

### Full Experiment

```bash
make full-experiment
```

Same 19 steps with full-scale parameters. Supports resuming:
```bash
make full-experiment-status   # Check progress
make full-experiment-reset    # Start fresh
```

### Artifacts

After running experiments:

```
results/
├── summary_v2.json           # Swarm v2 aggregated metrics
├── thesis_sweep.jsonl        # Multi-topology sweep results
├── c_calib_er.json           # Spectral constant calibration
├── bounds_validation.json    # Cheeger bounds validation
├── net_it_metrics.json       # Network MI/TE with CIs
├── ablation.jsonl            # Rank/scheme ablation
├── alt_train_merge/          # Convergence analysis
├── value_add/
│   ├── value_add.jsonl       # Per-cell statistical tests
│   └── value_add.md          # Human-readable report
└── figures/                  # Publication plots

out/
├── arithmetic/               # Domain adapters
├── legal/
├── medical/
└── monolithic_r4/            # Baseline comparison
```

## Information-Theoretic Metrics

### Mutual Information Estimators

| Estimator | Method | Module |
|-----------|--------|--------|
| **KSG k-NN** | Kraskov-Stögbauer-Grassberger | `plora.it_estimators.mi_knn` |
| **MINE** | Neural network variational bound | `plora.mine.mine_estimate` |

### Transfer Entropy

Discrete TE via histogram binning:
```python
from plora.te import transfer_entropy_discrete

te_a_to_b = transfer_entropy_discrete(series_a, series_b, k=1, bins=8)
```

### Swarm Metrics

| Metric | Description |
|--------|-------------|
| `coverage(knowledge)` | Fraction of agents possessing each domain |
| `mutual_information(knowledge)` | I(Agent; Domain) in bits |
| `spectral_gap(neighbours)` | λ₂ of graph Laplacian |
| `conductance_estimate(neighbours)` | Approximate Φ(G) via random cuts |
| `cheeger_bounds(neighbours)` | Lower/upper bounds on λ₂ |

## Graph Theory

### Supported Topologies

| Topology | Function | Key Parameters |
|----------|----------|----------------|
| Erdős–Rényi | `erdos_renyi_graph(n, p, seed)` | Edge probability `p` |
| Watts–Strogatz | `watts_strogatz_graph(n, k, beta, seed)` | Ring degree `k`, rewiring `β` |
| Barabási–Albert | `barabasi_albert_graph(n, m, seed)` | Edges per new node `m` |

### Diffusion Predictions

```python
from swarm.theory import predicted_rounds_spectral, cheeger_bounds

# Spectral prediction: t ≈ C log(n) / λ₂
t_pred = predicted_rounds_spectral(n=100, lambda2=0.3, C=0.7)

# Cheeger bounds on λ₂
lower, upper = cheeger_bounds(neighbours, normalized=True)
```

## Security Features

### Signature Verification

```python
from plora.signer import generate_keypair, sign_sha256_hex, verify_sha256_hex

# Generate keypair
generate_keypair(Path("keys/priv.pem"), Path("keys/pub.pem"))

# Sign and verify
sig = sign_sha256_hex(Path("keys/priv.pem"), sha256_hex)
ok = verify_sha256_hex(Path("keys/pub.pem"), sha256_hex, sig)
```

### Threshold Signatures

```python
from plora.threshold_sigs import aggregate_signatures, verify_aggregate

aggregate = aggregate_signatures([sig1, sig2, sig3])
ok = verify_aggregate(aggregate, sha256_hex, public_keys, quorum=2)
```

### Policy Configuration

```python
from plora.gate import Policy, alignment_gate

policy = Policy(
    base_model="google/gemma-3-1b-it",
    allowed_ranks=[4, 8, 16],
    allowed_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
    signatures_enabled=True,
    quorum_required=2,
    min_reputation=0.5,
    tau_trigger=0.2,        # Trigger compliance threshold
    tau_norm_z=3.0,         # Weight norm z-score threshold
    tau_clean_delta=-0.05,  # Clean accuracy regression threshold
)

gate_result = alignment_gate(adapter_dir, manifest, policy)
```

## Value-Add Experiments

Evaluate trained adapters with statistical rigor:

```bash
python -m scripts.run_lora_value_add \
    --domains arithmetic,legal,medical \
    --ranks 4,8,16 \
    --schemes attention,mlp,all \
    --seeds 41,42,43 \
    --samples 512 \
    --dev-size 1024
```

Outputs:
- **value_add.jsonl**: Per-cell metrics with Wilcoxon tests and bootstrap CIs
- **value_add.md**: Markdown tables with **bold** highlights for significant improvements

Guardrails:
- Abort if placebo beats baseline significantly
- Abort if inject/remove latency exceeds budget

## HuggingFace Authentication

This project uses `google/gemma-3-1b-it`, a gated model. You must accept the license and authenticate:

1. Go to https://huggingface.co/google/gemma-3-1b-it and click **"Agree and access repository"**
2. Create an access token at https://huggingface.co/settings/tokens (read access is sufficient)
3. Log in locally:

```bash
pip install huggingface_hub
huggingface-cli login   # paste your token when prompted
```

For Docker, pass the token as an environment variable:

```bash
make docker-test HF_TOKEN=hf_xxxxx
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `HF_TOKEN` | HuggingFace access token for gated models | (required) |
| `PLORA_BASE_MODEL` | HF model for training/metrics | `google/gemma-3-1b-it` |
| `PLORA_LATENCY_BUDGET_MS` | Latency guardrail (ms) | `250` |
| `PLORA_SAMPLES` | Global cap on HuggingFace samples | None |
| `PLORA_FORCE_CPU` | Force CPU even if GPU available | `0` |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | MPS memory setting | `0.0` (recommended) |

## Make Targets

### Core Workflow

| Target | Description |
|--------|-------------|
| `make setup` | Install dependencies via Poetry |
| `make test` | Run pytest suite |
| `make train-all` | Train all domain adapters |
| `make sign-all` | Sign all adapters |
| `make dry-run-lite` | 19-step validation run |
| `make full-experiment` | Thesis-grade experiment |

### Swarm Simulation

| Target | Description |
|--------|-------------|
| `make swarm-v2-smoke` | Push–pull gossip with security |
| `make swarm-v2-eval` | Summarize reports to JSON |
| `make thesis-sweep` | Multi-topology sweep (ER/WS/BA) |
| `make consensus-smoke` | Test consensus protocol |

### Calibration & Validation

| Target | Description |
|--------|-------------|
| `make calibrate-c` | Calibrate spectral constant C |
| `make validate-bounds` | Validate Cheeger bounds |
| `make probes-calib` | Calibrate behavioral probes |
| `make net-it` | Compute network MI/TE metrics |

### Analysis

| Target | Description |
|--------|-------------|
| `make figures` | Generate publication plots |
| `make ablation` | Rank/scheme ablation study |
| `make alt-train-merge` | Alternating train–merge stability |
| `make value-add-build-lowmem` | Build JSONL (memory-constrained) |

### Docker

| Target | Description |
|--------|-------------|
| `make docker-build` | Build the Docker image |
| `make docker-run` | Interactive shell (mounts results/ and out/) |
| `make docker-test` | Run pytest inside the container |
| `make docker-dry-run` | Run 19-step dry run inside the container |

### Utilities

| Target | Description |
|--------|-------------|
| `make dump-policy` | Print effective security policy |
| `make prepare-data` | Preload HuggingFace datasets |
| `make offer-all` | Start gRPC server for all domains |
| `make fetch-all` | Fetch all plasmids from server |

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `math_foundations.ipynb` | LaTeX-ready derivations (LoRA bounds, info theory, graph diffusion) |
| `experiment_analysis.ipynb` | Results visualization and statistical analysis |

Export math notebook to PDF:
```bash
make math-export  # Requires nbconvert/pandoc
```

## Testing

```bash
# Full test suite
make test

# Specific test categories
poetry run pytest tests/test_gate.py -v
poetry run pytest tests/test_swarm_v2.py -v
poetry run pytest tests/test_mi_knn.py -v
```

Key test coverage:
- Manifest validation and consistency
- Signature generation and verification
- Gate policy enforcement and quorum logic
- LoRA injection latency budgets
- Merge correctness (weighted sum, sequential)
- MI estimator calibration on known distributions
- Swarm diffusion and coverage bounds
- Consensus protocol safety

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Docker `Killed` / exit 137 | OOM — not enough RAM + swap. **macOS:** `make docker-setup-swap` after each `colima start`. **Linux:** ensure ≥ 8 GB host swap. **WSL2:** set `swap=8GB` in `.wslconfig` |
| Value-add OOM | Use `make value-add-build-lowmem` with `--skip-placebos --skip-cross` |
| HF 401/403 during model download | Ensure `HF_TOKEN` is set and you accepted the model license at https://huggingface.co/google/gemma-3-1b-it |
| gRPC signature failure | Ensure matching keypair between offer server and fetch client |
| Slow on CPU | Use `config/plora.dry.yml` and set `PLORA_SAMPLES=32` |
| MPS memory errors | Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` |
| Missing artifacts | Run `make artefacts-check` to diagnose |

## Author

Artem Pitertsev <artem.pitertsev@gmail.com>
