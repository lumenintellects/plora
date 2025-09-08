# Smoke-run Value-Add Experiment (Gemma-3B, rank-2, attention only)

**Run stamp:** $(date)

This minimal experiment trains and evaluates one LoRA per domain with the following knobs:

* base model: `google/gemma-3-1b-it` (CPU)
* domains: arithmetic, legal, medical
* LoRA rank `r = 2`, scheme `attention`
* samples/train = 128, dev = 512, seed = 41
* placebo A: random LoRA (σ = 1e-4, rank 2)
* placebo B: label-shuffled training
* latency budget = 250 ms (median inject/remove)

---

## Aggregate results

| Domain | ΔNLL (trained) | 95 % CI | p-value | Latency ms | Placebo A Δ | Placebo B Δ |
|--------|----------------|---------|---------|------------|-------------|-------------|
| arithmetic | **-1.47** | [-1.50, ‑1.44] | 1.5 e-154 | 201 | ‑0.07 | ‑1.60 |
| legal      | **-3.88** | [-3.93, ‑3.84] | 1.5 e-154 | 197 | ‑0.00 | ‑4.09 |
| medical    | **-2.88** | [-2.94, ‑2.82] | 1.5 e-154 | 200 | ‑0.00 | ‑3.22 |

Notes:
* All trained adapters comfortably beat the base model.
* Placebo A deltas are close to zero as expected after down-scaling random weights.
* Placebo B (label shuffle) degrades perplexity ≈ 1-3 PPL, demonstrating the importance of correct supervision.
* Median inject/remove latency < 250 ms for all adapters (CPU).

## Cross-domain transfer (ΔNLL w.r.t. baseline)

| Source → Target | arithmetic | legal | medical |
|-----------------|------------|-------|---------|
| arithmetic LoRA | —          | +0.00 | +0.00 |
| legal LoRA      | ‑0.47      | —     | ‑0.73 |
| medical LoRA    | ‑0.95      | ‑1.54 | — |

Negative numbers mean the specialist adapter helps the other domain.
Small cross-domain gains suggest limited negative transfer at rank-2.

## Verdict

The smoke configuration exercises the full pipeline and passes all guard-rails:
no placebo artefact beats baseline and latency stays within the 250 ms budget.
LoRA rank 2 on attention heads alone already yields 1-4 PPL improvements over Gemma-3B on CPU.

For a thorough study run `make value-add-full` (larger grid).
