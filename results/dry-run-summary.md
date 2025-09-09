# Experiment summary (dry-run-all)

This report summarizes the outputs produced by the dry-run workflow:
- Value-add rank sweep across three domains (arithmetic, legal, medical)
- Swarm v2 push–pull simulation with security gate enabled

Inputs:
- `results/value_add/value_add.jsonl` + `value_add.md`
- `results/value_add/nll_cache.json` (cache only; not plotted)
- `results/swarm_v2_report_seed42.json` and `results/summary_v2.json`
- `Makefile` targets indicate: base model `google/gemma-3-1b-it`, dev size 256, ranks 4/8/16, scheme `all`, seed 42, latency budget raised for experiments


## Value-add results

Configuration (from run):
- Base model: `google/gemma-3-1b-it`
- Domains: arithmetic, legal, medical
- Ranks: 4, 8, 16; Scheme: `all`; Seed: 42
- Dev set size: 256; Metric: mean token NLL per example; Stats: paired Wilcoxon + bootstrap CI
- Placebos: A = random weights; B = label-shuffle training

### Arithmetic

| r | Trained ΔNLL | 95% CI | p | Placebo A Δ | Placebo B Δ | Latency ms |
|---|---|---|---|---|---|---|
| 4 | -1.443 | [-1.487, -1.400] | 1.73e-77 | ~0.000 | -1.521 | 811 |
| 8 | -1.539 | [-1.584, -1.494] | 1.73e-77 | ~0.000 | -1.524 | 820 |
| 16 | -1.631 | [-1.680, -1.585] | 1.73e-77 | ~+0.002 | -1.531 | 902 |

Cross-domain (trained adapter applied to other domains):
- Into legal: Δ = 0.00, -0.12, -0.27 (r = 4, 8, 16)
- Into medical: Δ = 0.00, -0.15, -0.14

Notes:
- Clear, monotonic gains as rank increases.
- Placebo A ~0 as expected. Placebo B shows large gains comparable to trained (unexpected; see Remarks).
- Cross-domain mostly improves slightly (negative Δ), not the typical negative transfer pattern.

### Legal

| r | Trained ΔNLL | 95% CI | p | Placebo A Δ | Placebo B Δ | Latency ms |
|---|---|---|---|---|---|---|
| 4 | -3.883 | [-3.941, -3.825] | 1.73e-77 | ~-0.001 | -4.017 | 908 |
| 8 | -3.993 | [-4.049, -3.939] | 1.73e-77 | ~+0.003 | -3.997 | 837 |
| 16 | -4.094 | [-4.153, -4.038] | 1.73e-77 | ~-0.005 (p≈1.4e-3) | -4.024 | 854 |

Cross-domain:
- Into arithmetic: Δ = -0.53, -0.68, -0.70
- Into medical: Δ = -0.62, -0.93, -1.25

Notes:
- Very strong in-domain gains; modest additional improvements in other domains.
- Placebo B again mirrors trained scale (unexpected).

### Medical

| r | Trained ΔNLL | 95% CI | p | Placebo A Δ | Placebo B Δ | Latency ms |
|---|---|---|---|---|---|---|
| 4 | -2.830 | [-2.912, -2.754] | 1.73e-77 | ~0.000 | -2.935 | 1093 |
| 8 | -2.983 | [-3.070, -2.901] | 1.73e-77 | ~0.000 | -2.961 | 841 |
| 16 | -3.152 | [-3.241, -3.067] | 1.73e-77 | ~+0.006 (p≈1.3e-4) | -2.984 | 846 |

Cross-domain:
- Into arithmetic: Δ = -1.03, -1.01, -1.09
- Into legal: Δ = -1.50, -1.52, -1.57

Notes:
- Strong in-domain gains; cross-domain also improves notably.
- Placebo A mostly ~0 (rank16 shows tiny but statistically significant positive improvement ~+0.006).
- Placebo B again large.

### Latency

Median inject/remove latencies (3-run median per cell) were ~0.8–1.1 s. The guardrail threshold was increased (`PLORA_LATENCY_BUDGET_MS=$(LAT)` with LAT=5000), so runs proceeded. For production, consider tightening this back toward ≤250 ms.

### Value-add takeaways
- All trained adapters show large, statistically significant improvements; higher ranks yield larger gains.
- Cross-domain application often improves, suggesting the adapters are not overly specialized.
- Placebo A behaves as expected (~0). Placebo B (label-shuffle) unexpectedly shows large improvements comparable to trained across all domains.

Placebo B:
- This likely indicates the evaluation protocol (teacher-forced NLL over the full prompt+answer) rewards stylistic alignment or shortcut patterns learned even with shuffled labels.
- Actions to disambiguate:
  1) Evaluate on held-out, instruction-following prompts that force reasoning beyond surface style.
  2) Compute additional metrics (EM/chrF where applicable) and human-check a sample.
  3) Add a “random prompt pairing” placebo at inference (mismatch Q/A only at evaluation) to bound the protocol susceptibility.
  4) Re-run with shorter fine-tuning or lower LR for placebo B to see if magnitude collapses.


## Swarm v2 (push–pull) results

From `results/swarm_v2_report_seed42.json` and `results/summary_v2.json`:
- Topology: Erdos–Renyi, N=6, p=0.25
- Coverage did not change from initial: legal=0.33, medical=0.33, arithmetic=0.33
- Accepted offers: 0; Bytes on wire: 0
- Security gate totals: rejected_safety_total=60 (clean rejects=30; trojan rejects=30)
- Rejection reasons: `rank_not_allowed` (60), `trigger_rate_high` (30)

Interpretation:
- No diffusion occurred because every offer was rejected by the gate.
- The simulated adapters’ manifests use `lora.r = 1` (see the sim dummy manifest). With default allowed ranks {4,8,16}, all offers trip `rank_not_allowed`.
- A portion were also flagged `trigger_rate_high`, consistent with the configured trojan rate in the sim.

How to enable diffusion in security-on mode:
- Allow rank 1 during simulation or emit dummy manifests with an allowed rank:
  - Use: `make dump-policy RANKS=1,4,8,16` (and confirm TARGETS/signatures as needed), or
  - Adjust the sim to generate `lora.r` ∈ allowed ranks.
- If using allowed targets, ensure they match the dummy adapters (or set `allowed_targets` unset to accept any).
- For trojan stress tests, start with a lower trojan rate and/or increase `tau_trigger` to observe mixed accept/reject rather than total rejection.


## Next steps

1) Value-add protocol robustness
- Add alternative eval metrics (EM, chrF) and a small manual check.
- Add “random inference pairing” placebo to bound metric susceptibility.
- Track per-sample delta histograms to verify distributional shifts (not just means).

2) Placebo B anomaly
- Reduce training steps or LR for placebo B; compare curves.
- Evaluate on counterfactual prompts to ensure gains aren’t stylistic artifacts.

3) Swarm v2 diffusion under gate
- Re-run with `RANKS=1,4,8,16` or update dummy manifest `r`.
- Inspect `rejection_reasons` histogram over time once diffusion starts; watch FN/FP.

4) Latency
- Profile `inject()` on your hardware; consider preloading and adapter pooling.
- Target ≤250 ms median for interactive scenarios; keep the budget high only for research sweeps.


## Data leakage

Risks:
- Evaluating on the same distribution or exact items seen during training inflates gains.
- Cache reuse across splits may accidentally mix results.
- Cross-domain sets could share surface text (near-duplicates) with train.

Mitigations implemented:
- Split-aware dataset loading (`plora.dataset_loader.get_dataset(..., split=...)`) with default `PLORA_SPLIT` env and explicit `--eval-split` in value-add.
- Evaluation now uses `validation` (or `test`) while training remains on `train`.
- Cache keys include the eval split to avoid cross-contamination.

Additional guardrails:
- Add a simple lexical overlap check between train and eval (e.g., hash of normalized prompts) and report collision rate.
- Prefer dataset-native `validation`/`test` splits for GSM8K/LexGLUE/PubMedQA; avoid ad‑hoc shuffles for eval.
- Seed isolation: use separate RNG seeds for data sampling/shuffling vs training.
- Keep a provenance log (dataset name, subset, split, commit hash of code) in each record.


## Appendix
- Source config hints from `Makefile`:
  - Rank sweep target: `value-add-rank-sweep` with dev-size 256
  - Swarm v2 smoke with security on and trojan_rate=0.3
- Artifacts:
  - Value-add tables: `results/value_add/value_add.md`
  - Detailed records: `results/value_add/value_add.jsonl`
  - Swarm report: `results/swarm_v2_report_seed42.json` and `results/summary_v2.json` 
