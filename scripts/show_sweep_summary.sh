#!/bin/bash
# pLoRA Thesis Sweep Summary
# Usage: ./scripts/show_sweep_summary.sh [sweep_file]

SWEEP_FILE="${1:-results/thesis_sweep.jsonl}"

python3 -c "
import json

results = [json.loads(line) for line in open('$SWEEP_FILE')]

print('=' * 55)
print('       pLoRA THESIS SWEEP SUMMARY')
print('=' * 55)
print(f'  Total experiments:  {len(results)}')
print()

# Group by topology
topos = {}
for r in results:
    t = r['topology'].upper()
    if t not in topos:
        topos[t] = []
    topos[t].append(r)

print('-' * 55)
print('  RESULTS BY TOPOLOGY')
print('-' * 55)
for topo, exps in sorted(topos.items()):
    within = sum(1 for e in exps if e['t_obs'] <= e['t_pred'])
    avg_ratio = sum(e['t_obs']/max(e['t_pred'],1) for e in exps) / len(exps)
    print(f'  {topo:5s}  {len(exps):2d} experiments  {within}/{len(exps)} within bound ({100*within/len(exps):.0f}%)')

print()
print('-' * 55)
print('  SPECTRAL BOUND VALIDATION')
print('-' * 55)
within_bound = sum(1 for r in results if r['t_obs'] <= r['t_pred'])
pct = 100 * within_bound / len(results)
print(f'  Within spectral bound:  {within_bound}/{len(results)} ({pct:.1f}%)')
if pct >= 95:
    print(f'  Status:                 ✓ PASSES 95% threshold!')
else:
    print(f'  Status:                 ✗ Below 95% threshold')

print()
print('-' * 55)
print('  SECURITY GATE PERFORMANCE')
print('-' * 55)
total_fp = sum(r['gate'].get('false_positives', 0) for r in results)
total_fn = sum(r['gate'].get('false_negatives', 0) for r in results)
total_clean = sum(r['gate'].get('accepted_clean_total', 0) + r['gate'].get('rejected_clean_total', 0) for r in results)
total_trojan = sum(r['gate'].get('accepted_trojan_total', 0) + r['gate'].get('rejected_trojan_total', 0) for r in results)

print(f'  False Positives:  {total_fp}')
print(f'  False Negatives:  {total_fn}')
if total_fp == 0 and total_fn == 0:
    print(f'  Status:           ✓ PERFECT security gate!')
print()
print('=' * 55)
"
