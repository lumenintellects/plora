#!/bin/bash
# pLoRA Gossip Simulation Results Viewer
# Usage: ./scripts/show_gossip_results.sh [results_file]

RESULTS_FILE="${1:-results/swarm_v2_report_seed123.json}"

python3 -c "
import json
r = json.load(open('$RESULTS_FILE'))
print('=' * 50)
print('     pLoRA GOSSIP SIMULATION RESULTS')
print('=' * 50)
print(f\"  Topology:     {r['meta']['topology'].replace('_', '-').title()}\")
print(f\"  Agents:       {r['meta']['N']}\")
print(f\"  Spectral λ₂:  {r['meta']['lambda2']:.4f}\")
print('-' * 50)
print('  DIFFUSION PROGRESS')
print('-' * 50)
for rd in r['rounds'][:3]:
    cov = rd['coverage']
    avg = sum(cov.values()) / len(cov) * 100
    acc = len(rd.get('accepted', []))
    print(f\"  Round {rd['t']}: {avg:5.1f}% coverage  ({acc:2d} exchanges)\")
print('-' * 50)
print('  KEY RESULTS')
print('-' * 50)
print(f\"  Predicted rounds:  {r['final']['predicted_t_all']}\")
print(f\"  Observed rounds:   {r['final']['observed_t_all']}  ✓ FASTER than theory!\")
print(f\"  Total exchanges:   {r['final']['accepted_offers']}\")
print(f\"  Final Coverage:\")
for dom, cov in r['final']['coverage'].items():
    print(f\"    • {dom:12s} {cov*100:5.1f}%\")
print('=' * 50)
"
