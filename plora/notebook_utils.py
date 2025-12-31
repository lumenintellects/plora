from __future__ import annotations

"""plora.notebook_utils - Centralized utilities for experiment analysis notebook.

This module consolidates commonly used functions for data loading, processing,
and analysis that are repeated across the experiment analysis notebook.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

def find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml or Makefile."""
    current = Path.cwd()
    max_levels = 20
    levels = 0
    while current != current.parent and levels < max_levels:
        if (current / "pyproject.toml").exists() or (current / "Makefile").exists():
            return current
        # advance upward
        current = current.parent
        levels += 1
    return Path.cwd()


def load_experiment_data() -> Dict[str, Any]:
    """Load all experiment data in a standardized way.

    Returns:
        Dictionary containing all experiment data organized by category
    """
    project_root = find_project_root()
    results_path = project_root / "results"
    out_path = project_root / "out"
    config_path = project_root / "config"

    experiment_data = {}

    # Load swarm simulation results
    swarm_summary_file = results_path / "summary_v2.json"
    if swarm_summary_file.exists():
        with open(swarm_summary_file, 'r') as f:
            experiment_data['swarm_summary'] = json.load(f)

    swarm_files = list(results_path.glob("swarm_v2_report_*.json"))
    if swarm_files:
        experiment_data['swarm_reports'] = []
        for file in swarm_files:
            with open(file, 'r') as f:
                experiment_data['swarm_reports'].append(json.load(f))

    # Load value-add results
    value_add_file = results_path / "value_add" / "value_add.jsonl"
    if value_add_file.exists():
        experiment_data['value_add'] = []
        with open(value_add_file, 'r') as f:
            for line in f:
                experiment_data['value_add'].append(json.loads(line))

    # Load adapter manifests
    experiment_data['adapters'] = {}
    for domain in ['arithmetic', 'legal', 'medical']:
        manifest_file = out_path / domain / "plora.yml"
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                import yaml
                experiment_data['adapters'][domain] = yaml.safe_load(f)

    # Load configuration files
    config_files = {}
    for config_file in ['plora.full.yml', 'plora.dry.yml']:
        config_path_file = config_path / config_file
        if config_path_file.exists():
            with open(config_path_file, 'r') as f:
                import yaml
                config_files[config_file] = yaml.safe_load(f)
    experiment_data['configs'] = config_files

    # Load additional result files
    additional_files = [
        "thesis_sweep.jsonl",
        "c_calib_er.json",
        "bounds_validation.json",
        "probes_calib.json",
        "net_it_metrics.json"
    ]

    for filename in additional_files:
        file_path = results_path / filename
        if file_path.exists():
            if filename.endswith('.jsonl'):
                data = []
                with open(file_path, 'r') as f:
                    for line in f:
                        data.append(json.loads(line))
                experiment_data[filename.replace('.jsonl', '')] = data
            else:
                with open(file_path, 'r') as f:
                    experiment_data[filename.replace('.json', '')] = json.load(f)

    # Set default values for missing data
    for key, default in [
        ('swarm_summary', []),
        ('value_add', []),
        ('swarm_reports', [])
    ]:
        if key not in experiment_data:
            experiment_data[key] = default

    return experiment_data


def get_experiment_summary_stats(experiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics for all loaded experiment data.

    Args:
        experiment_data: Dictionary containing all experiment data

    Returns:
        Dictionary with summary statistics
    """
    stats = {}

    # Use thesis_sweep for swarm stats (the main experimental data)
    # thesis_sweep contains the full 72-experiment campaign used for RQ2/RQ3
    thesis_sweep = experiment_data.get('thesis_sweep', [])
    if thesis_sweep:
        stats['swarm_experiments'] = len(thesis_sweep)
        stats['swarm_data_source'] = 'thesis_sweep.jsonl'
        
        topologies = [exp.get('topology', 'unknown') for exp in thesis_sweep]
        stats['topologies'] = sorted(set(topologies))
        
        n_agents = [exp.get('N', 0) for exp in thesis_sweep]
        stats['agent_counts'] = {
            'min': min(n_agents) if n_agents else 0,
            'max': max(n_agents) if n_agents else 0,
            'unique': len(set(n_agents)),
            'values': sorted(set(n_agents))
        }
    # Use swarm_summary if thesis_sweep not available
    elif 'swarm_summary' in experiment_data:
        swarm_data = experiment_data['swarm_summary']
        stats['swarm_experiments'] = len(swarm_data)
        stats['swarm_data_source'] = 'summary_v2.json'

        if swarm_data:
            topologies = [exp.get('topology', 'unknown') for exp in swarm_data]
            stats['topologies'] = list(set(topologies))

            n_agents = [exp.get('N', 0) for exp in swarm_data]
            stats['agent_counts'] = {
                'min': min(n_agents) if n_agents else 0,
                'max': max(n_agents) if n_agents else 0,
                'unique': len(set(n_agents))
            }

    # Value-add stats
    if 'value_add' in experiment_data:
        value_add_data = experiment_data['value_add']
        stats['value_add_experiments'] = len(value_add_data)

        if value_add_data:
            domains = list(set(exp.get('config', {}).get('domain', 'unknown')
                             for exp in value_add_data))
            stats['value_add_domains'] = domains

            ranks = list(set(exp.get('config', {}).get('rank', 0)
                           for exp in value_add_data))
            stats['value_add_ranks'] = sorted(ranks)

    # Adapter stats
    if 'adapters' in experiment_data:
        adapter_data = experiment_data['adapters']
        stats['adapter_domains'] = list(adapter_data.keys())

    return stats


def extract_swarm_metrics(experiment_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract swarm simulation metrics into a pandas DataFrame.

    Adds new MI-derived metrics: cumulative absolute MI change, total MI loss (sum of
    negative deltas magnitudes), normalized final MI, and bootstrap CI for delta sum if
    available (delta_ci_low/high).
    """
    if not experiment_data.get('swarm_summary'):
        return pd.DataFrame()

    swarm_data = experiment_data['swarm_summary']
    records = []

    for exp in swarm_data:
        mi = exp.get('mi', {})
        record = {
            'N': exp.get('N'),
            'topology': exp.get('topology'),
            'lambda2': exp.get('lambda2', 0),
            'observed_t_all': exp.get('observed_t_all'),
            'predicted_t_all': exp.get('predicted_t_all'),
            'acceptance_rate': exp.get('acceptance_rate'),
            't_all': exp.get('t_all'),
            'bytes_on_wire': exp.get('bytes_on_wire', 0),
            'accepted_offers': exp.get('accepted_offers', 0),
            'coverage': exp.get('coverage', {}),
            'mi_final': mi.get('final'),
            'mi_max': mi.get('max'),
            'mi_min': mi.get('min'),
            'mi_delta_sum': mi.get('delta_sum'),
            'mi_delta_ci_low': (mi.get('delta_ci') or [None, None])[0],
            'mi_delta_ci_high': (mi.get('delta_ci') or [None, None])[1],
            'mi_cum_abs': mi.get('cum_abs'),
            'mi_total_loss': mi.get('total_loss'),
            'mi_norm_final': mi.get('norm_final'),
            'gate_rejected_hash_total': exp.get('gate', {}).get('rejected_hash_total', 0),
            'gate_rejected_safety_total': exp.get('gate', {}).get('rejected_safety_total', 0),
            'gate_accepted_clean_total': exp.get('gate', {}).get('accepted_clean_total', 0),
            'gate_accepted_trojan_total': exp.get('gate', {}).get('accepted_trojan_total', 0),
            'gate_rejected_clean_total': exp.get('gate', {}).get('rejected_clean_total', 0),
            'gate_rejected_trojan_total': exp.get('gate', {}).get('rejected_trojan_total', 0),
            'gate_false_negatives': exp.get('gate', {}).get('false_negatives', 0),
            'gate_false_positives': exp.get('gate', {}).get('false_positives', 0),
        }
        records.append(record)

    return pd.DataFrame(records)


def extract_swarm_round_metrics(experiment_data: Dict[str, Any]) -> pd.DataFrame:
    """Return a long-form per-round DataFrame across all available swarm reports.

    Columns: seed, topology, N, round, entropy_avg, mutual_information, mi_delta, mi_loss,
    mi_cum_abs, mi_norm, accepted_count, plus per-domain coverage columns (coverage_<domain>).
    """
    reports: List[dict] = experiment_data.get('swarm_reports') or []
    if not reports:
        return pd.DataFrame()

    rows: List[dict] = []
    for rep in reports:
        meta = rep.get('meta', {})
        rounds = rep.get('rounds', [])
        domains = meta.get('domains') or []
        seed = meta.get('seed')
        topology = meta.get('topology')
        N = meta.get('N')
        for r in rounds:
            base = {
                'seed': seed,
                'topology': topology,
                'N': N,
                'round': r.get('t'),
                'entropy_avg': r.get('entropy_avg'),
                'mutual_information': r.get('mutual_information'),
                'mi_delta': r.get('mi_delta'),
                'mi_loss': r.get('mi_loss'),
                'mi_cum_abs': r.get('mi_cum_abs'),
                'mi_norm': r.get('mi_norm'),
                'accepted_count': len(r.get('accepted', [])),
            }
            cov = r.get('coverage', {}) or {}
            for d in domains:
                base[f'coverage_{d}'] = cov.get(d, 0.0)
            rows.append(base)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Sort for ergonomic viewing
    return df.sort_values(['seed', 'round']).reset_index(drop=True)


def extract_value_add_metrics(experiment_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract value-add experiment metrics into a pandas DataFrame.

    Args:
        experiment_data: Dictionary containing experiment data

    Returns:
        DataFrame with value-add metrics
    """
    if not experiment_data.get('value_add'):
        return pd.DataFrame()

    value_add_data = experiment_data['value_add']
    records = []

    for exp in value_add_data:
        config = exp.get('config', {}) or {}
        trained = exp.get('trained', {}) or {}
        placebo_a = exp.get('placebo_a') or {}
        placebo_b = exp.get('placebo_b') or {}
        cross_domain = exp.get('cross_domain') or {}

        record = {
            'domain': config.get('domain'),
            'rank': config.get('rank'),
            'scheme': config.get('scheme'),
            'seed': config.get('seed'),
            'eval_split': config.get('eval_split'),
            'trained_delta_mean': trained.get('delta_mean', 0),
            'trained_wilcoxon_p': trained.get('wilcoxon_p', 1.0),
            'trained_ci_low': (trained.get('ci') or [0, 0])[0],
            'trained_ci_high': (trained.get('ci') or [0, 0])[1],
            'placebo_a_delta_mean': placebo_a.get('delta_mean', 0),
            'placebo_a_wilcoxon_p': placebo_a.get('wilcoxon_p', 1.0),
            'placebo_b_delta_mean': placebo_b.get('delta_mean', 0),
            'placebo_b_wilcoxon_p': placebo_b.get('wilcoxon_p', 1.0),
            'latency_ms': exp.get('latency_ms', 0),
        }

        # Add cross-domain metrics
        for other_domain, metrics in cross_domain.items():
            if not isinstance(metrics, dict):
                continue
            record[f'cross_{other_domain}_delta_mean'] = metrics.get('delta_mean', 0)

        records.append(record)

    return pd.DataFrame(records)


def calculate_diffusion_efficiency(experiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate diffusion efficiency metrics from swarm data.

    Args:
        experiment_data: Dictionary containing experiment data

    Returns:
        Dictionary with diffusion efficiency metrics
    """
    if not experiment_data.get('swarm_summary'):
        return {}

    swarm_data = experiment_data['swarm_summary']

    # Calculate predicted vs observed diffusion times
    diffusion_ratios = []
    for exp in swarm_data:
        predicted = exp.get('predicted_t_all')
        observed = exp.get('observed_t_all')

        if predicted is not None and observed is not None and predicted > 0:
            ratio = observed / predicted
            diffusion_ratios.append(ratio)

    if diffusion_ratios:
        return {
            'diffusion_efficiency_ratio': {
                'mean': np.mean(diffusion_ratios),
                'std': np.std(diffusion_ratios),
                'min': np.min(diffusion_ratios),
                'max': np.max(diffusion_ratios)
            },
            'n_experiments': len(diffusion_ratios)
        }

    return {}


def get_security_summary(experiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate security analysis summary from swarm data.

    Args:
        experiment_data: Dictionary containing experiment data

    Returns:
        Dictionary with security metrics
    """
    if not experiment_data.get('swarm_summary'):
        return {}

    swarm_data = experiment_data['swarm_summary']

    security_metrics: Dict[str, Any] = {
        'total_experiments': len(swarm_data),
        'rejection_rates': [],
        'false_positive_rates': [],
        'false_negative_rates': []
    }

    for exp in swarm_data:
        gate = exp.get('gate', {})

        rejected_total = gate.get('rejected_hash_total', 0) + gate.get('rejected_safety_total', 0)
        accepted_clean = gate.get('accepted_clean_total', 0)
        rejected_clean = gate.get('rejected_clean_total', 0)
        accepted_trojan = gate.get('accepted_trojan_total', 0)
        rejected_trojan = gate.get('rejected_trojan_total', 0)

        total_offers = (accepted_clean + rejected_clean + accepted_trojan + rejected_trojan)

        if total_offers > 0:
            rejection_rate = rejected_total / total_offers
            false_positive_rate = rejected_clean / (accepted_clean + rejected_clean) if (accepted_clean + rejected_clean) > 0 else 0
            false_negative_rate = accepted_trojan / (accepted_trojan + rejected_trojan) if (accepted_trojan + rejected_trojan) > 0 else 0

            security_metrics['rejection_rates'].append(rejection_rate)
            security_metrics['false_positive_rates'].append(false_positive_rate)
            security_metrics['false_negative_rates'].append(false_negative_rate)

    # Calculate summary statistics
    if security_metrics['rejection_rates']:
        security_metrics['rejection_rate_summary'] = {
            'mean': np.mean(security_metrics['rejection_rates']),
            'std': np.std(security_metrics['rejection_rates'])
        }

    if security_metrics['false_positive_rates']:
        security_metrics['false_positive_rate_summary'] = {
            'mean': np.mean(security_metrics['false_positive_rates']),
            'std': np.std(security_metrics['false_positive_rates'])
        }

    if security_metrics['false_negative_rates']:
        security_metrics['false_negative_rate_summary'] = {
            'mean': np.mean(security_metrics['false_negative_rates']),
            'std': np.std(security_metrics['false_negative_rates'])
        }

    return security_metrics


def get_swarm_df(experiment_data: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Convenience helper to obtain the swarm metrics DataFrame used in notebooks.

    This resolves "unresolved reference 'swarm_df'" warnings by centralizing the
    creation of the DataFrame instead of relying on a variable defined ad-hoc in
    a notebook cell.

    Args:
        experiment_data: Optional pre-loaded experiment data. If not supplied,
            the function loads data via load_experiment_data().

    Returns:
        pandas.DataFrame with swarm metrics (may be empty if no data present).
    """
    if experiment_data is None:
        experiment_data = load_experiment_data()
    return extract_swarm_metrics(experiment_data)
