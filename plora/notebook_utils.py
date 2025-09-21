from __future__ import annotations

"""plora.notebook_utils - Centralized utilities for experiment analysis notebook.

This module consolidates commonly used functions for data loading, processing,
and analysis that are repeated across the experiment analysis notebook.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

def find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml or Makefile."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / "Makefile").exists():
            return current
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

    # Swarm simulation stats
    if 'swarm_summary' in experiment_data:
        swarm_data = experiment_data['swarm_summary']
        stats['swarm_experiments'] = len(swarm_data)

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

    Args:
        experiment_data: Dictionary containing experiment data

    Returns:
        DataFrame with swarm metrics
    """
    if not experiment_data.get('swarm_summary'):
        return pd.DataFrame()

    swarm_data = experiment_data['swarm_summary']
    records = []

    for exp in swarm_data:
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
            'mi_final': exp.get('mi', {}).get('final'),
            'mi_max': exp.get('mi', {}).get('max'),
            'mi_min': exp.get('mi', {}).get('min'),
            'gate_rejected_hash_total': exp.get('gate', {}).get('rejected_hash_total', 0),
            'gate_rejected_safety_total': exp.get('gate', {}).get('rejected_safety_total', 0),
            'gate_accepted_clean_total': exp.get('gate', {}).get('accepted_clean_total', 0),
            'gate_accepted_trojan_total': exp.get('gate', {}).get('accepted_trojan_total', 0),
        }
        records.append(record)

    return pd.DataFrame(records)


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
        config = exp.get('config', {})
        trained = exp.get('trained', {})
        placebo_a = exp.get('placebo_a', {})
        placebo_b = exp.get('placebo_b', {})
        cross_domain = exp.get('cross_domain', {})

        record = {
            'domain': config.get('domain'),
            'rank': config.get('rank'),
            'scheme': config.get('scheme'),
            'seed': config.get('seed'),
            'eval_split': config.get('eval_split'),
            'trained_delta_mean': trained.get('delta_mean', 0),
            'trained_wilcoxon_p': trained.get('wilcoxon_p', 0),
            'trained_ci_low': trained.get('ci', [0, 0])[0],
            'trained_ci_high': trained.get('ci', [0, 0])[1],
            'placebo_a_delta_mean': placebo_a.get('delta_mean', 0),
            'placebo_a_wilcoxon_p': placebo_a.get('wilcoxon_p', 0),
            'placebo_b_delta_mean': placebo_b.get('delta_mean', 0),
            'placebo_b_wilcoxon_p': placebo_b.get('wilcoxon_p', 0),
            'latency_ms': exp.get('latency_ms', 0),
        }

        # Add cross-domain metrics
        for other_domain, metrics in cross_domain.items():
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

    security_metrics = {
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
        security_metrics['false_negative_rates'] = {
            'mean': np.mean(security_metrics['false_negative_rates']),
            'std': np.std(security_metrics['false_negative_rates'])
        }

    return security_metrics
