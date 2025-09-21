from __future__ import annotations

"""plora.plotting - Centralized plotting utilities for consistent visualizations.

This module provides standardized plotting functions with consistent styling,
color schemes, and layouts for the experiment analysis notebook.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Set consistent plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Consistent color scheme
COLORS = {
    'medical': '#2E86AB',
    'legal': '#A23B72',
    'arithmetic': '#F18F01',
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'neutral': '#6B7280'
}

# Consistent figure sizes
FIGURE_SIZES = {
    'single': (8, 6),
    'double': (12, 6),
    'quad': (15, 12),
    'wide': (16, 8),
    'tall': (10, 12)
}


def setup_plotting_style():
    """Set up consistent plotting style across all visualizations."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'lines.markersize': 6
    })


def create_swarm_dynamics_plot(swarm_report: Dict[str, Any], figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Create comprehensive swarm dynamics visualization.

    Args:
        swarm_report: Dictionary containing swarm simulation round data
        figsize: Optional figure size override

    Returns:
        Tuple of (figure, axes) for further customization
    """
    if figsize is None:
        figsize = FIGURE_SIZES['quad']

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if 'rounds' not in swarm_report:
        # Create empty plots with message
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'No Data Available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('No Swarm Data')
        return fig, axes

    rounds = swarm_report['rounds']
    domains = ['medical', 'arithmetic', 'legal']

    # Plot 1: Coverage over time
    for i, domain in enumerate(domains):
        coverage_values = [r.get('coverage', {}).get(domain, 0) for r in rounds]
        axes[0, 0].plot(range(len(rounds)), coverage_values,
                        marker='o', label=domain, color=COLORS[domain], linewidth=2)

    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Coverage')
    axes[0, 0].set_title('Domain Coverage Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Mutual Information over time
    mi_values = [r.get('mutual_information', 0) for r in rounds]
    axes[0, 1].plot(range(len(rounds)), mi_values, marker='o', color=COLORS['primary'], linewidth=2)
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Mutual Information')
    axes[0, 1].set_title('Mutual Information Evolution')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Entropy over time
    entropy_values = [r.get('entropy_avg', 0) for r in rounds]
    axes[1, 0].plot(range(len(rounds)), entropy_values, marker='o', color=COLORS['secondary'], linewidth=2)
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Average Entropy')
    axes[1, 0].set_title('Average Entropy Over Time')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Accepted offers per round
    accepted_per_round = [len(r.get('accepted', [])) for r in rounds]
    axes[1, 1].bar(range(len(rounds)), accepted_per_round, color=COLORS['accent'], alpha=0.7)
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Accepted Offers')
    axes[1, 1].set_title('Accepted Offers Per Round')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def create_value_add_summary_plot(experiment_data: Dict[str, Any], figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Create value-add experiment summary visualization.

    Args:
        experiment_data: Dictionary containing experiment data
        figsize: Optional figure size override

    Returns:
        Tuple of (figure, axes) for further customization
    """
    if figsize is None:
        figsize = FIGURE_SIZES['wide']

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    if not experiment_data.get('value_add'):
        for ax in axes:
            ax.text(0.5, 0.5, 'No Data Available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('No Value-Add Data')
        return fig, axes

    value_add_data = experiment_data['value_add']

    # Group data by domain and condition
    domains = ['arithmetic', 'legal', 'medical']
    conditions = ['trained', 'placebo_a', 'placebo_b']

    for i, domain in enumerate(domains):
        ax = axes[i]

        domain_data = [exp for exp in value_add_data
                      if exp.get('config', {}).get('domain') == domain]

        if not domain_data:
            ax.text(0.5, 0.5, f'No {domain} data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{domain.title()} Domain')
            continue

        # Extract delta means for each condition
        condition_means = {}
        for condition in conditions:
            deltas = [exp.get(condition, {}).get('delta_mean', 0)
                     for exp in domain_data]
            condition_means[condition] = np.mean(deltas) if deltas else 0

        # Create bar plot
        conditions_display = ['Trained', 'Placebo A\n(Random)', 'Placebo B\n(Shuffled)']
        values = [condition_means['trained'], condition_means['placebo_a'], condition_means['placebo_b']]
        colors = [COLORS['primary'], COLORS['neutral'], COLORS['neutral']]

        bars = ax.bar(conditions_display, values, color=colors, alpha=0.7)

        # Add value labels
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value".3f"}', ha='center', va='bottom', fontsize=9)

        ax.set_title(f'{domain.title()} Domain')
        ax.set_ylabel('ΔNLL (Mean)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    plt.tight_layout()
    return fig, axes


def create_security_analysis_plot(experiment_data: Dict[str, Any], figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Create security analysis visualization.

    Args:
        experiment_data: Dictionary containing experiment data
        figsize: Optional figure size override

    Returns:
        Tuple of (figure, axes) for further customization
    """
    if figsize is None:
        figsize = FIGURE_SIZES['double']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if not experiment_data.get('swarm_summary'):
        for ax in axes:
            ax.text(0.5, 0.5, 'No Data Available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('No Security Data')
        return fig, axes

    swarm_data = experiment_data['swarm_summary']

    # Extract security metrics
    rejection_rates = []
    false_positive_rates = []
    false_negative_rates = []
    topologies = []

    for exp in swarm_data:
        gate = exp.get('gate', {})
        topology = exp.get('topology', 'unknown')

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

            rejection_rates.append(rejection_rate)
            false_positive_rates.append(false_positive_rate)
            false_negative_rates.append(false_negative_rate)
            topologies.append(topology)

    # Plot rejection rates by topology
    if rejection_rates and topologies:
        unique_topologies = list(set(topologies))
        topology_data = {top: [] for top in unique_topologies}

        for rate, top in zip(rejection_rates, topologies):
            topology_data[top].append(rate)

        topology_means = [np.mean(topology_data[top]) for top in unique_topologies]

        axes[0].bar(unique_topologies, topology_means, color=COLORS['primary'], alpha=0.7)
        axes[0].set_title('Rejection Rates by Topology')
        axes[0].set_ylabel('Rejection Rate')
        axes[0].set_xlabel('Topology')
        axes[0].grid(True, alpha=0.3)

        # Add value labels
        for i, (top, mean_rate) in enumerate(zip(unique_topologies, topology_means)):
            axes[0].text(i, mean_rate + 0.01, f'{mean_rate".3f"}',
                        ha='center', va='bottom', fontsize=9)

    # Plot false positive vs false negative rates
    if false_positive_rates and false_negative_rates:
        axes[1].scatter(false_positive_rates, false_negative_rates,
                       color=COLORS['accent'], alpha=0.7, s=50)

        # Add reference lines
        axes[1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Target FP Rate (5%)')
        axes[1].axvline(x=0.1, color='blue', linestyle='--', alpha=0.5, label='Target FN Rate (10%)')

        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('False Negative Rate')
        axes[1].set_title('Security Trade-off Analysis')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Add quadrant labels
        axes[1].text(0.05, 0.15, 'Good Performance', fontsize=10, ha='center')
        axes[1].text(0.05, 0.05, 'Low FP, High FN\n(False negatives)', fontsize=8, ha='center')
        axes[1].text(0.15, 0.15, 'High FP, Low FN\n(False positives)', fontsize=8, ha='center')
        axes[1].text(0.15, 0.05, 'Poor Performance', fontsize=8, ha='center')

    plt.tight_layout()
    return fig, axes


def create_scalability_analysis_plot(experiment_data: Dict[str, Any], figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Create scalability analysis visualization.

    Args:
        experiment_data: Dictionary containing experiment data
        figsize: Optional figure size override

    Returns:
        Tuple of (figure, axes) for further customization
    """
    if figsize is None:
        figsize = FIGURE_SIZES['wide']

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    if not experiment_data.get('swarm_summary'):
        for ax in axes:
            ax.text(0.5, 0.5, 'No Data Available',
                   ha='center', va='center', transform=ax.transAxes)
            axes[0].set_title('No Scalability Data')
        return fig, axes

    swarm_data = experiment_data['swarm_summary']

    # Extract metrics by agent count
    agent_counts = []
    diffusion_times = []
    acceptance_rates = []
    topologies = []

    for exp in swarm_data:
        n_agents = exp.get('N')
        t_all = exp.get('observed_t_all')
        acceptance_rate = exp.get('acceptance_rate')
        topology = exp.get('topology')

        if n_agents and t_all and acceptance_rate is not None:
            agent_counts.append(n_agents)
            diffusion_times.append(t_all)
            acceptance_rates.append(acceptance_rate)
            topologies.append(topology)

    if not agent_counts:
        for ax in axes:
            ax.text(0.5, 0.5, 'Insufficient Data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('No Scalability Data')
        return fig, axes

    # Plot 1: Diffusion time vs agent count
    scatter = axes[0].scatter(agent_counts, diffusion_times, c=range(len(agent_counts)),
                             cmap='viridis', s=50, alpha=0.7)
    axes[0].set_xlabel('Number of Agents')
    axes[0].set_ylabel('Diffusion Time (Rounds)')
    axes[0].set_title('Diffusion Time vs Network Size')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Experiment Index')

    # Add trend line
    if len(set(agent_counts)) > 1:
        z = np.polyfit(agent_counts, diffusion_times, 1)
        p = np.poly1d(z)
        axes[0].plot(sorted(set(agent_counts)),
                    [p(x) for x in sorted(set(agent_counts))],
                    "r--", alpha=0.8, label=f'Trend: {z[0]".2f"}x + {z[1]".2f"}')
        axes[0].legend()

    # Plot 2: Acceptance rate vs agent count
    axes[1].scatter(agent_counts, acceptance_rates, c=range(len(agent_counts)),
                   cmap='plasma', s=50, alpha=0.7)
    axes[1].set_xlabel('Number of Agents')
    axes[1].set_ylabel('Acceptance Rate')
    axes[1].set_title('Acceptance Rate vs Network Size')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Performance by topology
    if topologies:
        unique_topologies = list(set(topologies))
        topology_stats = {}

        for top in unique_topologies:
            top_times = [diffusion_times[i] for i, t in enumerate(topologies) if t == top]
            top_rates = [acceptance_rates[i] for i, t in enumerate(topologies) if t == top]

            if top_times:
                topology_stats[top] = {
                    'mean_time': np.mean(top_times),
                    'mean_rate': np.mean(top_rates),
                    'count': len(top_times)
                }

        if topology_stats:
            topologies_sorted = sorted(topology_stats.keys())
            times = [topology_stats[top]['mean_time'] for top in topologies_sorted]
            rates = [topology_stats[top]['mean_rate'] for top in topologies_sorted]

            x = np.arange(len(topologies_sorted))
            width = 0.35

            bars1 = axes[2].bar(x - width/2, times, width, label='Diffusion Time',
                              color=COLORS['primary'], alpha=0.7)
            bars2 = axes[2].bar(x + width/2, rates, width, label='Acceptance Rate',
                              color=COLORS['secondary'], alpha=0.7)

            axes[2].set_xlabel('Topology')
            axes[2].set_ylabel('Value')
            axes[2].set_title('Performance by Topology')
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(topologies_sorted)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def save_figure(fig: plt.Figure, filename: str, output_dir: Path = None,
               dpi: int = 300, bbox_inches: str = 'tight') -> Path:
    """Save figure with consistent settings and return the saved path.

    Args:
        fig: Matplotlib figure to save
        filename: Filename for the saved figure
        output_dir: Optional output directory
        dpi: DPI for the saved figure
        bbox_inches: Bounding box setting

    Returns:
        Path to the saved figure
    """
    if output_dir is None:
        output_dir = Path.cwd() / 'figures'

    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)

    print(f"Saved figure to: {filepath}")
    return filepath


# Initialize plotting style
setup_plotting_style()
