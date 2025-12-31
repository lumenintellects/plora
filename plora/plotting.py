from __future__ import annotations

"""plora.plotting - Centralized plotting utilities for consistent visualizations.

This module provides standardized plotting functions with consistent styling,
color schemes, and layouts for the experiment analysis notebook.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
from plora.stats import bootstrap_ci_mean

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

    Adds overlay of new MI metrics (mi_norm, mi_cum_abs) and highlights per-round
    negative MI changes (mi_loss) via shaded bars.

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
                        marker='o', label=domain, color=COLORS.get(domain, COLORS['primary']), linewidth=2)

    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Coverage')
    axes[0, 0].set_title('Domain Coverage Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Mutual Information + new MI metrics
    mi_values = [r.get('mutual_information', 0) for r in rounds]
    mi_norm_values = [r.get('mi_norm', np.nan) for r in rounds]
    mi_cum_abs_values = [r.get('mi_cum_abs', np.nan) for r in rounds]
    mi_loss_values = [r.get('mi_loss', 0.0) for r in rounds]
    t_range = range(len(rounds))

    axes[0, 1].plot(t_range, mi_values, marker='o', color=COLORS['primary'], linewidth=2, label='MI')
    # Secondary normalized MI line (scaled if magnitude differs)
    if any(not np.isnan(v) for v in mi_norm_values):
        axes[0, 1].plot(t_range, mi_norm_values, linestyle='--', color=COLORS['secondary'], label='MI Norm')
    # Cumulative absolute change line
    if any(not np.isnan(v) for v in mi_cum_abs_values):
        axes[0, 1].plot(t_range, mi_cum_abs_values, linestyle=':', color=COLORS['accent'], label='MI Cum |Δ|')
    # Highlight per-round MI loss (negative deltas) as red bars at bottom
    loss_heights = [lv if lv > 0 else 0 for lv in mi_loss_values]
    if any(lh > 0 for lh in loss_heights):
        axes[0, 1].bar(t_range, [-lh for lh in loss_heights], color='#B22222', alpha=0.35, label='MI Loss (neg Δ)')

    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Mutual Information')
    axes[0, 1].set_title('Mutual Information & Derived Metrics')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Entropy over time
    entropy_values = [r.get('entropy_avg', 0) for r in rounds]
    axes[1, 0].plot(range(len(rounds)), entropy_values, marker='o', color=COLORS['secondary'], linewidth=2)
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Average Entropy')
    axes[1, 0].set_title('Average Entropy Over Time')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Accepted offers per round
    # Handle both formats: 'accepted' as list (legacy) or 'accepted_count' as int (thesis_sweep)
    accepted_per_round = []
    for r in rounds:
        if 'accepted_count' in r:
            accepted_per_round.append(r['accepted_count'])
        else:
            accepted_per_round.append(len(r.get('accepted', [])))
    axes[1, 1].bar(range(len(rounds)), accepted_per_round, color=COLORS['accent'], alpha=0.7)
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Accepted Offers')
    axes[1, 1].set_title('Accepted Offers Per Round')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def create_value_add_summary_plot(experiment_data: Dict[str, Any], figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Create value-add experiment summary visualization.

    Adds bootstrap 95% CI error bars for ΔNLL (mean delta across configs per domain).

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
    conditions = ['trained', 'placebo_a']

    for i, domain in enumerate(domains):
        ax = axes[i]

        domain_data = [exp for exp in value_add_data
                      if exp.get('config', {}).get('domain') == domain]

        if not domain_data:
            ax.text(0.5, 0.5, f'No {domain} data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{domain.title()} Domain')
            continue

        # Extract per-config delta means & bootstrap CI for each condition
        condition_stats = {}
        for condition in conditions:
            # Handle None values explicitly (placebo_a can be null in JSON)
            deltas = []
            for exp in domain_data:
                cond_data = exp.get(condition)
                if cond_data is not None and isinstance(cond_data, dict):
                    delta = cond_data.get('delta_mean')
                    if delta is not None:
                        deltas.append(delta)
            if deltas:
                mean_val = float(np.mean(deltas))
                try:
                    ci_lo, ci_hi = bootstrap_ci_mean(deltas, n_resamples=1000, ci=0.95, seed=42)
                except Exception:
                    ci_lo, ci_hi = mean_val, mean_val
            else:
                mean_val = 0.0
                ci_lo, ci_hi = 0.0, 0.0
            condition_stats[condition] = {
                'mean': mean_val,
                'ci_lo': ci_lo,
                'ci_hi': ci_hi,
            }

        # Create bar plot with error bars (CI)
        conditions_display = ['Trained', 'Placebo A\n(Random)']
        values = [condition_stats[c]['mean'] for c in conditions]
        yerr = [
            [condition_stats[c]['mean'] - condition_stats[c]['ci_lo'] for c in conditions],
            [condition_stats[c]['ci_hi'] - condition_stats[c]['mean'] for c in conditions]
        ]
        colors = [COLORS['primary'], COLORS['neutral']]

        bars = ax.bar(conditions_display, values, color=colors, alpha=0.75, yerr=yerr, capsize=4, linewidth=1.0, edgecolor='black')

        # Add value labels (mean and CI span)
        for bar, c in zip(bars, conditions):
            height = bar.get_height()
            ci_lo = condition_stats[c]['ci_lo']
            ci_hi = condition_stats[c]['ci_hi']
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f"{height:.3f}\n[{ci_lo:.3f},{ci_hi:.3f}]", ha='center', va='bottom', fontsize=8)

        ax.set_title(f'{domain.title()} Domain')
        ax.set_ylabel('ΔNLL (Mean, 95% CI)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

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
                    "r--", alpha=0.8, label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
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


# =============================================================================
# Thesis Sweep Plotting Functions
# =============================================================================

def plot_swarm_dynamics(
    experiment_data: Dict[str, Any],
    topology: str | None = None,
    N: int | None = None,
    figsize: Tuple[int, int] = (14, 10),
    title_prefix: str = ""
) -> plt.Figure:
    """Plot swarm dynamics: coverage, entropy, MI, and accepted offers over rounds.
    
    This creates an aggregated 4-panel plot showing per-round metrics across experiments.
    
    Args:
        experiment_data: Dictionary containing experiment data
        topology: Filter by topology (er, ws, ba). If None, aggregates all.
        N: Filter by network size. If None, aggregates all.
        figsize: Figure size tuple
        title_prefix: Optional prefix for the plot title
        
    Returns:
        matplotlib Figure object
    """
    from plora.notebook_utils import extract_thesis_sweep_rounds_df
    
    rounds_df = extract_thesis_sweep_rounds_df(experiment_data)
    if rounds_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No thesis_sweep data available", ha='center', va='center')
        return fig
    
    # Apply filters
    if topology:
        rounds_df = rounds_df[rounds_df['topology'] == topology]
    if N:
        rounds_df = rounds_df[rounds_df['N'] == N]
    
    if rounds_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"No data for topology={topology}, N={N}", ha='center', va='center')
        return fig
    
    # Aggregate by round (mean across seeds/configs)
    coverage_cols = [c for c in rounds_df.columns if c.startswith('coverage_')]
    agg_dict = {
        'mutual_information': ['mean', 'std'],
        'entropy_avg': ['mean', 'std'],
        'accepted_count': ['mean', 'std'],
    }
    for col in coverage_cols:
        agg_dict[col] = ['mean', 'std']
    
    grouped = rounds_df.groupby('round').agg(agg_dict)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Coverage per domain
    ax = axes[0, 0]
    for col in coverage_cols:
        domain = col.replace('coverage_', '')
        mean_vals = grouped[(col, 'mean')]
        std_vals = grouped[(col, 'std')]
        ax.plot(grouped.index, mean_vals, label=domain, marker='o', markersize=4)
        ax.fill_between(grouped.index, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Coverage')
    ax.set_title('Coverage by Domain')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    # 2. Entropy
    ax = axes[0, 1]
    mean_entropy = grouped[('entropy_avg', 'mean')]
    std_entropy = grouped[('entropy_avg', 'std')]
    ax.plot(grouped.index, mean_entropy, 'b-', marker='s', markersize=4, label='Avg Entropy')
    ax.fill_between(grouped.index, mean_entropy - std_entropy, mean_entropy + std_entropy, alpha=0.2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Entropy (bits)')
    ax.set_title('Average Binary Entropy')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # 3. Mutual Information
    ax = axes[1, 0]
    mean_mi = grouped[('mutual_information', 'mean')]
    std_mi = grouped[('mutual_information', 'std')]
    ax.plot(grouped.index, mean_mi, 'g-', marker='^', markersize=4, label='MI')
    ax.fill_between(grouped.index, mean_mi - std_mi, mean_mi + std_mi, alpha=0.2, color='green')
    ax.set_xlabel('Round')
    ax.set_ylabel('Mutual Information (bits)')
    ax.set_title('Mutual Information Decay')
    ax.grid(True, alpha=0.3)
    
    # 4. Accepted Offers
    ax = axes[1, 1]
    mean_accepted = grouped[('accepted_count', 'mean')]
    std_accepted = grouped[('accepted_count', 'std')]
    lower_err = np.minimum(std_accepted, mean_accepted)
    upper_err = std_accepted
    ax.bar(grouped.index, mean_accepted, yerr=[lower_err, upper_err], alpha=0.7, color='coral', capsize=3)
    ax.set_xlabel('Round')
    ax.set_ylabel('Accepted Offers')
    ax.set_title('Offers Accepted per Round')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Title
    filter_str = []
    if topology:
        filter_str.append(f"topology={topology}")
    if N:
        filter_str.append(f"N={N}")
    filter_label = f" ({', '.join(filter_str)})" if filter_str else " (all configurations)"
    
    n_exps = len(rounds_df[['topology', 'N', 'seed']].drop_duplicates())
    fig.suptitle(f"{title_prefix}Swarm Dynamics{filter_label}\n{n_exps} experiments", fontsize=12)
    plt.tight_layout()
    
    return fig


def plot_mi_decay_by_topology(
    experiment_data: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """Plot MI decay curves grouped by topology.
    
    Args:
        experiment_data: Dictionary containing experiment data
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    from plora.notebook_utils import extract_thesis_sweep_rounds_df
    
    rounds_df = extract_thesis_sweep_rounds_df(experiment_data)
    if rounds_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No thesis_sweep data available", ha='center', va='center')
        return fig
    
    topologies = sorted(rounds_df['topology'].unique())
    fig, axes = plt.subplots(1, len(topologies), figsize=figsize, sharey=True)
    if len(topologies) == 1:
        axes = [axes]
    
    topo_names = {'er': 'Erdős-Rényi', 'ws': 'Watts-Strogatz', 'ba': 'Barabási-Albert'}
    
    for ax, topo in zip(axes, topologies):
        topo_df = rounds_df[rounds_df['topology'] == topo]
        
        # Group by N and round
        for n_val in sorted(topo_df['N'].unique()):
            n_df = topo_df[topo_df['N'] == n_val]
            grouped = n_df.groupby('round')['mutual_information'].agg(['mean', 'std'])
            
            ax.plot(grouped.index, grouped['mean'], label=f'N={n_val}', marker='o', markersize=3)
            ax.fill_between(grouped.index, 
                          grouped['mean'] - grouped['std'], 
                          grouped['mean'] + grouped['std'], 
                          alpha=0.15)
        
        ax.set_xlabel('Round')
        ax.set_title(topo_names.get(topo, topo))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel('Mutual Information (bits)')
    fig.suptitle('MI Decay by Topology and Network Size', fontsize=12)
    plt.tight_layout()
    
    return fig


def plot_convergence_analysis(
    experiment_data: Dict[str, Any],
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """Plot convergence analysis: spectral gap vs t_obs, efficiency distribution.
    
    Args:
        experiment_data: Dictionary containing experiment data
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    from plora.notebook_utils import extract_thesis_sweep_df
    
    sweep_df = extract_thesis_sweep_df(experiment_data)
    if sweep_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No thesis_sweep data available", ha='center', va='center')
        return fig
    
    # Filter to valid convergence data
    valid_df = sweep_df[sweep_df['t_obs'].notna() & (sweep_df['t_pred'] > 0)].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    colors = {'er': '#2ecc71', 'ws': '#3498db', 'ba': '#e74c3c'}
    markers = {'er': 'o', 'ws': 's', 'ba': '^'}
    topo_names = {'er': 'Erdős-Rényi', 'ws': 'Watts-Strogatz', 'ba': 'Barabási-Albert'}
    
    # 1. Spectral Gap vs Observed Convergence Time
    ax = axes[0]
    for topo in sorted(valid_df['topology'].unique()):
        topo_data = valid_df[valid_df['topology'] == topo]
        ax.scatter(topo_data['lambda2'], topo_data['t_obs'], 
                  c=colors.get(topo, 'gray'), marker=markers.get(topo, 'o'),
                  label=topo_names.get(topo, topo), alpha=0.7, s=50)
    ax.set_xlabel('Spectral Gap (λ₂)')
    ax.set_ylabel('Observed Convergence (t_obs)')
    ax.set_title('Spectral Gap vs Convergence Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Predicted vs Observed
    ax = axes[1]
    for topo in sorted(valid_df['topology'].unique()):
        topo_data = valid_df[valid_df['topology'] == topo]
        ax.scatter(topo_data['t_pred'], topo_data['t_obs'],
                  c=colors.get(topo, 'gray'), marker=markers.get(topo, 'o'),
                  label=topo_names.get(topo, topo), alpha=0.7, s=50)
    
    # Add y=x line
    max_val = max(valid_df['t_pred'].max(), valid_df['t_obs'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y = x')
    ax.set_xlabel('Predicted t (spectral bound)')
    ax.set_ylabel('Observed t')
    ax.set_title('Predicted vs Observed Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Efficiency Ratio Distribution
    ax = axes[2]
    for topo in sorted(valid_df['topology'].unique()):
        topo_data = valid_df[valid_df['topology'] == topo]
        ratios = topo_data['efficiency_ratio'].dropna()
        ax.hist(ratios, bins=15, alpha=0.5, label=topo_names.get(topo, topo),
               color=colors.get(topo, 'gray'))
    
    ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.7, label='Ratio = 1.0')
    ax.axvline(x=1.5, color='red', linestyle=':', alpha=0.7, label='Threshold (1.5)')
    ax.set_xlabel('Efficiency Ratio (t_obs / t_pred)')
    ax.set_ylabel('Count')
    ax.set_title('Efficiency Ratio Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Summary stats
    within_bound = (valid_df['efficiency_ratio'] <= 1.5).sum()
    total = len(valid_df)
    pct = 100 * within_bound / total if total > 0 else 0
    
    fig.suptitle(f'Convergence Analysis ({total} experiments, {pct:.1f}% within 1.5× bound)', fontsize=12)
    plt.tight_layout()
    
    return fig


def plot_topology_comparison(
    experiment_data: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot comparison across topologies: convergence speed, efficiency by N.
    
    Args:
        experiment_data: Dictionary containing experiment data
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    from plora.notebook_utils import extract_thesis_sweep_df
    
    sweep_df = extract_thesis_sweep_df(experiment_data)
    if sweep_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No thesis_sweep data available", ha='center', va='center')
        return fig
    
    valid_df = sweep_df[sweep_df['t_obs'].notna()].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    colors = {'er': '#2ecc71', 'ws': '#3498db', 'ba': '#e74c3c'}
    topo_names = {'er': 'Erdős-Rényi', 'ws': 'Watts-Strogatz', 'ba': 'Barabási-Albert'}
    
    # 1. Convergence time by N and topology
    ax = axes[0, 0]
    topologies = sorted(valid_df['topology'].unique())
    sizes = sorted(valid_df['N'].unique())
    width = 0.25
    x = np.arange(len(sizes))
    
    for i, topo in enumerate(topologies):
        topo_data = valid_df[valid_df['topology'] == topo]
        means = [topo_data[topo_data['N'] == n]['t_obs'].mean() for n in sizes]
        stds = [topo_data[topo_data['N'] == n]['t_obs'].std() for n in sizes]
        ax.bar(x + i*width, means, width, yerr=stds, label=topo_names.get(topo, topo),
              color=colors.get(topo, 'gray'), alpha=0.8, capsize=3)
    
    ax.set_xlabel('Network Size (N)')
    ax.set_ylabel('Convergence Time (rounds)')
    ax.set_title('Convergence Time by Topology')
    ax.set_xticks(x + width)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Spectral gap by N and topology
    ax = axes[0, 1]
    for i, topo in enumerate(topologies):
        topo_data = valid_df[valid_df['topology'] == topo]
        means = [topo_data[topo_data['N'] == n]['lambda2'].mean() for n in sizes]
        stds = [topo_data[topo_data['N'] == n]['lambda2'].std() for n in sizes]
        ax.bar(x + i*width, means, width, yerr=stds, label=topo_names.get(topo, topo),
              color=colors.get(topo, 'gray'), alpha=0.8, capsize=3)
    
    ax.set_xlabel('Network Size (N)')
    ax.set_ylabel('Spectral Gap (λ₂)')
    ax.set_title('Spectral Gap by Topology')
    ax.set_xticks(x + width)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Efficiency ratio by topology (box plot)
    ax = axes[1, 0]
    data_for_box = [valid_df[valid_df['topology'] == t]['efficiency_ratio'].dropna() 
                   for t in topologies]
    bp = ax.boxplot(data_for_box, labels=[topo_names.get(t, t) for t in topologies], patch_artist=True)
    for patch, topo in zip(bp['boxes'], topologies):
        patch.set_facecolor(colors.get(topo, 'gray'))
        patch.set_alpha(0.7)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=1.5, color='red', linestyle=':', alpha=0.5)
    ax.set_ylabel('Efficiency Ratio (t_obs / t_pred)')
    ax.set_title('Efficiency by Topology')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_data = []
    for topo in topologies:
        topo_data = valid_df[valid_df['topology'] == topo]
        n_exp = len(topo_data)
        within_bound = (topo_data['efficiency_ratio'] <= 1.5).sum()
        pct_bound = 100 * within_bound / n_exp if n_exp > 0 else 0
        avg_t = topo_data['t_obs'].mean()
        avg_lambda = topo_data['lambda2'].mean()
        summary_data.append([
            topo_names.get(topo, topo),
            n_exp,
            f"{pct_bound:.0f}%",
            f"{avg_t:.1f}",
            f"{avg_lambda:.3f}"
        ])
    
    table = ax.table(
        cellText=summary_data,
        colLabels=['Topology', 'Experiments', '≤1.5× bound', 'Avg t_obs', 'Avg λ₂'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Summary Statistics', pad=20)
    
    fig.suptitle('Topology Comparison Analysis', fontsize=12)
    plt.tight_layout()
    
    return fig


def create_chapter2_plots(
    experiment_data: Dict[str, Any],
    save_path: Path | None = None
) -> list:
    """Create all Chapter 2 (Swarm Simulation) plots.
    
    Args:
        experiment_data: Dictionary containing experiment data
        save_path: Optional path to save figures
        
    Returns:
        List of matplotlib Figure objects
    """
    figures = []
    
    # 1. Overall swarm dynamics (aggregated)
    fig1 = plot_swarm_dynamics(experiment_data, title_prefix="Full Thesis Sweep: ")
    figures.append(('swarm_dynamics_all', fig1))
    
    # 2. MI decay by topology
    fig2 = plot_mi_decay_by_topology(experiment_data)
    figures.append(('mi_decay_by_topology', fig2))
    
    # 3. Convergence analysis
    fig3 = plot_convergence_analysis(experiment_data)
    figures.append(('convergence_analysis', fig3))
    
    # 4. Topology comparison
    fig4 = plot_topology_comparison(experiment_data)
    figures.append(('topology_comparison', fig4))
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        for name, fig in figures:
            fig.savefig(save_path / f"{name}.png", dpi=150, bbox_inches='tight')
            fig.savefig(save_path / f"{name}.pdf", bbox_inches='tight')
    
    return [fig for _, fig in figures]


# Initialize plotting style
setup_plotting_style()
