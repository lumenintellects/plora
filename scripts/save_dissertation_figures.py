#!/usr/bin/env python3
"""
Save all dissertation figures from the experiment analysis notebook.

This script generates and saves all figures used in the dissertation chapters.
Run from the project root: python scripts/save_dissertation_figures.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

from plora.notebook_utils import (
    load_experiment_data,
    extract_value_add_metrics,
    extract_thesis_sweep_df,
    extract_thesis_sweep_rounds_df,
)
from plora.plotting import (
    plot_swarm_dynamics,
    plot_mi_decay_by_topology,
    plot_convergence_analysis,
    plot_topology_comparison,
    create_value_add_summary_plot,
    create_swarm_dynamics_plot,
)


def save_figure(fig, name: str, figures_dir: Path, dpi: int = 300):
    """Save figure in both PNG and PDF formats."""
    png_path = figures_dir / f"{name}.png"
    pdf_path = figures_dir / f"{name}.pdf"
    
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight')
    
    print(f"  Saved: {name}.png, {name}.pdf")
    return png_path, pdf_path


def main():
    # Create figures directory
    figures_dir = project_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("GENERATING DISSERTATION FIGURES")
    print("=" * 60)
    print(f"Output directory: {figures_dir}\n")
    
    # Load experiment data
    print("Loading experiment data...")
    experiment_data = load_experiment_data()
    value_add_df = extract_value_add_metrics(experiment_data)
    sweep_df = extract_thesis_sweep_df(experiment_data)
    thesis_sweep = experiment_data.get('thesis_sweep', [])
    
    print(f"  Thesis sweep: {len(thesis_sweep)} experiments")
    print(f"  Value-add: {len(value_add_df)} experiments\n")
    
    # ==========================================================================
    # CHAPTER 2 / CHAPTER 3: Swarm Dynamics Figures
    # ==========================================================================
    print("--- Chapter 2/3: Swarm Dynamics ---")
    
    # Figure 1: Aggregated Swarm Dynamics (4-panel)
    fig1 = plot_swarm_dynamics(experiment_data, title_prefix="")
    save_figure(fig1, "fig_swarm_dynamics", figures_dir)
    plt.close(fig1)
    
    # Figure 2: MI Decay by Topology
    fig2 = plot_mi_decay_by_topology(experiment_data)
    save_figure(fig2, "fig_mi_decay_by_topology", figures_dir)
    plt.close(fig2)
    
    # Figure 3: Sample Individual Experiment (ER, N=40)
    sample_exp = next(
        (e for e in thesis_sweep 
         if e['topology'] == 'er' and e['N'] == 40 and e['trojan_rate'] == 0.0),
        None
    )
    if sample_exp and 'rounds' in sample_exp:
        fig_sample, _ = create_swarm_dynamics_plot(sample_exp)
        fig_sample.suptitle("Sample Experiment: Erdős-Rényi (N=40)", y=1.02)
        save_figure(fig_sample, "fig_sample_er_n40", figures_dir)
        plt.close(fig_sample)
    
    # ==========================================================================
    # CHAPTER 3: Spectral Diffusion Analysis (RQ2)
    # ==========================================================================
    print("\n--- Chapter 3: Spectral Diffusion (RQ2) ---")
    
    # Figure 4: Convergence Analysis (λ₂ vs t_obs, Predicted vs Observed, Distribution)
    fig4 = plot_convergence_analysis(experiment_data)
    save_figure(fig4, "fig_spectral_convergence", figures_dir)
    plt.close(fig4)
    
    # Figure 5: Topology Comparison
    fig5 = plot_topology_comparison(experiment_data)
    save_figure(fig5, "fig_topology_comparison", figures_dir)
    plt.close(fig5)
    
    # Figure 6: RQ2 4-panel analysis (custom)
    if not sweep_df.empty:
        fig6, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: λ₂ vs t_obs by topology
        ax1 = axes[0, 0]
        colors = {'er': '#2ecc71', 'ws': '#3498db', 'ba': '#e74c3c'}
        markers = {'er': 'o', 'ws': 's', 'ba': '^'}
        topo_names = {'er': 'Erdős-Rényi', 'ws': 'Watts-Strogatz', 'ba': 'Barabási-Albert'}
        for topo in sweep_df['topology'].unique():
            topo_data = sweep_df[sweep_df['topology'] == topo]
            ax1.scatter(topo_data['lambda2'], topo_data['t_obs'], 
                       label=topo_names.get(topo, topo), alpha=0.7, s=50,
                       c=colors.get(topo, 'gray'), marker=markers.get(topo, 'o'))
        ax1.set_xlabel('Spectral Gap (λ₂)')
        ax1.set_ylabel('Observed Diffusion Time (rounds)')
        ax1.set_title('(a) Spectral Gap vs Diffusion Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Predicted vs Observed
        ax2 = axes[0, 1]
        for topo in sweep_df['topology'].unique():
            topo_data = sweep_df[(sweep_df['topology'] == topo) & (sweep_df['t_obs'].notna())]
            ax2.scatter(topo_data['t_pred'], topo_data['t_obs'], 
                       label=topo_names.get(topo, topo), alpha=0.7, s=50,
                       c=colors.get(topo, 'gray'), marker=markers.get(topo, 'o'))
        max_t = max(sweep_df['t_pred'].max(), sweep_df['t_obs'].dropna().max())
        ax2.plot([0, max_t], [0, max_t], 'k--', alpha=0.5, label='Perfect prediction')
        ax2.set_xlabel('Predicted Diffusion Time')
        ax2.set_ylabel('Observed Diffusion Time')
        ax2.set_title('(b) Theory vs Observation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Efficiency ratio distribution
        ax3 = axes[1, 0]
        for topo in sweep_df['topology'].unique():
            topo_data = sweep_df[sweep_df['topology'] == topo]
            ratios = topo_data['efficiency_ratio'].dropna()
            ax3.hist(ratios, bins=15, alpha=0.5, label=topo_names.get(topo, topo),
                    color=colors.get(topo, 'gray'))
        ax3.axvline(x=1.0, color='black', linestyle='--', alpha=0.7, label='Ratio = 1.0')
        ax3.axvline(x=1.5, color='red', linestyle=':', alpha=0.7, label='Threshold (1.5)')
        ax3.set_xlabel('Efficiency Ratio (t_obs / t_pred)')
        ax3.set_ylabel('Count')
        ax3.set_title('(c) Efficiency Ratio Distribution')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Summary by network size
        ax4 = axes[1, 1]
        sizes = sorted(sweep_df['N'].unique())
        width = 0.25
        x = np.arange(len(sizes))
        for i, topo in enumerate(['er', 'ws', 'ba']):
            topo_data = sweep_df[sweep_df['topology'] == topo]
            means = [topo_data[topo_data['N'] == n]['t_obs'].mean() for n in sizes]
            stds = [topo_data[topo_data['N'] == n]['t_obs'].std() for n in sizes]
            ax4.bar(x + i*width, means, width, yerr=stds, 
                   label=topo_names.get(topo, topo),
                   color=colors.get(topo, 'gray'), alpha=0.8, capsize=3)
        ax4.set_xlabel('Network Size (N)')
        ax4.set_ylabel('Convergence Time (rounds)')
        ax4.set_title('(d) Diffusion Time by Network Size')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(sizes)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_figure(fig6, "fig_rq2_spectral_analysis", figures_dir)
        plt.close(fig6)
    
    # ==========================================================================
    # CHAPTER 4: Value-Add Experiments (RQ1)
    # ==========================================================================
    print("\n--- Chapter 4: Value-Add (RQ1) ---")
    
    # Figure 7: Value-Add Summary by Domain
    fig7, _ = create_value_add_summary_plot({'value_add': experiment_data.get('value_add', [])})
    save_figure(fig7, "fig_value_add_summary", figures_dir)
    plt.close(fig7)
    
    # Figure 8: Cross-Domain Transfer Heatmap
    if not value_add_df.empty:
        domains = ['arithmetic', 'legal', 'medical']
        fig8, ax = plt.subplots(figsize=(8, 6))
        
        transfer_matrix = []
        for src in domains:
            row = []
            src_subset = value_add_df[value_add_df['domain'] == src]
            for tgt in domains:
                if src == tgt:
                    effect = src_subset['trained_delta_mean'].mean()
                else:
                    col = f'cross_{tgt}_delta_mean'
                    effect = src_subset[col].mean() if col in src_subset.columns else np.nan
                row.append(effect)
            transfer_matrix.append(row)
        transfer_matrix = np.array(transfer_matrix, dtype=float)
        
        # Use diverging colormap centered at 0
        vmax = np.nanmax(np.abs(transfer_matrix))
        im = ax.imshow(transfer_matrix, cmap='RdYlGn_r', vmin=-vmax, vmax=vmax)
        
        ax.set_xticks(range(len(domains)))
        ax.set_yticks(range(len(domains)))
        ax.set_xticklabels([d.capitalize() for d in domains])
        ax.set_yticklabels([d.capitalize() for d in domains])
        ax.set_title('Cross-Domain Transfer Effects (ΔNLL)')
        ax.set_xlabel('Target Domain')
        ax.set_ylabel('Source Domain (Adapter Trained On)')
        
        # Annotate cells
        for i in range(len(domains)):
            for j in range(len(domains)):
                val = transfer_matrix[i, j]
                if np.isnan(val):
                    label = "—"
                else:
                    label = f"{val:.3f}"
                text_color = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=11, fontweight='bold')
        
        plt.colorbar(im, label='ΔNLL (negative = improvement)')
        plt.tight_layout()
        save_figure(fig8, "fig_cross_domain_transfer", figures_dir)
        plt.close(fig8)
    
    # Figure 9: Effect Size Comparison (Trained vs Placebo)
    if not value_add_df.empty:
        fig9, ax = plt.subplots(figsize=(10, 6))
        
        domains = ['arithmetic', 'legal', 'medical']
        x = np.arange(len(domains))
        width = 0.35
        
        trained_means = []
        trained_stds = []
        placebo_means = []
        placebo_stds = []
        
        for domain in domains:
            domain_data = value_add_df[value_add_df['domain'] == domain]
            trained_means.append(domain_data['trained_delta_mean'].mean())
            trained_stds.append(domain_data['trained_delta_mean'].std())
            placebo_means.append(domain_data['placebo_a_delta_mean'].mean())
            placebo_stds.append(domain_data['placebo_a_delta_mean'].std())
        
        bars1 = ax.bar(x - width/2, trained_means, width, yerr=trained_stds, 
                      label='Trained Adapter', color='#2E86AB', capsize=5)
        bars2 = ax.bar(x + width/2, placebo_means, width, yerr=placebo_stds,
                      label='Placebo (Random)', color='#A23B72', capsize=5)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('ΔNLL (negative = improvement)')
        ax.set_xlabel('Domain')
        ax.set_title('Trained Adapters vs Placebo Controls')
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in domains])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_figure(fig9, "fig_trained_vs_placebo", figures_dir)
        plt.close(fig9)
    
    # ==========================================================================
    # CHAPTER 5: Security Gate (RQ3)
    # ==========================================================================
    print("\n--- Chapter 5: Security Gate (RQ3) ---")
    
    if not sweep_df.empty:
        # Build security data
        sweep_security = []
        for exp in thesis_sweep:
            gate = exp.get('gate', {})
            sweep_security.append({
                'topology': exp.get('topology', 'unknown'),
                'N': exp.get('N', 0),
                'trojan_rate': exp.get('trojan_rate', 0),
                'rejected_clean': gate.get('rejected_clean_total', 0),
                'accepted_trojan': gate.get('accepted_trojan_total', 0),
                'accepted_clean': gate.get('accepted_clean_total', 0),
                'rejected_trojan': gate.get('rejected_trojan_total', 0),
            })
        sec_df = pd.DataFrame(sweep_security)
        
        # Figure 10: Security Gate Performance
        fig10, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # Compute per-topology rates
        topo_names = {'er': 'Erdős-Rényi', 'ws': 'Watts-Strogatz', 'ba': 'Barabási-Albert'}
        topos = ['er', 'ws', 'ba']
        fp_rates = []
        fn_rates = []
        clean_counts = []
        trojan_counts = []
        
        for topo in topos:
            topo_data = sec_df[sec_df['topology'] == topo]
            clean_offers = topo_data['accepted_clean'].sum() + topo_data['rejected_clean'].sum()
            trojan_offers = topo_data['accepted_trojan'].sum() + topo_data['rejected_trojan'].sum()
            fp_rates.append(topo_data['rejected_clean'].sum() / clean_offers if clean_offers > 0 else 0)
            fn_rates.append(topo_data['accepted_trojan'].sum() / trojan_offers if trojan_offers > 0 else 0)
            clean_counts.append(clean_offers)
            trojan_counts.append(trojan_offers)
        
        # Plot 1: FP/FN by topology
        x = np.arange(len(topos))
        width = 0.35
        axes[0].bar(x - width/2, fp_rates, width, label='FP Rate', color='#e74c3c', alpha=0.8)
        axes[0].bar(x + width/2, fn_rates, width, label='FN Rate', color='#3498db', alpha=0.8)
        axes[0].axhline(y=0.10, color='red', linestyle='--', linewidth=2, label='Threshold (0.10)')
        axes[0].set_xlabel('Topology')
        axes[0].set_ylabel('Error Rate')
        axes[0].set_title('(a) Security Gate by Topology')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([topo_names[t] for t in topos], rotation=15)
        axes[0].legend()
        axes[0].set_ylim(0, 0.15)
        
        # Plot 2: By network size
        sizes = sorted(sec_df['N'].unique())
        fp_by_size = []
        fn_by_size = []
        for N in sizes:
            n_data = sec_df[sec_df['N'] == N]
            clean_offers = n_data['accepted_clean'].sum() + n_data['rejected_clean'].sum()
            trojan_offers = n_data['accepted_trojan'].sum() + n_data['rejected_trojan'].sum()
            fp_by_size.append(n_data['rejected_clean'].sum() / clean_offers if clean_offers > 0 else 0)
            fn_by_size.append(n_data['accepted_trojan'].sum() / trojan_offers if trojan_offers > 0 else 0)
        
        x = np.arange(len(sizes))
        axes[1].bar(x - width/2, fp_by_size, width, label='FP Rate', color='#e74c3c', alpha=0.8)
        axes[1].bar(x + width/2, fn_by_size, width, label='FN Rate', color='#3498db', alpha=0.8)
        axes[1].axhline(y=0.10, color='red', linestyle='--', linewidth=2, label='Threshold')
        axes[1].set_xlabel('Network Size (N)')
        axes[1].set_ylabel('Error Rate')
        axes[1].set_title('(b) Security Gate by Network Size')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([str(s) for s in sizes])
        axes[1].legend()
        axes[1].set_ylim(0, 0.15)
        
        # Plot 3: Offer counts summary
        total_clean = sum(clean_counts)
        total_trojan = sum(trojan_counts)
        total_rejected_clean = sec_df['rejected_clean'].sum()
        total_accepted_trojan = sec_df['accepted_trojan'].sum()
        
        categories = ['Clean\nOffers', 'Trojan\nOffers']
        correct = [total_clean - total_rejected_clean, total_trojan - total_accepted_trojan]
        errors = [total_rejected_clean, total_accepted_trojan]
        
        x = np.arange(len(categories))
        axes[2].bar(x, correct, label='Correctly Handled', color='#2ecc71', alpha=0.8)
        axes[2].bar(x, errors, bottom=correct, label='Errors', color='#e74c3c', alpha=0.8)
        axes[2].set_ylabel('Number of Offers')
        axes[2].set_title('(c) Security Gate Decision Summary')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(categories)
        axes[2].legend()
        
        # Add count annotations
        for i, (c, e) in enumerate(zip(correct, errors)):
            axes[2].text(i, c/2, f'{c:,}', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
            if e > 0:
                axes[2].text(i, c + e/2, f'{e:,}', ha='center', va='center', fontsize=10, color='white')
        
        plt.tight_layout()
        save_figure(fig10, "fig_security_gate_performance", figures_dir)
        plt.close(fig10)
    
    # ==========================================================================
    # CHAPTER 5: Scalability Analysis
    # ==========================================================================
    print("\n--- Chapter 5: Scalability ---")
    
    if not sweep_df.empty:
        fig11, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        colors = {'er': '#2ecc71', 'ws': '#3498db', 'ba': '#e74c3c'}
        topo_names = {'er': 'Erdős-Rényi', 'ws': 'Watts-Strogatz', 'ba': 'Barabási-Albert'}
        
        # Plot 1: Diffusion time vs Network Size
        ax = axes[0, 0]
        for topo in ['er', 'ws', 'ba']:
            topo_data = sweep_df[sweep_df['topology'] == topo]
            grouped = topo_data.groupby('N')['t_obs'].agg(['mean', 'std']).reset_index()
            ax.errorbar(grouped['N'], grouped['mean'], yerr=grouped['std'], 
                       label=topo_names[topo], marker='o', capsize=3, color=colors[topo])
        ax.set_xlabel('Network Size (N)')
        ax.set_ylabel('Observed Diffusion Time (rounds)')
        ax.set_title('(a) Diffusion Time vs Network Size')
        ax.legend()
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Spectral Gap vs Network Size
        ax = axes[0, 1]
        for topo in ['er', 'ws', 'ba']:
            topo_data = sweep_df[sweep_df['topology'] == topo]
            grouped = topo_data.groupby('N')['lambda2'].agg(['mean', 'std']).reset_index()
            ax.errorbar(grouped['N'], grouped['mean'], yerr=grouped['std'],
                       label=topo_names[topo], marker='s', capsize=3, color=colors[topo])
        ax.set_xlabel('Network Size (N)')
        ax.set_ylabel('Spectral Gap (λ₂)')
        ax.set_title('(b) Spectral Gap by Network Size')
        ax.legend()
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Efficiency ratio boxplot
        ax = axes[1, 0]
        data_for_box = [sweep_df[sweep_df['topology'] == t]['efficiency_ratio'].dropna() 
                       for t in ['er', 'ws', 'ba']]
        bp = ax.boxplot(data_for_box, labels=[topo_names[t] for t in ['er', 'ws', 'ba']], 
                       patch_artist=True)
        for patch, topo in zip(bp['boxes'], ['er', 'ws', 'ba']):
            patch.set_facecolor(colors[topo])
            patch.set_alpha(0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect prediction')
        ax.axhline(y=1.5, color='red', linestyle=':', alpha=0.5, label='Threshold')
        ax.set_ylabel('Efficiency Ratio (t_obs / t_pred)')
        ax.set_title('(c) Efficiency by Topology')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Correlation heatmap
        ax = axes[1, 1]
        numeric_cols = ['N', 't_obs', 't_pred', 'lambda2', 'efficiency_ratio']
        valid_for_corr = sweep_df[numeric_cols].dropna()
        if len(valid_for_corr) >= 2:
            correlations = valid_for_corr.corr()
            im = ax.imshow(correlations, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            labels = ['N', 't_obs', 't_pred', 'λ₂', 'ratio']
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)
            ax.set_title('(d) Correlation Matrix')
            
            # Annotate
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    val = correlations.iloc[i, j]
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)
            
            plt.colorbar(im, ax=ax, label='Pearson r')
        
        plt.tight_layout()
        save_figure(fig11, "fig_scalability_analysis", figures_dir)
        plt.close(fig11)
    
    # ==========================================================================
    # CHAPTER 5: Convergence Analysis
    # ==========================================================================
    print("\n--- Chapter 5: Convergence ---")
    
    convergence_path = project_root / 'results' / 'alt_train_merge' / 'convergence.json'
    if convergence_path.exists():
        with open(convergence_path) as f:
            convergence_data = json.load(f)
        
        deltas = convergence_data.get('param_delta_fro', [])
        
        if len(deltas) >= 2:
            fig12, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            cycles = list(range(1, len(deltas) + 1))
            
            # Plot 1: Delta over cycles
            ax1 = axes[0]
            ax1.plot(cycles, deltas, 'bo-', linewidth=2, markersize=10)
            ax1.set_xlabel('Cycle Transition')
            ax1.set_ylabel('Parameter Delta (Frobenius Norm)')
            ax1.set_title('(a) Convergence of Alternating Train-Merge')
            ax1.set_xticks(cycles)
            ax1.set_xticklabels([f'{i}→{i+1}' for i in range(len(deltas))])
            ax1.grid(True, alpha=0.3)
            
            # Annotate reduction
            reduction = (1 - deltas[-1] / deltas[0]) * 100
            ax1.annotate(f'{reduction:.0f}% reduction', 
                        xy=(len(deltas), deltas[-1]), 
                        xytext=(len(deltas)-1, deltas[0]*0.6),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                        fontsize=12, color='green', fontweight='bold')
            
            # Plot 2: Log scale
            ax2 = axes[1]
            ax2.semilogy(cycles, deltas, 'ro-', linewidth=2, markersize=10)
            ax2.set_xlabel('Cycle Transition')
            ax2.set_ylabel('Parameter Delta (log scale)')
            ax2.set_title('(b) Exponential Decay Analysis')
            ax2.set_xticks(cycles)
            ax2.set_xticklabels([f'{i}→{i+1}' for i in range(len(deltas))])
            ax2.grid(True, alpha=0.3)
            
            # Add fit line
            try:
                from scipy.optimize import curve_fit
                def exp_decay(x, a, b):
                    return a * np.exp(-b * x)
                x_data = np.array(cycles)
                y_data = np.array(deltas)
                popt, _ = curve_fit(exp_decay, x_data, y_data, p0=[deltas[0], 0.5], maxfev=1000)
                x_fit = np.linspace(1, len(deltas), 100)
                y_fit = exp_decay(x_fit, *popt)
                ax2.plot(x_fit, y_fit, 'g--', linewidth=2, label=f'Fit: a·e^(-bt)\nHalf-life: {np.log(2)/popt[1]:.2f} cycles')
                ax2.legend()
            except:
                pass
            
            plt.tight_layout()
            save_figure(fig12, "fig_convergence_analysis", figures_dir)
            plt.close(fig12)
    
    # ==========================================================================
    # CHAPTER 5: Statistical Robustness
    # ==========================================================================
    print("\n--- Chapter 5: Statistical Robustness ---")
    
    if not value_add_df.empty:
        fig13, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        # Plot 1: Effect size distribution
        ax = axes[0]
        trained = value_add_df['trained_delta_mean'].dropna()
        placebo = value_add_df['placebo_a_delta_mean'].dropna()
        ax.hist(trained, bins=20, alpha=0.7, label='Trained', color='#2E86AB')
        ax.hist(placebo, bins=20, alpha=0.7, label='Placebo', color='#A23B72')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('ΔNLL')
        ax.set_ylabel('Count')
        ax.set_title('(a) ΔNLL Distribution: Trained vs Placebo')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: P-value distribution
        ax = axes[1]
        p_values = value_add_df['trained_wilcoxon_p'].dropna()
        ax.hist(p_values, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        ax.axvline(x=0.05/81, color='orange', linestyle=':', linewidth=2, label='Bonferroni α')
        ax.set_xlabel('p-value')
        ax.set_ylabel('Count')
        ax.set_title('(b) P-value Distribution (Trained Adapters)')
        ax.legend()
        ax.set_xlim(0, max(0.1, p_values.max()))
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: CV by configuration
        ax = axes[2]
        CV_TARGET = 0.15
        cv_values = []
        cv_labels = []
        for (domain, rank, scheme), group in value_add_df.groupby(['domain', 'rank', 'scheme']):
            if len(group) > 1:
                deltas = group['trained_delta_mean'].dropna()
                if len(deltas) > 1 and deltas.mean() != 0:
                    cv = deltas.std(ddof=1) / abs(deltas.mean())
                    cv_values.append(cv)
                    cv_labels.append(f"{domain[:3]}-r{rank}-{scheme[:3]}")
        
        colors = ['#2ecc71' if cv < CV_TARGET else '#e74c3c' for cv in cv_values]
        ax.barh(range(len(cv_values)), cv_values, color=colors, alpha=0.7)
        ax.axvline(x=CV_TARGET, color='red', linestyle='--', linewidth=2, label=f'Target CV = {CV_TARGET}')
        ax.set_xlabel('Coefficient of Variation')
        ax.set_ylabel('Configuration')
        ax.set_title('(c) Reproducibility Across Seeds')
        ax.set_yticks(range(len(cv_values)))
        ax.set_yticklabels(cv_labels, fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        save_figure(fig13, "fig_statistical_robustness", figures_dir)
        plt.close(fig13)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)
    
    # List all generated figures
    figures = sorted(figures_dir.glob("*.png"))
    print(f"\nGenerated {len(figures)} figures:")
    for fig_path in figures:
        print(f"  - {fig_path.name}")
    
    print(f"\nFigures saved to: {figures_dir}")
    print("\nTo use in dissertation chapters, reference figures as:")
    print("  ![Caption](figures/fig_name.png)")


if __name__ == '__main__':
    main()

