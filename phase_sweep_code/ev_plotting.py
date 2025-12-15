"""
Plotting utilities.
Optimized for Seaborn styling.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")

def _save(path):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=140, bbox_inches='tight')
        plt.close()
    return path

def plot_fanchart(traces_df, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"baseline": "steelblue", "subsidy": "darkorange"}
    
    for j, group in enumerate(["baseline", "subsidy"]):
        ax = axes[j]
        gdf = traces_df[traces_df["group"] == group]
        stats = gdf.groupby("time")["X"].agg(mean="mean", q10=lambda x: np.quantile(x,0.1), q90=lambda x: np.quantile(x,0.9))
        
        ax.fill_between(stats.index, stats['q10'], stats['q90'], color=colors[group], alpha=0.2)
        ax.plot(stats.index, stats['mean'], color=colors[group], lw=2)
        ax.set_title(f"{group.capitalize()} Dynamics")
        ax.set_ylim(0, 1.05)
    return _save(out_path)

def plot_spaghetti(traces_df, max_traces=50, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"baseline": "steelblue", "subsidy": "darkorange"}
    
    for j, group in enumerate(["baseline", "subsidy"]):
        ax = axes[j]
        gdf = traces_df[traces_df["group"] == group]
        trials = gdf["trial"].unique()
        if len(trials) > max_traces:
            gdf = gdf[gdf["trial"].isin(trials[:max_traces])]
            
        sns.lineplot(data=gdf, x="time", y="X", units="trial", estimator=None, ax=ax, color=colors[group], alpha=0.15, lw=0.8)
        ax.set_title(f"{group.capitalize()} Traces")
        ax.set_ylim(0, 1.05)
    return _save(out_path)

def plot_phase_plot(phase_df, out_path=None):
    """
    Heatmap of X* over (X0, ratio) with clean axis labels.
    """
    # 1. Create a copy to avoid modifying the original dataframe
    plot_df = phase_df.copy()
    
    # 2. Round values for better visualization
    plot_df['ratio'] = plot_df['ratio'].round(2)
    plot_df['X0'] = plot_df['X0'].round(2)
    
    # 3. Pivot
    pivot = plot_df.pivot(index="ratio", columns="X0", values="X_final").sort_index()
    
    plt.figure(figsize=(9, 7))
    
    # 4. Plot Heatmap
    
    ax = sns.heatmap(
        pivot, 
        cmap="viridis", 
        cbar_kws={'label': 'Final Adoption $X^*$'},
        vmin=0, vmax=1,
        xticklabels=2,
        yticklabels=4
    )
    
    # 5. Formatting
    ax.invert_yaxis() # Put standard Cartesian origin (low values) at the bottom
    plt.title("Phase Diagram: Adoption vs Initial Conditions")
    plt.xlabel("Initial Adoption $X_0$")
    plt.ylabel("Payoff Ratio $a_I / b$")
    
    # Optional: Rotate x-axis labels to be straight if they fit
    plt.xticks(rotation=0) 
    plt.yticks(rotation=0)
    
    return _save(out_path)