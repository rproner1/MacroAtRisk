"""
Consolidated figure generation module for quantile and mean forecast plots.
"""
import pandas as pd
import numpy as np
from src.train.fit_naive import expanding_stats
from src.utils.evaluation import estimate_mean_from_quantiles
import matplotlib.pyplot as plt

import os
import argparse
from pathlib import Path
from datetime import date


def make_quantile_plots(
    y_true: pd.Series,
    y_pred: pd.DataFrame,
    fig_dir: Path,
    models_to_plot: list[str],
    quantiles: list[float],
    target_name: str,
    country: str = 'us',
    horizon_in_quarters: int = 4
):
    """
    Generate quantile forecast plots for all models in `models_to_plot`
    """
    
    os.makedirs(fig_dir, exist_ok=True)
    
    int_quantiles = [int(q*100) for q in quantiles]

    # Generate plots for each model
    for model in models_to_plot:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Grab model quantile predictions
        model_cols = [c for c in y_pred.columns if c.startswith(f'{model}_')]
        model_preds = y_pred.loc[:, model_cols]

        # Plot each quantile prediction for the model
        cmap = 'RdYlBu_r'
        cmap_obj = plt.get_cmap(cmap)
        n_q = len(int_quantiles)

        if n_q > 1:
            colors = [cmap_obj(i / (n_q - 1)) for i in range(n_q)]
        else:
            colors = [cmap_obj(0.5)]
        
        ax.set_prop_cycle(color=colors)

        for i, q in enumerate(int_quantiles):

            line_color = colors[i]
            q_tail = 25
            if q == 50 or q == q_tail:
                r, g, b, a = line_color
                darker_color = (r*0.5, g*0.5, b*0.5, a)
                line_color = darker_color

            ax.plot(
                y_true.index,
                model_preds.loc[:, f'{model}_Q{q}'],
                label=f"{model} Q{q}",
                color=line_color,
                linestyle='-' if i == len(int_quantiles) // 2 else '--',
                alpha=0.7
            )

        # Fill the area between quantiles
        x = model_preds.index

        outer_low_col = f"{model}_Q5"
        outer_high_col = f"{model}_Q95"
        inner_low_col = f"{model}_Q25"
        inner_high_col = f"{model}_Q75"

        # Outer 5-95 band (lighter blue)
        if outer_low_col in model_preds.columns and outer_high_col in model_preds.columns:
            ax.fill_between(
                x,
                model_preds[outer_low_col],
                model_preds[outer_high_col],
                color="#cfe8ff",
                alpha=0.25,
                zorder=0,
                interpolate=True
            )

        # Inner 25-75 band (darker blue)
        if inner_low_col in model_preds.columns and inner_high_col in model_preds.columns:
            ax.fill_between(
                x,
                model_preds[inner_low_col],
                model_preds[inner_high_col],
                color="#7fbfff",
                alpha=0.35,
                zorder=1,
                interpolate=True
            )

        # Plot the actual values
        ax.plot(
            y_true.index,
            y_true.values,
            label="Actuals",
            color='black',
            linewidth=2
        )
        
        # Shade recession periods
        ax.axvspan('2001-03-01', '2001-11-01', -1, 1, color='grey', alpha=0.25)
        ax.axvspan('2007-12-01', '2009-06-01', -1, 1, color='grey', alpha=0.25)
        ax.axvspan('2020-02-01', '2020-04-01', -1, 1, color='grey', alpha=0.25)

        # Add title and legend
        ax.set_title(f"Quantile Predictions for {model}")
        ax.legend(fontsize='small')


        fig_name = (
            f'{model}_quantile_plot_{country}_{horizon_in_quarters}q_'
            f'{target_name}.png'
        )
        plt.tight_layout()
        plt.savefig(
            fig_dir / fig_name, 
            bbox_inches='tight', 
            dpi=300
        )
        plt.close()



def make_mean_plots(
    y_true: pd.Series,
    y_pred: pd.DataFrame,
    fig_dir: Path,
    models_to_plot: list[str],
    target_name: str,
    country: str = 'us',
    horizon_in_quarters: int = 4
):
    """
    Generate mean forecast plots for models in `models_to_plot`.
    """
    
    os.makedirs(fig_dir, exist_ok=True)
    
    # Generate plots for each model
    for model in models_to_plot:

        # Plot mean preds
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            y_true.index, 
            y_pred.loc[:, model], 
            label=f"{model} forecast", 
            color="#7fbfff", 
            linewidth=2
        )
        ax.plot(
            y_true.index,
            y_true, 
            label="Actual", 
            color='black', 
            linewidth=2
        )
        
        # Shade recession periods
        ax.axvspan('2001-03-01', '2001-11-01', -1, 1, color='grey', alpha=0.25)
        ax.axvspan('2007-12-01', '2009-06-01', -1, 1, color='grey', alpha=0.25)
        ax.axvspan('2020-02-01', '2020-04-01', -1, 1, color='grey', alpha=0.25)
        
        ax.legend()
        plt.tight_layout()

        fig_name  = (
            f"{model}_mean_plot_{country}_{horizon_in_quarters}q"
            f"_{target_name}.png"
        )
        plt.savefig(fig_dir / fig_name, dpi=300)
        plt.close(fig)
