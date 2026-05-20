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
    target_idx: int,
    targets_path: Path,
    pred_path: Path,
    fig_dir: Path,
    base_models_subset: list[str] = ['LR','LAS','QRF','QGB','DMQv0c','DMQv1c','DMQv2c'],
    country: str = 'us',
    horizon_in_quarters: int = 4,
    quantiles: list[float] = [0.05, 0.25, 0.50, 0.75, 0.95],
    test_start: str = '1998-01-01',
    test_end: str = '2024-12-01',
    date_str: str = None
    ):
    """
    Generate quantile forecast plots for all models.
    
    Parameters:
    -----------
    target_idx: int
        Target variable index (0: Infl_yoy, 1: IP_yoy, 2: Unrate_yoy)
    targets_path: Path
        Path to targets file
    pred_path: Path
        Path to predictions CSV file
    fig_dir: Path
        Directory to save figures
    base_models_subset: list[str]
        List of base model names to plot
    country: str
        Country code (us/ca)
    horizon_in_quarters: int
        Forecast horizon in quarters
    quantiles: list[float]
        List of quantiles to plot
    test_start: str
        Start date for test set
    test_end: str
        End date for test set
    date_str: str
        Date identifier for results directory
    """
    
    if date_str is None:
        date_str = str(date.today())
    
    os.makedirs(fig_dir, exist_ok=True)
    
    int_quantiles = [int(q*100) for q in quantiles]

    target_dict = {
        0: 'Infl_yoy',
        1: 'IP_yoy',
        2: 'Unrate_yoy'
    }

    if target_idx not in target_dict:
        raise ValueError(f"Invalid target_idx: {target_idx}. Must be 0, 1, or 2.")

    if target_idx == 0:
        benchmark_model = "IAR"
    elif target_idx == 1:
        benchmark_model = "VG"
    else:
        benchmark_model = "UAR"

    plot_list = ['AR1', benchmark_model] + base_models_subset

    # Load targets and predictions
    y_full = pd.read_csv(targets_path, index_col=0, parse_dates=True).loc['1961-01-01':test_end, :]
    preds = pd.read_csv(pred_path, index_col=0, parse_dates=True).loc[test_start:test_end]
    models_list = set([c.split('_')[0] for c in preds.columns if '_' in c])
    
    # Load actuals
    actuals = y_full.loc[test_start:test_end, target_dict[target_idx]]

    # Generate plots for each model
    for model in models_list:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Grab model quantile predictions
        model_preds = preds.loc[:, f"{model}_Q{int_quantiles[0]}":f"{model}_Q{int_quantiles[-1]}"]

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
                actuals.index,
                model_preds.iloc[:, i],
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
            actuals.index,
            actuals.values,
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

        plt.tight_layout()
        plt.savefig(fig_dir / f"{model}_quantile_plot_{country}_{horizon_in_quarters}q_{target_dict[target_idx]}_{test_start}-{test_end}.png", bbox_inches='tight', dpi=300)
        plt.close()

    print(f"Quantile plots generation complete for {target_dict[target_idx]}.")


def make_mean_plots(
    target_idx: int,
    targets_path: Path,
    pred_path: Path,
    fig_dir: Path,
    base_models_subset: list[str] = ['LR','LAS','QRF','QGB','DMQv0c','DMQv1c','DMQv2c'],
    country: str = 'us',
    horizon_in_quarters: int = 4,
    quantiles: list[float] = [0.05, 0.25, 0.50, 0.75, 0.95],
    test_start: str = '1998-01-01',
    test_end: str = '2024-12-01',
    date_str: str = None
    ):
    """
    Generate mean forecast plots for all models.
    
    Parameters:
    -----------
    target_idx: int
        Target variable index (0: Infl_yoy, 1: IP_yoy, 2: Unrate_yoy)
    targets_path: Path
        Path to targets file
    pred_path: Path
        Path to predictions CSV file
    fig_dir: Path
        Directory to save figures
    base_models_subset: list[str]
        List of base model names to plot
    country: str
        Country code (us/ca)
    horizon_in_quarters: int
        Forecast horizon in quarters
    quantiles: list[float]
        List of quantiles to use for mean estimation
    test_start: str
        Start date for test set
    test_end: str
        End date for test set
    date_str: str
        Date identifier for results directory
    """
    
    if date_str is None:
        date_str = str(date.today())
    
    os.makedirs(fig_dir, exist_ok=True)
    
    int_quantiles = [int(q*100) for q in quantiles]

    target_dict = {
        0: 'Infl_yoy',
        1: 'IP_yoy',
        2: 'Unrate_yoy'
    }

    target_name_dict = {
        0: 'Inflation',
        1: 'Industrial Production',
        2: 'Unemployment Rate'
    }

    if target_idx not in target_dict:
        raise ValueError(f"Invalid target_idx: {target_idx}. Must be 0, 1, or 2.")

    if target_idx == 0:
        benchmark_model = "IAR"
    elif target_idx == 1:
        benchmark_model = "VG"
    else:
        benchmark_model = "UAR"

    plot_list = ['AR1', benchmark_model] + base_models_subset

    # Load targets and predictions
    y_full = pd.read_csv(targets_path, index_col=0, parse_dates=True).loc['1961-01-01':test_end, :]
    preds = pd.read_csv(pred_path, index_col=0, parse_dates=True).loc[test_start:test_end]
    models_list = set([c.split('_')[0] for c in preds.columns if '_' in c])
    
    # Load actuals
    actuals = y_full.loc[test_start:test_end, target_dict[target_idx]]

    # Generate plots for each model
    for model in models_list:
        # Grab model quantile predictions
        model_preds = preds.loc[:, f"{model}_Q{int_quantiles[0]}":f"{model}_Q{int_quantiles[-1]}"]

        # Get mean preds
        model_mean_preds = estimate_mean_from_quantiles(model_preds.values)

        # Plot mean preds
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(actuals.index, model_mean_preds, label=f"{model} Mean", color="#7fbfff", linewidth=2)
        ax.plot(actuals.index, actuals, label="Actual", color='black', linewidth=2)
        
        # Shade recession periods
        ax.axvspan('2001-03-01', '2001-11-01', -1, 1, color='grey', alpha=0.25)
        ax.axvspan('2007-12-01', '2009-06-01', -1, 1, color='grey', alpha=0.25)
        ax.axvspan('2020-02-01', '2020-04-01', -1, 1, color='grey', alpha=0.25)
        
        ax.set_title(f"Mean Forecast for {model}")
        ax.set_ylabel(f"Y-o-y log change in {target_name_dict[target_idx]}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"{model}_mean_plot_{country}_{horizon_in_quarters}q_{target_dict[target_idx]}_{test_start}-{test_end}.png", dpi=300)
        plt.close(fig)

    print(f"Mean plots generation complete for {target_dict[target_idx]}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate forecast visualization plots")
    parser.add_argument("--target", type=int, default=0, help="Target index (0: Infl, 1: IP, 2: Unrate)")
    parser.add_argument("--country", type=str, default='us', help="Country code (us/ca)")
    parser.add_argument("--horizon", type=int, default=4, help="forecast horizon in quarters")
    parser.add_argument("--quantiles", type=float, nargs="*", default=[0.05, 0.25, 0.50, 0.75, 0.95], help="list of quantiles")
    parser.add_argument("--test-start", type=str, default="1998-01-01", help="start date for the test set")
    parser.add_argument("--test-end", type=str, default="2024-12-01", help="end date for the test set")
    parser.add_argument("--date", type=str, default=None, help="date identifier for results directory")
    parser.add_argument("--plot-type", type=str, choices=['quantile', 'mean', 'both'], default='both', help="type of plots to generate")

    args = parser.parse_args()
    
    # Parse target file path and other configuration from config
    import yaml
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    TARGET_FILE = config['target_file']
    DATA_DIR = Path(config.get('data_dir', './data/'))
    targets_path = DATA_DIR / TARGET_FILE
    
    if args.plot_type in ['quantile', 'both']:
        make_quantile_plots(
            target_idx=args.target,
            targets_path=targets_path,
            pred_path=Path(f"predictions_{args.country}_{args.horizon}q.csv"),
            fig_dir=Path("results_figures/"),
            country=args.country,
            horizon_in_quarters=args.horizon,
            quantiles=args.quantiles,
            test_start=args.test_start,
            test_end=args.test_end,
            date_str=args.date
        )
    
    if args.plot_type in ['mean', 'both']:
        make_mean_plots(
            target_idx=args.target,
            targets_path=targets_path,
            pred_path=Path(f"predictions_{args.country}_{args.horizon}q.csv"),
            fig_dir=Path("results_figures/"),
            country=args.country,
            horizon_in_quarters=args.horizon,
            quantiles=args.quantiles,
            test_start=args.test_start,
            test_end=args.test_end,
            date_str=args.date
        )
