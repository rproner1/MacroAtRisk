import pandas as pd
import numpy as np
from fit_naive import expanding_stats

from utils import compute_oos_r1_score, compute_oos_r2_score, estimate_mean_from_quantiles
import matplotlib.pyplot as plt

import os
import argparse
from datetime import datetime

DATE = "20260108"
base_plot_list = [
    'LR',
    'LAS',
    'QRF',
    'QGB',
    'DMQv0c',
    'DMQv1c',
    'DMQv2c'
]
PRED_DIR = f"/home/rproner/Documents/Projects/MacroAtRisk/Predictions/"
NAIVE_PRED_DIR = f"/home/rproner/Documents/Projects/MacroAtRisk/USNaivePredictions/"
RESULTS_DIR = f"/home/rproner/Documents/Projects/MacroAtRisk/Results/{DATE}/"
FIG_DIR = "/home/rproner/Documents/Projects/MacroAtRisk/ResultsFigures/"

os.makedirs(FIG_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Evaluate forecasts")
# parser.add_argument("--target", type=int, required=True, help="target variable index")
parser.add_argument("--country", type=str, default='us', help="Country code (us/ca)")
parser.add_argument("--horizon", type=int, default=4, help="forecast horizon in quarters")
parser.add_argument("--quantiles", type=float, nargs="*", default=[0.05,0.25,0.50,0.75,0.95], help="list of quantiles to predict")
parser.add_argument("--test-start", type=str, default="1998-01-01", help="start date for the test set: Tech bubble 2001-03-01; GFC 2007-12-01; Covid: 2020-02-1")
parser.add_argument("--test-end", type=str, default="2024-12-01", help="end date for the test set: Tech bubble 2001-11-01; GFC: 2009-06-01; Covid: 2024-12-01")
parser.add_argument("--models-to-plot", type=str, nargs="*", default=[], help="list of models to plot (if empty, plot none)")

args = parser.parse_args()
# TARGET_IDX = args.target
COUNTRY = args.country      
HORIZON_IN_QUARTERS = args.horizon
QUANTILES = args.quantiles
TEST_START = args.test_start
TEST_END = args.test_end
MODELS_TO_PLOT = args.models_to_plot

int_quantiles = [int(q*100) for q in QUANTILES]

target_dict = {
    0: 'Infl_yoy',
    1: 'IP_yoy',
    2: 'Unrate_yoy'
}

TEST_START = '1998-01-01'
TEST_END = '2024-12-01'

for TARGET_IDX in [0,1,2]:

    if TARGET_IDX==0:
        benchmark_model = "IAR"
    elif TARGET_IDX==1:
        benchmark_model = "VG"
    elif TARGET_IDX==2:
        benchmark_model = "UAR"

    plot_list = ['AR1', benchmark_model] + base_plot_list

    # Naive rolling mean and quantile predictions for computing out-of-sample R1 and R2
    y_full = pd.read_csv(f'/home/rproner/Documents/Data/MacroAtRisk/{COUNTRY}_targets_1961-01--2024-12.csv', index_col=0, parse_dates=True).loc['1961-01-01':'2024-12-01', :]
    naive_preds = expanding_stats(y_full, col=target_dict[TARGET_IDX], quantiles=[int(q*100) for q in QUANTILES], lag=(3*HORIZON_IN_QUARTERS + 1)).loc[TEST_START:TEST_END]

    naive_mean_test = naive_preds.loc[:, "Expanding_Mean"]

    # Load predictions
    preds = pd.read_csv(f"{PRED_DIR}all_models_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_dict[TARGET_IDX]}.csv", index_col=0, parse_dates=True).loc[TEST_START:TEST_END]
    models_list = set([c.split('_')[0] for c in preds.columns if '_' in c])

    # Load actuals
    actuals = pd.read_csv(f"/home/rproner/Documents/Data/MacroAtRisk/{COUNTRY}_targets_1961-01--2024-12.csv", index_col=0, parse_dates=True)
    actuals = actuals.loc[TEST_START:TEST_END, target_dict[TARGET_IDX]]

    for model in models_list:
        fig, ax = plt.subplots()  

        # Grab model quantile predictions
        model_preds = preds.loc[:, f"{model}_Q{int_quantiles[0]}":f"{model}_Q{int_quantiles[-1]}"]

        # Plot each quantile prediction for the model

        # keep a sequential cmap for the earlier branch

        cmap = 'RdYlBu' if TARGET_IDX == 1 else 'RdYlBu_r'
        cmap = plt.get_cmap(cmap)
        n_q = len(int_quantiles)

        if n_q > 1:
            colors = [cmap(i / (n_q - 1)) for i in range(n_q)]
        else:
            colors = [cmap(0.5)]
        
        ax.set_prop_cycle(color=colors)

        for i, q in enumerate(int_quantiles):

            line_color = colors[i]
            if q == 50 or q == 25:
                r,g,b,a = line_color
                darker_color = (r*0.5, g*0.5, b*0.5, a)
                line_color = darker_color

            ax.plot(
                actuals.index,
                model_preds.iloc[:, i],
                label=f"{model} Q{q}",
                color=line_color,
                linestyle='-' if i == len(int_quantiles) // 2 else '--',  # Solid line for median
                alpha=0.7
            )
            # Fill the area between Q5 and Q95 (outer band) and between Q25 and Q75 (inner band)
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
                color="#cfe8ff",  # light blue
                alpha=0.25,
                zorder=0,
                interpolate=True
                )

            # Inner 25-75 band (darker blue, drawn on top of outer band)
            if inner_low_col in model_preds.columns and inner_high_col in model_preds.columns:
                ax.fill_between(
                x,
                model_preds[inner_low_col],
                model_preds[inner_high_col],
                color="#7fbfff",  # darker blue
                alpha=0.35,
                zorder=1,
                interpolate=True
                )
        # Plot the naive predictions for comparison
        # for q in int_quantiles:
        #     ax.plot(
        #         actuals.index,
        #         naive_preds.loc[TEST_START:TEST_END, f"Expanding_Q{q}"].values.flatten(),
        #         label=f"Naive Q{q}",
        #         linestyle=':',
        #         alpha=0.8
        #     )

        # Plot the actual values
        ax.plot(
            actuals.index,
            actuals.values,
            label="Actuals",
            color='black',
            linewidth=2
        )
        ax.axvspan('2001-03-01', '2001-11-01', -1,1, color='grey', alpha=0.25)
        ax.axvspan('2007-12-01', '2009-06-01', -1,1, color='grey', alpha=0.25)
        ax.axvspan('2020-02-01', '2020-04-01', -1,1, color='grey', alpha=0.25)

        # Add title and legend for each subplot
        ax.set_title(f"Quantile Predictions for {model}")
        ax.legend(fontsize='small')

        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}{model}_quantile_plot_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_dict[TARGET_IDX]}.png", bbox_inches='tight', dpi=300)
        plt.close()