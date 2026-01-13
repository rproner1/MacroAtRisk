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

target_name_dict = {
    0: 'Inflation',
    1: 'Industrial Production',
    2: 'Unemployment Rate'
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
         
        # Grab model quantile predictions
        model_preds = preds.loc[:, f"{model}_Q{int_quantiles[0]}":f"{model}_Q{int_quantiles[-1]}"]

        # Get mean preds
        model_mean_preds = estimate_mean_from_quantiles(model_preds)

        # Plot mean preds
        fig, ax = plt.subplots()
        ax.plot(actuals.index, model_mean_preds, label=f"{model} Mean", color="#7fbfff")
        ax.plot(actuals.index, actuals, label="Actual", color='black')
        ax.axvspan('2001-03-01', '2001-11-01', -1,1, color='grey', alpha=0.25)
        ax.axvspan('2007-12-01', '2009-06-01', -1,1, color='grey', alpha=0.25)
        ax.axvspan('2020-02-01', '2020-04-01', -1,1, color='grey', alpha=0.25)
        ax.set_title(f"Mean Forecast for {model}")
        ax.set_ylabel(f"Y-o-y log change in {target_name_dict[TARGET_IDX]}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}{model}_mean_plot_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_dict[TARGET_IDX]}.png")
        plt.close(fig)