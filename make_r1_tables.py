import pandas as pd
import numpy as np
from fit_naive import expanding_stats

from utils import compute_oos_r1_score

import os
import argparse

DATE = "20260108"
base_model_subset = [
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
TABLES_DIR = "/home/rproner/Documents/Projects/MacroAtRisk/ResultsTables/"

os.makedirs(RESULTS_DIR, exist_ok=True)

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

    model_subset = ['AR1', benchmark_model] + base_model_subset

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

    # Evaluate forecasts

    for model in models_list:

        print("Evaluating model:", model)

        # Grab model quantile predictions
        model_preds = preds.loc[:, f"{model}_Q{int_quantiles[0]}":f"{model}_Q{int_quantiles[-1]}"]

    # Compute R1
    results = []

    for model in models_list:

        for q in QUANTILES:

            q_int = int(q*100)

            model_q_preds = preds.loc[:, f"{model}_Q{q_int}"]

            r1 = compute_oos_r1_score(
                y_true=actuals.values.flatten(),
                y_pred=model_q_preds.values.flatten(),
                benchmark_pred=naive_preds.loc[TEST_START:TEST_END, f"Expanding_Q{q_int}"].values.flatten(),
                q=q
            )

            results.append({
                'Model': model,
                'Quantile': q,
                'R1': r1
            })


    # Plot quantile forecasts for each model

    # if TEST_START=='1998-01-01' and TEST_END=='2024-12-01':
    #     n_models = len(plot_list)
    #     fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models), sharex=True)  # Create subplots for each model

    #     for ax, model in zip(axes, plot_list):
    #         # Grab model quantile predictions
    #         model_preds = preds.loc[:, f"{model}_Q{int_quantiles[0]}":f"{model}_Q{int_quantiles[-1]}"]

    #         # Plot each quantile prediction for the model
    #         for i, q in enumerate(int_quantiles):
    #             ax.plot(
    #                 actuals.index,
    #                 model_preds.iloc[:, i],
    #                 label=f"{model} Q{q}",
    #                 linestyle='-' if i == len(int_quantiles) // 2 else '--',  # Solid line for median
    #                 alpha=0.7
    #             )

    #         # Plot the naive predictions for comparison
    #         for q in int_quantiles:
    #             ax.plot(
    #                 actuals.index,
    #                 naive_preds.loc[TEST_START:TEST_END, f"Expanding_Q{q}"].values.flatten(),
    #                 label=f"Naive Q{q}",
    #                 linestyle=':',
    #                 alpha=0.8
    #             )

    #         # Plot the actual values
    #         ax.plot(
    #             actuals.index,
    #             actuals.values,
    #             label="Actuals",
    #             color='black',
    #             linewidth=2
    #         )
    #         ax.axvspan('2001-03-01', '2001-11-01', -1,1, color='grey', alpha=0.25)
    #         ax.axvspan('2007-12-01', '2009-06-01', -1,1, color='grey', alpha=0.25)
    #         ax.axvspan('2020-02-01', '2020-04-01', -1,1, color='grey', alpha=0.25)

    #         # Add title and legend for each subplot
    #         ax.set_title(f"Quantile Predictions for {model}")
    #         ax.legend(fontsize='small')

    #     # Adjust layout and save the figure
        
    #     plt.tight_layout()
    #     plt.savefig(f"{RESULTS_DIR}all_models_quantile_forecasts_panes_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_dict[TARGET_IDX]}.png", bbox_inches='tight', dpi=300)
    #     plt.close()

    r1_results_df = pd.DataFrame(results).pivot(index='Model', columns='Quantile', values='R1').reset_index().apply(lambda x: round(x, 1) if x.name!='Model' else x)
    r1_results_df['Mean'] = r1_results_df.loc[:, QUANTILES].mean(axis=1)
    r1_results_df.sort_values('Mean', ascending=False).to_csv(f"{RESULTS_DIR}oos_r1_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_dict[TARGET_IDX]}_{TEST_START}-{TEST_END}.csv", index=False)

    # Make latex table 
    
    r1_results_df = r1_results_df.set_index(['Model'])
    row_order = ['Mean'] + r1_results_df.columns[:-1].tolist()
    r1_report_df = r1_results_df.transpose().loc[row_order, model_subset]
    rename_rows_map = {k: f"Q{int(k*100)}" if k!='Mean' else 'Mean' for k in r1_report_df.index}
    r1_report_df.rename(index=rename_rows_map, inplace=True)
    r1_report_df.columns.name = None
    r1_report_df.index.name = None
    r1_report_df.to_latex(TABLES_DIR + f"r1_{target_dict[TARGET_IDX]}.tex", float_format="%.1f")