"""
Results Generation Entry Point
Combines predictions from all models and generates evaluation tables.
"""
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import yaml
import argparse
from pathlib import Path
from datetime import date
from src.eval.eval_utils import (
    concat_preds,
    get_r1_results_df,
    get_r2_results_df,
    get_mean_preds
)
from src.figures.make_figures import make_quantile_plots, make_mean_plots
load_dotenv()

# ----- Configuration -----
with open("./config/eval_config.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("./config/eval_config.yaml", "r") as f:
    eval_config = yaml.safe_load(f)

parser = argparse.ArgumentParser(description="Generate results and tables")

parser.add_argument(
    '--concat-preds',
    action='store_true',
    help='Whether predictions should be concatenated. Must be run once.'
)
parser.add_argument(
    '--targets', 
    type=str,
    nargs='+',
    default=['Infl_yoy', 'IP_yoy', 'Unrate_yoy'],
    help="Target(s) to evaluate: 'all' (default), 'Infl', 'IP', or 'Unrate'"
)
parser.add_argument(
    "--date", 
    type=str, 
    default=str(date.today()), 
    help="Date identifier for results (default: from config)"
)
parser.add_argument(
    "--shelf-date", 
    type=str, 
    default=None,
    help="Date identifier for shelf model predictions"
)
parser.add_argument(
    "--st-date", 
    type=str, 
    default=None,
    help="Date identifier for state-of-the-art model predictions"
)
parser.add_argument(
    "--plot-quantiles", 
    action="store_true", 
    help="Whether to generate quantile plots"
)
parser.add_argument(
    "--plot-means", 
    action="store_true", 
    help="Whether to generate mean plots"
)
parser.add_argument(
    "--dm-test", 
    action="store_true", 
    help="Whether to perform Diebold-Mariano tests for pairwise model comparisons"
)


args = parser.parse_args()


# Command line args
CONCAT_PREDS = args.concat_preds
TARGETS = args.targets
DATE = args.date
SHELF_DATE = args.shelf_date if args.shelf_date is not None else DATE
ST_DATE = args.st_date if args.st_date is not None else DATE
PLOT_QUANTILES = args.plot_quantiles
PLOT_MEANS = args.plot_means

# Paths
BASE_DIR =Path('.')
DATA_DIR = BASE_DIR / 'data' / 'processed'
NAIVE_PRED_DIR = BASE_DIR / 'predictions' / 'naive_preds'
LIT_BENCH_PRED_DIR = BASE_DIR / 'predictions' / 'lit_bench_preds' 
# SHELF_PRED_DIR = BASE_DIR / 'predictions' / 'shelf_preds' / SHELF_DATE
LINEAR_PRED_DIR = BASE_DIR / 'predictions' / 'linear_preds' / SHELF_DATE
TREE_PRED_DIR = BASE_DIR / 'predictions' / 'tree_preds' / SHELF_DATE
ST_PRED_DIR = BASE_DIR / 'predictions' / 'st_preds' / ST_DATE
CONCAT_PRED_DIR = BASE_DIR / 'predictions' / 'concatenated' / DATE
RESULTS_DIR = BASE_DIR / "results" / DATE
TABLES_DIR = BASE_DIR / "results_tables" / DATE
FIGURES_DIR = BASE_DIR / "results_figures" / DATE

for dir in [
    NAIVE_PRED_DIR,
    LIT_BENCH_PRED_DIR,
    # SHELF_PRED_DIR,
    ST_PRED_DIR,
    CONCAT_PRED_DIR, 
    RESULTS_DIR, 
    TABLES_DIR, 
    FIGURES_DIR
]:
    os.makedirs(dir, exist_ok=True)

# Config
COUNTRY = config['country']
HORIZON = config['horizon_in_quarters']
QUANTILES = config['quantiles']
INT_QUANTILES = [int(q*100) for q in QUANTILES]
TEST_START = config['test_start']
TEST_END = config['test_end']
START_YEAR = config['start_year']
END_YEAR = config['end_year']
TARGETS_FILE = config['target_file']
TARGETS_PATH = DATA_DIR / TARGETS_FILE
TARGET_NAME_DICT = {0: 'Infl_yoy', 1: 'IP_yoy', 2: 'Unrate_yoy'}
MODELS = config['models']
MODELS_TO_PLOT = config['models_to_plot']
TARGET_SCALE = config['target_scale']

def main():
    """Combine predictions and generate evaluation tables and figures."""
    
    if CONCAT_PREDS:
        print("Step 1: Concatenating predictions...")
        
        concat_preds(
            pred_dir_paths = [
                NAIVE_PRED_DIR, 
                LIT_BENCH_PRED_DIR,
                LINEAR_PRED_DIR,
                TREE_PRED_DIR,
                ST_PRED_DIR 
            ],
            out_dir = CONCAT_PRED_DIR,
            targets_to_concat = TARGETS,
            start_year = START_YEAR,
            end_year = END_YEAR,
            country = COUNTRY,
            horizon_in_quarters = HORIZON
        )

    # Load targets 
    targets = pd.read_csv(TARGETS_PATH, index_col=0, parse_dates=True) 
    targets *= TARGET_SCALE

    all_r1_results = {}
    all_r2_results = {}
    for target in TARGETS:

        preds_file = f'all_models_predictions_{COUNTRY}_{HORIZON}q_{target}.csv'

        # Load concatenated predictions for target
        target_preds = pd.read_csv(
            CONCAT_PRED_DIR / preds_file,
            index_col=0,
            parse_dates=True
        )
        target_preds.columns = target_preds.columns.str.replace(
            r'VG|IAR|UAR', 'LIT', regex=True
        )

        # Get true target values and benchmark
        y_true = targets.loc[TEST_START:TEST_END, target]
        
        q_benchmark_cols = [f'Naive_Q{q}' for q in INT_QUANTILES]
        target_q_benchmark = target_preds.loc[:, q_benchmark_cols]
        target_mean_benchmark = target_preds.loc[:, 'Naive_Mean']
        target_preds = target_preds.drop(
            columns=(q_benchmark_cols + ['Naive_Mean'])
        )
        print(target_preds.columns)

        if PLOT_QUANTILES:
            print("\nGenerating quantile plots...")
            make_quantile_plots(
                y_true=y_true,
                y_pred=target_preds,
                fig_dir=FIGURES_DIR,
                models_to_plot=MODELS_TO_PLOT,
                quantiles=QUANTILES,
                target_name=target,
                country=COUNTRY,
                horizon_in_quarters=HORIZON
            )

        # Compute R1 scores for each model and quantile 
        r1_results_df = get_r1_results_df(
            y_true=y_true,
            preds_df=target_preds,
            benchmark=target_q_benchmark,
            models=MODELS,
            quantiles=QUANTILES
        )
        all_r1_results[target] = r1_results_df
        
        # Get mean predictions from quantiles
        mean_preds = get_mean_preds(
            quantile_preds=target_preds.drop(columns=['AR1_Mean']),
            models=MODELS,
            weights=[0.15, 0.225, 0.25, 0.225, 0.15]
        )

        # Compute R2 results
        r2_df = get_r2_results_df(
            y_true=y_true,
            preds_df=mean_preds,
            benchmark=target_mean_benchmark,
            models=MODELS
        )
        all_r2_results[target] = r2_df
    
    # Save results

    # R1 results
    all_r1_results_df = pd.concat(all_r1_results)
    all_r1_results_df.index.names = ['Target', 'Quantile']
    all_r1_results_df.index = all_r1_results_df.index.set_levels(
        all_r1_results_df.index.levels[0].str.replace(r'_yoy', '', regex=False),
        level='Target'
    )

    all_r1_results_df.to_csv(
        RESULTS_DIR / f'r1_{COUNTRY}_{HORIZON}q_{TEST_START}-{TEST_END}.csv'
    )

    all_r1_results_df.to_latex(
        TABLES_DIR / f'r1_{COUNTRY}_{HORIZON}q_{TEST_START}-{TEST_END}.tex',
        multirow=True,
        float_format="%.2f"
    )

    # R2 results
    all_r2_results_df = pd.concat(all_r2_results).droplevel(1)
    all_r2_results_df.index.name = 'Target'

    all_r2_results_df.index = all_r2_results_df.index.str.replace(
        r'_yoy', '', regex=False
    )

    all_r2_results_df.to_csv(
        RESULTS_DIR / f'r2_{COUNTRY}_{HORIZON}q_{TEST_START}-{TEST_END}.csv'
    )

    all_r2_results_df.to_latex(
        TABLES_DIR / f'r2_{COUNTRY}_{HORIZON}q_{TEST_START}-{TEST_END}.tex',
        multirow=False,
        float_format="%.2f"
    )

    # if PLOT_MEANS:
    #     print("\nStep 6: Generating mean plots...")
    #     make_mean_plots(
    #         target_idx=TARGET_IDX,
    #         targets_path=DATA_DIR / config['target_file'],
    #         pred_path=PRED_DIR / f'all_models_predictions_{COUNTRY}_{HORIZON}q_{TARGET_NAME}.csv',
    #         fig_dir=FIGURES_DIR,
    #         country=COUNTRY,
    #         horizon_in_quarters=HORIZON,
    #         quantiles=QUANTILES,
    #         test_start=TEST_START,
    #         test_end=TEST_END,
    #         date_str=DATE
    #     )
    # print(f"\nResults complete. Tables saved to {TABLES_DIR}, Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
