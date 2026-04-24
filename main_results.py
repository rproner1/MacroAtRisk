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
from operator import itemgetter
from src.data.concat_preds import concat_predictions
from src.eval.make_tables import make_r1_multitarget_table_body, make_r2_multitarget_table
from src.eval.dm_tests import make_dm_tables
from src.figures.make_figures import make_quantile_plots, make_mean_plots
load_dotenv()

# ----- Configuration -----
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

parser = argparse.ArgumentParser(description="Generate results and tables")
parser.add_argument('--target', type=int, default=0, help="Target variable index (0: Infl_yoy, 1: IP_yoy, 2: Unrate_yoy)")
parser.add_argument("--date", type=str, default=None, 
                    help="Date identifier for results (default: from config)")
parser.add_argument("--country", type=str, default=None, 
                    help="Country code (default: from config)")
parser.add_argument("--horizon", type=int, default=None, 
                    help="Forecast horizon in quarters (default: from config)")
parser.add_argument("--quantiles", type=float, nargs="*", default=None,
                    help="List of quantiles (default: from config)")
parser.add_argument("--test-start", type=str, default='1998-01-01',
                    help="Test set start date. Options (with year-long buffer): {Tech bubble '2000-03-01', GFC '2006-12-01', Covid '2019-02-01'} ")
parser.add_argument("--test-end", type=str, default='2024-12-01',
                    help="Test set end date Options with year-long buffer, except COVID which we extend to the end of the sample (Tech bubble '2002-11-01', GFC '2010-06-01', Covid '2024-12-01')")
parser.add_argument("--start-year", type=int, default=None,
                    help="First prediction year (default: from config)")
parser.add_argument("--end-year", type=int, default=None,
                    help="Last prediction year (default: from config)")
parser.add_argument("--plot-quantiles", action="store_true", help="Whether to generate quantile plots")
parser.add_argument("--plot-means", action="store_true", help="Whether to generate mean plots")
parser.add_argument("--run-locally", action="store_true", help="Whether to run locally (adjusts file paths accordingly)")
args = parser.parse_args()

# Use config values as defaults, allow CLI overrides
TARGET_IDX = args.target
DATE = args.date if args.date is not None else config.get('date', str(date.today()))
COUNTRY = args.country if args.country is not None else config['country']
HORIZON = args.horizon if args.horizon is not None else config['horizon_in_quarters']
QUANTILES = args.quantiles if args.quantiles is not None else config['quantiles']
TEST_START = args.test_start if args.test_start is not None else config['test_start']
TEST_END = args.test_end if args.test_end is not None else config['test_end']
START_YEAR = args.start_year if args.start_year is not None else config['start_year']
END_YEAR = args.end_year if args.end_year is not None else config['end_year']
TARGET_FILE = config['target_file']
PLOT_QUANTILES = args.plot_quantiles
PLOT_MEANS = args.plot_means
RUN_LOCALLY = args.run_locally
BASE_DIR = Path(os.getenv('REMOTE_BASE_DIR')) if not RUN_LOCALLY else Path(os.getenv('LOCAL_BASE_DIR'))

DATA_DIR = BASE_DIR / 'data' / 'processed'
SHELF_MODEL_DIR = BASE_DIR / 'models' / 'shelf_models' / DATE
SHELF_PRED_DIR = BASE_DIR / 'predictions' / 'shelf_preds' / DATE
SHELF_TUNING_LOG_PATH = BASE_DIR / 'tuning_logs' / f"shelf_tuning_log_{DATE}.json"
LIT_BENCH_PRED_DIR = BASE_DIR  / 'lit_benchmark_predictions' 
ST_MODEL_DIR = BASE_DIR / 'models' / 'st_models' / DATE 
ST_PRED_DIR = BASE_DIR / 'predictions' / 'st_preds' / DATE

PRED_DIR = BASE_DIR / 'predictions' / 'concatenated' / DATE

RESULTS_DIR = BASE_DIR / "results" / DATE
TABLES_DIR = BASE_DIR / "results_tables" / DATE
FIGURES_DIR = BASE_DIR / "results_figures" / DATE

target_path = DATA_DIR / TARGET_FILE

target_name_dict = {0: 'Infl_yoy', 1: 'IP_yoy', 2: 'Unrate_yoy'}
target_name = target_name_dict[TARGET_IDX]

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    """Combine predictions and generate evaluation tables and figures."""
    
    print("Step 1: Concatenating predictions...")
    concat_predictions(
        country=COUNTRY,
        horizon_in_quarters=HORIZON,
        date=DATE,
        st_pred_dir=ST_PRED_DIR,
        lit_bench_pred_dir=LIT_BENCH_PRED_DIR,
        shelf_pred_dir=SHELF_PRED_DIR,
        pred_dir=PRED_DIR,
        start_year=START_YEAR,
        end_year=END_YEAR
    )
    
    print("\nStep 2: Generating combined R1 table body across INFL/IP/UNRATE...")
    make_r1_multitarget_table_body(
        targets_path=DATA_DIR / config['target_file'],
        pred_dir=PRED_DIR,
        results_dir=RESULTS_DIR,
        tables_dir=TABLES_DIR,
        country=COUNTRY,
        horizon_in_quarters=HORIZON,
        quantiles=QUANTILES,
        test_start=TEST_START,
        test_end=TEST_END,
        date_str=DATE
    )
    
    print("\nStep 3: Generating combined R2 table across INFL/IP/UNRATE...")
    make_r2_multitarget_table(
        targets_path=DATA_DIR / config['target_file'],
        pred_dir=PRED_DIR,
        results_dir=RESULTS_DIR,
        tables_dir=TABLES_DIR,
        country=COUNTRY,
        horizon_in_quarters=HORIZON,
        quantiles=QUANTILES,
        test_start=TEST_START,
        test_end=TEST_END,
        date_str=DATE
    )

    print("\nStep 4: Generating pairwise Diebold-Mariano tables...")
    make_dm_tables(
        targets_path=DATA_DIR / config['target_file'],
        pred_dir=PRED_DIR,
        results_dir=RESULTS_DIR,
        tables_dir=TABLES_DIR,
        country=COUNTRY,
        horizon_in_quarters=HORIZON,
        quantiles=QUANTILES,
        test_start=TEST_START,
        test_end=TEST_END,
        alpha=0.05,
        date_str=DATE,
    )
    
    if PLOT_QUANTILES:
        print("\nStep 5: Generating quantile plots...")
        make_quantile_plots(
            target_idx=TARGET_IDX,
            targets_path=DATA_DIR / config['target_file'],
            pred_path=PRED_DIR / f'all_models_predictions_{COUNTRY}_{HORIZON}q_{target_name}.csv',
            fig_dir=FIGURES_DIR,
            country=COUNTRY,
            horizon_in_quarters=HORIZON,
            quantiles=QUANTILES,
            test_start=TEST_START,
            test_end=TEST_END,
            date_str=DATE
        )
    
    if PLOT_MEANS:
        print("\nStep 6: Generating mean plots...")
        make_mean_plots(
            target_idx=TARGET_IDX,
            targets_path=DATA_DIR / config['target_file'],
            pred_path=PRED_DIR / f'all_models_predictions_{COUNTRY}_{HORIZON}q_{target_name}.csv',
            fig_dir=FIGURES_DIR,
            country=COUNTRY,
            horizon_in_quarters=HORIZON,
            quantiles=QUANTILES,
            test_start=TEST_START,
            test_end=TEST_END,
            date_str=DATE
        )
    print(f"\nResults complete. Tables saved to {TABLES_DIR}, Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
