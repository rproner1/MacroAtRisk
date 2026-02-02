from datetime import date
import pandas as pd
from dotenv import load_dotenv
import logging
from pathlib import Path
import argparse
import os
from operator import itemgetter
from src.data.run_data_pipeline import prepare_us_data
from src.utils.files import get_latest_file
from src.preprocessing.prepare_quantile_data import prepare_quantile_data
load_dotenv()

DATADIR = Path(os.getenv("DATADIR"))
raw_data_dir = DATADIR / "raw/"
processed_data_dir = DATADIR / "processed/"
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

ROOT_DIR = Path(os.getenv("ROOTDIR"))
conf_path = ROOT_DIR / 'config' / 'config.yaml'

PREDDIR = ROOT_DIR / "predictions/"
RESULTSDIR = ROOT_DIR / "results/"

# Load configuration parameters
import yaml
with open(conf_path, "r") as f:
    config = yaml.safe_load(f)

# Main parameters
process_data = config.get("process_data")

# Training parameters

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True, help="train cutoff year")
args = parser.parse_args()

YEAR = args.year
COUNTRY = config['country']
HORIZON_IN_QUARTERS = config['horizon_in_quarters']
QUANTILES = config['quantiles']
TARGET_IDX = config['target_idx']
RUN_LOCALLY = config['run_locally']
K_FOLDS = config['k_folds']
DATE = config.get('date', str(date.today()))
INPUT_FILES = config['input_files']
TARGET_FILE = config['target_file']

TIME_STEPS = config['time_steps']
TRIALS = config['trials']
N_ESTIMATORS = config['n_estimators']
VAL_YEARS  = config['val_years']

path_quantiles = [int(q*100) for q in QUANTILES]  # Quantiles as integers (e.g., 5 for 0.05) for file names

LOSS_WEIGHTS = [0.28, 0.17, 0.11, 0.17, 0.28]

if RUN_LOCALLY: 
    TRIALS = 2
    N_ESTIMATORS = 2


# Data construction parameters
DESIRED_START_DATE_OF_SAMPLES = pd.to_datetime(config.get("desired_start_date_of_samples", "1961-01-01"))
INITIAL_TRAINING_LAST_DATE = pd.to_datetime(config.get("initial_training_last_date", "1997-12-01"))
LAST_DATE_OF_SAMPLE = pd.to_datetime(config.get("last_date_of_sample", "2024-12-01"))
REMOVE_COLS_THRESHOLD = config.get("remove_cols_threshold", 0.3)
CONSTRUCT_OAP_SIGNALS = config["construct_oap_signals"]
SKIP_PROCESSED_DATA = config["skip_processed_data"]

def main():
    
    if process_data:
        prepare_us_data(
            get_latest_file(raw_data_dir / "2025-12-MD.csv", extension=".csv", directory=raw_data_dir), 
            get_latest_file(raw_data_dir / "signed_predictors_dl_wide.csv", extension=".csv", directory=raw_data_dir),
            get_latest_file(raw_data_dir / "crsp_monthly.parquet", extension=".parquet", directory=raw_data_dir),
            get_latest_file(raw_data_dir / "nfci_monthly.csv", extension=".csv", directory=raw_data_dir),
            get_latest_file(raw_data_dir / "NROU.csv", extension=".csv", directory=raw_data_dir),
            get_latest_file(raw_data_dir / "EXPINF10YR.csv", extension=".csv", directory=raw_data_dir),
            get_latest_file(raw_data_dir / "ebp_csv.csv", extension=".csv", directory=raw_data_dir),
            desired_start_date_of_samples=DESIRED_START_DATE_OF_SAMPLES,
            horizon_in_quarters=HORIZON_IN_QUARTERS,
            initial_training_last_date=INITIAL_TRAINING_LAST_DATE,
            last_date_of_sample=LAST_DATE_OF_SAMPLE,
            remove_cols_threshold=REMOVE_COLS_THRESHOLD,
            skip_processed_data=SKIP_PROCESSED_DATA,
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
            run_locally=RUN_LOCALLY,
            construct_oap_signals=CONSTRUCT_OAP_SIGNALS,
            config=config
        )    

    input_paths = [processed_data_dir / file for file in INPUT_FILES]
    target_path = processed_data_dir / TARGET_FILE

    logging.info(f"Preprocessing data with inputs: {INPUT_FILES} and targets: {TARGET_FILE}")

    non_rnn_data, rnn_data, meta_data = prepare_quantile_data(
        target=TARGET_IDX,
        time_steps=TIME_STEPS, 
        targets_path=target_path,
        input_paths=input_paths,
        start_date='1961-01-01', train_cutoff_year=YEAR, 
        n_quantiles=len(QUANTILES), val_years=5
    )

    (
        mq_y_train_rnn, mq_y_val_rnn, mq_y_train_full_rnn,
        X_train_rnn, X_val_rnn, X_train_full_rnn, X_test_rnn,
    ) = itemgetter(
        'mq_y_train_rnn', 'mq_y_val_rnn', 'mq_y_train_full_rnn', # single-task multi-quantile dynamic outputs
        'X_train_rnn', 'X_val_rnn', 'X_train_full_rnn', 'X_test_rnn', # Dynamic inputs
    )(rnn_data)

    (
        X_train_full, y_train_full,
        X_train, X_val, X_test,
        y_train, y_val,
        all_y_train
    ) = itemgetter(
        'X_train_full', 'y_train_full', # Full training set for tree-based models
        'X_train', 'X_val', 'X_test', # Static inputs
        'y_train', 'y_val', # multi-task multi-quantile static outputs
        'all_y_train' # For target names
    )(non_rnn_data)



if __name__ == "__main__":
    main()
