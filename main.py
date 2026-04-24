from datetime import date
import optuna
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
from src.train.shelf_models import *
# from src.tables.make_r1_tables import make_r1_tables

load_dotenv()

# ----- Configuration -----
# Load configuration parameters
import yaml
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Data construction parameters  
DESIRED_START_DATE_OF_SAMPLES = pd.to_datetime(config['desired_start_date_of_samples'])
INITIAL_TRAINING_LAST_DATE = pd.to_datetime(config['initial_training_last_date'])
LAST_DATE_OF_SAMPLE = pd.to_datetime(config['last_date_of_sample'])
REMOVE_COLS_THRESHOLD = config['remove_cols_threshold']
CONSTRUCT_OAP_SIGNALS = config["construct_oap_signals"]
PROCESS_DATA = config['process_data']
SKIP_PROCESSED_DATA = config["skip_processed_data"]
FIT_MODELS = config['fit_models']

# Model training parameters
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True, help="train cutoff year")
parser.add_argument("--target", type=int, required=True, help="Target index 0 to 2")
args = parser.parse_args()

YEAR = args.year
TARGET_IDX = args.target
COUNTRY = config['country']
HORIZON_IN_QUARTERS = config['horizon_in_quarters']
QUANTILES = config['quantiles']
RUN_LOCALLY = config['run_locally']
K_FOLDS = config['k_folds']
DATE = config.get('date', str(date.today()))
INPUT_FILES = config['input_files']
TARGET_FILE = config['target_file']

TIME_STEPS = config['time_steps']
TRIALS = config['trials']
N_ESTIMATORS = config['n_estimators']
VAL_YEARS  = config['val_years']
LOSS_WEIGHTS = config['loss_weights']

path_quantiles = [int(q*100) for q in QUANTILES]  # Quantiles as integers (e.g., 5 for 0.05) for file names

if RUN_LOCALLY: 
    TRIALS = 2
    N_ESTIMATORS = 2

# Evaluation parameters
MAKE_RESULTS = config['make_results']
BASE_MODELS_SUBSET = config['base_models_subset']

# ----- Environment Setup -----
DATADIR = Path(os.getenv("DATADIR"))
raw_data_dir = DATADIR / "raw/"
processed_data_dir = DATADIR / "processed/"
ROOT_DIR = Path(os.getenv("ROOTDIR"))
results_dir = ROOT_DIR / "results" / DATE 
tables_dir = ROOT_DIR / "results_tables" / DATE 

if RUN_LOCALLY:
    DATA_DIR = Path(os.getenv('LOCDATADIR')) / 'processed/'
    ST_MODEL_DIR = Path(os.getenv('LOCMODELDIR')) / 'st_models' / DATE
    ST_PRED_DIR = Path(os.getenv('LOCPREDDIR')) /'st_preds' / DATE
    ST_TUNING_LOG_PATH = Path(os.getenv('LOCTUNINGDIR')) / f"st_tuning_log_{DATE}.json" 
    SHELF_MODEL_DIR = Path(os.getenv('LOCMODELDIR')) / 'shelf_models' / DATE
    SHELF_PRED_DIR = Path(os.getenv('LOCPREDDIR')) /'shelf_preds' / DATE
    SHELF_TUNING_LOG_PATH = Path(os.getenv('LOCTUNINGDIR')) / f"shelf_tuning_log_{DATE}.json"
    PRED_DIR = Path(os.getenv('LOCPREDDIR')) / 'concatenated' / DATE

else:
    DATA_DIR = Path(os.getenv('DATADIR')) / 'processed/'
    ST_MODEL_DIR = Path(os.getenv('MODELDIR')) / 'st_models' / DATE
    ST_PRED_DIR = Path(os.getenv('PREDDIR')) / 'st_preds' / DATE
    ST_TUNING_LOG_PATH = Path(os.getenv('TUNINGDIR')) / f"st_tuning_log_{DATE}.json"
    SHELF_MODEL_DIR = Path(os.getenv('MODELDIR')) / 'shelf_models' / DATE
    SHELF_PRED_DIR = Path(os.getenv('PREDDIR')) / 'shelf_preds' / DATE
    SHELF_TUNING_LOG_PATH = Path(os.getenv('TUNINGDIR')) / f"shelf_tuning_log_{DATE}.json"
    PRED_DIR = Path(os.getenv('PREDDIR')) / 'concatenated' / DATE

storage_url = optuna.storages.InMemoryStorage()

for path in [
    ST_MODEL_DIR, 
    ST_PRED_DIR, 
    ST_TUNING_LOG_PATH.parent, 
    SHELF_MODEL_DIR, 
    SHELF_PRED_DIR, 
    SHELF_TUNING_LOG_PATH.parent,
    raw_data_dir,
    processed_data_dir
    ]:
    os.makedirs(path, exist_ok=True)



def main():

    input_paths = [processed_data_dir / file for file in INPUT_FILES]
    target_path = processed_data_dir / TARGET_FILE
    
    if PROCESS_DATA:
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

    if FIT_MODELS:

        target_name = all_y_train.columns[TARGET_IDX]
        all_models_preds = {}
        lit_bench_train_preds = {}

        if FIT_LIT_BENCH:

        lit_bench_data, _, _ = prepare_quantile_data(
            target=target_name, 
            time_steps=1, # no effect
            targets_path=target_path, 
            input_paths=[DATA_DIR / 'us_4q_vg_x.parquet'],
            start_date='1974-02-01', 
            train_cutoff_year=YEAR, 
            n_quantiles=len(QUANTILES), 
            val_years=5 # no effect
        )

        if target_name == 'IP_yoy':
            logging.info('Fitting vulnerable growth model...')
            vg_data, _ , _ = prepare_quantile_data(
                target='IP_yoy', 
                time_steps=1, # no effect
                targets_path=target_path, 
                input_paths=[DATA_DIR / 'us_4q_vg_x.parquet'],
                start_date='1974-02-01', 
                train_cutoff_year=YEAR, 
                n_quantiles=len(QUANTILES), 
                val_years=5 # no effect
            )

            (
                X_train, y_train, X_test 
            ) = itemgetter(
                'X_train_full', 'y_train_full', 'X_test' 
            )(vg_data)

            vg_train_preds, vg_preds = fit_lit_bench_model(
                X_train,
                y_train, 
                X_test,
                QUANTILES,
                'VG'
            )
            all_models_preds.update(vg_preds)
            lit_bench_train_preds.update(vg_train_preds)


        logging.info('Fitting inflation-at-risk model...')
        iar_data, _ , _ = prepare_quantile_data(
            target='Infl_yoy', 
            time_steps=12, 
            targets_path=target_path, 
            input_paths=[DATA_DIR / 'us_4q_iar_x.parquet'],
            start_date='1974-02-01',
            train_cutoff_year=YEAR, 
            n_quantiles=len(QUANTILES), 
            val_years=5
        )

        (
            X_train, y_train, X_test 
        ) = itemgetter(
            'X_train_full', 'y_train_full', 'X_test' 
        )(iar_data)

        iar_train_preds, iar_preds = fit_lit_bench_model(
            X_train,
            y_train, 
            X_test,
            QUANTILES,
            'IAR'
        )
        all_models_preds.update(iar_preds)
        lit_bench_train_preds.update(iar_train_preds)

        logging.info('Fitting unemployment-at-risk model...')
        uar_data, _ , _ = prepare_quantile_data(
            target='Unrate_yoy', 
            time_steps=1, 
            targets_path=target_path, 
            input_paths=[DATA_DIR / 'us_4q_uar_x.parquet'],
            start_date='1974-02-01',
            train_cutoff_year=YEAR, 
            n_quantiles=len(QUANTILES), 
            val_years=5
        )

        (
            X_train, y_train, X_test 
        ) = itemgetter(
            'X_train_full', 'y_train_full', 'X_test' 
        )(uar_data)

        uar_train_preds, uar_preds = fit_lit_bench_model(
            X_train,
            y_train, 
            X_test,
            QUANTILES,
            'UAR'
        )
        all_models_preds.update(uar_preds)
        lit_bench_train_preds.update(uar_train_preds)

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


        # Shelf Models
        linear_model_grids ={
            'LR': {},
            'LAS': {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
        }

        # Historical mean and quantile models
        dummy_preds = fit_dummy(
            X_train_full, 
            y_train_full,
            X_test, 
            QUANTILES
        )
        all_models_preds.update(dummy_preds)

        ar1_x_path = DATA_DIR / f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_ar1_x.parquet"
        X_ar_1 = pd.read_parquet(ar1_x_path)
        X_ar_1_train = X_ar_1.loc['1961-02-01':f'{YEAR}-12-01', f"{target_name}_t-1"]
        X_ar_1_test = X_ar_1.loc[f'{YEAR+1}-01-01': f'{YEAR+1}-12-01', f"{target_name}_t-1"]
        y_train_full_ar = y_train_full.loc['1961-02-01':f'{YEAR}-12-01'] # Get rid of first date because NaN from lag

        # AR(1) models
        ar1_preds = fit_ar1(
            X_ar_1_train, 
            y_train_full_ar,
            X_ar_1_test, 
            QUANTILES,
            target_name,
            YEAR
        )
        all_models_preds.update(ar1_preds)

        # LR and LASSO models
        linear_preds = fit_linear_models(
            X_train_full, 
            y_train_full,
            X_test, 
            QUANTILES,
            target_name,
            YEAR,
            linear_model_grids,
            K_FOLDS,
            SHELF_TUNING_LOG_PATH
        )

        all_models_preds.update(linear_preds)

        # QPCR model
        qpcr_preds = fit_qpcr(
            X_train_full, 
            y_train_full,
            X_test, 
            QUANTILES,
            target_name,
            YEAR,
            K_FOLDS,
            SHELF_TUNING_LOG_PATH
        )
        all_models_preds.update(qpcr_preds)

        # QRF model
        qrf_preds = fit_qrf(
            X_train_full, 
            y_train_full,
            X_test, 
            QUANTILES,
            target_name,
            YEAR,
            N_ESTIMATORS,
            K_FOLDS,
            SHELF_TUNING_LOG_PATH
        )
        all_models_preds.update(qrf_preds)

        # QGB model
        qgb_preds = fit_qgb(
            X_train_full, 
            y_train_full,
            X_test, 
            QUANTILES,
            target_name,
            YEAR,
            qgb_grid,
            K_FOLDS,
            SHELF_TUNING_LOG_PATH
        )


        # ----- Combine predictions and save -----

        all_model_preds_df = pd.DataFrame(all_models_preds, index=pd.date_range(start=meta_data['test_start'], end=meta_data['test_end'], freq='MS'))

        all_model_preds_df.to_csv(
            PRED_DIR / f'{COUNTRY}_{HORIZON_IN_QUARTERS}q_{YEAR}_all_model_preds.csv'
        )

        

if __name__ == "__main__":
    main()
