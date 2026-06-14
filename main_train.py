"""
Model Training Entry Point
Fits models based on specified type: shelf, lit_bench, deep, or all.
"""
import logging
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from pathlib import Path
import os
import yaml
import argparse
from datetime import date
import optuna
import warnings
import keras
from copy import deepcopy

from src.data.prepare_data import prepare_non_rnn_data, prepare_rnn_data
from src.train.shelf_models import *
from src.train.losses import make_tilted_loss, make_total_tilted_loss
from src.train.models import build_dmq
from src.train.train_utils import fit_models
from src.train.tuning import perform_hpo
from src.utils.files import (
    check_hps_exist,
    load_hyperparameters,
    repair_hyperparameters_log,
)

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # For parallelization

SEED = 1  # Set random seed for reproducibility
keras.utils.set_random_seed(SEED)

load_dotenv()

# ----- Configuration -----
with open("./config/config_file.yaml", "r") as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO)

# Command line arguments
parser = argparse.ArgumentParser(description="Train models")

# Required arguments
parser.add_argument(
    "--year", 
    type=int, 
    required=True, 
    help="Training cutoff year"
)
parser.add_argument(
    "--target", 
    type=int, 
    required=True, 
    help="Target index (0=Infl, 1=IP, 2=Unrate)"
)

# Optional arguments
parser.add_argument(
    "--model-type",
    type=str, 
    nargs='+', 
    default=["all"],
    choices=["linear", "trees", "deep", "all"],
    help="Type(s) of models to train (e.g. --model-type trees deep)"
)
parser.add_argument(
    "--date", 
    type=str, 
    default=str(date.today()),
    help="Date for organizing outputs (default: today's date)"
)
parser.add_argument(
    "--local-test", 
    action="store_true", 
    help=(
        "Whether to run a test locally "
        "(reduces hyperparameter tuning for quick testing)"
    )
)
parser.add_argument(
    "--fit-lit-bench", 
    action="store_true", 
    help="Fit models from the literature. Only needs to be run once."
)

args = parser.parse_args()

# Command-line args
DATE = args.date
YEAR = args.year
TARGET_IDX = args.target
MODEL_TYPE = args.model_type
LOCAL_TEST = args.local_test
FIT_LIT_BENCH = args.fit_lit_bench

# Paths
BASE_DIR = Path('./')
DATA_DIR = BASE_DIR / 'data' / 'processed'
SHELF_MODEL_DIR = BASE_DIR / 'models' / 'shelf_models' / DATE
SHELF_PRED_DIR = BASE_DIR / 'predictions' / 'shelf_preds' / DATE
SHELF_TUNING_LOG_PATH = BASE_DIR / 'tuning_logs' / f"shelf_tuning_log_{DATE}.json"
LIT_BENCH_PRED_DIR = BASE_DIR / 'predictions' / 'lit_bench_preds' / DATE
DEEP_MODEL_DIR = BASE_DIR / 'models' / 'st_models' / DATE 
DEEP_PRED_DIR = BASE_DIR / 'predictions' / 'st_preds' / DATE
DEEP_TUNING_LOG_PATH = BASE_DIR / 'tuning_logs' / f"st_tuning_log_{DATE}.json"
DEEP_OPTUNA_JOURNAL_PATH = BASE_DIR / 'tuning_logs' / f"st_optuna_journal_{DATE}.log"

for path in [SHELF_MODEL_DIR, SHELF_PRED_DIR, SHELF_TUNING_LOG_PATH.parent, LIT_BENCH_PRED_DIR, DEEP_MODEL_DIR, DEEP_PRED_DIR, DEEP_TUNING_LOG_PATH.parent]:
    os.makedirs(path, exist_ok=True)

# Other
TARGET_NAME_DICT = {0: 'Infl_yoy', 1: 'IP_yoy', 2: 'Unrate_yoy'}
TARGET_NAME = TARGET_NAME_DICT[TARGET_IDX]


# Data
data_config = config['data']
START_DATE = data_config['start_date']
COUNTRY = data_config['country']
HORIZON_IN_QUARTERS = data_config['horizon_in_quarters']
INPUT_FILES = data_config['input_files']
TARGET_FILE = data_config['target_file']
TARGET_PATH = DATA_DIR / TARGET_FILE
INPUT_PATHS = [
    DATA_DIR / file for file in INPUT_FILES 
]
BENCHMARK_FILE_DICT = {
    0: f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_iar_x.csv",
    1: f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_vg_x.csv",
    2: f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_uar_x.csv"
}
BENCHMARK_NAME_DICT = {0: 'IAR', 1: 'VG', 2: 'UAR'}
BENCHMARK_FILE = BENCHMARK_FILE_DICT[TARGET_IDX]
BENCHMARK_NAME = BENCHMARK_NAME_DICT[TARGET_IDX]
BENCHMARK_INPUT_PATHS = [DATA_DIR / BENCHMARK_FILE]
TIME_STEPS = data_config['time_steps']
VAL_MONTHS = data_config['val_months']
TEST_MONTHS = data_config['test_months']
TEST_IDX = pd.date_range(
    start=f'{YEAR+1}-01-01', 
    periods=TEST_MONTHS, 
    freq='MS'
)
TARGET_SCALE_FACTOR = data_config['target_scale_factor']

# General
QUANTILES = config['quantiles']
PATH_QUANTILES = [int(q*100) for q in QUANTILES]

# Tuning
tuning_config = config['tuning']
OPTUNA_STORAGE = tuning_config['optuna_storage']
TRIALS = tuning_config['trials']
K_FOLDS = tuning_config['k_folds']
VAL_SIZE = tuning_config['val_size']
LINEAR_GRIDS = tuning_config['linear_grids']
QRF_GRID = tuning_config['qrf_grid']
QGB_GRID = tuning_config['qgb_grid']
DMQ_GRID = tuning_config['dmq_grid']

# Training
training_config = config['training']
N_ESTIMATORS = training_config['n_estimators']
FIT_PARAMS = training_config['fit_params']
EARLY_STOPPING_ARGS = training_config['early_stopping']
BUILDER_PARAMS = training_config['builder_params']

# Local modifications for testing
if LOCAL_TEST:
    N_ESTIMATORS = 1
    FIT_PARAMS['epochs'] = 1
    K_FOLDS=2
    TRIALS=1
    OPTUNA_STORAGE = 'inmemory'




def train_linear_models():
    """Train linear shelf models (Naive, AR1, LR, LASSO)."""
    logging.info(f"Training linear models for {TARGET_NAME_DICT[TARGET_IDX]} ({YEAR})...")

    (
        X_train, X_val, X_test,
        t_train, t_val, t_test
    ) = prepare_non_rnn_data(
        targets_path=TARGET_PATH,
        input_paths=INPUT_PATHS,
        start_date=START_DATE,
        train_cutoff_year=YEAR,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
        target_scale_factor=TARGET_SCALE_FACTOR
    )

    X_train_full = pd.concat([X_train, X_val])

    y_train = t_train.iloc[:, TARGET_IDX]
    y_val = t_val.iloc[:, TARGET_IDX]
    y_train_full = pd.concat([y_train, y_val])

    all_preds = {}

    # Naive models
    naive_preds = fit_dummy(X_train_full, y_train_full, X_test, QUANTILES)
    all_preds.update(naive_preds)

    # AR(1) models
    ar1_x_path = DATA_DIR / f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_ar1_x.csv"
    X_ar1 = pd.read_csv(ar1_x_path, index_col=0, parse_dates=True)
    X_train_ar1 = X_ar1.loc['1961-02-01':f'{YEAR}-12-01', f"{TARGET_NAME}_t-1"]
    X_test_ar1 = X_ar1.loc[f'{YEAR+1}-01-01': f'{YEAR+1}-12-01', f"{TARGET_NAME}_t-1"]
    y_train_ar1 = y_train_full.loc['1961-02-01':f'{YEAR}-12-01']
    ar1_preds = fit_ar1(X_train_ar1, y_train_ar1, X_test_ar1, QUANTILES, TARGET_NAME, YEAR, verbose=False)
    all_preds.update(ar1_preds)

    fit_params = deepcopy(FIT_PARAMS) 
    fit_params['validation_data'] = (X_val, y_val)

    linear_preds = fit_linear_models(
        X_train=X_train, 
        y_train=y_train, 
        X_train_full=X_train_full, 
        y_train_full=y_train_full, 
        X_test=X_test,
        quantiles=QUANTILES, 
        target_name=TARGET_NAME, 
        year=YEAR,
        val_size=VAL_SIZE,
        k_folds=K_FOLDS,
        tuning_path=SHELF_TUNING_LOG_PATH,
        early_stopping_args=EARLY_STOPPING_ARGS,
        fit_params=fit_params,
        seed=SEED,
        trials=TRIALS,
        n_estimators=N_ESTIMATORS,
        model_dir_path=SHELF_MODEL_DIR,
        linear_grids=LINEAR_GRIDS,
        optuna_storage=OPTUNA_STORAGE,
    )
    all_preds.update(linear_preds)

    preds_df = pd.DataFrame(
        all_preds,
        index=TEST_IDX
    )
    output_path = SHELF_PRED_DIR / f"linear_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{TARGET_NAME}_{YEAR}.csv"
    preds_df.to_csv(output_path)
    logging.info(f"Linear model predictions saved to {output_path}")


def train_tree_models():
    """Train tree-based shelf models (QRF, QGB)."""
    logging.info(f"Training tree models for {TARGET_NAME_DICT[TARGET_IDX]} ({YEAR})...")

    # Prepare data
    (
        X_train, X_val, X_test,
        t_train, t_val, t_test
    ) = prepare_non_rnn_data(
        targets_path=TARGET_PATH,
        input_paths=INPUT_PATHS,
        start_date=START_DATE,
        train_cutoff_year=YEAR,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
        target_scale_factor=TARGET_SCALE_FACTOR
    )

    X_train_full = pd.concat([X_train, X_val])

    y_train = t_train.iloc[:, TARGET_IDX]
    y_val = t_val.iloc[:, TARGET_IDX]
    y_train_full = pd.concat([y_train, y_val])

    all_preds = {}

    # QRF
    qrf_preds = fit_qrf(
        X_train_full, 
        y_train_full, 
        X_test, 
        QUANTILES,
        TARGET_NAME, 
        YEAR, 
        QRF_GRID, 
        K_FOLDS, 
        SHELF_TUNING_LOG_PATH
    )
    all_preds.update(qrf_preds)

    # QGB
    qgb_preds = fit_qgb(
        X_train_full, 
        y_train_full,
        X_test, 
        QUANTILES,
        TARGET_NAME, 
        YEAR, 
        QGB_GRID, 
        K_FOLDS, 
        SHELF_TUNING_LOG_PATH
    )
    all_preds.update(qgb_preds)

    preds_df = pd.DataFrame(
        all_preds,
        index=TEST_IDX
    )

    output_path = (
        SHELF_PRED_DIR 
        / (
            f"tree_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_"
            f"{TARGET_NAME}_{YEAR}.csv"
        )
    )
    preds_df.to_csv(output_path)

    logging.info(f"Tree model predictions saved to {output_path}")


def train_lit_bench_models():
    """Train literature benchmark models (VG, IAR, UAR)."""
    logging.info(f"Training literature benchmark models for {TARGET_NAME} ({YEAR})...")
    
    # Prepare data
    (
        X_train, X_val, X_test,
        t_train, t_val, t_test
    ) = prepare_non_rnn_data(
        targets_path=TARGET_PATH,
        input_paths=BENCHMARK_INPUT_PATHS,
        start_date=START_DATE,
        train_cutoff_year=YEAR,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
        target_scale_factor=TARGET_SCALE_FACTOR
    )

    X_train_full = pd.concat([X_train, X_val])

    y_train = t_train.iloc[:, TARGET_IDX]
    y_val = t_val.iloc[:, TARGET_IDX]
    y_train_full = pd.concat([y_train, y_val])
    
    
    # Fit quantile regression models
    preds = {}
    for q in QUANTILES:
        Q = int(q * 100)
        model = QuantReg(y_train_full.values.flatten(), add_constant(X_train_full.values, has_constant='skip'))
        res = model.fit(q=q)
        preds[f'{BENCHMARK_NAME}_Q{Q}'] = res.predict(add_constant(X_test.values, has_constant='skip'))
    
    # Save predictions
    output_path = (
        LIT_BENCH_PRED_DIR 
        / (
            f"lit_bench_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_"
            f"{TARGET_NAME}_{YEAR}.csv"
        )
    )
    pd.DataFrame(preds, index=TEST_IDX).to_csv(output_path)

    print(f"Literature benchmark predictions saved to {output_path}")


def train_deep_models():
    """Train deep learning models"""
    logging.info(f"Training deep models for {TARGET_NAME} ({YEAR})...")

    # Prepare data (including RNN sequences)
    (
        X_train, X_val, X_test,
        t_train, t_val, t_test
    ) = prepare_rnn_data(
        targets_path=TARGET_PATH,
        input_paths=INPUT_PATHS,
        start_date=START_DATE,
        train_cutoff_year=YEAR,
        n_timesteps=TIME_STEPS,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
        target_scale_factor=TARGET_SCALE_FACTOR
    )

    X_train_full = np.concatenate([X_train, X_val], axis=0)

    y_train = t_train[:, TARGET_IDX]
    y_val = t_val[:, TARGET_IDX]
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    mq_y_train = np.repeat(
        y_train.reshape(-1,1), 
        repeats=len(QUANTILES),
        axis=1
    )
    mq_y_val = np.repeat(
        y_val.reshape(-1,1), 
        repeats=len(QUANTILES),
        axis=1
    )
    mq_y_train_full = np.repeat(
        y_train_full.reshape(-1,1), 
        repeats=len(QUANTILES),
        axis=1
    )
    
    # Custom objects for loading models
    custom_objects = {
        **{f'tilted_loss_{Q}': make_tilted_loss(Q) for Q in PATH_QUANTILES},
        **{
            f"total_tilted_loss_{'_'.join(map(str, PATH_QUANTILES))}": 
            make_total_tilted_loss(QUANTILES)
        }
    }
    
    model_names = list(BUILDER_PARAMS.keys())
    
    # Optuna storage
    if OPTUNA_STORAGE == "inmemory":
        storage = optuna.storages.InMemoryStorage()
    else:
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(str(DEEP_OPTUNA_JOURNAL_PATH))
        )

    all_model_preds = {}

    # Set training and validation sets
    X_tr = X_train_full
    y_tr = mq_y_train_full
    validation_data = (X_val, mq_y_val)
    X_te = X_test

    # set fit params
    fit_params = deepcopy(FIT_PARAMS)
    fit_params.update(
        {'validation_data': validation_data}
    )
    
    # Train each model variant
    for model_type in model_names:
        study_name = f'{model_type}_{TARGET_NAME}_{YEAR}'
        logging.info(f"Training {model_type}...")

        # Get basic model config
        builder_params = BUILDER_PARAMS[model_type]
        
        # Update builder params with runtime arguments
        builder_params.update(
            {
                'input_shapes': [X_train.shape[1:]],
                'lower_quantiles': [q for q in QUANTILES if q < 0.5],
                'upper_quantiles': [q for q in QUANTILES if q > 0.5]
            }
        )
        
        # Check if hyperparameters already exist
        if check_hps_exist(study_name, DEEP_TUNING_LOG_PATH):
            logging.info(f"Hyperparameters for {study_name} already exist. Loading...")
            best_params = load_hyperparameters(study_name, DEEP_TUNING_LOG_PATH)
        
        # If not perform hpo. If storage other than inmemory was used
        # perform_hpo will load best_params from storage if available.
        else:
            logging.info(f"No existing hyperparameters for {study_name}, optimizing...")
            
            best_params = perform_hpo(
                X_train=X_tr,
                y_train=y_tr,
                val_size=VAL_SIZE,
                n_splits=K_FOLDS,
                builder_func=build_dmq,
                fit_params=fit_params,
                early_stopping_args=EARLY_STOPPING_ARGS,
                grid=DMQ_GRID,
                study_name=study_name,
                trials=TRIALS,
                n_jobs=os.cpu_count()-1,
                storage=storage,
                sampler=optuna.samplers.RandomSampler(seed=SEED),
                pruner=optuna.pruners.MedianPruner(),
                save_hps=True if OPTUNA_STORAGE == 'inmemory' else False,
                log_path=DEEP_TUNING_LOG_PATH,
                **builder_params
            )

        best_params.update(builder_params)
        
        # Fit models with best hyperparameters
        estimators = fit_models(
            X_tr,
            y_tr,
            build_dmq,
            model_name=study_name,
            hps=best_params,
            fit_params=fit_params,
            early_stopping_args=EARLY_STOPPING_ARGS,
            n_estimators=N_ESTIMATORS,
            models_dir_path=DEEP_MODEL_DIR,
            save_models=True,
            custom_objects=custom_objects
        )
        
        # Generate predictions
        preds = []
        for e in estimators:
            e_preds = e.predict(X_te, verbose=0).reshape(-1, len(QUANTILES))
            preds.append(e_preds[:, :, np.newaxis])
        
        # Take the mean over estimators
        preds = np.concatenate(preds, axis=2).mean(axis=2) 
        
        # Store predictions for each quantile
        for i, q in enumerate(QUANTILES):
            Q = int(q * 100)
            all_model_preds[f'{model_type}_Q{Q}'] = preds[:, i]
    
    # Save predictions
    all_model_preds_df = pd.DataFrame(
        all_model_preds,
        index=pd.date_range(
            start=f'{YEAR+1}-01-01', 
            periods=TEST_MONTHS, 
            freq='MS')
    )
    
    # Save to file
    output_path = (
        DEEP_PRED_DIR 
        / (
            f"st_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_"
            f"{TARGET_NAME}_{YEAR}.csv"
        )
    )

    all_model_preds_df.to_csv(output_path)

    logging.info(f"Deep model predictions saved to {output_path}")


def main():
    """Run model training based on specified type."""

    for tuning_log_path in [SHELF_TUNING_LOG_PATH, DEEP_TUNING_LOG_PATH]:
        if tuning_log_path.exists():
            repaired = repair_hyperparameters_log(tuning_log_path)
            logging.info(
                "Recovered %d tuning-log entries from %s",
                len(repaired),
                tuning_log_path,
            )


    if FIT_LIT_BENCH:
        train_lit_bench_models()

    run_all = "all" in MODEL_TYPE

    if run_all or "linear" in MODEL_TYPE:
        train_linear_models()

    if run_all or "trees" in MODEL_TYPE:
        train_tree_models()

    if run_all or "deep" in MODEL_TYPE:
        train_deep_models()
    
    logging.info("Training complete.")


if __name__ == "__main__":
    main()
