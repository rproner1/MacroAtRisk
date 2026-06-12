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
from copy import deepcopy
import yaml
import argparse
from datetime import date
import optuna
from optuna.trial import TrialState
import statsmodels.formula.api as smf
import warnings
import keras

from src.data.prepare_data import prepare_non_rnn_data, prepare_rnn_data
from src.train.shelf_models import *
from src.train.losses import make_tilted_loss, make_total_tilted_loss
from src.train.models import build_dmq_v0, build_dmq_v1
from src.train.tuning import CVObjective
from src.train.train_utils import fit_models
from src.train.tuning import perform_hpo
from src.utils.files import (
    check_hps_exist,
    save_hyperparameters,
    load_hyperparameters,
    repair_hyperparameters_log,
)

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # For parallelization

SEED = 1  # Set random seed for reproducibility
keras.utils.set_random_seed(SEED)

load_dotenv()

# ----- Configuration -----
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("./config/data_config.yaml", 'r') as f:
    data_config = yaml.safe_load(f)

logging.basicConfig(level=config['logging_level'])

parser = argparse.ArgumentParser(description="Train models")
parser.add_argument("--year", type=int, required=True, help="Training cutoff year")
parser.add_argument("--target", type=int, required=True, help="Target index (0=Infl, 1=IP, 2=Unrate)")
parser.add_argument("--model-type", type=str, nargs='+', default=["all"],
                    choices=["linear", "trees", "deep", "all"],
                    help="Type(s) of models to train (e.g. --model-type trees deep)")
parser.add_argument("--date", type=str, default=str(date.today()), help="Date for organizing outputs (default: today's date)")
parser.add_argument("--run-locally", action="store_true", help="Whether to run locally (reduces hyperparameter tuning for quick testing)")
parser.add_argument("--fit-lit-bench", action="store_true", help="Fit models from the literature. Only needs to be run once.")
parser.add_argument(
    "--optuna-storage",
    type=str,
    default="inmemory",
    choices=["journal", "inmemory"],
    help="Optuna storage backend for tuning studies",
)
args = parser.parse_args()

YEAR = args.year
TARGET_IDX = args.target
MODEL_TYPE = args.model_type
RUN_LOCALLY = args.run_locally
FIT_LIT_BENCH = args.fit_lit_bench
OPTUNA_STORAGE = args.optuna_storage

COUNTRY = config['country']
HORIZON_IN_QUARTERS = config['horizon_in_quarters']
QUANTILES = config['quantiles']
K_FOLDS = config['k_folds']
DATE = args.date
INPUT_FILES = config['input_files']
TARGET_FILE = config['target_file']
TIME_STEPS = config['time_steps']
VAL_YEARS = config['val_years']
EPOCHS = config['epochs']
BATCH_SIZE = config['batch_size']

path_quantiles = [int(q*100) for q in QUANTILES]

BASE_DIR = Path('./')

DATA_DIR = BASE_DIR / 'data' / 'processed'
SHELF_MODEL_DIR = BASE_DIR / 'models' / 'shelf_models' / DATE
SHELF_PRED_DIR = BASE_DIR / 'predictions' / 'shelf_preds' / DATE
SHELF_TUNING_LOG_PATH = BASE_DIR / 'tuning_logs' / f"shelf_tuning_log_{DATE}.json"
LIT_BENCH_PRED_DIR = BASE_DIR / 'predictions' / 'lit_bench_preds' / DATE
DEEP_MODEL_DIR = BASE_DIR / 'models' / 'st_models' / DATE 
DEEP_PRED_DIR = BASE_DIR / 'predictions' / 'st_preds' / DATE
DEEP_TUNING_LOG_PATH =BASE_DIR / 'tuning_logs' / f"st_tuning_log_{DATE}.json"
DEEP_OPTUNA_JOURNAL_PATH = BASE_DIR / 'tuning_logs' / f"st_optuna_journal_{DATE}.log"

for path in [SHELF_MODEL_DIR, SHELF_PRED_DIR, SHELF_TUNING_LOG_PATH.parent, LIT_BENCH_PRED_DIR, DEEP_MODEL_DIR, DEEP_PRED_DIR, DEEP_TUNING_LOG_PATH.parent]:
    os.makedirs(path, exist_ok=True)

target_name_dict = {0: 'Infl_yoy', 1: 'IP_yoy', 2: 'Unrate_yoy'}
model_file_dict = {
    0: f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_iar_x.csv",
    1: f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_vg_x.csv",
    2: f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_uar_x.csv"
}
model_name_dict = {0: 'IAR', 1: 'VG', 2: 'UAR'}

target_path = DATA_DIR / TARGET_FILE
input_paths = [DATA_DIR / file for file in INPUT_FILES]

def train_linear_models():
    """Train linear shelf models (Naive, AR1, LR, LASSO)."""
    logging.info(f"Training linear models for {target_name_dict[TARGET_IDX]} ({YEAR})...")

    (
        X_train, X_val, X_test,
        t_train, t_val, t_test
    ) = prepare_non_rnn_data(
        targets_path=target_path,
        input_paths=input_paths,
        start_date='1961-01-01',
        train_cutoff_year=YEAR,
        val_split_style='date',
        val_months=data_config['val_months'],
        test_months=data_config['test_months'],
        target_scale_factor=data_config['target_scale_factor']
    )

    X_train_full = pd.concat([X_train, X_val])

    y_train = t_train.iloc[:, TARGET_IDX]
    y_val = t_val.iloc[:, TARGET_IDX]
    y_train_full = pd.concat([y_train, y_val])

    target_name = target_name_dict[TARGET_IDX]

    all_preds = {}

    # Naive models
    naive_preds = fit_dummy(X_train_full, y_train_full, X_test, QUANTILES)
    all_preds.update(naive_preds)

    # AR(1) models
    ar1_x_path = DATA_DIR / f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_ar1_x.csv"
    X_ar1 = pd.read_csv(ar1_x_path, index_col=0, parse_dates=True)
    X_train_ar1 = X_ar1.loc['1961-02-01':f'{YEAR}-12-01', f"{target_name}_t-1"]
    X_test_ar1 = X_ar1.loc[f'{YEAR+1}-01-01': f'{YEAR+1}-12-01', f"{target_name}_t-1"]
    y_train_ar1 = y_train_full.loc['1961-02-01':f'{YEAR}-12-01']
    ar1_preds = fit_ar1(X_train_ar1, y_train_ar1, X_test_ar1, QUANTILES, target_name, YEAR, verbose=False)
    all_preds.update(ar1_preds)

    early_stopping_args = {
        'monitor': 'val_loss',
        'min_delta': 1e-3,
        'patience': 5,
        'restore_best_weights': True,
        'verbose': 0
    }
    fit_params = {
        'epochs': 1 if RUN_LOCALLY else EPOCHS,
        'batch_size': 32 if RUN_LOCALLY else BATCH_SIZE,
        'validation_data': (X_val, y_val),
        'verbose': 0,
        'shuffle': False
    }

    linear_preds = fit_linear_models(
        X_train, y_train, X_train_full, y_train_full, X_test,
        QUANTILES, target_name, YEAR,
        val_size=config['val_size'],
        k_folds=K_FOLDS,
        tuning_path=SHELF_TUNING_LOG_PATH,
        early_stopping_args=early_stopping_args,
        fit_params=fit_params,
        seed=SEED,
        trials=1 if RUN_LOCALLY else config['trials'],
        n_estimators=1 if RUN_LOCALLY else config['n_estimators'],
        model_dir_path=SHELF_MODEL_DIR,
        linear_grids=config['tuning']['linear_grids'],
        optuna_storage=OPTUNA_STORAGE,
    )
    all_preds.update(linear_preds)

    preds_df = pd.DataFrame(
        all_preds,
        index=X_test.index
    )
    output_path = SHELF_PRED_DIR / f"linear_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}_{YEAR}.csv"
    preds_df.to_csv(output_path)
    logging.info(f"Linear model predictions saved to {output_path}")


def train_tree_models():
    """Train tree-based shelf models (QRF, QGB)."""
    logging.info(f"Training tree models for {target_name_dict[TARGET_IDX]} ({YEAR})...")

    (
        X_train, X_val, X_test,
        t_train, t_val, t_test
    ) = prepare_non_rnn_data(
        targets_path=target_path,
        input_paths=input_paths,
        start_date='1961-01-01',
        train_cutoff_year=YEAR,
        val_split_style='date',
        val_months=data_config['val_months'],
        test_months=data_config['test_months'],
        target_scale_factor=data_config['target_scale_factor']
    )

    X_train_full = pd.concat([X_train, X_val])

    y_train = t_train.iloc[:, TARGET_IDX]
    y_val = t_val.iloc[:, TARGET_IDX]
    y_train_full = pd.concat([y_train, y_val])

    target_name = target_name_dict[TARGET_IDX]
    all_preds = {}

    # QRF
    qrf_grid = deepcopy(config['tuning']['qrf_grid'])
    if RUN_LOCALLY:
        qrf_grid = {'n_estimators': [50], 'max_depth': [3]}
    qrf_preds = fit_qrf(
        X_train_full, y_train_full, X_test, QUANTILES,
        target_name, YEAR, qrf_grid, K_FOLDS, SHELF_TUNING_LOG_PATH
    )
    all_preds.update(qrf_preds)

    # QGB
    qgb_grid = deepcopy(config['tuning']['qgb_grid'])
    if RUN_LOCALLY:
        qgb_grid['n_estimators'] = [10]
        qgb_grid['learning_rate'] = [0.1]
        qgb_grid['subsample'] = [1.0]
        qgb_grid['max_depth'] = [1]
    qgb_preds = fit_qgb(
        X_train_full, y_train_full, X_test, QUANTILES,
        target_name, YEAR, qgb_grid, K_FOLDS, SHELF_TUNING_LOG_PATH
    )
    all_preds.update(qgb_preds)

    preds_df = pd.DataFrame(
        all_preds,
        index=X_test.index
    )
    output_path = SHELF_PRED_DIR / f"tree_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}_{YEAR}.csv"
    preds_df.to_csv(output_path)
    logging.info(f"Tree model predictions saved to {output_path}")


def train_lit_bench_models():
    """Train literature benchmark models (VG, IAR, UAR)."""
    logging.info(f"Training literature benchmark models for {target_name_dict[TARGET_IDX]} ({YEAR})...")
    
    target_path = DATA_DIR / TARGET_FILE
    
    # Get the specific benchmark's features
    model_file = model_file_dict[TARGET_IDX]
    model_name = model_name_dict[TARGET_IDX]
    target_name = target_name_dict[TARGET_IDX]

    (
        X_train, X_val, X_test,
        t_train, t_val, t_test
    ) = prepare_non_rnn_data(
        targets_path=target_path,
        input_paths=[DATA_DIR / model_file],
        start_date='1974-02-01',
        train_cutoff_year=YEAR,
        val_split_style='date',
        val_months=data_config['val_months'],
        test_months=data_config['test_months'],
        target_scale_factor=data_config['target_scale_factor']
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
        preds[f'{model_name}_Q{Q}'] = res.predict(add_constant(X_test.values, has_constant='skip'))
    
    # Save predictions
    output_path = LIT_BENCH_PRED_DIR / f"lit_bench_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}_{YEAR}.csv"
    pd.DataFrame(preds, index=X_test.index).to_csv(output_path)
    print(f"Literature benchmark predictions saved to {output_path}")


def train_deep_models():
    """Train deep learning models"""
    logging.info(f"Training deep models for {target_name_dict[TARGET_IDX]} ({YEAR})...")

    # Prepare data (including RNN sequences)
    (
        X_train, X_val, X_test,
        t_train, t_val, t_test
    ) = prepare_rnn_data(
        targets_path=target_path,
        input_paths=input_paths,
        start_date='1961-01-01',
        train_cutoff_year=YEAR,
        val_split_style='date',
        n_timesteps=config['time_steps'],
        val_months=data_config['val_months'],
        test_months=data_config['test_months'],
        target_scale_factor=data_config['target_scale_factor']
    )

    X_train_full = pd.concat([X_train, X_val])

    y_train = t_train.iloc[:, TARGET_IDX]
    y_val = t_val.iloc[:, TARGET_IDX]
    y_train_full = pd.concat([y_train, y_val])

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
    
    # Get target names
    target_name = target_name_dict[TARGET_IDX]
    
    # Custom objects for loading models
    path_quantiles = [int(q*100) for q in QUANTILES]
    custom_objects = {
        **{f'tilted_loss_{Q}': make_tilted_loss(Q) for Q in path_quantiles},
        **{f"total_tilted_loss_{'_'.join(map(str, path_quantiles))}": make_total_tilted_loss(QUANTILES)}
    }
    
    # Early stopping configuration
    early_stopping_args = {
        'monitor': 'val_loss',
        'min_delta': config['early_stopping']['min_delta'],
        'patience': config['early_stopping']['patience'],
        'restore_best_weights': True,
        'verbose': 0
    }
    
    model_builder_params_cfg = config['builder_params']['deep_models']
    model_names = list(model_builder_params_cfg.keys())
    
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
    
    # Train each model variant
    for model_type in model_names:
        study_name = f'{model_type}_{target_name}_{YEAR}'
        logging.info(f"Training {model_type}...")

        # Get basic model config
        builder_params = dict(model_builder_params_cfg[model_type])
        
        # Update builder params with runtime arguments
        builder_params.update(
            {
                'input_shape': X_train.shape[1:],
                'loss_weights': config['loss_weights'],
                'seed': SEED,
            }
        )
        builder_params['lower_quantiles'] = [q for q in QUANTILES if q < 0.5]
        builder_params['upper_quantiles'] = [q for q in QUANTILES if q > 0.5]

        # set fit params
        fit_params = {
            'epochs': 1 if RUN_LOCALLY else config['epochs'],
            'batch_size': 32 if RUN_LOCALLY else config['batch_size'],
            'validation_data': validation_data,
            'verbose': 0,
            'shuffle': config['shuffle']
        }
        
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
                val_size=config['val_size'],
                n_splits=config['k_folds'],
                builder_func=build_dmq_v0,
                fit_params=fit_params,
                early_stopping_args=early_stopping_args,
                grid=config['tuning']['dmq_grid'],
                study_name=study_name,
                trials=1 if RUN_LOCALLY else config['trials'],
                n_jobs=os.cpu_count()-1,
                storage=storage,
                sampler=optuna.samplers.RandomSampler(seed=SEED),
                pruner=optuna.pruners.MedianPruner(),
                save_hps=True if OPTUNA_STORAGE == 'inmemory' else False,
                log_path=DEEP_TUNING_LOG_PATH
            )
        
        # Fit models with best hyperparameters
        estimators = fit_models(
            X_tr,
            y_tr,
            build_dmq_v0,
            model_name=study_name,
            hps=best_params,
            fit_params=fit_params,
            early_stopping_args=early_stopping_args,
            n_estimators=1 if RUN_LOCALLY else config['n_estimators'],
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
            end=f'{YEAR+1}-12-01', 
            freq='MS')
    )
    
    # Save to file
    output_path = DEEP_PRED_DIR / f"st_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}_{YEAR}.csv"
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
