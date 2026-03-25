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
from operator import itemgetter
import optuna
import statsmodels.formula.api as smf
import warnings
import tensorflow as tf

from src.preprocessing.prepare_quantile_data import prepare_quantile_data
from src.train.shelf_models import *
from src.train.losses import make_tilted_loss, make_total_tilted_loss
from src.train.models import build_dmq_v0, build_dmq_v1, build_dmq_v2
from src.train.tuning import CVObjective
from src.train.train_utils import fit_models
from src.utils.files import check_hps_exist, save_hyperparameters, load_hyperparameters

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # For parallelization

SEED = 1  # Set random seed for reproducibility
tf.random.set_seed(SEED)  # Set TensorFlow random seed

load_dotenv()

# ----- Configuration -----
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=config['logging_level'])

parser = argparse.ArgumentParser(description="Train models")
parser.add_argument("--year", type=int, required=True, help="Training cutoff year")
parser.add_argument("--target", type=int, required=True, help="Target index (0=Infl, 1=IP, 2=Unrate)")
parser.add_argument("--model-type", type=str, default="all", 
                    choices=["shelf", "deep", "all"],
                    help="Type of models to train")
parser.add_argument("--date", type=str, default=str(date.today()), help="Date for organizing outputs (default: today's date)")
parser.add_argument("--run-locally", action="store_true", help="Whether to run locally (reduces hyperparameter tuning for quick testing)")
parser.add_argument("--fit-lit-bench", action="store_true", help="Fit models from the literature. Only needs to be run once.")
args = parser.parse_args()

YEAR = args.year
TARGET_IDX = args.target
MODEL_TYPE = args.model_type
RUN_LOCALLY = args.run_locally
FIT_LIT_BENCH = args.fit_lit_bench

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

BASE_DIR = Path(os.getenv('REMOTE_BASE_DIR')) if not RUN_LOCALLY else Path(os.getenv('LOCAL_BASE_DIR'))

DATA_DIR = BASE_DIR / 'data' / 'processed'
SHELF_MODEL_DIR = BASE_DIR / 'models' / 'shelf_models' / DATE
SHELF_PRED_DIR = BASE_DIR / 'predictions' / 'shelf_preds' / DATE
SHELF_TUNING_LOG_PATH = BASE_DIR / 'tuning_logs' / f"shelf_tuning_log_{DATE}.json"
LIT_BENCH_PRED_DIR = BASE_DIR / 'predictions' / 'lit_bench_preds' / DATE
DEEP_MODEL_DIR = BASE_DIR / 'models' / 'st_models' / DATE 
DEEP_PRED_DIR = BASE_DIR / 'st_preds' / DATE
DEEP_TUNING_LOG_PATH =BASE_DIR / f"st_tuning_log_{DATE}.json"

for path in [SHELF_MODEL_DIR, SHELF_PRED_DIR, SHELF_TUNING_LOG_PATH.parent, LIT_BENCH_PRED_DIR, DEEP_MODEL_DIR, DEEP_PRED_DIR, DEEP_TUNING_LOG_PATH.parent]:
    os.makedirs(path, exist_ok=True)

target_name_dict = {0: 'Infl_yoy', 1: 'IP_yoy', 2: 'Unrate_yoy'}
model_file_dict = {
    0: f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_iar_x.parquet",
    1: f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_vg_x.parquet",
    2: f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_uar_x.parquet"
}
model_name_dict = {0: 'IAR', 1: 'VG', 2: 'UAR'}


def train_shelf_models():
    """Train shelf models (Naive, AR1, LR, LASSO, QRF, QGB)."""
    logging.info(f"Training shelf models for {target_name_dict[TARGET_IDX]} ({YEAR})...")
    
    # if RUN_LOCALLY:
    #     quantiles = [QUANTILES[0]]
    # else:
    quantiles = QUANTILES

    target_path = DATA_DIR / TARGET_FILE
    input_paths = [DATA_DIR / file for file in INPUT_FILES]
    
    non_rnn_data, _, meta_data = prepare_quantile_data(
        target=TARGET_IDX,
        time_steps=1,
        targets_path=target_path,
        input_paths=input_paths,
        start_date='1961-01-01',
        train_cutoff_year=YEAR,
        n_quantiles=len(QUANTILES),
        val_years=VAL_YEARS
    )
    
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_train_full, 
        y_train_full, 
        X_test, 
        all_y_train
    ) = itemgetter(
        'X_train',
        'y_train',
        'X_val',
        'y_val',
        'X_train_full', 
        'y_train_full', 
        'X_test', 
        'all_y_train'
    )(non_rnn_data)
    
    target_name = target_name_dict[TARGET_IDX]
    all_preds = {}
    
    # Naive models
    naive_preds = fit_dummy(X_train_full, y_train_full, X_test, quantiles)
    all_preds.update(naive_preds)
    
    # AR(1) models
    ar1_x_path = DATA_DIR / f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_ar1_x.parquet"
    X_ar1 = pd.read_parquet(ar1_x_path)
    X_train_ar1 = X_ar1.loc['1961-02-01':f'{YEAR}-12-01', f"{target_name}_t-1"]
    X_test_ar1 = X_ar1.loc[f'{YEAR+1}-01-01': f'{YEAR+1}-12-01', f"{target_name}_t-1"]
    y_train_ar1 = y_train_full.loc['1961-02-01':f'{YEAR}-12-01'] # Get rid of first date because NaN from lag

    ar1_preds = fit_ar1(X_train_ar1, y_train_ar1, X_test_ar1, quantiles, target_name, YEAR, verbose=False)
    all_preds.update(ar1_preds)
    
    # Linear models
    # linear_grids = {'LR': {}, 'LAS': {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]}}
    # if RUN_LOCALLY:
    #     linear_grids['LAS']['alpha'] = [1.0]  # Reduce for local runs

    # linear_preds = fit_linear_models(
    #     X_train_full, y_train_full, X_test, quantiles, 
    #     target_name, YEAR, linear_grids, K_FOLDS, SHELF_TUNING_LOG_PATH
    # )

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
        X_train,
        y_train,
        X_train_full,
        y_train_full,
        X_test,
        QUANTILES,
        target_name,
        YEAR,
        val_size=config['val_size'],
        k_folds=K_FOLDS,
        tuning_path=SHELF_TUNING_LOG_PATH,
        early_stopping_args=early_stopping_args,
        fit_params=fit_params,
        seed=SEED,
        trials=1 if RUN_LOCALLY else config['trials'],
        n_estimators= 1 if RUN_LOCALLY else config['n_estimators'],
        model_dir_path=SHELF_MODEL_DIR
    )

    all_preds.update(linear_preds)
    
    # QRF
    qrf_grid = {'n_estimators': [100, 500, 1000], 'max_depth': list(range(1, 13))}
    if RUN_LOCALLY:
        qrf_grid = {'n_estimators': [50], 'max_depth': [3]}
    qrf_preds = fit_qrf(
        X_train_full, y_train_full, X_test, quantiles,
        target_name, YEAR, qrf_grid, K_FOLDS, SHELF_TUNING_LOG_PATH
    )
    all_preds.update(qrf_preds)
    
    # QGB
    qgb_grid = {
        'learning_rate' : [0.1, 0.01, 0.001],
        'n_estimators' : [50, 100, 200],
        'subsample' : [0.25, 0.5, 1.0], 
        'max_depth' : list(range(1, 6+1))
    }
    if RUN_LOCALLY:
        qgb_grid['n_estimators'] = [10]  # Reduce for local runs
        qgb_grid['learning_rate'] = [0.1]  
        qgb_grid['subsample'] = [1.0]  
        qgb_grid['max_depth'] = [1]
    
    qgb_preds = fit_qgb(
        X_train_full, y_train_full, X_test, quantiles,
        target_name, YEAR, qgb_grid, K_FOLDS, SHELF_TUNING_LOG_PATH
    )
    all_preds.update(qgb_preds)
    
    # Save predictions
    preds_df = pd.DataFrame(
        all_preds,
        index=pd.date_range(start=meta_data['test_start'], end=meta_data['test_end'], freq='MS')
    )
    output_path = SHELF_PRED_DIR / f"shelf_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}_{YEAR}.csv"
    preds_df.to_csv(output_path)
    logging.info(f"Shelf model predictions saved to {output_path}")


def train_lit_bench_models():
    """Train literature benchmark models (VG, IAR, UAR)."""
    logging.info(f"Training literature benchmark models for {target_name_dict[TARGET_IDX]} ({YEAR})...")
    
    target_path = DATA_DIR / TARGET_FILE
    
    # Get the model file for the target
    model_file = model_file_dict[TARGET_IDX]
    model_name = model_name_dict[TARGET_IDX]
    target_name = target_name_dict[TARGET_IDX]
    
    lit_bench_data, _, _ = prepare_quantile_data(
        target=target_name,
        time_steps=1,
        targets_path=target_path,
        input_paths=[DATA_DIR / model_file],
        start_date='1974-02-01',
        train_cutoff_year=YEAR,
        n_quantiles=len(QUANTILES),
        val_years=5
    )
    
    (X_train, y_train, X_test) = itemgetter(
        'X_train_full', 'y_train_full', 'X_test'
    )(lit_bench_data)
    
    # Fit quantile regression models
    preds = {}
    for q in QUANTILES:
        Q = int(q * 100)
        model = QuantReg(y_train.values.flatten(), add_constant(X_train.values, has_constant='skip'))
        res = model.fit(q=q)
        preds[f'{model_name}_Q{Q}'] = res.predict(add_constant(X_test.values, has_constant='skip'))
    
    # Save predictions
    output_path = LIT_BENCH_PRED_DIR / f"lit_bench_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}_{YEAR}.csv"
    pd.DataFrame(preds).to_csv(output_path)
    print(f"Literature benchmark predictions saved to {output_path}")


def train_deep_models():
    """Train deep learning models (DMQv0, DMQv1, DMQv2)."""
    logging.info(f"Training deep models for {target_name_dict[TARGET_IDX]} ({YEAR})...")
    
    # Adjust parameters for local runs
    trials = 1 if RUN_LOCALLY else config['trials']
    n_estimators = 1 if RUN_LOCALLY else config['n_estimators']
    
    target_path = DATA_DIR / TARGET_FILE
    input_paths = [DATA_DIR / file for file in INPUT_FILES]
    
    # Prepare data (including RNN sequences)
    non_rnn_data, rnn_data, meta_data = prepare_quantile_data(
        target=TARGET_IDX,
        time_steps=TIME_STEPS,
        targets_path=target_path,
        input_paths=input_paths,
        start_date='1961-01-01',
        train_cutoff_year=YEAR,
        n_quantiles=len(QUANTILES),
        val_years=VAL_YEARS
    )
    
    # Extract RNN data
    (
        mq_y_train_rnn, mq_y_val_rnn, mq_y_train_full_rnn,
        X_train_rnn, X_val_rnn, X_train_full_rnn, X_test_rnn,
    ) = itemgetter(
        'mq_y_train_rnn', 'mq_y_val_rnn', 'mq_y_train_full_rnn',
        'X_train_rnn', 'X_val_rnn', 'X_train_full_rnn', 'X_test_rnn',
    )(rnn_data)
    
    # Get target names
    all_y_train = non_rnn_data['all_y_train']
    target_name = target_name_dict[TARGET_IDX]
    
    # Custom objects for loading models
    path_quantiles = [int(q*100) for q in QUANTILES]
    custom_objects = {
        **{f'tilted_loss_{Q}': make_tilted_loss(Q) for Q in path_quantiles},
        **{f"total_tilted_loss_{'_'.join(map(str, path_quantiles))}": make_total_tilted_loss(QUANTILES)}
    }
    
    # Loss weights
    LOSS_WEIGHTS = config['loss_weights']
    
    # Early stopping configuration
    early_stopping_args = {
        'monitor': 'val_loss',
        'min_delta': 1e-3,
        'patience': 5,
        'restore_best_weights': True,
        'verbose': 0
    }
    
    # Define model configurations
    mq_model_params_dict = {
        'DMQv0': {
            'X_tr': X_train_full_rnn,
            'y_tr': mq_y_train_full_rnn,
            'validation_data': (X_val_rnn, mq_y_val_rnn),
            'X_te': X_test_rnn,
            'builder_fn': build_dmq_v0,
            'builder_params': { 
                'input_shape': X_train_rnn.shape[1:],
                'n_recurrent_layers': 3,
                'n_shared_layers': 2,
                'n_qtask_layers': 2,
                'n_recurrent_nodes': 64,
                'n_shared_nodes': 32,
                'n_task_nodes': 16,
                'recurrent_layer_type': 'lstm',
                'norm_fn': 'layer',
                'quantiles': QUANTILES,
                'loss_weights': LOSS_WEIGHTS,
                'seed': SEED
            }
        },
        'DMQv0c': {
            'X_tr': X_train_full_rnn,
            'y_tr': mq_y_train_full_rnn,
            'validation_data': (X_val_rnn, mq_y_val_rnn),
            'X_te': X_test_rnn,
            'builder_fn': build_dmq_v0,
            'builder_params': { 
                'input_shape': X_train_rnn.shape[1:],
                'n_recurrent_layers': 3,
                'n_shared_layers': 2,
                'n_qtask_layers': 2,
                'n_recurrent_nodes': 32,
                'n_shared_nodes': 32,
                'n_task_nodes': 32,
                'recurrent_layer_type': 'lstm',
                'norm_fn': 'layer',
                'quantiles': QUANTILES,
                'loss_weights': LOSS_WEIGHTS,
                'seed': SEED
            }
        },
        'DMQv1': {
            'X_tr': X_train_full_rnn,
            'y_tr': mq_y_train_full_rnn,
            'validation_data': (X_val_rnn, mq_y_val_rnn),
            'X_te': X_test_rnn,
            'builder_fn': build_dmq_v1,
            'builder_params': { 
                'input_shape': X_train_rnn.shape[1:],
                'n_recurrent_layers': 3,
                'n_shared_layers': 2,
                'n_qtask_layers': 2,
                'n_recurrent_nodes': 64,
                'n_shared_nodes': 32,
                'n_task_nodes': 16,
                'recurrent_layer_type': 'lstm',
                'norm_fn': 'layer',
                'lower_quantiles': [q for q in QUANTILES if q < 0.5],
                'upper_quantiles': [q for q in QUANTILES if q > 0.5],
                'loss_weights': LOSS_WEIGHTS,
                'seed': SEED
            }
        },
        'DMQv1c': {
            'X_tr': X_train_full_rnn,
            'y_tr': mq_y_train_full_rnn,
            'validation_data': (X_val_rnn, mq_y_val_rnn),
            'X_te': X_test_rnn,
            'builder_fn': build_dmq_v1,
            'builder_params': { 
                'input_shape': X_train_rnn.shape[1:],
                'n_recurrent_layers': 3,
                'n_shared_layers': 2,
                'n_qtask_layers': 2,
                'n_recurrent_nodes': 32,
                'n_shared_nodes': 32,
                'n_task_nodes': 32,
                'recurrent_layer_type': 'lstm',
                'norm_fn': 'layer',
                'lower_quantiles': [q for q in QUANTILES if q < 0.5],
                'upper_quantiles': [q for q in QUANTILES if q > 0.5],
                'loss_weights': LOSS_WEIGHTS,
                'seed': SEED
            }
        },
        'DMQv2': {
            'X_tr': X_train_full_rnn,
            'y_tr': mq_y_train_full_rnn,
            'validation_data': (X_val_rnn, mq_y_val_rnn),
            'X_te': X_test_rnn,
            'builder_fn': build_dmq_v2,
            'builder_params': { 
                'input_shape': X_train_rnn.shape[1:],
                'n_recurrent_layers': 3,
                'n_shared_layers': 2,
                'n_qtask_layers': 2,
                'n_recurrent_nodes': 64,
                'n_shared_nodes': 32,
                'n_task_nodes': 16,
                'recurrent_layer_type': 'lstm',
                'norm_fn': 'layer',
                'lower_quantiles': [q for q in QUANTILES if q < 0.5],
                'upper_quantiles': [q for q in QUANTILES if q > 0.5],
                'loss_weights': LOSS_WEIGHTS,
                'seed': SEED
            }
        },
        'DMQv2c': {
            'X_tr': X_train_full_rnn,
            'y_tr': mq_y_train_full_rnn,
            'validation_data': (X_val_rnn, mq_y_val_rnn),
            'X_te': X_test_rnn,
            'builder_fn': build_dmq_v2,
            'builder_params': { 
                'input_shape': X_train_rnn.shape[1:],
                'n_recurrent_layers': 3,
                'n_shared_layers': 2,
                'n_qtask_layers': 2,
                'n_recurrent_nodes': 32,
                'n_shared_nodes': 32,
                'n_task_nodes': 32,
                'recurrent_layer_type': 'lstm',
                'norm_fn': 'layer',
                'lower_quantiles': [q for q in QUANTILES if q < 0.5],
                'upper_quantiles': [q for q in QUANTILES if q > 0.5],
                'loss_weights': LOSS_WEIGHTS,
                'seed': SEED
            }
        }
    }
    
    # Optuna storage
    storage_url = optuna.storages.InMemoryStorage()
    all_model_preds = {}
    
    # Train each model variant
    for model_type in mq_model_params_dict.keys():
        study_name = f'{model_type}_{target_name}_{YEAR}'
        logging.info(f"Training {model_type}...")
        
        X_tr = mq_model_params_dict[model_type]['X_tr']
        y_tr = mq_model_params_dict[model_type]['y_tr']
        validation_data = mq_model_params_dict[model_type]['validation_data']
        X_te = mq_model_params_dict[model_type]['X_te']
        builder_fn = mq_model_params_dict[model_type]['builder_fn']
        builder_params = mq_model_params_dict[model_type]['builder_params']
        
        fit_params = {
            'epochs': 1 if RUN_LOCALLY else EPOCHS,
            'batch_size': 32 if RUN_LOCALLY else BATCH_SIZE,
            'validation_data': validation_data,
            'verbose': 0,
            'shuffle': False
        }
        
        # Check if hyperparameters already exist
        if check_hps_exist(study_name, DEEP_TUNING_LOG_PATH):
            logging.info(f"Hyperparameters for {study_name} already exist. Loading...")
            best_params = load_hyperparameters(study_name, DEEP_TUNING_LOG_PATH)
        else:
            logging.info(f"No existing hyperparameters for {study_name}, optimizing...")
            objective = CVObjective(
                X_tr=X_tr,
                y_tr=y_tr,
                val_size=config['val_size'],
                n_splits=K_FOLDS,
                builder_func=builder_fn,
                fit_params=fit_params,
                early_stopping_args=early_stopping_args,
                n_jobs=os.cpu_count(),
                tune_l1=False,
                tune_l2=True,
                tune_lr=True,
                tune_rec_drop=False,
                tune_dropout=False,
                tune_n_layers=False,
                tune_n_nodes=False,
                tune_norm=False,
                tune_recurrent_layer_type=False,
                **builder_params
            )
            
            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                storage=storage_url,
                load_if_exists=True,
                sampler=optuna.samplers.RandomSampler(SEED),
                pruner=None
            )
            
            study.optimize(
                objective,
                n_trials=trials,
                n_jobs=os.cpu_count()
            )
            
            best_params = study.best_params
            best_params.update(builder_params)
            
            save_hyperparameters(
                best_params,
                study_name,
                log_path=DEEP_TUNING_LOG_PATH
            )
        
        # Fit models with best hyperparameters
        estimators = fit_models(
            X_tr,
            y_tr,
            builder_fn,
            model_name=study_name,
            hps=best_params,
            fit_params=fit_params,
            early_stopping_args=early_stopping_args,
            n_estimators=n_estimators,
            models_dir_path=DEEP_MODEL_DIR,
            save_models=True,
            custom_objects=custom_objects
        )
        
        # Generate predictions
        preds = []
        for e in estimators:
            e_preds = e.predict(X_te).reshape(-1, len(QUANTILES))
            preds.append(e_preds[:, :, np.newaxis])
        
        preds = np.concatenate(preds, axis=2).mean(axis=2) # Take the mean over estimators
        
        # Store predictions for each quantile
        for i, q in enumerate(QUANTILES):
            Q = int(q * 100)
            all_model_preds[f'{model_type}_Q{Q}'] = preds[:, i]
    
    # Save predictions
    all_model_preds_df = pd.DataFrame(
        all_model_preds,
        index=pd.date_range(start=meta_data['test_start'], end=meta_data['test_end'], freq='MS')
    )
    
    # Create ensemble predictions (average all DMQ models)
    # all_dmq_preds = []
    # for model in mq_model_params_dict.keys():
    #     preds = all_model_preds_df.loc[:, f'{model}_Q5':f'{model}_Q95']
    #     all_dmq_preds.append(preds.values[:, :, np.newaxis])
    
    # all_dmq_preds = np.concatenate(all_dmq_preds, axis=2)
    # dmqe_preds = np.mean(all_dmq_preds, axis=2) # Take the mean over all DMQ models
    # dmqe_all_preds_df = pd.DataFrame(
    #     dmqe_preds,
    #     columns=[f'DMQe_all_Q{Q}' for Q in path_quantiles],
    #     index=all_model_preds_df.index
    # )
    
    # Combine individual and ensemble predictions
    # all_model_preds_df = pd.concat([all_model_preds_df, dmqe_all_preds_df], axis=1)
    
    # Save to file
    output_path = DEEP_PRED_DIR / f"st_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}_{YEAR}.csv"
    all_model_preds_df.to_csv(output_path)
    logging.info(f"Deep model predictions saved to {output_path}")


def main():
    """Run model training based on specified type."""

    if FIT_LIT_BENCH:
        train_lit_bench_models()

    if MODEL_TYPE == "shelf" or MODEL_TYPE == "all":
        train_shelf_models()
    
    if MODEL_TYPE == "deep" or MODEL_TYPE == "all":
        train_deep_models()
    
    logging.info("Training complete.")


if __name__ == "__main__":
    main()
