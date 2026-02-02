# ******************************** Imports ********************************
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd

load_dotenv()

# Data Libraries
import warnings
warnings.filterwarnings("ignore")

# Machine Learning libraries
import tensorflow as tf
import optuna
from sklearn.linear_model import LinearRegression
from src.preprocessing.prepare_quantile_data import prepare_quantile_data
from src.train.losses import make_tilted_loss, make_total_tilted_loss
from src.train.models import build_dmq_v0, build_dmq_v1, build_dmq_v2


from keras.callbacks import EarlyStopping
from datetime import date
import os # for checking if files exist
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # For parallelization

import argparse
from operator import itemgetter
# from utils.utils import *
import yaml

SEED = 1  # Set random seed for reproducibility
tf.random.set_seed(SEED)  # Set TensorFlow random seed

# ******************************** Arguments ********************************

from src.utils.files import check_hps_exist, save_hyperparameters, load_hyperparameters

with open("./config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


parser = argparse.ArgumentParser(description="Tune MTMQ/MQ models")
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

TIME_STEPS = config['time_steps']
TRIALS = config['trials']
N_ESTIMATORS = config['n_estimators']
VAL_YEARS  = config['val_years']

path_quantiles = [int(q*100) for q in QUANTILES]  # Quantiles as integers (e.g., 5 for 0.05) for file names

LOSS_WEIGHTS = [0.28, 0.17, 0.11, 0.17, 0.28]

if RUN_LOCALLY: 
    TRIALS = 2
    N_ESTIMATORS = 2

# ******************************** Paths ********************************

if RUN_LOCALLY:
    DATA_DIR = Path(os.getenv('LOCDATADIR')) / 'processed/'
    MODEL_DIR = Path(os.getenv('LOCMODELDIR')) / 'st_models' / f"{DATE}/"
    PRED_DIR = Path(os.getenv('LOCPREDDIR')) /'st_preds' / f"{DATE}/"
    tuning_log_path = Path(os.getenv('LOCTUNINGDIR')) / f"st_tuning_log_{DATE}.json" 
else:
    DATA_DIR = Path(os.getenv('DATADIR')) / 'processed/'
    MODEL_DIR = Path(os.getenv('MODELDIR')) / 'st_models' / f"{DATE}/"
    PRED_DIR = Path(os.getenv('PREDDIR')) / 'st_preds' / f"{DATE}/"
    tuning_log_path = Path(os.getenv('TUNINGDIR')) / f"st_tuning_log_{DATE}.json"

storage_url = optuna.storages.InMemoryStorage()

for path in [MODEL_DIR, PRED_DIR]:
    os.makedirs(path, exist_ok=True)


# ******************************** Data ********************************

INPUT_FILES = config['input_files']
TARGET_FILE = config['target_file']
input_paths = [DATA_DIR / f for f in INPUT_FILES]

non_rnn_data, rnn_data, meta_data = prepare_quantile_data(
    target=TARGET_IDX,
    time_steps=TIME_STEPS, 
    targets_path=DATA_DIR / TARGET_FILE, input_paths=input_paths,
    start_date='1961-01-01', train_cutoff_year=YEAR, 
    n_quantiles=len(QUANTILES), val_years=VAL_YEARS
)

(
    mq_y_train_rnn, mq_y_val_rnn, mq_y_train_full_rnn,
    X_train_rnn, X_val_rnn, X_train_full_rnn, X_test_rnn,
) = itemgetter(
    'mq_y_train_rnn', 'mq_y_val_rnn', 'mq_y_train_full_rnn', # single-task multi-quantile dynamic outputs
    'X_train_rnn', 'X_val_rnn', 'X_train_full_rnn', 'X_test_rnn', # Dynamic inputs
)(rnn_data)

(
    all_y_train
) = itemgetter(
    'all_y_train'
)(non_rnn_data)

custom_objects = {
    **{f'tilted_loss_{Q}': make_tilted_loss(Q) for Q in path_quantiles},
    **{f"total_tilted_loss_{'_'.join(map(str, path_quantiles))}": make_total_tilted_loss(QUANTILES)}
}
all_model_preds = {}

target_name_dict = {i: name for i, name in enumerate(all_y_train.columns)}

target_name = target_name_dict[TARGET_IDX]

early_stopping_args = {
    'monitor': 'val_loss',
    'min_delta': 1e-3,
    'patience': 5,
    'restore_best_weights': True,
    'verbose': 0
}

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

for model_type in mq_model_params_dict.keys():

    study_name = f'{model_type}_{target_name}_{YEAR}'

    X_tr = mq_model_params_dict[model_type]['X_tr']
    y_tr = mq_model_params_dict[model_type]['y_tr']
    validation_data = mq_model_params_dict[model_type]['validation_data']
    X_te = mq_model_params_dict[model_type]['X_te']
    builder_fn = mq_model_params_dict[model_type]['builder_fn']
    builder_params = mq_model_params_dict[model_type]['builder_params']
    fit_params = {
        'epochs': 100,
        'batch_size': 4,
        'validation_data': validation_data,
        'verbose':0,
        'shuffle': False
    }

    # Check if hyperparameters already exist
    if check_hps_exist(study_name, tuning_log_path):
        print(f"Hyperparameters for {study_name} already exist. Loading...")
        
        best_params = load_hyperparameters(study_name, tuning_log_path)

    else: 
        print(f"No existing hyperparameters for {study_name}, optimizing...")
        objective = CVObjective(
            X_tr=X_tr, 
            y_tr=y_tr, 
            val_size=0.1,
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
            n_trials=TRIALS,
            n_jobs=os.cpu_count()
        )

        best_params = study.best_params
        best_params.update(builder_params)

        save_hyperparameters(
            best_params, 
            study_name, 
            log_path=tuning_log_path,
            overwrite=OVERWRITE_LOG
        )

    estimators = fit_models(
        X_tr,
        y_tr,
        builder_fn,
        model_name=study_name,
        hps=best_params,
        fit_params=fit_params,
        early_stopping_args=early_stopping_args,
        n_estimators=N_ESTIMATORS,
        models_dir_path=MODEL_DIR,
        save_models=True,
        custom_objects=custom_objects
    ) 

    preds = []
    for e in estimators: 
        e_preds = e.predict(X_te).reshape(-1, len(QUANTILES)) # shape (n_samples, n_quantiles)
        preds.append(e_preds[:,:,np.newaxis]) # shape (n_samples, n_quantiles, 1)

    preds = np.concatenate(preds, axis=2).mean(axis=2)

    # Generate predictions
    for i, q in enumerate(QUANTILES):
        Q = int(q*100)
        all_model_preds[f'{model_type}_Q{Q}'] = preds[:, i]

# Save predictions
all_model_preds_df = pd.DataFrame(all_model_preds, index=pd.date_range(start=f'{YEAR+1}-01-01', end=f'{YEAR+1}-12-01', freq='MS'))

all_dmq_preds = []
for model in mq_model_params_dict.keys():
    preds = all_model_preds_df.loc[:, f'{model}_Q5':f'{model}_Q95']
    all_dmq_preds.append(preds.values[:,:,np.newaxis])
all_dmq_preds = np.concatenate(all_dmq_preds, axis=2)
dmqe_preds = np.mean(all_dmq_preds, axis=2)
dmqe_all_preds_df = pd.DataFrame(dmqe_preds, columns=[f'DMQe_all_Q{Q}' for Q in [5,25,50,75,95]], index=all_model_preds_df.index)

all_model_preds_df = pd.concat([all_model_preds_df, dmqe_all_preds_df], axis=1)
all_model_preds_df.to_csv(f"{PRED_DIR}st_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}_{YEAR}.csv")


print('COMPLETE.')