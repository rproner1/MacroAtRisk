# ******************************** Imports ********************************
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np

# Data Libraries
import warnings
warnings.filterwarnings("ignore")

# Machine Learning libraries
import tensorflow as tf
SEED = 1  # Set random seed for reproducibility
tf.random.set_seed(SEED)  # Set TensorFlow random seed

import optuna
from sklearn.linear_model import QuantileRegressor, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from quantile_forest import RandomForestQuantileRegressor
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools import add_constant

from src.preprocessing.prepare_quantile_data import prepare_quantile_data
from src.train.models import fit_qpcr


from datetime import date
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # For parallelization

import argparse
from operator import itemgetter
from pathlib import Path
import yaml

from src.utils.files import check_hps_exist, save_hyperparameters, load_hyperparameters

with open("./config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# ******************************** Arguments ********************************

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

if RUN_LOCALLY: 
    QUANTILES = [0.50]  # Only median for local testing

path_quantiles = [int(q*100) for q in QUANTILES]  # Quantiles as integers (e.g., 5 for 0.05) for file names

# ******************************** Paths ********************************

if RUN_LOCALLY:
    DATA_DIR = Path(os.getenv('LOCDATADIR')) / 'processed/'
    MODEL_DIR = Path(os.getenv('LOCMODELDIR')) / 'shelf_models' / DATE
    PRED_DIR = Path(os.getenv('LOCPREDDIR')) / 'shelf_preds' / DATE
    tuning_log_path = Path(os.getenv('LOCTUNINGDIR')) / f"shelf_tuning_log_{DATE}.json" 
else:
    DATA_DIR = Path(os.getenv('DATADIR')) / 'processed/'
    MODEL_DIR = Path(os.getenv('MODELDIR')) / 'shelf_models' / DATE
    PRED_DIR = Path(os.getenv('PREDDIR')) / 'shelf_preds' / DATE
    tuning_log_path = Path(os.getenv('TUNINGDIR')) / f"shelf_tuning_log_{DATE}.json"

storage_url = optuna.storages.InMemoryStorage()

for path in [MODEL_DIR, PRED_DIR, tuning_log_path.parent]:
    os.makedirs(path, exist_ok=True)

    
# ******************************** Data ********************************

input_files = config['input_files']
INPUT_PATHS = [DATA_DIR / file for file in input_files]
TARGETS_PATH = DATA_DIR / config['target_file']

non_rnn_data, rnn_data, meta_data = prepare_quantile_data(
    target=TARGET_IDX,
    time_steps=1, # No impact for shelf models.
    targets_path=TARGETS_PATH, input_paths=INPUT_PATHS,
    start_date='1961-01-01', train_cutoff_year=YEAR, 
    n_quantiles=len(QUANTILES), val_years=5
)

(
    X_train_full, y_train_full,
    X_train, X_val, X_test,
    y_train, y_val,
    all_y_train
) = itemgetter(
    'X_train_full', 'y_train_full', # Full training set for tree-based models
    'X_train', 'X_val', 'X_test', # Static inputs
    'y_train', 'y_val', # multi-task multi-quantile static outputs
    'all_y_train'
)(non_rnn_data)

# Create PredefinedSplit for static models
val_fold = [-1] * len(X_train) + [0] * len(X_val)  # -1 for training, 0 for validation
pds = PredefinedSplit(val_fold)


linear_models ={
    **{f'LR_Q{Q}': QuantileRegressor(quantile=q, alpha=0.0) for Q, q in zip(path_quantiles, QUANTILES)},
    **{f'LAS_Q{Q}': QuantileRegressor(quantile=q) for Q, q in zip(path_quantiles, QUANTILES)}
}
linear_model_grids ={
    'LR': {},
    'LAS': {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
}
all_model_preds = {}

target_name_dict = {i: name for i, name in enumerate(all_y_train.columns)}

target_name = target_name_dict[TARGET_IDX]

for model in linear_models:

    print(f"Training model {model}...")

    model_name = f"{model}_{target_name}_{YEAR}"
    grid = linear_model_grids[model.split('_')[0]]

    split = pds
    X_tr, y_tr = (X_train_full, y_train_full)
    X_te = X_test

    if check_hps_exist(model_name, tuning_log_path):
        print(f"Hyperparameters for {model_name} already exist. Loading...")
        
        best_params = load_hyperparameters(model_name, tuning_log_path)

        best_fit = linear_models[model].set_params(**best_params)
        best_fit.fit(X_tr, y_tr.values.flatten())

    else: 
        print(f"Tuning {model_name}...")
        search = GridSearchCV(estimator=linear_models[model], param_grid=grid, cv=K_FOLDS, n_jobs=-1)
        best_fit = search.fit(X_tr, y_tr.values.flatten())
        best_params = best_fit.best_params_
        save_hyperparameters(best_params, model_name, tuning_log_path)

    all_model_preds[model] = best_fit.predict(X_te)


# QGB
for Q, q in zip(path_quantiles, QUANTILES):
    study_name = f'QGB_Q{Q}_{target_name}_{YEAR}'

    # Check if study already exists
    gbt = GradientBoostingRegressor(random_state=1, loss='quantile', alpha=q)
    if check_hps_exist(study_name, tuning_log_path):
        print(f"Study {study_name} already exists. Loading...")
        best_params = load_hyperparameters(study_name, tuning_log_path)
        gbt.set_params(**best_params)
        best_fit = gbt.fit(X_train_full, y_train_full.values.flatten())
    else:
        print(f'Tuning {study_name}...')
        gbt_grid = {'learning_rate' : [0.1, 0.01, 0.001],'n_estimators' : [50, 100, 200],'subsample' : [0.25, 0.5, 1.0], 'max_depth' : list(range(1, 6+1, 2))}
        grid_search = GridSearchCV(gbt, gbt_grid, refit=True, cv=K_FOLDS, n_jobs=-1)

        # Perform grid search and refit with best params
        best_fit = grid_search.fit(X_train_full, y_train_full.values.flatten())

        # Save hps
        best_params = best_fit.best_params_
        save_hyperparameters(best_params, study_name, tuning_log_path)

    all_model_preds[f'QGB_Q{Q}'] = best_fit.predict(X_test)

# QRF (learns all quantiles at the same time)
study_name = f'QRF_{target_name}_{YEAR}'
qrf = RandomForestQuantileRegressor(random_state=1, max_features='sqrt')
# Check if study already exists
if check_hps_exist(study_name, tuning_log_path):
    print(f"Study {study_name} already exists. Loading...")
    best_params = load_hyperparameters(study_name, tuning_log_path)
    
else:
    print(f'Tuning {study_name}...')

    qrf_grid = {'n_estimators':[100,500,1000], 'max_depth':list(range(1, 12+1))}
    grid_search = GridSearchCV(qrf, qrf_grid, refit=True, cv=K_FOLDS, n_jobs=-1)

    best_fit = grid_search.fit(X_train_full, y_train_full.values.flatten())

    # Save hps
    best_params = best_fit.best_params_
    save_hyperparameters(best_params, study_name, tuning_log_path)
qrf.set_params(**best_params)
qrf.fit(X_train_full, y_train_full.values.flatten())
preds = qrf.predict(X_test, quantiles=QUANTILES).reshape(-1,len(QUANTILES))  # Shape (n_samples, n_quantiles)
for i, Q in enumerate(path_quantiles):
    all_model_preds[f'QRF_Q{Q}'] = preds[:, i]

# QPCR 
for q in QUANTILES:

    Q = int(q*100)

    study_name = f'QPCR_Q{Q}_{TARGET_IDX}_{YEAR}'

    if not os.path.exists(f'{MODEL_DIR}{study_name}_bestsubset.npy'):
        print(f'Training {study_name}...')

        qpcr, best_subset = fit_qpcr(
            X=X_train_full.values, # Constant added inside function
            y=y_train_full.values, 
            q=q
        )

        np.save(f'{MODEL_DIR}{study_name}_bestsubset.npy', np.array(best_subset))

    else:

        print(f'{study_name} already exists. Loading...')

        best_subset = list(np.load(f'{MODEL_DIR}{study_name}_bestsubset.npy', allow_pickle=True).flatten())

        X_train_subset = X_train_full.values[:, best_subset]
        qpcr = QuantReg(y_train_full.values, add_constant(X_train_subset, has_constant='skip')).fit(q=q)

    # print(f"best_subset: {best_subset}")

    X_test_subset = X_test.values[:, best_subset]
    preds = qpcr.predict(add_constant(X_test_subset, has_constant='skip')).flatten()

    all_model_preds[f'QPCR_Q{Q}'] = preds

# Benchmark AR(1) model
ar1_x_path = DATA_DIR / f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_ar1_x.parquet"
X_ar_1 = pd.read_parquet(ar1_x_path)
X_ar_1_train = X_ar_1.loc['1961-02-01':f'{YEAR}-12-01', f"{target_name}_t-1"]
X_ar_1_test = X_ar_1.loc[f'{YEAR+1}-01-01': f'{YEAR+1}-12-01', f"{target_name}_t-1"]
y_train_full_ar = y_train_full.loc['1961-02-01':f'{YEAR}-12-01'] # Get rid of first date because NaN from lag

for q in QUANTILES: 
    ar_1 = QuantileRegressor(quantile=q)
    ar_1.fit(X_ar_1_train.values.reshape(-1,1), y_train_full_ar.loc[X_ar_1_train.index].values)
    preds = ar_1.predict(X_ar_1_test.values.reshape(-1,1)) # Shape (n_samples,)
    all_model_preds[f'AR1_Q{int(q*100)}'] = preds

# Train an AR(1) specifically for the mean
ar1_mean = LinearRegression()
ar1_mean.fit(X_ar_1_train.values.reshape(-1,1), y_train_full_ar.loc[X_ar_1_train.index].values)
mean_preds = ar1_mean.predict(X_ar_1_test.values.reshape(-1,1))
all_model_preds['AR1_Mean'] = mean_preds

# Naive Regressor 
from sklearn.dummy import DummyRegressor

for q in QUANTILES:
    Q = int(q*100)
    naive = DummyRegressor(strategy='quantile', quantile=q)
    # X does not matter for DummyRegressor but required for consistency
    naive.fit(X_train_full, y_train_full)
    preds = naive.predict(X_test)
    all_model_preds[f'Naive_Q{Q}'] = preds

# Naive Mean Regressor

naive_mean = DummyRegressor(strategy='mean')
naive_mean.fit(X_train_full, y_train_full)
mean_preds = naive_mean.predict(X_test)
all_model_preds['Naive_Mean'] = mean_preds

all_model_preds_df = pd.DataFrame(all_model_preds, index=pd.date_range(start=meta_data['test_start'], end=meta_data['test_end'], freq='MS'))
preds_path = PRED_DIR / f"shelf_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}_{YEAR}.csv"
all_model_preds_df.to_csv(preds_path)

print('COMPLETE.')