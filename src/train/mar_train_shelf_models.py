# ******************************** Imports ********************************

# Data Libraries
import warnings
warnings.filterwarnings("ignore")

# Machine Learning libraries
import tensorflow as tf
import optuna
from sklearn.linear_model import QuantileRegressor, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from quantile_forest import RandomForestQuantileRegressor

from datetime import date
import os # for checking if files exist
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # For parallelization

import argparse
from operator import itemgetter
from utils.utils import *

SEED = 1  # Set random seed for reproducibility
tf.random.set_seed(SEED)  # Set TensorFlow random seed


# ******************************** Arguments ********************************

parser = argparse.ArgumentParser(description="Tune MTMQ/MQ models")
parser.add_argument("--year", type=int, required=True, help="train cutoff year")
parser.add_argument("--target", type=int, required=True, help="target variable index")
parser.add_argument("--country", type=str, default="us", help="country code (us/ca)")
parser.add_argument("--horizon", type=int, default=4, help="forecast horizon in quarters")
parser.add_argument("--trials", type=int, default=50, help="number of trials per model/layer combination")
parser.add_argument("--time-steps", type=int, default=12, help="number of time steps for RNN models")
parser.add_argument("--quantiles", type=float, nargs="*", default=[0.05,0.25,0.50,0.75,0.95], help="list of quantiles to predict")
parser.add_argument("--overwrite-log", action="store_true", help="overwrite existing log file")
parser.add_argument("--local", action="store_true", help="run locally (use local data/DB)")
parser.add_argument("--n-estimators", type=int, default=5, help="number of estimators per ensemble model")
parser.add_argument("--k-folds", type=int, default=10, help="number of folds for cross-validation")
parser.add_argument("--date", type=str, default=str(date.today()), help="date string for file paths")
args = parser.parse_args()

print(f"Arguments: {args}")

YEAR = args.year
COUNTRY = args.country
HORIZON_IN_QUARTERS = args.horizon
RUN_LOCALLY = args.local
OVERWRITE_LOG = args.overwrite_log
QUANTILES = args.quantiles
TARGET_IDX = args.target  
TRIALS = args.trials
TIME_STEPS = args.time_steps
N_ESTIMATORS = args.n_estimators
K_FOLDS = args.k_folds
DATE = args.date

if RUN_LOCALLY: 
    TRIALS = 1
    QUANTILES = [0.50]  # Only median for local testing
    N_ESTIMATORS = 1

path_quantiles = [int(q*100) for q in QUANTILES]  # Quantiles as integers (e.g., 5 for 0.05) for file names

# ******************************** Paths ********************************

if RUN_LOCALLY:
    DATA_DIR = "/home/rproner/Documents/Data/MacroAtRisk/"
    MODEL_DIR = "/home/rproner/Documents/Projects/MacroAtRisk/TestModels/"
    PRED_DIR = "/home/rproner/Documents/Projects/MacroAtRisk/TestPredictions/"
    tuning_log_path = f"localtest_tuning_log_{DATE}.json"
else:
    DATA_DIR = "/home/rproner/projects/rrg-camera/rproner/Data/MacroAtRisk/"
    TUNING_LOG_DIR = "/home/rproner/projects/rrg-camera/rproner/MacroAtRisk/TuningLogs/"
    MODEL_DIR = f"/home/rproner/projects/rrg-camera/rproner/MacroAtRisk/Models/Models_{DATE}/"
    PRED_DIR = f"/home/rproner/projects/rrg-camera/rproner/MacroAtRisk/Predictions/Shelf_Predictions_{DATE}/"
    tuning_log_path = f"{TUNING_LOG_DIR}st_tuning_log_{DATE}.json"
    os.makedirs(TUNING_LOG_DIR, exist_ok=True)

storage_url = optuna.storages.InMemoryStorage()

for path in [MODEL_DIR, PRED_DIR]:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)



# ******************************** Data ********************************

input_paths = [
    f'{DATA_DIR}{COUNTRY}_macro_predictors_{HORIZON_IN_QUARTERS}q_1961-01--2024-12.csv' 
    # f'{DATA_DIR}{COUNTRY}_oap_firm_avg_diff_financial_predictors_{HORIZON_IN_QUARTERS}q_1961-01--2024-12.csv'
]

non_rnn_data, rnn_data, meta_data = prepare_quantile_data(
    target=TARGET_IDX,
    time_steps=TIME_STEPS, 
    targets_path=f'{DATA_DIR}{COUNTRY}_targets_1961-01--2024-12.csv', input_paths=input_paths,
    start_date='1961-01-01', train_cutoff_year=YEAR, 
    n_quantiles=len(QUANTILES), val_years=5
)

(
    X_train_rnn, X_val_rnn, X_test_rnn,
    y_train_rnn, y_val_rnn,
) = itemgetter(
    'X_train_rnn', 'X_val_rnn', 'X_test_rnn', # Dynamic inputs
    'y_train_rnn', 'y_val_rnn' # multi-task multi-quantile dynamic outputs
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
    'all_y_train'
)(non_rnn_data)

# Data with lags
input_paths_lags = [
    f'{DATA_DIR}{COUNTRY}_macro_predictors_with_12lags_{HORIZON_IN_QUARTERS}q_1962-01--2024-12.csv', 
]
non_rnn_data_lags, _, _ = prepare_quantile_data(
    target=TARGET_IDX,
    time_steps=TIME_STEPS, 
    targets_path=f'{DATA_DIR}{COUNTRY}_targets_1961-01--2024-12.csv', input_paths=input_paths_lags,
    start_date='1962-01-01', train_cutoff_year=YEAR, 
    n_quantiles=len(QUANTILES), val_years=5
)
(
    X_train_full_lags, y_train_full_lags,
    X_train_lags, X_val_lags, X_test_lags,
) = itemgetter(
    'X_train_full', 'y_train_full', # Full training set for tree-based models
    'X_train', 'X_val', 'X_test', # Static inputs
)(non_rnn_data_lags)

# Create PredefinedSplit for static models
val_fold = [-1] * len(X_train) + [0] * len(X_val)  # -1 for training, 0 for validation
pds = PredefinedSplit(val_fold)

val_fold_lags = [-1] * len(X_train_lags) + [0] * len(X_val_lags)  # -1 for training, 0 for validation
pds_lags = PredefinedSplit(val_fold_lags)

linear_models ={
    **{f'LR_Q{Q}': QuantileRegressor(quantile=q, alpha=0.0) for Q, q in zip(path_quantiles, QUANTILES)},
    **{f'LAS_Q{Q}': QuantileRegressor(quantile=q) for Q, q in zip(path_quantiles, QUANTILES)},
    **{f'LAS-l_Q{Q}': QuantileRegressor(quantile=q) for Q, q in zip(path_quantiles, QUANTILES)},
}
linear_model_grids ={
    'LR': {},
    'LAS': {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
    'LAS-l': {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]},
}
all_model_preds = {}

target_name_dict = {i: name for i, name in enumerate(all_y_train.columns)}

target_name = target_name_dict[TARGET_IDX]

for model in linear_models:

    print(f"Training model {model}...")

    model_name = f"{model}_{target_name}_{YEAR}"
    grid = linear_model_grids[model.split('_')[0]]

    split = pds_lags if 'LAS-l' in model else pds
    X_tr, y_tr = (X_train_full_lags, y_train_full_lags) if 'LAS-l' in model else (X_train_full, y_train_full)
    X_te = X_test_lags if 'LAS-l' in model else X_test

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


# model_params_dict = {
#     **{f'LR_Q{Q}': {
#         'builder_fn': build_qlr, 
#         'mode': 'qlr',
#         'builder_params': {'q': Q}, 
#         'grid': {'lr': [1e-4, 1e-3], 'l1': [0.0], 'l2': [0.0]},
#         'validation_data': [X_val, y_val],
#         'X_tr': X_train, 'y_tr': y_train, 'X_te': X_test
#     } for Q in path_quantiles},
#     **{f'LAS_Q{Q}': {
#         'builder_fn': build_qlr, 
#         'mode': 'las',
#         'builder_params': {'q': Q}, 
#         'grid': {'lr': [1e-4, 1e-3], 'l1': [1e-5, 1e-4, 1e-3], 'l2': [0.0]},
#         'validation_data': [X_val, y_val],
#         'X_tr': X_train, 'y_tr': y_train, 'X_te': X_test
#     } for Q in path_quantiles},
#     **{f'RID_Q{Q}': {
#         'builder_fn': build_qlr, 
#         'mode': 'rid',
#         'builder_params': {'q': Q},
#         'grid': {'lr': [1e-4, 1e-3], 'l1': [0.0], 'l2': [1e-5, 1e-4, 1e-3]}, 
#         'validation_data': [X_val, y_val],
#         'X_tr': X_train, 'y_tr': y_train, 'X_te': X_test
#     } for Q in path_quantiles},
#     **{f'EN_Q{Q}': {
#         'builder_fn': build_qlr, 
#         'mode': 'en',
#         'builder_params': {'q': Q}, 
#         'grid': {'lr': [1e-4, 1e-3], 'l1': [1e-5, 1e-4, 1e-3], 'l2': [1e-5, 1e-4, 1e-3]},
#         'validation_data': [X_val, y_val],
#         'X_tr': X_train, 'y_tr': y_train, 'X_te': X_test,
#     } for Q in path_quantiles},
#      **{f'RNN_Q{Q}': {
#         'builder_fn': build_rnn, 
#         'mode': 'rnn',
#         'builder_params': {
#             'n_recurrent_layers': 2,  # Number of recurrent layers
#             'n_dense_layers': 1,  # Number of shared layers
#             'n_nodes': 32,  # Number of nodes
#             'recurrent_layer_type': 'gru',
#             'q': Q
#         },
#         'grid': {'lr': [1e-4, 1e-3], 'l1': [1e-5, 1e-4, 1e-3], 'l2': [1e-5, 1e-4, 1e-3]},
#         'validation_data': [X_val_rnn, y_val_rnn],
#         'X_tr': X_train_rnn, 'y_tr': y_train_rnn, 'X_te': X_test_rnn
#     } for Q in path_quantiles}
# }


# custom_objects = {
#     **{f'tilted_loss_{Q}': make_tilted_loss(Q) for Q in path_quantiles},
#     **{f"total_tilted_loss_{'_'.join(map(str, path_quantiles))}": make_total_tilted_loss(QUANTILES)}
# }

# early_stopping_args = {
#     'monitor': 'val_loss',
#     'min_delta': 1e-3,
#     'patience': 5,
#     'restore_best_weights': True,
#     'verbose': 0
# }

# for model in model_params_dict:

#     study_name = f"{model}_{target_name}_{YEAR}" # model already includes quantile info

#     # Load model-specific params/args
#     X_tr = model_params_dict[model]['X_tr']
#     y_tr = model_params_dict[model]['y_tr']
#     validation_data = model_params_dict[model]['validation_data']
#     X_te = model_params_dict[model]['X_te']
#     builder_fn = model_params_dict[model]['builder_fn']
#     mode = model_params_dict[model]['mode']
#     builder_params = model_params_dict[model]['builder_params']
#     fit_params = {
#         'epochs': 100,
#         'batch_size': 4,
#         'validation_data': validation_data,
#         'verbose':0
#     }

#     # Check if hyperparameters already exist
#     if check_hps_exist(study_name, tuning_log_path):
#         print(f"Hyperparameters for {study_name} already exist. Loading...")
        
#         best_params = load_hyperparameters(study_name, tuning_log_path)

#     else: 
#         objective = Objective(X_tr=X_tr, y_tr=y_tr, builder_func=builder_fn, fit_params=fit_params, early_stopping_args=early_stopping_args, mode=mode, **builder_params)

#         study = optuna.create_study(
#             direction="minimize", # Minimize validation loss
#             study_name=study_name,
#             storage=storage_url,
#             load_if_exists=True,
#             sampler=optuna.samplers.RandomSampler(SEED),
#             pruner=None
#         )

#         study.optimize(
#             objective,
#             n_trials=TRIALS,
#             n_jobs=-1
#         )

#         best_params = study.best_params
#         best_params.update(builder_params)
#         save_hyperparameters(
#             best_params, 
#             study_name, 
#             log_path=tuning_log_path,
#             overwrite=OVERWRITE_LOG
#         )

#     estimators = fit_models(
#         X_tr, 
#         y_tr,
#         builder_fn,
#         model_name=study_name,
#         hps=best_params,
#         fit_params=fit_params,
#         early_stopping_args=early_stopping_args,
#         n_estimators=N_ESTIMATORS,
#         models_dir_path=MODEL_DIR,
#         save_models=True,
#         custom_objects=custom_objects
#     )    

#     preds = []
#     for e in estimators:
#         e_pred = e.predict(X_te).reshape(-1,1)
#         preds.append(e_pred)
#     preds = np.concatenate(preds, axis=1).mean(axis=1)

#     all_model_preds[model] = preds

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

# QRF-l (learns all quantiles at the same time)
study_name = f'QRF-l_{target_name}_{YEAR}'
qrf = RandomForestQuantileRegressor(random_state=1, max_features='sqrt')
# Check if study already exists
if check_hps_exist(study_name, tuning_log_path):
    print(f"Study {study_name} already exists. Loading...")
    best_params = load_hyperparameters(study_name, tuning_log_path)
    
else:
    print(f'Tuning {study_name}...')

    qrf_grid = {'n_estimators':[100,500,1000], 'max_depth':list(range(1, 12+1))}
    grid_search = GridSearchCV(qrf, qrf_grid, refit=True, cv=K_FOLDS, n_jobs=-1)

    best_fit = grid_search.fit(X_train_full_lags, y_train_full_lags.values.flatten())

    # Save hps
    best_params = best_fit.best_params_
    save_hyperparameters(best_params, study_name, tuning_log_path)

qrf.set_params(**best_params)
qrf.fit(X_train_full_lags, y_train_full_lags.values.flatten())
preds = qrf.predict(X_test_lags, quantiles=QUANTILES).reshape(-1,len(QUANTILES))  # Shape (n_samples, n_quantiles)
for i, Q in enumerate(path_quantiles):
    all_model_preds[f'QRF-l_Q{Q}'] = preds[:, i]

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

    print(f"best_subset: {best_subset}")
    X_test_subset = X_test.values[:, best_subset]
    preds = qpcr.predict(add_constant(X_test_subset, has_constant='skip')).flatten()

    all_model_preds[f'QPCR_Q{Q}'] = preds

# Benchmark AR(1) model
X_ar_1 = pd.read_csv(f'{DATA_DIR}{COUNTRY}_ar1_predictors_{HORIZON_IN_QUARTERS}q_2024-12.csv', index_col=0, parse_dates=True)
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
all_model_preds_df.to_csv(f"{PRED_DIR}shelf_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}_{YEAR}.csv")

print('COMPLETE.')