
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import os
from pathlib import Path
from figures.make_quantile_plots import QUANTILES
from src.utils.files import check_hps_exist, load_hyperparameters, save_hyperparameters
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from quantile_forest import RandomForestQuantileRegressor
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools import add_constant
from sklearn.dummy import DummyRegressor

def fit_linear_models(
        X_train_full: pd.DataFrame, 
        y_train_full: pd.Series,
        X_test: pd.DataFrame,
        quantiles: list[float],
        target_name: str,
        year: int,
        grids: dict[dict],
        k_folds: int,
        tuning_log_path: Path
    ):

    preds = {}

    path_quantiles = [int(q*100) for q in quantiles]  # Quantiles as integers (e.g., 5 for 0.05) for file names

    linear_models ={
        **{f'LR_Q{Q}': QuantileRegressor(quantile=q, alpha=0.0) for Q, q in zip(path_quantiles, quantiles)},
        **{f'LAS_Q{Q}': QuantileRegressor(quantile=q) for Q, q in zip(path_quantiles, quantiles)}
    }

    for model in linear_models:

        print(f"Training model {model}...")

        model_name = f"{model}_{target_name}_{year}"
        grid = grids[model.split('_')[0]]

        X_tr, y_tr = (X_train_full, y_train_full)
        X_te = X_test

        if check_hps_exist(model_name, tuning_log_path):
            print(f"Hyperparameters for {model_name} already exist. Loading...")
            
            best_params = load_hyperparameters(model_name, tuning_log_path)

            best_fit = linear_models[model].set_params(**best_params)
            best_fit.fit(X_tr, y_tr.values.flatten())

        else: 
            print(f"Tuning {model_name}...")
            search = GridSearchCV(estimator=linear_models[model], param_grid=grid, cv=k_folds, n_jobs=-1)
            best_fit = search.fit(X_tr, y_tr.values.flatten())
            best_params = best_fit.best_params_
            save_hyperparameters(best_params, model_name, tuning_log_path)

        preds[model] = best_fit.predict(X_te)
    
    return preds

def fit_qgb(
    X_train_full: pd.DataFrame, 
    y_train_full: pd.Series,
    X_test: pd.DataFrame,
    quantiles: list[float],
    target_name: str,
    year: int,
    grid: dict[list],
    k_folds: int,
    tuning_log_path: Path
):

    preds = {}
    path_quantiles = [int(q*100) for q in quantiles]  # Quantiles as integers (e.g., 5 for 0.05) for file names

    for Q, q in zip(path_quantiles, quantiles):
        study_name = f'QGB_Q{Q}_{target_name}_{year}'

        # Check if study already exists
        gbt = GradientBoostingRegressor(random_state=1, loss='quantile', alpha=q)
        if check_hps_exist(study_name, tuning_log_path):
            print(f"Study {study_name} already exists. Loading...")
            best_params = load_hyperparameters(study_name, tuning_log_path)
            gbt.set_params(**best_params)
            best_fit = gbt.fit(X_train_full, y_train_full.values.flatten())
        else:
            print(f'Tuning {study_name}...')
            grid_search = GridSearchCV(gbt, grid, refit=True, cv=k_folds, n_jobs=-1)

            # Perform grid search and refit with best params
            best_fit = grid_search.fit(X_train_full, y_train_full.values.flatten())

            # Save hps
            best_params = best_fit.best_params_
            save_hyperparameters(best_params, study_name, tuning_log_path)

        preds[f'QGB_Q{Q}'] = best_fit.predict(X_test)
    return preds

def fit_qrf(
        X_train_full: pd.DataFrame, 
        y_train_full: pd.Series,
        X_test: pd.DataFrame,
        quantiles: list[float],
        target_name: str,
        year: int,
        grid: dict[list],
        k_folds: int,
        tuning_log_path: Path
    ):

    path_quantiles = [int(q*100) for q in quantiles]  # Quantiles as integers (e.g., 5 for 0.05) for file names
    preds = {}

    study_name = f'QRF_{target_name}_{year}'
    qrf = RandomForestQuantileRegressor(random_state=1, max_features='sqrt')
    # Check if study already exists
    if check_hps_exist(study_name, tuning_log_path):
        print(f"Study {study_name} already exists. Loading...")
        best_params = load_hyperparameters(study_name, tuning_log_path)
        
    else:
        print(f'Tuning {study_name}...')

        grid_search = GridSearchCV(qrf, grid, refit=True, cv=k_folds, n_jobs=-1)

        best_fit = grid_search.fit(X_train_full, y_train_full.values.flatten())

        # Save hps
        best_params = best_fit.best_params_
        save_hyperparameters(best_params, study_name, tuning_log_path)

    qrf.set_params(**best_params)
    qrf.fit(X_train_full, y_train_full.values.flatten())
    preds_array = qrf.predict(X_test, quantiles=quantiles).reshape(-1,len(quantiles))  # Shape (n_samples, n_quantiles)
    
    for i, Q in enumerate(path_quantiles):
        preds[f'QRF_Q{Q}'] = preds_array[:, i]

    return preds

def fit_qpcr(
        X_train_full: pd.DataFrame, 
        y_train_full: pd.Series,
        X_test: pd.DataFrame,
        quantiles: list[float],
        target_idx: int, 
        year: int,
        models_dir: Path
    ):

    preds_dict = {}

    for q in quantiles:

        Q = int(q*100)

        study_name = f'QPCR_Q{Q}_{target_idx}_{year}'

        if not os.path.exists( models_dir / f'{study_name}_bestsubset.npy'):
            print(f'Training {study_name}...')

            qpcr, best_subset = fit_qpcr(
                X=X_train_full.values, # Constant added inside function
                y=y_train_full.values, 
                q=q
            )

            np.save(models_dir / f'{study_name}_bestsubset.npy', np.array(best_subset))

        else:

            print(f'{study_name} already exists. Loading...')

            best_subset = list(np.load(f'{models_dir}{study_name}_bestsubset.npy', allow_pickle=True).flatten())

            X_train_subset = X_train_full.values[:, best_subset]
            qpcr = QuantReg(y_train_full.values, add_constant(X_train_subset, has_constant='skip')).fit(q=q)

        # print(f"best_subset: {best_subset}")

        X_test_subset = X_test.values[:, best_subset]
        preds = qpcr.predict(add_constant(X_test_subset, has_constant='skip')).flatten()

        preds_dict[f'QPCR_Q{Q}'] = preds

def fit_ar1(X_train_full: pd.DataFrame, y_train_full: pd.DataFrame, X_test: pd.DataFrame, quantiles: list[float], target_name: str, year: int):

    preds_dict = {}
    # X_ar_1 = pd.read_parquet(ar1_x_path)
    # X_ar_1_train = X_ar_1.loc['1961-02-01':f'{year}-12-01', f"{target_name}_t-1"]
    # X_ar_1_test = X_ar_1.loc[f'{year+1}-01-01': f'{year+1}-12-01', f"{target_name}_t-1"]
    # y_train_full_ar = y_train_full.loc['1961-02-01':f'{year}-12-01'] # Get rid of first date because NaN from lag

    for q in quantiles: 
        ar_1 = QuantileRegressor(quantile=q)
        ar_1.fit(X_train_full.values.reshape(-1,1), y_train_full.values)
        preds = ar_1.predict(X_test.values.reshape(-1,1)) # Shape (n_samples,)
        preds_dict[f'AR1_Q{int(q*100)}'] = preds

    # Train an AR(1) specifically for the mean
    ar1_mean = LinearRegression()
    ar1_mean.fit(X_train_full.values.reshape(-1,1), y_train_full.values)
    mean_preds = ar1_mean.predict(X_test.values.reshape(-1,1))
    preds_dict['AR1_Mean'] = mean_preds

    return preds_dict

def fit_dummy(X_train_full: pd.DataFrame, 
              y_train_full: pd.Series,
              X_test: pd.DataFrame,
              quantiles: list[float]):
    
    preds_dict = {}

    for q in quantiles:
        Q = int(q*100)
        naive = DummyRegressor(strategy='quantile', quantile=q)
        # X does not matter for DummyRegressor but required for consistency
        naive.fit(X_train_full, y_train_full)
        preds = naive.predict(X_test)
        preds_dict[f'Naive_Q{Q}'] = preds

    # Naive Mean Regressor
    naive_mean = DummyRegressor(strategy='mean')
    naive_mean.fit(X_train_full, y_train_full)
    mean_preds = naive_mean.predict(X_test)
    preds_dict['Naive_Mean'] = mean_preds

    return preds_dict