
import logging

import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import os
from pathlib import Path
from src.utils.files import check_hps_exist, load_hyperparameters, save_hyperparameters
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from quantile_forest import RandomForestQuantileRegressor
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools import add_constant
from sklearn.dummy import DummyRegressor

from src.train.losses import make_tilted_loss
from src.train.models import build_qlr
from src.train.train_utils import fit_models
from src.train.tuning import CVObjective

def fit_lit_bench_model(
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        quantiles: list[float], 
        model_name: str
    ) -> dict:

    train_preds_dict = {}
    preds_dict = {}

    for q in quantiles:

        Q = int(q*100)

        model = QuantReg(y_train.values, add_constant(X_train.values, has_constant='skip'))
        res = model.fit(q=q)
        print(f"Quantile: {q}, Params: {res.params}")

        # predict
        preds = res.predict(add_constant(X_test.values, has_constant='skip'))
        train_preds = res.predict(add_constant(X_train.values, has_constant='skip'))
        preds_dict[f'{model_name}_Q{Q}'] = preds
        train_preds_dict[f'{model_name}_Q{Q}'] = train_preds

    return train_preds_dict, preds_dict

def fit_linear_models(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_train_full: np.ndarray,
        y_train_full: np.ndarray,
        X_test: np.ndarray,
        quantiles: list[float],
        target_name: str,
        year: int,
        val_size: float,
        k_folds: int,
        tuning_path: Path,
        early_stopping_args: dict,
        fit_params: dict,
        seed: int,
        trials: int,
        n_estimators: int,
        model_dir_path: Path,
        linear_grids: dict | None = None,
    ):

    path_quantiles = [int(q*100) for q in quantiles]
    custom_objects = {
        **{f'tilted_loss_{Q}': make_tilted_loss(Q) for Q in path_quantiles}
    }
    
    if linear_grids is None:
        linear_grids = {
            'QR': {
                'lr': {'type': 'float', 'values': [5e-4, 2e-3], 'log_scale': True}
            },
            'RID': {
                'l2': {'type': 'float', 'values': [1e-5, 1e-4], 'log_scale': True},
                'lr': {'type': 'float', 'values': [5e-4, 2e-3], 'log_scale': True},
            },
            'LAS': {
                'l1': {'type': 'float', 'values': [1e-7, 1e-5], 'log_scale': True},
                'lr': {'type': 'float', 'values': [5e-4, 2e-3], 'log_scale': True},
            },
            'EN': {
                'l1': {'type': 'float', 'values': [1e-7, 1e-5], 'log_scale': True},
                'l2': {'type': 'float', 'values': [1e-5, 1e-4], 'log_scale': True},
                'lr': {'type': 'float', 'values': [5e-4, 2e-3], 'log_scale': True},
            },
        }

    models = [m for m in ['QR', 'RID', 'LAS', 'EN'] if m in linear_grids]
    
    # Optuna storage
    storage_url = optuna.storages.InMemoryStorage()

    all_model_preds = {}
    # Train each model variant
    for model in models:
        for q,Q in zip(quantiles, path_quantiles):
            study_name = f'{model}_Q{Q}_{target_name}_{year}'
            
            logging.info(f"Training {model}_Q{Q}...")
        
            # Check if hyperparameters already exist
            if check_hps_exist(study_name, tuning_path):
                logging.info(f"Hyperparameters for {study_name} already exist. Loading...")
                best_params = load_hyperparameters(study_name, tuning_path)
            else:
                logging.info(f"No existing hyperparameters for {study_name}, optimizing...")

                builder_params = {'q': q}

                grid = linear_grids[model]

                objective = CVObjective(
                    X_tr=X_train_full,
                    y_tr=y_train_full,
                    val_size=val_size,
                    n_splits=k_folds,
                    builder_func=build_qlr,
                    fit_params=fit_params,
                    early_stopping_args=early_stopping_args,
                    n_jobs=os.cpu_count(),
                    grid=grid,
                    **builder_params
                )
                
                study = optuna.create_study(
                    direction="minimize",
                    study_name=study_name,
                    storage=storage_url,
                    load_if_exists=True,
                    sampler=optuna.samplers.RandomSampler(seed),
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
                    log_path=tuning_path
                )
            
            # Fit models with best hyperparameters
            estimators = fit_models(
                X_train,
                y_train,
                build_qlr,
                model_name=study_name,
                hps=best_params,
                fit_params=fit_params,
                early_stopping_args=early_stopping_args,
                n_estimators=n_estimators,
                models_dir_path=model_dir_path,
                save_models=True,
                custom_objects=custom_objects
            )
            
            # Generate predictions
            preds = []
            for e in estimators:
                e_preds = np.asarray(e.predict(X_test)).reshape(-1)
                preds.append(e_preds[:, np.newaxis])
            
            preds = np.concatenate(preds, axis=1).mean(axis=1) # Take the mean over estimators

            # Store predictions for the current quantile
            all_model_preds[f'{model}_Q{Q}'] = preds
    
    # # Save predictions
    # all_model_preds_df = pd.DataFrame(
    #     all_model_preds,
    #     index=pd.date_range(start=meta_data['test_start'], end=meta_data['test_end'], freq='MS')
    # )

    return all_model_preds

def fit_sklearn_linear_models(
        X_train: pd.DataFrame, 
        y_train: pd.Series,
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

        X_tr, y_tr = (X_train, y_train)
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
    X_train: pd.DataFrame, 
    y_train: pd.Series,
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
            best_fit = gbt.fit(X_train, y_train.values.flatten())
        else:
            print(f'Tuning {study_name}...')
            grid_search = GridSearchCV(gbt, grid, refit=True, cv=k_folds, n_jobs=-1)

            # Perform grid search and refit with best params
            best_fit = grid_search.fit(X_train, y_train.values.flatten())

            # Save hps
            best_params = best_fit.best_params_
            save_hyperparameters(best_params, study_name, tuning_log_path)

        preds[f'QGB_Q{Q}'] = best_fit.predict(X_test)
    return preds

def fit_qrf(
        X_train: pd.DataFrame, 
        y_train: pd.Series,
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

        best_fit = grid_search.fit(X_train, y_train.values.flatten())

        # Save hps
        best_params = best_fit.best_params_
        save_hyperparameters(best_params, study_name, tuning_log_path)

    qrf.set_params(**best_params)
    qrf.fit(X_train, y_train.values.flatten())
    preds_array = qrf.predict(X_test, quantiles=quantiles).reshape(-1,len(quantiles))  # Shape (n_samples, n_quantiles)
    
    for i, Q in enumerate(path_quantiles):
        preds[f'QRF_Q{Q}'] = preds_array[:, i]

    return preds

def fit_qpcr(
        X_train: pd.DataFrame, 
        y_train: pd.Series,
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
                X=X_train.values, # Constant added inside function
                y=y_train.values, 
                q=q
            )

            np.save(models_dir / f'{study_name}_bestsubset.npy', np.array(best_subset))

        else:

            print(f'{study_name} already exists. Loading...')

            best_subset = list(np.load(f'{models_dir}{study_name}_bestsubset.npy', allow_pickle=True).flatten())

            X_train_subset = X_train.values[:, best_subset]
            qpcr = QuantReg(y_train.values, add_constant(X_train_subset, has_constant='skip')).fit(q=q)

        # print(f"best_subset: {best_subset}")

        X_test_subset = X_test.values[:, best_subset]
        preds = qpcr.predict(add_constant(X_test_subset, has_constant='skip')).flatten()

        preds_dict[f'QPCR_Q{Q}'] = preds

def fit_ar1(
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        quantiles: list[float], 
        target_name: str, 
        year: int,
        verbose: bool=True
    ):

    preds_dict = {}

    for q in quantiles:
        if verbose: print(f'Fitting quantile {q}') 
        ar_1 = QuantileRegressor(quantile=q)
        ar_1.fit(X_train.values.reshape(-1,1), y_train.values)
        preds = ar_1.predict(X_test.values.reshape(-1,1)) # Shape (n_samples,)
        preds_dict[f'AR1_Q{int(q*100)}'] = preds

    # Train an AR(1) specifically for the mean
    ar1_mean = LinearRegression()
    ar1_mean.fit(X_train.values.reshape(-1,1), y_train.values)
    mean_preds = ar1_mean.predict(X_test.values.reshape(-1,1))
    preds_dict['AR1_Mean'] = mean_preds

    return preds_dict

def fit_dummy(X_train: pd.DataFrame, 
              y_train: pd.Series,
              X_test: pd.DataFrame,
              quantiles: list[float]):
    
    preds_dict = {}

    for q in quantiles:
        Q = int(q*100)
        naive = DummyRegressor(strategy='quantile', quantile=q)
        # X does not matter for DummyRegressor but required for consistency
        naive.fit(X_train, y_train)
        preds = naive.predict(X_test)
        preds_dict[f'Naive_Q{Q}'] = preds

    # Naive Mean Regressor
    naive_mean = DummyRegressor(strategy='mean')
    naive_mean.fit(X_train, y_train)
    mean_preds = naive_mean.predict(X_test)
    preds_dict['Naive_Mean'] = mean_preds

    return preds_dict