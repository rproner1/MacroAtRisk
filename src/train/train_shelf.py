

from sklearn.linear_model import QuantileRegressor

def fit_linear_models():

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

def train_shelf(
        quantiles, 
        linear_models: dict,
        linear_model_grids: dict[dict]
    ):

    path_quantiles = [int(q * 100) for q in quantiles]

    preds = {}

    # linear_models ={
    #     **{f'LR_Q{Q}': QuantileRegressor(quantile=q, alpha=0.0) for Q, q in zip(path_quantiles, quantiles)},
    #     **{f'LAS_Q{Q}': QuantileRegressor(quantile=q) for Q, q in zip(path_quantiles, quantiles)},
    #     **{f'LAS-l_Q{Q}': QuantileRegressor(quantile=q) for Q, q in zip(path_quantiles, quantiles)},
    # }
    # linear_model_grids ={
    #     'LR': {},
    #     'LAS': {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
    #     'LAS-l': {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]},
    # }

    return preds