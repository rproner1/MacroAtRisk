from keras.callbacks import EarlyStopping
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import numpy as np
from typing import Tuple
import inspect
import optuna
import logging

from src.utils.files import save_hyperparameters

def _take_rows(data, indices):
    
    if hasattr(data, 'iloc'):
        return data.iloc[indices]
    else:
        return np.asarray(data)[indices]


def _filter_builder_kwargs(builder_func, params):
    signature = inspect.signature(builder_func)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return params
    accepted = {
        name for name, parameter in signature.parameters.items()
        if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {name: value for name, value in params.items() if name in accepted}

def get_early_stopping(**kwargs):
    return EarlyStopping(**kwargs)


class CVObjective:

    def __init__(
            self, 
            X_tr, 
            y_tr, 
            val_size: float, 
            n_splits: float, 
            fit_params: dict, 
            early_stopping_args: dict, 
            builder_func: callable, 
            n_jobs: int, 
            grid: dict | None = None,
            **kwargs
        ):
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.val_size = val_size
        self.n_splits = n_splits
        self.fit_params = fit_params
        self.builder_func = builder_func
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.early_stopping_args = early_stopping_args
        self.grid = grid

    def _sample_trial_kwargs(self, trial):
        trial_kwargs = self.kwargs.copy()

        for param_name, param_config in self.grid.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'categorical')
                param_values = param_config.get('values', [])
                log_scale = param_config.get('log_scale', False)
            else:
                param_type = 'categorical'
                param_values = param_config
                log_scale = False

            if param_type == 'categorical':
                trial_kwargs[param_name] = trial.suggest_categorical(param_name, param_values)
            elif param_type == 'float':
                low = float(min(param_values))
                high = float(max(param_values))
                if low == high:
                    trial_kwargs[param_name] = low
                else:
                    trial_kwargs[param_name] = trial.suggest_float(param_name, low, high, log=log_scale)
            elif param_type == 'int':
                low = int(min(param_values))
                high = int(max(param_values))
                if low == high:
                    trial_kwargs[param_name] = low
                else:
                    trial_kwargs[param_name] = trial.suggest_int(param_name, low, high, log=log_scale)
            else:
                raise ValueError(f"Unknown parameter type '{param_type}' for '{param_name}'")

        return trial_kwargs

    def _fit_and_evaluate_on_split(self, trial_kwargs, train_idx, test_idx):
        
        if isinstance(self.X_tr, list):
            X_train = [_take_rows(x, train_idx) for x in self.X_tr]
            X_test = [_take_rows(x, test_idx) for x in self.X_tr]
        else:
            X_train = _take_rows(self.X_tr, train_idx)
            X_test = _take_rows(self.X_tr, test_idx)

        y_train = _take_rows(self.y_tr, train_idx)
        y_test = _take_rows(self.y_tr, test_idx)

        # Split the training set further for early stopping
        if isinstance(self.X_tr, list):
            split_idx = int(len(X_train[0]) * (1 - self.val_size))
            X_train_split = [x[:split_idx] for x in X_train]
            X_val_split = [x[split_idx:] for x in X_train]
        else:
            split_idx = int(len(X_train) * (1 - self.val_size))
            X_train_split = X_train[:split_idx]
            X_val_split = X_train[split_idx:]

        y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]

        model = self.builder_func(**_filter_builder_kwargs(self.builder_func, trial_kwargs))

        es = get_early_stopping(**self.early_stopping_args)
        fit_params = {
            **self.fit_params,
            'callbacks': [es],
            'validation_data': (X_val_split, y_val_split),
        }

        model.fit(X_train_split, y_train_split, **fit_params)
        val_score = model.evaluate(X_test, y_test, verbose=0)
        if isinstance(val_score, list):
            val_score = val_score[0]

        return val_score

    def __call__(self, trial):
        trial_kwargs = self._sample_trial_kwargs(trial)

        cv_losses = Parallel(n_jobs=self.n_jobs, return_as='generator', prefer='threads')(
            delayed(self._fit_and_evaluate_on_split)(trial_kwargs.copy(), train_idx, test_idx)
            for train_idx, test_idx in KFold(n_splits=self.n_splits).split(self.X_tr)
        )

        mean_cv_loss = np.mean(list(cv_losses))
        return float(mean_cv_loss)

def perform_hpo(
        X_train,
        y_train,
        builder_func,
        fit_params,
        early_stopping_args,
        grid,
        study_name,
        val_size=0.1,
        n_splits=10,
        trials=50,
        n_jobs=-1,
        storage=None,
        sampler=None,
        pruner=None,
        save_hps=True,
        log_path='tuning_log.json',
        **kwargs
):
    
    if storage is None:
        storage = optuna.storages.InMemoryStorage()
    if sampler is None:
        sampler = optuna.samplers.RandomSampler()
    if pruner is None:
        pruner = optuna.pruners.MedianPruner()
    
    objective = CVObjective(
        X_tr=X_train,
        y_tr=y_train,
        val_size=val_size,
        n_splits=n_splits,
        builder_func=builder_func,
        fit_params=fit_params,
        early_stopping_args=early_stopping_args,
        n_jobs=n_jobs,
        grid=grid,
        **kwargs
    )

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )

    n_completed_trials = len(
        study.get_trials(
            deepcopy=False, 
            states=[optuna.trial.TrialState.COMPLETE]
        )
    )
    remaining_trials = max(0, trials - n_completed_trials)

    if remaining_trials == 0:
        logging.info(
            f"Study {study_name} already has {n_completed_trials} completed trials. Skipping..."
        )
    else:
        logging.info(
            f"Study {study_name} has {n_completed_trials} completed trials. Running {remaining_trials} more..."
        )
        study.optimize(
            objective,
            n_trials=remaining_trials,
            n_jobs=n_jobs,
            gc_after_trial=True,
            show_progress_bar=True,
        )

    best_params = study.best_params

    if save_hps:
        save_hyperparameters(
            best_params,
            study_name,
            log_path=log_path
        )

    return best_params
