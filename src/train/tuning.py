from keras.callbacks import EarlyStopping
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import numpy as np
from typing import Tuple

def get_early_stopping(**kwargs):
    return EarlyStopping(**kwargs)

def grid_search(param_grid: dict, builder_func: callable, X_tr: np.ndarray, y_tr: np.ndarray, fit_params: dict, early_stopping_args: dict, n_jobs: int, **kwargs):

    from itertools import product

    keys, values = zip(*param_grid.items())
    candidates = product(*values)

    def evaluate_candidate(hps):

        params = dict(zip(keys, hps))
        kwargs.update(params)

        model = builder_func(**kwargs)

        es = get_early_stopping(**early_stopping_args)
        fit_params.update({'callbacks': [es]})

        model.fit(X_tr, y_tr, **fit_params)

        val_score = model.evaluate(*fit_params['validation_data'])
        if isinstance(val_score, list):
            val_score = val_score[0]  # If multiple metrics, take the total val loss

        return params, val_score

    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_candidate)(hps) for hps in candidates)
    params, scores = zip(*results)
    best_params = params[np.argmin(scores)] 

    return best_params, min(scores) 


# Tuning 
class Objective:

    def __init__(
            self, 
            X_tr, 
            y_tr, 
            fit_params: dict, 
            early_stopping_args: dict, 
            builder_func: callable, 
            tune_l1: bool=False,
            tune_l2: bool=True,
            tune_lr: bool=True,
            tune_rec_drop: bool=False,
            tune_dropout: bool=False,
            tune_n_layers: bool=False,
            tune_n_nodes: bool=False,
            tune_norm: bool=False,
            **kwargs
        ):
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.fit_params = fit_params
        self.builder_func = builder_func
        self.kwargs = kwargs
        self.early_stopping_args = early_stopping_args
        self.tune_l1 = tune_l1
        self.tune_l2 = tune_l2
        self.tune_lr = tune_lr
        self.tune_rec_drop = tune_rec_drop
        self.tune_dropout = tune_dropout
        self.tune_n_layers = tune_n_layers
        self.tune_n_nodes = tune_n_nodes
        self.tune_norm = tune_norm

    def __call__(self, trial):

        l1_choices = list(np.logspace(-7, -5, num=50))
        l2_choices = list(np.logspace(-5, -4, num=50))
        rec_drop_choices = [0.0, 0.02, 0.05, 0.1]
        dropout_choices = [0.0, 0.02, 0.05, 0.1]
        layer_choices = [1, 2, 3]
        node_choices = [16, 32, 64]
        norm_choices = [False, True]

        if self.tune_l1:
            l1 = trial.suggest_categorical('l1', l1_choices)
        else:
            l1 = 0.0
        if self.tune_l2:
            l2 = trial.suggest_categorical('l2', l2_choices)
        else: 
            l2 = 0.0
        if self.tune_lr:
            lr = trial.suggest_float('lr', 5e-4, 2e-3, log=True)
        else: 
            lr=5e-4
        if self.tune_rec_drop:
            rec_drop = trial.suggest_categorical('rec_drop', rec_drop_choices)
        else:
            rec_drop = 0.0
        if self.tune_dropout:
            dropout = trial.suggest_categorical('dropout', dropout_choices)
        else:
            dropout=0.0
        if self.tune_n_layers:
            n_recurrent_layers = trial.suggest_categorical('n_recurrent_layers', layer_choices)
            n_shared_layers = trial.suggest_categorical('n_shared_layers', layer_choices)
            n_qtask_layers = trial.suggest_categorical('n_qtask_layers', layer_choices)
        else:
            n_recurrent_layers = self.kwargs['n_recurrent_layers']
            n_shared_layers = self.kwargs['n_shared_layers']
            n_qtask_layers = self.kwargs['n_qtask_layers']
        if self.tune_n_nodes:
            n_recurrent_nodes = trial.suggest_categorical('n_recurrent_nodes', node_choices)
            n_shared_nodes = trial.suggest_categorical('n_shared_nodes', node_choices)
            n_task_nodes = trial.suggest_categorical('n_task_nodes', node_choices)
        else: 
            n_recurrent_nodes = self.kwargs['n_recurrent_nodes']
            n_shared_nodes = self.kwargs['n_shared_nodes']
            n_task_nodes = self.kwargs['n_task_nodes']
        if self.tune_norm:
            recurrent_norm = trial.suggest_categorical('recurrent_norm', norm_choices)
            shared_norm = trial.suggest_categorical('shared_norm', norm_choices)
            task_norm = trial.suggest_categorical('task_specific_norm', norm_choices)
        else:
            recurrent_norm = self.kwargs['recurrent_norm']
            shared_norm = self.kwargs['shared_norm']
            task_norm = self.kwargs['task_specific_norm']

        self.kwargs.update(
            {
                'l1': l1,
                'l2': l2,
                'lr': lr,
                'rec_drop': rec_drop,
                'dropout': dropout,
                'n_recurrent_layers': n_recurrent_layers,
                'n_shared_layers': n_shared_layers,
                'n_qtask_layers': n_qtask_layers,
                'n_recurrent_nodes': n_recurrent_nodes,
                'n_shared_nodes': n_shared_nodes,
                'n_task_nodes': n_task_nodes,
                'recurrent_norm': recurrent_norm,
                'shared_norm': shared_norm,
                'task_specific_norm': task_norm
            }
        )

        model = self.builder_func(**self.kwargs)

        es = get_early_stopping(**self.early_stopping_args)
        self.fit_params.update({'callbacks': [es]})

        model.fit(self.X_tr, self.y_tr, **self.fit_params)

        val_score = model.evaluate(*self.fit_params['validation_data'])
        if isinstance(val_score, list):
            val_score = val_score[0]  # If multiple metrics, take the total val loss

        return val_score


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
            tune_l1: bool=False,
            tune_l2: bool=True,
            tune_lr: bool=True,
            tune_rec_drop: bool=False,
            tune_dropout: bool=False,
            tune_n_layers: bool=False,
            tune_n_nodes: bool=False,
            tune_norm: bool=False,
            tune_recurrent_layer_type: bool=False,
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
        self.tune_l1 = tune_l1
        self.tune_l2 = tune_l2
        self.tune_lr = tune_lr
        self.tune_rec_drop = tune_rec_drop
        self.tune_dropout = tune_dropout
        self.tune_n_layers = tune_n_layers
        self.tune_n_nodes = tune_n_nodes
        self.tune_norm = tune_norm
        self.tune_recurrent_layer_type = tune_recurrent_layer_type

    def __call__(self, trial):

        l1_choices = list(np.logspace(-7, -5, num=50))
        l2_choices = list(np.logspace(-5, -4, num=50))
        rec_drop_choices = [0.0, 0.02, 0.05, 0.1]
        dropout_choices = [0.0, 0.02, 0.05, 0.1]
        layer_choices = [1, 2, 3]
        node_choices = [16, 32, 64]
        norm_choices = [False, True]
        recurrent_layer_type_choices = ['gru', 'lstm']

        if self.tune_l1:
            l1 = trial.suggest_categorical('l1', l1_choices)
        else:
            l1 = 0.0
        if self.tune_l2:
            l2 = trial.suggest_categorical('l2', l2_choices)
        else: 
            l2 = 1e-4
        if self.tune_lr:
            lr = trial.suggest_float('lr', 5e-4, 2e-3, log=True)
        else: 
            lr=5e-4
        if self.tune_rec_drop:
            rec_drop = trial.suggest_categorical('rec_drop', rec_drop_choices)
        else:
            rec_drop = 0.0
        if self.tune_dropout:
            dropout = trial.suggest_categorical('dropout', dropout_choices)
        else:
            dropout=0.0
        if self.tune_n_layers:
            n_recurrent_layers = trial.suggest_categorical('n_recurrent_layers', layer_choices)
            n_shared_layers = trial.suggest_categorical('n_shared_layers', layer_choices)
            n_qtask_layers = trial.suggest_categorical('n_qtask_layers', layer_choices)
        else:
            n_recurrent_layers = self.kwargs.get('n_recurrent_layers', 3)
            n_shared_layers = self.kwargs.get('n_shared_layers', 3)
            n_qtask_layers = self.kwargs.get('n_qtask_layers', 2)
        if self.tune_n_nodes:
            n_recurrent_nodes = trial.suggest_categorical('n_recurrent_nodes', node_choices)
            n_shared_nodes = trial.suggest_categorical('n_shared_nodes', node_choices)
            n_task_nodes = trial.suggest_categorical('n_task_nodes', node_choices)
        else: 
            n_recurrent_nodes = self.kwargs.get('n_recurrent_nodes', 32)
            n_shared_nodes = self.kwargs.get('n_shared_nodes', 32)
            n_task_nodes = self.kwargs.get('n_task_nodes', 32)
        if self.tune_norm:
            recurrent_norm = trial.suggest_categorical('recurrent_norm', norm_choices)
            shared_norm = trial.suggest_categorical('shared_norm', norm_choices)
            task_norm = trial.suggest_categorical('task_specific_norm', norm_choices)
        else:
            recurrent_norm = self.kwargs.get('recurrent_norm', False)
            shared_norm = self.kwargs.get('shared_norm', False)
            task_norm = self.kwargs.get('task_specific_norm', False)
        if self.tune_recurrent_layer_type:
            recurrent_layer_type = trial.suggest_categorical('recurrent_layer_type', recurrent_layer_type_choices)
        else:
            recurrent_layer_type = self.kwargs.get('recurrent_layer_type', 'lstm')

        self.kwargs.update(
            {
                'l1': l1,
                'l2': l2,
                'lr': lr,
                'rec_drop': rec_drop,
                'dropout': dropout,
                'n_recurrent_layers': n_recurrent_layers,
                'n_shared_layers': n_shared_layers,
                'n_qtask_layers': n_qtask_layers,
                'n_recurrent_nodes': n_recurrent_nodes,
                'n_shared_nodes': n_shared_nodes,
                'n_task_nodes': n_task_nodes,
                'recurrent_norm': recurrent_norm,
                'shared_norm': shared_norm,
                'task_specific_norm': task_norm,
                'recurrent_layer_type': recurrent_layer_type
            }
        )

        def fit_and_evaluate_on_split(X_train, y_train, X_test, y_test):

            # reserve val_size fraction of training data for monitoring early stopping
            split_idx = int(len(X_train) * (1-self.val_size))
            X_train_split, X_val_split = X_train[:split_idx], X_train[split_idx:]
            y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]

            model = self.builder_func(**self.kwargs)

            es = get_early_stopping(**self.early_stopping_args)
            self.fit_params.update({'callbacks': [es], 'validation_data': (X_test, y_test)})

            model.fit(X_train_split, y_train_split, **self.fit_params)
            val_score = model.evaluate(X_val_split, y_val_split)
            if isinstance(val_score, list):
                val_score = val_score[0]  # If multiple metrics, take the total val loss
            return val_score
        
        cv_losses = Parallel(n_jobs=self.n_jobs, return_as='generator')(delayed(fit_and_evaluate_on_split)(
            self.X_tr[train_idx],
            self.y_tr[train_idx],
            self.X_tr[test_idx],
            self.y_tr[test_idx]
        ) for train_idx, test_idx in KFold(n_splits=self.n_splits).split(self.X_tr))

        mean_cv_loss = np.mean(list(cv_losses))
        return mean_cv_loss


class CVObjectiveTest:

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

    def __call__(self, trial):

        l1_choices = [0.0] + list(np.logspace(-7, -5, num=50))
        l2_choices = [0.0] +  list(np.logspace(-5, -4, num=50))
        rec_drop_choices = [0.0] + [0.05,0.1,0.15,0.2,0.3,0.5]
        dropout_choices = [0.0] + list(np.linspace(0.01,0.2,20))
        n_recurrent_layer_choices = [1,2,3]
        n_shared_layer_choices = [1,2,3] 
        n_qtask_layer_choices = [1,2,3]

        l1 = trial.suggest_categorical('l1', l1_choices)
        l2 = trial.suggest_categorical('l2', l2_choices)
        lr = trial.suggest_float('lr', 5e-4, 2e-3, log=True)
        rec_drop = trial.suggest_categorical('rec_drop', rec_drop_choices)
        dropout = trial.suggest_categorical('dropout', dropout_choices)
        n_recurrent_layers = trial.suggest_categorical('n_recurrent_layers', n_recurrent_layer_choices)
        n_shared_layers = trial.suggest_categorical('n_shared_layers', n_shared_layer_choices)
        n_qtask_layers = trial.suggest_categorical('n_qtask_layers', n_qtask_layer_choices)
        n_recurrent_nodes = trial.suggest_categorical('n_recurrent_nodes', [16,32,64])
        n_shared_nodes = trial.suggest_categorical('n_shared_nodes', [16,32,64])
        n_task_nodes = trial.suggest_categorical('n_task_nodes', [16,32,64])

        self.kwargs.update(
            {
                'l1': l1,
                'l2': l2,
                'lr': lr,
                'rec_drop': rec_drop,
                'dropout': dropout,
                'n_recurrent_layers': n_recurrent_layers,
                'n_shared_layers': n_shared_layers,
                'n_qtask_layers': n_qtask_layers,
                'n_recurrent_nodes': n_recurrent_nodes,
                'n_shared_nodes': n_shared_nodes,
                'n_task_nodes': n_task_nodes
            }
        )

        def fit_and_evaluate_on_split(X_train, y_train, X_test, y_test):

            # reserve val_size fraction of training data for monitoring early stopping
            split_idx = int(len(X_train) * (1-self.val_size))
            X_train_split, X_val_split = X_train[:split_idx], X_train[split_idx:]
            y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]

            model = self.builder_func(**self.kwargs)

            es = get_early_stopping(**self.early_stopping_args)
            self.fit_params.update({'callbacks': [es], 'validation_data': (X_test, y_test)})

            model.fit(X_train_split, y_train_split, **self.fit_params)
            val_score = model.evaluate(X_val_split, y_val_split)
            if isinstance(val_score, list):
                val_score = val_score[0]  # If multiple metrics, take the total val loss
            return val_score
        
        cv_losses = Parallel(n_jobs=self.n_jobs)(delayed(fit_and_evaluate_on_split)(
            self.X_tr[train_idx],
            self.y_tr[train_idx],
            self.X_tr[test_idx],
            self.y_tr[test_idx]
        ) for train_idx, test_idx in KFold(n_splits=self.n_splits).split(self.X_tr))

        mean_cv_loss = np.mean(cv_losses)
        return mean_cv_loss
