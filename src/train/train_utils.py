from typing import Callable, List, Union
import os
import numpy as np
import pandas as pd
from keras.models import load_model
from src.train.tuning import get_early_stopping

def fit_models(X: Union[List[np.ndarray], pd.DataFrame, np.ndarray], y: Union[dict, pd.DataFrame, np.ndarray], 
              builder_func: Callable, model_name: str, hps: dict, fit_params: dict, early_stopping_args: dict,
              n_estimators: int, models_dir_path: str, custom_objects=None, save_models: bool=True, overwrite: bool=False) -> list:
    
    """
    Fits an ensemble of estimators using builder_func(**hps).

    Parameters
    ----------
    X: 
        Training features.
    y: 
        Training targets
    builder_func:
        Function that returns keras.models.Sequential | keras.models.Functional
    model_name:
        In the format of "<model name>_qx" for x in {5,25,50,75,95}.
    hps:
        kwargs to pass to builder_func
    fit_params:
        Parameters to pass to model.fit()
    n_estimators:
        Number of estimators in the ensemble.
    train_end_year:
        Last year in X,y, e.g. "1989"
    models_dir_path:
        Directory to save the models in or load models from.
    custom_objects:
        Custom object to pass to keras.models.load_model()

    Returns
    -------
    estimators:
        A list of keras.models.Sequential | keras.models.Functional
    """

    estimators = []

    # Make model directory if it does not exist
    dir_path = models_dir_path + model_name + '/'
    os.makedirs(dir_path, exist_ok=True)

    for i in range(n_estimators):

        model_path = dir_path + model_name + '_estimator' + str(i) + '.keras'
        
        # Check if the model is already trained, if not, train it
        if not os.path.isfile(model_path) or overwrite:

            model = builder_func(**hps)

            print(f'Training estimator {i+1} of {n_estimators}...')

            es = get_early_stopping(**early_stopping_args)
            fit_params.update({'callbacks': [es]})
            model.fit(
                X, y,
                **fit_params
            )
            
            if save_models:
                model.save(model_path)
        
        else:
            print('Existing model found, loading...')
            model = load_model(model_path, custom_objects, safe_mode=False)
            
        # Save estimator
        estimators.append(model)
    
    return estimators


def fit_models_par(X: Union[List[np.ndarray], pd.DataFrame, np.ndarray], y: Union[dict, pd.DataFrame, np.ndarray], 
              builder_func: Callable, model_name: str, hps: dict, fit_params: dict, early_stopping_args: dict,
              n_estimators: int, models_dir_path: str, custom_objects=None, save_models: bool=True) -> list:
    
    """
    Fits an ensemble of estimators using builder_func(**hps).

    Parameters
    ----------
    X: 
        Training features.
    y: 
        Training targets
    builder_func:
        Function that returns keras.models.Sequential | keras.models.Functional
    model_name:
        In the format of "<model name>_qx" for x in {5,25,50,75,95}.
    hps:
        kwargs to pass to builder_func
    fit_params:
        Parameters to pass to model.fit()
    n_estimators:
        Number of estimators in the ensemble.
    train_end_year:
        Last year in X,y, e.g. "1989"
    models_dir_path:
        Directory to save the models in or load models from.
    custom_objects:
        Custom object to pass to keras.models.load_model()

    Returns
    -------
    estimators:
        A list of keras.models.Sequential | keras.models.Functional
    """

    from joblib import Parallel, delayed

    estimators = []

    # Make model directory if it does not exist
    dir_path = models_dir_path + model_name + '/'
    os.makedirs(dir_path, exist_ok=True)

    def train_estimator(hps, model_path):
        # Check if the model is already trained, if not, train it
        if not os.path.isfile(model_path):

            model = builder_func(**hps)


            es = get_early_stopping(**early_stopping_args)
            fit_params.update({'callbacks': [es]})
            model.fit(
                X, y,
                **fit_params
            )
            
            if save_models:
                model.save(model_path)
        
        else:
            print('Existing model found, loading...')
            model = load_model(model_path, custom_objects)

        return model

    estimators = Parallel(delayed(train_estimator)(hps, f"{dir_path}{model_name}_estimator{i}.keras") for i in range(n_estimators))
    
    return estimators
