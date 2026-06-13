from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.data.data_utils import split_sequences
from pathlib import Path
import logging

def _split_dates(
        start_date,
        train_cutoff_year,
        val_months=24,
        test_months=12
):
    # Define split dates
    train_start = start_date
    train_cutoff = f'{train_cutoff_year}-12-01'
    train_end = (datetime.strptime(train_cutoff, '%Y-%m-%d') 
                 - relativedelta(months=val_months))
    val_start = train_end + relativedelta(months=1)
    val_end = val_start + relativedelta(months=(val_months-1))
    test_start = val_end + relativedelta(months=1)
    test_end = test_start + relativedelta(months=(test_months-1))

    logging.info(f"Train: {train_start} to {train_end}")
    logging.info(f"Validation: {val_start} to {val_end}")
    logging.info(f"Test: {test_start} to {test_end}")

    return train_start, train_end, val_start, val_end, test_start, test_end


def _read_and_merge_data(
        input_paths,
        targets_path
):
    
    # Read and merge inputs
    inputs = [
        pd.read_csv(
            path, 
            index_col=0, 
            parse_dates=True
        ) for path in input_paths
    ]
    X = pd.concat(inputs, axis=1)

    targets = pd.read_csv(targets_path, index_col=0, parse_dates=True)

    # Convert indices to datetime
    X.index = pd.to_datetime(X.index, format='%Y-%m-%d')
    targets.index = pd.to_datetime(targets.index, format='%Y-%m-%d')

    # Make index sizes consistent
    common_idx = X.index.intersection(targets.index)
    X = X.loc[common_idx]
    targets = targets.loc[common_idx]

    return X, targets

def _fractional_train_val_split(
    X,
    y,
    val_size=0.1,
    val_style='random',
    val_buffer=60
):
    
    n = X.shape[0]
    n_val = int(val_size * n)

    if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
    
        if val_style == 'last':
            X_train = X.iloc[:-n_val]
            X_val = X.iloc[-n_val:]

            y_train = y.iloc[:-n_val]
            y_val = y.iloc[-n_val:]

        elif val_style == 'first':
            X_train = X.iloc[n_val:]
            X_val = X.iloc[:n_val]

            y_train = y.iloc[n_val:]
            y_val = y.iloc[:n_val]

        elif val_style == 'random':
            val_idx = np.random.choice(n-val_buffer, size=n_val)
            mask = ~np.isin(np.arange(n), val_idx)   
            
            X_train = X.iloc[mask]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[mask]
            y_val = y.iloc[val_idx]
        
        else:
            raise TypeError(f'Unrecognized argument {val_style}'
                            'Please provide one of {"last", "first", "random"}')
        
    elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):

        if val_style == 'last':
            X_train = X[:-n_val]
            X_val = X[-n_val:]

            y_train = y[:-n_val]
            y_val = y[-n_val:]

        elif val_style == 'first':
            X_train = X[n_val:]
            X_val = X[:n_val]

            y_train = y[n_val:]
            y_val = y[:n_val]

        elif val_style == 'random':
            val_idx = np.random.choice(n-val_buffer, size=n_val)
            mask = ~np.isin(np.arange(n), val_idx)   
            
            X_train = X[mask]
            X_val = X[val_idx]

            y_train = y[mask]
            y_val = y[val_idx]
        
        else:
            raise TypeError(f'Unrecognized argument {val_style}'
                            'Please provide one of {"last", "first", "random"}')

    return X_train, X_val, y_train, y_val

def _train_test_split(
        X, 
        y,
        train_start: str|datetime,
        train_end: str|datetime,
        val_start: str|datetime,
        val_end: str|datetime,
        test_start: str|datetime,
        test_end: str|datetime
    ):

    """
    Parameters:
    *_start: start date of subsample
    *_end: end date of subsample

    Returns:
    X_*: train, val, test features
    y_*: train, val, test targets
    """

    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y have a different number of rows!')

    # Split features
    X_train = X.loc[train_start:train_end]
    X_val = X.loc[val_start:val_end]
    X_test = X.loc[test_start:test_end]

    # Split targets
    y_train = y.loc[train_start:train_end]
    y_val = y.loc[val_start:val_end]
    y_test = y.loc[test_start:test_end]

    return X_train, X_val, X_test, y_train, y_val, y_test 


def _impute_missing_features(
        X_train,
        X_val,
        X_test
):
    
    imputer = SimpleImputer()
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_val_imp = pd.DataFrame(
        imputer.transform(X_val), 
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )

    return X_train_imp, X_val_imp, X_test_imp

def _scale_features(
        X_train,
        X_val,
        X_test
):
    # Standardize features
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), 
        columns=X_val.columns, 
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )

    return X_train_scaled, X_val_scaled, X_test_scaled



def prepare_non_rnn_data(
        targets_path,
        input_paths,
        start_date,
        train_cutoff_year,
        val_months,
        test_months,
        imputer=None,
        scaler=None,
        target_scale_factor=100
    ):

    # Read and merge data
    X, targets = _read_and_merge_data(
        input_paths=input_paths,
        targets_path=targets_path
    )

    (
        train_start,
        train_end,
        val_start,
        val_end,
        test_start,
        test_end
    ) = _split_dates(
        start_date=start_date,
        train_cutoff_year=train_cutoff_year,
        val_months=val_months,
        test_months=test_months
    )

    # Split data 
    (
        X_train, X_val, X_test, 
        targets_train, targets_val, targets_test
    ) = _train_test_split(
        X, 
        targets,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end
    )

    # impute data
    X_train, X_val, X_test = _impute_missing_features(
        X_train, X_val, X_test
    )

    # Scale data
    X_train, X_val, X_test = _scale_features(
        X_train, X_val, X_test
    )

    if target_scale_factor:
        targets_train *= target_scale_factor
        targets_val *= target_scale_factor
        targets_test *= target_scale_factor

    return X_train, X_val, X_test, targets_train, targets_val, targets_test


def prepare_rnn_data(
        targets_path,
        input_paths,
        start_date,
        train_cutoff_year,
        val_months,
        test_months,
        n_timesteps=12,
        target_scale_factor=100
    ):


    if n_timesteps <= 1:
        raise ValueError('The number of time steps must be greater than 1.')

    (
        X_train, X_val, X_test, 
        targets_train, targets_val, targets_test
    ) = prepare_non_rnn_data(
        targets_path,
        input_paths,
        start_date,
        train_cutoff_year,
        val_months,
        test_months,
        target_scale_factor=target_scale_factor
    )

    train_data = pd.concat([X_train, targets_train], axis=1)

    val_data = pd.concat([X_val, targets_val], axis=1)
    val_data = pd.concat([train_data.iloc[-(n_timesteps-1):], val_data])

    test_data = pd.concat([X_test, targets_test], axis=1)
    test_data = pd.concat([val_data.iloc[-(n_timesteps-1):], test_data])

    # Make sequences for recurrent neural nets
    X_train_rnn, targets_train_rnn = split_sequences(
        train_data,
        n_timesteps=n_timesteps,
        n_targets=targets_train.shape[1]
    ) 

    X_val_rnn, targets_val_rnn = split_sequences(
        val_data,
        n_timesteps=n_timesteps,
        n_targets=targets_train.shape[1]
    )

    X_test_rnn, targets_test_rnn = split_sequences(
        test_data,
        n_timesteps=n_timesteps,
        n_targets=targets_train.shape[1]
    )

    return (X_train_rnn, X_val_rnn, X_test_rnn, 
            targets_train_rnn, targets_val_rnn, targets_test_rnn)
