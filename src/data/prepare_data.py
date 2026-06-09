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

    all_y = pd.read_csv(targets_path, index_col=0, parse_dates=True)

    # Convert indices to datetime
    X.index = pd.to_datetime(X.index, format='%Y-%m-%d')
    all_y.index = pd.to_datetime(all_y.index, format='%Y-%m-%d')

    return X, all_y

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
        raise Warning('X and y have a different number of rows!')

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
        X_test,
        imputer=SimpleImputer
):
    imp = imputer()
    X_train_imp = pd.DataFrame(
        imp.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_val_imp = pd.DataFrame(
        imp.transform(X_val), 
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_imp = pd.DataFrame(
        imp.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )

    return X_train_imp, X_val_imp, X_test_imp

def _scale_features(
        X_train,
        X_val,
        X_test,
        scaler=StandardScaler
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
        val_split_style='fractional',
        val_style='random',
        val_buffer=60,
        val_size=0.1,
        imputer=SimpleImputer,
        scaler=StandardScaler
    ):

    if val_split_style not in ['fractional', 'date']:
        raise TypeError(f'Unrecognized argument {val_split_style}'
                        'Please provide one of {"fractional", "date"}')
    
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

    if val_split_style=='fractional':
        X_train_full = pd.concat([X_train,X_val])
        targets_train_full = pd.concat([targets_train, targets_val])

        (
            X_train, X_val, 
            targets_train, targets_val
        ) = _fractional_train_val_split(
            X=X_train_full,
            y=targets_train_full,
            val_size=val_size,
            val_style=val_style,
            val_buffer=val_buffer
        )

    # impute data
    X_train, X_val, X_test = _impute_missing_features(
        X_train, X_val, X_test, imputer=imputer
    )

    # Scale data
    X_train, X_val, X_test = _scale_features(
        X_train, X_val, X_test, scaler=scaler
    )

    return X_train, X_val, X_test, targets_train, targets_val, targets_test


def prepare_rnn_data(
        targets_path,
        input_paths,
        start_date,
        train_cutoff_year,
        val_months,
        test_months,
        val_split_style='fractional',
        val_style='random',
        val_buffer=60,
        val_size=0.1,
        n_timesteps=12,
        imputer=SimpleImputer,
        scaler=StandardScaler
    ):

    if val_split_style not in ['fractional', 'date']:
        raise TypeError(f'Unrecognized argument {val_split_style}'
                        'Please provide one of {"fractional", "date"}')

    if n_timesteps <= 1:
        raise Warning('The number of time steps must be greater than 1.')

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
        split_style='date',
        imputer=imputer,
        scaler=scaler 
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

    if val_split_style == 'fractional':
        X_train_rnn_full = np.concatenate([X_train_rnn, X_val_rnn], axis=0)
        targets_train_rnn_full =  np.concatenate(
            [targets_train_rnn, targets_val_rnn],
            axis=0
        )
        (
            X_train_rnn, X_val_rnn, 
            targets_train_rnn, targets_val_rnn
        ) = _fractional_train_val_split(
            X_train_rnn_full,
            targets_train_rnn_full,
            val_size=val_size,
            val_style=val_style,
            val_buffer=val_buffer
        )


    return (X_train_rnn, X_val_rnn, X_test_rnn, 
            targets_train_rnn, targets_val_rnn, targets_test_rnn)
    

def prepare_quantile_data(target: int, time_steps: int, 
                          targets_path: Path, input_paths: List[Path],
                          start_date: str, train_cutoff_year: int, 
                          n_quantiles: int, val_years: int) -> Tuple[dict, dict, dict]:

    # Define split dates
    train_start = start_date
    train_cutoff = f'{train_cutoff_year}-12-01'
    train_end = datetime.strptime(train_cutoff, '%Y-%m-%d') - relativedelta(years=val_years)
    val_start = train_end + relativedelta(months=1)
    val_end = val_start + relativedelta(years=val_years-1, months=11)
    test_start = val_end + relativedelta(months=1)
    test_end = test_start + relativedelta(months=11)

    print(f"Train: {train_start} to {train_end}")
    print(f"Validation: {val_start} to {val_end}")
    print(f"Test: {test_start} to {test_end}")

    # Read and merge data
    inputs = [pd.read_csv(path, index_col=0, parse_dates=True) for path in input_paths]
    X = pd.concat(inputs, axis=1)
    all_y = pd.read_csv(targets_path, index_col=0, parse_dates=True)
    n_targets = all_y.shape[1]

    # Ensure target is an integer index for array indexing
    if isinstance(target, str):
        target_idx = list(all_y.columns).index(target)
    else:
        target_idx = target

    # Convert indices to datetime
    X.index = pd.to_datetime(X.index, format='%Y-%m-%d')
    all_y.index = pd.to_datetime(all_y.index, format='%Y-%m-%d')

    # Split features
    X_train = X.loc[train_start:train_end]
    X_val = X.loc[val_start:val_end]
    X_train_full = X.loc[train_start:val_end]
    X_test = X.loc[test_start:test_end]
    X_test_full = X.loc[test_start:]

    # Impute missing values
    imputer = SimpleImputer()
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    X_train_full_imp = pd.DataFrame(imputer.transform(X_train_full), columns=X_train_full.columns, index=X_train_full.index)
    X_test_full_imp = pd.DataFrame(imputer.transform(X_test_full), columns=X_test_full.columns, index=X_test_full.index)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train_imp.columns, index=X_train_imp.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_imp), columns=X_val_imp.columns, index=X_val_imp.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test_imp.columns, index=X_test_imp.index)
    X_train_full_scaled = pd.DataFrame(scaler.transform(X_train_full_imp), columns=X_train_full_imp.columns, index=X_train_full_imp.index)
    X_test_full_scaled = pd.DataFrame(scaler.transform(X_test_full_imp), columns=X_test_full_imp.columns, index=X_test_full_imp.index)

    # Split targets
    all_y_train = all_y.loc[train_start:train_end]
    all_y_val = all_y.loc[val_start:val_end]
    all_y_test = all_y.loc[test_start:test_end]
    all_y_train_full = all_y.loc[train_start:val_end]
    all_y_test_full = all_y.loc[test_start:]

    y_train = all_y_train.iloc[:,target_idx]
    y_val = all_y_val.iloc[:,target_idx]
    y_train_full = all_y_train_full.iloc[:,target_idx]
    y_test = all_y_test.iloc[:,target_idx]
    
    # Targets for single-task multi-quantile models. repeats same array n_quantiles times. Different quantile loss functions computed on the same series
    mq_y_train = np.repeat(y_train.values.reshape(-1,1), n_quantiles, axis=1)
    mq_y_val = np.repeat(y_val.values.reshape(-1,1), n_quantiles, axis=1)
    mq_y_test = np.repeat(y_test.values.reshape(-1,1), n_quantiles, axis=1)
    mq_y_train_full = np.repeat(y_train_full.values.reshape(-1,1), n_quantiles, axis=1)

    # MT targets for non-rnn mt models
    # Creates a list of n_targets arrays of shape (T, 5), one for each target, 5 for 5 quantiles
    mtmq_y_train = [np.repeat(all_y_train.values[:,i].reshape(-1,1), n_quantiles, axis=1) for i in range(n_targets) ] 
    mtmq_y_val = [ np.repeat(all_y_val.values[:,i].reshape(-1,1), n_quantiles, axis=1) for i in range(n_targets) ]
    mtmq_y_test = [ np.repeat(all_y_test.values[:,i].reshape(-1,1), n_quantiles, axis=1) for i in range(n_targets) ]
    mtmq_y_train_full = [ np.repeat(all_y_train_full.values[:,i].reshape(-1,1), n_quantiles, axis=1) for i in range(n_targets) ]

    # Prepare RNN sequences
    X_full = pd.concat([X_train_scaled, X_val_scaled, X_test_scaled])
    all_y_full = pd.concat([all_y_train, all_y_val, all_y_test])
    dataset = pd.concat([X_full, all_y_full], axis=1)
    
    train_dataset = dataset.loc[train_start:train_end]
    train_full_dataset = dataset.loc[train_start:val_end]
    full_dataset = pd.concat([X_test_full_scaled, all_y_test_full], axis=1)

    # RNNs need a warm-up period to fill the initial sequence. not data leakage because these last observations are not used for prediction in the training phase
    if time_steps > 1:
        val_dataset = pd.concat([train_dataset.iloc[-(time_steps-1):], dataset.loc[val_start:val_end]])
        test_dataset = pd.concat([val_dataset.iloc[-(time_steps-1):], dataset.loc[test_start:test_end]])
        full_test_dataset = pd.concat([val_dataset.iloc[-(time_steps-1):], full_dataset.loc[test_start:]])
    else:
        val_dataset = dataset.loc[val_start:val_end]
        test_dataset = dataset.loc[test_start:test_end]
        full_test_dataset = full_dataset.loc[test_start:]

    # Make sequences 
    X_train_rnn, all_y_train_rnn_arr = split_sequences(train_dataset.values, time_steps, n_targets)
    X_val_rnn, all_y_val_rnn_arr = split_sequences(val_dataset.values, time_steps, n_targets)
    X_test_rnn, all_y_test_rnn_arr = split_sequences(test_dataset.values, time_steps, n_targets)
    X_train_full_rnn, all_y_train_full_rnn_arr = split_sequences(train_full_dataset.values, time_steps, n_targets)
    X_test_rnn_full, all_y_test_rnn_arr_full = split_sequences(full_test_dataset.values, time_steps, n_targets)
    
    # Create lists where each entry is (T, 1, n_quantiles) for each target
    mtmq_y_train_rnn = [np.repeat(all_y_train_rnn_arr[:, i].reshape(-1, 1), n_quantiles, axis=1)
                      for i in range(n_targets)]
    mtmq_y_val_rnn = [np.repeat(all_y_val_rnn_arr[:, i].reshape(-1, 1), n_quantiles, axis=1)
                    for i in range(n_targets)]
    mtmq_y_test_rnn = [np.repeat(all_y_test_rnn_arr[:, i].reshape(-1, 1), n_quantiles, axis=1)
                     for i in range(n_targets)]
    mtmq_y_train_full_rnn = [np.repeat(all_y_train_full_rnn_arr[:, i].reshape(-1, 1), n_quantiles, axis=1)
                          for i in range(n_targets)]

    

    # MQ models are single-task model with multiple quantile outputs so we repeat the single target n_quantiles times
    mq_y_train_rnn = np.repeat(all_y_train_rnn_arr[:,target_idx].reshape(-1,1), n_quantiles, axis=1)
    mq_y_val_rnn = np.repeat(all_y_val_rnn_arr[:,target_idx].reshape(-1,1), n_quantiles, axis=1)
    mq_y_test_rnn = np.repeat(all_y_test_rnn_arr[:,target_idx].reshape(-1,1), n_quantiles, axis=1)
    mq_y_train_full_rnn = np.repeat(all_y_train_full_rnn_arr[:,target_idx].reshape(-1,1), n_quantiles, axis=1)

    # Get rnn targets for single task RNN
    y_train_rnn = mtmq_y_train_rnn[target_idx]
    y_val_rnn = mtmq_y_val_rnn[target_idx]
    y_test_rnn = mtmq_y_test_rnn[target_idx]
    y_train_full_rnn = mtmq_y_train_full_rnn[target_idx]

    # Split into per-input feature groups
    # compute column index ranges from the original inputs list
    col_counts = [df.shape[1] for df in inputs]
    offsets = [0]
    for c in col_counts:
        offsets.append(offsets[-1] + c)

    # Give input indicies for each input from the total feature set
    input_indicies = [list(range(offsets[i], offsets[i + 1])) for i in range(len(input_paths))] 

    # Separates inputs into lists of inputs for multi-input models
    X_inputs_train = [X_train_scaled.iloc[:,idx] for idx in input_indicies]
    X_inputs_val = [X_val_scaled.iloc[:,idx] for idx in input_indicies]
    X_inputs_test = [X_test_scaled.iloc[:,idx] for idx in input_indicies]
    X_inputs_train_full = [X_train_full_scaled.iloc[:,idx] for idx in input_indicies]
    X_inputs_test_full = [X_test_full_scaled.iloc[:,idx] for idx in input_indicies]

    X_inputs_train_rnn = [X_train_rnn[:,:,idx] for idx in input_indicies]
    X_inputs_val_rnn = [X_val_rnn[:,:,idx] for idx in input_indicies]
    X_inputs_test_rnn = [X_test_rnn[:,:,idx] for idx in input_indicies]
    X_inputs_train_full_rnn = [X_train_full_rnn[:,:,idx] for idx in input_indicies]
    X_inputs_test_rnn_full = [X_test_rnn_full[:,:,idx] for idx in input_indicies]
    """
    MQ:
        Tuning: (X_inputs_train, mq_y_train) and (X_inputs_val, mq_y_val) for validation
        Final training: (X_inputs_train_full, mq_y_train_full)
    DMQ:
        Tuning: (X_inputs_train_rnn, mq_y_train_rnn) and (X_inputs_val_rnn, mq_y_val_rnn) for validation
        Final training: (X_inputs_train_full_rnn, mq_y_train_full_rnn)
    MTMQ:
        Tuning: (X_inputs_train, mtmq_y_train) and (X_inputs_val, mtmq_y_val) for validation
        Final training: (X_inputs_train_full, mtmq_y_train_full)
    DMTMQ:
        Tuning: (X_inputs_train_rnn, mtmq_y_train_rnn) and (X_inputs_val_rnn, mtmq_y_val_rnn) for validation
        Final training: (X_inputs_train_full_rnn, mtmq_y_train_full_rnn)
    """

    # Package results
    non_rnn_data = {
        'X_train': X_train_scaled, 'X_val': X_val_scaled, 'X_test': X_test_scaled, 'X_test_full': X_test_full_scaled, # Standard data for static models
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,  # targets for static single-task models
        'mtmq_y_train': mtmq_y_train, 'mtmq_y_val': mtmq_y_val, 'mtmq_y_test': mtmq_y_test, 'mtmq_y_train_full': mtmq_y_train_full, # multi-task multi-quantile targets for static models
        'mq_y_train': mq_y_train, 'mq_y_val': mq_y_val, 'mq_y_test': mq_y_test, 'mq_y_train_full': mq_y_train_full, # multi-quantile targets for static models
        'X_train_full': X_train_full_scaled, 'y_train_full': y_train_full, # Static full training data for shelf models
        'all_y_train': all_y_train, 'all_y_val': all_y_val, 'all_y_test': all_y_test, 'all_y_test_full': all_y_test_full, # All targets, mainly for convenience
        'X_inputs_train': X_inputs_train, 'X_inputs_val': X_inputs_val, 'X_inputs_test': X_inputs_test, 'X_inputs_train_full': X_inputs_train_full, 'X_inputs_test_full': X_inputs_test_full # Static mulit-modal data
    }
    rnn_data = {
        'X_train_rnn': X_train_rnn, 'X_val_rnn': X_val_rnn, 'X_test_rnn': X_test_rnn,'X_train_full_rnn': X_train_full_rnn, # single input dynamic data
        'mtmq_y_train_rnn': mtmq_y_train_rnn, 'mtmq_y_val_rnn': mtmq_y_val_rnn, 'mtmq_y_test_rnn': mtmq_y_test_rnn, 'mtmq_y_train_full_rnn': mtmq_y_train_full_rnn, # multi-task mulit-quantile targets for dynamic models
        'y_train_rnn': y_train_rnn, 'y_val_rnn': y_val_rnn, 'y_test_rnn': y_test_rnn, 'y_train_full_rnn': y_train_full_rnn, # single-task targets for dynamic models
        'mq_y_train_rnn': mq_y_train_rnn, 'mq_y_val_rnn': mq_y_val_rnn, 'mq_y_test_rnn': mq_y_test_rnn, 'mq_y_train_full_rnn': mq_y_train_full_rnn, # multi-quantile targets for dynamic models
        'X_inputs_train_rnn': X_inputs_train_rnn, 'X_inputs_val_rnn': X_inputs_val_rnn, 'X_inputs_test_rnn': X_inputs_test_rnn, 'X_inputs_train_full_rnn': X_inputs_train_full_rnn, # data for multi-model dyanmic models
        'X_test_rnn_full': X_test_rnn_full, 'X_inputs_test_rnn_full': X_inputs_test_rnn_full, # data for dyanmic models on full test set
    }
    meta_data = {
        'target': target_idx,
        'train_start': train_start,
        'train_end': train_end,
        'val_start': val_start,
        'val_end': val_end,
        'test_start': test_start,
        'test_end': test_end
    }
    return non_rnn_data, rnn_data, meta_data
