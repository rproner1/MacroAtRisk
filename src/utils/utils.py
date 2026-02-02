# ******************************** Dependencies ********************************

import pandas as pd 
import numpy as np
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy import integrate
import tensorflow as tf
import keras
from keras.layers import Activation, Lambda, Input, Dense, RNN, LSTM, GRU, LayerNormalization, BatchNormalization, Concatenate, Dropout, Bidirectional, Add, Subtract, Conv1D, AveragePooling1D, MaxPooling1D, GlobalAveragePooling1D, Flatten, Reshape, TimeDistributed
from keras.callbacks import EarlyStopping 
from keras.models import Sequential, Model, load_model
from keras.regularizers import L1,L1L2
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_pinball_loss, mean_squared_error
from datetime import datetime
from dateutil.relativedelta import relativedelta
import optuna
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools import add_constant
import random
import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from joblib import Parallel, delayed
import os
from typing import Union, List, Callable, Type, List, Tuple
import json

import portalocker

# For custom layers
from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell
from keras.src.layers.rnn.rnn import RNN



def remove_outliers(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes outliers from a dataset by replacing them with NaN.

    Parameters:
        X (numpy.ndarray): Dataset (one series per column).

    Returns:
        Y (numpy.ndarray): Dataset with outliers replaced with NaN.
        n (numpy.ndarray): Number of outliers found in each series.
    """
    import numpy as np
    # Calculate the median of each series
    median_X = np.nanmedian(X, axis=0)

    # Calculate the interquartile range (IQR) of each series
    Q1 = np.nanpercentile(X, 25, axis=0)
    Q3 = np.nanpercentile(X, 75, axis=0)
    IQR = Q3 - Q1

    # Determine outliers
    Z = np.abs(X - median_X)
    outlier = Z > (10 * IQR)

    # Replace outliers with NaN
    Y = X.copy()
    Y[outlier] = np.nan

    # Count the number of outliers in each series
    n = np.sum(outlier, axis=0)

    return Y, n

def prepare_missing(rawdata: np.ndarray, tcode: int) -> np.ndarray:
    """
    Transforms raw data based on each series' transformation code.

    Parameters:
        rawdata (numpy.ndarray): Raw data (each column is a series).
        tcode (list or numpy.ndarray): Transformation codes for each series.

    Returns:
        numpy.ndarray: Transformed data.
    """
    import numpy as np
    # Initialize output variable
    yt = []

    # Number of series
    N = rawdata.shape[1]

    # Perform transformation using the subfunction transxf
    for i in range(N):
        transformed_series = transxf(rawdata[:, i], tcode[i])
        yt.append(transformed_series)

    # Stack transformed series column-wise
    yt = np.column_stack(yt)
    return yt

def transxf(x: np.ndarray, tcode: int) -> np.ndarray:
    """
    Transforms a single series as specified by a given transformation code.

    Parameters:
        x (numpy.ndarray): Series (1D array) to be transformed.
        tcode (int): Transformation code (1-7).

    Returns:
        numpy.ndarray: Transformed series.
    """
    import numpy as np
    # Number of observations
    n = len(x)

    # Value close to zero
    small = 1e-6

    # Allocate output variable
    y = np.full(n, np.nan)

    # Apply transformation based on the transformation code
    if tcode == 1:
        # Level (no transformation): x(t)
        y = x

    elif tcode == 2:
        # First difference: x(t) - x(t-1)
        y[1:] = x[1:] - x[:-1]

    elif tcode == 3:
        # Second difference: (x(t) - x(t-1)) - (x(t-1) - x(t-2))
        y[2:] = x[2:] - 2 * x[1:-1] + x[:-2]

    elif tcode == 4:
        # Natural log: ln(x)
        y = np.where(x > small, np.log(x), np.nan)

    elif tcode == 5:
        # First difference of natural log: ln(x) - ln(x-1)
        x = np.where(x > small, x, np.nan)  # Replace invalid values with NaN
        x = np.log(x)
        y[1:] = x[1:] - x[:-1]

    elif tcode == 6:
        # Second difference of natural log: (ln(x) - ln(x-1)) - (ln(x-1) - ln(x-2))
        x = np.where(x > small, x, np.nan)  # Replace invalid values with NaN
        x = np.log(x)
        y[2:] = x[2:] - 2 * x[1:-1] + x[:-2]

    elif tcode == 7:
        # First difference of percent change: (x(t)/x(t-1) - 1) - (x(t-1)/x(t-2) - 1)
        y1 = np.full(n, np.nan)
        y1[1:] = np.where(x[:-1] != 0, (x[1:] - x[:-1]) / x[:-1], np.nan)
        y[2:] = y1[2:] - y1[1:-1]

    return y

def prepare_quantile_data(target: int, time_steps: int, 
                          targets_path: str, input_paths: List[str],
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

# Utils

def get_mt_labels(data: Union[pd.DataFrame, np.ndarray]):

    if not isinstance(data, np.ndarray):
        data = data.values

    mt_labels = [data[:,j] for j in range(data.shape[1])]

    return mt_labels


def split_sequences(data: Union[np.ndarray, pd.DataFrame], n_timesteps: int, n_targets: int):
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    n_rows, n_cols = data.shape
    if n_rows < n_timesteps:
        raise ValueError(f"Not enough rows for sequences: n_rows={n_rows}, n_timesteps={n_timesteps}")

    n_features = n_cols - n_targets
    n_samples = n_rows - n_timesteps + 1

    X = np.zeros((n_samples, n_timesteps, n_features), dtype=data.dtype)
    y = np.zeros((n_samples, n_targets), dtype=data.dtype)

    for i in range(n_samples):
        end_ind = i + n_timesteps
        X[i] = data[i:end_ind, :n_features]
        y[i] = data[end_ind - 1, -n_targets:]

    # return X (samples, timesteps, features) and y (samples, 1, n_targets) as expected downstream
    return X, y

# Loss functions

def tilted_loss(y_true: tf.Tensor, y_pred: tf.Tensor, q: float=0.5) -> tf.Tensor:
    """
    Computes tilted loss for quantile regression.

    Parameters:
    ----------
    y_true: 
    """
    # Cast both as float32 to avoid dtype issues
    e = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e), axis=-1)

def temporal_smooth_penalty(y_pred):
    diff = y_pred[1:,:] - y_pred[:-1,:]
    return tf.reduce_mean(tf.abs(diff))

@keras.saving.register_keras_serializable()
def make_tilted_loss(q: Union[float, int]):
    q = float(q/100.0) if q > 1 else float(q)
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))
    loss.__name__ = f"tilted_loss_{int(q*100)}"
    return loss

@keras.saving.register_keras_serializable()
def make_total_tilted_loss(quantiles: List[Union[float, int]], q_loss_weights: List[float]=[1.0]*5):
    """
    Returns a loss function that computes the mean of tilted losses for the given quantiles.

    Parameters:
    ----------
    quantiles: List of quantiles (as floats in (0,1) or ints in (1,100))
        The quantiles for which to compute the tilted losses.
    """
    qs = [q/100.0 if q > 1 else float(q) for q in quantiles]
    loss_fns = [make_tilted_loss(q) for q in qs]
    def total_tilted_loss(y_true, y_pred):
        # y_pred shape: (batch, len(quantiles))
        losses = []
        # Compute loss on each quantile
        for i, lf in enumerate(loss_fns):
            losses.append(q_loss_weights[i] * lf(y_true, y_pred[:, i:i+1]))

        return tf.add_n(losses) / tf.cast(len(losses), tf.float32) 

    total_tilted_loss.__name__ = "total_tilted_loss_" + "_".join(str(int(q*100)) for q in qs)
    return total_tilted_loss


# Builder functions

@keras_export("keras.layers.MyGRUCell")
class MyGRUCell(Layer, DropoutRNNCell):
    """Cell class for the GRU layer.

    This class processes one step within the whole time sequence input, whereas
    `keras.layer.GRU` processes the whole sequence.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
            applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation
            of the recurrent state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). False = "before",
            True = "after" (default and cuDNN compatible).
        seed: Random seed for dropout.

    Call arguments:
        inputs: A 2D tensor, with shape `(batch, features)`.
        states: A 2D tensor with shape `(batch, units)`, which is the state
            from the previous time step.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.

    Example:

    >>> inputs = np.random.random((32, 10, 8))
    >>> rnn = keras.layers.RNN(keras.layers.GRUCell(4))
    >>> output = rnn(inputs)
    >>> output.shape
    (32, 4)
    >>> rnn = keras.layers.RNN(
    ...    keras.layers.GRUCell(4),
    ...    return_sequences=True,
    ...    return_state=True)
    >>> whole_sequence_output, final_state = rnn(inputs)
    >>> whole_sequence_output.shape
    (32, 10, 4)
    >>> final_state.shape
    (32, 4)
    """

    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        reset_after=True,
        seed=None,
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        implementation = kwargs.pop("implementation", 2)
        super().__init__(**kwargs)
        self.implementation = implementation
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.seed = seed
        self.seed_generator = backend.random.SeedGenerator(seed=seed)

        self.reset_after = reset_after
        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape):
        super().build(input_shape)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        if self.use_bias:
            if not self.reset_after:
                bias_shape = (3 * self.units,)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU
                # biases `(2 * 3 * self.units,)`, so that we can distinguish the
                # classes when loading and converting saved weights.
                bias_shape = (2, 3 * self.units)
            self.bias = self.add_weight(
                shape=bias_shape,
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=False):
        h_tm1 = (
            states[0] if tree.is_nested(states) else states
        )  # previous state

        dp_mask = self.get_dropout_mask(inputs)
        rec_dp_mask = self.get_recurrent_dropout_mask(h_tm1)

        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = (
                    ops.squeeze(e, axis=0)
                    for e in ops.split(self.bias, self.bias.shape[0], axis=0)
                )

        if training and 0.0 < self.dropout < 1.0:
            inputs = inputs * dp_mask

        if self.implementation == 1:
            inputs_z = inputs
            inputs_r = inputs
            inputs_h = inputs

            x_z = ops.matmul(inputs_z, self.kernel[:, : self.units])
            x_r = ops.matmul(
                inputs_r, self.kernel[:, self.units : self.units * 2]
            )
            x_h = ops.matmul(inputs_h, self.kernel[:, self.units * 2 :])

            if self.use_bias:
                x_z += input_bias[: self.units]
                x_r += input_bias[self.units : self.units * 2]
                x_h += input_bias[self.units * 2 :]

            h_tm1_z = h_tm1
            h_tm1_r = h_tm1
            h_tm1_h = h_tm1

            recurrent_z = ops.matmul(
                h_tm1_z, self.recurrent_kernel[:, : self.units]
            )
            recurrent_r = ops.matmul(
                h_tm1_r, self.recurrent_kernel[:, self.units : self.units * 2]
            )
            if self.reset_after and self.use_bias:
                recurrent_z += recurrent_bias[: self.units]
                recurrent_r += recurrent_bias[self.units : self.units * 2]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = ops.matmul(
                    h_tm1_h, self.recurrent_kernel[:, self.units * 2 :]
                )
                if self.use_bias:
                    recurrent_h += recurrent_bias[self.units * 2 :]
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = ops.matmul(
                    r * h_tm1_h, self.recurrent_kernel[:, self.units * 2 :]
                )

            hh = self.activation(x_h + recurrent_h)
        else:
            # inputs projected by all gate matrices at once
            matrix_x = ops.matmul(inputs, self.kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x += input_bias

            x_z, x_r, x_h = ops.split(matrix_x, 3, axis=-1)

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = ops.matmul(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner += recurrent_bias
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = ops.matmul(
                    h_tm1, self.recurrent_kernel[:, : 2 * self.units]
                )

            recurrent_z = matrix_inner[:, : self.units]
            recurrent_r = matrix_inner[:, self.units : self.units * 2]
            recurrent_h = matrix_inner[:, self.units * 2 :]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            if self.reset_after:
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = ops.matmul(
                    r * h_tm1, self.recurrent_kernel[:, 2 * self.units :]
                )

            hh = self.activation(x_h + recurrent_h)


        if training and 0.0 < self.recurrent_dropout < 1.0:
            hh = hh * rec_dp_mask

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh
        new_state = [h] if tree.is_nested(states) else h
        return h, new_state

    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "reset_after": self.reset_after,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def get_initial_state(self, batch_size=None):
        return [
            ops.zeros((batch_size, self.state_size), dtype=self.compute_dtype)
        ]

@keras_export("keras.layers.MyGRU")
class MyGRU(RNN):
    """Gated Recurrent Unit - Cho et al. 2014.

    Based on available runtime hardware and constraints, this layer
    will choose different implementations (cuDNN-based or backend-native)
    to maximize the performance. If a GPU is available and all
    the arguments to the layer meet the requirement of the cuDNN kernel
    (see below for details), the layer will use a fast cuDNN implementation
    when using the TensorFlow backend.

    The requirements to use the cuDNN implementation are:

    1. `activation` == `tanh`
    2. `recurrent_activation` == `sigmoid`
    3. `dropout` == 0 and `recurrent_dropout` == 0
    4. `unroll` is `False`
    5. `use_bias` is `True`
    6. `reset_after` is `True`
    7. Inputs, if use masking, are strictly right-padded.
    8. Eager execution is enabled in the outermost context.

    There are two variants of the GRU implementation. The default one is based
    on [v3](https://arxiv.org/abs/1406.1078v3) and has reset gate applied to
    hidden state before matrix multiplication. The other one is based on
    [original](https://arxiv.org/abs/1406.1078v1) and has the order reversed.

    The second variant is compatible with CuDNNGRU (GPU-only) and allows
    inference on CPU. Thus it has separate biases for `kernel` and
    `recurrent_kernel`. To use this variant, set `reset_after=True` and
    `recurrent_activation='sigmoid'`.

    For example:

    >>> inputs = np.random.random((32, 10, 8))
    >>> gru = keras.layers.GRU(4)
    >>> output = gru(inputs)
    >>> output.shape
    (32, 4)
    >>> gru = keras.layers.GRU(4, return_sequences=True, return_state=True)
    >>> whole_sequence_output, final_state = gru(inputs)
    >>> whole_sequence_output.shape
    (32, 10, 4)
    >>> final_state.shape
    (32, 4)

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step.
            Default: sigmoid (`sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent
            state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation"). Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        seed: Random seed for dropout.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition
            to the output. Default: `False`.
        go_backwards: Boolean (default `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default: `False`). If `True`, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default: `False`).
            If `True`, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). `False` is `"before"`,
            `True` is `"after"` (default and cuDNN compatible).
        use_cudnn: Whether to use a cuDNN-backed implementation. `"auto"` will
            attempt to use cuDNN when feasible, and will fallback to the
            default implementation if not.

    Call arguments:
        inputs: A 3D tensor, with shape `(batch, timesteps, feature)`.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether
            a given timestep should be masked  (optional).
            An individual `True` entry indicates that the corresponding timestep
            should be utilized, while a `False` entry indicates that the
            corresponding timestep should be ignored. Defaults to `None`.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed to the
            cell when calling it. This is only relevant if `dropout` or
            `recurrent_dropout` is used  (optional). Defaults to `None`.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell (optional, `None` causes creation
            of zero-filled initial state tensors). Defaults to `None`.
    """

    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        reset_after=True,
        use_cudnn="auto",
        **kwargs,
    ):
        cell = MyGRUCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            reset_after=reset_after,
            dtype=kwargs.get("dtype", None),
            trainable=kwargs.get("trainable", True),
            name="gru_cell",
            seed=seed,
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            activity_regularizer=activity_regularizer,
            **kwargs,
        )
        self.input_spec = InputSpec(ndim=3)
        if use_cudnn not in ("auto", True, False):
            raise ValueError(
                "Invalid valid received for argument `use_cudnn`. "
                "Expected one of {'auto', True, False}. "
                f"Received: use_cudnn={use_cudnn}"
            )
        self.use_cudnn = use_cudnn
        if (
            backend.backend() == "tensorflow"
            and backend.cudnn_ok(
                cell.activation,
                cell.recurrent_activation,
                self.unroll,
                cell.use_bias,
                reset_after=reset_after,
            )
            and use_cudnn in (True, "auto")
        ):
            self.supports_jit = False

    def inner_loop(self, sequences, initial_state, mask, training=False):
        if tree.is_nested(initial_state):
            initial_state = initial_state[0]
        if tree.is_nested(mask):
            mask = mask[0]
        if self.use_cudnn in ("auto", True):
            if not self.dropout and not self.recurrent_dropout:
                try:
                    # Backends are allowed to specify (optionally) optimized
                    # implementation of the inner GRU loop. In the case of
                    # TF for instance, it will leverage cuDNN when feasible, and
                    # it will raise NotImplementedError otherwise.
                    out = backend.gru(
                        sequences,
                        initial_state,
                        mask,
                        kernel=self.cell.kernel,
                        recurrent_kernel=self.cell.recurrent_kernel,
                        bias=self.cell.bias,
                        activation=self.cell.activation,
                        recurrent_activation=self.cell.recurrent_activation,
                        return_sequences=self.return_sequences,
                        go_backwards=self.go_backwards,
                        unroll=self.unroll,
                        reset_after=self.cell.reset_after,
                    )
                    # We disable jit_compile for the model in this case,
                    # since cuDNN ops aren't XLA compatible.
                    if backend.backend() == "tensorflow":
                        self.supports_jit = False
                    return out
                except NotImplementedError:
                    pass
        if self.use_cudnn is True:
            raise ValueError(
                "use_cudnn=True was specified, "
                "but cuDNN is not supported for this layer configuration "
                "with this backend. Pass use_cudnn='auto' to fallback "
                "to a non-cuDNN implementation."
            )
        return super().inner_loop(
            sequences, initial_state, mask=mask, training=training
        )

    def call(self, sequences, initial_state=None, mask=None, training=False):
        return super().call(
            sequences, mask=mask, training=training, initial_state=initial_state
        )

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def reset_after(self):
        return self.cell.reset_after

    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "reset_after": self.reset_after,
            "seed": self.cell.seed,
        }
        base_config = super().get_config()
        del base_config["cell"]
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class GRUResidualBlock(Layer):
    def __init__(self, units, norm: bool=False, **kwargs):
        super().__init__()
        self.gru1 = GRU(units, **kwargs)
        self.gru2 = GRU(units, **kwargs)
        self.norm = norm
        if self.norm:
            self.norm1 = LayerNormalization()
            self.norm2 = LayerNormalization()
        self.add1 = Add()
        self.add2 = Add()
        self.relu1 = Activation('relu')

    def call(self, inputs):

        x = self.gru1(inputs)

        if self.norm:
            x = self.norm1(x)

        x = self.gru2(x)

        if self.norm:
            x = self.norm2(x)

        x = self.add2([x, inputs])

        x = self.relu1(x)

        return x

class DenseResidualBlock(Layer):
    def __init__(self, units, norm: bool=False, **kwargs):
        super().__init__()
        self.dense1 = Dense(units, **kwargs)
        self.dense2 = Dense(units, **kwargs)
        self.norm = norm
        if self.norm:
            self.norm1 = LayerNormalization()
            self.norm2 = LayerNormalization()
        self.add1 = Add()
        self.add2 = Add()
        self.relu1 = Activation('relu')

    def call(self, inputs):

        x = self.dense1(inputs)

        if self.norm:
            x = self.norm1(x)

        x = self.dense2(x)

        if self.norm:
            x = self.norm2(x)

        x = self.add2([x, inputs])

        x = self.relu1(x)

        return x
    

def build_qlr(q: Union[float, int]=0.5, l1: float=0.0, l2: float=0.0, lr: float=0.001):
    model = Sequential()
    model.add(Dense(int(1), activation = 'linear', kernel_regularizer=L1L2(l1=l1,l2=l2)))
    opt = Adam(learning_rate=lr)
    loss = make_tilted_loss(q)
    model.compile(loss = loss, optimizer = opt)
    return model


def build_nn(q: Union[float, int]=0.5, n_dense_layers: int=1, n_nodes: int=16,
             l1: float=0.0, l2: float=0.0, lr: float=0.001, norm_fn: str='batch'):
    """
    Builds a single task quantile regression neural network.

    Parameters
    ----------
    l1: float (default=0.0)
        The l1 penalty for the neural network weights
    l2: float (default=0.0)
        The l2 penalty for the neural network weights
    q: float in (0.0,1.0) (default=0.5, i.e., the median)
        The quantile of interest used in defining the tilted loss.
    lr: float (default=0.001)
        The initial learning rate used for the optimization algorithm.
    n_dense_layers: int (default=1)
        The number of dense layers in the model.
    n_nodes: int (default=32)
        The number of nodes in each shared layer.
        
    Returns
    ----------
    model: a compiled keras model
    """

    norm_fn = norm_fn.lower()
    if norm_fn == 'batch':
        norm_fn = BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")

    model = Sequential()
    for i in range(1,n_dense_layers+1):
        model.add(Dense(n_nodes, 'relu', kernel_regularizer=L1L2(l1,l2)))
        if i<n_dense_layers:
            model.add(norm_fn())

    model.add(Dense(int(1), activation='linear', kernel_regularizer=L1L2(l1,l2)))
    
    opt = Adam(learning_rate=lr)
    loss = make_tilted_loss(q)
    model.compile(loss = loss, optimizer = opt)
    return model


def build_rnn(q: Union[float, int]=0.5, n_recurrent_layers: int=2, n_dense_layers: int=1, n_nodes: int=32,
               l1: float=0.0, l2: float=0.0, lr: float=0.001, recurrent_layer_type: str='gru'):
    """
    Builds a single task quantile regression recurrent neural network.

    Parameters
    ----------
    l1: float (default=0.0)
        The l1 penalty for the neural network weights
    l2: float (default=0.0)
        The l2 penalty for the neural network weights
    q: float in (0.0,1.0) (default=0.5, i.e., the median)
        The quantile of interest used in defining the tilted loss.
    lr: float (default=0.001)
        The initial learning rate used for the optimization algorithm.
    n_recurrent_layers: int (default=1)
        The number of recurrent layers in the model.
    n_nodes: int (default=32)
        The number of nodes in each recurrent layer.
    n_dense_layers: int (default=1)
        The number of shared layers in the model.
    n_nodes: int (default=32)
        The number of nodes in each shared layer.
        
    Returns
    ----------
    model: a compiled keras model
    """
    recurrent_layer_type = recurrent_layer_type.lower()
    if recurrent_layer_type == 'lstm':
        recurrent_layer_type = LSTM
    elif recurrent_layer_type == 'gru':    
        recurrent_layer_type = GRU

    # Recurrent layers
    model = Sequential()
    for i in range(1,n_recurrent_layers+1):
        model.add(recurrent_layer_type(n_nodes, return_sequences=(i < n_recurrent_layers), kernel_regularizer=L1L2(l1,l2)))

    # Dense layers
    for i in range(1,n_dense_layers+1):
        model.add(Dense(n_nodes, 'relu', kernel_regularizer=L1L2(l1,l2)))

    # Output layer    
    model.add(Dense(int(1), activation='linear', kernel_regularizer=L1L2(l1,l2)))
    
    opt = Adam(learning_rate=lr)
    loss = make_tilted_loss(q)
    model.compile(loss = loss, optimizer = opt)
    return model


def build_mq_v0(input_shape: tuple, n_shared_layers: int=1, n_qtask_layers: int=2, n_nodes: int=32, l1: float=0.0, l2: float=0.0, lr: float=0.001, norm_fn: str='batch', quantiles: list[int]=[0.05,0.25,0.50,0.75,0.95], task_specific_norm: bool=False):

    if norm_fn == 'batch':
        norm_fn = BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")

    inputs = Input(shape=input_shape)

    shared_layers = []
    for i in range(1, n_shared_layers + 1):
        shared_layers.append(
            Dense(n_nodes, activation='relu', kernel_regularizer=L1L2(l1,l2))
        )
        shared_layers.append(norm_fn())

    shared_net = Sequential(shared_layers, name='shared')(inputs)

    outputs = []
    for q in quantiles:
        name = f"Q{q}"
        
        qtask_layers = []
        for i in range(1, n_qtask_layers+1):
            qtask_layers.append(
                Dense(
                    n_nodes, 
                    activation='relu',
                    kernel_regularizer=L1L2(l1, l2)
                )
            )
            if task_specific_norm and i < n_qtask_layers:
                qtask_layers.append(norm_fn())

        # Append output node
        qtask_layers.append(
            Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2))
        )
        
        # Build output net
        output_q = Sequential(qtask_layers, name=name)(shared_net)

        outputs.append(output_q)

    out_concat = Concatenate(name='out_layer')(outputs)

    model = Model(inputs=inputs, outputs=out_concat)

    loss = make_total_tilted_loss(quantiles)

    model.compile(
        loss=loss, 
        optimizer=Adam(learning_rate=lr) 
    )

    return model


def build_mq(input_shapes: Union[tuple, list[tuple]], n_input_processing_layers: int=2, n_shared_layers: int=1, n_qtask_layers: int=2, n_nodes: int=32, l1: float=0.0, l2: float=0.0,  lr: float=0.001, norm_fn: str='batch', quantiles: list[int]=[0.05,0.25,0.50,0.75,0.95], task_specific_norm: bool=False):

    if norm_fn == 'batch':
        norm_fn = BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")

    inputs = []
    input_processing_nets = []
    for i, shape in enumerate(input_shapes):
        net_input = Input(shape=shape)
        inputs.append(net_input)
        
        input_processing_layers = []
        # build input processing layers
        for j in range(1, n_input_processing_layers + 1):
            input_processing_layers.append(Dense(n_nodes, activation='relu', kernel_regularizer=L1L2(l1,l2)))
            input_processing_layers.append(norm_fn())

        # Make model
        input_processing_net = Sequential(input_processing_layers, name=f'input_processing_{i}')(net_input)
        input_processing_nets.append(input_processing_net)

    # Concatenate the layers from each input
    concat = Concatenate()(input_processing_nets)

    shared_layers = []
    for i in range(1, n_shared_layers + 1):
        shared_layers.append(
            Dense(
                n_nodes, 
                activation='relu', 
                kernel_regularizer=L1L2(l1,l2)
            )
        )
        shared_layers.append(norm_fn())


    shared_net = Sequential(shared_layers, name='shared')(concat)


    outputs = []
    for q in quantiles:
        name = f"Q{q}"
        
        qtask_layers = []
        for i in range(1, n_qtask_layers+1):
            qtask_layers.append(
                Dense(
                    n_nodes, 
                    activation='relu',
                    kernel_regularizer=L1L2(l1, l2)
                )
            )
            if task_specific_norm and i < n_qtask_layers:
                qtask_layers.append(norm_fn())

        # Append output node
        qtask_layers.append(
            Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2))
        )
        
        # Build output net
        output_q = Sequential(qtask_layers, name=name)(shared_net)

        outputs.append(output_q)

    out_concat = Concatenate(name='out_layer')(outputs)

    model = Model(inputs=inputs, outputs=out_concat)

    loss = make_total_tilted_loss(quantiles)

    model.compile(
        loss=loss, 
        optimizer=Adam(learning_rate=lr) 
    )

    return model


def build_dmq_v5(
        input_shape: tuple, 
        n_recurrent_layers: int=1,
        n_shared_layers: int=1, 
        n_qtask_layers: int=1, 
        n_conv_filters: int=32,
        kernel_size: int=12,
        n_recurrent_nodes: int=32,
        n_shared_nodes: int=32,
        n_task_nodes: int=32,
        l1: float=0.0, 
        l2: float=0.0, 
        rec_drop: float=0.0,
        dropout: float=0.0,
        lr: float=0.001, 
        norm_fn: str='layer', 
        recurrent_layer_type: str='gru', 
        lower_quantiles: List[float]=[0.05,0.25], 
        upper_quantiles: List[float]=[0.75,0.95],
        recurrent_norm: bool=False,
        shared_norm: bool=False, 
        task_specific_norm: bool=False, 
        loss_weights: list[float]=[1.0]*5,
        seed: int=1
    ):

    """
    DMQv0 + quantile spacing and Conv1D at the beginning. Should be used with more timesteps
    """

    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    norm_fn = norm_fn.lower()
    if norm_fn == 'batch':
        norm_fn = BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")
    
    recurrent_layer_type = recurrent_layer_type.lower()
    if recurrent_layer_type == 'lstm':
        recurrent_layer_type = LSTM
    elif recurrent_layer_type == 'gru':   
        recurrent_layer_type = GRU
    elif recurrent_layer_type == 'mygru':
        recurrent_layer_type = MyGRU
    else:
        raise ValueError("recurrent_layer_type must be 'lstm' or 'gru'")

    inputs = Input(shape=input_shape)

    shared_layers = []

    shared_layers.append(
        Conv1D(
            32,
            12,
            activation='relu',
            padding='causal'
        )
    )
    shared_layers.append(
        Conv1D(
            64,
            12,
            activation='relu',
            padding='causal'
        )
    )

    for i in range(1, n_recurrent_layers + 1):
        shared_layers.append(
            recurrent_layer_type(
                n_recurrent_nodes, 
                return_sequences=(i < n_recurrent_layers), 
                kernel_regularizer=L1L2(l1,l2), 
                recurrent_dropout=rec_drop,
                kernel_initializer=initializer
            )
        )
        if recurrent_norm:
            shared_layers.append(norm_fn())

    
    for i in range(1, n_shared_layers + 1):
        shared_layers.append(
            Dense(
                n_shared_nodes, 
                activation='relu', 
                kernel_regularizer=L1L2(l1,l2),
                kernel_initializer=initializer
            )
        )
        if shared_norm:
            shared_layers.append(norm_fn())

    # shared_layers.append(norm_fn())

    shared_net = Sequential(shared_layers, name='shared')(inputs)

    # Median head
    median_head = Sequential(name='Q50')
    for i in range(1, n_qtask_layers+1):
        median_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            median_head.add(norm_fn())
        if dropout > 0.0 and i < n_qtask_layers:
            median_head.add(Dropout(dropout))

    median_head.add(
        Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2))
    )

    median_output = median_head(shared_net)

    # Lower quantile head
    lower_resid_head = Sequential(name='Q_lower_raw')
    for i in range(1, n_qtask_layers+1):
        lower_resid_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            lower_resid_head.add(norm_fn())
        if dropout > 0.0 and i < n_qtask_layers:
            lower_resid_head.add(Dropout(dropout))

    lower_resid_head.add(
        Dense(len(lower_quantiles), activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
    )
    lower_raw = lower_resid_head(shared_net)
    lower_resid = Activation('softplus')(lower_raw)


    # Upper quantile head
    upper_resid_head = Sequential(name='Q_upper_raw')
    for i in range(1, n_qtask_layers+1):
        upper_resid_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            upper_resid_head.add(norm_fn())
        if dropout > 0.0 and i < n_qtask_layers:    
            upper_resid_head.add(Dropout(dropout))
        
    upper_resid_head.add(
        Dense(len(upper_quantiles), activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
    )
    upper_raw = upper_resid_head(shared_net)
    upper_resid = Activation('softplus')(upper_raw)

    # Combine outputs
    Q50 = median_output
    Q_lower = Q50 - lower_resid
    Q_upper = Q50 + upper_resid

    out_concat = Concatenate()([Q_lower, Q50, Q_upper])

    model = Model(inputs=inputs, outputs=out_concat)

    loss = make_total_tilted_loss(lower_quantiles + [0.5] + upper_quantiles, q_loss_weights=loss_weights)

    model.compile(
        loss=loss, 
        optimizer=Adam(learning_rate=lr),
    )

    return model


def build_dmq_v4(
        input_shape: tuple, 
        n_recurrent_layers: int=2, 
        n_shared_layers: int=1, 
        n_qtask_layers: int=2, 
        n_recurrent_nodes: int=32,
        n_shared_nodes: int=32,
        n_task_nodes: int=32,
        l1: float=0.0, 
        l2: float=0.0, 
        lr: float=0.001, 
        rec_drop: float=0.0,
        dropout: float=0.0,
        norm_fn: str='layer', 
        recurrent_layer_type: str='gru', 
        lower_quantiles: List[float]=[0.05,0.25], 
        upper_quantiles: List[float]=[0.75,0.95],
        recurrent_norm: bool=False,
        shared_norm: bool=False, 
        task_specific_norm: bool=False, 
        loss_weights: list[float]=[1.0]*5,
        seed: int=1
    ):

    """
    DMQv2 + skip connection to task layers
    """

    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    norm_fn = norm_fn.lower()
    if norm_fn == 'batch':
        norm_fn = BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")
    
    recurrent_layer_type = recurrent_layer_type.lower()
    if recurrent_layer_type == 'lstm':
        recurrent_layer_type = LSTM
    elif recurrent_layer_type == 'gru':   
        recurrent_layer_type = GRU
    else:
        raise ValueError("recurrent_layer_type must be 'lstm' or 'gru'")


    inputs = Input(shape=input_shape)

    # ============================================================
    # LEARNABLE RESIDUAL SKIP (corrected)
    # ============================================================

    # last time step of raw input
    skip_input = Lambda(lambda x: x[:, -1, :], output_shape=lambda s: (s[0],s[2]), name="skip_raw")(inputs)

    # normalize skip so spikes don't dominate
    skip_norm = norm_fn(name="skip_norm")(skip_input)

    # learnable linear projection (zero initialized)
    skip_proj = Dense(
        n_shared_nodes,
        activation="linear",
        kernel_initializer=tf.keras.initializers.Zeros(),
        bias_initializer=tf.keras.initializers.Zeros(),
        name="skip_projection"
    )(skip_norm)

    # ============================================================
    # SHARED LAYERS
    # ============================================================

    shared_layers = []
    
    for i in range(1, n_recurrent_layers + 1):
        shared_layers.append(
            recurrent_layer_type(
                n_recurrent_nodes, 
                return_sequences=(i < n_recurrent_layers), 
                kernel_regularizer=L1L2(l1,l2), 
                kernel_initializer=initializer,
                recurrent_dropout=rec_drop
            )
        )
        if recurrent_norm:
            shared_layers.append(norm_fn())

    
    for i in range(1, n_shared_layers + 1):
        shared_layers.append(
            Dense(
                n_shared_nodes, 
                activation='relu', 
                kernel_regularizer=L1L2(l1,l2),
                kernel_initializer=initializer
            )
        )
        if shared_norm:
            shared_layers.append(norm_fn())

    # shared_layers.append(norm_fn())

    shared_base = Sequential(shared_layers, name='shared')(inputs)
    shared_net = Add(name="shared_plus_skip")([shared_base, skip_proj])

    # ====================================================
    # MEDIAN HEAD
    # ====================================================

    median_head = Sequential(name='Q50')
    for i in range(1, n_qtask_layers+1):
        median_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            median_head.add(norm_fn())
        
        if dropout > 0.0 and i < n_qtask_layers:
            median_head.add(Dropout(dropout))

    median_head.add(
        Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2))
    )

    median_output = median_head(shared_net)

    # ====================================================
    # LOWER QUANTILE HEADS
    # ====================================================

    fifth_resid_head = Sequential(name='Q5_lower_raw')
    for i in range(1, n_qtask_layers+1):
        fifth_resid_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            fifth_resid_head.add(norm_fn())
        
        if dropout > 0.0 and i < n_qtask_layers:
            fifth_resid_head.add(Dropout(dropout))

    fifth_resid_head.add(
        Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
    )
    fifth_raw = fifth_resid_head(shared_net)
    fifth_resid = Activation('softplus')(fifth_raw)

    twentyfifth_resid_head = Sequential(name='Q25_lower_raw')
    for i in range(1, n_qtask_layers+1):
        twentyfifth_resid_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            twentyfifth_resid_head.add(norm_fn())

        if dropout > 0.0 and i < n_qtask_layers:
            twentyfifth_resid_head.add(Dropout(dropout))

    twentyfifth_resid_head.add(
        Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
    )
    twentyfifth_raw = twentyfifth_resid_head(shared_net)
    twentyfifth_resid = Activation('softplus')(twentyfifth_raw)


    # ====================================================
    # UPPER QUANTILE HEADS
    # ====================================================
    ninetyfifth_resid_head = Sequential(name='Q95_upper_raw')
    for i in range(1, n_qtask_layers+1):
        ninetyfifth_resid_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            ninetyfifth_resid_head.add(norm_fn())

        if dropout > 0.0 and i < n_qtask_layers:
            ninetyfifth_resid_head.add(Dropout(dropout))

    ninetyfifth_resid_head.add(
        Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
    )
    ninetyfifth_raw = ninetyfifth_resid_head(shared_net)
    ninetyfifth_resid = Activation('softplus')(ninetyfifth_raw)

    # Upper quantile head
    seventyfifth_resid_head = Sequential(name='Q75_upper_raw')
    for i in range(1, n_qtask_layers+1):
        seventyfifth_resid_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            seventyfifth_resid_head.add(norm_fn())
        
        if dropout > 0.0 and i < n_qtask_layers:
            seventyfifth_resid_head.add(Dropout(dropout))
    
    seventyfifth_resid_head.add(
        Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
    )
    seventyfifth_raw = seventyfifth_resid_head(shared_net)
    seventyfifth_resid = Activation('softplus')(seventyfifth_raw)

    # ====================================================
    # OUTPUTS
    # ====================================================

    # Combine outputs
    Q50 = median_output
    Q5 = Q50 - fifth_resid
    Q25 = Q50 - twentyfifth_resid
    Q75 = Q50 + seventyfifth_resid
    Q95 = Q50 + ninetyfifth_resid

    out_concat = Concatenate()([Q5, Q25, Q50, Q75, Q95])

    model = Model(inputs=inputs, outputs=out_concat)

    loss = make_total_tilted_loss(lower_quantiles + [0.5] + upper_quantiles, q_loss_weights=loss_weights)

    model.compile(
        loss=loss, 
        optimizer=Adam(learning_rate=lr),
    )

    return model


def build_dmq_v3(
    input_shape: tuple,
    n_recurrent_layers: int = 2,
    n_shared_layers: int = 1,
    n_qtask_layers: int = 2,
    n_recurrent_nodes: int = 32,
    n_shared_nodes: int = 32,
    n_task_nodes: int = 32,
    l1: float = 0.0,
    l2: float = 0.0,
    lr: float = 0.001,
    rec_drop: float=0.0,
    dropout: float=0.0,
    resid_scale: float=1.0,
    norm_fn: str = 'layer',
    recurrent_layer_type: str = 'gru',
    lower_quantiles: List[float] = [0.05, 0.25],
    upper_quantiles: List[float] = [0.75, 0.95],
    recurrent_norm: bool = False,
    shared_norm: bool = False,
    task_specific_norm: bool = False,
    loss_weights: list[float] = [1.0] * 5,
    seed: int=1
):

    """
    DMQv1 + skip connection to task layers
    """

    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    # ----- Normalization selection -----
    norm_fn = norm_fn.lower()
    if norm_fn == 'batch':
        norm_fn = BatchNormalization
    elif norm_fn == 'layer':
        norm_fn = LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")

    # ----- RNN Type -----
    recurrent_layer_type = recurrent_layer_type.lower()
    if recurrent_layer_type == 'lstm':
        rnn_cls = LSTM
    elif recurrent_layer_type == 'gru':
        rnn_cls = GRU
    else:
        raise ValueError("recurrent_layer_type must be 'lstm' or 'gru'")

    # ----- Inputs -----
    inputs = Input(shape=input_shape)

    # ============================================================
    # LEARNABLE RESIDUAL SKIP (corrected)
    # ============================================================

    # last time step of raw input
    skip_input = Lambda(lambda x: x[:, -1, :], output_shape=lambda s: (s[0],s[2]), name="skip_raw")(inputs)

    # normalize skip so spikes don't dominate
    skip_norm = norm_fn(name="skip_norm")(skip_input)

    # learnable linear projection (zero initialized)
    skip_proj = Dense(
        n_shared_nodes,
        activation="linear",
        kernel_initializer=tf.keras.initializers.Zeros(),
        bias_initializer=tf.keras.initializers.Zeros(),
        name="skip_projection"
    )(skip_norm)

    skip_proj = skip_proj * resid_scale

    # ============================================================
    # 2. SHARED REPRESENTATION
    # ============================================================

    shared_layers = []
    for i in range(1, n_recurrent_layers + 1):

        return_seq = (i < n_recurrent_layers)

        shared_layers.append(
            rnn_cls(
                n_recurrent_nodes,
                return_sequences=return_seq,
                kernel_regularizer=L1L2(l1, l2),
                recurrent_dropout=rec_drop,
                kernel_initializer=initializer,
                name=f"rnn_{i}"
            )
        )
        if recurrent_norm:
            shared_layers.append(norm_fn())

    for i in range(1, n_shared_layers + 1):
        shared_layers.append(
            Dense(
                n_shared_nodes,
                activation="relu",
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer,
                name=f"shared_dense_{i}"
            )
        )
        if shared_norm:
            shared_layers.append(norm_fn())

    shared_base = Sequential(shared_layers, name="shared_base")(inputs)

    # Add skip (true residual)
    shared_net = Add(name="shared_plus_skip")([shared_base, skip_proj])

    # ============================================================
    # 3. MEDIAN HEAD
    # ============================================================

    median_head = Sequential(name="Q50_head")
    for i in range(1, n_qtask_layers + 1):
        median_head.add(Dense(n_task_nodes, activation="relu",
                              kernel_regularizer=L1L2(l1, l2),
                              kernel_initializer=initializer,
                              name=f"median_dense_{i}"))
        if task_specific_norm and i < n_qtask_layers:
            median_head.add(norm_fn(name=f"median_norm_{i}"))
        
        if dropout > 0.0 and i < n_qtask_layers:
            median_head.add(Dropout(dropout))

    median_head.add(
        Dense(1, activation="linear", kernel_regularizer=L1L2(l1, l2),
              kernel_initializer=initializer,
              name="median_output")
    )

    Q50 = median_head(shared_net)

    # ============================================================
    # 4. LOWER QUANTILES
    # ============================================================

    lower_resid_head = Sequential(name="Q_lower_head")
    for i in range(1, n_qtask_layers + 1):
        lower_resid_head.add(
            Dense(n_task_nodes, activation="relu",
                  kernel_regularizer=L1L2(l1, l2),
                  kernel_initializer=initializer,
                  name=f"lower_dense_{i}")
        )
        if task_specific_norm and i < n_qtask_layers:
            lower_resid_head.add(norm_fn(name=f"lower_norm_{i}"))
        
        if dropout > 0.0 and i < n_qtask_layers:
            lower_resid_head.add(Dropout(dropout))

    lower_resid_head.add(
        Dense(len(lower_quantiles), activation="linear",
              kernel_regularizer=L1L2(l1, l2),
              kernel_initializer=initializer,
              name="lower_raw")
    )

    lower_raw = lower_resid_head(shared_net)
    lower_resid = Activation('softplus')(lower_raw)
    Q_lower = Subtract(name="Q_lower")([Q50, lower_resid])

    # ============================================================
    # 5. UPPER QUANTILES
    # ============================================================

    upper_resid_head = Sequential(name="Q_upper_head")
    for i in range(1, n_qtask_layers + 1):
        upper_resid_head.add(
            Dense(n_task_nodes, activation="relu",
                  kernel_regularizer=L1L2(l1, l2),
                  kernel_initializer=initializer,
                  name=f"upper_dense_{i}")
        )
        if task_specific_norm and i < n_qtask_layers:
            upper_resid_head.add(norm_fn(name=f"upper_norm_{i}"))
        
        if dropout > 0.0 and i < n_qtask_layers:
            upper_resid_head.add(Dropout(dropout))

    upper_resid_head.add(
        Dense(len(upper_quantiles), activation="linear",
              kernel_regularizer=L1L2(l1, l2),
              kernel_initializer=initializer,
              name="upper_raw")
    )

    upper_raw = upper_resid_head(shared_net)
    upper_resid = Activation('softplus')(upper_raw)
    Q_upper = Add(name="Q_upper")([Q50, upper_resid])

    # ============================================================
    # 6. CONCAT OUTPUT
    # ============================================================

    outputs = Concatenate(name="all_quantiles")([Q_lower, Q50, Q_upper])

    model = Model(inputs=inputs, outputs=outputs)

    # correct loss function
    loss = make_total_tilted_loss(
        lower_quantiles + [0.5] + upper_quantiles,
        
        q_loss_weights=loss_weights
    )

    model.compile(
        loss=loss,
        optimizer=Adam(learning_rate=lr),
    )

    return model


def build_dmq_v2(
        input_shape: tuple, 
        n_recurrent_layers: int=2, 
        n_shared_layers: int=1, 
        n_qtask_layers: int=2, 
        n_recurrent_nodes: int=32,
        n_shared_nodes: int=32,
        n_task_nodes: int=32,
        l1: float=0.0, 
        l2: float=0.0, 
        lr: float=0.001, 
        rec_drop: float=0.0,
        dropout: float=0.0,
        norm_fn: str='layer', 
        recurrent_layer_type: str='gru', 
        lower_quantiles: List[float]=[0.05,0.25], 
        upper_quantiles: List[float]=[0.75,0.95],
        recurrent_norm: bool=False,
        shared_norm: bool=False, 
        task_specific_norm: bool=False, 
        loss_weights: list[float]=[1.0]*5,
        seed: int=1
    ):

    """
    DMQv0 + quantile spacing with separate heads for each quantile.
    """

    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    norm_fn = norm_fn.lower()
    if norm_fn == 'batch':
        norm_fn = BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")
    
    recurrent_layer_type = recurrent_layer_type.lower()
    if recurrent_layer_type == 'lstm':
        recurrent_layer_type = LSTM
    elif recurrent_layer_type == 'gru':   
        recurrent_layer_type = GRU
    else:
        raise ValueError("recurrent_layer_type must be 'lstm' or 'gru'")

    inputs = Input(shape=input_shape)

    shared_layers = []
    
    for i in range(1, n_recurrent_layers + 1):
        shared_layers.append(
            recurrent_layer_type(
                n_recurrent_nodes, 
                return_sequences=(i < n_recurrent_layers), 
                kernel_regularizer=L1L2(l1,l2), 
                recurrent_dropout=rec_drop,
                kernel_initializer=initializer
            )
        )
        if recurrent_norm:
            shared_layers.append(norm_fn())

    
    for i in range(1, n_shared_layers + 1):
        shared_layers.append(
            Dense(
                n_shared_nodes, 
                activation='relu', 
                kernel_regularizer=L1L2(l1,l2),
                kernel_initializer=initializer
            )
        )
        if shared_norm:
            shared_layers.append(norm_fn())

    # shared_layers.append(norm_fn())

    shared_net = Sequential(shared_layers, name='shared')(inputs)

    # Median head
    median_head = Sequential(name='Q50')
    for i in range(1, n_qtask_layers+1):
        median_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            median_head.add(norm_fn())
    
        if dropout > 0.0 and i < n_qtask_layers:
            median_head.add(Dropout(dropout))

    median_head.add(
        Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2))
    )

    median_output = median_head(shared_net)

    # Lower quantile heads
    fifth_resid_head = Sequential(name='Q5_lower_raw')
    for i in range(1, n_qtask_layers+1):
        fifth_resid_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            fifth_resid_head.add(norm_fn())
        
        if dropout > 0.0 and i < n_qtask_layers:
            fifth_resid_head.add(Dropout(dropout))

    fifth_resid_head.add(
        Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
    )
    fifth_raw = fifth_resid_head(shared_net)
    fifth_resid = Activation('softplus')(fifth_raw)

    twentyfifth_resid_head = Sequential(name='Q25_lower_raw')
    for i in range(1, n_qtask_layers+1):
        twentyfifth_resid_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            twentyfifth_resid_head.add(norm_fn())
        
        if dropout > 0.0 and i < n_qtask_layers:
            twentyfifth_resid_head.add(Dropout(dropout))

    twentyfifth_resid_head.add(
        Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
    )
    twentyfifth_raw = twentyfifth_resid_head(shared_net)
    twentyfifth_resid = Activation('softplus')(twentyfifth_raw)


    # Upper quantile heads
    ninetyfifth_resid_head = Sequential(name='Q95_upper_raw')
    for i in range(1, n_qtask_layers+1):
        ninetyfifth_resid_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            ninetyfifth_resid_head.add(norm_fn())

        if dropout > 0.0 and i < n_qtask_layers:
            ninetyfifth_resid_head.add(Dropout(dropout))

    ninetyfifth_resid_head.add(
        Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
    )
    ninetyfifth_raw = ninetyfifth_resid_head(shared_net)
    ninetyfifth_resid = Activation('softplus')(ninetyfifth_raw)

    # Upper quantile head
    seventyfifth_resid_head = Sequential(name='Q75_upper_raw')
    for i in range(1, n_qtask_layers+1):
        seventyfifth_resid_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            seventyfifth_resid_head.add(norm_fn())
        if dropout > 0.0 and i < n_qtask_layers:
            seventyfifth_resid_head.add(Dropout(dropout))

    seventyfifth_resid_head.add(
        Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
    )
    seventyfifth_raw = seventyfifth_resid_head(shared_net)
    seventyfifth_resid = Activation('softplus')(seventyfifth_raw)

    # Combine outputs
    Q50 = median_output
    Q5 = Q50 - fifth_resid
    Q25 = Q50 - twentyfifth_resid
    Q75 = Q50 + seventyfifth_resid
    Q95 = Q50 + ninetyfifth_resid

    out_concat = Concatenate()([Q5, Q25, Q50, Q75, Q95])

    model = Model(inputs=inputs, outputs=out_concat)

    loss = make_total_tilted_loss(lower_quantiles + [0.5] + upper_quantiles, q_loss_weights=loss_weights)

    model.compile(
        loss=loss, 
        optimizer=Adam(learning_rate=lr),
    )

    return model


def build_dmq_v1(
        input_shape: tuple, 
        n_recurrent_layers: int=2, 
        n_shared_layers: int=1, 
        n_qtask_layers: int=2, 
        n_recurrent_nodes: int=32,
        n_shared_nodes: int=32,
        n_task_nodes: int=32,
        l1: float=0.0, 
        l2: float=0.0, 
        rec_drop: float=0.0,
        dropout: float=0.0,
        lr: float=0.001, 
        norm_fn: str='layer', 
        recurrent_layer_type: str='gru', 
        lower_quantiles: List[float]=[0.05,0.25], 
        upper_quantiles: List[float]=[0.75,0.95],
        recurrent_norm: bool=False,
        shared_norm: bool=False, 
        task_specific_norm: bool=False, 
        loss_weights: list[float]=[1.0]*5,
        seed: int=1
    ):

    """
    DMQv0 + quantile spacing
    """

    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    norm_fn = norm_fn.lower()
    if norm_fn == 'batch':
        norm_fn = BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")
    
    recurrent_layer_type = recurrent_layer_type.lower()
    if recurrent_layer_type == 'lstm':
        recurrent_layer_type = LSTM
    elif recurrent_layer_type == 'gru':   
        recurrent_layer_type = GRU
    elif recurrent_layer_type == 'mygru':
        recurrent_layer_type = MyGRU
    else:
        raise ValueError("recurrent_layer_type must be 'lstm' or 'gru'")

    inputs = Input(shape=input_shape)

    shared_layers = []
    
    for i in range(1, n_recurrent_layers + 1):
        shared_layers.append(
            recurrent_layer_type(
                n_recurrent_nodes, 
                return_sequences=(i < n_recurrent_layers), 
                kernel_regularizer=L1L2(l1,l2), 
                recurrent_dropout=rec_drop,
                kernel_initializer=initializer
            )
        )
        if recurrent_norm:
            shared_layers.append(norm_fn())

    
    for i in range(1, n_shared_layers + 1):
        shared_layers.append(
            Dense(
                n_shared_nodes, 
                activation='relu', 
                kernel_regularizer=L1L2(l1,l2),
                kernel_initializer=initializer
            )
        )
        if shared_norm:
            shared_layers.append(norm_fn())

    # shared_layers.append(norm_fn())

    shared_net = Sequential(shared_layers, name='shared')(inputs)

    # Median head
    median_head = Sequential(name='Q50')
    for i in range(1, n_qtask_layers+1):
        median_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            median_head.add(norm_fn())
        if dropout > 0.0 and i < n_qtask_layers:
            median_head.add(Dropout(dropout))

    median_head.add(
        Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2))
    )

    median_output = median_head(shared_net)

    # Lower quantile head
    lower_resid_head = Sequential(name='Q_lower_raw')
    for i in range(1, n_qtask_layers+1):
        lower_resid_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            lower_resid_head.add(norm_fn())
        if dropout > 0.0 and i < n_qtask_layers:
            lower_resid_head.add(Dropout(dropout))

    lower_resid_head.add(
        Dense(len(lower_quantiles), activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
    )
    lower_raw = lower_resid_head(shared_net)
    lower_resid = Activation('softplus')(lower_raw)


    # Upper quantile head
    upper_resid_head = Sequential(name='Q_upper_raw')
    for i in range(1, n_qtask_layers+1):
        upper_resid_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_specific_norm and i < n_qtask_layers:
            upper_resid_head.add(norm_fn())
        if dropout > 0.0 and i < n_qtask_layers:    
            upper_resid_head.add(Dropout(dropout))
        
    upper_resid_head.add(
        Dense(len(upper_quantiles), activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
    )
    upper_raw = upper_resid_head(shared_net)
    upper_resid = Activation('softplus')(upper_raw)

    # Combine outputs
    Q50 = median_output
    Q_lower = Q50 - lower_resid
    Q_upper = Q50 + upper_resid

    out_concat = Concatenate()([Q_lower, Q50, Q_upper])

    model = Model(inputs=inputs, outputs=out_concat)

    loss = make_total_tilted_loss(lower_quantiles + [0.5] + upper_quantiles, q_loss_weights=loss_weights)

    model.compile(
        loss=loss, 
        optimizer=Adam(learning_rate=lr),
    )

    return model


def build_dmq_v0(
        input_shape: tuple, 
        n_recurrent_layers: int=2, 
        n_shared_layers: int=1, 
        n_qtask_layers: int=2, 
        n_recurrent_nodes: int=32,
        n_shared_nodes: int=32,
        n_task_nodes: int=32,
        l1: float=0.0, 
        l2: float=0.0, 
        lr: float=0.001, 
        rec_drop: float=0.0,
        dropout: float=0.0,
        norm_fn: str='layer', 
        recurrent_layer_type: str='gru', 
        quantiles: list[int]=[0.05,0.25,0.50,0.75,0.95], 
        recurrent_norm: bool=False,
        shared_norm: bool=False,
        task_specific_norm: bool=False, 
        loss_weights: list[float]=[1.0]*5,
        seed: int=1
    ):

    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    norm_fn = norm_fn.lower()
    if norm_fn == 'batch':
        norm_fn = BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")
    
    recurrent_layer_type = recurrent_layer_type.lower()
    if recurrent_layer_type == 'lstm':
        recurrent_layer_type = LSTM
    elif recurrent_layer_type == 'gru':   
        recurrent_layer_type = GRU
    else:
        raise ValueError("recurrent_layer_type must be 'lstm' or 'gru'")

    inputs = Input(shape=input_shape)

    shared_layers = []
    
    for i in range(1, n_recurrent_layers + 1):
        shared_layers.append(
            recurrent_layer_type(
                n_recurrent_nodes, 
                return_sequences=(i < n_recurrent_layers), 
                kernel_regularizer=L1L2(l1,l2), 
                recurrent_dropout=rec_drop,
                kernel_initializer=initializer
            )
        )
        if recurrent_norm:
            shared_layers.append(norm_fn())

    for i in range(1, n_shared_layers + 1):
        shared_layers.append(
            Dense(
                n_shared_nodes, 
                activation='relu', 
                kernel_regularizer=L1L2(l1,l2),
                kernel_initializer=initializer
            )
        )
        if shared_norm:
            shared_layers.append(norm_fn())

    shared_net = Sequential(shared_layers, name='shared')(inputs)

    outputs = []
    for q in quantiles:
        name = f"Q{q}"
        
        qtask_layers = []
        for i in range(1, n_qtask_layers+1):
            qtask_layers.append(
                Dense(
                    n_task_nodes, 
                    activation='relu',
                    kernel_regularizer=L1L2(l1, l2),
                    kernel_initializer=initializer
                )
            )
            if task_specific_norm and i < n_qtask_layers:
                qtask_layers.append(norm_fn())
            if dropout > 0.0 and i < n_qtask_layers:
                qtask_layers.append(Dropout(dropout))

        # Append output node
        qtask_layers.append(
            Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
        )
        
        # Build output net
        output_q = Sequential(qtask_layers, name=name)(shared_net)

        outputs.append(output_q)

    out_concat = Concatenate(name='out_layer')(outputs)

    model = Model(inputs=inputs, outputs=out_concat)

    loss = make_total_tilted_loss(quantiles, q_loss_weights=loss_weights)

    model.compile(
        loss=loss, 
        optimizer=Adam(learning_rate=lr),
    )

    return model

def train_estimator(estimator, X, y, early_stopping_args, fit_kwargs):

    es = EarlyStopping(**early_stopping_args)
    fit_kwargs.update({'callbacks': [es]})
    estimator.fit(
        X, y,
        **fit_kwargs
    )
    return estimator

class DMQEnsemble:

    def __init__(self, builder_func, n_models: int, **dmq_kwargs):
        self.n_models = n_models
        self.builder_func = builder_func
        self.dmq_kwargs = dmq_kwargs
        self.models = [
            self.builder_func(**self.dmq_kwargs) for _ in range(n_models)
        ]

    def fit(self,
            X, y,
            early_stopping_args: dict | None = None,
            n_jobs: int = 1,
            fit_kwargs: dict | None = None
    ):
        """
        Fit all ensemble members. If n_jobs > 1, runs workers in parallel;
        each worker builds/fits/saves its own model file (if model_path provided).
        """

        # joblib Parallel expects a sequence of tasks
        tasks = (
            delayed(train_estimator)(
                model,
                X,
                y,
                early_stopping_args,
                fit_kwargs
            )
            for model in self.models
        )
        

        if n_jobs == 1:
            # run sequentially (keeps training in main process)
            trained_models = [fn() if callable(fn) else fn for fn in [t for t in tasks]]
            # note: above is a fallback; for clarity we run _train_or_load_estimator directly
            trained_models = [
                train_estimator(
                    model,
                    X,
                    y,
                    early_stopping_args,
                    fit_kwargs
                ) for model in self.models
            ]
        else:
            trained_models = Parallel(n_jobs=n_jobs)(tasks)

        # assign trained models into self.models
        for i, m in enumerate(trained_models):
            self.models[i] = m

        return self.models

    def predict(self, X):
        preds = [model.predict(X)[:,:,np.newaxis] for model in self.models]
        return np.mean(np.concatenate(preds, axis=2), axis=2)   

# class MemoryPureLSTM(RNN):

#     def __init__(self, *args, g_recurrent_dropout: float = 0.0, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.g_recurrent_dropout = float(g_recurrent_dropout)
#         # keep existing masks behavior if built-in recurrent_dropout used
#         if self.g_recurrent_dropout != 0.0:
#             # force implementation that allows mask usage (keeps compatibility)
#             self.implementation = 1

#     def call(self, inputs, states, training=False):
#         # copy of LSTMCell.call but only modify how 'g' is dropped
#         h_tm1 = states[0]
#         c_tm1 = states[1]

#         if self.implementation == 1:
#             if training and 0.0 < self.dropout < 1.0:
#                 dp_mask = self.get_dropout_mask(inputs)
#                 inputs_i = inputs * dp_mask[0]
#                 inputs_f = inputs * dp_mask[1]
#                 inputs_c = inputs * dp_mask[2]
#                 inputs_o = inputs * dp_mask[3]
#             else:
#                 inputs_i = inputs_f = inputs_c = inputs_o = inputs

#             k_i, k_f, k_c, k_o = ops.split(self.kernel, 4, axis=1)
#             x_i = ops.matmul(inputs_i, k_i)
#             x_f = ops.matmul(inputs_f, k_f)
#             x_c = ops.matmul(inputs_c, k_c)
#             x_o = ops.matmul(inputs_o, k_o)
#             if self.use_bias:
#                 b_i, b_f, b_c, b_o = ops.split(self.bias, 4, axis=0)
#                 x_i += b_i; x_f += b_f; x_c += b_c; x_o += b_o

#             if training and 0.0 < self.recurrent_dropout < 1.0:
#                 rec_dp_mask = self.get_recurrent_dropout_mask(h_tm1)
#                 h_tm1_i = h_tm1 * rec_dp_mask[0]
#                 h_tm1_f = h_tm1 * rec_dp_mask[1]
#                 h_tm1_c = h_tm1 * rec_dp_mask[2]
#                 h_tm1_o = h_tm1 * rec_dp_mask[3]
#             else:
#                 h_tm1_i = h_tm1_f = h_tm1_c = h_tm1_o = h_tm1

#             # compute gates manually so we can apply dropout to g
#             i = self.recurrent_activation(
#                 x_i + ops.matmul(h_tm1_i, self.recurrent_kernel[:, : self.units])
#             )
#             f = self.recurrent_activation(
#                 x_f + ops.matmul(h_tm1_f, self.recurrent_kernel[:, self.units : self.units * 2])
#             )
#             # compute raw candidate
#             raw_g = self.activation(
#                 x_c + ops.matmul(h_tm1_c, self.recurrent_kernel[:, self.units * 2 : self.units * 3])
#             )
#             # g = i * raw_g  -> apply dropout to g (only during training)
#             g = i * raw_g
#             if training and 0.0 < self.g_recurrent_dropout < 1.0:
#                 # tf.nn.dropout expects rate; scales during training, leaves unchanged in inference
#                 g = tf.nn.dropout(g, rate=self.g_recurrent_dropout, seed=self.seed)
#             c = f * c_tm1 + g

#             o = self.recurrent_activation(
#                 x_o + ops.matmul(h_tm1_o, self.recurrent_kernel[:, self.units * 3 :])
#             )

#         else:
#             if training and 0.0 < self.dropout < 1.0:
#                 dp_mask = self.get_dropout_mask(inputs)
#                 inputs = inputs * dp_mask
#             z = ops.matmul(inputs, self.kernel)
#             z = ops.add(z, ops.matmul(h_tm1, self.recurrent_kernel))
#             if self.use_bias:
#                 z = ops.add(z, self.bias)
#             z = ops.split(z, 4, axis=1)

#             # fused path: z0,z1,z2,z3 correspond to i,f,raw_c,o before activations
#             z0, z1, z2, z3 = z
#             i = self.recurrent_activation(z0)
#             f = self.recurrent_activation(z1)
#             raw_g = self.activation(z2)
#             g = i * raw_g
#             if training and 0.0 < self.g_recurrent_dropout < 1.0:
#                 g = tf.nn.dropout(g, rate=self.g_recurrent_dropout, seed=self.seed)
#             c = f * c_tm1 + g
#             o = self.recurrent_activation(z3)

#         h = o * self.activation(c)
#         return h, [h, c]

def compute_quantile_subgradient(u: np.ndarray, q: float) -> float:
    """Check function for quantile regression."""
    return (q - (u < 0).astype(float))

def quantile_loss(u: np.ndarray, q: float) -> float:
    """Quantile loss function."""
    return u * compute_quantile_subgradient(u, q)

def compute_qpc(y: np.ndarray, X_s: np.ndarray , X_j: np.ndarray, q: float) -> float:

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=IterationLimitWarning)
        # add intercept if missing
        X_s_const = add_constant(X_s, has_constant='skip')
        reg_s = QuantReg(y, X_s_const).fit(q=q)
        resid_s = y - reg_s.predict(X_s_const)

        X_j_const = add_constant(X_j, has_constant='skip')
        reg_j = QuantReg(y, X_j_const).fit(q=q)
        resid_j = y - reg_j.predict(X_j_const)

    var_j = np.var(resid_j)

    qpc = np.mean(compute_quantile_subgradient(resid_s, q) * resid_j )/ np.sqrt(q*(1-q)*var_j)

    return qpc


def get_confounding_set(X: np.ndarray, m: int, j: int) -> list:

    """
    Generates the confounding set for variable j based on the m largest correlations with other variables.

    Parameters:
    - X: 2D numpy array of shape (n_samples, n_features)
    - m: Number of variables to include in the confounding set
    - j: Index of the variable for which the confounding set is to be generated

    Returns:
    - confounding_set: List of indices representing the confounding set for variable j
    """

    idx_no_j = list(range(X.shape[1]))
    idx_no_j.remove(j)

    # Get all correlations
    corrs = np.abs(np.corrcoef(X, rowvar=False)[j, idx_no_j])

    # Take variables whose correlation with j is above the mth largest correlation
    mth_largest_corr_idx = np.argsort(corrs)[-m:]
    confounding_set = [idx_no_j[i] for i in mth_largest_corr_idx]

    return confounding_set


def fit_qpcr(X: np.ndarray, y: np.ndarray, q: float, n_updates: int=None, max_predictors: int=None, size_of_confounding_set: int=None, ebic_const: int=1):

    """
    Fits a quantile partial correlation regression model using the specified parameters.

    Parameters:
    - X: 2D numpy array of shape (n_samples, n_features) representing the predictor variables
    - y: 1D numpy array of shape (n_samples,) representing the response variable
    - q: Quantile to be estimated (e.g., 0.05, 0.5, 0.95).
    - n_updates: Number of updates to the confounding set to perform (default is [sqrt(T/log(T))])
    - max_predictors: Maximum number of predictors to include in the model (default is [T/log(T)])
    - size_of_confounding_set: Size of the confounding set to consider for each variable (default is [sqrt(T/log(T))])
    - ebic_const: Constant for the EBIC criterion (default is 1)

    Returns:
    - model: Fitted QPCR model
    """

    T = X.shape[0]
    if n_updates is None:
        n_updates = int((T / np.log(T))**(1/2))
    if size_of_confounding_set is None:
        size_of_confounding_set = int((T / np.log(T))**(1/2))
    if max_predictors is None:
        max_predictors = int(T/np.log(T))

    active_sets = [[]]
    while len(active_sets[-1]) < n_updates:

        # Set active set
        active_set = active_sets[-1]

        X_candidates_idx = [i for i in range(X.shape[1]) if i not in active_set]

        # For each candidate, update conditioning set
        def get_all_qpcs(i):
            # Update conditional set
            confounding_set = get_confounding_set(X, m=size_of_confounding_set, j=i)
            conditional_set = active_set + confounding_set
            # Compute qpc
            qpc = compute_qpc(y, X[:, conditional_set], X[:, i].reshape(-1, 1), q)
            return qpc

        all_qpcs = Parallel(n_jobs=-1)(delayed(get_all_qpcs)(i) for i in X_candidates_idx)

        # Select covariate index
        selected_idx = X_candidates_idx[np.argmax(np.abs(all_qpcs))]

        updated_active_set = active_sets[-1] + [selected_idx]
        active_sets.append(updated_active_set)

        # print(f"Updated active set: {updated_active_set}")

    active_set_dstar = active_sets[-1]

    while len(active_sets[-1]) < max_predictors:

        # Select set to search over
        X_candidates_idx = [i for i in range(X.shape[1]) if i not in active_sets[-1]]

        def get_all_qpcs(i):
            # Update conditional set
            confounding_set = get_confounding_set(X, m=size_of_confounding_set, j=i)
            conditional_set = active_set_dstar + confounding_set
            # Compute qpc
            qpc = compute_qpc(y, X[:, conditional_set], X[:, i].reshape(-1, 1), q)
            return qpc

        # Select covariate index
        all_qpcs = Parallel(n_jobs=-1)(delayed(get_all_qpcs)(i) for i in X_candidates_idx)
        selected_idx = X_candidates_idx[np.argmax(np.abs(all_qpcs))]
        updated_active_set = active_sets[-1] + [selected_idx]
        active_sets.append(updated_active_set)
        # print(f"Updated active set: {updated_active_set}")
    
    losses = []
    active_sets = active_sets[1:]  # Remove empty set
    for model_candidate in active_sets:

        # Fit qreg 
        X_active = X[:, model_candidate]
        X_active_const = add_constant(X_active, has_constant='skip')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=IterationLimitWarning)
            reg = QuantReg(y, X_active_const).fit(q=q)

        # Compute loss 
        loss = np.log(quantile_loss(y - reg.predict(X_active_const), q).mean()) + ebic_const * (np.log(X.shape[0]) * np.log(len(model_candidate))) / X.shape[0]
        losses.append(loss)

    # Fit best model
    best_model_idx = np.argmin(losses)
    X_active = X[:, active_sets[best_model_idx]]
    X_active_const = add_constant(X_active, has_constant='skip')
    reg = QuantReg(y, X_active_const).fit(q=q)

    return reg, active_sets[best_model_idx]

def get_early_stopping(**kwargs):
    return tf.keras.callbacks.EarlyStopping(**kwargs)


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



def load_tuning_log(log_path):
    if not os.path.exists(log_path):
        return {}
    try:
        with open(log_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def study_exists(study_name, log):
    return study_name in log


def check_hps_exist(model_name: str, log_path: str) -> bool:
    """
    Checks if hyperparameters for a model already exist in the log file.

    Parameters
    ----------
    model_name: str
        Name of the model, e.g., "model_q5".
    log_path: str
        Path to the log file, e.g., "tuning_log.txt".

    Returns
    -------
    bool:
        True if hyperparameters exist, False otherwise.
    """
    
    if not os.path.exists(log_path):
        return False
    
    try:
        with open(log_path, 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            log = json.load(f)
            portalocker.unlock(f)
        return model_name in log
    
    except Exception:
        return False


def save_hyperparameters(hps: dict, model_name: str, log_path: str, overwrite: bool=False):
    """
    Saves hyperparameters to a log file.

    Parameters
    ----------
    hps: dict
        Hyperparameters to save.
    model_name: str
        Name of the model, e.g., "model_q5".
    log_path: str
        Path to the log file, e.g., "tuning_log.txt".
    """
    # Check if log exists, create log file if not
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump({}, f)
            portalocker.unlock(f)

    # Load log and update
    with open(log_path, 'r+') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        log = json.load(f)

        # If model not in log, always save
        if model_name not in log:
            print('Saving model hyperparameters (first entry)...')
            log[model_name] = hps
            f.seek(0)
            json.dump(log, f)
            f.truncate()
        else:
            if overwrite:
                old_value = log[model_name].get('value', 0.0)
                if hps.get('value', 0.0) > old_value:
                    print('Saving model hyperparameters (better value)...')
                    log[model_name] = hps
                    f.seek(0)
                    json.dump(log, f)
                    f.truncate()
        portalocker.unlock(f)
    return None


def load_hyperparameters(model_name: str, log_path: str) -> dict:
    """
    Loads hyperparameters from a log file.

    Parameters
    ----------
    model_name: str
        Name of the model, e.g., "model_q5".
    log_path: str
        Path to the log file, e.g., "tuning_log.txt".

    Returns
    -------
    dict:
        Hyperparameters for the specified model.
    """
    import portalocker, json

    with open(log_path, 'r') as f:
        portalocker.lock(f, portalocker.LOCK_SH)
        log = json.load(f)
        portalocker.unlock(f)

    return log.get(model_name, {})


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
            model = load_model(model_path, custom_objects)

        return model

    estimators = Parallel(delayed(train_estimator)(hps, f"{dir_path}{model_name}_estimator{i}.keras") for i in range(n_estimators))
    
    return estimators

def compute_oos_r1_score(benchmark_pred, y_true, y_pred, q):
    
    """
    Computes the R1 score of a set of quantile forecasts and a set of returns.
    """

    return (1 - mean_pinball_loss(y_true, y_pred, alpha=q)/mean_pinball_loss(y_true, benchmark_pred, alpha=q))*100

def compute_oos_r2_score(y_true, y_pred, benchmark):
    
    """
    Computes out-of-sample (OOS) R2.
    """
    
    return (1 - mean_squared_error(y_true, y_pred) / mean_squared_error(y_true, benchmark))*100

def estimate_mean_from_quantiles(preds, weights: List[float]=[0.15, 0.225, 0.25, 0.225, 0.15]):
    return preds @ np.array(weights).reshape(-1,1)


def evaluate_model(y_pred, y_true, benchmark, target, model_name, quantiles, suppress_recession_dates: bool=False):
    import matplotlib.pyplot as plt
    r2_scores = {}

    target_preds = y_pred
    mean_preds = estimate_mean_from_quantiles(target_preds)

    r2 = compute_oos_r2_score(y_true, mean_preds.flatten(), benchmark[target]['Expanding_Mean'].values.flatten())

    plt.plot(y_true.index, mean_preds, label=f'Pred - R2={r2:.0f}%')
    plt.plot(y_true.index, y_true, label='Actual')
    plt.plot(y_true.index, benchmark[target]['Expanding_Mean'].values.flatten(), label='Naive')
    if not suppress_recession_dates:
        plt.axvspan('2001-03-01', '2001-11-01', -1,1, color='grey', alpha=0.25)
        plt.axvspan('2007-12-01', '2009-06-01', -1,1, color='grey', alpha=0.25)
        plt.axvspan('2020-02-01', '2020-04-01', -1,1, color='grey', alpha=0.25)
    plt.legend()
    plt.show()

    quantile_performance = {}
    for q in quantiles:
        q_preds = y_pred[:,quantiles.index(q)].flatten()
        q_r1 = compute_oos_r1_score(benchmark[target][f'Expanding_Q{int(q*100)}'].values.flatten(), y_true, q_preds, q)
        quantile_performance[f'Quantile {q}'] = round(q_r1,1)
        plt.plot(y_true.index, q_preds, linestyle='--', label=f'Quantile {q} -- R1={q_r1:.2f}%')
        # plt.plot(benchmark, linestyle=':', label=f'Benchmark Q{int(q*100)}')
    plt.plot(y_true.index, y_true, color='black', label='Actual')
    # plt.axvspan('1969-12-01', '1970-11-01', -1,1, color='grey', alpha=0.25)
    # plt.axvspan('1973-11-01', '1975-03-01', -1,1, color='grey', alpha=0.25)
    # plt.axvspan('1980-01-01', '1980-07-01', -1,1, color='grey', alpha=0.25)
    # plt.axvspan('1981-07-01', '1982-11-01', -1,1, color='grey', alpha=0.25)
    # plt.axvspan('1990-07-01', '1991-03-01', -1,1, color='grey', alpha=0.25)
    plt.axvspan('2001-03-01', '2001-11-01', -1,1, color='grey', alpha=0.25)
    plt.axvspan('2007-12-01', '2009-06-01', -1,1, color='grey', alpha=0.25)
    plt.axvspan('2020-02-01', '2020-04-01', -1,1, color='grey', alpha=0.25)
    plt.legend()
    plt.savefig(f'{model_name}_quantile_predictions_{target}.png')
    plt.show()
    quantile_performance.update({'Mean R1': round(np.mean(list(quantile_performance.values())),2), 'R2': round(r2,2)})
    print(quantile_performance)

    coverages = compute_quantile_coverage(y_true.values, y_pred, quantiles)

    for q, c in coverages.items():
        print(f"Quantile {q:.2f}: empirical coverage = {100*c:.2f}% (target {100*q:.2f}%)")

    plot_coverage(coverages)

    return quantile_performance

def compute_quantile_coverage(y_true, y_pred, quantiles):
    """
    Computes empirical coverage for each quantile.

    Parameters
    ----------
    y_true : np.ndarray, shape (T,)
        Actual values.
    y_pred : np.ndarray, shape (T, Q)
        Predicted quantiles (each column corresponds to a quantile).
    quantiles : list or np.ndarray, shape (Q,)
        Quantile levels, e.g. [0.05, 0.25, 0.5, 0.75, 0.95]

    Returns
    -------
    dict : {quantile: coverage}
        Dictionary mapping quantiles to their empirical coverage.
    """
    coverages = {}
    for i, q in enumerate(quantiles):
        coverage = np.mean(y_true <= y_pred[:, i])
        coverages[q] = coverage
    return coverages


def plot_coverage(coverages):
    import matplotlib.pyplot as plt
    qs = np.array(list(coverages.keys()))
    cs = np.array(list(coverages.values()))
    
    plt.figure(figsize=(5, 5))
    plt.plot(qs, cs, 'o-', label='Empirical coverage')
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal 45° line')
    plt.xlabel("Nominal quantile (target)")
    plt.ylabel("Empirical coverage")
    plt.title("Quantile calibration plot")
    plt.legend()
    plt.grid(True)
    plt.show()


def qskt(q, loc, scale, shape, df):
    """
    Quantile function for the skewed t-distribution.
    
    Parameters:
        q: Probability values (between 0 and 1)
        loc: Location parameter
        scale: Scale parameter
        shape: Shape parameter (skewness)
        df: Degrees of freedom
    
    Returns:
        Quantiles of the skewed t-distribution
    """
    from scipy.stats import t
    import numpy as np
    
    # Formula for Azzalini-Capitanio skewed t-distribution quantiles
    z = t.ppf(q, df)
    a = shape * z * np.sqrt((df+1)/(df+z**2))
    
    # Handle numeric issues safely
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        z2 = z * z
        inv = np.where(z2 == 0.0, np.inf, df / z2)
        ratio = (df + 1.0) / (1.0 + inv)
        ratio = np.maximum(ratio, 0.0)
        a = shape * np.sign(z) * np.sqrt(ratio)
    
    # Calculate skewed-t quantile (location-scale form)
    delta = a / np.sqrt(1 + a**2)
    return loc + scale * z * (1 + delta) / np.sqrt(1 - delta**2)

def quantiles_interpolation(qq_targ, QQ, lc0=None, sc0=None, sh0=None):
    """
    Fits a skewed t-distribution to match target quantiles.
    
    Parameters:
        qq_targ: Vector containing the target quantiles
        QQ: Vector of quantile levels (should include 0.05, 0.25, 0.50, 0.75, 0.95)
        lc0: Initial condition for location parameter
        sc0: Initial condition for scale parameter
        sh0: Initial condition for shape parameter
    
    Returns:
        lc: Fitted location parameter
        sc: Fitted scale parameter
        sh: Fitted shape parameter
        df: Fitted degrees of freedom parameter
    """

    import numpy as np
    from scipy import optimize
    from scipy.stats import norm

    # Set bounds for optimization
    LB = [-20, 1e-6, -30]  # lower bounds: location, scale, shape
    UB = [20, 50, 30]      # upper bounds
    
    # Find indices of target quantiles
    jq50 = np.argmin(np.abs(QQ - 0.50))
    jq25 = np.argmin(np.abs(QQ - 0.25))
    jq75 = np.argmin(np.abs(QQ - 0.75))
    jq05 = np.argmin(np.abs(QQ - 0.05))
    jq95 = np.argmin(np.abs(QQ - 0.95))
    
    # Set initial conditions if not provided
    if lc0 is None or sc0 is None or sh0 is None:
        iqn = norm.ppf(0.75) - norm.ppf(0.25)
        lc0 = qq_targ[jq50]
        sc0 = (qq_targ[jq75] - qq_targ[jq25]) / iqn
        sh0 = 0
    
    X0 = [lc0, sc0, sh0]
    
    # Select target quantiles
    select = [jq05, jq25, jq75, jq95]
    QQ_select = QQ[select]
    qq_targ_select = qq_targ[select]
    
    # Optimize for each possible value of degrees of freedom
    par = np.full((30, 3), np.nan)
    ssq = np.full(30, np.nan)
    
    for df in range(1, 31):
        def objective(x):
            return qq_targ_select - qskt(QQ_select, x[0], x[1], x[2], df)
        
        result = optimize.least_squares(
            objective, X0, bounds=(LB, UB), 
            method='trf', ftol=1e-6, xtol=1e-6
        )
        
        par[df-1, :] = result.x
        ssq[df-1] = np.sum(result.fun**2)
    
    # Find best fit
    best_df_idx = np.argmin(ssq)
    df = best_df_idx + 1  # df ranges from 1 to 30
    lc = par[best_df_idx, 0]
    sc = par[best_df_idx, 1]
    sh = par[best_df_idx, 2]
    
    return lc, sc, sh, df

class SkewedTDistribution:
    def __init__(self, loc, scale, shape, df):
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.df = df

    

    def pdf(self, x):
        """Probability density function"""
        
        z = (x - self.loc) / self.scale

        # Stable computation of a
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            z2 = z * z
            inv = np.where(z2 == 0.0, np.inf, self.df / z2)
            ratio = (self.df + 1.0) / (1.0 + inv)
            ratio = np.maximum(ratio, 0.0)
            a = self.shape * np.sign(z) * np.sqrt(ratio)

        # Clip a to prevent overflow
        a = np.clip(a, -1e6, 1e6)

        pdf_z = t.pdf(z, self.df)
        cdf_a = t.cdf(a, self.df + 1)
        return 2.0 / self.scale * pdf_z * cdf_a

    def cdf(self, x):
        """Cumulative distribution function"""
        
        # For more accurate results with extreme values, we integrate the PDF
        # For typical cases, we use a direct formula
        
        if np.isscalar(x):
            if x < self.loc - 10*self.scale:  # Far in left tail
                return 0.0
            if x > self.loc + 10*self.scale:  # Far in right tail
                return 1.0
                
            # Numerical integration for accuracy
            result, _ = integrate.quad(self.pdf, -np.inf, x)
            return result
        else:
            # Vectorized version
            result = np.zeros_like(x, dtype=float)
            
            # Far left and right tails
            result[x < self.loc - 10*self.scale] = 0.0
            result[x > self.loc + 10*self.scale] = 1.0

            # Middle range - integrate each point
            middle = (x >= self.loc - 10*self.scale) & (x <= self.loc + 10*self.scale)
            for i, xi in enumerate(x[middle]):
                result[middle][i], _ = integrate.quad(self.pdf, -np.inf, xi)

            return result

    def ppf(self, q):
        """Percent point function (inverse CDF)"""
        # For a scalar, use binary search
        if np.isscalar(q):
            if q <= 0:
                return -np.inf
            if q >= 1:
                return np.inf
                
            # Binary search
            left = self.loc - 10*self.scale
            right = self.loc + 10*self.scale

            for _ in range(50):  # Usually converges in < 50 iterations
                mid = (left + right) / 2
                if self.cdf(mid) < q:
                    left = mid
                else:
                    right = mid
                    
                if right - left < 1e-10:
                    break
                    
            return (left + right) / 2
        else:
            # Vectorized version
            return np.array([self.ppf(qi) for qi in q])

    def plot(self, x_range=None, fig=None, ax=None, plot_type='pdf', **kwargs):
        """
        Plot the distribution's PDF or CDF.
        
        Parameters:
            x_range: Optional range for x-values as [min, max]
            fig, ax: Optional matplotlib figure and axes
            plot_type: 'pdf' or 'cdf'
            **kwargs: Additional arguments for matplotlib plot function
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        if x_range is None:
            # Determine reasonable plotting range
            if self.shape < 0:  # Left skewed
                x_range = [self.loc - 3*self.scale, self.loc + 1.5*self.scale]
            elif self.shape > 0:  # Right skewed
                x_range = [self.loc - 1.5*self.scale, self.loc + 3*self.scale]
            else:  # Symmetric
                x_range = [self.loc - 3*self.scale, self.loc + 3*self.scale]

        x = np.linspace(x_range[0], x_range[1], 1000)
        
        if plot_type.lower() == 'pdf':
            y = self.pdf(x)
            ax.plot(x, y, **kwargs)
            ax.set_title(f'Skewed t-distribution PDF (loc={self.loc}, scale={self.scale}, shape={self.shape}, df={self.df})')
            ax.set_ylabel('Probability Density')
        elif plot_type.lower() == 'cdf':
            y = self.cdf(x)
            ax.plot(x, y, **kwargs)
            ax.set_title(f'Skewed t-distribution CDF (loc={self.loc}, scale={self.scale}, shape={self.shape}, df={self.df})')
            ax.set_ylabel('Cumulative Probability')
        
        ax.set_xlabel('x')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    


def create_skewed_t_distribution(loc, scale, shape, df):
    """
    Creates a skewed t-distribution with specified parameters.
    
    Parameters:
        loc (float): Location parameter
        scale (float): Scale parameter (must be positive)
        shape (float): Shape parameter (skewness)
        df (float): Degrees of freedom (must be positive)
    
    Returns:
        A dictionary with methods pdf, cdf, ppf, and plot
    """
    import numpy as np
    from scipy.stats import t
    import matplotlib.pyplot as plt
    from scipy import integrate
    
    if scale <= 0:
        raise ValueError("Scale parameter must be positive")
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")
        
    def pdf(x):
        """Probability density function"""
        z = (x - loc) / scale
        
        # Stable computation of a
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            z2 = z * z
            inv = np.where(z2 == 0.0, np.inf, df / z2)
            ratio = (df + 1.0) / (1.0 + inv)
            ratio = np.maximum(ratio, 0.0)
            a = shape * np.sign(z) * np.sqrt(ratio)
        
        # Clip a to prevent overflow
        a = np.clip(a, -1e6, 1e6)
        
        pdf_z = t.pdf(z, df)
        cdf_a = t.cdf(a, df + 1)
        return 2.0 / scale * pdf_z * cdf_a
    
    def cdf(x):
        """Cumulative distribution function"""
        # For more accurate results with extreme values, we integrate the PDF
        # For typical cases, we use a direct formula
        
        if np.isscalar(x):
            if x < loc - 10*scale:  # Far in left tail
                return 0.0
            if x > loc + 10*scale:  # Far in right tail
                return 1.0
                
            # Numerical integration for accuracy
            result, _ = integrate.quad(pdf, -np.inf, x)
            return result
        else:
            # Vectorized version
            result = np.zeros_like(x, dtype=float)
            
            # Far left and right tails
            result[x < loc - 10*scale] = 0.0
            result[x > loc + 10*scale] = 1.0
            
            # Middle range - integrate each point
            middle = (x >= loc - 10*scale) & (x <= loc + 10*scale)
            for i, xi in enumerate(x[middle]):
                result[middle][i], _ = integrate.quad(pdf, -np.inf, xi)
                
            return result
    
    def ppf(q):
        """Percent point function (inverse CDF)"""
        # For a scalar, use binary search
        if np.isscalar(q):
            if q <= 0:
                return -np.inf
            if q >= 1:
                return np.inf
                
            # Binary search
            left = loc - 10*scale
            right = loc + 10*scale
            
            for _ in range(50):  # Usually converges in < 50 iterations
                mid = (left + right) / 2
                if cdf(mid) < q:
                    left = mid
                else:
                    right = mid
                    
                if right - left < 1e-10:
                    break
                    
            return (left + right) / 2
        else:
            # Vectorized version
            return np.array([ppf(qi) for qi in q])
    
    def plot(x_range=None, fig=None, ax=None, plot_type='pdf', **kwargs):
        """
        Plot the distribution's PDF or CDF.
        
        Parameters:
            x_range: Optional range for x-values as [min, max]
            fig, ax: Optional matplotlib figure and axes
            plot_type: 'pdf' or 'cdf'
            **kwargs: Additional arguments for matplotlib plot function
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        if x_range is None:
            # Determine reasonable plotting range
            if shape < 0:  # Left skewed
                x_range = [loc - 3*scale, loc + 1.5*scale]
            elif shape > 0:  # Right skewed
                x_range = [loc - 1.5*scale, loc + 3*scale]
            else:  # Symmetric
                x_range = [loc - 3*scale, loc + 3*scale]
                
        x = np.linspace(x_range[0], x_range[1], 1000)
        
        if plot_type.lower() == 'pdf':
            y = pdf(x)
            ax.plot(x, y, **kwargs)
            ax.set_title(f'Skewed t-distribution PDF (loc={loc}, scale={scale}, shape={shape}, df={df})')
            ax.set_ylabel('Probability Density')
        elif plot_type.lower() == 'cdf':
            y = cdf(x)
            ax.plot(x, y, **kwargs)
            ax.set_title(f'Skewed t-distribution CDF (loc={loc}, scale={scale}, shape={shape}, df={df})')
            ax.set_ylabel('Cumulative Probability')
        
        ax.set_xlabel('x')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    # Return dictionary with all methods
    return {
        'pdf': pdf,
        'cdf': cdf,
        'ppf': ppf,
        'plot': plot,
        'params': {'loc': loc, 'scale': scale, 'shape': shape, 'df': df}
    }

