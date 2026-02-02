import numpy as np
from typing import Tuple, Union
import pandas as pd

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
