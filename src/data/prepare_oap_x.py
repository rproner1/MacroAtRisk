
import pandas as pd
from dateutil.relativedelta import relativedelta
from pathlib import Path
from statsmodels.tsa.stattools import adfuller

from src.data.data_utils import remove_cols


"""
Variables that do not aggregate to a economically meaningful variable:
- Decline in analyst coverage (ChNAnalyst)
So the filter is as follows: 1) If non-stationary first difference. If first-difference is non-stationary exclude. 2) Remove variables with insufficient time series variation |mean|/sd < 0.06. 3) Remove variables that do not aggregate to something that is economically meaningful. 

The third part is a bit tricky. There are eight variables excluded according to this criteria: (i) Analyst Coverage; (ii) Forecast Dispersion Change; (iii) Idiosyncratic Risk, Distant; (iv) Number of Analysts, Change; (v) Book to Assets, Change; (vi) Misvalued Innovation; (vii) Sales Growth; and (vii) Agnostic Value  The OAP variables  

"""

def stationarity_filter(df: pd.DataFrame, alpha=0.05) -> pd.DataFrame:
    """
    Tests each variable for stationarity using the Augmented Dickey-Fuller test. If a variable is non-stationary, it is first-differenced and tested again. If it remains non-stationary after first-differencing, it is excluded from the dataset.

    Parameters: 
        df (pd.DataFrame): 
            DataFrame containing the variables to be tested for stationarity.
        alpha: float, optional
            Significance level for the Augmented Dickey-Fuller test (default is 0.05).
    Returns:
        pd.DataFrame: 
            DataFrame with non-stationary variables removed.
    
    """
    stationary_cols = []
    for col in df.columns:
        result = adfuller(df[col].dropna())
        p_value = result[1]
        if p_value < alpha:
            stationary_cols.append(col)
        else: 
            result_diff = adfuller(df[col].diff().dropna())
            p_val_diff = result_diff[1]
            if p_val_diff < alpha:
                col_diff = col + '_diff'
                df[col_diff] = df[col].diff()
                stationary_cols.append(col_diff)

    return df[stationary_cols]

def get_firm_level_x(
        file_path: Path, 
        first_difference: bool,
        horizon_in_quarters: int,
        desired_start_date_of_samples: pd.Timestamp,
        last_date_of_sample: pd.Timestamp,
        remove_cols_threshold: float,
        exclude_non_stationary: bool,
        alpha: float,
        initial_training_last_date: pd.Timestamp
    ) -> pd.DataFrame:
    """
    Prepares firm-level characteristics data from Open Asset Pricing (OAP) for modeling.

    Returns:
        pd.DataFrame: Prepared firm-level characteristics data ready for modeling.
    """
    df = pd.read_csv(file_path)
    df['yyyymm'] = pd.to_datetime(df['yyyymm'])
    df.rename(columns={'yyyymm': 'date'}, inplace=True)
    df.set_index('date', inplace=True)

    # Subset data to avoid removing too many columns due to initial NaNs
    buffer = int(first_difference) 
    df = df.loc[desired_start_date_of_samples - relativedelta(months=3*horizon_in_quarters + buffer): last_date_of_sample]

    # First difference if specified
    if first_difference:
        df = df.diff()

    # Remove columns with too many missing values
    df = remove_cols(remove_cols_threshold, df, initial_training_last_date)

    # Filter columns for stationarity 
    if exclude_non_stationary:
        df = stationarity_filter(df, alpha)

    # Lag predictors
    df = df.shift(3*horizon_in_quarters)

    # Subset data 
    df = df.loc[desired_start_date_of_samples: last_date_of_sample]

    return df


