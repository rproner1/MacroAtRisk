import pandas as pd
import numpy as np
import warnings
from dateutil.relativedelta import relativedelta
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta
from pathlib import Path

from src.data.data_utils import (
    remove_cols, 
    stationarity_filter,
    get_value_weights
)


def _clean_jkp(
        df
):
    # Remove missing permnos
    df = (
        df
        .dropna(subset=['permno'])
        .drop(
            columns = [
                'obs_main', 'exch_main', 'common', 'primary_sec', 'excntry', 'eom'
            ]
        )
        .rename(columns={'me': 'size'})
    )

    df['date'] = (
        pd.to_datetime(df['date'], format='%Y-%m-%d')
        .dt.to_period('M').dt.to_timestamp()
    )

    df['permno'] = df['permno'].astype('int')

    return df

def _get_firm_avg_jkp(
        df: pd.DataFrame, 
        value_weight: bool = False, 
        sample_start_date: str = '1961-01-01', 
        sample_end_date: str = '2024-12-01', 
        horizon_in_quarters: int=4
) -> pd.DataFrame:
    """
    Computes the firm-level average of financial variables from OAP data.

    Parameters:
    df: pd.DataFrame
        DataFrame containing firm-level signals with a 'yyyymm' column in %Y%m format.
    value_weight: bool, optional
        Whether to weight the average calculation by firm value. Defaults to False.
    size: pd.DataFrame, optional
        DataFrame containing size for each firm at each date. Required if value_weight is True.

    Returns:
    pd.DataFrame: DataFrame with average firm-level signals per date.
    """
    
    # Subset data
    start_date =(
        pd.to_datetime(sample_start_date) 
        - relativedelta(months=3*horizon_in_quarters + 1)
    )
    df = df[(df['date'] >= start_date) & (df['date'] <= sample_end_date)]

    # Group by 'date' and compute the cross-sectional mean for each financial variable
    if value_weight:
        # Compute size weights 
        size = df.loc[:, ['date', 'permno', 'size']]
        df.drop(columns=['size'], inplace=True)
        weights = get_value_weights(size)

        # Ensure weights sum to 1 for each date. If not, raise an error. This is a sanity check to catch misalignments between df and size.
        if not weights.groupby('date')['weight'].sum().round().eq(1.0).all():
            raise ValueError("Weights do not sum to 1 for all dates.")
        
        before_rows = len(df)

        # Merge weights with characteristics df
        df = df.merge(weights, on=['date', 'permno'], how='inner')
        dropped_share = 1 - (len(df) / before_rows)
        if dropped_share > 0:
            warnings.warn(
                f"Dropped {dropped_share:.1%} of rows due to missing weights. "
                "Check that 'permno' and 'date' align between df and size.",
                RuntimeWarning,
            )
        
        df.replace([-np.inf, np.inf], np.nan, inplace=True)
        signal_cols = df.drop(columns=['permno', 'date', 'weight']).columns 
        
        # Take the weighted average of each signal by month. This is done by multiplying each signal by the weight and then summing across firms for each month.
        firm_avg_df = (
            df[signal_cols]
            .mul(df['weight'], axis=0)
            .groupby(df['date'])
            .sum()
            .reset_index()
        )

    else:
        # Take the average signal value across firms for each month. 
        firm_avg_df = (
            df
            .replace([-np.inf, np.inf], np.nan)
            .drop(columns=['permno'])
            .groupby('date')
            .mean()
            .reset_index()
        )

    # Replace zeros with NaN. Zeros arise when there are too few firms in a month to populate bins. 
    # Some signals are sparse and this is frequent. Such signals will later be removed.
    firm_avg_df.replace(0, np.nan, inplace=True)

    return firm_avg_df

def get_jkp_x(
        file_path: Path, 
        first_difference: bool,
        horizon_in_quarters: int,
        desired_start_date: pd.Timestamp,
        desired_end_date: pd.Timestamp,
        remove_cols_threshold: float,
        exclude_non_stationary: bool,
        alpha: float,
        initial_training_last_date: pd.Timestamp
    ) -> pd.DataFrame:
    """
    Prepares features from JKP characteristics
    """
    df = pd.read_parquet(file_path)
    
    df = _clean_jkp(df)

    df = _get_firm_avg_jkp(df)

    df.set_index('date', inplace=True)

    # Subset data to avoid removing too many columns due to initial NaNs
    buffer = int(first_difference) 

    effective_start_date = (
        desired_start_date 
        - relativedelta(months=3*horizon_in_quarters + buffer)
    )
    df = df.loc[effective_start_date: desired_end_date]

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
    df = df.loc[desired_start_date: desired_end_date]

    return df

