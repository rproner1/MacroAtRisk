
import pandas as pd
from dateutil.relativedelta import relativedelta
from src.utils.clean_data import remove_cols
from pathlib import Path


def get_firm_level_x(
        file_path: Path, 
        first_difference: bool,
        horizon_in_quarters: int,
        desired_start_date_of_samples: pd.Timestamp,
        last_date_of_sample: pd.Timestamp,
        remove_cols_threshold: float,
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

    # Lag predictors
    df = df.shift(3*horizon_in_quarters)

    # Subset data 
    df = df.loc[desired_start_date_of_samples: last_date_of_sample]

    return df


