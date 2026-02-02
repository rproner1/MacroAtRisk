from pathlib import Path
import pandas as pd
import numpy as np

def get_ar1_x(
        file_path: Path,
        desired_start_date_of_samples: pd.Timestamp,
        horizon_in_quarters: int,
        last_date_of_sample: pd.Timestamp
    ) -> pd.DataFrame:
    """
    Prepares AR(1) features from FRED target variables for modeling.

    Parameters:
        fred_file_name (str): Name of the FRED data CSV file.

    Returns:
        pd.DataFrame: Prepared AR(1) features ready for modeling.
    """
    df = pd.read_csv(file_path)
    
    # Fix date format and set as index
    df = df.loc[1:, :] # skip first row with transformation codes
    df['sasdate'] = pd.to_datetime(df['sasdate'], format='%m/%d/%Y')
    df.set_index('sasdate', inplace=True)
    df.index.name = 'date'

    # Create target variables
    infl_t = np.log(df['CPIAUCSL']) - np.log(df['CPIAUCSL'].shift(12))
    ip_t = np.log(df['INDPRO']) - np.log(df['INDPRO'].shift(12))
    lu_t = np.log(df['UNRATE']) - np.log(df['UNRATE'].shift(12))

    targets = pd.DataFrame({
        'Infl_yoy': infl_t,
        'IP_yoy': ip_t,
        'Unrate_yoy': lu_t
    })

    ar1_features = targets.shift(3*horizon_in_quarters + 1)
    ar1_features.columns = [f"{col}_t-1" for col in targets.columns]

    ar1_features = ar1_features.loc[desired_start_date_of_samples:last_date_of_sample]

    return ar1_features
