import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from dateutil.relativedelta import relativedelta
from utils.clean_data import remove_cols

load_dotenv()

DATA_DIR = os.getenv("DATADIR")
raw_data_dir = os.path.join(DATA_DIR, "raw/")

# Load configuration parameters
import yaml
with open("conf/prepare_us_data_conf.yaml", "r") as f:
    config = yaml.safe_load(f)

DESIRED_START_DATE_OF_SAMPLES = pd.to_datetime(config["desired_start_date_of_samples"])
HORIZON_IN_QUARTERS = config["horizon_in_quarters"]
REMOVE_COLS_THRESHOLD = config["remove_cols_threshold"]
INITIAL_TRAINING_LAST_DATE = pd.to_datetime(config["intial_training_last_date"])
LAST_DATE_OF_SAMPLE = pd.to_datetime(config["last_date_of_sample"])

financial_fred_vars = [
    'S&P 500', 
    'S&P div yield', 
    'S&P PE ratio', 
    'VIXCLSx', 
    'OILPRICEx', 
    'FEDFUNDS',
    'CP3Mx',
    'TB3MS',
    'TB6MS',
    'GS1',
    'GS5',
    'AAA',
    'BAA',
    'COMPAPFFx',
    'TB3SMFFM',
    'TB6SMFFM',
    'T1YFFM',
    'T10YFFM',
    'AAAFFM',
    'BAAFFM',
    'TWEXAFEGSMTHx',
    'EXSZUS',
    'EXPUSx',
    'EXUSUKx',
    'EXCAUSx'
]

def transfn(x: np.ndarray, tcode: int) -> np.ndarray:
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

def prepare_missing(rawdata: np.ndarray, tcode: list | np.ndarray) -> np.ndarray:
    """
    Transforms raw data based on each series' transformation code.

    Parameters:
        rawdata (numpy.ndarray): Raw data (each column is a series).
        tcode (list or numpy.ndarray): Transformation codes for each series.

    Returns:
        numpy.ndarray: Transformed data.
    """
    # Initialize output variable
    yt = []

    # Number of series
    N = rawdata.shape[1]

    for i in range(N):
        transformed_series = transfn(rawdata[:, i], tcode[i])
        yt.append(transformed_series)

    # Stack transformed series column-wise
    yt = np.column_stack(yt)
    return yt

def get_fred_md_x(fred_file_name: str) -> pd.DataFrame:
    
    """
    Prepares FRED data for modeling by applying transformations, lagging variables, and cleaning missing data.

    Parameters:
        fred_file_name (str): Name of the FRED data CSV file.   
    Returns:
        pd.DataFrame: Prepared FRED data ready for modeling.
    """
    
    df = pd.read_csv(os.path.join(raw_data_dir, fred_file_name))
    
    # Apply stationary transformations
    col_names = df.columns.tolist()
    tcodes = df.iloc[0,1:].values
    rawdata = df.iloc[1:,1:].values
    date_col = df.iloc[1:,0].values
    fred = prepare_missing(rawdata, tcodes)
    fred = pd.DataFrame(fred, index=df.index[1:], columns=col_names[1:])
    fred['date'] = pd.to_datetime(date_col, format='%m/%d/%Y')
    fred.set_index('date', inplace=True)

    # Trim data to desired date range plus buffer for lags
    fred = fred.loc[(DESIRED_START_DATE_OF_SAMPLES - relativedelta(months=3*HORIZON_IN_QUARTERS + 1)):LAST_DATE_OF_SAMPLE]

    # Drop columns with too many missing values in first training window
    fred = remove_cols(REMOVE_COLS_THRESHOLD, fred, train_end=INITIAL_TRAINING_LAST_DATE)

    fred_macro_cols = [c for c in fred if c not in financial_fred_vars]
    fred_fin_cols = [c for c in fred if c in financial_fred_vars]

    # Lag predictors, adding extra lag for macro variables for announcement delay
    fred[fred_fin_cols] = fred[fred_fin_cols].shift(3*HORIZON_IN_QUARTERS)
    fred[fred_macro_cols] = fred[fred_macro_cols].shift(3*HORIZON_IN_QUARTERS+1)

    fred = fred.loc[DESIRED_START_DATE_OF_SAMPLES:LAST_DATE_OF_SAMPLE]

    return fred

if __name__ == "__main__":
    fred_x = get_fred_md_x("2025-12-MD.csv")
    print(fred_x.head())