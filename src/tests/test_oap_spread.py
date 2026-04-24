import pandas as pd
import numpy as np
from src.data.make_oap_signals import get_firm_avg, get_firm_spread
from dotenv import load_dotenv
import os
from pathlib import Path
from src.utils.files import get_latest_file
load_dotenv()

DATADIR = Path(os.getenv("DATADIR"))
raw_data_dir = DATADIR / "raw/"

# file = get_latest_file(raw_data_dir / "signed_predictors_dl_wide.csv")
# df = pd.read_csv(file, skiprows=range(1,10000), nrows=2000)

df = pd.DataFrame(
    {
        'permno': [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010]*12,
        'yyyymm': list(range(202001, 202012+1))*10,
        'signal1': np.random.randn(12*10),
        'signal2': np.random.randn(12*10)
    }
)

df.loc[0:5, 'signal1'] = np.nan  # introduce some NaNs
df.loc[10:15, 'signal2'] = np.inf  # introduce some infs

# print(df.head())
spread_df = get_firm_spread(df, quantiles=2)
print(spread_df.head())

