import pandas as pd
import numpy as np
from src.data.make_oap_signals import get_firm_avg, get_value_weights

df = pd.DataFrame(
    {
        'permno': [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010]*12,
        'yyyymm': list(range(202001, 202012+1))*10,
        'signal1': np.random.randn(12*10),
        'signal2': np.random.randn(12*10)
    }
)

size = pd.DataFrame(
    {
        'permno': [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010]*12,
        'yyyymm': list(range(202001, 202012+1))*10,
        'altprc': np.random.rand(12*10),
        'shrout': np.random.rand(12*10) * 1_000_000,
        'size': (np.random.rand(12*10) * 1000)
    }
)

df.loc[0:5, 'signal1'] = np.nan  # introduce some NaNs
df.loc[10:15, 'signal2'] = np.inf  # introduce some infs

firm_avg_df = get_firm_avg(df, value_weight=False)
print(firm_avg_df.head())

firm_vw_avg_df = get_firm_avg(df, value_weight=True, size=size)
print(firm_vw_avg_df.head())