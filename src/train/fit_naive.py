import pandas as pd 
import numpy as np
import os
from pathlib import Path
import yaml
from sklearn.

from src.data.prepare_data import prepare_non_rnn_data


DATA_DIR = Path('data/processed/')
PRED_DIR = Path('naive_predictions/')

os.makedirs(PRED_DIR, exist_ok=True)

with open("./config/config_file.yaml", "r") as f:
    config = yaml.safe_load(f)

# data
data_cfg = config['data']
COUNTRY = data_cfg['country']
HORIZON_IN_QUARTERS = data_cfg['horizon_in_quarters'] # In quarters, 1 or 4

# Other
QUANTILES = config['quantiles']
TARGET_DICT = {0: 'Infl_yoy', 1: 'IP_yoy', 2: 'Unrate_yoy'}


# Expanding mean and quantiles (lagged by 12 months)
LAG = 3*HORIZON_IN_QUARTERS + 1  # months extra for announcement delay
def expanding_stats(y, col, quantiles=QUANTILES, lag=LAG):
    stats = {}
    stats['Expanding_Mean'] = y[col].shift(lag).expanding(min_periods=1).mean()
    for q in quantiles:
        stats[f'Expanding_Q{q}'] = y[col].shift(lag).expanding(min_periods=1).quantile(q/100)
    return pd.DataFrame(stats, index=y.index)

def main():

    expanding_results = {}
    for target in range(0, 3):
        # Load full target series for expanding calculation
        target_path = f'{DATA_DIR}{COUNTRY}_targets_1961-01--2024-12.csv'
        y_full = pd.read_csv(target_path, index_col=0, parse_dates=True)
        col = y_full.columns[target]
        expanding_results[target] = expanding_stats(y_full, col)

        # Save expanding stats to disk (optional)
        expanding_results[target].to_csv(os.path.join(PRED_DIR, f'{COUNTRY}_{HORIZON_IN_QUARTERS}q_naive_preds_{target_dict[target]}.csv'))

if __name__ == "__main__":
    main()