import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime

DATE = datetime.now().strftime("%Y%m%d")
ST_PRED_DIR = f"/home/rproner/Documents/Projects/MacroAtRisk/ST_Predictions/"
LIT_BENCH_PRED_DIR = f"/home/rproner/Documents/Projects/MacroAtRisk/LitBenchmarkPredictions/"
SHELF_PRED_DIR = f"/home/rproner/Documents/Projects/MacroAtRisk/Shelf_Predictions/"
PRED_DIR = f"/home/rproner/Documents/Projects/MacroAtRisk/Predictions/{DATE}/"

os.makedirs(PRED_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Concatenate predictions")
parser.add_argument("--target", type=int, required=False, help="target variable index")
parser.add_argument("--country", type=str, default='us', help="Country code (us/ca)")
parser.add_argument("--horizon", type=int, default=4, help="forecast horizon in quarters")

args = parser.parse_args()
TARGET_IDX = args.target
COUNTRY = args.country
HORIZON_IN_QUARTERS = args.horizon

TEST_START = '1998-01-01'
TEST_END = '2024-12-01'
target_name_dict = {
    0: 'Infl_yoy',
    1: 'IP_yoy',
    2: 'Unrate_yoy'
}
for TARGET_IDX in target_name_dict.keys():
    preds = []
    for yr in range(1997,2024):
        start = f'{yr+1}-01-01'
        end = f'{yr+1}-12-01'
        st_preds = pd.read_csv(f"{ST_PRED_DIR}st_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name_dict[TARGET_IDX]}_{yr}.csv", index_col=0, parse_dates=True).loc[start:end]
        lit_bench_preds = pd.read_csv(f"{LIT_BENCH_PRED_DIR}lit_bench_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name_dict[TARGET_IDX]}_{yr}.csv", index_col=0, parse_dates=True).loc[start:end]
        shelf_preds = pd.read_csv(f"{SHELF_PRED_DIR}shelf_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name_dict[TARGET_IDX]}_{yr}.csv", index_col=0, parse_dates=True).loc[start:end]

        data_list = [
            lit_bench_preds,
            shelf_preds,
            st_preds
        ]

        preds_y = pd.concat(data_list, axis=1)   
        preds.append(preds_y)

    preds = pd.concat(preds)

    preds.to_csv(f"{PRED_DIR}all_models_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name_dict[TARGET_IDX]}.csv")
