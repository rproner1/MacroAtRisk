import pandas as pd
import numpy as np
# import statsmodels.api as sm
import statsmodels.formula.api as smf
from utils import prepare_quantile_data

from operator import itemgetter
import argparse
import os

# Arguments and Paths
parser = argparse.ArgumentParser(description="fit quantile regression from Adrian et al. 2019.")
parser.add_argument("--year", type=int, required=True, help="train cutoff year")
parser.add_argument("--country", type=str, default="us", help="country code (us/ca)")
parser.add_argument("--horizon", type=int, default=4, help="forecast horizon in quarters")
parser.add_argument("--quantiles", type=float, nargs="*", default=[0.05,0.25,0.50,0.75,0.95], help="list of quantiles to predict")
# parser.add_argument("--local", action="store_true", help="run locally (use local data/DB)")
args = parser.parse_args()

print(f"Arguments: {args}")

YEAR = args.year
COUNTRY = args.country
HORIZON_IN_QUARTERS = args.horizon
QUANTILES = args.quantiles

DATE = '25-12-23'

DATA_DIR = "/home/rproner/Documents/Data/MacroAtRisk/"
PRED_DIR = f"/home/rproner/Documents/Projects/MacroAtRisk/LitBenchmarkPredictions_{DATE}/"
TRAIN_PRED_DIR = f"/home/rproner/Documents/Projects/MacroAtRisk/LitBenchmarkTrainPredictions_{DATE}/"

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(TRAIN_PRED_DIR, exist_ok=True)

# Vulnerable Growth
print('Fitting Vulnerable Growth Benchmark Model...')
vg_data, _ , _ = prepare_quantile_data(
    target='IP_yoy', 
    time_steps=12, 
    targets_path=f'{DATA_DIR}{COUNTRY}_targets_1961-01--2024-12.csv', 
    input_paths=[f'{DATA_DIR}{COUNTRY}_vulnerable_growth_predictors_{HORIZON_IN_QUARTERS}q_2024-12.csv'],
    start_date='1974-02-01', 
    train_cutoff_year=YEAR, 
    n_quantiles=len(QUANTILES), 
    val_years=5
)

(
    X_train, y_train, X_test 
) = itemgetter(
    'X_train_full', 'y_train_full', 'X_test' 
)(vg_data)

# Combine X_train and y_train into a DataFrame for formula API
train_df = X_train.copy()
train_df['target'] = y_train

# Build formula string: target ~ col1 + col2 + ...
predictors = " + ".join(X_train.columns)
formula = f"target ~ {predictors}"

# Fit quantile regression models for each quantile
vg_preds = {}
vg_train_preds = {}
for q in QUANTILES:

    Q = int(q*100)

    model = smf.quantreg(formula, data=train_df)
    res = model.fit(q=q)
    print(f"Quantile: {q}, Params: {res.params}")

    # predict
    preds = res.predict(X_test)
    train_preds = res.predict(X_train)
    vg_preds[f'VG_Q{Q}'] = preds
    vg_train_preds[f'VG_Q{Q}'] = train_preds

# print("Vulnerable Growth Predictions:")
# print(pd.DataFrame(vg_preds).head())
pd.DataFrame(vg_preds).to_csv(f'{PRED_DIR}lit_bench_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_IP_yoy_{YEAR}.csv')
pd.DataFrame(vg_train_preds).to_csv(f'{TRAIN_PRED_DIR}lit_bench_train_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_IP_yoy_{YEAR}.csv')

# Unemployment at Risk
print("Fitting Unemployment at Risk model...")
uar_data, _ , _ = prepare_quantile_data(
    target='Unrate_yoy', 
    time_steps=12, 
    targets_path=f'{DATA_DIR}{COUNTRY}_targets_1961-01--2024-12.csv', 
    input_paths=[f'{DATA_DIR}{COUNTRY}_unemployment_at_risk_predictors_{HORIZON_IN_QUARTERS}q_2024-12.csv'],
    start_date='1974-02-01',
    train_cutoff_year=YEAR, 
    n_quantiles=len(QUANTILES), 
    val_years=2
)

(
    X_train, y_train, X_test 
) = itemgetter(
    'X_train_full', 'y_train_full', 'X_test' 
)(uar_data)

# Combine X_train and y_train into a DataFrame for formula API
train_df = X_train.copy()
train_df['target'] = y_train

# Build formula string: target ~ col1 + col2 + ...
predictors = " + ".join(X_train.columns)
formula = f"target ~ {predictors}"

uar_preds = {}
uar_train_preds = {}

for q in QUANTILES:

    Q = int(q*100)

    model = smf.quantreg(formula, data=train_df)
    res = model.fit(q=q)
    print(f"Quantile: {q}, Params: {res.params}")

    # predict
    preds = res.predict(X_test)
    uar_preds[f'UAR_Q{Q}'] = preds
    train_preds = res.predict(X_train)
    uar_train_preds[f'UAR_Q{Q}'] = train_preds

print("Saving UaR Predictions...")
pd.DataFrame(uar_preds).to_csv(f'{PRED_DIR}lit_bench_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_Unrate_yoy_{YEAR}.csv')
pd.DataFrame(uar_train_preds).to_csv(f'{TRAIN_PRED_DIR}lit_bench_train_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_Unrate_yoy_{YEAR}.csv')

# Inflation at Risk
print('Fitting Inflation at Risk model...')
iar_data, _ , _ = prepare_quantile_data(
    target='Infl_yoy', 
    time_steps=12, 
    targets_path=f'{DATA_DIR}{COUNTRY}_targets_1961-01--2024-12.csv', 
    input_paths=[f'{DATA_DIR}{COUNTRY}_inflation_at_risk_predictors_{HORIZON_IN_QUARTERS}q_2024-12.csv'],
    start_date='1974-02-01',
    train_cutoff_year=YEAR, 
    n_quantiles=len(QUANTILES), 
    val_years=2
)

(
    X_train, y_train, X_test 
) = itemgetter(
    'X_train_full', 'y_train_full', 'X_test' 
)(iar_data)

# Combine X_train and y_train into a DataFrame for formula API
train_df = X_train.copy()
train_df['target'] = y_train

# Build formula string: target ~ col1 + col2 + ...
predictors = " + ".join(X_train.columns)
formula = f"target ~ {predictors}"

iar_preds = {}
iar_train_preds = {}
for q in QUANTILES:

    Q = int(q*100)

    model = smf.quantreg(formula, data=train_df)
    res = model.fit(q=q)
    print(f"Quantile: {q}, Params: {res.params}")

    # predict
    preds = res.predict(X_test)
    iar_preds[f'IAR_Q{Q}'] = preds
    train_preds = res.predict(X_train)
    iar_train_preds[f'IAR_Q{Q}'] = train_preds
print("Saving IaR Predictions...")
pd.DataFrame(iar_preds).to_csv(f'{PRED_DIR}lit_bench_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_Infl_yoy_{YEAR}.csv')
pd.DataFrame(iar_train_preds).to_csv(f'{TRAIN_PRED_DIR}lit_bench_train_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_Infl_yoy_{YEAR}.csv')

print('COMPLETE.')
