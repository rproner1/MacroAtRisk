import pandas as pd
import numpy as np
from src.train.fit_naive import expanding_stats
from src.utils.evaluation import compute_oos_r2_score, estimate_mean_from_quantiles

import os
import argparse
from pathlib import Path
from datetime import date

def make_r2_tables(
    target_idx: int,
    targets_path: Path,
    pred_path: Path,
    results_dir: Path,
    tables_dir: Path,
    base_models_subset: list[str] = ['LR','LAS','QRF','QGB','DMQv0c','DMQv1c','DMQv2c'],
    country: str = 'us',
    horizon_in_quarters: int = 4,
    quantiles: list[float] = [0.05, 0.25, 0.50, 0.75, 0.95],
    test_start: str = '1998-01-01',
    test_end: str = '2024-12-01',
    date: str = str(date.today())
    ):
    """
    Generate R2 score tables for model evaluation.
    
    Parameters:
    -----------
    target_idx: int
        Target variable index (0: Infl_yoy, 1: IP_yoy, 2: Unrate_yoy)
    targets_path: Path
        Path to targets file
    pred_path: Path
        Path to predictions CSV file
    results_dir: Path
        Directory to save results CSV
    tables_dir: Path
        Directory to save LaTeX tables
    base_models_subset: list[str]
        List of base model names to include
    country: str
        Country code (us/ca)
    horizon_in_quarters: int
        Forecast horizon in quarters
    quantiles: list[float]
        List of quantiles to evaluate
    test_start: str
        Start date for test set
    test_end: str
        End date for test set
    date: str
        Date identifier for results directory
    """
    
    target_dict = {
        0: 'Infl_yoy',
        1: 'IP_yoy',
        2: 'Unrate_yoy'
    }

    if target_idx not in target_dict:
        raise ValueError(f"Invalid target_idx: {target_idx}. Must be 0, 1, or 2.")

    if target_idx == 0:
        benchmark_model = "IAR"
    elif target_idx == 1:
        benchmark_model = "VG"
    else:
        benchmark_model = "UAR"

    model_subset = ['AR1', benchmark_model] + base_models_subset

    # Naive rolling mean and quantile predictions for computing out-of-sample R2
    y_full = pd.read_csv(targets_path, index_col=0, parse_dates=True).loc['1961-01-01':'2024-12-01', :]
    naive_preds = expanding_stats(
        y_full,
        col=target_dict[target_idx],
        quantiles=[int(q*100) for q in quantiles],
        lag=(3*horizon_in_quarters + 1)
    ).loc[test_start:test_end]

    naive_mean_test = naive_preds.loc[:, "Expanding_Mean"]

    # Load predictions
    preds = pd.read_csv(pred_path, index_col=0, parse_dates=True).loc[test_start:test_end]
    models_list = set([c.split('_')[0] for c in preds.columns if '_' in c])

    # Load actuals
    actuals = y_full.loc[test_start:test_end, target_dict[target_idx]]

    # Evaluate forecasts and compute R2
    r2_report = {
        'Model': [],
        'R2': []
    }

    all_model_mean_preds = {}
    for model in models_list:

        # Grab model quantile predictions
        int_quantiles = [int(q*100) for q in quantiles]
        model_preds = preds.loc[:, f"{model}_Q{int_quantiles[0]}":f"{model}_Q{int_quantiles[-1]}"]

        # Estimate the mean from quantiles
        model_mean_preds = estimate_mean_from_quantiles(model_preds.values, weights=[0.15, 0.225, 0.25, 0.225, 0.15])
        all_model_mean_preds[model] = model_mean_preds

        # Compute R2
        r2 = compute_oos_r2_score(
            y_true=actuals.values.flatten(),
            y_pred=model_mean_preds.flatten(),
            benchmark=naive_mean_test.values.flatten()
        )

        r2_report['Model'].append(model)
        r2_report['R2'].append(r2)

    # Compute R2 for AR1_Mean model
    r2 = compute_oos_r2_score(
        y_true=actuals.values.flatten(),
        y_pred=preds.loc[:, 'AR1_Mean'].values.flatten(),
        benchmark=naive_mean_test.values.flatten()
    )
    r2_report['Model'].append('AR1_Mean')
    r2_report['R2'].append(r2)

    # Create results dataframe and save CSV
    r2_report_df = pd.DataFrame(r2_report).apply(lambda x: round(x, 1) if x.name=='R2' else x)
    r2_report_df.sort_values('R2', ascending=False).to_csv(
        results_dir / f"oos_r2_{country}_{horizon_in_quarters}q_{target_dict[target_idx]}_{test_start}-{test_end}.csv", 
        index=False
    )

    # Make LaTeX table
    r2_report_df = r2_report_df.set_index('Model').transpose().loc[:, model_subset]
    r2_report_df.to_latex(tables_dir / f"r2_{target_dict[target_idx]}_{test_start}-{test_end}.tex", float_format="%.1f")
    
    print(f"R2 tables generation complete for {target_dict[target_idx]}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate R2 evaluation tables")
    parser.add_argument("--target", type=int, default=0, help="Target index (0: Infl, 1: IP, 2: Unrate)")
    parser.add_argument("--country", type=str, default='us', help="Country code (us/ca)")
    parser.add_argument("--horizon", type=int, default=4, help="forecast horizon in quarters")
    parser.add_argument("--quantiles", type=float, nargs="*", default=[0.05, 0.25, 0.50, 0.75, 0.95], help="list of quantiles to predict")
    parser.add_argument("--test-start", type=str, default="1998-01-01", help="start date for the test set")
    parser.add_argument("--test-end", type=str, default="2024-12-01", help="end date for the test set")
    parser.add_argument("--date", type=str, default="20260108", help="date identifier for results directory")

    args = parser.parse_args()
    
    make_r2_tables(
        target_idx=args.target,
        country=args.country,
        horizon_in_quarters=args.horizon,
        quantiles=args.quantiles,
        test_start=args.test_start,
        test_end=args.test_end,
        date=args.date
    )