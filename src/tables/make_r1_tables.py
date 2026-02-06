import pandas as pd
import numpy as np
from src.train.fit_naive import expanding_stats

from src.utils.evaluation import compute_oos_r1_score

import os
import argparse
from pathlib import Path
from datetime import date

def make_r1_tables(
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
    Generate R1 score tables for model evaluation.
    
    Parameters:
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
    pred_dir: str
        Directory containing predictions
    naive_pred_dir: str
        Directory containing naive predictions
    results_dir: str
        Directory to save results (default: Results/{date}/)
    tables_dir: str
        Directory to save LaTeX tables
    """
    
    int_quantiles = [int(q*100) for q in quantiles]

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

    # Naive rolling mean and quantile predictions for computing out-of-sample R1 and R2
    y_full = pd.read_parquet(targets_path).loc['1961-01-01':'2024-12-01', :]
    naive_preds = expanding_stats(
        y_full,
        col=target_dict[target_idx],
        quantiles=[int(q*100) for q in quantiles],
        lag=(3*horizon_in_quarters + 1)
    ).loc[test_start:test_end]

    # Load predictions
    preds = pd.read_csv(pred_path, index_col=0, parse_dates=True).loc[test_start:test_end]
    models_list = set([c.split('_')[0] for c in preds.columns if '_' in c])

    # Load actuals
    actuals = y_full.loc[test_start:test_end, target_dict[target_idx]]

    # Compute R1
    results = []
    for model in models_list:

        for q in quantiles:

            q_int = int(q*100)

            model_q_preds = preds.loc[:, f"{model}_Q{q_int}"]

            r1 = compute_oos_r1_score(
                y_true=actuals.values.flatten(),
                y_pred=model_q_preds.values.flatten(),
                benchmark_pred=naive_preds.loc[test_start:test_end, f"Expanding_Q{q_int}"].values.flatten(),
                q=q
            )

            results.append({
                'Model': model,
                'Quantile': q,
                'R1': r1
            })


    r1_results_df = pd.DataFrame(results).pivot(index='Model', columns='Quantile', values='R1').reset_index().apply(lambda x: round(x, 1) if x.name!='Model' else x)
    r1_results_df['Mean'] = r1_results_df.loc[:, quantiles].mean(axis=1)
    r1_results_df.sort_values('Mean', ascending=False).to_csv(f"{results_dir}oos_r1_{country}_{horizon_in_quarters}q_{target_dict[target_idx]}_{test_start}-{test_end}.csv", index=False)

    # Make latex table
    r1_results_df = r1_results_df.set_index(['Model'])
    row_order = ['Mean'] + r1_results_df.columns[:-1].tolist()
    r1_report_df = r1_results_df.transpose().loc[row_order, model_subset]
    rename_rows_map = {k: f"Q{int(k*100)}" if k!='Mean' else 'Mean' for k in r1_report_df.index}
    r1_report_df.rename(index=rename_rows_map, inplace=True)
    r1_report_df.columns.name = None
    r1_report_df.index.name = None
    r1_report_df.to_latex(tables_dir / f"r1_{target_dict[target_idx]}.tex", float_format="%.1f")
    
    print("R1 tables generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate forecasts")
    parser.add_argument("--target", type=int, default=0, help="Target index (0: Infl, 1: IP, 2: Unrate)")
    parser.add_argument("--country", type=str, default='us', help="Country code (us/ca)")
    parser.add_argument("--horizon", type=int, default=4, help="forecast horizon in quarters")
    parser.add_argument("--quantiles", type=float, nargs="*", default=[0.05, 0.25, 0.50, 0.75, 0.95], help="list of quantiles to predict")
    parser.add_argument("--test-start", type=str, default="1998-01-01", help="start date for the test set")
    parser.add_argument("--test-end", type=str, default="2024-12-01", help="end date for the test set")
    parser.add_argument("--date", type=str, default="20260108", help="date identifier for results directory")

    args = parser.parse_args()
    
    make_r1_tables(
        target_idx=args.target,
        country=args.country,
        horizon_in_quarters=args.horizon,
        quantiles=args.quantiles,
        test_start=args.test_start,
        test_end=args.test_end,
        date=args.date
    )