from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import os
from datetime import date


def concat_predictions(
    st_pred_dir: Path,
    lit_bench_pred_dir: Path,
    shelf_pred_dir: Path,
    pred_dir: Path,
    country: str = 'us',
    horizon_in_quarters: int = 4,
    date: str = date.today(),
    start_year: int = 1997,
    end_year: int = 2023
):
    """
    Concatenate predictions from different model types.
    
    Parameters:
    country: str
        Country code (us/ca)
    horizon_in_quarters: int
        Forecast horizon in quarters
    date: str
        Date identifier for output directory
    st_pred_dir: Path
        Directory containing state-of-the-art model predictions
    lit_bench_pred_dir: Path
        Directory containing literature benchmark predictions
    shelf_pred_dir: Path
        Directory containing shelf model predictions
    pred_dir: Path
        Output directory for concatenated predictions (default: Predictions/{date}/)
    start_year: int
        First year to process. Years are training cutoff years.
    end_year: int
        Last year to process (inclusive). Years are training cutoff years.
    """

    target_name_dict = {
        0: 'Infl_yoy',
        1: 'IP_yoy',
        2: 'Unrate_yoy'
    }
    
    for TARGET_IDX in target_name_dict.keys():
        preds = []
        for yr in range(start_year, end_year + 1):
            start = f'{yr+1}-01-01'
            end = f'{yr+1}-12-01'
            st_preds = pd.read_csv(st_pred_dir / f"st_model_predictions_{country}_{horizon_in_quarters}q_{target_name_dict[TARGET_IDX]}_{yr}.csv", index_col=0, parse_dates=True).loc[start:end]
            lit_bench_preds = pd.read_csv(lit_bench_pred_dir / f"lit_bench_predictions_{country}_{horizon_in_quarters}q_{target_name_dict[TARGET_IDX]}_{yr}.csv", index_col=0, parse_dates=True).loc[start:end]
            shelf_preds = pd.read_csv(shelf_pred_dir / f"shelf_model_predictions_{country}_{horizon_in_quarters}q_{target_name_dict[TARGET_IDX]}_{yr}.csv", index_col=0, parse_dates=True).loc[start:end]

            data_list = [
                lit_bench_preds,
                shelf_preds,
                st_preds
            ]

            preds_y = pd.concat(data_list, axis=1)   
            preds.append(preds_y)

        preds = pd.concat(preds)

        preds.to_csv(pred_dir / f"all_models_predictions_{country}_{horizon_in_quarters}q_{target_name_dict[TARGET_IDX]}.csv")
    
    print(f"Concatenated predictions saved to {pred_dir}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate predictions")
    parser.add_argument("--target", type=int, required=False, help="target variable index")
    parser.add_argument("--country", type=str, default='us', help="Country code (us/ca)")
    parser.add_argument("--horizon", type=int, default=4, help="forecast horizon in quarters")
    parser.add_argument("--date", type=str, default="20260108", help="date identifier for output directory")
    parser.add_argument("--start-year", type=int, default=1997, help="first year to process")
    parser.add_argument("--end-year", type=int, default=2023, help="last year to process")

    args = parser.parse_args()
    
    concat_predictions(
        country=args.country,
        horizon_in_quarters=args.horizon,
        date=args.date,
        start_year=args.start_year,
        end_year=args.end_year
    )
