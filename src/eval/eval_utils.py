import pandas as pd
import numpy as np
from pathlib import Path
import glob
import warnings
import re
from sklearn.metrics import mean_pinball_loss, mean_squared_error

def _get_missing_years(files, start_year: int = 1997, end_year: int = 2023):
    
    years_present = []
    for f in files:
        year = re.search(r'\d{4}').group()
        years_present.append(year)
    
    years_expected = list(range(start_year, end_year + 1))

    missing = list(set(years_expected) - set(years_present))

    return missing


def concat_preds(
        pred_dir_paths: list[str|Path],
        out_dir: str|Path,
        targets_to_concat: list[int] = ['Infl_yoy', 'IP_yoy', 'Unrate_yoy'],
        start_year: int = 1997,
        end_year: int = 2023,
        country: str = 'us',
        horizon_in_quarters: int = 4
):
    
    n_years = end_year - start_year + 1

    for target in targets_to_concat:
        
        target_preds = []
        for dir_path in pred_dir_paths:

            dir_target_pred_files = sorted(
                glob.glob(
                    str(
                        Path(dir_path) 
                        / f'*{country}*{horizon_in_quarters}*{target}*'
                    )
                )
            )

            dir_preds = []
            for f in dir_target_pred_files:


                dir_pred_yr = pd.read_csv(
                    f, 
                    index_col=0, 
                    parse_dates=True
                )
            
                dir_preds.append(dir_pred_yr)
            
            if len(dir_preds) != n_years:
                missing = _get_missing_years(
                    dir_target_pred_files,
                    start_year=start_year,
                    end_year=end_year
                )
                warnings.warn(f'Predictions missing for years {missing}')

            # Concatenate predictions within model families
            dir_preds_df = pd.concat(dir_preds)
            
            target_preds.append(dir_preds_df)

        # Concatenate predictions across model families
        target_preds_df = pd.concat(target_preds, axis=1)

        out_path = (
            out_dir / 
            (
                f'all_models_predictions_{country}_{horizon_in_quarters}q_'
                f'{target}.csv'
            )
        )
        target_preds_df.to_csv(out_path)

def r1_score(
        y_true: np.ndarray|pd.Series, 
        y_pred: np.ndarray|pd.Series, 
        benchmark: np.ndarray|pd.Series, 
        q: float
    ):

    '''
    Computes r1 score given by
        1 - A/B
    where A is the pinball loss of the model and B is the pinball loss of
    the constant-only model.
    '''

    r1 = (
        1 
        - mean_pinball_loss(y_true=y_true, y_pred=y_pred, alpha=q)
        / mean_pinball_loss(y_true=y_true, y_pred=benchmark, alpha=q)
    )

    return r1
        
def r2_score(
        y_true: np.ndarray|pd.Series, 
        y_pred: np.ndarray|pd.Series, 
        benchmark: np.ndarray|pd.Series,
):
    
    '''
    Computes r1 score given by
        1 - A/B
    where A is the mse of the model and B is the mse of
    the constant-only model.
    '''

    r2 = (
        1 
        - mean_squared_error(y_true=y_true, y_pred=y_pred)
        / mean_squared_error(y_true=y_true, y_pred=benchmark)
    )

    return r2