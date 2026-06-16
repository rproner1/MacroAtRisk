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
        year = re.search(r'\d{4}', f).group()
        years_present.append(year)
    
    years_expected = list(range(start_year, end_year + 1))

    missing = list(set(years_expected) - set(years_present))

    return missing

def estimate_mean_from_quantiles(
        preds, 
        weights: list[float]|None=None
    ):
    if weights is None:
        return np.mean(preds, axis=1).flatten()
    else:
        return (preds @ np.array(weights).reshape(-1,1)).flatten()

def get_mean_preds(
        quantile_preds: pd.DataFrame,
        models: list[str],
        weights: list[float]|None=None
):

    mean_preds = {}           
    for model in models:
        model_cols = [c for c in quantile_preds.columns if model in c]
        if 'Naive_Mean' in model_cols:
            model_cols.remove('Naive_Mean')
        
        model_preds = quantile_preds.loc[:, model_cols]
        model_mean_preds = estimate_mean_from_quantiles(
            model_preds.values,
            weights=weights
        )
        mean_preds[model] = model_mean_preds
    
    mean_preds_df = pd.DataFrame(mean_preds, index=quantile_preds.index)
    return mean_preds_df


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
            # Allow for some model types to be missing
            if len(dir_preds) > 0:
                dir_preds_df = pd.concat(dir_preds)
            else:
                dir_preds_df = pd.DataFrame()
            
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
    ) * 100

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
    ) * 100

    return r2

def _compute_r1_scores(
        y_true: pd.DataFrame|pd.Series|np.ndarray,
        y_pred: pd.DataFrame|np.ndarray, 
        benchmark: pd.DataFrame|np.ndarray,
        quantiles: list[float]|None = None
    ):

    '''
    Given a dataframe of a model's predictions and a list of quantiels, 
    compute R1 for each quantile and the average R1.
    Predictions should be arranged from lowest quantile to highest
    '''

    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values.flatten()
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    if isinstance(benchmark, pd.DataFrame):
        benchmark = benchmark.values

    if quantiles is None:
        quantiles = [.05, .25, .5, .75, .95]
        
    int_quantiles = [int(q*100) for q in quantiles]

    scores = {}

    for i, q in enumerate(quantiles):
        y_pred_q = y_pred[:,i]
        benchmark_q = benchmark[:, i]
        r1_q = r1_score(
            y_true=y_true,
            y_pred=y_pred_q,
            benchmark=benchmark_q,
            q=q
        )
        scores[str(int_quantiles[i])] = r1_q

    scores['Mean'] = np.mean(list(scores.values()))

    return scores

def get_r1_results_df(
        y_true: pd.Series,
        preds_df: pd.DataFrame,
        benchmark: pd.DataFrame,
        models: list[str],
        quantiles: list[float]
):
    
    results = {}
    for model in models:
        model_cols = [c for c in preds_df if model in c]
        model_preds = preds_df.loc[:, model_cols]

        # Compute R1 for each quantile for the model
        model_r1_scores = _compute_r1_scores(
            y_true=y_true, 
            y_pred=model_preds,
            benchmark=benchmark,
            quantiles=quantiles
        )
        results[model] = model_r1_scores

    return pd.DataFrame(results)

def get_r2_results_df(
        y_true: pd.Series,
        preds_df: pd.DataFrame,
        benchmark: pd.Series,
        models: list[str],
):
    r2s = {}
    for model in models:
        r2s[model] = [
            r2_score(
                y_true=y_true,
                y_pred=preds_df.loc[:,model],
                benchmark=benchmark
            )
        ]
    
    return pd.DataFrame(r2s)