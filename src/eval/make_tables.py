"""
Consolidated table generation module for R1 and R2 score evaluation.
"""
import pandas as pd
import numpy as np
from src.train.fit_naive import expanding_stats
from src.utils.evaluation import compute_oos_r1_score, compute_oos_r2_score, estimate_mean_from_quantiles

import os
import argparse
from pathlib import Path
from datetime import date


TARGET_DICT = {
    0: 'Infl_yoy',
    1: 'IP_yoy',
    2: 'Unrate_yoy'
}

BENCHMARK_MODEL_BY_TARGET = {
    0: 'IAR',
    1: 'VG',
    2: 'UAR'
}


def _compute_r1_results_df(
    target_idx: int,
    targets_path: Path,
    pred_path: Path,
    horizon_in_quarters: int,
    quantiles: list[float],
    test_start: str,
    test_end: str
) -> pd.DataFrame:
    """Compute R1 results by model for a single target."""

    if target_idx not in TARGET_DICT:
        raise ValueError(f"Invalid target_idx: {target_idx}. Must be 0, 1, or 2.")

    target_name = TARGET_DICT[target_idx]

    y_full = pd.read_parquet(targets_path).loc['1961-01-01':'2024-12-01', :]
    naive_preds = expanding_stats(
        y_full,
        col=target_name,
        quantiles=[int(q * 100) for q in quantiles],
        lag=(3 * horizon_in_quarters + 1)
    ).loc[test_start:test_end]

    preds = pd.read_csv(pred_path, index_col=0, parse_dates=True).loc[test_start:test_end]
    models_list = sorted(set([c.split('_')[0] for c in preds.columns if '_' in c]))
    actuals = y_full.loc[test_start:test_end, target_name]

    results = []
    for model in models_list:
        for q in quantiles:
            q_int = int(q * 100)
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

    r1_results_df = pd.DataFrame(results).pivot(index='Model', columns='Quantile', values='R1').reset_index().apply(
        lambda x: round(x, 1) if x.name != 'Model' else x
    )
    r1_results_df['Mean'] = r1_results_df.loc[:, quantiles].mean(axis=1)
    r1_results_df = r1_results_df.sort_values('Mean', ascending=False)
    return r1_results_df


def _format_latex_cell(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.1f}"


def _compute_r2_results_df(
    target_idx: int,
    targets_path: Path,
    pred_path: Path,
    horizon_in_quarters: int,
    quantiles: list[float],
    test_start: str,
    test_end: str,
) -> pd.DataFrame:
    """Compute R2 results by model for a single target."""

    if target_idx not in TARGET_DICT:
        raise ValueError(f"Invalid target_idx: {target_idx}. Must be 0, 1, or 2.")

    target_name = TARGET_DICT[target_idx]

    y_full = pd.read_parquet(targets_path).loc['1961-01-01':'2024-12-01', :]
    naive_preds = expanding_stats(
        y_full,
        col=target_name,
        quantiles=[int(q * 100) for q in quantiles],
        lag=(3 * horizon_in_quarters + 1)
    ).loc[test_start:test_end]
    naive_mean_test = naive_preds.loc[:, "Expanding_Mean"]

    preds = pd.read_csv(pred_path, index_col=0, parse_dates=True).loc[test_start:test_end]
    models_list = sorted(set([c.split('_')[0] for c in preds.columns if '_' in c]))
    actuals = y_full.loc[test_start:test_end, target_name]

    r2_report = {
        'Model': [],
        'R2': []
    }

    for model in models_list:
        int_quantiles = [int(q * 100) for q in quantiles]
        model_preds = preds.loc[:, f"{model}_Q{int_quantiles[0]}":f"{model}_Q{int_quantiles[-1]}"]

        model_mean_preds = estimate_mean_from_quantiles(
            model_preds.values,
            weights=[0.15, 0.225, 0.25, 0.225, 0.15]
        )

        r2 = compute_oos_r2_score(
            y_true=actuals.values.flatten(),
            y_pred=model_mean_preds.flatten(),
            benchmark=naive_mean_test.values.flatten()
        )

        r2_report['Model'].append(model)
        r2_report['R2'].append(r2)

    if 'AR1_mean' in preds.columns:
        r2 = compute_oos_r2_score(
            y_true=actuals.values.flatten(),
            y_pred=preds.loc[:, 'AR1_Mean'].values.flatten(),
            benchmark=naive_mean_test.values.flatten()
        )
        r2_report['Model'].append('AR1_Mean')
        r2_report['R2'].append(r2)

    return pd.DataFrame(r2_report).apply(lambda x: round(x, 1) if x.name == 'R2' else x)


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
    date_str: str = None
    ):
    """
    Generate R1 score tables for model evaluation.
    
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
    date_str: str
        Date identifier for results directory
    """
    
    if date_str is None:
        date_str = str(date.today())
    
    target_name = TARGET_DICT[target_idx]
    benchmark_model = BENCHMARK_MODEL_BY_TARGET[target_idx]

    model_subset = ['AR1', benchmark_model] + base_models_subset

    r1_results_df = _compute_r1_results_df(
        target_idx=target_idx,
        targets_path=targets_path,
        pred_path=pred_path,
        horizon_in_quarters=horizon_in_quarters,
        quantiles=quantiles,
        test_start=test_start,
        test_end=test_end,
    )
    r1_results_df['Mean'] = r1_results_df.loc[:, quantiles].mean(axis=1)
    r1_results_df = r1_results_df.sort_values('Mean', ascending=False)
    r1_results_df.to_csv(
       results_dir / f"oos_r1_{country}_{horizon_in_quarters}q_{target_name}_{test_start}-{test_end}.csv", 
       index=False
    )

    # Make latex table
    r1_results_df = r1_results_df.set_index(['Model'])
    row_order = ['Mean'] + r1_results_df.columns[:-1].tolist()
    r1_report_df = r1_results_df.transpose().loc[row_order, model_subset]
    rename_rows_map = {k: f"Q{int(k*100)}" if k!='Mean' else 'Mean' for k in r1_report_df.index}
    r1_report_df.rename(index=rename_rows_map, inplace=True)
    r1_report_df.columns.name = None
    r1_report_df.index.name = None
    r1_report_df.to_latex(tables_dir / f"r1_{target_name}_{test_start}-{test_end}.tex", float_format="%.1f")
    
    print(f"R1 tables generation complete for {target_name}.")


def make_r1_multitarget_table_body(
    targets_path: Path,
    pred_dir: Path,
    results_dir: Path,
    tables_dir: Path,
    base_models_subset: list[str] = ['LR', 'LAS', 'QRF', 'QGB', 'DMQv0c', 'DMQv1c', 'DMQv2c'],
    country: str = 'us',
    horizon_in_quarters: int = 4,
    quantiles: list[float] = [0.05, 0.25, 0.50, 0.75, 0.95],
    test_start: str = '1998-01-01',
    test_end: str = '2024-12-01',
    date_str: str = None
):
    """Generate one combined multi-target R1 LaTeX table with tabular wrapper."""

    if date_str is None:
        date_str = str(date.today())

    target_order = [0, 1, 2]
    target_labels = {
        0: 'INFL',
        1: 'IP',
        2: 'UNRATE'
    }
    model_header_labels = ['AR1', 'Benchmark'] + base_models_subset

    body_lines = []
    row_labels = ['Mean'] + [f"Q{int(q * 100)}" for q in quantiles]
    n_rows_per_target = len(row_labels)

    for idx, target_idx in enumerate(target_order):
        target_name = TARGET_DICT[target_idx]
        benchmark_model = BENCHMARK_MODEL_BY_TARGET[target_idx]
        pred_path = pred_dir / f"all_models_predictions_{country}_{horizon_in_quarters}q_{target_name}.csv"

        r1_results_df = _compute_r1_results_df(
            target_idx=target_idx,
            targets_path=targets_path,
            pred_path=pred_path,
            horizon_in_quarters=horizon_in_quarters,
            quantiles=quantiles,
            test_start=test_start,
            test_end=test_end,
        )

        r1_results_df.to_csv(
            results_dir / f"oos_r1_{country}_{horizon_in_quarters}q_{target_name}_{test_start}-{test_end}.csv",
            index=False
        )

        report_df = r1_results_df.set_index('Model')
        ordered_models = ['AR1', benchmark_model] + base_models_subset

        ordered_rows = ['Mean'] + quantiles
        for row_i, row_key in enumerate(ordered_rows):
            first_col = f"\\multirow{{{n_rows_per_target}}}{{*}}{{{target_labels[target_idx]}}}" if row_i == 0 else ""
            metric_label = 'Mean' if row_key == 'Mean' else f"Q{int(row_key * 100)}"

            values = []
            for model in ordered_models:
                if model in report_df.index and row_key in report_df.columns:
                    values.append(_format_latex_cell(report_df.loc[model, row_key]))
                else:
                    values.append("--")

            line = f"{first_col} & {metric_label} & " + " & ".join(values) + r" \\" 
            body_lines.append(line)

        if idx < len(target_order) - 1:
            body_lines.append(r"\midrule")

    out_path = tables_dir / f"r1_combined_body_{country}_{horizon_in_quarters}q_{test_start}-{test_end}.tex"
    tabular_spec = "ll" + ("r" * len(model_header_labels))
    header_line = "Target & Metric & " + " & ".join(model_header_labels) + r" \\"
    wrapped_lines = [
        f"\\begin{{tabular}}{{{tabular_spec}}}",
        r"\toprule",
        header_line,
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
    ]

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(wrapped_lines) + "\n")

    print(f"Combined R1 table written to {out_path}")


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
    date_str: str = None
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
    date_str: str
        Date identifier for results directory
    """
    
    if date_str is None:
        date_str = str(date.today())
    
    target_name = TARGET_DICT[target_idx]
    benchmark_model = BENCHMARK_MODEL_BY_TARGET[target_idx]

    model_subset = ['AR1', benchmark_model] + base_models_subset

    # Create results dataframe and save CSV
    r2_report_df = _compute_r2_results_df(
        target_idx=target_idx,
        targets_path=targets_path,
        pred_path=pred_path,
        horizon_in_quarters=horizon_in_quarters,
        quantiles=quantiles,
        test_start=test_start,
        test_end=test_end,
    )
    r2_report_df.sort_values('R2', ascending=False).to_csv(
        results_dir / f"oos_r2_{country}_{horizon_in_quarters}q_{target_name}_{test_start}-{test_end}.csv", 
        index=False
    )

    # Make LaTeX table
    r2_report_df = r2_report_df.set_index('Model').transpose().loc[:, model_subset]
    r2_report_df.to_latex(tables_dir / f"r2_{target_name}_{test_start}-{test_end}.tex", float_format="%.1f")
    
    print(f"R2 tables generation complete for {target_name}.")


def make_r2_multitarget_table(
    targets_path: Path,
    pred_dir: Path,
    results_dir: Path,
    tables_dir: Path,
    base_models_subset: list[str] = ['LR', 'LAS', 'QRF', 'QGB', 'DMQv0c', 'DMQv1c', 'DMQv2c'],
    country: str = 'us',
    horizon_in_quarters: int = 4,
    quantiles: list[float] = [0.05, 0.25, 0.50, 0.75, 0.95],
    test_start: str = '1998-01-01',
    test_end: str = '2024-12-01',
    date_str: str = None,
):
    """Generate one combined multi-target R2 LaTeX table (targets as rows, models as columns)."""

    if date_str is None:
        date_str = str(date.today())

    target_order = [0, 1, 2]
    target_labels = {
        0: 'INFL',
        1: 'IP',
        2: 'UNRATE'
    }

    ordered_models = ['AR1', 'Benchmark'] + base_models_subset
    combined_rows = []

    for target_idx in target_order:
        target_name = TARGET_DICT[target_idx]
        benchmark_model = BENCHMARK_MODEL_BY_TARGET[target_idx]
        pred_path = pred_dir / f"all_models_predictions_{country}_{horizon_in_quarters}q_{target_name}.csv"

        r2_report_df = _compute_r2_results_df(
            target_idx=target_idx,
            targets_path=targets_path,
            pred_path=pred_path,
            horizon_in_quarters=horizon_in_quarters,
            quantiles=quantiles,
            test_start=test_start,
            test_end=test_end,
        )

        r2_report_df.sort_values('R2', ascending=False).to_csv(
            results_dir / f"oos_r2_{country}_{horizon_in_quarters}q_{target_name}_{test_start}-{test_end}.csv",
            index=False
        )

        row_dict = {'Target': target_labels[target_idx]}
        row_dict['AR1'] = r2_report_df.loc[r2_report_df['Model'] == 'AR1_Mean', 'R2'].iloc[0] if (r2_report_df['Model'] == 'AR1_Mean').any() else np.nan
        row_dict['Benchmark'] = r2_report_df.loc[r2_report_df['Model'] == benchmark_model, 'R2'].iloc[0] if (r2_report_df['Model'] == benchmark_model).any() else np.nan
        for model in base_models_subset:
            row_dict[model] = r2_report_df.loc[r2_report_df['Model'] == model, 'R2'].iloc[0] if (r2_report_df['Model'] == model).any() else np.nan

        combined_rows.append(row_dict)

    combined_df = pd.DataFrame(combined_rows).set_index('Target').loc[['INFL', 'IP', 'UNRATE'], ordered_models]
    combined_df.index.name = None

    out_path = tables_dir / f"r2_combined_{country}_{horizon_in_quarters}q_{test_start}-{test_end}.tex"
    combined_df.to_latex(out_path, float_format="%.1f")

    print(f"Combined R2 table written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate R1 and R2 evaluation tables")
    parser.add_argument("--target", type=int, default=0, help="Target index (0: Infl, 1: IP, 2: Unrate)")
    parser.add_argument("--country", type=str, default='us', help="Country code (us/ca)")
    parser.add_argument("--horizon", type=int, default=4, help="forecast horizon in quarters")
    parser.add_argument("--quantiles", type=float, nargs="*", default=[0.05, 0.25, 0.50, 0.75, 0.95], help="list of quantiles to predict")
    parser.add_argument("--test-start", type=str, default="1998-01-01", help="start date for the test set")
    parser.add_argument("--test-end", type=str, default="2024-12-01", help="end date for the test set")
    parser.add_argument("--date", type=str, default=None, help="date identifier for results directory")

    args = parser.parse_args()
    
    # Parse target file path and other configuration from config
    import yaml
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    TARGET_FILE = config['target_file']
    DATA_DIR = Path(config.get('data_dir', './data/'))
    targets_path = DATA_DIR / TARGET_FILE
    
    make_r1_tables(
        target_idx=args.target,
        targets_path=targets_path,
        pred_path=Path(f"predictions_{args.country}_{args.horizon}q.csv"),
        results_dir=Path("results/"),
        tables_dir=Path("results_tables/"),
        country=args.country,
        horizon_in_quarters=args.horizon,
        quantiles=args.quantiles,
        test_start=args.test_start,
        test_end=args.test_end,
        date_str=args.date
    )
    
    make_r2_tables(
        target_idx=args.target,
        targets_path=targets_path,
        pred_path=Path(f"predictions_{args.country}_{args.horizon}q.csv"),
        results_dir=Path("results/"),
        tables_dir=Path("results_tables/"),
        country=args.country,
        horizon_in_quarters=args.horizon,
        quantiles=args.quantiles,
        test_start=args.test_start,
        test_end=args.test_end,
        date_str=args.date
    )
