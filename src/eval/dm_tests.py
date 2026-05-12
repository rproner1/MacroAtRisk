"""Pairwise Diebold-Mariano forecast comparison utilities."""
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from src.train.losses import tilted_loss

TARGET_DICT = {
    0: "Infl_yoy",
    1: "IP_yoy",
    2: "Unrate_yoy",
}

BENCHMARK_MODEL_BY_TARGET = {
    0: "IAR",
    1: "VG",
    2: "UAR",
}

def _tilted_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> np.ndarray:
    """Compute tilted loss for quantile q."""
    e = y_true - y_pred
    return np.where(e >= 0, q * e, (q - 1) * e)

def get_quantile_weights(quantiles: list[float]) -> np.ndarray:
    """Compute quadrature-style weights from ordered quantile levels."""
    if len(quantiles) < 2:
        raise ValueError("Need at least two quantiles to compute weights.")

    weights = [0.5 * (quantiles[0] + quantiles[1])]
    for i in range(1, len(quantiles) - 1):
        weights.append(0.5 * (quantiles[i + 1] - quantiles[i - 1]))
    weights.append(1.0 - 0.5 * (quantiles[-1] + quantiles[-2]))
    return np.array(weights, dtype=float)


def _newey_west_lags(n_obs: int) -> int:
    """Automatic Newey-West lag length: floor(4*(T/100)^(2/9))."""
    if n_obs <= 0:
        return 0
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


def _dm_from_differential(differential: np.ndarray) -> tuple[float, float]:
    """Run DM regression d_t ~ const and return (t-stat, p-value)."""
    d = pd.Series(np.asarray(differential, dtype=float)).dropna()
    if d.shape[0] < 5:
        return np.nan, np.nan

    data = pd.DataFrame({"d": d.values})
    lags = _newey_west_lags(data.shape[0])
    fit = smf.ols("d ~ 1", data=data).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues["Intercept"]), float(fit.pvalues["Intercept"])


def diebold_mariano_test(
    y_true: np.ndarray,
    forecast_a: np.ndarray,
    forecast_b: np.ndarray,
) -> tuple[float, float]:
    """
    Compute DM t-statistic and two-sided p-value for squared-error loss.

    Statistic uses d_t = (e_a,t^2 - e_b,t^2). Positive t implies model B
    has lower loss than model A on average.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    fa = np.asarray(forecast_a, dtype=float).reshape(-1)
    fb = np.asarray(forecast_b, dtype=float).reshape(-1)

    if not (y.shape[0] == fa.shape[0] == fb.shape[0]):
        raise ValueError("y_true and forecasts must have the same length.")

    d = (y - fa) ** 2 - (y - fb) ** 2
    return _dm_from_differential(d)


def _build_model_mean_forecasts(
    preds: pd.DataFrame,
    benchmark_model: str,
    base_models_subset: list[str],
    quantiles: list[float],
) -> dict[str, np.ndarray]:
    """Construct mean forecasts for each model in display order."""
    weights = get_quantile_weights(quantiles)
    int_quantiles = [int(q * 100) for q in quantiles]

    model_forecasts: dict[str, np.ndarray] = {}

    if "AR1_Mean" not in preds.columns:
        raise ValueError("Expected AR1_Mean column in predictions file.")
    model_forecasts["AR1"] = preds["AR1_Mean"].values

    ordered_models = [benchmark_model] + base_models_subset
    for model in ordered_models:
        needed_cols = [f"{model}_Q{q}" for q in int_quantiles]
        missing_cols = [c for c in needed_cols if c not in preds.columns]
        if missing_cols:
            raise ValueError(f"Missing quantile columns for model {model}: {missing_cols}")

        q_preds = preds.loc[:, needed_cols].values
        model_forecasts[model] = (q_preds @ weights.reshape(-1, 1)).reshape(-1)

    return model_forecasts


def _build_model_quantile_forecasts(
    preds: pd.DataFrame,
    benchmark_model: str,
    base_models_subset: list[str],
    quantiles: list[float],
) -> dict[str, np.ndarray]:
    """Construct quantile forecast matrices (T x Q) for each model."""
    int_quantiles = [int(q * 100) for q in quantiles]
    ordered_models = [benchmark_model] + base_models_subset

    model_q_forecasts: dict[str, np.ndarray] = {}

    # AR1 fallback for quantile-based DM tests: if AR1 quantiles are unavailable,
    # replicate AR1_Mean across quantiles so AR1 remains in pairwise matrices.
    ar1_q_cols = [f"AR1_Q{q}" for q in int_quantiles]
    has_ar1_q = all(col in preds.columns for col in ar1_q_cols)
    if has_ar1_q:
        model_q_forecasts["AR1"] = preds.loc[:, ar1_q_cols].values
    elif "AR1_Mean" in preds.columns:
        ar1_mean = preds["AR1_Mean"].values.reshape(-1, 1)
        model_q_forecasts["AR1"] = np.repeat(ar1_mean, repeats=len(int_quantiles), axis=1)
    else:
        raise ValueError("Expected either AR1 quantile columns or AR1_Mean in predictions file.")

    for model in ordered_models:
        needed_cols = [f"{model}_Q{q}" for q in int_quantiles]
        missing_cols = [c for c in needed_cols if c not in preds.columns]
        if missing_cols:
            raise ValueError(f"Missing quantile columns for model {model}: {missing_cols}")
        model_q_forecasts[model] = preds.loc[:, needed_cols].values

    return model_q_forecasts


def compute_pairwise_dm_matrices(
    y_true: np.ndarray,
    model_forecasts: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return pairwise DM t-stats and p-values for all model pairs."""
    models = list(model_forecasts.keys())
    t_stats = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    p_vals = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)

    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i == j:
                t_stats.loc[model_i, model_j] = 0.0
                p_vals.loc[model_i, model_j] = np.nan
                continue

            t_stat, p_val = diebold_mariano_test(
                y_true=y_true,
                forecast_a=model_forecasts[model_i],
                forecast_b=model_forecasts[model_j],
            )
            t_stats.loc[model_i, model_j] = t_stat
            p_vals.loc[model_i, model_j] = p_val

    return t_stats, p_vals


def compute_pairwise_dm_matrices_per_quantile(
    y_true: np.ndarray,
    model_quantile_forecasts: dict[str, np.ndarray],
    quantiles: list[float],
) -> dict[float, tuple[pd.DataFrame, pd.DataFrame]]:
    """Return pairwise DM matrices for each quantile using pinball loss."""
    models = list(model_quantile_forecasts.keys())
    y = np.asarray(y_true, dtype=float).reshape(-1)
    out: dict[float, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for q_idx, q in enumerate(quantiles):
        t_stats = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
        p_vals = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)

        for i, model_i in enumerate(models):
            for j, model_j in enumerate(models):
                if i == j:
                    t_stats.loc[model_i, model_j] = 0.0
                    p_vals.loc[model_i, model_j] = np.nan
                    continue

                f_i = np.asarray(model_quantile_forecasts[model_i][:, q_idx], dtype=float)
                f_j = np.asarray(model_quantile_forecasts[model_j][:, q_idx], dtype=float)
                d = _tilted_loss(y, f_i, q=q) - _tilted_loss(y, f_j, q=q)
                t_stat, p_val = _dm_from_differential(d)
                t_stats.loc[model_i, model_j] = t_stat
                p_vals.loc[model_i, model_j] = p_val

        out[q] = (t_stats, p_vals)

    return out


def compute_pairwise_dm_matrices_quantile_pooled(
    y_true: np.ndarray,
    model_quantile_forecasts: dict[str, np.ndarray],
    quantiles: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return pairwise DM matrices pooled across quantiles (per target)."""
    models = list(model_quantile_forecasts.keys())
    y = np.asarray(y_true, dtype=float).reshape(-1)

    t_stats = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    p_vals = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)

    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i == j:
                t_stats.loc[model_i, model_j] = 0.0
                p_vals.loc[model_i, model_j] = np.nan
                continue

            q_diffs = []
            for q_idx, q in enumerate(quantiles):
                f_i = np.asarray(model_quantile_forecasts[model_i][:, q_idx], dtype=float)
                f_j = np.asarray(model_quantile_forecasts[model_j][:, q_idx], dtype=float)
                q_diffs.append(_tilted_loss(y, f_i, q=q) - _tilted_loss(y, f_j, q=q))

            d_pooled = np.mean(np.column_stack(q_diffs), axis=1)
            # print(f"Quantile-pooled DM differential for {model_i} vs {model_j}:", d_pooled) # One element vector resulting in nans
            t_stat, p_val = _dm_from_differential(d_pooled)
            t_stats.loc[model_i, model_j] = t_stat
            p_vals.loc[model_i, model_j] = p_val

    return t_stats, p_vals


def _format_dm_cell(t_stat: float, p_val: float, alpha: float) -> str:
    if not np.isfinite(t_stat):
        return "--"
    text = f"{t_stat:.2f}"
    if np.isfinite(p_val) and p_val < alpha:
        return f"\\textbf{{{text}}}"
    return text


def write_upper_triangular_dm_table(
    t_stats: pd.DataFrame,
    p_vals: pd.DataFrame,
    out_path: Path,
    alpha: float = 0.05,
) -> None:
    """Write an upper-triangular LaTeX table with significant entries in bold."""
    models = list(t_stats.columns)
    tabular_spec = "l" + ("r" * len(models))

    lines = [
        f"\\begin{{tabular}}{{{tabular_spec}}}",
        "\\toprule",
        " & " + " & ".join(models) + r" \\",
        "\\midrule",
    ]

    for i, row_model in enumerate(models):
        row_cells = [row_model]
        for j, col_model in enumerate(models):
            if j <= i:
                row_cells.append("--" if j == i else "")
            else:
                row_cells.append(
                    _format_dm_cell(
                        t_stat=t_stats.loc[row_model, col_model],
                        p_val=p_vals.loc[row_model, col_model],
                        alpha=alpha,
                    )
                )
        lines.append(" & ".join(row_cells) + r" \\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def make_dm_tables(
    targets_path: Path,
    pred_dir: Path,
    results_dir: Path,
    tables_dir: Path,
    base_models_subset: list[str] = ["LR", "LAS", "QRF", "QGB", "DMQv0c", "DMQv1c", "DMQv2c"],
    country: str = "us",
    horizon_in_quarters: int = 4,
    quantiles: list[float] = [0.05, 0.25, 0.50, 0.75, 0.95],
    test_start: str = "1998-01-01",
    test_end: str = "2024-12-01",
    alpha: float = 0.05,
    date_str: str = None,
) -> None:
    """
    Generate pairwise DM tables for all targets.

    Outputs three DM families:
    1) Mean DM (means aggregated from quantiles)
    2) Quantile-specific DM (one matrix per quantile)
    3) Quantile-pooled DM (pooling differential across quantiles per target)

    Saves upper-triangular LaTeX tables and CSV matrices of t-stats/p-values.
    """
    if date_str is None:
        date_str = str(date.today())

    y_full = pd.read_parquet(targets_path).loc["1961-01-01":"2024-12-01", :]

    for target_idx in [0, 1, 2]:
        target_name = TARGET_DICT[target_idx]
        benchmark_model = BENCHMARK_MODEL_BY_TARGET[target_idx]

        pred_path = pred_dir / f"all_models_predictions_{country}_{horizon_in_quarters}q_{target_name}.csv"
        preds = pd.read_csv(pred_path, index_col=0, parse_dates=True).loc[test_start:test_end]
        actuals = y_full.loc[test_start:test_end, target_name].values.reshape(-1)

        model_forecasts = _build_model_mean_forecasts(
            preds=preds,
            benchmark_model=benchmark_model,
            base_models_subset=base_models_subset,
            quantiles=quantiles,
        )
        model_q_forecasts = _build_model_quantile_forecasts(
            preds=preds,
            benchmark_model=benchmark_model,
            base_models_subset=base_models_subset,
            quantiles=quantiles,
        )

        ordered_keys = ["AR1", benchmark_model] + base_models_subset
        ordered_forecasts = {k: model_forecasts[k] for k in ordered_keys}
        ordered_q_forecasts = {k: model_q_forecasts[k] for k in ordered_keys}

        # 1) Mean DM
        # print(actuals, ordered_forecasts)
        t_stats, p_vals = compute_pairwise_dm_matrices(
            y_true=actuals,
            model_forecasts=ordered_forecasts,
        )

        out_tex_mean = tables_dir / f"dm_mean_upper_{target_name}_{country}_{horizon_in_quarters}q_{test_start}-{test_end}.tex"
        write_upper_triangular_dm_table(
            t_stats=t_stats,
            p_vals=p_vals,
            out_path=out_tex_mean,
            alpha=alpha,
        )

        # 2) Quantile-specific DM
        q_mats = compute_pairwise_dm_matrices_per_quantile(
            y_true=actuals,
            model_quantile_forecasts=ordered_q_forecasts,
            quantiles=quantiles,
        )
        for q in quantiles:
            q_int = int(round(q * 100))
            qt, qp = q_mats[q]

            out_q = tables_dir / f"dm_quantile_upper_{target_name}_Q{q_int}_{country}_{horizon_in_quarters}q_{test_start}-{test_end}.tex"
            write_upper_triangular_dm_table(
                t_stats=qt,
                p_vals=qp,
                out_path=out_q,
                alpha=alpha,
            )

        # 3) Quantile-pooled DM
        t_pool, p_pool = compute_pairwise_dm_matrices_quantile_pooled(
            y_true=actuals,
            model_quantile_forecasts=ordered_q_forecasts,
            quantiles=quantiles,
        )

        out_pool = tables_dir / f"dm_qpooled_upper_{target_name}_{country}_{horizon_in_quarters}q_{test_start}-{test_end}.tex"
        write_upper_triangular_dm_table(
            t_stats=t_pool,
            p_vals=p_pool,
            out_path=out_pool,
            alpha=alpha,
        )

        print(f"DM table written to {out_pool}")
