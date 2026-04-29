import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path    
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import SmoothBivariateSpline
from typing import Optional, List, Tuple


def _shade_points_by_zscore(base_color, z_values):
    """Return shades of base_color from light (low z) to dark (high z)."""
    z = np.asarray(z_values, dtype=float)
    z = np.clip(z, -2.5, 2.5)
    t = (z + 2.5) / 5.0

    r, g, b = base_color[:3]
    white_mix = 0.75 * (1.0 - t)
    dark_mix = 0.45 * t

    rr = r * (1.0 - white_mix) + 1.0 * white_mix
    gg = g * (1.0 - white_mix) + 1.0 * white_mix
    bb = b * (1.0 - white_mix) + 1.0 * white_mix

    rr = rr * (1.0 - dark_mix)
    gg = gg * (1.0 - dark_mix)
    bb = bb * (1.0 - dark_mix)

    return np.column_stack([rr, gg, bb])


def _aggregate_lag_tensor(arr, method: str = "last"):
    """
    Aggregate a (n_samples, n_lags, n_features) tensor over lag axis.

    Methods:
      - "last": use most recent lag (recommended for calendar-time views)
      - "mean": arithmetic mean over lags
      - "sum": sum over lags
      - "abs_mean": mean of absolute values over lags
      - "abs_sum": sum of absolute values over lags
    """
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"Expected 3D tensor (n,t,f), got shape {a.shape}")

    method = str(method).lower().strip()
    if method == "last":
        return a[:, -1, :]
    if method == "mean":
        return np.mean(a, axis=1)
    if method == "sum":
        return np.sum(a, axis=1)
    if method == "abs_mean":
        return np.mean(np.abs(a), axis=1)
    if method == "abs_sum":
        return np.sum(np.abs(a), axis=1)

    raise ValueError(
        f"Unknown aggregation method '{method}'. "
        "Use one of: last, mean, sum, abs_mean, abs_sum."
    )


def _add_macro_recession_shading(ax, idx):
    """Add standard recession bands used in existing macro plots."""
    if len(idx) == 0:
        return
    start = pd.Timestamp(min(idx))
    end = pd.Timestamp(max(idx))

    for s, e in [
        (pd.Timestamp("2001-03-01"), pd.Timestamp("2001-11-01")),
        (pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01")),
        (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-01")),
    ]:
        if e < start or s > end:
            continue
        ax.axvspan(max(s, start), min(e, end), color="grey", alpha=0.2, zorder=0)

def make_overall_importance_plot_agg_time_lags(
    sv_q,
    model_features,
    q,
    target_name,
    model_to_explain,
    fig_dir,
    top_n: int = 25,
    global_imp_func: str="median",
):
    """
    Plot overall feature importance using mean absolute SHAP over samples and lags.

    For sv_q with shape (n_samples, n_lags, n_features), importance is:
    mean_{samples,lags}(|SHAP|) for each feature.
    """
    sv_q = np.asarray(sv_q)
    if sv_q.ndim != 3:
        raise ValueError(f"Expected sv_q with shape (n, t, f), got {sv_q.shape}")

    _, _, n_features = sv_q.shape
    if n_features != len(model_features):
        raise ValueError(
            f"Feature count mismatch: sv_q has {n_features}, model_features has {len(model_features)}"
        )

    # Sum over lags, then average over samples -> (n_features,)
    if global_imp_func == "mean":
        overall_imp = np.mean(np.abs(np.sum(sv_q, axis=1)), axis=0)
    elif global_imp_func == "median":
        overall_imp = np.median(np.abs(np.sum(sv_q, axis=1)), axis=0)
    imp_s = pd.Series(overall_imp, index=model_features).sort_values(ascending=False)

    top_n = min(top_n, n_features)
    imp_top = imp_s.head(top_n).sort_values(ascending=True)

    fig_h = max(6, 0.30 * top_n)
    plt.figure(figsize=(10.5, fig_h))
    plt.barh(imp_top.index, imp_top.values, color="#4f6d7a")
    q_label = int(round(float(q) * 100))
    # plt.title(f"Overall SHAP Importance")
    plt.xlabel(f"{global_imp_func.capitalize()} |EG| value")
    plt.ylabel("Feature")
    plt.tight_layout()

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"overall_importance_{model_to_explain}_{target_name}_q{q_label}.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def make_feature_x_attribution_by_regime_plot(
    sv_q,
    x_q,
    model_features,
    q,
    target_name,
    model_to_explain,
    fig_dir,
    top_k: int = 5,
    smooth_window: int = 12,
    feature_value_color: bool = True,
    contrib_agg: str = "sum",
    value_agg: str = "last",
    rank_agg: str = "abs_mean",
    global_imp_func: str = "mean",
    z_clip: float = 2.5,
    values_already_standardized: bool = True,
):
    if isinstance(q, float):
        q_label = int(q*100)
    else:
        q_label = q

    # Get top 5 features 
    if global_imp_func == "mean":
        global_imp = np.mean(np.abs(np.sum(sv_q, axis=1)), axis=0)
    elif global_imp_func == "median":
        global_imp = np.median(np.abs(np.sum(sv_q, axis=1)), axis=0)
    top_k = min(top_k, sv_q.shape[2])
    top_idx = np.argsort(global_imp)[-top_k:][::-1]
    top_features = [model_features[i] for i in top_idx]
    features_dict = {model_features[i]: i for i in top_idx}

    if x_q.ndim == 3:
        x_collapsed = _aggregate_lag_tensor(x_q, method=value_agg)
    elif x_q.ndim == 2:
        x_collapsed = x_q
    
    # infl_idx = features_dict['CPIAUCSL']
    # ip_idx = features_dict['INDPRO']
    # fedfunds_idx = features_dict['FEDFUNDS']
    # infl_t = x_collapsed[:,infl_idx]
    # ip_t = x_collapsed[:,ip_idx]
    # fed_funds_t = x_collapsed[:, fedfunds_idx]


    for feat_idx in top_idx:
        feat_name = model_features[feat_idx]
        feat_values = x_q[:, -1, feat_idx]  # Use last lag for feature values
        feat_contrib = np.sum(sv_q, axis=1)[:, feat_idx]  # Summing contributions across all lags gives overall contribution for the feature at each time point

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(
            feat_values,
            feat_contrib,
        )
        ax.set_xlabel(feat_name)
        ax.set_ylabel("Attribution")

        plt.tight_layout()
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        style_suffix = '_valuecolor' if feature_value_color else ''
        out_path = fig_dir / f"{model_to_explain}_{target_name}_{feat_name}_value_attribution_q{q_label}.png"
        plt.savefig(out_path, bbox_inches='tight', dpi=220)
        plt.close()


def make_top_feature_contrib_timeseries_plot(
    sv_q,
    x_q,
    model_features,
    q,
    target_name,
    model_to_explain,
    fig_dir,
    time_index,
    top_k: int = 5,
    smooth_window: int = 12,
    feature_value_color: bool = True,
    contrib_agg: str = "sum",
    value_agg: str = "last",
    rank_agg: str = "sum_abs",
    global_imp_func: str = "median",
    z_clip: float = 2.5,
    values_already_standardized: bool = True,
    low_value_marker: str = "v",
    high_value_marker: str = "^",
    value_split: float = 0.0,
):
    """Factor_risk-style top-k SHAP contribution time-series for one quantile.

    Args:
        global_imp_func:
            The function to use for computing global feature importance. Options: "mean" or "median"
        values_already_standardized:
            If True, use aggregated feature values directly for color shading
            (clipped to [-z_clip, z_clip]). If False, compute per-feature z-score
            within the plotted sample window before shading.
    """
    sv_q = np.asarray(sv_q)
    x_q = np.asarray(x_q)
    idx = pd.DatetimeIndex(time_index)

    if sv_q.ndim != 3 or x_q.ndim != 3:
        raise ValueError(f"Expected (n,t,f) tensors, got sv_q={sv_q.shape}, x_q={x_q.shape}")
    if sv_q.shape != x_q.shape:
        raise ValueError(f"Shape mismatch: sv_q {sv_q.shape} vs x_q {x_q.shape}")
    if sv_q.shape[0] != len(idx):
        raise ValueError(f"Index length mismatch: {len(idx)} vs n_samples={sv_q.shape[0]}")

    # Aggregate lag dimension for plotting series.
    # For RNNs, using "last" generally aligns best with calendar-time interpretation.
    sv_collapsed = _aggregate_lag_tensor(sv_q, method=contrib_agg)  # (n, f)
    x_collapsed = _aggregate_lag_tensor(x_q, method=value_agg)      # (n, f)

    if global_imp_func == "mean":
        func = np.mean
    elif global_imp_func == "median":   
        func = np.median
    else:
        raise ValueError(f"Unknown global_imp_func '{global_imp_func}'. Use 'mean' or 'median'.")

    # Rank top features without signed-cancellation artifacts.
    rank_agg = str(rank_agg).lower().strip()
    if rank_agg == "sum_abs":
        global_imp = func(np.abs(np.sum(sv_q, axis=1)), axis=0)
    elif rank_agg == "last_abs":
        global_imp = func(np.abs(sv_q[:, -1, :]), axis=0)
    else:
        raise ValueError(
            f"Unknown rank_agg '{rank_agg}'. "
            "Use one of: sum_abs, last_abs."
        )

    top_k = min(top_k, sv_collapsed.shape[1])
    top_idx = np.argsort(global_imp)[-top_k:][::-1]

    plt.figure(figsize=(12, 6))
    tab_colors = plt.get_cmap("tab10").colors
    legend_handles = []

    for rank, feat_idx in enumerate(top_idx):
        feat_name = model_features[feat_idx]
        base_color = tab_colors[rank % len(tab_colors)]

        y_series = pd.Series(sv_collapsed[:, feat_idx], index=idx)
        y_plot = y_series.rolling(window=max(1, smooth_window), min_periods=1).mean() if smooth_window > 1 else y_series

        legend_handles.append(
            Line2D(
                [0], [0],
                color=base_color,
                lw=3.2,
                alpha=1.0,
                marker='o',
                markersize=6,
                markerfacecolor=base_color,
                markeredgecolor=base_color,
                label=feat_name,
            )
        )

        if feature_value_color:
            x_series = pd.Series(x_collapsed[:, feat_idx], index=idx).astype(float)
            if values_already_standardized:
                x_z = x_series.copy()
            else:
                x_mu = float(np.nanmean(x_series.values))
                x_sigma = float(np.nanstd(x_series.values))
                x_z = (x_series - x_mu) / x_sigma if x_sigma > 0 else pd.Series(0.0, index=idx)
            x_z = x_z.clip(lower=-abs(float(z_clip)), upper=abs(float(z_clip)))
            plt.plot(y_plot.index, y_plot.values, linewidth=0.8, alpha=0.30, color=base_color)
            point_colors = _shade_points_by_zscore(base_color, x_z.values)

            low_mask = x_z.values < float(value_split)
            high_mask = ~low_mask

            if np.any(low_mask):
                plt.scatter(
                    y_plot.index[low_mask],
                    y_plot.values[low_mask],
                    color=point_colors[low_mask],
                    marker=low_value_marker,
                    s=20,
                    alpha=0.9,
                    edgecolors='none',
                )
            if np.any(high_mask):
                plt.scatter(
                    y_plot.index[high_mask],
                    y_plot.values[high_mask],
                    color=point_colors[high_mask],
                    marker=high_value_marker,
                    s=20,
                    alpha=0.9,
                    edgecolors='none',
                )
        else:
            plt.plot(y_plot.index, y_plot.values, linewidth=1.0, color=base_color)

    ax = plt.gca()
    _add_macro_recession_shading(ax, idx)
    plt.axhline(y=0.0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel('Date')
    plt.ylabel('EG attribution')

    q_label = int(round(float(q) * 100))
    plt.title(f"Top {top_k} Feature Attributions Over Time - {model_to_explain} {target_name} Q{q_label}")
    if feature_value_color:
        legend_handles.extend([
            Line2D(
                [0], [0],
                linestyle='none',
                marker=low_value_marker,
                markersize=7,
                markerfacecolor='0.4',
                markeredgecolor='0.4',
                label=f'Low value (< {value_split:g})',
            ),
            Line2D(
                [0], [0],
                linestyle='none',
                marker=high_value_marker,
                markersize=7,
                markerfacecolor='0.4',
                markeredgecolor='0.4',
                label=f'High value (>= {value_split:g})',
            ),
        ])

    legend_title = 'Feature (light=low, dark=high)' if feature_value_color else None
    plt.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        ncol=1,
        title=legend_title,
        frameon=True,
        handlelength=3.2,
        handletextpad=0.8,
        borderpad=0.6,
    )
    plt.tight_layout()

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    style_suffix = '_valuecolor' if feature_value_color else ''
    out_path = fig_dir / f"contrib_timeseries_{model_to_explain}_{target_name}_top{top_k}_q{q_label}_ma{smooth_window}{style_suffix}.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=220)
    plt.close()


def make_feat_x_time_importance_plot(
    sv_q,
    model_features,
    q,
    target_name,
    fig_dir,
    top_n: int = 20,
    global_imp_func: str = "mean",
):
    """
    Create a feature importance plot that shows the importance of the top features at each time step for an LSTM/GRU model changes over time.

    Args:
        sv_q: SHAP values for quantile q, expected shape (n_samples, time_steps, n_features)
        model_features: List of feature names corresponding to the n_features dimension
        q: The quantile for which the SHAP values are computed
        target_name: Name of the target variable (for labeling)
        fig_dir: Directory to save the figure
    """
    sv_q = np.asarray(sv_q)
    if sv_q.ndim != 3:
        raise ValueError(f"Expected sv_q with 3 dimensions (n, t, f), got shape {sv_q.shape}")

    n_samples, n_lags, n_features = sv_q.shape
    if n_features != len(model_features):
        raise ValueError(
            f"Feature count mismatch: sv_q has {n_features} features, "
            f"but model_features has {len(model_features)}"
        )

    # Select the most imortant features by average absolute SHAP across samples and lags.
    if global_imp_func == "mean":
        overall_feat_imp = np.mean(np.abs(np.sum(sv_q, axis=1)), axis=0)  # (n_features,)
    elif global_imp_func == "median":
        overall_feat_imp = np.median(np.abs(np.sum(sv_q, axis=1)), axis=0)  # (n_features,)
    top_features = np.argsort(overall_feat_imp)[::-1][:top_n]

    # Average absolute SHAP over samples -> (lag, feature)
    lag_feature_imp = np.mean(np.abs(sv_q), axis=0)
    lag_labels = list(range(-n_lags, 0))

    # Build frame with rows=lag and columns=feature, then rank within each lag.
    imp_df = pd.DataFrame(lag_feature_imp, index=lag_labels, columns=model_features)
    rank_df = imp_df.rank(axis=1, ascending=False, method="dense")

    # Aggregate rank across lags (lower sum = more important), then take top features.
    rank_sum = rank_df.sum(axis=0).sort_values(ascending=True)
    top_features = rank_sum.head(min(top_n, n_features)).index.tolist()

    # Heatmap matrix with rows=features and columns=lags.
    heatmap_df = rank_df.loc[:, top_features].T
    heatmap_df = heatmap_df.loc[top_features]

    fig_w = max(12, 0.8 * n_lags)
    fig_h = max(8, 0.4 * len(heatmap_df.index))
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        heatmap_df,
        cmap="Greys_r",
        vmin=1,
        vmax=float(heatmap_df.to_numpy().max()),
        cbar_kws={"label": "Rank (1 = most important)"},
    )

    ax.set_title(f"Feature Rank Heatmap Over Time: {target_name}, q={q}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Feature")
    ax.set_xticklabels([str(int(x)) for x in heatmap_df.columns], rotation=0)
    ax.tick_params(axis='y', labelsize=10)  # Adjust font size for y-axis labels
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"feature_rank_heatmap_{target_name}_q{q}.png", dpi=200)
    plt.close()


def make_top_feature_time_heatmap_agg_lags(
    sv_q,
    model_features,
    q,
    target_name,
    fig_dir,
    time_index,
    top_n: int = 10,
):
    """
    Plot a feature-by-time heatmap of predictor ranks after aggregating across lags.

    - Aggregate within each date by averaging |SHAP| across lags.
    - Convert feature influence to cross-feature rank at each date (1 = most influential).
    - Select top features by best average rank over time.
    - Render heat map with rows=features and columns=time.
    """
    sv_q = np.asarray(sv_q)
    idx = pd.DatetimeIndex(time_index)

    if sv_q.ndim != 3:
        raise ValueError(f"Expected sv_q with shape (n, t, f), got {sv_q.shape}")

    n_samples, _, n_features = sv_q.shape
    if n_features != len(model_features):
        raise ValueError(
            f"Feature count mismatch: sv_q has {n_features}, model_features has {len(model_features)}"
        )
    if len(idx) != n_samples:
        raise ValueError(f"Index length mismatch: len(idx)={len(idx)} vs n_samples={n_samples}")

    # Aggregate across lags for each time point -> (n_samples, n_features)
    time_feature_imp = np.mean(np.abs(sv_q), axis=1)

    rank_df_all = pd.DataFrame(time_feature_imp, index=idx, columns=model_features)
    rank_df_all = rank_df_all.rank(axis=1, ascending=False, method="dense")

    avg_rank = rank_df_all.mean(axis=0)
    top_n = min(top_n, n_features)
    top_features = avg_rank.sort_values(ascending=True).head(top_n).index.tolist()

    heatmap_df = pd.DataFrame(
        rank_df_all.loc[:, top_features].T,
        index=top_features,
        columns=idx,
    )

    # Keep x-axis readable by showing annual ticks.
    year_starts = [i for i, ts in enumerate(idx) if ts.month == 1]
    year_labels = [str(idx[i].year) for i in year_starts]

    fig_w = max(14, len(idx) / 9)
    fig_h = max(6, 0.45 * top_n)
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        heatmap_df,
        cmap="Greys_r",
        vmin=1,
        vmax=float(heatmap_df.to_numpy().max()),
        cbar_kws={"label": "Rank (1 = most influential)"},
    )

    q_label = int(round(float(q) * 100))
    ax.set_title(f"Top {top_n} Predictor Ranks Over Time (Lag-Aggregated) - {target_name}, Q{q_label}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Feature")
    ax.tick_params(axis='y', labelsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    if year_starts:
        ax.set_xticks(year_starts)
        ax.set_xticklabels(year_labels, rotation=45, ha='right')

    plt.tight_layout()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"top{top_n}_predictor_rank_heatmap_time_agg_lags_{target_name}_q{q_label}.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def make_event_force_plot(
    sv_event,
    model_features,
    event_date,
    event_name,
    q,
    target_name,
    fig_dir,
    top_n: int = 20,
):
    """
    Create a force-style plot for a single event using lagged SHAP contributions.

    Args:
        sv_event: SHAP values for one event sample, shape (n_lags, n_features)
        model_features: List of feature names
        event_date: Timestamp/date for the selected event sample
        event_name: Label for the event
        q: Quantile label
        target_name: Target variable name
        fig_dir: Output directory
        top_n: Number of top features to plot
    """
    sv_event = np.asarray(sv_event)
    if sv_event.ndim != 2:
        raise ValueError(f"Expected sv_event with shape (t, f), got {sv_event.shape}")

    n_lags, n_features = sv_event.shape
    if n_features != len(model_features):
        raise ValueError(
            f"Feature count mismatch: sv_event has {n_features}, model_features has {len(model_features)}"
        )

    # Pick top features based on event-specific absolute contribution summed over lags.
    total_abs = np.sum(np.abs(sv_event), axis=0)
    top_idx = np.argsort(total_abs)[::-1][: min(top_n, n_features)]
    top_features = [model_features[i] for i in top_idx]
    top_contrib = sv_event[:, top_idx]

    lags = np.arange(-n_lags, 0)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(top_features)))

    fig_w = max(11, 0.75 * n_lags)
    fig_h = 8
    plt.figure(figsize=(fig_w, fig_h))

    pos_bottom = np.zeros(n_lags)
    neg_bottom = np.zeros(n_lags)

    for i, feat in enumerate(top_features):
        vals = top_contrib[:, i]
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)

        plt.bar(lags, pos, bottom=pos_bottom, width=0.8, color=colors[i], label=feat)
        plt.bar(lags, neg, bottom=neg_bottom, width=0.8, color=colors[i])

        pos_bottom = pos_bottom + pos
        neg_bottom = neg_bottom + neg

    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.title(
        f"Event Force Plot ({event_name}) for {target_name}, q={q}\n"
        f"Event date: {pd.Timestamp(event_date).date()}"
    )
    plt.xlabel("Lag")
    plt.ylabel("EG attribution")
    plt.xticks(lags)
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    safe_event = str(event_name).strip().lower().replace(" ", "_")
    plt.savefig(
        fig_dir / f"event_force_plot_{safe_event}_{target_name}_q{q}.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()


def make_pair_value_vs_attribution_cubic_spline_3d_plot(
    sv_q,
    x_q,
    model_features,
    q,
    target_name,
    model_to_explain,
    fig_dir,
    feature_x,
    feature_y,
    value_agg: str = "last",
    contrib_agg: str = "sum",
    spline_s: Optional[float] = None,
    grid_size: int = 40,
    point_alpha: float = 0.35,
    time_index=None,
    event_windows: Optional[List[Tuple[str, str, str]]] = None,
    annotate_event_points: bool = True,
    event_point_size: float = 38.0,
    event_label_fontsize: float = 7.0,
    make_interactive_html: bool = False,
    html_include_plotlyjs: str = "cdn",
    highlight_n: int = 0,
    highlight_method: str = "abs",
    plot_scatter: bool = True,
):
    """Plot a 3D cubic spline surface of pairwise feature effects on attribution.

    The x- and y-axes are feature values, and z-axis is the combined attribution
    of the selected pair (feature_x attribution + feature_y attribution).
    """
    sv_q = np.asarray(sv_q)
    x_q = np.asarray(x_q)

    if sv_q.ndim != 3 or x_q.ndim != 3:
        raise ValueError(f"Expected (n,t,f) tensors, got sv_q={sv_q.shape}, x_q={x_q.shape}")
    if sv_q.shape != x_q.shape:
        raise ValueError(f"Shape mismatch: sv_q {sv_q.shape} vs x_q {x_q.shape}")
    if sv_q.shape[2] != len(model_features):
        raise ValueError(
            f"Feature count mismatch: tensor has {sv_q.shape[2]} features, model_features has {len(model_features)}"
        )
    if time_index is not None and len(time_index) != sv_q.shape[0]:
        raise ValueError(
            f"Index length mismatch: len(time_index)={len(time_index)} vs n_samples={sv_q.shape[0]}"
        )

    model_features_norm = [str(f).strip().lower() for f in model_features]

    def _resolve_feature_idx(feature_name: str) -> int:
        needle = str(feature_name).strip().lower()
        exact = [i for i, n in enumerate(model_features_norm) if n == needle]
        if exact:
            return exact[0]
        contains = [i for i, n in enumerate(model_features_norm) if needle in n]
        if contains:
            return contains[0]
        raise ValueError(f"Feature '{feature_name}' not found in model_features")

    ix = _resolve_feature_idx(feature_x)
    iy = _resolve_feature_idx(feature_y)

    x_vals = _aggregate_lag_tensor(x_q, method=value_agg)[:, ix]
    y_vals = _aggregate_lag_tensor(x_q, method=value_agg)[:, iy]

    sv_collapsed = _aggregate_lag_tensor(sv_q, method=contrib_agg)
    z_vals = sv_collapsed[:, ix] + sv_collapsed[:, iy]

    valid = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(z_vals)

    if time_index is None:
        time_vals = pd.date_range(start="1900-01-01", periods=sv_q.shape[0], freq="MS")
    else:
        time_vals = pd.DatetimeIndex(time_index)
    time_vals = time_vals[valid]

    x_vals = x_vals[valid]
    y_vals = y_vals[valid]
    z_vals = z_vals[valid]

    min_points = 25
    if len(x_vals) < min_points:
        raise ValueError(f"Need at least {min_points} finite points for cubic spline fit, got {len(x_vals)}")

    x_lo, x_hi = np.quantile(x_vals, [0.01, 0.99])
    y_lo, y_hi = np.quantile(y_vals, [0.01, 0.99])

    spline = SmoothBivariateSpline(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        kx=3,
        ky=3,
        s=spline_s,
    )

    gx = np.linspace(x_lo, x_hi, int(max(15, grid_size)))
    gy = np.linspace(y_lo, y_hi, int(max(15, grid_size)))
    xx, yy = np.meshgrid(gx, gy)
    zz = spline.ev(xx.ravel(), yy.ravel()).reshape(xx.shape)

    z_max = np.max(z_vals)
    z_min = np.min(z_vals)
    zz = np.clip(zz, z_min - 0.1 * abs(z_min), z_max + 0.1 * abs(z_max))

    z_all = np.concatenate([z_vals.ravel(), zz.ravel()])
    z_abs_max = float(np.nanmax(np.abs(z_all))) if z_all.size else 0.0
    if not np.isfinite(z_abs_max) or z_abs_max <= 0:
        z_abs_max = 1e-9
    zero_center_norm = TwoSlopeNorm(vmin=-z_abs_max, vcenter=0.0, vmax=z_abs_max)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        xx,
        yy,
        zz,
        cmap="coolwarm",
        norm=zero_center_norm,
        linewidth=0,
        antialiased=True,
        alpha=0.85,
    )
    if plot_scatter:
        ax.scatter(
            x_vals,
            y_vals,
            z_vals,
            c=z_vals,
            cmap="coolwarm",
            norm=zero_center_norm,
            s=10,
            alpha=float(point_alpha),
            depthshade=True,
        )

    feat_x_label = str(model_features[ix]).strip()
    feat_y_label = str(model_features[iy]).strip()
    q_label = int(round(float(q) * 100)) if isinstance(q, (float, np.floating)) else int(q)

    ax.set_xlabel(f"{feat_x_label} value")
    ax.set_ylabel(f"{feat_y_label} value")
    ax.set_zlabel("Combined attribution")
    ax.set_title(f"3D cubic spline - {model_to_explain} {target_name} q={q_label}")

    # Plot only selected historical event points and annotate each with date.
    if event_windows is None:
        event_windows = [
            # ("DotCom", "2000-03-01", "2002-12-01"),
            ("GFC", "2007-12-01", "2010-12-01"),
            # ("COVID", "2020-02-01", "2020-04-01"),
        ]

    event_mask = np.zeros(len(time_vals), dtype=bool)
    for _, start_s, end_s in event_windows:
        start_ts = pd.Timestamp(start_s)
        end_ts = pd.Timestamp(end_s)
        event_mask |= (time_vals >= start_ts) & (time_vals <= end_ts) & (time_vals.month % 3 == 0) 

    if np.any(event_mask):
        ex = x_vals[event_mask]
        ey = y_vals[event_mask]
        ez = z_vals[event_mask]
        et = time_vals[event_mask]

        ax.scatter(
            ex,
            ey,
            ez,
            c=ez,
            cmap="coolwarm",
            norm=zero_center_norm,
            s=float(event_point_size),
            alpha=max(0.2, min(1.0, float(point_alpha) + 0.25)),
            depthshade=True,
            edgecolors="black",
            linewidths=0.45,
        )

        if annotate_event_points:
            z_span = float(np.nanmax(z_vals) - np.nanmin(z_vals)) if len(z_vals) else 0.0
            z_offset = 0.02 * z_span if z_span > 0 else 0.001
            for x_i, y_i, z_i, ts_i in zip(ex, ey, ez, et):
                ax.text(
                    float(x_i),
                    float(y_i),
                    float(z_i + z_offset),
                    pd.Timestamp(ts_i).strftime("%Y-%m"),
                    fontsize=float(event_label_fontsize),
                    color="black",
                    zorder=9,
                )

    # Optionally highlight observations with most extreme total contributions
    # Compute per-sample total contribution (aggregated across features and lags)
    if highlight_n and highlight_n > 0:
        # sv_collapsed has shape (n_samples, n_features)
        total_signed = np.sum(sv_collapsed, axis=1)
        if highlight_method == "abs":
            order_idx = np.argsort(np.abs(total_signed))
        elif highlight_method == "posneg":
            # pick top positives then top negatives
            pos_idx = np.argsort(total_signed)
            order_idx = np.concatenate([pos_idx[::-1], pos_idx])
        else:
            # default fall back to signed magnitude
            order_idx = np.argsort(total_signed)

        sel_idx = order_idx[-int(highlight_n):]
        sel_idx = np.asarray(sel_idx, dtype=int)
        hx = x_vals[sel_idx]
        hy = y_vals[sel_idx]
        hz = z_vals[sel_idx]
        ht = time_vals[sel_idx]

        ax.scatter(
            hx,
            hy,
            hz,
            c=hz,
            cmap="coolwarm",
            norm=zero_center_norm,
            s=float(event_point_size) * 1.6,
            marker="X",
            alpha=1.0,
            depthshade=True,
            edgecolors="black",
            linewidths=0.8,
            label="Extreme total contrib",
        )

        if annotate_event_points:
            for x_i, y_i, z_i, ts_i in zip(hx, hy, hz, ht):
                ax.text(
                    float(x_i),
                    float(y_i),
                    float(z_i + 0.02 * (np.nanmax(z_vals) - np.nanmin(z_vals) if len(z_vals) else 1.0)),
                    pd.Timestamp(ts_i).strftime("%Y-%m"),
                    fontsize=float(event_label_fontsize),
                    color="black",
                    zorder=10,
                )

    cbar = fig.colorbar(surface, ax=ax, shrink=0.72, pad=0.08)
    cbar.set_label("Spline fitted combined attribution")

    def _safe_token(s: str) -> str:
        t = str(s).strip().lower()
        out = [ch if (ch.isalnum() or ch == "_") else "_" for ch in t]
        token = "".join(out)
        while "__" in token:
            token = token.replace("__", "_")
        return token.strip("_")

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / (
        f"spline3d_{_safe_token(model_to_explain)}_{_safe_token(target_name)}"
        f"_q{q_label}_{_safe_token(feat_x_label)}_vs_{_safe_token(feat_y_label)}.png"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    html_path = None
    if make_interactive_html:
        import plotly.graph_objects as go

        colorscale = [
            [0.0, "rgb(59,76,192)"],
            [0.5, "rgb(255,255,255)"],
            [1.0, "rgb(180,4,38)"],
        ]

        fig_html = go.Figure()
        fig_html.add_trace(
            go.Surface(
                x=xx,
                y=yy,
                z=zz,
                surfacecolor=zz,
                colorscale=colorscale,
                cmin=-z_abs_max,
                cmax=z_abs_max,
                opacity=0.86,
                colorbar={"title": "Spline fitted combined attribution"},
                showscale=True,
            )
        )

        if np.any(event_mask):
            ex = x_vals[event_mask]
            ey = y_vals[event_mask]
            ez = z_vals[event_mask]
            et = time_vals[event_mask]
            event_text = [pd.Timestamp(ts).strftime("%Y-%m") for ts in et]

            fig_html.add_trace(
                go.Scatter3d(
                    x=ex,
                    y=ey,
                    z=ez,
                    mode="markers+text" if annotate_event_points else "markers",
                    text=event_text if annotate_event_points else None,
                    textposition="top center",
                    textfont={"size": max(7, int(event_label_fontsize))},
                    marker={
                        "size": max(4, int(event_point_size / 6.0)),
                        "color": ez,
                        "colorscale": colorscale,
                        "cmin": -z_abs_max,
                        "cmax": z_abs_max,
                        "line": {"color": "black", "width": 1},
                        "opacity": max(0.2, min(1.0, float(point_alpha) + 0.25)),
                    },
                    name="Event points",
                    showlegend=True,
                )
            )

            # Highlight extreme-total-contribution points in the interactive plot as well
            if highlight_n and (highlight_n > 0):
                total_signed = np.sum(sv_collapsed, axis=1)
                if highlight_method == "abs":
                    order_idx = np.argsort(np.abs(total_signed))
                elif highlight_method == "posneg":
                    pos_idx = np.argsort(total_signed)
                    order_idx = np.concatenate([pos_idx[::-1], pos_idx])
                else:
                    order_idx = np.argsort(total_signed)

                sel_idx = order_idx[-int(highlight_n):]
                sel_idx = np.asarray(sel_idx, dtype=int)
                hx = x_vals[sel_idx]
                hy = y_vals[sel_idx]
                hz = z_vals[sel_idx]
                ht = time_vals[sel_idx]
                highlight_text = [pd.Timestamp(ts).strftime("%Y-%m") for ts in ht]

                fig_html.add_trace(
                    go.Scatter3d(
                        x=hx,
                        y=hy,
                        z=hz,
                        mode="markers+text" if annotate_event_points else "markers",
                        text=highlight_text if annotate_event_points else None,
                        textposition="top center",
                        textfont={"size": max(7, int(event_label_fontsize))},
                        marker={
                            "size": max(6, int(event_point_size / 4.0)),
                            "color": hz,
                            "colorscale": colorscale,
                            "cmin": -z_abs_max,
                            "cmax": z_abs_max,
                            "line": {"color": "black", "width": 1.5},
                            "opacity": 1.0,
                        },
                        name="Extreme total contrib",
                        showlegend=True,
                    )
                )

        fig_html.update_layout(
            title=f"3D cubic spline - {model_to_explain} {target_name} q={q_label}",
            scene={
                "xaxis_title": f"{feat_x_label} value",
                "yaxis_title": f"{feat_y_label} value",
                "zaxis_title": "Combined attribution",
            },
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )

        html_path = out_path.with_suffix(".html")
        fig_html.write_html(str(html_path), include_plotlyjs=html_include_plotlyjs)

    return {
        "feature_x": feat_x_label,
        "feature_y": feat_y_label,
        "n_points": int(len(x_vals)),
        "n_event_points": int(np.sum(event_mask)),
        "out_path": str(out_path),
        "html_path": str(html_path) if html_path is not None else None,
    }