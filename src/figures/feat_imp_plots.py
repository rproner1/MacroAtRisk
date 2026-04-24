import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path    
from matplotlib.lines import Line2D


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
):
    """Factor_risk-style top-k SHAP contribution time-series for one quantile."""
    sv_q = np.asarray(sv_q)
    x_q = np.asarray(x_q)
    idx = pd.DatetimeIndex(time_index)

    if sv_q.ndim != 3 or x_q.ndim != 3:
        raise ValueError(f"Expected (n,t,f) tensors, got sv_q={sv_q.shape}, x_q={x_q.shape}")
    if sv_q.shape != x_q.shape:
        raise ValueError(f"Shape mismatch: sv_q {sv_q.shape} vs x_q {x_q.shape}")
    if sv_q.shape[0] != len(idx):
        raise ValueError(f"Index length mismatch: {len(idx)} vs n_samples={sv_q.shape[0]}")

    # Collapse lag first to get sample-level contribution/value series per feature.
    sv_collapsed = np.mean(sv_q, axis=1)  # (n, f)
    x_collapsed = np.mean(x_q, axis=1)    # (n, f)

    global_imp = np.mean(np.abs(sv_collapsed), axis=0)
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
            x_mu = float(np.nanmean(x_series.values))
            x_sigma = float(np.nanstd(x_series.values))
            x_z = (x_series - x_mu) / x_sigma if x_sigma > 0 else pd.Series(0.0, index=idx)
            plt.plot(y_plot.index, y_plot.values, linewidth=0.8, alpha=0.30, color=base_color)
            point_colors = _shade_points_by_zscore(base_color, x_z.values)
            plt.scatter(y_plot.index, y_plot.values, color=point_colors, s=16, alpha=0.9, edgecolors='none')
        else:
            plt.plot(y_plot.index, y_plot.values, linewidth=1.0, color=base_color)

    ax = plt.gca()
    _add_macro_recession_shading(ax, idx)
    plt.axhline(y=0.0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel('Date')
    plt.ylabel('SHAP contribution')

    q_label = int(round(float(q) * 100))
    plt.title(f"Top {top_k} Feature Contributions Over Time - {model_to_explain} {target_name} Q{q_label} (MA{smooth_window})")
    legend_title = 'Feature (light=low value, dark=high value)' if feature_value_color else None
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
    out_path = fig_dir / f"{model_to_explain}_{target_name}_top{top_k}_contrib_timeseries_q{q_label}_ma{smooth_window}{style_suffix}.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=220)
    plt.close()

def make_feat_x_time_importance_plot(
    sv_q,
    model_features,
    q,
    target_name,
    fig_dir,
    top_n: int = 20,
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
    overall_feat_imp = np.mean(np.abs(sv_q), axis=(0, 1))
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


def make_overall_importance_plot_agg_time_lags(
    sv_q,
    model_features,
    q,
    target_name,
    fig_dir,
    top_n: int = 20,
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

    overall_imp = np.mean(np.abs(sv_q), axis=(0, 1))
    imp_s = pd.Series(overall_imp, index=model_features).sort_values(ascending=False)

    top_n = min(top_n, n_features)
    imp_top = imp_s.head(top_n).sort_values(ascending=True)

    fig_h = max(6, 0.30 * top_n)
    plt.figure(figsize=(10.5, fig_h))
    plt.barh(imp_top.index, imp_top.values, color="#4f6d7a")
    q_label = int(round(float(q) * 100))
    plt.title(f"Overall SHAP Importance (Avg |SHAP| over Time and Lags) - {target_name}, Q{q_label}")
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.tight_layout()

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"overall_importance_agg_time_lags_{target_name}_q{q_label}.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def make_feat_force_time_plot(
    sv_q,
    model_features,
    q,
    target_name,
    fig_dir,
    top_n: int = 20,
):
    """
    Create a force-style time plot for average SHAP contributions.

    The x-axis is lag, and stacked positive/negative bars show average signed
    contribution per lag for the top features.

    Args:
        sv_q: SHAP values for quantile q, expected shape (n_samples, time_steps, n_features)
        model_features: List of feature names corresponding to the n_features dimension
        q: The quantile for which SHAP values are computed
        target_name: Name of the target variable
        fig_dir: Directory to save the figure
        top_n: Number of features to include in the plot
    """
    sv_q = np.asarray(sv_q)
    if sv_q.ndim != 3:
        raise ValueError(f"Expected sv_q with 3 dimensions (n, t, f), got shape {sv_q.shape}")

    _, n_lags, n_features = sv_q.shape
    if n_features != len(model_features):
        raise ValueError(
            f"Feature count mismatch: sv_q has {n_features} features, "
            f"but model_features has {len(model_features)}"
        )

    # Average signed SHAP over samples -> (lag, feature)
    lag_feature_contrib = np.mean(sv_q, axis=0)

    # Select top features by total absolute contribution over lags.
    total_abs = np.sum(np.abs(lag_feature_contrib), axis=0)
    top_idx = np.argsort(total_abs)[::-1][: min(top_n, n_features)]
    top_features = [model_features[i] for i in top_idx]
    top_contrib = lag_feature_contrib[:, top_idx]

    lags = np.arange(-n_lags, 0)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(top_features)))

    fig_w = max(11, 0.75 * n_lags)
    fig_h = 8
    plt.figure(figsize=(fig_w, fig_h))

    pos_bottom = np.zeros(n_lags)
    neg_bottom = np.zeros(n_lags)

    # Stack positive and negative contributions separately.
    for i, feat in enumerate(top_features):
        vals = top_contrib[:, i]
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)

        plt.bar(lags, pos, bottom=pos_bottom, width=0.8, color=colors[i], label=feat)
        plt.bar(lags, neg, bottom=neg_bottom, width=0.8, color=colors[i])

        pos_bottom = pos_bottom + pos
        neg_bottom = neg_bottom + neg

    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.title(f"Average SHAP Contributions Over Time: {target_name}, q={q}")
    plt.xlabel("Lag")
    plt.ylabel("Average SHAP contribution")
    plt.xticks(lags)
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"feature_force_time_{target_name}_q{q}.png", dpi=200)
    plt.close()


def make_feat_force_time_smooth_plot(
    sv_q,
    model_features,
    q,
    target_name,
    fig_dir,
    top_n: int = 20,
    n_points: int = 200,
):
    """
    Create a smooth-line version of the force-style time plot.

    The x-axis is lag and each line is the average signed SHAP contribution
    through time for one of the top features.
    """
    sv_q = np.asarray(sv_q)
    if sv_q.ndim != 3:
        raise ValueError(f"Expected sv_q with 3 dimensions (n, t, f), got shape {sv_q.shape}")

    _, n_lags, n_features = sv_q.shape
    if n_features != len(model_features):
        raise ValueError(
            f"Feature count mismatch: sv_q has {n_features} features, "
            f"but model_features has {len(model_features)}"
        )

    lag_feature_contrib = np.mean(sv_q, axis=0)
    total_abs = np.sum(np.abs(lag_feature_contrib), axis=0)
    top_idx = np.argsort(total_abs)[::-1][: min(top_n, n_features)]
    top_features = [model_features[i] for i in top_idx]
    top_contrib = lag_feature_contrib[:, top_idx]

    lags = np.arange(-n_lags, 0).astype(float)
    x_dense = np.linspace(lags.min(), lags.max(), n_points)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(top_features)))

    fig_w = max(11, 0.75 * n_lags)
    fig_h = 8
    plt.figure(figsize=(fig_w, fig_h))

    use_spline = True
    try:
        from scipy.interpolate import make_interp_spline
    except Exception:
        use_spline = False

    for i, feat in enumerate(top_features):
        y = top_contrib[:, i]
        if use_spline and n_lags >= 4:
            spline = make_interp_spline(lags, y, k=3)
            y_smooth = spline(x_dense)
        else:
            y_smooth = np.interp(x_dense, lags, y)

        plt.plot(x_dense, y_smooth*100, color=colors[i], linewidth=1.8, label=feat)

    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.title(f"Smooth Average SHAP Contributions Over Time: {target_name}, q={q}")
    plt.xlabel("Lag")
    plt.ylabel("Average SHAP contribution")
    plt.xticks(np.arange(-n_lags, 0, 1))
    plt.grid(axis="y", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"feature_force_time_smooth_{target_name}_q{q}.png", dpi=200)
    plt.close()


def make_feat_time_value_conditioned_plot(
    sv_q,
    x_q,
    model_features,
    q,
    target_name,
    fig_dir,
    top_n: int = 20,
):
    """
    Plot top lag-agnostic features with color conditioned on feature value.

    Time is collapsed first, so feature importance is independent of sequence position.
    - x-axis: sample-level time-averaged SHAP contribution
    - y-axis: top features (global rank)
    - point color: sample-level time-averaged feature value (z-scored within feature)
    """
    sv_q = np.asarray(sv_q)
    x_q = np.asarray(x_q)

    if sv_q.ndim != 3:
        raise ValueError(f"Expected sv_q with shape (n, t, f), got {sv_q.shape}")
    if x_q.ndim != 3:
        raise ValueError(f"Expected x_q with shape (n, t, f), got {x_q.shape}")
    if sv_q.shape != x_q.shape:
        raise ValueError(f"Shape mismatch: sv_q {sv_q.shape} vs x_q {x_q.shape}")

    n_samples, _, n_features = sv_q.shape
    if n_features != len(model_features):
        raise ValueError(
            f"Feature count mismatch: sv_q has {n_features}, model_features has {len(model_features)}"
        )

    # Collapse time first: (n, t, f) -> (n, f)
    sv_time_avg = np.mean(sv_q, axis=1)
    x_time_avg = np.mean(x_q, axis=1)

    # Global rank by mean absolute SHAP over samples.
    global_imp = np.mean(np.abs(sv_time_avg), axis=0)
    top_idx = np.argsort(global_imp)[::-1][: min(top_n, n_features)]
    top_features = [model_features[i] for i in top_idx]

    fig_h = max(7, 0.35 * len(top_features))
    fig, ax = plt.subplots(figsize=(11, fig_h))

    cmap = plt.get_cmap("coolwarm")
    vmin, vmax = -2.0, 2.0

    y_pos = np.arange(len(top_features))
    for i, feat_idx in enumerate(top_idx):
        shap_vals = sv_time_avg[:, feat_idx]
        feat_vals = x_time_avg[:, feat_idx]

        # Per-feature z-score for comparable color scale.
        std = float(np.std(feat_vals))
        if std > 0:
            color_vals = (feat_vals - np.mean(feat_vals)) / std
        else:
            color_vals = np.zeros_like(feat_vals)

        # Add vertical jitter around each feature row for readability.
        # jitter = np.random.uniform(-0.22, 0.22, size=n_samples)
        sc = ax.scatter(
            shap_vals,
            np.full(n_samples, y_pos[i]),
            c=np.clip(color_vals, vmin, vmax),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=14,
            alpha=0.8,
            edgecolors="none",
        )

    ax.axvline(0.0, color="#222222", linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Time-averaged SHAP contribution")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top-{len(top_features)} Features (Time-Averaged, Lag-Agnostic): {target_name}, q={q}")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Feature value (z-score)")

    plt.tight_layout()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"feature_value_conditioned_time_avg_{target_name}_q{q}.png", dpi=220, bbox_inches="tight")
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
    plt.ylabel("SHAP contribution")
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