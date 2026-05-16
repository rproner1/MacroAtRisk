import argparse
import logging
import os
import warnings
from datetime import date
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import yaml
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

from src.figures.feat_imp_plots import (
    make_event_force_plot,
    make_feat_x_time_importance_plot,
    make_overall_importance_plot_agg_time_lags,
    make_feature_x_attribution_by_regime_plot,
    make_top_feature_contrib_timeseries_plot,
    make_pair_value_vs_attribution_cubic_spline_3d_plot,
)
from src.preprocessing.prepare_quantile_data import prepare_quantile_data
from src.train.losses import make_tilted_loss, make_total_tilted_loss
from src.utils.files import concat_shap_values

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # For parallelization

SEED = 1  # Set random seed for reproducibility
tf.random.set_seed(SEED)  # Set TensorFlow random seed

load_dotenv()

# ----- Configuration -----
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=config['logging_level'])

parser = argparse.ArgumentParser(description="Compute and/or plot SHAP outputs")
parser.add_argument("--targets", nargs="+", type=int, default=[0, 1, 2], help="Target index list (0=Infl, 1=IP, 2=Unrate)")
parser.add_argument("--model-type", type=str, default="deep", choices=["shelf", "deep", "all"], help="Type of models for which to compute SHAP values")
parser.add_argument("--model-to-explain", type=str, default="DMQv0c")
parser.add_argument("--time-steps", type=int, default=12, help="Number of time steps for RNN input")
parser.add_argument("--date", type=str, default=str(date.today()), help="Date for organizing outputs (default: today's date)")
parser.add_argument("--country", type=str, default=None, help="Country code override (default from config)")
parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon override (default from config)")
parser.add_argument("--quantiles", type=float, nargs="*", default=None, help="Quantiles override (default from config)")
parser.add_argument("--target", type=int, default=0, help="Single target used by plotting stage")
parser.add_argument("--baseline-start-date", type=str, default='1985-01-01', help="Baseline start date for integrated gradients (default from config)")
parser.add_argument("--baseline-end-date", type=str, default='1997-12-01', help="Baseline end date for integrated gradients (default from config)")
parser.add_argument("--nsamples-shap", type=int, default=1000, help="Number of samples to use for SHAP value estimation (default 1000)")
parser.add_argument("--start-year", type=int, default=None, help="First prediction year (default from config)")
parser.add_argument("--end-year", type=int, default=None, help="Last prediction year (default from config)")
parser.add_argument("--create-full-test-set", action="store_true", help="Force rebuild and save the full test set cache")
parser.add_argument("--compute-shap", action="store_true", help="Compute and save yearly SHAP values")
parser.add_argument("--feat-imp", action="store_true", help="Generate SHAP feature-importance plots")
parser.add_argument("--concat-shap", action="store_true", help="Concatenate yearly SHAP arrays before plotting")
parser.add_argument("--event-force", action="store_true", help="Generate event-specific force plots")
parser.add_argument("--event-date", type=str, default="2008-09-01", help="Event date for event-specific force plot (YYYY-MM-DD)")
parser.add_argument("--event-name", type=str, default="Great Financial Crisis", help="Label for the event-specific force plot")
parser.add_argument('--smooth-window', type=int, default=12, help='Rolling window for SHAP contribution timeseries plots')
parser.add_argument('--feature-value-color', action=argparse.BooleanOptionalAction, default=True, help='Color SHAP timeseries points by standardized feature value')
parser.add_argument('--top-k', type=int, default=5, help='Number of top features to include in SHAP contribution timeseries')
parser.add_argument("--global-imp-func", type=str, default="mean", choices=["mean", "median"], help="Function to aggregate SHAP values for global importance ranking")
parser.add_argument("--check-additivity", action="store_true", help="Whether to check SHAP additivity on a sample of predictions")
parser.add_argument("--spline-3d-interactions", action="store_true", help="Generate requested 3D cubic-spline interaction plots")
parser.add_argument("--spline-grid-size", type=int, default=45, help="Grid size for 3D spline surface evaluation")
parser.add_argument("--spline-point-alpha", type=float, default=0.28, help="Point alpha for overlayed 3D scatter")
parser.add_argument("--spline-smoothing", type=float, default=None, help="Optional smoothing parameter for cubic spline (default auto)")
parser.add_argument("--spline-outdir-name", type=str, default="interaction_3d_spline_requested_quantiles", help="Subdirectory under results_figures/DATE for spline outputs")
parser.add_argument("--spline-interactive-html", action="store_true", help="Also save interactive HTML versions of spline plots")
parser.add_argument("--spline-html-plotlyjs", type=str, default="cdn", choices=["cdn", "directory", "inline"], help="How Plotly JS is embedded in HTML output")
parser.add_argument("--highlight-n", type=int, default=0, help="Number of points to highlight in spline interaction plots based on highest absolute SHAP value")
parser.add_argument("--run-locally", action="store_true", help="Whether to run locally")
args = parser.parse_args()

TARGETS = args.targets
MODEL_TYPE = args.model_type
RUN_LOCALLY = args.run_locally
# FIT_LIT_BENCH = args.fit_lit_bench

COUNTRY = args.country if args.country is not None else config['country']
HORIZON_IN_QUARTERS = args.horizon if args.horizon is not None else config['horizon_in_quarters']
QUANTILES = args.quantiles if args.quantiles is not None else config['quantiles']
# K_FOLDS = config['k_folds']
DATE = args.date
INPUT_FILES = config['input_files']
TARGET_FILE = config['target_file']
TIME_STEPS = config['time_steps']
VAL_YEARS = config['val_years']
# EPOCHS = config['epochs']
# BATCH_SIZE = config['batch_size']
MODEL_TO_EXPLAIN = args.model_to_explain
TARGET_IDX_FOR_PLOTS = args.target
START_YEAR = args.start_year if args.start_year is not None else config['start_year']
END_YEAR = args.end_year if args.end_year is not None else config['end_year']
EVENT_FORCE = args.event_force
EVENT_DATE = args.event_date
EVENT_NAME = args.event_name
SMOOTH_WINDOW = args.smooth_window
FEATURE_VALUE_COLOR = args.feature_value_color
TOP_K = args.top_k

REQUESTED_SPLINE_INTERACTIONS = {
    "Infl_yoy": [
        (0.05, "Starts:  MW", "Baa-FF spread"),
        (0.95, "Starts:  MW", "Baa-FF spread"),
    ],
    "IP_yoy": [
        (0.05, "10 yr-FF spread", "VIX"),
    ],
    "Unrate_yoy": [
        (0.95, "10 yr-FF spread", "VIX"),
    ],
}

path_quantiles = [int(q*100) for q in QUANTILES]

BASE_DIR = Path(os.getenv('REMOTE_BASE_DIR')) if not RUN_LOCALLY else Path(os.getenv('LOCAL_BASE_DIR'))

DATA_DIR = BASE_DIR / 'data' / 'processed'
SHELF_MODEL_DIR = BASE_DIR / 'models' / 'shelf_models' / DATE
SHELF_PRED_DIR = BASE_DIR / 'predictions' / 'shelf_preds' / DATE
SHELF_TUNING_LOG_PATH = BASE_DIR / 'tuning_logs' / f"shelf_tuning_log_{DATE}.json"
LIT_BENCH_PRED_DIR = BASE_DIR / 'predictions' / 'lit_bench_preds' / DATE
DEEP_MODEL_DIR = BASE_DIR / 'models' / 'st_models' / DATE
DEEP_PRED_DIR = BASE_DIR / 'st_preds' / DATE
DEEP_TUNING_LOG_PATH = BASE_DIR / 'tuning_logs' / f"st_tuning_log_{DATE}.json"
SHAP_DIR = BASE_DIR / 'shap_values' / DATE
SHAP_CONCAT_DIR = BASE_DIR / 'shap_values' / 'concatenated'
FIGURES_DIR = BASE_DIR / 'results_figures' / DATE

for path in [SHELF_MODEL_DIR, SHELF_PRED_DIR, SHELF_TUNING_LOG_PATH.parent, LIT_BENCH_PRED_DIR, DEEP_MODEL_DIR, DEEP_PRED_DIR, DEEP_TUNING_LOG_PATH.parent, SHAP_DIR, SHAP_CONCAT_DIR, FIGURES_DIR]:
    os.makedirs(path, exist_ok=True)

target_name_dict = {0: 'Infl_yoy', 1: 'IP_yoy', 2: 'Unrate_yoy'}
model_file_dict = {
    0: f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_iar_x.csv",
    1: f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_vg_x.csv",
    2: f"{COUNTRY}_{HORIZON_IN_QUARTERS}q_uar_x.csv"
}
model_name_dict = {0: 'IAR', 1: 'VG', 2: 'UAR'}


linear_model_names =['QR', 'RID', 'LAS', 'EN']
tree_model_names = ['QRF', 'QGB']


custom_objects = {
    **{f'tilted_loss_{Q}': make_tilted_loss(Q) for Q in path_quantiles},
    **{f"total_tilted_loss_{'_'.join(map(str, path_quantiles))}": make_total_tilted_loss(QUANTILES)}
}

def check_additivity(model, X_eval, X_baseline, shap_values, tol=1e-2):

    baseline_preds = model.predict(X_baseline)
    expected_output = np.mean(baseline_preds, axis=0)
    logging.info(f"Expected output (mean prediction on baseline): {expected_output}")
    preds = model.predict(X_eval)
    
    for q in range(shap_values.shape[-1]):
        sv_q = shap_values[..., q]
        sv_sum = np.sum(sv_q, axis=(1, 2))
        deviation = preds[:, q] - expected_output[q] - sv_sum

        logging.debug(f"Quantile {QUANTILES[q]}: SHAP sum = {sv_sum}, Predictions = {preds[:, q]}, Expected output = {expected_output[q]}, Deviation = {deviation}")
        
        if not np.allclose(deviation, 0, atol=tol):
            logging.warning(f"Additivity check failed for quantile {QUANTILES[q]}: max deviation {np.max(np.abs(deviation))}")
        else:
            logging.info(f"Additivity check passed for quantile {QUANTILES[q]}")


def load_or_build_full_test_set(fred_to_group: dict, fred_to_gsi: dict):
    cache_path = SHAP_DIR / "full_test_set_cache.npz"

    if cache_path.exists() and not args.create_full_test_set:
        cache = np.load(cache_path, allow_pickle=True)
        X_test_rnn = cache["X_test_rnn"]
        model_features = cache["model_features"].tolist()
        event_index = pd.DatetimeIndex(pd.to_datetime(cache["event_dates"]))
        logging.info(f"Loaded shared cached full test set from {cache_path}")
        return X_test_rnn, model_features, event_index

    target_path = DATA_DIR / TARGET_FILE
    input_paths = [DATA_DIR / file for file in INPUT_FILES]

    X_test_rnn_all = []
    event_dates = []
    model_features = None

    for year in range(START_YEAR, END_YEAR + 1):
        non_rnn_data_year, rnn_data_year, _ = prepare_quantile_data(
            target=TARGETS[0],
            time_steps=TIME_STEPS,
            targets_path=target_path,
            input_paths=input_paths,
            start_date='1961-01-01',
            train_cutoff_year=year,
            n_quantiles=len(QUANTILES),
            val_years=VAL_YEARS,
        )

        if model_features is None:
            model_features_raw = non_rnn_data_year['X_test'].columns.tolist()
            model_features = [f' {fred_to_group.get(f, "Unknown")} | {fred_to_gsi.get(f, f)}' for f in model_features_raw]

        X_test_rnn_all.append(np.asarray(rnn_data_year['X_test_rnn']))
        event_dates.extend(list(non_rnn_data_year['X_test'].index))

    X_test_rnn = np.concatenate(X_test_rnn_all, axis=0)
    event_index = pd.DatetimeIndex(event_dates)

    np.savez_compressed(
        cache_path,
        X_test_rnn=X_test_rnn,
        model_features=np.asarray(model_features, dtype=object),
        event_dates=np.asarray(event_index, dtype='datetime64[ns]'),
    )
    logging.info(f"Saved shared full test set cache to {cache_path}")

    return X_test_rnn, model_features, event_index


def compute_shap_values():

    # Load data 
    target_path = DATA_DIR / TARGET_FILE
    input_paths = [DATA_DIR / file for file in INPUT_FILES]    

    for target_idx in TARGETS:
        for year in list(range(START_YEAR, END_YEAR + 1)):

            non_rnn_data, rnn_data, meta_data = prepare_quantile_data(
                target=target_idx,
                time_steps=TIME_STEPS,
                targets_path=target_path,
                input_paths=input_paths,
                start_date='1961-01-01',
                train_cutoff_year=year,
                n_quantiles=len(QUANTILES),
                val_years=VAL_YEARS
            )

            # For feature names
            X_train_full = non_rnn_data['X_train_full']
            X_test = non_rnn_data['X_test']
            model_features = X_test.columns.tolist()
            
            (
                X_train_rnn, X_test_rnn,
            ) = itemgetter(
                'X_train_full_rnn','X_test_rnn',
            )(rnn_data)

            # Explain DMQ model
            if 'dmq' in MODEL_TO_EXPLAIN.lower(): 
                # Load model
                model = load_model(
                    DEEP_MODEL_DIR / f"{MODEL_TO_EXPLAIN}_{target_name_dict[target_idx]}_{year}_estimator0.keras",
                    custom_objects=custom_objects
                )

                target_dates = pd.date_range(start=args.baseline_start_date, end=args.baseline_end_date, freq='MS')
                date_indicies = X_train_full.iloc[(args.time_steps-1):].index.get_indexer(target_dates) # Drop first date to align with rnn data which starts at t=1 due to lag.

                # Filter out invalid indices (-1)
                valid_indices = date_indicies[date_indicies >= 0]

                # Check if any dates were not found
                if len(valid_indices) < len(date_indicies):
                    missing_dates = target_dates[date_indicies == -1]
                    logging.warning(f"The following dates were not found in the DataFrame index: {missing_dates}")

                X_bg = np.asarray(X_train_rnn[date_indicies], dtype=np.float32)
                X_eval = np.asarray(X_test_rnn, dtype=np.float32)

                explainer = shap.GradientExplainer(model, X_train_rnn)
                sv = explainer.shap_values(X_test_rnn, nsamples=args.nsamples_shap) # Same (n_samples, time_steps, n_features, n_quantiles)
                if args.check_additivity:
                    check_additivity(model, X_test_rnn, X_train_rnn, sv, tol=1e-3)

                sv = np.asarray(sv)
            
                np.save(SHAP_DIR / f"{MODEL_TO_EXPLAIN}_{target_name_dict[target_idx]}_{year}.npy", sv)
            

def plot_shap_feature_importance(target_idx: int):
    target_name = target_name_dict[target_idx]

    if args.concat_shap:
        sv = concat_shap_values(
            shap_dir=SHAP_DIR,
            output_path=SHAP_CONCAT_DIR / f"{MODEL_TO_EXPLAIN}_concatenated_shap_values_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}.npy",
            glob_pattern=f"{MODEL_TO_EXPLAIN}_{target_name}_*.npy",
        )
    else:
        sv = np.load(
            SHAP_CONCAT_DIR / f"{MODEL_TO_EXPLAIN}_concatenated_shap_values_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}.npy"
        )

    appendix_path = BASE_DIR / 'data' / 'FRED-MD_updated_appendix.csv'
    appendix_df = pd.read_csv(appendix_path, encoding='latin-1')
    group_dict = {
        1: 'Output',
        2: 'Labor',
        3: 'Housing',
        4: 'Consumption',
        5: 'Money',
        6: 'Rates',
        7: 'Prices',
        8: 'Stocks',
    }
    appendix_df['group'] = appendix_df['group'].map(group_dict)
    fred_to_group = pd.Series(appendix_df['group'].values, index=appendix_df['fred']).to_dict()
    fred_to_gsi = pd.Series(appendix_df['gsi:description'].values, index=appendix_df['fred']).to_dict()

    X_test_rnn, model_features, event_index = load_or_build_full_test_set(
        fred_to_group=fred_to_group,
        fred_to_gsi=fred_to_gsi,
    )

    if sv.shape[:3] != X_test_rnn.shape:
        raise ValueError(f"Concatenated SHAP and feature tensors are misaligned: sv {sv.shape[:3]} vs X {X_test_rnn.shape}.")

    print(f"SHAP values shape: {sv.shape}, X_test_rnn shape: {X_test_rnn.shape}")

    for q in QUANTILES:
        q_idx = QUANTILES.index(q)
        sv_q = sv[..., q_idx]

        make_overall_importance_plot_agg_time_lags(
            sv_q=sv_q,
            model_features=model_features,
            q=q,
            target_name=target_name,
            model_to_explain=MODEL_TO_EXPLAIN,
            fig_dir=FIGURES_DIR,
            top_n=25,
            global_imp_func=args.global_imp_func,
        )

        make_feat_x_time_importance_plot(
            sv_q=sv_q,
            model_features=model_features,
            q=q,
            target_name=target_name,
            fig_dir=FIGURES_DIR,
            global_imp_func=args.global_imp_func,
        )
        
        # make_feature_x_attribution_by_regime_plot(
        #     sv_q=sv_q,
        #     x_q=X_test_rnn,
        #     model_features=model_features,
        #     q=q,
        #     target_name=target_name,
        #     model_to_explain=MODEL_TO_EXPLAIN,
        #     fig_dir=FIGURES_DIR,
        # )
        make_top_feature_contrib_timeseries_plot(
            sv_q=sv_q,
            x_q=X_test_rnn,
            model_features=model_features,
            q=q,
            target_name=target_name,
            model_to_explain=MODEL_TO_EXPLAIN,
            fig_dir=FIGURES_DIR,
            time_index=event_index,
            top_k=TOP_K,
            global_imp_func=args.global_imp_func,
            smooth_window=SMOOTH_WINDOW,
            feature_value_color=FEATURE_VALUE_COLOR,
        )

        if EVENT_FORCE:
            event_ts = pd.Timestamp(EVENT_DATE)
            nearest_pos = int(np.argmin(np.abs(event_index - event_ts)))
            matched_date = event_index[nearest_pos]
            make_event_force_plot(
                sv_event=sv_q[nearest_pos],
                model_features=model_features,
                event_date=matched_date,
                event_name=EVENT_NAME,
                q=q,
                target_name=target_name,
                fig_dir=FIGURES_DIR,
                top_n=10,
            )

    if args.spline_3d_interactions:
        target_interactions = REQUESTED_SPLINE_INTERACTIONS.get(target_name, [])
        if target_interactions:
            spline_fig_dir = FIGURES_DIR / args.spline_outdir_name
            summary_rows = []
            q_arr = np.asarray(QUANTILES, dtype=float)

            for q_req, feature_x, feature_y in target_interactions:
                q_matches = np.where(np.isclose(q_arr, float(q_req), rtol=0, atol=1e-10))[0]
                if len(q_matches) == 0:
                    logging.warning(
                        "Requested spline quantile %.4f is not in QUANTILES=%s for target %s; skipping.",
                        float(q_req),
                        QUANTILES,
                        target_name,
                    )
                    continue

                q_idx = int(q_matches[0])
                sv_q = sv[..., q_idx]

                try:
                    meta = make_pair_value_vs_attribution_cubic_spline_3d_plot(
                        sv_q=sv_q,
                        x_q=X_test_rnn,
                        model_features=model_features,
                        q=float(q_req),
                        target_name=target_name,
                        model_to_explain=MODEL_TO_EXPLAIN,
                        fig_dir=spline_fig_dir,
                        feature_x=feature_x,
                        feature_y=feature_y,
                        value_agg="last",
                        contrib_agg="sum",
                        spline_s=args.spline_smoothing,
                        grid_size=args.spline_grid_size,
                        point_alpha=args.spline_point_alpha,
                        time_index=event_index,
                        make_interactive_html=args.spline_interactive_html,
                        html_include_plotlyjs=(True if args.spline_html_plotlyjs == "inline" else args.spline_html_plotlyjs),
                        highlight_n=args.highlight_n,
                        plot_scatter=False,
                    )
                    summary_rows.append({
                        "target": target_name,
                        "quantile": float(q_req),
                        "feature_x": meta["feature_x"],
                        "feature_y": meta["feature_y"],
                        "n_points": meta["n_points"],
                        "n_event_points": meta.get("n_event_points", np.nan),
                        "plot_file": Path(meta["out_path"]).name,
                        "html_file": Path(meta["html_path"]).name if meta.get("html_path") else None,
                    })
                except Exception as exc:
                    logging.exception(
                        "Failed spline interaction plot for target=%s, q=%.4f, features=(%s, %s): %s",
                        target_name,
                        float(q_req),
                        feature_x,
                        feature_y,
                        exc,
                    )

            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_path = spline_fig_dir / f"spline3d_summary_{MODEL_TO_EXPLAIN}_{target_name}.csv"
                summary_df.to_csv(summary_path, index=False)
                logging.info("Saved spline interaction summary to %s", summary_path)


def main():
    if args.compute_shap:
        compute_shap_values()

    if args.feat_imp:
        # Generate feature-importance outputs for each requested target.
        for target_idx in TARGETS:
            plot_shap_feature_importance(target_idx=target_idx)


if __name__ == "__main__":
    main()