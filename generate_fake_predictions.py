"""
Generate fake predictions for all models, targets, and years for testing main_results.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
import yaml

load_dotenv()

# Load configuration
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Configuration
COUNTRY = config['country']
HORIZON_IN_QUARTERS = config['horizon_in_quarters']
QUANTILES = config['quantiles']
RUN_LOCALLY = config['run_locally']
DATE = config.get('date', '2026-02-02')

# Year and target ranges
START_YEAR = 1997
END_YEAR = 2023
TARGETS = ['Infl_yoy', 'IP_yoy', 'Unrate_yoy']

# Directories
if RUN_LOCALLY:
    SHELF_PRED_DIR = Path(os.getenv('LOCPREDDIR')) / 'shelf_preds' / DATE
    LIT_BENCH_PRED_DIR = Path(os.getenv('LOCPREDDIR')) / 'lit_bench_preds' / DATE
    DEEP_PRED_DIR = Path(os.getenv('LOCPREDDIR')) / 'st_preds' / DATE
else:
    SHELF_PRED_DIR = Path(os.getenv('PREDDIR')) / 'shelf_preds' / DATE
    LIT_BENCH_PRED_DIR = Path(os.getenv('PREDDIR')) / 'lit_bench_preds' / DATE
    DEEP_PRED_DIR = Path(os.getenv('PREDDIR')) / 'st_preds' / DATE

# Create directories
for path in [SHELF_PRED_DIR, LIT_BENCH_PRED_DIR, DEEP_PRED_DIR]:
    os.makedirs(path, exist_ok=True)

# Model configurations
SHELF_MODELS = ['Naive', 'AR1', 'LR', 'LAS', 'QRF', 'QGB']
LIT_BENCH_MODELS = {'Infl_yoy': 'IAR', 'IP_yoy': 'VG', 'Unrate_yoy': 'UAR'}
DEEP_MODELS = ['DMQv0', 'DMQv0c', 'DMQv1', 'DMQv1c', 'DMQv2', 'DMQv2c']

# Quantile integers for column names
Q_INTS = [int(q * 100) for q in QUANTILES]

np.random.seed(42)  # For reproducibility


def generate_fake_quantile_predictions(n_samples, mean_value=0.03, std_dev=0.02):
    """
    Generate fake quantile predictions that are monotonically increasing.
    
    Args:
        n_samples: Number of time steps (months)
        mean_value: Central tendency of predictions
        std_dev: Spread of predictions
    
    Returns:
        Dictionary with predictions for each quantile
    """
    predictions = {}
    
    # Generate base predictions with some temporal variation
    base = mean_value + np.random.randn(n_samples) * std_dev * 0.3
    
    # Generate predictions for each quantile
    for i, q in enumerate(QUANTILES):
        # Quantile-specific offset from mean
        q_offset = (q - 0.5) * std_dev * 2
        noise = np.random.randn(n_samples) * std_dev * 0.1
        predictions[q] = base + q_offset + noise
    
    return predictions


def generate_shelf_predictions(year, target_name):
    """Generate fake shelf model predictions."""
    # 12 months of predictions
    date_index = pd.date_range(start=f'{year+1}-01-01', end=f'{year+1}-12-01', freq='MS')
    n_samples = len(date_index)
    
    # Target-specific parameters
    target_params = {
        'Infl_yoy': {'mean': 0.025, 'std': 0.015},
        'IP_yoy': {'mean': 0.02, 'std': 0.04},
        'Unrate_yoy': {'mean': 0.0, 'std': 0.015}
    }
    params = target_params[target_name]
    
    all_preds = {}
    
    for model in SHELF_MODELS:
        # Generate predictions for this model
        preds = generate_fake_quantile_predictions(n_samples, params['mean'], params['std'])
        
        # Add to dictionary with proper column names
        for q_val, q_int in zip(QUANTILES, Q_INTS):
            all_preds[f'{model}_Q{q_int}'] = preds[q_val]
        
        # Add mean prediction for Naive and AR1
        if model in ['Naive', 'AR1']:
            all_preds[f'{model}_Mean'] = preds[0.5]  # Use median as mean
    
    # Create DataFrame
    df = pd.DataFrame(all_preds, index=date_index)
    
    # Save
    output_path = SHELF_PRED_DIR / f"shelf_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}_{year}.csv"
    df.to_csv(output_path)
    print(f"Created: {output_path.name}")


def generate_lit_bench_predictions(year, target_name):
    """Generate fake literature benchmark predictions."""
    # 12 months of predictions
    date_index = pd.date_range(start=f'{year+1}-01-01', end=f'{year+1}-12-01', freq='MS')
    n_samples = len(date_index)
    
    # Target-specific parameters
    target_params = {
        'Infl_yoy': {'mean': 0.025, 'std': 0.015},
        'IP_yoy': {'mean': 0.02, 'std': 0.04},
        'Unrate_yoy': {'mean': 0.0, 'std': 0.015}
    }
    params = target_params[target_name]
    
    # Get model name for this target
    model_name = LIT_BENCH_MODELS[target_name]
    
    # Generate predictions
    preds = generate_fake_quantile_predictions(n_samples, params['mean'], params['std'])
    
    # Create dictionary with proper column names
    all_preds = {}
    for q_val, q_int in zip(QUANTILES, Q_INTS):
        all_preds[f'{model_name}_Q{q_int}'] = preds[q_val]
    
    # Create DataFrame
    df = pd.DataFrame(all_preds, index=date_index)
    
    # Save
    output_path = LIT_BENCH_PRED_DIR / f"lit_bench_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}_{year}.csv"
    df.to_csv(output_path)
    print(f"Created: {output_path.name}")


def generate_deep_predictions(year, target_name):
    """Generate fake deep learning model predictions."""
    # 12 months of predictions
    date_index = pd.date_range(start=f'{year+1}-01-01', end=f'{year+1}-12-01', freq='MS')
    n_samples = len(date_index)
    
    # Target-specific parameters
    target_params = {
        'Infl_yoy': {'mean': 0.025, 'std': 0.015},
        'IP_yoy': {'mean': 0.02, 'std': 0.04},
        'Unrate_yoy': {'mean': 0.0, 'std': 0.015}
    }
    params = target_params[target_name]
    
    all_preds = {}
    all_model_preds = []
    
    for model in DEEP_MODELS:
        # Generate predictions for this model
        preds = generate_fake_quantile_predictions(n_samples, params['mean'], params['std'])
        
        # Store for ensemble calculation
        model_preds = []
        
        # Add to dictionary with proper column names
        for q_val, q_int in zip(QUANTILES, Q_INTS):
            all_preds[f'{model}_Q{q_int}'] = preds[q_val]
            model_preds.append(preds[q_val])
        
        all_model_preds.append(np.array(model_preds).T)  # Shape: (n_samples, n_quantiles)
    
    # Create ensemble predictions (average across all models)
    # all_model_preds = np.stack(all_model_preds, axis=2)  # Shape: (n_samples, n_quantiles, n_models)
    # ensemble_preds = np.mean(all_model_preds, axis=2)  # Shape: (n_samples, n_quantiles)
    
    # for i, q_int in enumerate(Q_INTS):
    #     all_preds[f'DMQe_all_Q{q_int}'] = ensemble_preds[:, i]
    
    # Create DataFrame
    df = pd.DataFrame(all_preds, index=date_index)
    
    # Save
    output_path = DEEP_PRED_DIR / f"st_model_predictions_{COUNTRY}_{HORIZON_IN_QUARTERS}q_{target_name}_{year}.csv"
    df.to_csv(output_path)
    print(f"Created: {output_path.name}")


def main():
    """Generate all fake predictions."""
    print(f"Generating fake predictions for years {START_YEAR}-{END_YEAR}")
    print(f"Targets: {TARGETS}")
    print(f"Date: {DATE}\n")
    
    total_files = 0
    
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n=== Year {year} ===")
        
        for target in TARGETS:
            print(f"\nTarget: {target}")
            
            # Generate shelf predictions
            generate_shelf_predictions(year, target)
            
            # Generate literature benchmark predictions
            generate_lit_bench_predictions(year, target)
            
            # Generate deep model predictions
            generate_deep_predictions(year, target)
            
            total_files += 3
    
    print(f"\n{'='*60}")
    print(f"Complete! Generated {total_files} prediction files.")
    print(f"Shelf predictions: {SHELF_PRED_DIR}")
    print(f"Lit bench predictions: {LIT_BENCH_PRED_DIR}")
    print(f"Deep predictions: {DEEP_PRED_DIR}")
    print(f"\nYou can now test main_results.py with:")
    print(f"  python main_results.py --date {DATE}")


if __name__ == "__main__":
    main()
