# MacroAtRisk: Deep Learning for Economic Nowcasting & Forecasting

This repository implements deep learning models for quantile regression on macroeconomic targets (inflation, industrial production, unemployment) using firm-level characteristics and FRED macroeconomic indicators.

## Project Overview

**Goal**: Predict probability distributions (quantiles) of macroeconomic variables using neural networks trained on firm characteristics (JKP) and/or macroeconomic features (FRED).

**Key Features**:
- **Data Sources**: 
  - FRED economic indicators (~100 features across 8 categories)
  - JKP firm characteristics (~170 features across 12 categories)
- **Targets**: Inflation (YoY), Industrial Production (YoY), Unemployment (YoY)
- **Models**: Linear (Elastic-Net), Neural Networks, RNN, Deep Multi-Task Quantile (DMQ)
- **Loss**: Tilted (quantile) loss for 5 quantiles: 0.05, 0.25, 0.5, 0.75, 0.95
- **Data Preprocessing**: Imputation, winsorization, standardization, stationarity filtering

## Project Structure

```
macroatrisk/
├── README.md                          # This file
├── PIPELINE.md                        # Data processing pipeline details
├── requirements.txt                   # Python dependencies
├── model.keras                        # Trained model (if saved)
├── config/                            # Configuration files
│   ├── config_file.yaml              # Main config (data paths, targets, dates)
│   ├── data_config.yaml              # Data pipeline config
│   └── eval_config.yaml              # Evaluation config
├── data/
│   ├── raw/                          # Raw downloaded data
│   └── processed/                    # Cleaned, merged feature sets
│       ├── us_4q_fred_x.csv          # FRED features (quarterly)
│       ├── us_4q_jkp_vw_x.csv        # JKP firm characteristics (value-weighted)
│       ├── us_4q_*_targets.csv       # Target variables
│       └── ...
├── src/
│   ├── data/
│   │   ├── prepare_data.py           # Main data pipeline (impute, scale, split, winsorize)
│   │   ├── prepare_jkp_signals.py    # JKP-specific preprocessing (clean, aggregate, filter)
│   │   └── data_utils.py             # Utilities (stationarity, column removal)
│   ├── train/
│   │   ├── models.py                 # Model builders (linear, NN, RNN, DMQ)
│   │   ├── losses.py                 # Quantile loss functions
│   │   └── train_utils.py            # Training utilities
│   ├── eval/
│   │   └── evaluation.py             # R² and R1 score computation
│   ├── figures/                      # Figure generation scripts
│   ├── tables/                       # Table generation scripts
│   └── tests/                        # Unit tests
├── tuning.ipynb                       # Interactive tuning & experimentation notebook
├── sandbox.ipynb                      # Exploratory analysis notebook
├── tune_dmq.ipynb                    # DMQ-specific tuning notebook
├── tuning_logs/                      # Hyperparameter tuning logs
├── models/
│   ├── shelf_models/                 # Fitted benchmark models
│   └── st_models/                    # Fitted state-of-art models
├── predictions/                       # Model predictions (test set)
├── results/                           # Training results & metrics
├── results_tables/                   # Output tables
├── results_figures/                  # Output figures
├── latex/                            # Paper source files
└── scripts/
    ├── main_train.sh                 # Shell script for training
    ├── test_main_train.ps1           # PowerShell test runner
    └── train_naive_and_lit.ps1       # Benchmark model training
```

## Main Entry Points

### 1. **Data Preparation**
**File**: `src/data/prepare_data.py`

**Main Functions**:
- `prepare_non_rnn_data()` - Prepares flat feature arrays with imputation, winsorization, scaling, train/val/test split
- `prepare_rnn_data()` - Creates sequential data for RNN models (wraps `prepare_non_rnn_data`)
- `_impute_missing_features()` - Fills NaN with mean (fit on train, apply to val/test)
- `_winsorize_features()` - Clips extreme outliers to train-based percentiles (0.5%, 99.5%)
- `_scale_features()` - Standardizes features (mean=0, std=1)

**Key Parameters**:
```python
prepare_non_rnn_data(
    targets_path='data/processed/targets.csv',
    input_paths=['data/processed/us_4q_fred_x.csv'],
    start_date='1990-01-01',
    train_cutoff_year=1997,
    val_months=60,
    test_months=240,
    winsorize=True,           # NEW: Enable outlier clipping
    lower=0.005,              # Clip to 0.5% quantile
    upper=0.995               # Clip to 99.5% quantile
)
```

### 2. **Model Training**
**File**: `src/train/models.py`

**Model Builders**:
- `build_linear_model(lr, l1, l2, loss, q)` - Single Dense layer with L1L2 regularization
- `build_nn(n_layers, n_nodes, lr, l1, l2, loss, q)` - Multi-layer Dense network
- `build_rnn(n_recurrent_layers, n_dense_layers, n_nodes, lr, l1, l2, rec_drop, loss, q)` - LSTM-based RNN
- `build_dmq(input_shapes, n_recurrent_layers, n_shared_layers, n_qtask_layers, ...)` - Deep Multi-Task Quantile network

**Loss Functions** (`src/train/losses.py`):
- `make_tilted_loss(q)` - Creates quantile loss closure: `mean(max(q*e, (q-1)*e))` where `e = y_true - y_pred`
- Supports 5 independent quantiles or multi-task training

**Example Usage**:
```python
from src.train.models import build_linear_model
from src.train.losses import make_tilted_loss

model = build_linear_model(
    lr=3e-4,         # Learning rate (reduced for JKP)
    l1=1.0,          # L1 regularization
    l2=0.0,          # L2 regularization
    loss='mse',      # Or provide custom loss
    q=0.5            # Quantile (0.5 = median)
)
```

### 3. **Hyperparameter Tuning**
**File**: `tuning.ipynb` (Interactive notebook)

**Workflow**:
1. Load data with `prepare_non_rnn_data()`
2. Create model with builder function
3. Train with `model.fit()` and `EarlyStopping` callback
4. Evaluate on test set with `evaluate_predictions()`
5. Plot results with `plot_predictions()`

**Known Issues & Solutions**:
- **Problem**: JKP data has extreme outliers (max: 3.6 trillion) → causes 1e23 loss on first epoch
- **Solution**: Enable `winsorize=True` in `prepare_non_rnn_data()` to clip to train-based percentiles

**Best Configurations Found**:
- **Elastic-Net**: `lr=3e-4, l1≤2.0, l2=0.0`
- **NN**: `1 layer, 32 nodes, lr=3e-4, l1≤1.0, l2≤1.0`

### 4. **Model Evaluation**
**File**: `src/eval/evaluation.py`

**Metrics**:
- `compute_oos_r2_score()` - Out-of-sample R² against mean benchmark
- `compute_oos_r1_score(y_pred, y_true, benchmark_pred, q)` - Tilted loss reduction vs. quantile benchmark

**Usage**:
```python
from src.utils.evaluation import compute_oos_r2_score

r2 = compute_oos_r2_score(
    y_true=y_test.values,
    y_pred=predictions,
    benchmark=np.full_like(y_test, y_test.mean())
)
```

## Configuration

All settings are in `config/config_file.yaml`:

```yaml
data:
  start_date: '1990-01-01'
  target_file: 'us_4q_targets.csv'
  raw_data_path: 'data/raw/'
  processed_data_path: 'data/processed/'

model:
  train_cutoff_year: 1997
  val_months: 60
  test_months: 240
  winsorize: true
  winsorize_lower: 0.005   # 0.5% quantile
  winsorize_upper: 0.995   # 99.5% quantile
```

## Data Pipeline Details

### Step 1: Feature Preparation

**FRED Features** (`src/data/prepare_fred_signals.py`):
- Downloaded from FRED API
- 8 categories: Growth, Employment, Housing, Real Activity, Money, Interest Rates, Prices, Stock Market
- Quarterly aggregation, differencing for stationarity

**JKP Features** (`src/data/prepare_jkp_signals.py`):
- Clean: Drop missing permnos, rename columns
- Aggregate: Compute value-weighted cross-sectional means per month
- Replace zeros with NaN (sparse signals)
- Filter: Remove columns with >30% NaN in training period
- Stationarity: ADF test, remove non-stationary columns
- Lag: 4-quarter lag for prediction

### Step 2: Preprocessing (`prepare_non_rnn_data`)

```
Raw Features
    ↓
[1] WINSORIZE (if enabled)
    - Compute 0.5% and 99.5% quantiles from X_train
    - Clip X_train, X_val, X_test to these bounds
    ↓
[2] IMPUTE
    - SimpleImputer().fit(X_train) → fill NaN with mean
    - Apply to val/test
    ↓
[3] SCALE
    - StandardScaler().fit(X_train) → mean=0, std=1
    - Apply to val/test
    ↓
[4] TRAIN/VAL/TEST SPLIT
    - Train: 1990 to train_cutoff_year - val_months
    - Val: (train_end+1) to (train_end+val_months)
    - Test: (val_end+1) to (val_end+test_months)
    ↓
Scaled Features Ready for Training
```

### Step 3: For RNN Models

If using `prepare_rnn_data()`:
- Concatenate features with targets to create sequences
- Use `split_sequences()` to create (n_samples, n_timesteps, n_features) arrays
- Each RNN sample sees 12 quarters (3 years) of history

## Implementation Details

### Quantile Loss

For a single quantile $q$:

$$\text{Loss} = \mathbb{E}[\max(q \cdot e, (q-1) \cdot e)]$$

where $e = y_{\text{true}} - y_{\text{pred}}$.

This penalizes underprediction by $q$ and overprediction by $(1-q)$, centering the predictions at the desired quantile.

### Multi-Quantile Models

Models like DMQ train 5 outputs simultaneously (one per quantile) with a shared feature extraction backbone:
- Recurrent layers → extract temporal patterns
- Shared dense layers → learn common representations
- Task-specific dense layers → per-quantile refinement

### Regularization

- **L1** (Lasso): Encourages sparsity, useful when features are potentially redundant
- **L2** (Ridge): Prevents weight explosion, useful with many features
- **Dropout** (RNN): Randomly zeros activations to prevent co-adaptation

### Why Winsorization?

JKP features contain rare extreme values (e.g., unusual financial ratios during crises). These outliers:
1. Distort the scaler's mean/std
2. Create massive gradients during backprop
3. Cause optimizer divergence (1e23 loss on first epoch)

**Solution**: Clip to train-based percentiles before scaling. This keeps the natural distribution but removes tails.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```python
from src.data.prepare_data import prepare_non_rnn_data

X_train, X_val, X_test, y_train, y_val, y_test = prepare_non_rnn_data(
    targets_path='data/processed/us_4q_targets.csv',
    input_paths=['data/processed/us_4q_jkp_vw_x.csv'],
    start_date='1990-01-01',
    train_cutoff_year=1997,
    winsorize=True
)
```

### 3. Build & Train Model
```python
from src.train.models import build_linear_model
from tensorflow.keras.callbacks import EarlyStopping

model = build_linear_model(lr=3e-4, l1=1.0, l2=0.0, loss='mse', q=0.5)

model.fit(
    X_train.values, y_train.values,
    epochs=200,
    batch_size=4,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
)
```

### 4. Evaluate
```python
from src.utils.evaluation import compute_oos_r2_score

y_pred = model.predict(X_test)
r2 = compute_oos_r2_score(y_test.values, y_pred, benchmark=y_test.mean())
print(f"R² Score: {r2}")
```

### 5. Interactive Tuning
Open `tuning.ipynb` for an end-to-end workflow with visualization and benchmark comparisons.

## Known Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| NaN loss on first epoch (JKP) | Extreme outliers distort scaler | Set `winsorize=True` in data prep |
| Very slow convergence | Learning rate too small | Increase `lr` to 1e-4 or 3e-4 |
| Overfitting | Model too complex for data | Add L1/L2 regularization or dropout |
| Validation loss diverges | Gradient instability | Reduce `lr` or enable gradient clipping |

## References

- **Quantile Regression**: Koenker & Bassett (1978)
- **FRED Data**: Federal Reserve Economic Data API
- **JKP Characteristics**: Hou, Xue, Zhang (2015, 2021)
- **Deep Learning for Economics**: Various recent papers on neural forecasting

## License

Internal research project. Contact for usage rights.
