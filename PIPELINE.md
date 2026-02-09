# MacroAtRisk Pipeline

This project uses a modular pipeline with three separate entry points.

## Entry Points

### 1. Data Processing: `main_data.py`
Processes raw data into model-ready formats.

```bash
python main_data.py
```

**When to run:** Once when data changes or initially setting up the project.

**What it does:**
- Loads raw FRED-MD, OAP, CRSP data
- Constructs OAP signals (avg, value-weighted avg, spreads)
- Prepares predictor datasets (VG, IAR, UAR, AR1, FRED-MD)
- Saves processed data to `DATA_DIR/processed/`

**Config:** Controlled by `config.yaml` settings:
- `process_data`: Enable/disable
- `construct_oap_signals`: Build firm-level signals
- `skip_processed_data`: Skip if files exist

---

### 2. Model Training: `main_train.py`
Fits models for a specific year and target.

```bash
# Train only shelf models
python main_train.py --year 2008 --target 0 --model-type shelf

# Train literature benchmarks
python main_train.py --year 2008 --target 1 --model-type lit_bench

# Train deep models
python main_train.py --year 2008 --target 2 --model-type deep

# Train all model types
python main_train.py --year 2008 --target 0 --model-type all
```

**Arguments:**
- `--year`: Training cutoff year (e.g., 2008 trains on 1961-2008, tests on 2009)
- `--target`: Target variable index
  - `0` = Inflation (Infl_yoy)
  - `1` = Industrial Production (IP_yoy)
  - `2` = Unemployment (Unrate_yoy)
- `--model-type`: Which models to train
  - `shelf`: LR, LASSO, QRF, QGB, AR(1), Naive
  - `lit_bench`: VG, IAR, UAR
  - `deep`: RNN-based models
  - `all`: All of the above

**What it does:**
- Loads processed data
- Trains specified model types
- Saves predictions to year-specific CSV files
- Logs hyperparameters to tuning log

**Output:** `PRED_DIR/{model_type}_preds/{date}/{model}_predictions_{country}_{horizon}q_{target}_{year}.csv`

---

### 3. Results Generation: `main_results.py`
Combines predictions and generates evaluation tables.

```bash
python main_results.py --date 2026-02-04 --test-start 1998-01-01 --test-end 2024-12-01
```

**Arguments:**
- `--date`: Date identifier for this results run
- `--country`: Country code (default: us)
- `--horizon`: Forecast horizon in quarters (default: 4)
- `--quantiles`: List of quantiles (default: 0.05 0.25 0.50 0.75 0.95)
- `--test-start`: Test period start date
- `--test-end`: Test period end date
- `--start-year`: First prediction year (default: 1997)
- `--end-year`: Last prediction year (default: 2023)

**What it does:**
1. Concatenates all yearly predictions into full test-period files
2. Computes out-of-sample R1 scores vs naive benchmarks
3. Generates LaTeX tables

**Output:**
- Concatenated predictions: `PRED_DIR/concatenated/{date}/all_models_predictions_{country}_{horizon}q_{target}.csv`
- R1 scores: `results/{date}/oos_r1_{country}_{horizon}q_{target}_{test_start}-{test_end}.csv`
- LaTeX tables: `results_tables/{date}/r1_{target}.tex`

---

## Typical Workflows

### Full Pipeline (First Run)
```bash
# 1. Process data
python main_data.py

# 2. Train all models for all years and targets
for year in {1997..2023}; do
  for target in 0 1 2; do
    python main_train.py --year $year --target $target --model-type all
  done
done

# 3. Generate results
python main_results.py --date $(date +%Y-%m-%d)
```

### Iterating on Shelf Models Only
```bash
# Train only shelf models for a specific year/target
python main_train.py --year 2008 --target 0 --model-type shelf

# Retrain for all years if satisfied
for year in {1997..2023}; do
  python main_train.py --year $year --target 0 --model-type shelf
done

# Regenerate results
python main_results.py --date $(date +%Y-%m-%d)
```

### Parallel Training on HPC
```bash
# Submit separate jobs for different model types
sbatch train_shelf.sh    # Trains shelf models
sbatch train_litbench.sh # Trains literature benchmarks
sbatch train_deep.sh     # Trains deep models

# Once all complete, combine results
python main_results.py --date 2026-02-04
```

---

## Configuration

Edit `config/config.yaml` to control pipeline behavior:

```yaml
# Data processing
process_data: true
construct_oap_signals: true
skip_processed_data: false

# Model training
fit_models: true
k_folds: 5
quantiles: [0.05, 0.25, 0.50, 0.75, 0.95]

# Evaluation
test_start: "1998-01-01"
test_end: "2024-12-01"
```

---

## Directory Structure

```
MacroAtRisk/
├── main_data.py          # Data processing entry point
├── main_train.py         # Model training entry point
├── main_results.py       # Results generation entry point
├── config/
│   └── config.yaml       # Configuration file
├── src/
│   ├── data/             # Data processing modules
│   ├── train/            # Model training modules
│   ├── tables/           # Results generation modules
│   └── preprocessing/    # Data preparation utilities
└── {DATA_DIR}/
    ├── raw/              # Raw input data
    ├── processed/        # Processed predictors/targets
    └── {PRED_DIR}/
        ├── shelf_preds/
        ├── lit_bench_preds/
        ├── st_preds/
        └── concatenated/
```
