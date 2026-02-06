from datetime import date
from dotenv import load_dotenv
from pathlib import Path
import os
from src.tables.make_r1_tables import make_r1_tables
from src.data.concat_preds import concat_predictions
load_dotenv()

# ----- Configuration -----
# Load configuration parameters
import yaml
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

COUNTRY = config['country']
HORIZON_IN_QUARTERS = config['horizon_in_quarters']
QUANTILES = config['quantiles']
DATE = config.get('date', str(date.today()))
INPUT_FILES = config['input_files']
TARGET_FILE = config['target_file']
CONCAT_PREDS = config['concatenate_predictions']

# Evaluation parameters
MAKE_RESULTS = config['make_results']
BASE_MODELS_SUBSET = config['base_models_subset']
TEST_START = config['test_start']
TEST_END = config['test_end']

import os
from pathlib import Path

# ----- Environment Setup -----
DATADIR = Path(os.getenv("DATADIR"))
processed_data_dir = DATADIR / "processed/"
ROOT_DIR = Path(os.getenv("ROOTDIR"))
results_dir = ROOT_DIR / "results" / DATE 
tables_dir = ROOT_DIR / "results_tables" / DATE 

ST_PRED_DIR = Path(os.getenv('LOCPREDDIR')) /'st_preds' / DATE
SHELF_PRED_DIR = Path(os.getenv('LOCPREDDIR')) /'shelf_preds' / DATE
PRED_DIR = Path(os.getenv('LOCPREDDIR')) / 'concatenated' / DATE

for dir in [results_dir, tables_dir, PRED_DIR]:
    os.makedirs(dir, exist_ok=True)

def main():

    if CONCAT_PREDS:
        concat_predictions(
            st_pred_dir=ST_PRED_DIR,
            lit_bench_pred_dir=processed_data_dir,
            shelf_pred_dir=SHELF_PRED_DIR,
            country=COUNTRY,
            horizon_in_quarters=HORIZON_IN_QUARTERS,
            date=DATE,
            pred_dir=PRED_DIR,
            start_year=1997,
            end_year=2023
        )

    input_paths = [processed_data_dir / file for file in INPUT_FILES]
    target_path = processed_data_dir / TARGET_FILE

    make_r1_tables(
        targets_path=target_path,
        pred_path=PRED_DIR,
        results_dir=results_dir,
        tables_dir=tables_dir,
        base_models_subset=BASE_MODELS_SUBSET,
        country=COUNTRY,
        horizon_in_quarters=HORIZON_IN_QUARTERS,
        quantiles=QUANTILES,
        test_start=TEST_START,
        test_end=TEST_END,
        date=DATE
    )

if __name__ == "__main__":
    main()