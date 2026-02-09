"""
Results Generation Entry Point
Combines predictions from all models and generates evaluation tables.
"""
from dotenv import load_dotenv
import os
import yaml
import argparse
from pathlib import Path
from datetime import date

from src.data.concat_preds import concat_predictions
from src.tables.make_r1_tables import make_r1_tables

load_dotenv()

# ----- Configuration -----
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

parser = argparse.ArgumentParser(description="Generate results and tables")
parser.add_argument('--target', type=int, default=0, help="Target variable index (0: Infl_yoy, 1: IP_yoy, 2: Unrate_yoy)")
parser.add_argument("--date", type=str, default=None, 
                    help="Date identifier for results (default: from config)")
parser.add_argument("--country", type=str, default=None, 
                    help="Country code (default: from config)")
parser.add_argument("--horizon", type=int, default=None, 
                    help="Forecast horizon in quarters (default: from config)")
parser.add_argument("--quantiles", type=float, nargs="*", default=None,
                    help="List of quantiles (default: from config)")
parser.add_argument("--test-start", type=str, default=None,
                    help="Test set start date (default: from config)")
parser.add_argument("--test-end", type=str, default=None,
                    help="Test set end date (default: from config)")
parser.add_argument("--start-year", type=int, default=None,
                    help="First prediction year (default: from config)")
parser.add_argument("--end-year", type=int, default=None,
                    help="Last prediction year (default: from config)")
args = parser.parse_args()

# Use config values as defaults, allow CLI overrides
TARGET_IDX = args.target
DATE = args.date if args.date is not None else config.get('date', str(date.today()))
COUNTRY = args.country if args.country is not None else config['country']
HORIZON = args.horizon if args.horizon is not None else config['horizon_in_quarters']
QUANTILES = args.quantiles if args.quantiles is not None else config['quantiles']
TEST_START = args.test_start if args.test_start is not None else config['test_start']
TEST_END = args.test_end if args.test_end is not None else config['test_end']
START_YEAR = args.start_year if args.start_year is not None else config['start_year']
END_YEAR = args.end_year if args.end_year is not None else config['end_year']
TARGET_FILE = config['target_file']

RUN_LOCALLY = config['run_locally']
ROOT_DIR = Path(os.getenv("LOCROOTDIR" if RUN_LOCALLY else "ROOTDIR"))

# Prediction directories
if RUN_LOCALLY:
    DATA_DIR = Path(os.getenv("LOCDATADIR")) / "processed"
    ST_PRED_DIR = Path(os.getenv('LOCPREDDIR')) / 'st_preds' / DATE
    LIT_BENCH_PRED_DIR = Path(os.getenv('LOCPREDDIR')) / 'lit_bench_preds' / DATE
    SHELF_PRED_DIR = Path(os.getenv('LOCPREDDIR')) / 'shelf_preds' / DATE
    PRED_DIR = Path(os.getenv('LOCPREDDIR')) / 'concatenated' / DATE
else:
    DATA_DIR = Path(os.getenv("DATADIR")) / "processed"
    ST_PRED_DIR = Path(os.getenv('PREDDIR')) / 'st_preds' / DATE
    LIT_BENCH_PRED_DIR = Path(os.getenv('PREDDIR')) / 'lit_bench_preds' / DATE
    SHELF_PRED_DIR = Path(os.getenv('PREDDIR')) / 'shelf_preds' / DATE
    PRED_DIR = Path(os.getenv('PREDDIR')) / 'concatenated' / DATE

RESULTS_DIR = ROOT_DIR / "results" / DATE
TABLES_DIR = ROOT_DIR / "results_tables" / DATE

target_path = DATA_DIR / TARGET_FILE

target_name_dict = {0: 'Infl_yoy', 1: 'IP_yoy', 2: 'Unrate_yoy'}
target_name = target_name_dict[TARGET_IDX]

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def main():
    """Combine predictions and generate evaluation tables."""
    
    print("Step 1: Concatenating predictions...")
    concat_predictions(
        country=COUNTRY,
        horizon_in_quarters=HORIZON,
        date=DATE,
        st_pred_dir=ST_PRED_DIR,
        lit_bench_pred_dir=LIT_BENCH_PRED_DIR,
        shelf_pred_dir=SHELF_PRED_DIR,
        pred_dir=PRED_DIR,
        start_year=START_YEAR,
        end_year=END_YEAR
    )
    
    print("\nStep 2: Generating R1 tables...")
    make_r1_tables(
        target_idx=TARGET_IDX,
        targets_path=target_path,
        pred_path=PRED_DIR / f'all_models_predictions_{COUNTRY}_{HORIZON}q_{target_name}.csv',
        results_dir=RESULTS_DIR,
        tables_dir=TABLES_DIR,
        country=COUNTRY,
        horizon_in_quarters=HORIZON,
        quantiles=QUANTILES,
        test_start=TEST_START,
        test_end=TEST_END,
        date=DATE
    )
    
    print(f"\nResults complete. Tables saved to {TABLES_DIR}")


if __name__ == "__main__":
    main()
