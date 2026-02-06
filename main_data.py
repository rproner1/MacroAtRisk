"""
Data Pipeline Entry Point
Processes raw data into formats ready for final preprocessing and model training.
"""
import logging
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import os
import yaml

from src.data.prepare_ar1_x import get_ar1_x
from src.data.prepare_fred import get_fred_md_x, get_targets
from src.data.prepare_oap_x import get_firm_level_x
from src.data.make_oap_signals import get_firm_avg, get_firm_spread
from src.data.prepare_lit_x import get_vg_x, get_iar_x, get_uar_x
from src.utils.files import get_latest_file, timestamp_file

load_dotenv()

# ----- Configuration -----
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=config['logging_level'])

DESIRED_START_DATE_OF_SAMPLES = pd.to_datetime(config['desired_start_date_of_samples'])
INITIAL_TRAINING_LAST_DATE = pd.to_datetime(config['initial_training_last_date'])
LAST_DATE_OF_SAMPLE = pd.to_datetime(config['last_date_of_sample'])
REMOVE_COLS_THRESHOLD = config['remove_cols_threshold']
CONSTRUCT_OAP_SIGNALS = config["construct_oap_signals"]
SKIP_PROCESSED_DATA = config["skip_processed_data"]
RUN_LOCALLY = config['run_locally']
HORIZON_IN_QUARTERS = config['horizon_in_quarters']

# ----- Paths -----
DATADIR = Path(os.getenv("LOCDATADIR" if RUN_LOCALLY else "DATADIR"))
raw_data_dir = DATADIR / "raw/"
processed_data_dir = DATADIR / "processed/"

os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)


def main():
    """Run the data processing pipeline."""
    logging.info("Starting data preparation pipeline...")
    
    # Get input file paths
    fred_file_path = get_latest_file(raw_data_dir / "2025-12-MD.csv", extension=".csv")
    oap_panel_path = get_latest_file(raw_data_dir / "signed_predictors_dl_wide.csv", extension=".csv")
    size_path = get_latest_file(raw_data_dir / "crsp_monthly.parquet", extension=".parquet")
    nfci_file_path = get_latest_file(raw_data_dir / "nfci_monthly.csv", extension=".csv")
    nrou_file_path = get_latest_file(raw_data_dir / "NROU.csv", extension=".csv")
    lte_file_path = get_latest_file(raw_data_dir / "EXPINF10YR.csv", extension=".csv")
    ebp_file_path = get_latest_file(raw_data_dir / "ebp_csv.csv", extension=".csv")
    
    h = HORIZON_IN_QUARTERS
    
    # VG benchmark
    vg_file = processed_data_dir / f"us_{h}q_vg_x.parquet"
    if SKIP_PROCESSED_DATA and get_latest_file(vg_file) is not None:
        logging.info(f"Skipping VG predictors - file already exists")
    else:
        logging.info("Preparing VG benchmark predictors...")
        vg_x = get_vg_x(
            fred_file_path=fred_file_path,
            nfci_file_path=nfci_file_path,
            desired_start_date_of_samples=DESIRED_START_DATE_OF_SAMPLES,
            last_date_of_sample=LAST_DATE_OF_SAMPLE,
            horizon_in_quarters=HORIZON_IN_QUARTERS
        )
        vg_x.to_parquet(vg_file)
        logging.info(f"VG predictors saved")
    
    # IAR benchmark
    iar_file = processed_data_dir / f"us_{h}q_iar_x.parquet"
    if SKIP_PROCESSED_DATA and get_latest_file(iar_file) is not None:
        logging.info(f"Skipping IAR predictors - file already exists")
    else:
        logging.info("Preparing IAR benchmark predictors...")
        iar_x = get_iar_x(
            fred_file_path=fred_file_path,
            lte_file_path=lte_file_path,
            nrou_file_path=nrou_file_path,
            ebp_file_path=ebp_file_path,
            desired_start_date_of_samples=DESIRED_START_DATE_OF_SAMPLES,
            last_date_of_sample=LAST_DATE_OF_SAMPLE,
            horizon_in_quarters=HORIZON_IN_QUARTERS
        )
        iar_x.to_parquet(iar_file)
        logging.info(f"IAR predictors saved")
    
    # UAR benchmark
    uar_file = processed_data_dir / f"us_{h}q_uar_x.parquet"
    if SKIP_PROCESSED_DATA and get_latest_file(uar_file) is not None:
        logging.info(f"Skipping UAR predictors - file already exists")
    else:
        logging.info("Preparing UAR benchmark predictors...")
        uar_x = get_uar_x(
            fred_file_path=fred_file_path,
            desired_start_date_of_samples=DESIRED_START_DATE_OF_SAMPLES,
            last_date_of_sample=LAST_DATE_OF_SAMPLE,
            horizon_in_quarters=HORIZON_IN_QUARTERS
        )
        uar_x.to_parquet(uar_file)
        logging.info(f"UAR predictors saved")

    # FRED-MD predictors
    fred_file = processed_data_dir / f"us_{h}q_fred_x.parquet"
    if SKIP_PROCESSED_DATA and get_latest_file(fred_file) is not None:
        logging.info(f"Skipping FRED-MD predictors - file already exists")
    else:
        logging.info("Preparing FRED-MD predictors...")
        fred_x = get_fred_md_x(
            file_path=fred_file_path,
            horizon_in_quarters=HORIZON_IN_QUARTERS,
            desired_start_date_of_samples=DESIRED_START_DATE_OF_SAMPLES,
            last_date_of_sample=LAST_DATE_OF_SAMPLE,
            initial_training_last_date=INITIAL_TRAINING_LAST_DATE,
            remove_cols_threshold=REMOVE_COLS_THRESHOLD
        )
        fred_x.to_parquet(fred_file)
        logging.info(f"FRED-MD predictors saved")
    
    # AR(1) predictors
    ar1_file = processed_data_dir / f"us_{h}q_ar1_x.parquet"
    if SKIP_PROCESSED_DATA and get_latest_file(ar1_file) is not None:
        logging.info(f"Skipping AR(1) predictors - file already exists")
    else:
        logging.info("Preparing AR(1) predictors...")
        ar1_x = get_ar1_x(
            file_path=fred_file_path, 
            desired_start_date_of_samples=DESIRED_START_DATE_OF_SAMPLES, 
            horizon_in_quarters=HORIZON_IN_QUARTERS, 
            last_date_of_sample=LAST_DATE_OF_SAMPLE
        )
        ar1_x.to_parquet(ar1_file)
        logging.info(f"AR(1) predictors saved")
    
    # Target variables
    target_file = processed_data_dir / f"us_{h}q_fred_y.parquet"
    if SKIP_PROCESSED_DATA and get_latest_file(target_file) is not None:
        logging.info(f"Skipping target variables - file already exists")
    else:
        logging.info("Preparing target variables...")
        fred_y = get_targets(
            file_path=fred_file_path,
            desired_start_date_of_samples=DESIRED_START_DATE_OF_SAMPLES,
            last_date_of_sample=LAST_DATE_OF_SAMPLE
        )
        fred_y.to_parquet(target_file)
        logging.info(f"Target variables saved")

    # ----- Firm signals -----
    if CONSTRUCT_OAP_SIGNALS:
        logging.info("Constructing OAP signals...")
        # Load the OAP data file
        nrows = 100000 if RUN_LOCALLY else None
        oap_panel = pd.read_csv(oap_panel_path, nrows=nrows)
        size = pd.read_parquet(size_path)

        # Process all OAP signals from config
        oap_signals = config["oap_signals"]
        for signal_config in oap_signals:
            signal_type = signal_config["signal"]
            output_name = signal_config["output_name"]

            logging.info(f"Processing OAP signal: {output_name} of type {signal_type}")

            # Construct signal based on type
            if signal_type == "avg":
                logging.info(f"Constructing {output_name}...")
                oap_signal = get_firm_avg(oap_panel)
            elif signal_type == "vw_avg":
                logging.info(f"Constructing {output_name} with value weighting...")
                oap_signal = get_firm_avg(oap_panel, value_weight=True, size=size)
            elif signal_type == "spread":
                quantiles = signal_config.get("quantiles", 10)
                logging.info(f"Constructing {output_name} with {quantiles} quantiles...")
                oap_signal = get_firm_spread(oap_panel, quantiles=quantiles)
            else:
                logging.info(f"Unknown signal type: {signal_type}")
                continue
            
            # Save the signal
            if RUN_LOCALLY:
                output_name = f"{output_name}_TEST"
                
            output_path = timestamp_file(raw_data_dir / f"{output_name}.csv")
            oap_signal.to_csv(output_path, index=False)
            logging.info(f"{output_name} saved to {output_path}")
    
    # Process all OAP datasets from config
    oap_datasets = config["oap_datasets"]
    
    for oap_config in oap_datasets:
        oap_name = oap_config["name"]
        oap_filename = oap_config["file"]

        if RUN_LOCALLY:
            oap_filename = oap_filename.replace(".csv", "_TEST.csv")
        
        first_diff = oap_config["first_difference"]
        output_name = oap_config["output_name"]
        
        oap_file_path = get_latest_file(raw_data_dir / oap_filename, extension=".csv")
        if oap_file_path is None:
            logging.info(f"Warning: Could not find OAP file matching pattern '{oap_filename}' in {raw_data_dir}")
            continue
            
        logging.info(f"Loading OAP data from {oap_file_path}...")

        if RUN_LOCALLY:
            output_name = f"{output_name}_TEST"             

        output_file = processed_data_dir / f"us_{h}q_{output_name}.parquet"
        
        if SKIP_PROCESSED_DATA and get_latest_file(output_file) is not None:
            logging.info(f"Skipping {oap_name} - file already exists")
        else:
            logging.info(f"Preparing {oap_name} financial variables...")
            oap_data = get_firm_level_x(
                file_path=oap_file_path,
                first_difference=first_diff,
                horizon_in_quarters=HORIZON_IN_QUARTERS,
                desired_start_date_of_samples=DESIRED_START_DATE_OF_SAMPLES,
                last_date_of_sample=LAST_DATE_OF_SAMPLE,
                remove_cols_threshold=REMOVE_COLS_THRESHOLD,
                initial_training_last_date=INITIAL_TRAINING_LAST_DATE
            )
            oap_data.to_parquet(output_file)
            logging.info(f"{oap_name} saved to {output_file}")
    
    logging.info("Data pipeline complete.")


if __name__ == "__main__":
    main()
