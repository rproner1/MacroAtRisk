import os
from src.data.prepare_ar1_x import get_ar1_x
from src.data.prepare_fred import get_fred_md_x, get_targets
from src.data.prepare_oap_x import get_firm_level_x
from src.data.make_oap_signals import get_firm_avg, get_firm_spread
from src.data.prepare_lit_x import get_vg_x, get_iar_x, get_uar_x
from src.utils.files import get_latest_file, timestamp_file
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

DATA_DIR = Path(os.getenv("DATADIR"))
raw_data_dir = DATA_DIR / "raw/"
processed_data_dir = DATA_DIR / "processed/"
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

ROOT_DIR = Path(os.getenv("ROOTDIR"))
conf_path = ROOT_DIR / 'config' / 'config.yaml'

# Load configuration parameters
import yaml
with open(conf_path, "r") as f:
    config = yaml.safe_load(f)

DESIRED_START_DATE_OF_SAMPLES = pd.to_datetime(config["desired_start_date_of_samples"])
HORIZON_IN_QUARTERS = config["horizon_in_quarters"]
INITIAL_TRAINING_LAST_DATE = pd.to_datetime(config["initial_training_last_date"])
LAST_DATE_OF_SAMPLE = pd.to_datetime(config["last_date_of_sample"])
REMOVE_COLS_THRESHOLD = config["remove_cols_threshold"]
CONSTRUCT_OAP_SIGNALS = config.get("construct_oap_signals", False)
SKIP_PROCESSED_DATA = config.get("skip_processed_data", False)

def prepare_us_data(
        fred_file_path: Path,
        oap_panel_path: Path,
        size_path: Path,
        nfci_file_path: Path,
        nrou_file_path: Path,
        lte_file_path: Path,
        ebp_file_path: Path,
        desired_start_date_of_samples: pd.Timestamp = DESIRED_START_DATE_OF_SAMPLES,
        horizon_in_quarters: int = HORIZON_IN_QUARTERS,
        initial_training_last_date: pd.Timestamp = INITIAL_TRAINING_LAST_DATE,
        last_date_of_sample: pd.Timestamp = LAST_DATE_OF_SAMPLE,
        remove_cols_threshold: float = REMOVE_COLS_THRESHOLD,
        skip_processed_data: bool = SKIP_PROCESSED_DATA
    ) -> pd.DataFrame:

    print("Starting data preparation pipeline...")
    
    h = horizon_in_quarters
    
    # VG benchmark
    vg_file = processed_data_dir / f"us_{h}q_vg_x.parquet"
    if skip_processed_data and get_latest_file(vg_file) is not None:
        print(f"Skipping VG predictors - file already exists")
    else:
        print("Preparing VG benchmark predictors...")
        vg_x = get_vg_x(
            fred_file_path=fred_file_path,
            nfci_file_path=nfci_file_path,
            desired_start_date_of_samples=desired_start_date_of_samples,
            last_date_of_sample=last_date_of_sample,
            horizon_in_quarters=horizon_in_quarters
        )
        vg_x.to_parquet(timestamp_file(vg_file))
        print(f"VG predictors saved")
    
    # IAR benchmark
    iar_file = processed_data_dir / f"us_{h}q_iar_x.parquet"
    if skip_processed_data and get_latest_file(iar_file) is not None:
        print(f"Skipping IAR predictors - file already exists")
    else:
        print("Preparing IAR benchmark predictors...")
        iar_x = get_iar_x(
            fred_file_path=fred_file_path,
            lte_file_path=lte_file_path,
            nrou_file_path=nrou_file_path,
            ebp_file_path=ebp_file_path,
            desired_start_date_of_samples=desired_start_date_of_samples,
            last_date_of_sample=last_date_of_sample,
            horizon_in_quarters=horizon_in_quarters
        )
        iar_x.to_parquet(timestamp_file(iar_file))
        print(f"IAR predictors saved")
    
    # UAR benchmark
    uar_file = processed_data_dir / f"us_{h}q_uar_x.parquet"
    if skip_processed_data and get_latest_file(uar_file) is not None:
        print(f"Skipping UAR predictors - file already exists")
    else:
        print("Preparing UAR benchmark predictors...")
        uar_x = get_uar_x(
            fred_file_path=fred_file_path,
            desired_start_date_of_samples=desired_start_date_of_samples,
            last_date_of_sample=last_date_of_sample,
            horizon_in_quarters=horizon_in_quarters
        )
        uar_x.to_parquet(timestamp_file(uar_file))
        print(f"UAR predictors saved")

    # FRED-MD predictors
    fred_file = processed_data_dir / f"us_{h}q_fred_x.parquet"
    if skip_processed_data and get_latest_file(fred_file) is not None:
        print(f"Skipping FRED-MD predictors - file already exists")
    else:
        print("Preparing FRED-MD predictors...")
        fred_x = get_fred_md_x(
            file_path=fred_file_path,
            horizon_in_quarters=horizon_in_quarters,
            desired_start_date_of_samples=desired_start_date_of_samples,
            last_date_of_sample=last_date_of_sample,
            initial_training_last_date=initial_training_last_date,
            remove_cols_threshold=remove_cols_threshold
        )
        fred_x.to_parquet(timestamp_file(fred_file))
        print(f"FRED-MD predictors saved")
    
    # AR(1) predictors
    ar1_file = processed_data_dir / f"us_{h}q_ar1_x.parquet"
    if skip_processed_data and get_latest_file(ar1_file) is not None:
        print(f"Skipping AR(1) predictors - file already exists")
    else:
        print("Preparing AR(1) predictors...")
        ar1_x = get_ar1_x(
            file_path=fred_file_path, 
            desired_start_date_of_samples=desired_start_date_of_samples, 
            horizon_in_quarters=horizon_in_quarters, 
            last_date_of_sample=last_date_of_sample
        )
        ar1_x.to_parquet(timestamp_file(ar1_file))
        print(f"AR(1) predictors saved")
    
    # Target variables
    target_file = processed_data_dir / f"us_{h}q_fred_y.parquet"
    if skip_processed_data and get_latest_file(target_file) is not None:
        print(f"Skipping target variables - file already exists")
    else:
        print("Preparing target variables...")
        fred_y = get_targets(
            file_path=fred_file_path,
            desired_start_date_of_samples=desired_start_date_of_samples,
            last_date_of_sample=last_date_of_sample
        )
        fred_y.to_parquet(timestamp_file(target_file))
        print(f"Target variables saved")

    # ----- Firm signals -----
    if CONSTRUCT_OAP_SIGNALS:
        print("Constructing OAP signals...")
        # Load the OAP data file
        oap_panel = pd.read_csv(oap_panel_path)
        size = pd.read_parquet(size_path)

        # Process all OAP signals from config
        oap_signals = config.get("oap_signals", [])
        for signal_config in oap_signals:
            signal_type = signal_config["signal"]
            output_name = signal_config["output_name"]

            print(f"Processing OAP signal: {output_name} of type {signal_type}")

            # Construct signal based on type
            if signal_type == "avg":
                print(f"Constructing {output_name}...")
                oap_signal = get_firm_avg(oap_panel)
            if signal_type == "vw_avg":
                print(f"Constructing {output_name} with value weighting...")
                oap_signal = get_firm_avg(oap_panel, value_weight=True, size=size)
            elif signal_type == "spread":
                quantiles = signal_config.get("quantiles", 10)
                print(f"Constructing {output_name} with {quantiles} quantiles...")
                oap_signal = get_firm_spread(oap_panel, quantiles=quantiles)
            else:
                print(f"Unknown signal type: {signal_type}")
                continue
            
            # Save the signal
            output_path = timestamp_file(raw_data_dir / f"{output_name}.csv")
            oap_signal.to_csv(output_path, index=False)
            print(f"{output_name} saved to {output_path}")
    
    # Process all OAP datasets from config
    oap_datasets = config.get("oap_datasets", [])
    for oap_config in oap_datasets:
        oap_name = oap_config["name"]
        oap_filename = oap_config["file"]
        first_diff = oap_config["first_difference"]
        output_name = oap_config["output_name"]
        
        oap_file_path = get_latest_file(raw_data_dir / oap_filename, extension=".csv", directory=raw_data_dir)
        output_file = processed_data_dir / f"us_{h}q_{output_name}.parquet"
        
        if skip_processed_data and get_latest_file(output_file) is not None:
            print(f"Skipping {oap_name} - file already exists")
        else:
            print(f"Preparing {oap_name} financial variables...")
            oap_data = get_firm_level_x(
                file_path=oap_file_path,
                first_difference=first_diff,
                horizon_in_quarters=horizon_in_quarters,
                desired_start_date_of_samples=desired_start_date_of_samples,
                last_date_of_sample=last_date_of_sample,
                remove_cols_threshold=remove_cols_threshold,
                initial_training_last_date=initial_training_last_date
            )
            oap_data.to_parquet(timestamp_file(output_file))
            print(f"{oap_name} saved")
    
    return None

if __name__ == "__main__":
    prepare_us_data(
        get_latest_file(raw_data_dir / "2025-12-MD.csv", extension=".csv", directory=raw_data_dir), 
        get_latest_file(raw_data_dir / "signed_predictors_dl_wide.csv", extension=".csv", directory=raw_data_dir),
        get_latest_file(raw_data_dir / "crsp_monthly.parquet", extension=".parquet", directory=raw_data_dir),
        get_latest_file(raw_data_dir / "nfci_monthly.csv", extension=".csv", directory=raw_data_dir),
        get_latest_file(raw_data_dir / "NROU.csv", extension=".csv", directory=raw_data_dir),
        get_latest_file(raw_data_dir / "EXPINF10YR.csv", extension=".csv", directory=raw_data_dir),
        get_latest_file(raw_data_dir / "ebp_csv.csv", extension=".csv", directory=raw_data_dir)
    )    