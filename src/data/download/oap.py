# import os
import pandas as pd
import openassetpricing as oap
from dotenv import load_dotenv
import os
from datetime import datetime, UTC
load_dotenv()

DATA_DIR = os.getenv("DATADIR")
raw_data_dir = os.path.join(DATA_DIR, "raw/")

os.makedirs(raw_data_dir, exist_ok=True)

def get_oap_firm_level_data(release_version: int, save: bool=True) -> pd.DataFrame:
    """
    Downloads and reads the Open Asset Pricing (OAP) firm-level characteristics data.

    Returns:
        pd.DataFrame: DataFrame containing the OAP firm-level characteristics data.
    """

    openap = oap.OpenAP(release_version)

    df = openap.dl_all_signals('polars')

    return df

get_oap_firm_level_data(release_version=202510, save=True)