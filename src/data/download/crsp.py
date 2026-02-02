import numpy as np
import pandas as pd
import wrds
from dotenv import load_dotenv
import os
load_dotenv()
from src.utils.files import timestamp_file
from pathlib import Path


DATA_DIR = os.getenv("DATADIR")
raw_data_dir = Path(DATA_DIR) / "raw"

os.makedirs(raw_data_dir, exist_ok=True)

def get_crsp_monthly(
    wrds_username: str,
    wrds_password: str,
    start_date: str="19590101",
    end_date: str="20251231",
    save: bool=True
) -> pd.DataFrame:
    """
    Download CRSP monthly stock file data and return a DataFrame.
    By the default, stocks with SHRCD between 10 and 11 and EXCHCD between 1 and 3 are selected.

    Args:
        wrds_username (str): A WRDS username to use for the connection.
        wrds_password (str): A WRDS password to use for the connection.
        CRSP_START_DATE (str): The start date for the data.
        CRSP_END_DATE (str): The end date for the data.
    """

    conn = wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)

    query = (
        "SELECT msf.permno, date_trunc('month', msf.mthcaldt)::date AS date, msf.shrout, msf.mthprc AS altprc, "
        "ssih.primaryexch, ssih.siccd "
        "FROM crsp.msf_v2 as msf "
        "INNER JOIN crsp.stksecurityinfohist AS ssih "
        "ON msf.permno = ssih.permno AND "
        "ssih.secinfostartdt <= msf.mthcaldt AND "
        "msf.mthcaldt <= ssih.secinfoenddt "
        f"WHERE msf.mthcaldt BETWEEN '{start_date}' AND '{end_date}' "
        "AND ssih.sharetype = 'NS' "
        "AND ssih.securitytype = 'EQTY' "  
        "AND ssih.securitysubtype = 'COM' " 
        "AND ssih.usincflg = 'Y' " 
        "AND ssih.issuertype in ('ACOR', 'CORP') " 
        "AND ssih.primaryexch in ('N', 'A', 'Q') "
        "AND ssih.conditionaltype in ('RW', 'NW') "
        "AND ssih.tradingstatusflg = 'A'"
    )

    df = conn.raw_sql(query)
    df["date"] = pd.to_datetime(df["date"])

    # push date to the end of the month
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    df["altprc"] = np.abs(df["altprc"])
    df["permno"] = df["permno"].astype(int)
    df['yyyymm'] = df['date'].dt.strftime('%Y%m')
    df.drop(columns=['date'], inplace=True)
    df['size'] = df['altprc'] * df['shrout'] / 1_000_000

    conn.close()

    if save:

        df.to_parquet(
            timestamp_file(raw_data_dir / f'crsp_monthly.parquet')
        )

    return df

get_crsp_monthly(
    wrds_username=os.getenv("WRDS_USERNAME"),
    wrds_password=os.getenv("WRDS_PASSWORD"),
    start_date="19590101",
    end_date="20251231",
    save=True
)