import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.interpolate import PchipInterpolator
from pathlib import Path


def get_vg_x(
        fred_file_path: Path, 
        nfci_file_path: Path, 
        horizon_in_quarters: int, 
        desired_start_date_of_samples: pd.Timestamp,
        last_date_of_sample: pd.Timestamp
    ) -> pd.DataFrame:
    """
    Prepare predictors for Vulnerable Growth dataset and save to CSV.
    Args:
        fred_file_name (str): Name of the FRED monthly data CSV file (not transformed) with 'INDPRO'.
        nfci_file_name (str): Name of the NFCI monthly data CSV file with 'NFCI'.
        output_path (str): Path to save the output CSV.
        horizon_in_quarters (int): Forecast horizon in quarters.
        desired_start_date_of_samples (pd.Timestamp): Start date for the sample.
        last_date_of_sample (pd.Timestamp): End date for the sample.
    Returns:
        pd.DataFrame: DataFrame of predictors.
    """

    fred = pd.read_csv(fred_file_path)
    
    # Fix date format and set as index
    fred = fred.loc[1:, :] # skip first row with transformation codes
    fred['sasdate'] = pd.to_datetime(fred['sasdate'], format='%m/%d/%Y')
    fred.set_index('sasdate', inplace=True)
    fred.index.name = 'date'

    lagged_ip_growth = (np.log(fred['INDPRO']) - np.log(fred['INDPRO']).shift(12)).shift(3*horizon_in_quarters + 1)
    lagged_1q_ip_growth = (np.log(fred['INDPRO']) - np.log(fred['INDPRO']).shift(3)).shift(3*horizon_in_quarters + 1)
    
    nfci = pd.read_csv(nfci_file_path)
    nfci['observation_date'] = pd.to_datetime(nfci['observation_date'], format='%m/%d/%Y')
    nfci.set_index('observation_date', inplace=True)
    nfci.index.name = 'date'
    lagged_nfci = nfci['NFCI'].shift(3*horizon_in_quarters + 1)

    lagged_ip_growth = lagged_ip_growth.loc[desired_start_date_of_samples:last_date_of_sample]
    lagged_1q_ip_growth = lagged_1q_ip_growth.loc[desired_start_date_of_samples:last_date_of_sample]
    lagged_nfci = lagged_nfci.loc[desired_start_date_of_samples:last_date_of_sample]
    

    # Merge all predictors using inner join on the date index
    vg_X = (
        lagged_nfci.to_frame('NFCI')
        .join(lagged_ip_growth.to_frame('lagged_IP_yoy'), how='inner')
        .join(lagged_1q_ip_growth.to_frame('lagged_IP_qoq'), how='inner')
    )
    return vg_X


def get_uar_x(
        fred_file_path: Path, 
        horizon_in_quarters: int, 
        desired_start_date_of_samples: pd.Timestamp,
        last_date_of_sample: pd.Timestamp,
        plot: bool=False
    ) -> pd.DataFrame:
    """
    Prepare predictors for Unemployment at Risk dataset and save to CSV.
    Args:
        fred_y (pd.DataFrame): FRED monthly data with required columns.
        output_path (str): Path to save the output CSV.
        horizon_in_quarters (int): Forecast horizon in quarters.
        start_date (str): Start date for the sample (YYYY-MM-DD).
        end_date (str): End date for the sample (YYYY-MM-DD).
        plot (bool): If True, plot the predictors.
    Returns:
        pd.DataFrame: DataFrame of predictors.
    """

    fred = pd.read_csv(fred_file_path)
    
    # Fix date format and set as index
    fred = fred.loc[1:, :] # skip first row with transformation codes
    fred['sasdate'] = pd.to_datetime(fred['sasdate'], format='%m/%d/%Y')
    fred.set_index('sasdate', inplace=True)
    fred.index.name = 'date'

    # Make features
    u_t = fred['UNRATE'].shift(3*horizon_in_quarters + 1)
    pce_gr = (np.log(fred['DPCERA3M086SBEA']) - np.log(fred['DPCERA3M086SBEA']).shift(12)).shift(3*horizon_in_quarters + 1)
    busloans_to_ip = (fred['BUSLOANS'] / fred['INDPRO'])
    busloans_to_ip_gr = (np.log(busloans_to_ip) - np.log(busloans_to_ip).shift(48)).shift(3*horizon_in_quarters + 1)
    baa_gs10 = (fred['BAA'] - fred['GS10']).shift(3*horizon_in_quarters + 1)
    gs10_ff = (fred['GS10'] - fred['FEDFUNDS']).shift(3*horizon_in_quarters + 1)
    
    # Subset data
    u_t = u_t.loc[desired_start_date_of_samples:last_date_of_sample]
    pce_gr = pce_gr.loc[desired_start_date_of_samples:last_date_of_sample]
    busloans_to_ip_gr = busloans_to_ip_gr.loc[desired_start_date_of_samples:last_date_of_sample]
    baa_gs10 = baa_gs10.loc[desired_start_date_of_samples:last_date_of_sample]
    gs10_ff = gs10_ff.loc[desired_start_date_of_samples:last_date_of_sample]
    
    # Merge all predictors using inner join on the date index
    ur_X = (
        u_t.to_frame('UNRATE_lvl')
        .join(pce_gr.to_frame('PCE_gr_4q'), how='inner')
        .join(busloans_to_ip_gr.to_frame('BUSLOANS_to_IP_gr_16q'), how='inner')
        .join(baa_gs10.to_frame('BAA_GS10'), how='inner')
        .join(gs10_ff.to_frame('GS10_FEDFUNDS'), how='inner')
    )
    if plot:
        ur_X.plot()
        plt.title('Unemployment at Risk Predictors')
        plt.legend(ur_X.columns)
        plt.show()
    return ur_X

def get_iar_x(
    fred_file_path: Path, 
    lte_file_path: Path, 
    nrou_file_path: Path,
    ebp_file_path: Path, 
    horizon_in_quarters: int, 
    desired_start_date_of_samples: pd.Timestamp,
    last_date_of_sample: pd.Timestamp, 
    plot: bool=False
):
    """
    Prepare predictors for Inflation at Risk dataset and save to CSV.
    Args:
        fred_file_path (Path): FRED monthly data file path with required columns.
        lte_file_path (Path): Long-term expectations (EXPINF10YR) file path.
        gs10 (pd.Series): 10-year Treasury yield.
        nrou (pd.DataFrame or pd.Series): NROU quarterly, will be interpolated to monthly.
        ebp (pd.DataFrame): Excess bond premium data with 'gz_spread'.
        output_path (str): Path to save the output CSV.
        horizon_in_quarters (int): Forecast horizon in quarters.
        start_date (str): Start date for the sample (YYYY-MM-DD).
        end_date (str): End date for the sample (YYYY-MM-DD).
        plot (bool): If True, plot the predictors.
    Returns:
        pd.DataFrame: DataFrame of predictors.
    """

    fred = pd.read_csv(fred_file_path)
    
    # Fix date format and set as index
    fred = fred.loc[1:, :] # skip first row with transformation codes
    fred['sasdate'] = pd.to_datetime(fred['sasdate'], format='%m/%d/%Y')
    fred.set_index('sasdate', inplace=True)
    fred.index.name = 'date'

    lte = pd.read_csv(lte_file_path, index_col='observation_date', parse_dates=True)
    lte.index.name = 'date'

    gs10 = fred['GS10']

    # Backcast LTE for missing periods using GS10
    backcast_data = pd.concat([gs10, lte], axis=1).dropna()
    backcast_predict = gs10.loc[:lte.index.min()]
    backcast_predict_df = backcast_predict.to_frame(name='GS10')

    backcast_model = smf.ols('EXPINF10YR ~ GS10', data=backcast_data).fit()
    preds = backcast_model.predict(backcast_predict_df)
    

    if plot:
        fig, ax = plt.subplots()
        line1, = ax.plot(gs10, color='green', linestyle='-', label='GS10 (actual)')
        ax2 = ax.twinx()
        line2, = ax2.plot(preds, color='orange', linestyle='--', label='LTE predicted')
        line3, = ax2.plot(lte, color='blue', linestyle=':', label='LTE')

        # Collect handles and labels from both axes
        lines = [line1, line2, line3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')

        plt.show()

    nrou = pd.read_csv(nrou_file_path, index_col='observation_date', parse_dates=True)
    nrou.index.name = 'date'

    quarterly_dates = nrou.index.to_period('Q').to_timestamp()
    quarterly_ord = quarterly_dates.map(pd.Timestamp.toordinal).values
    
    monthly_dates = pd.date_range(quarterly_dates.min(), quarterly_dates.max(), freq='MS')
    monthly_ord = monthly_dates.map(pd.Timestamp.toordinal).values
    
    hermite = PchipInterpolator(quarterly_ord, nrou.values.flatten())
    monthly_nrou = pd.Series(hermite(monthly_ord), index=monthly_dates, name=nrou.columns[0])

    if plot:
        monthly_nrou.plot(title='NROU Monthly Interpolated (Hermite)')

    ebp = pd.read_csv(ebp_file_path, index_col='date', parse_dates=True)

    # Prepare predictors
    lagged_infl = (np.log(fred['CPIAUCSL']) - np.log(fred['CPIAUCSL']).shift(12)).shift(3*horizon_in_quarters + 1)
    avg_yoy_infl_past_year = lagged_infl.rolling(window=12, min_periods=1).mean()
    lagged_lte_full = pd.concat([preds.to_frame(name='EXPINF10YR'), lte], axis=0).sort_index().shift(3*horizon_in_quarters + 1)
    lagged_nrou = monthly_nrou.shift(3*horizon_in_quarters)
    u_t = fred['UNRATE'].shift(3*horizon_in_quarters + 1)
    ugap = u_t - lagged_nrou
    lagged_oil_infl = (np.log(fred['OILPRICEx']) - np.log(fred['OILPRICEx']).shift(12)).shift(3*horizon_in_quarters)
    lagged_credit_spread = ebp[['gz_spread']].shift(3*horizon_in_quarters)
    
    # Subset dates
    avg_yoy_infl_past_year = avg_yoy_infl_past_year.loc[desired_start_date_of_samples:last_date_of_sample]
    lagged_lte_full = lagged_lte_full.loc[desired_start_date_of_samples:last_date_of_sample]
    ugap = ugap.loc[desired_start_date_of_samples:last_date_of_sample]
    lagged_oil_infl = lagged_oil_infl.loc[desired_start_date_of_samples:last_date_of_sample]
    lagged_credit_spread = lagged_credit_spread.loc[desired_start_date_of_samples:last_date_of_sample]
    
    # Merge all predictors using inner join on the date index
    ir_X = (
        avg_yoy_infl_past_year
        .to_frame('Infl_yoy_avg_12m')
        .join(lagged_lte_full, how='inner')
        .join(ugap.to_frame('ugap'), how='inner')
        .join(lagged_oil_infl.to_frame('Oil_infl_yoy'), how='inner')
        .join(lagged_credit_spread, how='inner')
    )

    if plot:
        ir_X.plot()
        plt.title('Inflation at Risk Predictors')
        plt.legend(ir_X.columns)
        plt.show()
    
    return ir_X


# df = get_vg_x(
#     fred_file_name='2025-12-MD.csv',
#     nfci_file_name='nfci_monthly.csv',
#     horizon_in_quarters=4,
#     desired_start_date_of_samples=pd.Timestamp('1961-01-01'),
#     last_date_of_sample=pd.Timestamp('2024-12-01')
# )

# df = get_uar_x(
#     fred_file_name='2025-12-MD.csv',
#     horizon_in_quarters=4,
#     desired_start_date_of_samples=pd.Timestamp('1961-01-01'),
#     last_date_of_sample=pd.Timestamp('2024-12-01'),
#     plot=True
# )

# df = get_iar_x(
#     fred_file_name='2025-12-MD.csv',
#     lte_file_name='EXPINF10YR.csv',
#     nrou_file_name='NROU.csv',
#     ebp_file_name='ebp_csv.csv',
#     horizon_in_quarters=4,
#     desired_start_date_of_samples=pd.Timestamp('1974-02-01'),
#     last_date_of_sample=pd.Timestamp('2024-12-01'),
#     plot=True
# )

# print(df.head())