import pandas as pd
import numpy as np

def get_value_weights(df: pd.DataFrame) -> pd.DataFrame:

    df['weight'] = df.groupby('yyyymm')['size'].transform(lambda x: x / x.sum())

    return df[['yyyymm', 'permno', 'weight']]

def get_firm_avg(df: pd.DataFrame, value_weight: bool = False, size: pd.DataFrame = None) -> pd.DataFrame:
    """
    Computes the firm-level average of financial variables from OAP data.

    Parameters:
    df: pd.DataFrame
        DataFrame containing firm-level signals with a 'yyyymm' column in %Y%m format.
    value_weight: bool, optional
        Whether to weight the average calculation by firm value. Defaults to False.
    size: pd.DataFrame, optional
        DataFrame containing size for each firm at each date. Required if value_weight is True.

    Returns:
    pd.DataFrame: DataFrame with average firm-level signals per date.
    """
    if value_weight and size is None:
        raise ValueError("Size must be provided if value_weight is True.")
    
    # Ensure 'Date' column is in datetime format
    df['yyyymm'] = pd.to_datetime(df['yyyymm'], format='%Y%m')

    

    # Group by 'date' and compute the cross-sectional mean for each financial variable
    if value_weight:
        size['yyyymm'] = pd.to_datetime(size['yyyymm'], format='%Y%m')
        weights = get_value_weights(size)
        df = df.merge(weights, on=['yyyymm', 'permno'], how='left')
        df.replace([-np.inf, np.inf], np.nan, inplace=True)
        signal_cols = df.drop(columns=['permno', 'yyyymm', 'weight']).columns 
        firm_avg_df = (
            df[signal_cols]
            .mul(df['weight'], axis=0)
            .groupby(df['yyyymm'])
            .sum()
            .reset_index()
        )

    else:
        firm_avg_df = (
            df
            .replace([-np.inf, np.inf], np.nan)
            .drop(columns=['permno'])
            .groupby('yyyymm')
            .mean()
            .reset_index()
        )

    return firm_avg_df

def compute_xs_spread(x, quantiles: int):

    # Compute the spread between the top and bottom quantile for a Series
    bins = pd.qcut(x, quantiles, labels=False, duplicates='drop')
    top = x[bins == quantiles - 1]
    bottom = x[bins == 0]
    # If multiple values in top/bottom, take mean
    spread = top.mean() - bottom.mean()
    return spread

def get_firm_spread(df: pd.DataFrame, quantiles: int = 10) -> pd.DataFrame:
    """
    Computes the firm-level spread of financial variables from OAP data.

    Parameters:
    df (pd.DataFrame):
        DataFrame containing firm-level signals with a 'date' column.
    quantiles (int, optional):
        Number of quantiles to use for spread calculation. Defaults to 10.

    Returns:
    pd.DataFrame: DataFrame with spread average of top quantile bin - average of bottom quantile bin of firm-level signals per date.
    """
    # Read data
    df['yyyymm'] = pd.to_datetime(df['yyyymm'], format='%Y%m')

    exclude = ['permno', 'yyyymm']

    cols = [col for col in df.columns if col not in exclude]

    # Compute cross-sectional spread for the specified column per date
    spread_df = {}
    for col in cols:
        spread_df[col] = (
            df
            .groupby('yyyymm')[col]
            .apply(lambda x: compute_xs_spread(
                x.replace([-np.inf, np.inf], np.nan).dropna(), 
                quantiles=quantiles
            ))
        )
    
    spread_df = pd.DataFrame(spread_df)
    spread_df.reset_index(inplace=True)

    return spread_df
