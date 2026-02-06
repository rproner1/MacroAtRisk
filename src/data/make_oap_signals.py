import pandas as pd
import numpy as np
import warnings

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

        df['permno'] = pd.to_numeric(df['permno'], errors='coerce').astype('Int64')
        size['permno'] = pd.to_numeric(size['permno'], errors='coerce').astype('Int64')
        weights = get_value_weights(size)
        if not weights.groupby('yyyymm')['weight'].sum().round(6).eq(1.0).all():
            raise ValueError("Weights do not sum to 1 for all dates.")
        before_rows = len(df)
        df = df.merge(weights, on=['yyyymm', 'permno'], how='inner')
        dropped_share = 1 - (len(df) / before_rows)
        if dropped_share > 0:
            warnings.warn(
                f"Dropped {dropped_share:.1%} of rows due to missing weights. "
                "Check that 'permno' and 'yyyymm' align between df and size.",
                RuntimeWarning,
            )
        
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

    # Replace zeros with NaN. Zeros arise when there are too few firms in a month to populate bins. 
    # Some signals are sparse and this is frequent. Such signals will later be removed.
    firm_avg_df.replace(0, np.nan, inplace=True)

    return firm_avg_df

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
    df = df.copy()
    df['yyyymm'] = pd.to_datetime(df['yyyymm'], format='%Y%m')
    df.replace([-np.inf, np.inf], np.nan, inplace=True)
    
    exclude = ['permno', 'yyyymm']
    signal_cols = [col for col in df.columns if col not in exclude]
    
    # Vectorized approach: compute quantile bins for all columns at once per date
    def compute_spread_for_group(group):
        result = {}
        for col in signal_cols:
            x = group[col].dropna()
            if len(x) < quantiles:
                result[col] = np.nan
                continue
            bins = pd.qcut(x, quantiles, labels=False, duplicates='drop')
            top = x[bins == bins.max()].mean()
            bottom = x[bins == 0].mean()
            result[col] = top - bottom
        return pd.Series(result)
    
    spread_df = df.groupby('yyyymm', group_keys=False).apply(compute_spread_for_group)
    spread_df.reset_index(inplace=True)

    # Replace zeros with NaN. Zeros arise when there are too few firms in a month to populate bins. 
    # Some signals are sparse and this is frequent. Such signals will later be removed.
    spread_df.replace(0, np.nan, inplace=True)

    return spread_df
