import pandas as pd
import numpy as np

df = pd.read_csv('signed_predictors_dl_wide.csv')

df = df.replace([-np.inf, np.inf], np.nan)
df_new = pd.DataFrame(columns=df.columns)
char_cols = df.columns[2:] # ignore firm number and date columns

def compute_decile_diff(g, col):
    """Compute top decile mean - bottom decile mean for a column in a group."""
    s = g[col].dropna()  # Drop NaNs for this characteristic
    if len(s) < 10:
        return np.nan
    k = max(1, len(s) // 10)  # Number of firms in decile (at least 1)
    bottom_mean = s.nsmallest(k).mean()  # Mean of lowest k firms
    top_mean = s.nlargest(k).mean()      # Mean of highest k firms
    return top_mean - bottom_mean

# Compute diffs using groupby (much faster)
df_new = df.groupby('yyyymm').apply(
    lambda g: pd.Series({f"{col}_top_bottom_diff": compute_decile_diff(g, col) for col in char_cols})
).reset_index()

# Sort by date
df_new = df_new.sort_values('yyyymm').reset_index(drop=True)

