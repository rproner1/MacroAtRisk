import pandas as pd

def remove_cols(threshold: float, df: pd.DataFrame, train_end: str):

    train_df = df.loc[:train_end]
    cols_to_keep = [c for c in df.columns if (train_df[c].isna().sum() / train_df.shape[1] < threshold)]
    df_clean = df.loc[:, cols_to_keep]
    return df_clean