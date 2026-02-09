import pandas as pd
from utils.utils import prepare_missing

X = pd.read_csv("/home/rproner/Documents/Data/2025-10-MD.csv", index_col=0)

tcodes = X.iloc[0, :].values.astype(int)
X = X.iloc[1:, :]
X.index = pd.to_datetime(X.index, format="%m/%d/%Y")
X.index = X.index.strftime("%Y-%m-%d")

X_transformed = prepare_missing(X.values, tcodes)
X_transformed_df = pd.DataFrame(X_transformed, index=X.index, columns=X.columns)

X_transformed_df.to_csv("/home/rproner/Documents/Data/2025-10-MD-transformed.csv")