import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from src.utils.files import get_latest_file
from src.utils.clean_data import remove_cols

TRAIN_END = '1997-12-01'
ALPHA = 0.05

COLS = list(set([
    'Leverage', 'MomSeason11YrPlus', 'ChEQ', 'GrLTNOA', 'EP', 'Accruals', 'MomOffSeason11YrPlus', 'IdioVol3F', 'RealizedVol', 'Coskewness', 'RDAbility', 'OrderBacklog', 'HerfBE', 'MomSeason06YrPlus', 'AM', 'RIO_Turnover', 'IntanCFP', 'BetaFP', 'CompEquIss', 'MomOffSeason', 'DivYieldST', 'IntanBM', 'RIO_MB', 'BetaLiquidityPS', 'VolSD', 'RoE', 'IntanSP', 'CashProd', 'MomSeason', 'RD', 'RIO_Volatility', 'ShareIss1Y', 'DelLTI', 'MomSeason16YrPlus', 'IdioVolAHT', 'Tax', 'MaxRet', 'DelEqu', 'EarningsConsistency', 'MomSeason16YrPlus', 'CompEquIss', 'OrderBacklog', 'IntanCFP', 'MaxRet', 'Leverage', 'DivYieldST', 'Accruals', 'RIO_Turnover', 'MomSeason', 'MomSeason06YrPlus', 'IntanSP', 'ChEQ', 'DelLTI', 'std_turn', 'IdioVol3F', 'IntanBM', 'RDS', 'AbnormalAccruals', 'CashProd', 'BetaLiquidityPS', 'RD', 'MomOffSeason', 'HerfBE', 'GP', 'MomSeason11YrPlus', 'RIO_MB', 'MomOffSeason11YrPlus', 'Coskewness', 'EarningsConsistency', 'RDAbility', 'GrLTNOA', 'RoE', 'OperProf', 'IdioVolAHT', 'AM', 'ShareIss1Y', 'RealizedVol', 'BetaFP', 'RIO_Volatility', 'DelEqu', 'Tax', 'OperProfRD', 'std_turn', 'OperPro', 'VolMkt'
]))

data_path = get_latest_file(prefix='oap_avg', directory=Path('data/raw'))
x = pd.read_csv(data_path, index_col=0, parse_dates=True)
x = x.loc['1961-01-01':]
x = remove_cols(0.3, x, TRAIN_END)

missing = [c for c in COLS if c not in x.columns]
if missing:
    print(f"Columns not found after NaN filter: {missing}")

header = (
    f"{'Column':<25}"
    f"{'Train levels':>14}"
    f"{'Train diff':>12}"
    f"{'Full levels':>13}"
    f"{'Full diff':>11}"
)
print(header)
print("-" * 77)

for col in COLS:
    if col not in x.columns:
        continue

    train = x.loc[:TRAIN_END, col]
    full  = x.loc[:, col]

    p_train_levels = adfuller(train.dropna())[1]
    p_train   = adfuller(train.diff().dropna())[1]
    p_full_levels  = adfuller(full.dropna())[1]
    p_full    = adfuller(full.diff().dropna())[1]

    def fmt(p):
        return f"{p:.4f}{'*' if p < ALPHA else ' '}"

    print(
        f"{col:<25}"
        f"{fmt(p_train_levels):>14}"
        f"{fmt(p_train):>12}"
        f"{fmt(p_full_levels):>13}"
        f"{fmt(p_full):>11}"
    )

print("\n* = stationary at 5%")
print("Training-filter keeps a column if: train diff is stationary (train diff *)")
print("Full-filter keeps a column if:     full diff is stationary  (full diff *)")
