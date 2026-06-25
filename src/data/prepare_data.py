from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import logging
from functools import reduce
from src.data.data_utils import split_sequences


FRED_DICT = {
    "1": [
        "RPI",
        "W875RX1",
        "INDPRO",
        "IPFPNSS",
        "IPFINAL",
        "IPCONGD",
        "IPDCONGD",
        "IPNCONGD",
        "IPBUSEQ",
        "IPMAT",
        "IPDMAT",
        "IPNMAT",
        "IPMANSICS",
        "IPB51222s",
        "IPFUELS",
        "CUMFNS"
    ],
    "2": [
        "HWI",
        "HWIURATIO",
        "CLF16OV",
        "CE16OV",
        "UNRATE",
        "UEMPMEAN",
        "UEMPLT5",
        "UEMP5TO14",
        "UEMP15OV",
        "UEMP15T26",
        "UEMP27OV",
        "CLAIMSx",
        "PAYEMS",
        "USGOOD",
        "CES1021000001",
        "USCONS",
        "MANEMP",
        "DMANEMP",
        "NDMANEMP",
        "SRVPRD",
        "USTPU",
        "USWTRADE",
        "USTRADE",
        "USFIRE",
        "USGOVT",
        "CES0600000007",
        "AWOTMAN",
        "AWHMAN",
        "CES0600000008",
        "CES2000000008",
        "CES3000000008"
    ],
    "3": [
        "HOUST",
        "HOUSTNE",
        "HOUSTMW",
        "HOUSTS",
        "HOUSTW",
        "PERMIT",
        "PERMITNE",
        "PERMITMW",
        "PERMITS",
        "PERMITW"
    ],
    "4": [
        "DPCERA3M086SBEA",
        "CMRMTSPLx",
        "RETAILx",
        "ACOGNO",
        "AMDMNOx",
        "ANDENOx",
        "AMDMUOx",
        "BUSINVx",
        "ISRATIOx",
        "UMCSENTx"
    ],
    "5": [
        "M1SL",
        "M2SL",
        "M2REAL",
        "BOGMBASE",
        "TOTRESNS",
        "NONBORRES",
        "BUSLOANS",
        "REALLN",
        "NONREVSL",
        "CONSPI",
        "DTCOLNVHFNM",
        "DTCTHFNM",
        "INVEST"
    ],
    "6": [
        "FEDFUNDS",
        "CP3Mx",
        "TB3MS",
        "TB6MS",
        "GS1",
        "GS5",
        "GS10",
        "AAA",
        "BAA",
        "COMPAPFFx",
        "TB3SMFFM",
        "TB6SMFFM",
        "T1YFFM",
        "T5YFFM",
        "T10YFFM",
        "AAAFFM",
        "BAAFFM",
        "TWEXAFEGSMTHx",
        "EXSZUSx",
        "EXJPUSx",
        "EXUSUKx",
        "EXCAUSx"
    ],
    "7": [
        "WPSFD49207",
        "WPSFD49502",
        "WPSID61",
        "WPSID62",
        "OILPRICEx",
        "PPICMM",
        "CPIAUCSL",
        "CPIAPPSL",
        "CPITRNSL",
        "CPIMEDSL",
        "CUSR0000SAC",
        "CUSR0000SAD",
        "CUSR0000SAS",
        "CPIULFSL",
        "CUSR0000SA0L2",
        "CUSR0000SA0L5",
        "PCEPI",
        "DDURRG3M086SBEA",
        "DNDGRG3M086SBEA",
        "DSERRG3M086SBEA"
    ],
    "8": [
        "S&P 500",
        "S&P div yield",
        "S&P PE ratio",
        "VIXCLSx"
    ]
}

JKP_DICT = {
    "Accruals": [
        "cowc_gr1a",
        "oaccruals_at",
        "oaccruals_ni",
        "seas_16_20na",
        "taccruals_at",
        "taccruals_ni"
    ],
    "Debt Issuance": [
        "capex_abn",
        "debt_gr3",
        "fnl_gr1a",
        "ncol_gr1a",
        "nfna_gr1a",
        "ni_ar1",
        "noa_at"
    ],
    "Investment": [
        "aliq_at",
        "at_gr1",
        "be_gr1a",
        "capx_gr1",
        "capx_gr2",
        "capx_gr3",
        "coa_gr1a",
        "col_gr1a",
        "emp_gr1",
        "inv_gr1",
        "inv_gr1a",
        "lnoa_gr1a",
        "mispricing_mgmt",
        "ncoa_gr1a",
        "nncoa_gr1a",
        "noa_gr1a",
        "ppeinv_gr1a",
        "ret_60_12",
        "sale_gr1",
        "sale_gr3",
        "saleq_gr1",
        "seas_2_5na"
    ],
    "Low Leverage": [
        "age",
        "aliq_mat",
        "at_be",
        "bidaskhl_21d",
        "cash_at",
        "netdebt_me",
        "ni_ivol",
        "rd_sale",
        "rd5_at",
        "tangibility",
        "z_score"
    ],
    "Low Risk": [
        "beta_60m",
        "beta_dimson_21d",
        "betabab_1260d",
        "betadown_252d",
        "earnings_variability",
        "ivol_capm_21d",
        "ivol_capm_252d",
        "ivol_ff3_21d",
        "ivol_hxz4_21d",
        "ocfq_saleq_std",
        "rmax1_21d",
        "rmax5_21d",
        "rvol_21d",
        "seas_6_10na",
        "turnover_126d",
        "zero_trades_126d",
        "zero_trades_21d",
        "zero_trades_252d"
    ],
    "Momentum": [
        "prc_highprc_252d",
        "resff3_12_1",
        "resff3_6_1",
        "ret_12_1",
        "ret_3_1",
        "ret_6_1",
        "ret_9_1",
        "seas_1_1na"
    ],
    "Profit Growth": [
        "dsale_dinv",
        "dsale_drec",
        "dsale_dsga",
        "niq_at_chg1",
        "niq_be_chg1",
        "niq_su",
        "ocf_at_chg1",
        "ret_12_7",
        "sale_emp_gr1",
        "saleq_su",
        "seas_1_1an",
        "tax_gr1a"
    ],
    "Profitability": [
        "dolvol_var_126d",
        "ebit_bev",
        "ebit_sale",
        "f_score",
        "ni_be",
        "niq_be",
        "o_score",
        "ocf_at",
        "ope_be",
        "ope_bel1",
        "turnover_var_126d"
    ],
    "Quality": [
        "at_turnover",
        "cop_at",
        "cop_atl1",
        "dgp_dsale",
        "gp_at",
        "gp_atl1",
        "mispricing_perf",
        "ni_inc8q",
        "niq_at",
        "op_at",
        "op_atl1",
        "opex_at",
        "qmj",
        "qmj_growth",
        "qmj_prof",
        "qmj_safety",
        "sale_bev"
    ],
    "Seasonality": [
        "corr_1260d",
        "coskew_21d",
        "dbnetis_at",
        "kz_index",
        "lti_gr1a",
        "pi_nix",
        "seas_11_15an",
        "seas_11_15na",
        "seas_16_20an",
        "seas_2_5an",
        "seas_6_10an",
        "sti_gr1a"
    ],
    "Short-Term Reversal": [
        "iskew_capm_21d",
        "iskew_ff3_21d",
        "iskew_hxz4_21d",
        "ret_1_0",
        "rmax5_rvol_21d",
        "rskew_21d"
    ],
    "Size": [
        "ami_126d",
        "dolvol_126d",
        "market_equity",
        "prc",
        "rd_me"
    ],
    "Value": [
        "at_me",
        "be_me",
        "bev_mev",
        "chcsho_12m",
        "debt_me",
        "div12m_me",
        "ebitda_mev",
        "eq_dur",
        "eqnetis_at",
        "eqnpo_12m",
        "eqnpo_me",
        "eqpo_me",
        "fcf_me",
        "ival_me",
        "netis_at",
        "ni_me",
        "ocf_me",
        "sale_me"
    ]
}

MACRO_VARS = sum(FRED_DICT.values(), [])
FIN_VARS = sum(JKP_DICT.values(), [])



def _split_dates(
        start_date,
        train_cutoff_year,
        val_months=24,
        test_months=12
):
    # Define split dates
    train_start = start_date
    train_cutoff = f'{train_cutoff_year}-12-01'
    train_end = (datetime.strptime(train_cutoff, '%Y-%m-%d') 
                 - relativedelta(months=val_months))
    val_start = train_end + relativedelta(months=1)
    val_end = val_start + relativedelta(months=(val_months-1))
    test_start = val_end + relativedelta(months=1)
    test_end = test_start + relativedelta(months=(test_months-1))

    logging.info(f"Train: {train_start} to {train_end}")
    logging.info(f"Validation: {val_start} to {val_end}")
    logging.info(f"Test: {test_start} to {test_end}")

    return train_start, train_end, val_start, val_end, test_start, test_end


def _read_and_merge_data(
        input_paths,
        targets_path
):
    
    # Read and merge inputs
    inputs = [
        pd.read_csv(
            path, 
            index_col=0, 
            parse_dates=True
        ) for path in input_paths
    ]

    # Get common indicies
    common_idx = reduce(
        lambda left, right: left.intersection(right), 
        [df.index for df in inputs]
    )

    X = pd.concat(inputs, axis=1).loc[common_idx]

    targets = pd.read_csv(targets_path, index_col=0, parse_dates=True)

    # Convert indices to datetime
    X.index = pd.to_datetime(X.index, format='%Y-%m-%d')
    targets.index = pd.to_datetime(targets.index, format='%Y-%m-%d')

    # Make index sizes consistent
    common_idx = X.index.intersection(targets.index)
    X = X.loc[common_idx]
    targets = targets.loc[common_idx]

    return X, targets

def _fractional_train_val_split(
    X,
    y,
    val_size=0.1,
    val_style='random',
    val_buffer=60
):
    
    n = X.shape[0]
    n_val = int(val_size * n)

    if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
    
        if val_style == 'last':
            X_train = X.iloc[:-n_val]
            X_val = X.iloc[-n_val:]

            y_train = y.iloc[:-n_val]
            y_val = y.iloc[-n_val:]

        elif val_style == 'first':
            X_train = X.iloc[n_val:]
            X_val = X.iloc[:n_val]

            y_train = y.iloc[n_val:]
            y_val = y.iloc[:n_val]

        elif val_style == 'random':
            val_idx = np.random.choice(n-val_buffer, size=n_val)
            mask = ~np.isin(np.arange(n), val_idx)   
            
            X_train = X.iloc[mask]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[mask]
            y_val = y.iloc[val_idx]
        
        else:
            raise TypeError(f'Unrecognized argument {val_style}'
                            'Please provide one of {"last", "first", "random"}')
        
    elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):

        if val_style == 'last':
            X_train = X[:-n_val]
            X_val = X[-n_val:]

            y_train = y[:-n_val]
            y_val = y[-n_val:]

        elif val_style == 'first':
            X_train = X[n_val:]
            X_val = X[:n_val]

            y_train = y[n_val:]
            y_val = y[:n_val]

        elif val_style == 'random':
            val_idx = np.random.choice(n-val_buffer, size=n_val)
            mask = ~np.isin(np.arange(n), val_idx)   
            
            X_train = X[mask]
            X_val = X[val_idx]

            y_train = y[mask]
            y_val = y[val_idx]
        
        else:
            raise TypeError(f'Unrecognized argument {val_style}'
                            'Please provide one of {"last", "first", "random"}')

    return X_train, X_val, y_train, y_val

def _train_test_split(
        X, 
        y,
        train_start: str|datetime,
        train_end: str|datetime,
        val_start: str|datetime,
        val_end: str|datetime,
        test_start: str|datetime,
        test_end: str|datetime
    ):

    """
    Parameters:
    *_start: start date of subsample
    *_end: end date of subsample

    Returns:
    X_*: train, val, test features
    y_*: train, val, test targets
    """

    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y have a different number of rows!')

    # Split features
    X_train = X.loc[train_start:train_end]
    X_val = X.loc[val_start:val_end]
    X_test = X.loc[test_start:test_end]

    # Split targets
    y_train = y.loc[train_start:train_end]
    y_val = y.loc[val_start:val_end]
    y_test = y.loc[test_start:test_end]

    return X_train, X_val, X_test, y_train, y_val, y_test 


def _impute_missing_features(
        X_train,
        X_val,
        X_test
):
    
    imputer = SimpleImputer()
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_val_imp = pd.DataFrame(
        imputer.transform(X_val), 
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )

    return X_train_imp, X_val_imp, X_test_imp

def _scale_features(
        X_train,
        X_val,
        X_test
):
    # Standardize features
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), 
        columns=X_val.columns, 
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )

    return X_train_scaled, X_val_scaled, X_test_scaled

def _get_feature_indicies(
        X: pd.DataFrame,
        groups: list[list[str]]
):
    idx = []
    for g in groups:
        g_idx = []
        for c in g:
            if c in X.columns:
                g_idx.append(X.columns.get_loc(c))
            else:
                continue
        idx.append(g_idx)
        
    return idx

def prepare_non_rnn_data(
        targets_path: list[str|Path],
        input_paths: list[str|Path],
        start_date: str,
        train_cutoff_year: int,
        val_months: int = 24,
        test_months: int = 12,
        target_scale_factor: int|float = 100,
        split_groups: list[list[str]] = None
    ):

    # Read and merge data
    X, targets = _read_and_merge_data(
        input_paths=input_paths,
        targets_path=targets_path
    )

    (
        train_start,
        train_end,
        val_start,
        val_end,
        test_start,
        test_end
    ) = _split_dates(
        start_date=start_date,
        train_cutoff_year=train_cutoff_year,
        val_months=val_months,
        test_months=test_months
    )

    # Split data 
    (
        X_train, X_val, X_test, 
        targets_train, targets_val, targets_test
    ) = _train_test_split(
        X, 
        targets,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end
    )

    # impute data
    X_train, X_val, X_test = _impute_missing_features(
        X_train, X_val, X_test
    )

    # Scale data
    X_train, X_val, X_test = _scale_features(
        X_train, X_val, X_test
    )

    # Split inputs into lists of groups
    if split_groups is not None:
        feat_idx = _get_feature_indicies(X_train, split_groups)
        X_train = _split_inputs_into_categories(X_train, feat_idx)
        X_val = _split_inputs_into_categories(X_val, feat_idx)
        X_test = _split_inputs_into_categories(X_test, feat_idx)

    if target_scale_factor:
        targets_train *= target_scale_factor
        targets_val *= target_scale_factor
        targets_test *= target_scale_factor

    return X_train, X_val, X_test, targets_train, targets_val, targets_test

def _split_inputs_into_categories(
        X: pd.DataFrame|np.ndarray, 
        groups: list[list[int]],
        axis: int|None = None
):  
    
    if isinstance(X, pd.DataFrame):
        return [X.iloc[:,g] for g in groups]
    elif isinstance(X, np.ndarray):
        
        if axis is None:
            result = [np.take(X, g, axis=0) for g in groups]
        else:
            result = [np.take(X, g, axis=axis) for g in groups]
        
        return result


def prepare_rnn_data(
        targets_path: list[str|Path],
        input_paths: list[str|Path],
        start_date: str,
        train_cutoff_year: int,
        val_months: int = 24,
        test_months: int = 12,
        n_timesteps: int = 12,
        target_scale_factor: int|float = 100,
        split_groups: list[list[str]] = None
    ):


    if n_timesteps <= 1:
        raise ValueError('The number of time steps must be greater than 1.')

    (
        X_train, X_val, X_test, 
        targets_train, targets_val, targets_test
    ) = prepare_non_rnn_data(
        targets_path,
        input_paths,
        start_date,
        train_cutoff_year,
        val_months,
        test_months,
        target_scale_factor=target_scale_factor
    )


    train_data = pd.concat([X_train, targets_train], axis=1)

    val_data = pd.concat([X_val, targets_val], axis=1)
    val_data = pd.concat([train_data.iloc[-(n_timesteps-1):], val_data])

    test_data = pd.concat([X_test, targets_test], axis=1)
    test_data = pd.concat([val_data.iloc[-(n_timesteps-1):], test_data])

    # Make sequences for recurrent neural nets
    X_train_rnn, targets_train_rnn = split_sequences(
        train_data,
        n_timesteps=n_timesteps,
        n_targets=targets_train.shape[1]
    ) 

    X_val_rnn, targets_val_rnn = split_sequences(
        val_data,
        n_timesteps=n_timesteps,
        n_targets=targets_train.shape[1]
    )

    X_test_rnn, targets_test_rnn = split_sequences(
        test_data,
        n_timesteps=n_timesteps,
        n_targets=targets_train.shape[1]
    )

    if split_groups is not None:
        feat_idx = _get_feature_indicies(X_train, split_groups)
        X_train_rnn = _split_inputs_into_categories(
            X_train_rnn, 
            feat_idx,
            axis=2
        )
        X_val_rnn = _split_inputs_into_categories(
            X_val_rnn, 
            feat_idx,
            axis=2
        )
        X_test_rnn = _split_inputs_into_categories(
            X_test_rnn, 
            feat_idx,
            axis=2
        )


    return (X_train_rnn, X_val_rnn, X_test_rnn, 
            targets_train_rnn, targets_val_rnn, targets_test_rnn)

def concatenate_multi_input_data(
        X1: list[pd.DataFrame|np.ndarray],
        X2: list[pd.DataFrame|np.ndarray],
        axis: int = 0
):
    
    zipped = zip(X1, X2)

    is_pandas = all(isinstance(x1, pd.DataFrame) for x1 in X1)
    is_numpy = all(isinstance(x1, np.ndarray) for x1 in X1)
    if is_pandas:
        X_c = [pd.concat([x1,x2], axis=axis) for x1,x2 in zipped]
    elif is_numpy:
        X_c = [np.concatenate([x1,x2], axis=axis) for x1,x2 in zipped]
    else:
        raise ValueError(
            'List elements must be either pd.DataFrame '
            f'or np.ndarray. Received {type(X1[0])}'
        )
    
    return X_c 