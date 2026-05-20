import numpy as np
from src.train.losses import compute_qpc, quantile_loss, make_tilted_loss, make_total_tilted_loss
from joblib import Parallel, delayed
from statsmodels.regression.quantile_regression import QuantReg
import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from statsmodels.tools import add_constant
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.layers import (
    Dense, 
    Input, 
    Concatenate, 
    LSTM, 
    GRU, 
    RNN,
    Conv1D,
    BatchNormalization, 
    LayerNormalization, 
    Dropout, 
    Activation,
    Lambda,
    Add,
    Subtract
)
from keras.regularizers import L1L2
from keras.optimizers import Adam
from keras.initializers import GlorotUniform
from typing import Union, List

from src.train.slstm import sLSTM, sLSTMCell

def get_confounding_set(X: np.ndarray, m: int, j: int) -> list:

    """
    Generates the confounding set for variable j based on the m largest correlations with other variables.

    Parameters:
    - X: 2D numpy array of shape (n_samples, n_features)
    - m: Number of variables to include in the confounding set
    - j: Index of the variable for which the confounding set is to be generated

    Returns:
    - confounding_set: List of indices representing the confounding set for variable j
    """

    idx_no_j = list(range(X.shape[1]))
    idx_no_j.remove(j)

    # Get all correlations
    corrs = np.abs(np.corrcoef(X, rowvar=False)[j, idx_no_j])

    # Take variables whose correlation with j is above the mth largest correlation
    mth_largest_corr_idx = np.argsort(corrs)[-m:]
    confounding_set = [idx_no_j[i] for i in mth_largest_corr_idx]

    return confounding_set


def fit_qpcr(X: np.ndarray, y: np.ndarray, q: float, n_updates: int=None, max_predictors: int=None, size_of_confounding_set: int=None, ebic_const: int=1):

    """
    Fits a quantile partial correlation regression model using the specified parameters.

    Parameters:
    - X: 2D numpy array of shape (n_samples, n_features) representing the predictor variables
    - y: 1D numpy array of shape (n_samples,) representing the response variable
    - q: Quantile to be estimated (e.g., 0.05, 0.5, 0.95).
    - n_updates: Number of updates to the confounding set to perform (default is [sqrt(T/log(T))])
    - max_predictors: Maximum number of predictors to include in the model (default is [T/log(T)])
    - size_of_confounding_set: Size of the confounding set to consider for each variable (default is [sqrt(T/log(T))])
    - ebic_const: Constant for the EBIC criterion (default is 1)

    Returns:
    - model: Fitted QPCR model
    """

    T = X.shape[0]
    if n_updates is None:
        n_updates = int((T / np.log(T))**(1/2))
    if size_of_confounding_set is None:
        size_of_confounding_set = int((T / np.log(T))**(1/2))
    if max_predictors is None:
        max_predictors = int(T/np.log(T))

    active_sets = [[]]
    while len(active_sets[-1]) < n_updates:

        # Set active set
        active_set = active_sets[-1]

        X_candidates_idx = [i for i in range(X.shape[1]) if i not in active_set]

        # For each candidate, update conditioning set
        def get_all_qpcs(i):
            # Update conditional set
            confounding_set = get_confounding_set(X, m=size_of_confounding_set, j=i)
            conditional_set = active_set + confounding_set
            # Compute qpc
            qpc = compute_qpc(y, X[:, conditional_set], X[:, i].reshape(-1, 1), q)
            return qpc

        all_qpcs = Parallel(n_jobs=-1)(delayed(get_all_qpcs)(i) for i in X_candidates_idx)

        # Select covariate index
        selected_idx = X_candidates_idx[np.argmax(np.abs(all_qpcs))]

        updated_active_set = active_sets[-1] + [selected_idx]
        active_sets.append(updated_active_set)

        # print(f"Updated active set: {updated_active_set}")

    active_set_dstar = active_sets[-1]

    while len(active_sets[-1]) < max_predictors:

        # Select set to search over
        X_candidates_idx = [i for i in range(X.shape[1]) if i not in active_sets[-1]]

        def get_all_qpcs(i):
            # Update conditional set
            confounding_set = get_confounding_set(X, m=size_of_confounding_set, j=i)
            conditional_set = active_set_dstar + confounding_set
            # Compute qpc
            qpc = compute_qpc(y, X[:, conditional_set], X[:, i].reshape(-1, 1), q)
            return qpc

        # Select covariate index
        all_qpcs = Parallel(n_jobs=-1)(delayed(get_all_qpcs)(i) for i in X_candidates_idx)
        selected_idx = X_candidates_idx[np.argmax(np.abs(all_qpcs))]
        updated_active_set = active_sets[-1] + [selected_idx]
        active_sets.append(updated_active_set)
        # print(f"Updated active set: {updated_active_set}")
    
    losses = []
    active_sets = active_sets[1:]  # Remove empty set
    for model_candidate in active_sets:

        # Fit qreg 
        X_active = X[:, model_candidate]
        X_active_const = add_constant(X_active, has_constant='skip')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=IterationLimitWarning)
            reg = QuantReg(y, X_active_const).fit(q=q)

        # Compute loss 
        loss = np.log(quantile_loss(y - reg.predict(X_active_const), q).mean()) + ebic_const * (np.log(X.shape[0]) * np.log(len(model_candidate))) / X.shape[0]
        losses.append(loss)

    # Fit best model
    best_model_idx = np.argmin(losses)
    X_active = X[:, active_sets[best_model_idx]]
    X_active_const = add_constant(X_active, has_constant='skip')
    reg = QuantReg(y, X_active_const).fit(q=q)

    return reg, active_sets[best_model_idx]



def build_qlr(q: float=0.5, l1: float=0.0, l2: float=0.0, lr: float=0.001):
    model = Sequential()
    model.add(Dense(int(1), activation = 'linear', kernel_regularizer=L1L2(l1=l1,l2=l2)))
    opt = Adam(learning_rate=lr)
    loss = make_tilted_loss(q)
    model.compile(loss = loss, optimizer = opt)
    return model


def build_nn(q: Union[float, int]=0.5, n_dense_layers: int=1, n_nodes: int=16,
             l1: float=0.0, l2: float=0.0, lr: float=0.001, norm_fn: str='batch'):
    """
    Builds a single task quantile regression neural network.

    Parameters
    ----------
    l1: float (default=0.0)
        The l1 penalty for the neural network weights
    l2: float (default=0.0)
        The l2 penalty for the neural network weights
    q: float in (0.0,1.0) (default=0.5, i.e., the median)
        The quantile of interest used in defining the tilted loss.
    lr: float (default=0.001)
        The initial learning rate used for the optimization algorithm.
    n_dense_layers: int (default=1)
        The number of dense layers in the model.
    n_nodes: int (default=32)
        The number of nodes in each shared layer.
        
    Returns
    ----------
    model: a compiled keras model
    """

    norm_fn = norm_fn.lower()
    if norm_fn == 'batch':
        norm_fn = BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")

    model = Sequential()
    for i in range(1,n_dense_layers+1):
        model.add(Dense(n_nodes, 'relu', kernel_regularizer=L1L2(l1,l2)))
        if i<n_dense_layers:
            model.add(norm_fn())

    model.add(Dense(int(1), activation='linear', kernel_regularizer=L1L2(l1,l2)))
    
    opt = Adam(learning_rate=lr)
    loss = make_tilted_loss(q)
    model.compile(loss = loss, optimizer = opt)
    return model


def build_rnn(q: Union[float, int]=0.5, n_recurrent_layers: int=2, n_dense_layers: int=1, n_nodes: int=32,
               l1: float=0.0, l2: float=0.0, lr: float=0.001, recurrent_layer_type: str='gru'):
    """
    Builds a single task quantile regression recurrent neural network.

    Parameters
    ----------
    l1: float (default=0.0)
        The l1 penalty for the neural network weights
    l2: float (default=0.0)
        The l2 penalty for the neural network weights
    q: float in (0.0,1.0) (default=0.5, i.e., the median)
        The quantile of interest used in defining the tilted loss.
    lr: float (default=0.001)
        The initial learning rate used for the optimization algorithm.
    n_recurrent_layers: int (default=1)
        The number of recurrent layers in the model.
    n_nodes: int (default=32)
        The number of nodes in each recurrent layer.
    n_dense_layers: int (default=1)
        The number of shared layers in the model.
    n_nodes: int (default=32)
        The number of nodes in each shared layer.
        
    Returns
    ----------
    model: a compiled keras model
    """
    recurrent_layer_type = recurrent_layer_type.lower()
    if recurrent_layer_type == 'lstm':
        recurrent_layer_type = LSTM
    elif recurrent_layer_type == 'gru':    
        recurrent_layer_type = GRU

    # Recurrent layers
    model = Sequential()
    for i in range(1,n_recurrent_layers+1):
        model.add(recurrent_layer_type(n_nodes, return_sequences=(i < n_recurrent_layers), kernel_regularizer=L1L2(l1,l2)))

    # Dense layers
    for i in range(1,n_dense_layers+1):
        model.add(Dense(n_nodes, 'relu', kernel_regularizer=L1L2(l1,l2)))

    # Output layer    
    model.add(Dense(int(1), activation='linear', kernel_regularizer=L1L2(l1,l2)))
    
    opt = Adam(learning_rate=lr)
    loss = make_tilted_loss(q)
    model.compile(loss = loss, optimizer = opt)
    return model


def build_mq_v0(input_shape: tuple, n_shared_layers: int=1, n_qtask_layers: int=2, n_nodes: int=32, l1: float=0.0, l2: float=0.0, lr: float=0.001, norm_fn: str='batch', quantiles: list[int]=[0.05,0.25,0.50,0.75,0.95], task_specific_norm: bool=False):

    if norm_fn == 'batch':
        norm_fn = BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")

    inputs = Input(shape=input_shape)

    shared_layers = []
    for i in range(1, n_shared_layers + 1):
        shared_layers.append(
            Dense(n_nodes, activation='relu', kernel_regularizer=L1L2(l1,l2))
        )
        shared_layers.append(norm_fn())

    shared_net = Sequential(shared_layers, name='shared')(inputs)

    outputs = []
    for q in quantiles:
        name = f"Q{q}"
        
        qtask_layers = []
        for i in range(1, n_qtask_layers+1):
            qtask_layers.append(
                Dense(
                    n_nodes, 
                    activation='relu',
                    kernel_regularizer=L1L2(l1, l2)
                )
            )
            if task_specific_norm and i < n_qtask_layers:
                qtask_layers.append(norm_fn())

        # Append output node
        qtask_layers.append(
            Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2))
        )
        
        # Build output net
        output_q = Sequential(qtask_layers, name=name)(shared_net)

        outputs.append(output_q)

    out_concat = Concatenate(name='out_layer')(outputs)

    model = Model(inputs=inputs, outputs=out_concat)

    loss = make_total_tilted_loss(quantiles)

    model.compile(
        loss=loss, 
        optimizer=Adam(learning_rate=lr) 
    )

    return model


def build_mq(input_shapes: Union[tuple, list[tuple]], n_input_processing_layers: int=2, n_shared_layers: int=1, n_qtask_layers: int=2, n_nodes: int=32, l1: float=0.0, l2: float=0.0,  lr: float=0.001, norm_fn: str='batch', quantiles: list[int]=[0.05,0.25,0.50,0.75,0.95], task_specific_norm: bool=False):

    if norm_fn == 'batch':
        norm_fn = BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")

    inputs = []
    input_processing_nets = []
    for i, shape in enumerate(input_shapes):
        net_input = Input(shape=shape)
        inputs.append(net_input)
        
        input_processing_layers = []
        # build input processing layers
        for j in range(1, n_input_processing_layers + 1):
            input_processing_layers.append(Dense(n_nodes, activation='relu', kernel_regularizer=L1L2(l1,l2)))
            input_processing_layers.append(norm_fn())

        # Make model
        input_processing_net = Sequential(input_processing_layers, name=f'input_processing_{i}')(net_input)
        input_processing_nets.append(input_processing_net)

    # Concatenate the layers from each input
    concat = Concatenate()(input_processing_nets)

    shared_layers = []
    for i in range(1, n_shared_layers + 1):
        shared_layers.append(
            Dense(
                n_nodes, 
                activation='relu', 
                kernel_regularizer=L1L2(l1,l2)
            )
        )
        shared_layers.append(norm_fn())


    shared_net = Sequential(shared_layers, name='shared')(concat)


    outputs = []
    for q in quantiles:
        name = f"Q{q}"
        
        qtask_layers = []
        for i in range(1, n_qtask_layers+1):
            qtask_layers.append(
                Dense(
                    n_nodes, 
                    activation='relu',
                    kernel_regularizer=L1L2(l1, l2)
                )
            )
            if task_specific_norm and i < n_qtask_layers:
                qtask_layers.append(norm_fn())

        # Append output node
        qtask_layers.append(
            Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2))
        )
        
        # Build output net
        output_q = Sequential(qtask_layers, name=name)(shared_net)

        outputs.append(output_q)

    out_concat = Concatenate(name='out_layer')(outputs)

    model = Model(inputs=inputs, outputs=out_concat)

    loss = make_total_tilted_loss(quantiles)

    model.compile(
        loss=loss, 
        optimizer=Adam(learning_rate=lr) 
    )

    return model

def build_dmq_v1(
        input_shape: tuple, 
        n_recurrent_layers: int=2, 
        n_shared_layers: int=1, 
        n_qtask_layers: int=2, 
        n_recurrent_nodes: int=32,
        n_shared_nodes: int=32,
        n_task_nodes: int=32,
        l1: float=0.0, 
        l2: float=0.0, 
        lr: float=0.001, 
        rec_drop: float=0.0,
        dropout: float=0.0,
        norm_fn: str='layer', 
        recurrent_layer_type: str='gru', 
        lower_quantiles: List[float]=[0.05,0.25], 
        upper_quantiles: List[float]=[0.75,0.95],
        recurrent_norm: bool=False,
        shared_norm: bool=False, 
        task_norm: bool=False, 
        loss_weights: list[float]=[1.0]*5,
        seed: int=1
    ):

    """
    DMQv0 + quantile spacings
    """

    initializer = GlorotUniform(seed=seed)
    lower_quantiles = sorted(lower_quantiles)
    upper_quantiles = sorted(upper_quantiles)
    quantiles = lower_quantiles + [0.5] + upper_quantiles
    if len(loss_weights) != len(quantiles):
        loss_weights = [1.0] * len(quantiles)

    norm_fn = norm_fn.lower()
    if norm_fn == 'batch':
        norm_fn = BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")
    
    recurrent_layer_type = recurrent_layer_type.lower()
    if recurrent_layer_type == 'lstm':
        recurrent_layer_type = LSTM
    elif recurrent_layer_type == 'gru':   
        recurrent_layer_type = GRU
    else:
        raise ValueError("recurrent_layer_type must be 'lstm' or 'gru'")

    inputs = Input(shape=input_shape)

    shared_layers = []
    
    for i in range(1, n_recurrent_layers + 1):
        shared_layers.append(
            recurrent_layer_type(
                n_recurrent_nodes, 
                return_sequences=(i < n_recurrent_layers), 
                kernel_regularizer=L1L2(l1,l2), 
                recurrent_dropout=rec_drop,
                kernel_initializer=initializer
            )
        )
        if recurrent_norm:
            shared_layers.append(norm_fn())

    
    for i in range(1, n_shared_layers + 1):
        shared_layers.append(
            Dense(
                n_shared_nodes, 
                activation='relu', 
                kernel_regularizer=L1L2(l1,l2),
                kernel_initializer=initializer
            )
        )
        if shared_norm:
            shared_layers.append(norm_fn())

    shared_net = Sequential(shared_layers, name='shared')(inputs)

    # Median head
    median_head = Sequential(name='Q50')
    for i in range(1, n_qtask_layers+1):
        median_head.add(
            Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=L1L2(l1, l2),
                kernel_initializer=initializer
            )
        )
        if task_norm and i < n_qtask_layers:
            median_head.add(norm_fn())
    
        if dropout > 0.0 and i < n_qtask_layers:
            median_head.add(Dropout(dropout))

    median_head.add(
        Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2))
    )

    median_output = median_head(shared_net)

    # Lower quantile heads (monotonic chain)
    lower_outputs = []
    prev = median_output
    for q in sorted(lower_quantiles, reverse=True):
        qtask_layers = []
        for i in range(1, n_qtask_layers+1):
            qtask_layers.append(
                Dense(
                    n_task_nodes,
                    activation='relu',
                    kernel_regularizer=L1L2(l1, l2),
                    kernel_initializer=initializer
                )
            )
            if task_norm and i < n_qtask_layers:
                qtask_layers.append(norm_fn())
            if dropout > 0.0 and i < n_qtask_layers:
                qtask_layers.append(Dropout(dropout))

        qtask_layers.append(
            Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
        )
        q_task_resid = Sequential(qtask_layers, name=f'Q{int(q*100)}_lower_raw')(shared_net)
        q_out = Subtract(name=f'Q{int(q*100)}_from_prev')([prev, Activation('softplus')(q_task_resid)])
        lower_outputs.append(q_out)
        prev = q_out

    # Upper quantile heads (monotonic chain)
    upper_outputs = []
    prev = median_output
    for q in upper_quantiles:
        qtask_layers = []
        for i in range(1, n_qtask_layers+1):
            qtask_layers.append(
                Dense(
                    n_task_nodes,
                    activation='relu',
                    kernel_regularizer=L1L2(l1, l2),
                    kernel_initializer=initializer
                )
            )
            if task_norm and i < n_qtask_layers:
                qtask_layers.append(norm_fn())
            if dropout > 0.0 and i < n_qtask_layers:
                qtask_layers.append(Dropout(dropout))

        qtask_layers.append(
            Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
        )
        q_task_resid = Sequential(qtask_layers, name=f'Q{int(q*100)}_upper_raw')(shared_net)
        q_out = Add(name=f'Q{int(q*100)}_from_prev')([prev, Activation('softplus')(q_task_resid)])
        upper_outputs.append(q_out)
        prev = q_out

    outputs = list(reversed(lower_outputs)) + [median_output] + upper_outputs
    out_concat = Concatenate(name='out_layer')(outputs)

    model = Model(inputs=inputs, outputs=out_concat)

    loss = make_total_tilted_loss(quantiles, q_loss_weights=loss_weights)

    model.compile(
        loss=loss, 
        optimizer=Adam(learning_rate=lr),
    )

    return model

def build_dmq_v0(
        input_shape: tuple, 
        n_recurrent_layers: int=1, 
        n_shared_layers: int=1, 
        n_qtask_layers: int=1, 
        n_recurrent_nodes: int=32,
        n_shared_nodes: int=32,
        n_task_nodes: int=32,
        l1: float=0.0, 
        l2: float=0.0, 
        lr: float=0.001, 
        rec_drop: float=0.0,
        dropout: float=0.0,
        norm_fn: str='layer', 
        recurrent_layer_type: str='gru', 
        lower_quantiles: List[float] = [0.05, 0.25],
        upper_quantiles: List[float] = [0.75, 0.95], 
        recurrent_norm: bool=False,
        shared_norm: bool=False,
        task_norm: bool=False, 
        loss_weights: list[float]=[1.0]*5,
        seed: int=1
    ):

    """
    Base DMQ model
    """

    initializer = GlorotUniform(seed=seed)
    lower_quantiles = sorted(lower_quantiles)
    upper_quantiles = sorted(upper_quantiles)
    quantiles = lower_quantiles + [0.5] + upper_quantiles
    if len(loss_weights) != len(quantiles):
        loss_weights = [1.0] * len(quantiles)

    norm_fn = norm_fn.lower()
    if norm_fn == 'batch':
        norm_fn = BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")
    
    recurrent_layer_type = recurrent_layer_type.lower()
    if recurrent_layer_type == 'lstm':
        recurrent_layer_type = LSTM
    elif recurrent_layer_type == 'gru':   
        recurrent_layer_type = GRU
    elif recurrent_layer_type == 'slstm':
        pass
    else:
        raise ValueError("recurrent_layer_type must be 'lstm', 'slstm', or 'gru'")

    inputs = Input(shape=input_shape)

    shared_layers = []
    
    for i in range(1, n_recurrent_layers + 1):
    
        if recurrent_layer_type == 'slstm':
            shared_layers.append(
                RNN(
                    sLSTMCell(
                        n_recurrent_nodes,
                        kernel_regularizer=L1L2(l1,l2),
                        kernel_initializer=initializer
                    ), 
                    return_sequences=(i < n_recurrent_layers)
                )
            )
        else:
            shared_layers.append(
                recurrent_layer_type(
                    n_recurrent_nodes, 
                    return_sequences=(i < n_recurrent_layers), 
                    kernel_regularizer=L1L2(l1,l2), 
                    recurrent_dropout=rec_drop,
                    kernel_initializer=initializer
                )
            )
        
        if recurrent_norm:
            shared_layers.append(norm_fn())

    for i in range(1, n_shared_layers + 1):
        shared_layers.append(
            Dense(
                n_shared_nodes, 
                activation='relu', 
                kernel_regularizer=L1L2(l1,l2),
                kernel_initializer=initializer
            )
        )
        if shared_norm:
            shared_layers.append(norm_fn())

    shared_net = Sequential(shared_layers, name='shared')(inputs)

    outputs = []
    for q in quantiles:
        qtask_layers = []
        for i in range(1, n_qtask_layers + 1):
            qtask_layers.append(
                Dense(
                    n_task_nodes,
                    activation='relu',
                    kernel_regularizer=L1L2(l1, l2),
                    kernel_initializer=initializer,
                )
            )
            if task_norm and i < n_qtask_layers:
                qtask_layers.append(norm_fn())
            if dropout > 0.0 and i < n_qtask_layers:
                qtask_layers.append(Dropout(dropout))

        qtask_layers.append(
            Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
        )

        q_out = Sequential(qtask_layers, name=f'Q{int(q*100)}_head')(shared_net)
        outputs.append(q_out)

    out_concat = Concatenate(name='out_layer')(outputs)

    model = Model(inputs=inputs, outputs=out_concat)

    loss = make_total_tilted_loss(quantiles, q_loss_weights=loss_weights)

    model.compile(
        loss=loss, 
        optimizer=Adam(learning_rate=lr),
    )

    return model

# Old

# def build_dmq_v5(
#         input_shape: tuple, 
#         n_recurrent_layers: int=1,
#         n_shared_layers: int=1, 
#         n_qtask_layers: int=1, 
#         n_conv_filters: int=32,
#         kernel_size: int=12,
#         n_recurrent_nodes: int=32,
#         n_shared_nodes: int=32,
#         n_task_nodes: int=32,
#         l1: float=0.0, 
#         l2: float=0.0, 
#         rec_drop: float=0.0,
#         dropout: float=0.0,
#         lr: float=0.001, 
#         norm_fn: str='layer', 
#         recurrent_layer_type: str='gru', 
#         lower_quantiles: List[float]=[0.05,0.25], 
#         upper_quantiles: List[float]=[0.75,0.95],
#         recurrent_norm: bool=False,
#         shared_norm: bool=False, 
#         task_norm: bool=False, 
#         loss_weights: list[float]=[1.0]*5,
#         seed: int=1
#     ):

#     """
#     DMQv0 + quantile spacing and Conv1D at the beginning. Should be used with more timesteps
#     """

#     initializer = GlorotUniform(seed=seed)

#     norm_fn = norm_fn.lower()
#     if norm_fn == 'batch':
#         norm_fn = BatchNormalization
#     elif norm_fn == 'layer':    
#         norm_fn = LayerNormalization
#     else:
#         raise ValueError("norm_fn must be 'batch' or 'layer'")
    
#     recurrent_layer_type = recurrent_layer_type.lower()
#     if recurrent_layer_type == 'lstm':
#         recurrent_layer_type = LSTM
#     elif recurrent_layer_type == 'gru':   
#         recurrent_layer_type = GRU
#     else:
#         raise ValueError("recurrent_layer_type must be 'lstm' or 'gru'")

#     inputs = Input(shape=input_shape)

#     shared_layers = []

#     shared_layers.append(
#         Conv1D(
#             32,
#             12,
#             activation='relu',
#             padding='causal'
#         )
#     )
#     shared_layers.append(
#         Conv1D(
#             64,
#             12,
#             activation='relu',
#             padding='causal'
#         )
#     )

#     for i in range(1, n_recurrent_layers + 1):
#         shared_layers.append(
#             recurrent_layer_type(
#                 n_recurrent_nodes, 
#                 return_sequences=(i < n_recurrent_layers), 
#                 kernel_regularizer=L1L2(l1,l2), 
#                 recurrent_dropout=rec_drop,
#                 kernel_initializer=initializer
#             )
#         )
#         if recurrent_norm:
#             shared_layers.append(norm_fn())

    
#     for i in range(1, n_shared_layers + 1):
#         shared_layers.append(
#             Dense(
#                 n_shared_nodes, 
#                 activation='relu', 
#                 kernel_regularizer=L1L2(l1,l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if shared_norm:
#             shared_layers.append(norm_fn())

#     # shared_layers.append(norm_fn())

#     shared_net = Sequential(shared_layers, name='shared')(inputs)

#     # Median head
#     median_head = Sequential(name='Q50')
#     for i in range(1, n_qtask_layers+1):
#         median_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             median_head.add(norm_fn())
#         if dropout > 0.0 and i < n_qtask_layers:
#             median_head.add(Dropout(dropout))

#     median_head.add(
#         Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2))
#     )

#     median_output = median_head(shared_net)

#     # Lower quantile head
#     lower_resid_head = Sequential(name='Q_lower_raw')
#     for i in range(1, n_qtask_layers+1):
#         lower_resid_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             lower_resid_head.add(norm_fn())
#         if dropout > 0.0 and i < n_qtask_layers:
#             lower_resid_head.add(Dropout(dropout))

#     lower_resid_head.add(
#         Dense(len(lower_quantiles), activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
#     )
#     lower_raw = lower_resid_head(shared_net)
#     lower_resid = Activation('softplus')(lower_raw)


#     # Upper quantile head
#     upper_resid_head = Sequential(name='Q_upper_raw')
#     for i in range(1, n_qtask_layers+1):
#         upper_resid_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             upper_resid_head.add(norm_fn())
#         if dropout > 0.0 and i < n_qtask_layers:    
#             upper_resid_head.add(Dropout(dropout))
        
#     upper_resid_head.add(
#         Dense(len(upper_quantiles), activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
#     )
#     upper_raw = upper_resid_head(shared_net)
#     upper_resid = Activation('softplus')(upper_raw)

#     # Combine outputs
#     Q50 = median_output
#     Q_lower = Q50 - lower_resid
#     Q_upper = Q50 + upper_resid

#     out_concat = Concatenate()([Q_lower, Q50, Q_upper])

#     model = Model(inputs=inputs, outputs=out_concat)

#     loss = make_total_tilted_loss(lower_quantiles + [0.5] + upper_quantiles, q_loss_weights=loss_weights)

#     model.compile(
#         loss=loss, 
#         optimizer=Adam(learning_rate=lr),
#     )

#     return model


# def build_dmq_v4(
#         input_shape: tuple, 
#         n_recurrent_layers: int=2, 
#         n_shared_layers: int=1, 
#         n_qtask_layers: int=2, 
#         n_recurrent_nodes: int=32,
#         n_shared_nodes: int=32,
#         n_task_nodes: int=32,
#         l1: float=0.0, 
#         l2: float=0.0, 
#         lr: float=0.001, 
#         rec_drop: float=0.0,
#         dropout: float=0.0,
#         norm_fn: str='layer', 
#         recurrent_layer_type: str='gru', 
#         lower_quantiles: List[float]=[0.05,0.25], 
#         upper_quantiles: List[float]=[0.75,0.95],
#         recurrent_norm: bool=False,
#         shared_norm: bool=False, 
#         task_specific_norm: bool=False, 
#         loss_weights: list[float]=[1.0]*5,
#         seed: int=1
#     ):

#     """
#     DMQv2 + skip connection to task layers
#     """

#     initializer = GlorotUniform(seed=seed)

#     norm_fn = norm_fn.lower()
#     if norm_fn == 'batch':
#         norm_fn = BatchNormalization
#     elif norm_fn == 'layer':    
#         norm_fn = LayerNormalization
#     else:
#         raise ValueError("norm_fn must be 'batch' or 'layer'")
    
#     recurrent_layer_type = recurrent_layer_type.lower()
#     if recurrent_layer_type == 'lstm':
#         recurrent_layer_type = LSTM
#     elif recurrent_layer_type == 'gru':   
#         recurrent_layer_type = GRU
#     else:
#         raise ValueError("recurrent_layer_type must be 'lstm' or 'gru'")


#     inputs = Input(shape=input_shape)

#     # ============================================================
#     # LEARNABLE RESIDUAL SKIP (corrected)
#     # ============================================================

#     # last time step of raw input
#     skip_input = Lambda(lambda x: x[:, -1, :], output_shape=lambda s: (s[0],s[2]), name="skip_raw")(inputs)

#     # normalize skip so spikes don't dominate
#     skip_norm = norm_fn(name="skip_norm")(skip_input)

#     # learnable linear projection (zero initialized)
#     skip_proj = Dense(
#         n_shared_nodes,
#         activation="linear",
#         kernel_initializer='zeros',
#         bias_initializer='zeros',
#         name="skip_projection"
#     )(skip_norm)

#     # ============================================================
#     # SHARED LAYERS
#     # ============================================================

#     shared_layers = []
    
#     for i in range(1, n_recurrent_layers + 1):
#         shared_layers.append(
#             recurrent_layer_type(
#                 n_recurrent_nodes, 
#                 return_sequences=(i < n_recurrent_layers), 
#                 kernel_regularizer=L1L2(l1,l2), 
#                 kernel_initializer=initializer,
#                 recurrent_dropout=rec_drop
#             )
#         )
#         if recurrent_norm:
#             shared_layers.append(norm_fn())

    
#     for i in range(1, n_shared_layers + 1):
#         shared_layers.append(
#             Dense(
#                 n_shared_nodes, 
#                 activation='relu', 
#                 kernel_regularizer=L1L2(l1,l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if shared_norm:
#             shared_layers.append(norm_fn())

#     # shared_layers.append(norm_fn())

#     shared_base = Sequential(shared_layers, name='shared')(inputs)
#     shared_net = Add(name="shared_plus_skip")([shared_base, skip_proj])

#     # ====================================================
#     # MEDIAN HEAD
#     # ====================================================

#     median_head = Sequential(name='Q50')
#     for i in range(1, n_qtask_layers+1):
#         median_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             median_head.add(norm_fn())
        
#         if dropout > 0.0 and i < n_qtask_layers:
#             median_head.add(Dropout(dropout))

#     median_head.add(
#         Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2))
#     )

#     median_output = median_head(shared_net)

#     # ====================================================
#     # LOWER QUANTILE HEADS
#     # ====================================================

#     fifth_resid_head = Sequential(name='Q5_lower_raw')
#     for i in range(1, n_qtask_layers+1):
#         fifth_resid_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             fifth_resid_head.add(norm_fn())
        
#         if dropout > 0.0 and i < n_qtask_layers:
#             fifth_resid_head.add(Dropout(dropout))

#     fifth_resid_head.add(
#         Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
#     )
#     fifth_raw = fifth_resid_head(shared_net)
#     fifth_resid = Activation('softplus')(fifth_raw)

#     twentyfifth_resid_head = Sequential(name='Q25_lower_raw')
#     for i in range(1, n_qtask_layers+1):
#         twentyfifth_resid_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             twentyfifth_resid_head.add(norm_fn())

#         if dropout > 0.0 and i < n_qtask_layers:
#             twentyfifth_resid_head.add(Dropout(dropout))

#     twentyfifth_resid_head.add(
#         Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
#     )
#     twentyfifth_raw = twentyfifth_resid_head(shared_net)
#     twentyfifth_resid = Activation('softplus')(twentyfifth_raw)


#     # ====================================================
#     # UPPER QUANTILE HEADS
#     # ====================================================
#     ninetyfifth_resid_head = Sequential(name='Q95_upper_raw')
#     for i in range(1, n_qtask_layers+1):
#         ninetyfifth_resid_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             ninetyfifth_resid_head.add(norm_fn())

#         if dropout > 0.0 and i < n_qtask_layers:
#             ninetyfifth_resid_head.add(Dropout(dropout))

#     ninetyfifth_resid_head.add(
#         Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
#     )
#     ninetyfifth_raw = ninetyfifth_resid_head(shared_net)
#     ninetyfifth_resid = Activation('softplus')(ninetyfifth_raw)

#     # Upper quantile head
#     seventyfifth_resid_head = Sequential(name='Q75_upper_raw')
#     for i in range(1, n_qtask_layers+1):
#         seventyfifth_resid_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             seventyfifth_resid_head.add(norm_fn())
        
#         if dropout > 0.0 and i < n_qtask_layers:
#             seventyfifth_resid_head.add(Dropout(dropout))
    
#     seventyfifth_resid_head.add(
#         Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
#     )
#     seventyfifth_raw = seventyfifth_resid_head(shared_net)
#     seventyfifth_resid = Activation('softplus')(seventyfifth_raw)

#     # ====================================================
#     # OUTPUTS
#     # ====================================================

#     # Combine outputs
#     Q50 = median_output
#     Q5 = Q50 - fifth_resid
#     Q25 = Q50 - twentyfifth_resid
#     Q75 = Q50 + seventyfifth_resid
#     Q95 = Q50 + ninetyfifth_resid

#     out_concat = Concatenate()([Q5, Q25, Q50, Q75, Q95])

#     model = Model(inputs=inputs, outputs=out_concat)

#     loss = make_total_tilted_loss(lower_quantiles + [0.5] + upper_quantiles, q_loss_weights=loss_weights)

#     model.compile(
#         loss=loss, 
#         optimizer=Adam(learning_rate=lr),
#     )

#     return model


# def build_dmq_v3(
#     input_shape: tuple,
#     n_recurrent_layers: int = 2,
#     n_shared_layers: int = 1,
#     n_qtask_layers: int = 2,
#     n_recurrent_nodes: int = 32,
#     n_shared_nodes: int = 32,
#     n_task_nodes: int = 32,
#     l1: float = 0.0,
#     l2: float = 0.0,
#     lr: float = 0.001,
#     rec_drop: float=0.0,
#     dropout: float=0.0,
#     resid_scale: float=1.0,
#     norm_fn: str = 'layer',
#     recurrent_layer_type: str = 'gru',
#     lower_quantiles: List[float] = [0.05, 0.25],
#     upper_quantiles: List[float] = [0.75, 0.95],
#     recurrent_norm: bool = False,
#     shared_norm: bool = False,
#     task_specific_norm: bool = False,
#     loss_weights: list[float] = [1.0] * 5,
#     seed: int=1
# ):

#     """
#     DMQv1 + skip connection to task layers
#     """

#     initializer = GlorotUniform(seed=seed)

#     # ----- Normalization selection -----
#     norm_fn = norm_fn.lower()
#     if norm_fn == 'batch':
#         norm_fn = BatchNormalization
#     elif norm_fn == 'layer':
#         norm_fn = LayerNormalization
#     else:
#         raise ValueError("norm_fn must be 'batch' or 'layer'")

#     # ----- RNN Type -----
#     recurrent_layer_type = recurrent_layer_type.lower()
#     if recurrent_layer_type == 'lstm':
#         rnn_cls = LSTM
#     elif recurrent_layer_type == 'gru':
#         rnn_cls = GRU
#     else:
#         raise ValueError("recurrent_layer_type must be 'lstm' or 'gru'")

#     # ----- Inputs -----
#     inputs = Input(shape=input_shape)

#     # ============================================================
#     # LEARNABLE RESIDUAL SKIP (corrected)
#     # ============================================================

#     # last time step of raw input
#     skip_input = Lambda(lambda x: x[:, -1, :], output_shape=lambda s: (s[0],s[2]), name="skip_raw")(inputs)

#     # normalize skip so spikes don't dominate
#     skip_norm = norm_fn(name="skip_norm")(skip_input)

#     # learnable linear projection (zero initialized)
#     skip_proj = Dense(
#         n_shared_nodes,
#         activation="linear",
#         kernel_initializer='zeros',
#         bias_initializer='zeros',
#         name="skip_projection"
#     )(skip_norm)

#     skip_proj = skip_proj * resid_scale

#     # ============================================================
#     # 2. SHARED REPRESENTATION
#     # ============================================================

#     shared_layers = []
#     for i in range(1, n_recurrent_layers + 1):

#         return_seq = (i < n_recurrent_layers)

#         shared_layers.append(
#             rnn_cls(
#                 n_recurrent_nodes,
#                 return_sequences=return_seq,
#                 kernel_regularizer=L1L2(l1, l2),
#                 recurrent_dropout=rec_drop,
#                 kernel_initializer=initializer,
#                 name=f"rnn_{i}"
#             )
#         )
#         if recurrent_norm:
#             shared_layers.append(norm_fn())

#     for i in range(1, n_shared_layers + 1):
#         shared_layers.append(
#             Dense(
#                 n_shared_nodes,
#                 activation="relu",
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer,
#                 name=f"shared_dense_{i}"
#             )
#         )
#         if shared_norm:
#             shared_layers.append(norm_fn())

#     shared_base = Sequential(shared_layers, name="shared_base")(inputs)

#     # Add skip (true residual)
#     shared_net = Add(name="shared_plus_skip")([shared_base, skip_proj])

#     # ============================================================
#     # 3. MEDIAN HEAD
#     # ============================================================

#     median_head = Sequential(name="Q50_head")
#     for i in range(1, n_qtask_layers + 1):
#         median_head.add(Dense(n_task_nodes, activation="relu",
#                               kernel_regularizer=L1L2(l1, l2),
#                               kernel_initializer=initializer,
#                               name=f"median_dense_{i}"))
#         if task_specific_norm and i < n_qtask_layers:
#             median_head.add(norm_fn(name=f"median_norm_{i}"))
        
#         if dropout > 0.0 and i < n_qtask_layers:
#             median_head.add(Dropout(dropout))

#     median_head.add(
#         Dense(1, activation="linear", kernel_regularizer=L1L2(l1, l2),
#               kernel_initializer=initializer,
#               name="median_output")
#     )

#     Q50 = median_head(shared_net)

#     # ============================================================
#     # 4. LOWER QUANTILES
#     # ============================================================

#     lower_resid_head = Sequential(name="Q_lower_head")
#     for i in range(1, n_qtask_layers + 1):
#         lower_resid_head.add(
#             Dense(n_task_nodes, activation="relu",
#                   kernel_regularizer=L1L2(l1, l2),
#                   kernel_initializer=initializer,
#                   name=f"lower_dense_{i}")
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             lower_resid_head.add(norm_fn(name=f"lower_norm_{i}"))
        
#         if dropout > 0.0 and i < n_qtask_layers:
#             lower_resid_head.add(Dropout(dropout))

#     lower_resid_head.add(
#         Dense(len(lower_quantiles), activation="linear",
#               kernel_regularizer=L1L2(l1, l2),
#               kernel_initializer=initializer,
#               name="lower_raw")
#     )

#     lower_raw = lower_resid_head(shared_net)
#     lower_resid = Activation('softplus')(lower_raw)
#     Q_lower = Subtract(name="Q_lower")([Q50, lower_resid])

#     # ============================================================
#     # 5. UPPER QUANTILES
#     # ============================================================

#     upper_resid_head = Sequential(name="Q_upper_head")
#     for i in range(1, n_qtask_layers + 1):
#         upper_resid_head.add(
#             Dense(n_task_nodes, activation="relu",
#                   kernel_regularizer=L1L2(l1, l2),
#                   kernel_initializer=initializer,
#                   name=f"upper_dense_{i}")
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             upper_resid_head.add(norm_fn(name=f"upper_norm_{i}"))
        
#         if dropout > 0.0 and i < n_qtask_layers:
#             upper_resid_head.add(Dropout(dropout))

#     upper_resid_head.add(
#         Dense(len(upper_quantiles), activation="linear",
#               kernel_regularizer=L1L2(l1, l2),
#               kernel_initializer=initializer,
#               name="upper_raw")
#     )

#     upper_raw = upper_resid_head(shared_net)
#     upper_resid = Activation('softplus')(upper_raw)
#     Q_upper = Add(name="Q_upper")([Q50, upper_resid])

#     # ============================================================
#     # 6. CONCAT OUTPUT
#     # ============================================================

#     outputs = Concatenate(name="all_quantiles")([Q_lower, Q50, Q_upper])

#     model = Model(inputs=inputs, outputs=outputs)

#     # correct loss function
#     loss = make_total_tilted_loss(
#         lower_quantiles + [0.5] + upper_quantiles,
        
#         q_loss_weights=loss_weights
#     )

#     model.compile(
#         loss=loss,
#         optimizer=Adam(learning_rate=lr),
#     )

#     return model


# def build_dmq_v2(
#         input_shape: tuple, 
#         n_recurrent_layers: int=2, 
#         n_shared_layers: int=1, 
#         n_qtask_layers: int=2, 
#         n_recurrent_nodes: int=32,
#         n_shared_nodes: int=32,
#         n_task_nodes: int=32,
#         l1: float=0.0, 
#         l2: float=0.0, 
#         lr: float=0.001, 
#         rec_drop: float=0.0,
#         dropout: float=0.0,
#         norm_fn: str='layer', 
#         recurrent_layer_type: str='gru', 
#         lower_quantiles: List[float]=[0.05,0.25], 
#         upper_quantiles: List[float]=[0.75,0.95],
#         recurrent_norm: bool=False,
#         shared_norm: bool=False, 
#         task_specific_norm: bool=False, 
#         loss_weights: list[float]=[1.0]*5,
#         seed: int=1
#     ):

#     """
#     DMQv0 + quantile spacing with separate heads for each quantile.
#     """

#     initializer = GlorotUniform(seed=seed)

#     norm_fn = norm_fn.lower()
#     if norm_fn == 'batch':
#         norm_fn = BatchNormalization
#     elif norm_fn == 'layer':    
#         norm_fn = LayerNormalization
#     else:
#         raise ValueError("norm_fn must be 'batch' or 'layer'")
    
#     recurrent_layer_type = recurrent_layer_type.lower()
#     if recurrent_layer_type == 'lstm':
#         recurrent_layer_type = LSTM
#     elif recurrent_layer_type == 'gru':   
#         recurrent_layer_type = GRU
#     else:
#         raise ValueError("recurrent_layer_type must be 'lstm' or 'gru'")

#     inputs = Input(shape=input_shape)

#     shared_layers = []
    
#     for i in range(1, n_recurrent_layers + 1):
#         shared_layers.append(
#             recurrent_layer_type(
#                 n_recurrent_nodes, 
#                 return_sequences=(i < n_recurrent_layers), 
#                 kernel_regularizer=L1L2(l1,l2), 
#                 recurrent_dropout=rec_drop,
#                 kernel_initializer=initializer
#             )
#         )
#         if recurrent_norm:
#             shared_layers.append(norm_fn())

    
#     for i in range(1, n_shared_layers + 1):
#         shared_layers.append(
#             Dense(
#                 n_shared_nodes, 
#                 activation='relu', 
#                 kernel_regularizer=L1L2(l1,l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if shared_norm:
#             shared_layers.append(norm_fn())

#     # shared_layers.append(norm_fn())

#     shared_net = Sequential(shared_layers, name='shared')(inputs)

#     # Median head
#     median_head = Sequential(name='Q50')
#     for i in range(1, n_qtask_layers+1):
#         median_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             median_head.add(norm_fn())
    
#         if dropout > 0.0 and i < n_qtask_layers:
#             median_head.add(Dropout(dropout))

#     median_head.add(
#         Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2))
#     )

#     median_output = median_head(shared_net)

#     # Lower quantile heads
#     fifth_resid_head = Sequential(name='Q5_lower_raw')
#     for i in range(1, n_qtask_layers+1):
#         fifth_resid_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             fifth_resid_head.add(norm_fn())
        
#         if dropout > 0.0 and i < n_qtask_layers:
#             fifth_resid_head.add(Dropout(dropout))

#     fifth_resid_head.add(
#         Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
#     )
#     fifth_raw = fifth_resid_head(shared_net)
#     fifth_resid = Activation('softplus')(fifth_raw)

#     twentyfifth_resid_head = Sequential(name='Q25_lower_raw')
#     for i in range(1, n_qtask_layers+1):
#         twentyfifth_resid_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             twentyfifth_resid_head.add(norm_fn())
        
#         if dropout > 0.0 and i < n_qtask_layers:
#             twentyfifth_resid_head.add(Dropout(dropout))

#     twentyfifth_resid_head.add(
#         Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
#     )
#     twentyfifth_raw = twentyfifth_resid_head(shared_net)
#     twentyfifth_resid = Activation('softplus')(twentyfifth_raw)


#     # Upper quantile heads
#     ninetyfifth_resid_head = Sequential(name='Q95_upper_raw')
#     for i in range(1, n_qtask_layers+1):
#         ninetyfifth_resid_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             ninetyfifth_resid_head.add(norm_fn())

#         if dropout > 0.0 and i < n_qtask_layers:
#             ninetyfifth_resid_head.add(Dropout(dropout))

#     ninetyfifth_resid_head.add(
#         Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
#     )
#     ninetyfifth_raw = ninetyfifth_resid_head(shared_net)
#     ninetyfifth_resid = Activation('softplus')(ninetyfifth_raw)

#     # Upper quantile head
#     seventyfifth_resid_head = Sequential(name='Q75_upper_raw')
#     for i in range(1, n_qtask_layers+1):
#         seventyfifth_resid_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             seventyfifth_resid_head.add(norm_fn())
#         if dropout > 0.0 and i < n_qtask_layers:
#             seventyfifth_resid_head.add(Dropout(dropout))

#     seventyfifth_resid_head.add(
#         Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
#     )
#     seventyfifth_raw = seventyfifth_resid_head(shared_net)
#     seventyfifth_resid = Activation('softplus')(seventyfifth_raw)

#     # Combine outputs
#     Q50 = median_output
#     Q5 = Q50 - fifth_resid
#     Q25 = Q50 - twentyfifth_resid
#     Q75 = Q50 + seventyfifth_resid
#     Q95 = Q50 + ninetyfifth_resid

#     out_concat = Concatenate()([Q5, Q25, Q50, Q75, Q95])

#     model = Model(inputs=inputs, outputs=out_concat)

#     loss = make_total_tilted_loss(lower_quantiles + [0.5] + upper_quantiles, q_loss_weights=loss_weights)

#     model.compile(
#         loss=loss, 
#         optimizer=Adam(learning_rate=lr),
#     )

#     return model


# def build_dmq_v1(
#         input_shape: tuple, 
#         n_recurrent_layers: int=2, 
#         n_shared_layers: int=1, 
#         n_qtask_layers: int=2, 
#         n_recurrent_nodes: int=32,
#         n_shared_nodes: int=32,
#         n_task_nodes: int=32,
#         l1: float=0.0, 
#         l2: float=0.0, 
#         rec_drop: float=0.0,
#         dropout: float=0.0,
#         lr: float=0.001, 
#         norm_fn: str='layer', 
#         recurrent_layer_type: str='gru', 
#         lower_quantiles: List[float]=[0.05,0.25], 
#         upper_quantiles: List[float]=[0.75,0.95],
#         recurrent_norm: bool=False,
#         shared_norm: bool=False, 
#         task_specific_norm: bool=False, 
#         loss_weights: list[float]=[1.0]*5,
#         seed: int=1
#     ):

#     """
#     DMQv0 + quantile spacing + task layers for tails and median
#     """

#     initializer = GlorotUniform(seed=seed)

#     norm_fn = norm_fn.lower()
#     if norm_fn == 'batch':
#         norm_fn = BatchNormalization
#     elif norm_fn == 'layer':    
#         norm_fn = LayerNormalization
#     else:
#         raise ValueError("norm_fn must be 'batch' or 'layer'")
    
#     recurrent_layer_type = recurrent_layer_type.lower()
#     if recurrent_layer_type == 'lstm':
#         recurrent_layer_type = LSTM
#     elif recurrent_layer_type == 'gru':   
#         recurrent_layer_type = GRU
#     else:
#         raise ValueError("recurrent_layer_type must be 'lstm' or 'gru'")

#     inputs = Input(shape=input_shape)

#     shared_layers = []
    
#     for i in range(1, n_recurrent_layers + 1):
#         shared_layers.append(
#             recurrent_layer_type(
#                 n_recurrent_nodes, 
#                 return_sequences=(i < n_recurrent_layers), 
#                 kernel_regularizer=L1L2(l1,l2), 
#                 recurrent_dropout=rec_drop,
#                 kernel_initializer=initializer
#             )
#         )
#         if recurrent_norm:
#             shared_layers.append(norm_fn())

    
#     for i in range(1, n_shared_layers + 1):
#         shared_layers.append(
#             Dense(
#                 n_shared_nodes, 
#                 activation='relu', 
#                 kernel_regularizer=L1L2(l1,l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if shared_norm:
#             shared_layers.append(norm_fn())

#     # shared_layers.append(norm_fn())

#     shared_net = Sequential(shared_layers, name='shared')(inputs)

#     # Median head
#     median_head = Sequential(name='Q50')
#     for i in range(1, n_qtask_layers+1):
#         median_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             median_head.add(norm_fn())
#         if dropout > 0.0 and i < n_qtask_layers:
#             median_head.add(Dropout(dropout))

#     median_head.add(
#         Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2))
#     )

#     median_output = median_head(shared_net)

#     # Lower quantile head
#     lower_resid_head = Sequential(name='Q_lower_raw')
#     for i in range(1, n_qtask_layers+1):
#         lower_resid_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             lower_resid_head.add(norm_fn())
#         if dropout > 0.0 and i < n_qtask_layers:
#             lower_resid_head.add(Dropout(dropout))

#     lower_resid_head.add(
#         Dense(len(lower_quantiles), activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
#     )
#     lower_raw = lower_resid_head(shared_net)
#     lower_resid = Activation('softplus')(lower_raw)


#     # Upper quantile head
#     upper_resid_head = Sequential(name='Q_upper_raw')
#     for i in range(1, n_qtask_layers+1):
#         upper_resid_head.add(
#             Dense(
#                 n_task_nodes, 
#                 activation='relu',
#                 kernel_regularizer=L1L2(l1, l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if task_specific_norm and i < n_qtask_layers:
#             upper_resid_head.add(norm_fn())
#         if dropout > 0.0 and i < n_qtask_layers:    
#             upper_resid_head.add(Dropout(dropout))
        
#     upper_resid_head.add(
#         Dense(len(upper_quantiles), activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
#     )
#     upper_raw = upper_resid_head(shared_net)
#     upper_resid = Activation('softplus')(upper_raw)

#     # Combine outputs
#     Q50 = median_output
#     Q_lower = Q50 - lower_resid
#     Q_upper = Q50 + upper_resid

#     out_concat = Concatenate()([Q_lower, Q50, Q_upper])

#     model = Model(inputs=inputs, outputs=out_concat)

#     loss = make_total_tilted_loss(lower_quantiles + [0.5] + upper_quantiles, q_loss_weights=loss_weights)

#     model.compile(
#         loss=loss, 
#         optimizer=Adam(learning_rate=lr),
#     )

#     return model


# def build_dmq_v0(
#         input_shape: tuple, 
#         n_recurrent_layers: int=2, 
#         n_shared_layers: int=1, 
#         n_qtask_layers: int=2, 
#         n_recurrent_nodes: int=32,
#         n_shared_nodes: int=32,
#         n_task_nodes: int=32,
#         l1: float=0.0, 
#         l2: float=0.0, 
#         lr: float=0.001, 
#         rec_drop: float=0.0,
#         dropout: float=0.0,
#         norm_fn: str='layer', 
#         recurrent_layer_type: str='gru', 
#         quantiles: list[int]=[0.05,0.25,0.50,0.75,0.95], 
#         recurrent_norm: bool=False,
#         shared_norm: bool=False,
#         task_specific_norm: bool=False, 
#         loss_weights: list[float]=[1.0]*5,
#         seed: int=1
#     ):

#     initializer = GlorotUniform(seed=seed)

#     norm_fn = norm_fn.lower()
#     if norm_fn == 'batch':
#         norm_fn = BatchNormalization
#     elif norm_fn == 'layer':    
#         norm_fn = LayerNormalization
#     else:
#         raise ValueError("norm_fn must be 'batch' or 'layer'")
    
#     recurrent_layer_type = recurrent_layer_type.lower()
#     if recurrent_layer_type == 'lstm':
#         recurrent_layer_type = LSTM
#     elif recurrent_layer_type == 'gru':   
#         recurrent_layer_type = GRU
#     else:
#         raise ValueError("recurrent_layer_type must be 'lstm' or 'gru'")

#     inputs = Input(shape=input_shape)

#     shared_layers = []
    
#     for i in range(1, n_recurrent_layers + 1):
#         shared_layers.append(
#             recurrent_layer_type(
#                 n_recurrent_nodes, 
#                 return_sequences=(i < n_recurrent_layers), 
#                 kernel_regularizer=L1L2(l1,l2), 
#                 recurrent_dropout=rec_drop,
#                 kernel_initializer=initializer
#             )
#         )
#         if recurrent_norm:
#             shared_layers.append(norm_fn())

#     for i in range(1, n_shared_layers + 1):
#         shared_layers.append(
#             Dense(
#                 n_shared_nodes, 
#                 activation='relu', 
#                 kernel_regularizer=L1L2(l1,l2),
#                 kernel_initializer=initializer
#             )
#         )
#         if shared_norm:
#             shared_layers.append(norm_fn())

#     shared_net = Sequential(shared_layers, name='shared')(inputs)

#     outputs = []
#     for q in quantiles:
#         name = f"Q{q}"
        
#         qtask_layers = []
#         for i in range(1, n_qtask_layers+1):
#             qtask_layers.append(
#                 Dense(
#                     n_task_nodes, 
#                     activation='relu',
#                     kernel_regularizer=L1L2(l1, l2),
#                     kernel_initializer=initializer
#                 )
#             )
#             if task_specific_norm and i < n_qtask_layers:
#                 qtask_layers.append(norm_fn())
#             if dropout > 0.0 and i < n_qtask_layers:
#                 qtask_layers.append(Dropout(dropout))

#         # Append output node
#         qtask_layers.append(
#             Dense(1, activation='linear', kernel_regularizer=L1L2(l1, l2), kernel_initializer=initializer)
#         )
        
#         # Build output net
#         output_q = Sequential(qtask_layers, name=name)(shared_net)

#         outputs.append(output_q)

#     out_concat = Concatenate(name='out_layer')(outputs)

#     model = Model(inputs=inputs, outputs=out_concat)

#     loss = make_total_tilted_loss(quantiles, q_loss_weights=loss_weights)

#     model.compile(
#         loss=loss, 
#         optimizer=Adam(learning_rate=lr),
#     )

#     return model

