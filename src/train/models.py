import warnings
from typing import Union, List

import numpy as np
from joblib import Parallel, delayed
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from statsmodels.tools import add_constant
import keras

from src.train.slstm import sLSTMCell, LayerNormLSTMCell
from src.xlstm.blocks import sLSTMBlock
from src.train.losses import (
    compute_qpc, 
    quantile_loss, 
    make_tilted_loss, 
    make_total_tilted_loss
)

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
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(int(1), activation = 'linear', kernel_regularizer=keras.regularizers.L1L2(l1=l1,l2=l2)))
    opt = keras.optimizers.Adam(learning_rate=lr)
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
        norm_fn = keras.layers.BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = keras.layers.LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")

    model = keras.models.Sequential()
    for i in range(1,n_dense_layers+1):
        model.add(
            keras.layers.Dense(
                n_nodes, 
                'relu', 
                kernel_regularizer=keras.regularizers.L1L2(l1,l2)
            )
        )

        if i<n_dense_layers:
            model.add(norm_fn())

    model.add(
        keras.layers.Dense(
            int(1), 
            activation='linear', 
            kernel_regularizer=keras.regularizers.L1L2(l1,l2)
        )
    )
    
    opt = keras.optimizers.Adam(learning_rate=lr)
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
        recurrent_layer_type = keras.layers.LSTM
    elif recurrent_layer_type == 'gru':    
        recurrent_layer_type = keras.layers.GRU

    # Recurrent layers
    model = keras.models.Sequential()
    for i in range(1,n_recurrent_layers+1):
        model.add(
            recurrent_layer_type(
                n_nodes, 
                return_sequences=(i < n_recurrent_layers), kernel_regularizer=keras.regularizers.L1L2(l1,l2)
            )
        )

    # Dense layers
    for i in range(1,n_dense_layers+1):
        model.add(
            keras.layers.Dense(
                n_nodes, 
                'relu', 
                kernel_regularizer=keras.regularizers.L1L2(l1,l2)
            )
        )

    # Output layer    
    model.add(
        keras.layers.Dense(
            int(1), 
            activation='linear', 
            kernel_regularizer=keras.regularizers.L1L2(l1,l2)
        )
    )
    
    opt = keras.optimizers.Adam(learning_rate=lr)
    loss = make_tilted_loss(q)
    model.compile(loss = loss, optimizer = opt)
    return model

def _build_input_layer(
        input_shapes: list[tuple]
):
    inputs = []
    for input_shape in input_shapes:
        in_layer = keras.layers.Input(input_shape)
        inputs.append(in_layer)

    return inputs

def _get_recurrent_layer(
        size=32,
        type='lstm',
        num_heads=None,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        return_sequences=False,
        name='rnn_layer'
):
    """
    Returns a keras recurrent layer.

    Parameters:
    -----------
    size: int
        The number of hidden units in the layer
    type: str
        One of {'lstm', 'ln_lstm', 'gru', 'slstm', 'slstm_block'}
    
    
    """
    
    if type == 'lstm':
        return keras.layers.LSTM(
            units=size,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            return_sequences=return_sequences,
            name=name
        )
    
    if type == 'ln_lstm':
        return keras.layers.RNN(
            LayerNormLSTMCell(
                units=size,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
            ),
            return_sequences=return_sequences,
            name=name
        )
    
    elif type == 'gru':
        return keras.layers.GRU(
            units=size,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            return_sequences=return_sequences,
            name=name
        )

    elif type == 'slstm':
        return keras.layers.RNN(
            sLSTMCell(
                units=size,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
            ),
            return_sequences=return_sequences,
            name=name
        )
    
    elif type == 'slstm_block':
        if not num_heads:
            raise TypeError(
                'num_heads must be provided for layer type slstm_block.'
                )
        
        return sLSTMBlock(
            units=size,
            num_heads=num_heads,
            return_sequences=return_sequences,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            name=name
        )

def _build_rnn_layers(
        hidden_sizes,
        num_heads=None,
        layer_type='lstm',
        normalization_layer=None,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        return_sequences=True
):

    rnn_layers = []
    for i, size in enumerate(hidden_sizes):
        is_not_last = (i < len(hidden_sizes)-1)
        if not return_sequences:
            ret_seq = is_not_last
        else:
            ret_seq = return_sequences

        rnn_layers.append(
            _get_recurrent_layer(
                size=size,
                type=layer_type,
                num_heads=num_heads,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
                return_sequences=ret_seq,
                name=f'rnn_layer_{i+1}'
            )
        )

        if normalization_layer:
            rnn_layers.append(
                normalization_layer()
            )

    return rnn_layers

def _build_dense_layers(
        hidden_sizes,
        kernel_initializer='he_normal',
        activation='relu',
        normalization_layer=None,
        kernel_regularizer=None,
        normalize_last=True
):
    layers = []
    for i, size in enumerate(hidden_sizes):
        layers.append(
            keras.layers.Dense(
                units=size,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'dense_layer_{i+1}'
            )
        ) 

        if normalization_layer:

            if normalize_last or (i < len(hidden_sizes)-1):
                layers.append(
                    normalization_layer()
                )

        if activation:
            layers.append(
                keras.layers.Activation(activation)
            )
    
    return layers

def build_dmq(
        input_shapes,
        shared_recurrent_sizes = [32],
        shared_dense_sizes = [16],
        task_sizes = [8],
        l2=0.0,
        lr=3e-4,
        lower_quantiles = [0.05,0.25], 
        upper_quantiles = [0.75,0.95],
        recurrent_type='lstm',
        dense_activation='relu',
        dense_kernel_initializer='he_normal',
        bias_initializers = None,
        loss_weights = None
):
    
    inputs = _build_input_layer(input_shapes)

    quantiles = lower_quantiles + [0.5] + upper_quantiles

    if not loss_weights:
        loss_weights = [1/len(quantiles)] * len(quantiles)

    if not bias_initializers:
        bias_initializers = {
            q: 'zeros' for q in quantiles
        }

    recurrent_layers = _build_rnn_layers(
        hidden_sizes=shared_recurrent_sizes,
        num_heads=None,
        normalization_layer=keras.layers.LayerNormalization,
        layer_type=recurrent_type,
        kernel_regularizer=keras.regularizers.L2(l2),
        return_sequences=False
    )

    dense_layers = _build_dense_layers(
        hidden_sizes=shared_dense_sizes,
        activation=dense_activation,
        normalization_layer=keras.layers.LayerNormalization,
        kernel_initializer=dense_kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(l2),
        normalize_last=True
    )

    shared_layers = recurrent_layers + dense_layers

    shared_net = keras.models.Sequential(
        shared_layers, 
        name='shared_net'
    )(inputs)

    outputs = []
    for q in quantiles:
        qtask_layers = _build_dense_layers(
            hidden_sizes=task_sizes,
            activation=dense_activation,
            normalization_layer=keras.layers.LayerNormalization,
            kernel_initializer=dense_kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(l2),
            normalize_last=False
        )

        qtask_layers.append(
            keras.layers.Dense(
                1, 
                activation='linear', 
                kernel_regularizer=keras.regularizers.L2(l2), 
                bias_initializer=bias_initializers.get(q, 'zeros'),
                name=f'q{q}_task_out_layer'
            )
        )

        q_out = keras.models.Sequential(
            qtask_layers, 
            name=f'Q{int(q*100)}_head'
        )(shared_net)
        outputs.append(q_out)

    out_concat = keras.layers.Concatenate(name='out_layer')(outputs)

    model = keras.models.Model(inputs=inputs, outputs=out_concat)

    loss = make_total_tilted_loss(quantiles, q_loss_weights=loss_weights)

    model.compile(
        loss=loss, 
        optimizer=keras.optimizers.Adam(learning_rate=lr),
    )

    return model

def build_dmq_v1(
        input_shape: tuple, 
        n_recurrent_layers: int=2, 
        n_shared_layers: int=1, 
        n_qtask_layers: int=2,
        num_heads: int=4, 
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

    initializer = keras.initializers.GlorotUniform(seed=seed)
    lower_quantiles = sorted(lower_quantiles)
    upper_quantiles = sorted(upper_quantiles)
    quantiles = lower_quantiles + [0.5] + upper_quantiles
    if len(loss_weights) != len(quantiles):
        loss_weights = [1.0] * len(quantiles)

    norm_fn = norm_fn.lower()
    if norm_fn == 'batch':
        norm_fn = keras.layers.BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = keras.layers.LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")
    
    recurrent_layer_type = recurrent_layer_type.lower()
    if recurrent_layer_type == 'lstm':
        recurrent_layer = keras.layers.LSTM
    elif recurrent_layer_type == 'gru':   
        recurrent_layer = keras.layers.GRU
    elif recurrent_layer_type in ['slstm', 'slstm_block']:
        pass
    else:
        raise ValueError("recurrent_layer_type must be 'lstm', 'slstm', or 'gru'")

    inputs = keras.layers.Input(shape=input_shape)

    shared_layers = []
    
    for i in range(1, n_recurrent_layers + 1):
    
        if recurrent_layer_type == 'slstm':
            shared_layers.append(
                keras.layers.RNN(
                    sLSTMCell(
                        n_recurrent_nodes,
                        kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                        kernel_initializer=initializer
                    ), 
                    return_sequences=(i < n_recurrent_layers),
                    name=f'{recurrent_layer_type}_layer_{i}'
                )
            )
        elif recurrent_layer_type == 'slstm_block':
            shared_layers.append(
                sLSTMBlock(
                    n_recurrent_nodes,
                    num_heads=num_heads,
                    return_sequences=(i < n_recurrent_layers),
                    kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                    recurrent_regularizer=keras.regularizers.L1L2(l1,l2),
                    name=f'{recurrent_layer_type}_layer_{i}'
                )
            )
        else:
            shared_layers.append(
                recurrent_layer(
                    n_recurrent_nodes, 
                    return_sequences=(i < n_recurrent_layers), 
                    kernel_regularizer=keras.regularizers.L1L2(l1,l2), 
                    recurrent_regularizer=keras.regularizers.L1L2(l1,l2),
                    recurrent_dropout=rec_drop,
                    kernel_initializer=initializer,
                    name=f'{recurrent_layer_type}_layer_{i}'
                )
            )
        
        if recurrent_norm:
            shared_layers.append(norm_fn())

    if not recurrent_layer_type == 'slstm_block':
        for i in range(1, n_shared_layers + 1):
            shared_layers.append(
                keras.layers.Dense(
                    n_shared_nodes, 
                    activation='relu', 
                    kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                    kernel_initializer=initializer,
                    name=f'shared_dense_layer_{i}'
                )
            )
            if shared_norm:
                shared_layers.append(norm_fn())

    shared_net = keras.models.Sequential(shared_layers, name='shared_layers')(inputs)

    # Median head
    median_head = keras.models.Sequential(name='Q50')
    for i in range(1, n_qtask_layers+1):
        median_head.add(
            keras.layers.Dense(
                n_task_nodes, 
                activation='relu',
                kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                kernel_initializer=initializer,
                name=f'q0.50_dense_layer_{i}'
            )
        )
        if task_norm and i < n_qtask_layers:
            median_head.add(norm_fn())
    
        if dropout > 0.0 and i < n_qtask_layers:
            median_head.add(keras.layers.Dropout(dropout))

    median_head.add(
        keras.layers.Dense(
            1, 
            activation='linear', 
            kernel_regularizer=keras.regularizers.L1L2(l1,l2),
            name=f'q0.50_output'
            )
    )

    median_output = median_head(shared_net)

    # Lower quantile heads (monotonic chain)
    lower_outputs = []
    prev = median_output
    for q in sorted(lower_quantiles, reverse=True):
        qtask_layers = []
        for i in range(1, n_qtask_layers+1):
            qtask_layers.append(
                keras.layers.Dense(
                    n_task_nodes,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                    kernel_initializer=initializer,
                    name=f'q{q}_dense_layer_{i}'
                )
            )
            if task_norm and i < n_qtask_layers:
                qtask_layers.append(norm_fn())
            if dropout > 0.0 and i < n_qtask_layers:
                qtask_layers.append(keras.layers.Dropout(dropout))

        qtask_layers.append(
            keras.layers.Dense(1, activation='linear', kernel_regularizer=keras.regularizers.L1L2(l1,l2), kernel_initializer=initializer)
        )
        q_task_resid = keras.models.Sequential(qtask_layers, name=f'Q{int(q*100)}_lower_raw')(shared_net)
        q_out = keras.layers.Subtract(name=f'Q{int(q*100)}_from_prev')([prev, keras.layers.Activation('softplus')(q_task_resid)])
        lower_outputs.append(q_out)
        prev = q_out

    # Upper quantile heads (monotonic chain)
    upper_outputs = []
    prev = median_output
    for q in upper_quantiles:
        qtask_layers = []
        for i in range(1, n_qtask_layers+1):
            qtask_layers.append(
                keras.layers.Dense(
                    n_task_nodes,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                    kernel_initializer=initializer,
                    name=f'q{q}_dense_layer_{i}'
                )
            )
            if task_norm and i < n_qtask_layers:
                qtask_layers.append(norm_fn())
            if dropout > 0.0 and i < n_qtask_layers:
                qtask_layers.append(keras.layers.Dropout(dropout))

        qtask_layers.append(
            keras.layers.Dense(1, activation='linear', kernel_regularizer=keras.regularizers.L1L2(l1,l2), kernel_initializer=initializer)
        )
        q_task_resid = keras.models.Sequential(qtask_layers, name=f'Q{int(q*100)}_upper_raw')(shared_net)
        q_out = keras.layers.Add(name=f'Q{int(q*100)}_from_prev')([prev, keras.layers.Activation('softplus')(q_task_resid)])
        upper_outputs.append(q_out)
        prev = q_out

    outputs = list(reversed(lower_outputs)) + [median_output] + upper_outputs
    out_concat = keras.layers.Concatenate(name='out_layer')(outputs)

    model = keras.model.keras.models.Model(inputs=inputs, outputs=out_concat)

    loss = make_total_tilted_loss(quantiles, q_loss_weights=loss_weights)

    model.compile(
        loss=loss, 
        optimizer=keras.optimizers.Adam(learning_rate=lr),
    )

    return model

def build_dmq_v0(
        input_shape, 
        n_recurrent_layers=1, 
        num_heads=1,
        n_shared_layers=1, 
        n_qtask_layers=1, 
        n_recurrent_nodes=32,
        n_shared_nodes=32,
        n_task_nodes=32,
        l1=0.0, 
        l2=0.0, 
        lr=0.001, 
        rec_drop=0.0,
        dropout=0.0,
        norm_fn='layer', 
        recurrent_layer_type='gru', 
        lower_quantiles=[0.05, 0.25],
        upper_quantiles=[0.75, 0.95], 
        recurrent_norm=False,
        shared_norm=False,
        task_norm=False, 
        loss_weights=[1/5]*5,
        bias_initializer='zeros'
    ):

    """
    Base DMQ model
    """

    initializer = 'glorot_uniform'
    
    lower_quantiles = sorted(lower_quantiles)
    upper_quantiles = sorted(upper_quantiles)
    quantiles = lower_quantiles + [0.5] + upper_quantiles
    if len(loss_weights) != len(quantiles):
        loss_weights = [1.0] * len(quantiles)

    norm_fn = norm_fn.lower()
    if norm_fn == 'batch':
        norm_fn = keras.layers.BatchNormalization
    elif norm_fn == 'layer':    
        norm_fn = keras.layers.LayerNormalization
    else:
        raise ValueError("norm_fn must be 'batch' or 'layer'")
    
    recurrent_layer_type = recurrent_layer_type.lower()
    if recurrent_layer_type == 'lstm':
        recurrent_layer = keras.layers.LSTM
    elif recurrent_layer_type == 'gru':   
        recurrent_layer = keras.layers.GRU
    elif recurrent_layer_type in ['slstm', 'slstm_block']:
        pass
    else:
        raise ValueError("recurrent_layer_type must be 'lstm', 'slstm', or 'gru'")

    inputs = keras.layers.Input(shape=input_shape)

    shared_layers = []
    
    for i in range(1, n_recurrent_layers + 1):
    
        if recurrent_layer_type == 'slstm':
            shared_layers.append(
                keras.layers.RNN(
                    sLSTMCell(
                        n_recurrent_nodes,
                        kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                        kernel_initializer=initializer
                    ), 
                    return_sequences=(i < n_recurrent_layers),
                    name=f'{recurrent_layer_type}_layer_{i}'
                )
            )
        elif recurrent_layer_type == 'slstm_block':
            shared_layers.append(
                sLSTMBlock(
                    n_recurrent_nodes,
                    num_heads=num_heads,
                    return_sequences=(i < n_recurrent_layers),
                    kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                    recurrent_regularizer=keras.regularizers.L1L2(l1,l2),
                    name=f'{recurrent_layer_type}_layer_{i}'
                )
            )
        else:
            shared_layers.append(
                recurrent_layer(
                    n_recurrent_nodes, 
                    return_sequences=(i < n_recurrent_layers), 
                    kernel_regularizer=keras.regularizers.L1L2(l1,l2), 
                    recurrent_regularizer=keras.regularizers.L1L2(l1,l2),
                    recurrent_dropout=rec_drop,
                    kernel_initializer=initializer,
                    name=f'{recurrent_layer_type}_layer_{i}'
                )
            )
        
        if recurrent_norm:
            shared_layers.append(norm_fn())

    if not recurrent_layer_type == 'slstm_block':
        for i in range(1, n_shared_layers + 1):
            shared_layers.append(
                keras.layers.Dense(
                    n_shared_nodes, 
                    activation='relu', 
                    kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                    kernel_initializer=initializer,
                    name=f'shared_dense_layer_{i}'
                )
            )
            if shared_norm:
                shared_layers.append(norm_fn())

    shared_net = keras.models.Sequential(shared_layers, name='shared_layers')(inputs)

    outputs = []
    for q in quantiles:
        qtask_layers = []
        for i in range(1, n_qtask_layers + 1):
            qtask_layers.append(
                keras.layers.Dense(
                    n_task_nodes,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                    kernel_initializer=initializer,
                    name=f'q{q}_task_dense_layer_{i}'
                )
            )
            if task_norm and i < n_qtask_layers:
                qtask_layers.append(norm_fn())
            if dropout > 0.0 and i < n_qtask_layers:
                qtask_layers.append(keras.layers.Dropout(dropout))

        qtask_layers.append(
            keras.layers.Dense(
                1, 
                activation='linear', 
                kernel_regularizer=keras.regularizers.L1L2(l1,l2), 
                kernel_initializer=initializer,
                bias_initializer=bias_initializer,
                name=f'q{q}_task_out_layer_{i}'
            )
        )

        q_out = keras.models.Sequential(qtask_layers, name=f'Q{int(q*100)}_head')(shared_net)
        outputs.append(q_out)

    out_concat = keras.layers.Concatenate(name='out_layer')(outputs)

    model = keras.models.Model(inputs=inputs, outputs=out_concat)

    loss = make_total_tilted_loss(quantiles, q_loss_weights=loss_weights)

    model.compile(
        loss=loss, 
        optimizer=keras.optimizers.Adam(learning_rate=lr),
    )

    return model