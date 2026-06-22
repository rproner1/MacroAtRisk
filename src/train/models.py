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

def _build_recurrent_layers(
        n_layers: int = 1,
        n_units: int = 32,
        l1: float = 0.0,
        l2: float = 0.0,
        rec_drop: float = 0.0,
        initializer: str = 'glorot_uniform'
):
    layers = []
    for i in range(1, n_layers+1):
        layers.append(
            keras.layers.LSTM(
                n_units, 
                return_sequences=(i < n_layers), 
                kernel_regularizer=keras.regularizers.L1L2(l1,l2), 
                recurrent_dropout=rec_drop,
                kernel_initializer=initializer,
                name=f'recurrent_layer_{i}'
            )
        )

        layers.append(
            keras.layers.LayerNormalization()
        )

    return layers

def _build_dense_layers(
        n_layers: int = 1,
        n_units: int = 32,
        l1: float = 0.0,
        l2: float = 0.0,
        initializer: str = 'glorot_uniform',
        name_prefix: str | None = None
):
    layers = []
    for i in range(1, n_layers+1):
        layers.append(
            keras.layers.Dense(
                units=n_units,
                kernel_initializer=initializer,
                kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                name=f'{name_prefix}_dense_layer_{i}'
            )
        ) 

        layers.append(
            keras.layers.LayerNormalization()
        )
        
    return layers

def _build_task_heads(
        inputs,
        n_layers: int,
        n_units: int,
        quantiles: list[float],
        initializer: str = 'he_normal',
        l1: float = 0.0,
        l2: float = 0.0,
        bias_initializers: dict[keras.initializers.Constant] | None = None
):
    
    if bias_initializers is None:
        bias_initializers = {q: 'zeros' for q in quantiles}
    
    outputs = []
    for q in quantiles:
        qtask_layers = []
        for i in range(1, n_layers + 1):
            qtask_layers.append(
                keras.layers.Dense(
                    n_units,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                    kernel_initializer=initializer,
                    name=f'q{q}_task_dense_layer_{i}'
                )
            )
            if i < n_layers:
                qtask_layers.append(
                    keras.layers.LayerNormalization()
                )

        qtask_layers.append(
            keras.layers.Dense(
                1, 
                activation='linear', 
                kernel_regularizer=keras.regularizers.L1L2(l1,l2), 
                kernel_initializer='glorot_uniform',
                bias_initializer=bias_initializers[q],
                name=f'q{q}_task_out_layer_{i}'
            )
        )

        q_out = keras.models.Sequential(qtask_layers, name=f'Q{int(q*100)}_head')(inputs)
        outputs.append(q_out)
    
    return outputs

def _build_spaced_task_heads(
        inputs,
        n_layers: int,
        n_units: int,
        quantiles: list[float],
        initializer: str = 'he_normal',
        l1: float = 0.0,
        l2: float = 0.0,
        bias_initializers: dict[keras.initializers.Constant] | None = None
):
    quantiles = sorted(quantiles)
    lower_quantiles = [q for q in quantiles if q < .5]
    upper_quantiles = [q for q in quantiles if q > .5]
    
    if bias_initializers is None:
        bias_initializers = {
            q: 'zeros' for q in quantiles
        }

    # Median head
    median_head = keras.models.Sequential(name='Q50')
    for i in range(1, n_layers+1):
        median_head.add(
            keras.layers.Dense(
                n_units, 
                activation='relu',
                kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                kernel_initializer=initializer,
                name=f'Q50_dense_layer_{i}'
            )
        )
        if i < n_layers:
            median_head.add(keras.layers.LayerNormalization())
    
    median_head.add(
        keras.layers.Dense(
            1, 
            activation='linear', 
            kernel_regularizer=keras.regularizers.L1L2(l1,l2),
            bias_initializer=bias_initializers[.5],
            name=f'Q50_output'
            )
    )

    median_output = median_head(inputs)

    # Lower quantile heads (monotonic chain)
    lower_outputs = []
    prev = median_output
    for q in sorted(lower_quantiles, reverse=True):
        qtask_layers = []
        for i in range(1, n_layers+1):
            qtask_layers.append(
                keras.layers.Dense(
                    n_units,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                    kernel_initializer=initializer,
                    name=f'Q{int(q*100)}_dense_layer_{i}'
                )
            )
            if i < n_layers:
                qtask_layers.append(keras.layers.LayerNormalization())

        qtask_layers.append(
            keras.layers.Dense(
                1, 
                activation='linear', 
                kernel_regularizer=keras.regularizers.L1L2(l1,l2), 
                kernel_initializer=initializer,
                bias_initializer=bias_initializers[q]
            )
        )
        q_task_resid = keras.models.Sequential(
            qtask_layers, 
            name=f'Q{int(q*100)}_lower_raw'
        )(inputs)

        q_out = keras.layers.Subtract(
            name=f'Q{int(q*100)}_from_prev'
        )([prev, keras.layers.Activation('softplus')(q_task_resid)])

        lower_outputs.append(q_out)
        prev = q_out

    # Upper quantile heads (monotonic chain)
    upper_outputs = []
    prev = median_output
    for q in upper_quantiles:
        qtask_layers = []
        for i in range(1, n_layers+1):
            qtask_layers.append(
                keras.layers.Dense(
                    n_units,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.L1L2(l1,l2),
                    kernel_initializer=initializer,
                    name=f'Q{int(q*100)}_dense_layer_{i}'
                )
            )
            if i < n_layers:
                qtask_layers.append(keras.layers.LayerNormalization())

        qtask_layers.append(
            keras.layers.Dense(
                1, 
                activation='linear',
                kernel_regularizer=keras.regularizers.L1L2(l1,l2), 
                kernel_initializer=initializer,
                bias_initializer=bias_initializers[q]
            )
        )
        q_task_resid = keras.models.Sequential(
            qtask_layers, 
            name=f'Q{int(q*100)}_upper_raw'
        )(inputs)

        q_out = keras.layers.Add(
            name=f'Q{int(q*100)}_from_prev'
        )([prev, keras.layers.Activation('softplus')(q_task_resid)])

        upper_outputs.append(q_out)

        prev = q_out

    outputs = list(reversed(lower_outputs)) + [median_output] + upper_outputs

    return outputs

def build_dmq(
        input_shapes, 
        n_recurrent_layers: int = 1, 
        n_shared_layers: int = 1, 
        n_qtask_layers: int = 1, 
        n_recurrent_nodes: int = 32,
        n_shared_nodes: int = 32,
        n_task_nodes: int = 32,
        initializer: str = 'glorot_uniform',
        l1: float = 0.0, 
        l2: float = 0.0, 
        lr: float = 0.001, 
        rec_drop: float = 0.0,
        quantiles: list[float] | None = None, 
        loss_weights: list[float] | None = None,
        bias_initializers: dict[str|keras.Initializer] | None = None,
        space_quantiles: bool = False
):
    
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    
    inputs = _build_input_layer(input_shapes)

    if loss_weights is None:
        loss_weights = [1.0]*len(quantiles)

    if bias_initializers is None:
        bias_initializers = {
            q: 'zeros' for q in quantiles
        }

    if len(inputs) > 1:
        rnn_out = []
        for i, x in enumerate(inputs):
            x_rnn_layers = _build_recurrent_layers(
                n_layers=n_recurrent_layers,
                n_units=n_recurrent_nodes,
                l1=l1,
                l2=l2,
                rec_drop=rec_drop,
                initializer=initializer
            )
            x_rnn = keras.models.Sequential(
                x_rnn_layers,
                name=f'input_{i}_rnn'
            )(x)
            rnn_out.append(x_rnn)
        
        rnn_out = keras.layers.Concatenate(name='input_wise_rnn')(rnn_out)

    
    else:
        rnn_layers = _build_recurrent_layers(
            n_layers=n_recurrent_layers,
            n_units=n_recurrent_nodes,
            l1=l1,
            l2=l2,
            rec_drop=rec_drop,
            initializer=initializer
        )
        rnn_out = keras.models.Sequential(
            rnn_layers,
            name='shared_rnn'
        )(inputs[0])
    

    dense_layers = _build_dense_layers(
        n_layers=n_shared_layers,
        n_units=n_shared_nodes,
        l1=l1,
        l2=l2,
        initializer=initializer,
        name_prefix='shared'
    )

    dense_net = keras.models.Sequential(
        dense_layers, 
        name='shared_dense'
    )(rnn_out)

    if space_quantiles:
        outputs = _build_spaced_task_heads(
            dense_net,
            n_layers=n_qtask_layers,
            n_units=n_task_nodes,
            quantiles=quantiles,
            initializer=initializer,
            l1=l1,
            l2=l2,
            bias_initializers=bias_initializers
        )
    else:
        outputs = _build_task_heads(
            dense_net,
            n_layers=n_qtask_layers,
            n_units=n_task_nodes,
            quantiles=quantiles,
            initializer=initializer,
            l1=l1,
            l2=l2,
            bias_initializers=bias_initializers
        )

    outputs = keras.layers.Concatenate()(outputs)

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    loss = make_total_tilted_loss(quantiles, q_loss_weights=loss_weights)

    model.compile(
        loss=loss, 
        optimizer=keras.optimizers.Adam(
            learning_rate=lr
        ),
    )

    return model

def build_dmq_v0(
        input_shapes, 
        n_recurrent_layers: int = 1, 
        n_shared_layers: int = 1, 
        n_qtask_layers: int = 1, 
        n_recurrent_nodes: int = 32,
        n_shared_nodes: int = 32,
        n_task_nodes: int = 32,
        initializer: str = 'glorot_uniform',
        l1: float = 0.0, 
        l2: float = 0.0, 
        lr: float = 0.001, 
        rec_drop: float = 0.0,
        lower_quantiles: list[float] | None = None, 
        upper_quantiles: list[float] | None = None, 
        loss_weights: list[float] | None = None,
        bias_initializers: dict[str|keras.initializers.Constant] | None = None,
        space_quantiles: bool = False
    ):

    """
    Base DMQ model
    """

    if lower_quantiles is None:
        lower_quantiles = [.05, .25]
    if upper_quantiles is None:
        upper_quantiles = [.75, .95]

    lower_quantiles = sorted(lower_quantiles)
    upper_quantiles = sorted(upper_quantiles)
    quantiles = lower_quantiles + [0.5] + upper_quantiles
    
    # Set defaults
    if loss_weights is None:
        loss_weights = [1.0]*len(quantiles)

    if bias_initializers is None:
        bias_initializers = {
            q: 'zeros' for q in quantiles
        }

    inputs = _build_input_layer(input_shapes)

    if isinstance(inputs, list):
        x_nets = []
        for x in inputs: 
            x_layers = _build_recurrent_layers(
                n_layers=n_recurrent_layers,
                n_units=n_recurrent_nodes,
                l1=l1,
                l2=l2,
                rec_drop=rec_drop,
                initializer=initializer
            )
            x_net = keras.models.Sequential(x_layers)(x)
            x_nets.append(x_net)

        recurrent_net = keras.layers.Concatenate()(x_nets)
    
    else:
        recurrent_layers = _build_recurrent_layers(
            n_layers=n_recurrent_layers,
            n_units=n_recurrent_nodes,
            l1=l1,
            l2=l2,
            rec_drop=rec_drop,
            initializer=initializer
        )
        recurrent_net = keras.models.Sequential(recurrent_layers)(inputs)

    shared_layers = []
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
        
        shared_layers.append(
            keras.layers.LayerNormalization()
        )

    shared_net = keras.models.Sequential(
        shared_layers, 
        name='shared_layers'
    )(recurrent_net)

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
            if i < n_qtask_layers:
                qtask_layers.append(
                    keras.layers.LayerNormalization()
                )

        qtask_layers.append(
            keras.layers.Dense(
                1, 
                activation='linear', 
                kernel_regularizer=keras.regularizers.L1L2(l1,l2), 
                kernel_initializer='glorot_uniform',
                bias_initializer=bias_initializers[q],
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
