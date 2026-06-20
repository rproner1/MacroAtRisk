import numpy as np
import warnings
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from statsmodels.tools import add_constant
import keras

def compute_quantile_subgradient(u: np.ndarray, q: float) -> float:
    """Check function for quantile regression."""
    return (q - (u < 0).astype(float))

def quantile_loss(u: np.ndarray, q: float) -> float:
    """Quantile loss function."""
    return u * compute_quantile_subgradient(u, q)

def compute_qpc(y: np.ndarray, X_s: np.ndarray , X_j: np.ndarray, q: float) -> float:

    '''Computes quantile partial correlation between X_s and X_j'''

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=IterationLimitWarning)
        # add intercept if missing
        X_s_const = add_constant(X_s, has_constant='skip')
        reg_s = QuantReg(y, X_s_const).fit(q=q)
        resid_s = y - reg_s.predict(X_s_const)

        X_j_const = add_constant(X_j, has_constant='skip')
        reg_j = QuantReg(y, X_j_const).fit(q=q)
        resid_j = y - reg_j.predict(X_j_const)

    var_j = np.var(resid_j)

    qpc = np.mean(compute_quantile_subgradient(resid_s, q) * resid_j )/ np.sqrt(q*(1-q)*var_j)

    return qpc

def tilted_loss(
        y_true,
        y_pred, 
        q
):
    """
    Computes tilted loss for quantile regression.
    """
    e = y_true - y_pred
    return keras.ops.mean(keras.ops.maximum(q * e, (q - 1.0) * e), axis=-1)

@keras.saving.register_keras_serializable()
def make_tilted_loss(q: float):

    def loss(y_true, y_pred):
        e = y_true - y_pred
        return keras.ops.mean(keras.ops.maximum(q * e, (q - 1.0) * e))
    
    loss.__name__ = f"tilted_loss_{int(q*100)}"

    return loss

@keras.saving.register_keras_serializable()
def make_total_tilted_loss(
    quantiles: list[float], 
    q_loss_weights: list[float]|None = None
):
    """
    Returns a loss function that computes the mean of tilted losses for the given quantiles.
    """

    if q_loss_weights is None:
        q_loss_weights = [1.0] * len(quantiles)

    loss_fns = [make_tilted_loss(q) for q in quantiles]
    
    def total_tilted_loss(y_true, y_pred):
        # y_pred shape: (batch, len(quantiles))
        losses = []
        # Compute loss on each quantile
        for i, lf in enumerate(loss_fns):
            losses.append(q_loss_weights[i] * lf(y_true, y_pred[:, i:i+1]))

        return keras.ops.mean(losses) 

    total_tilted_loss.__name__ = "total_tilted_loss_" + "_".join(str(int(q*100)) for q in quantiles)

    return total_tilted_loss


