import numpy as np
from numba import njit

@njit(cache=True)
def ols_residuals(y: np.ndarray, x: np.ndarray):
    """
    Compute OLS slope & intercept (beta1, beta0) via closed‐form,
    then return residual array.
    """
    n = y.shape[0]
    # means
    mean_y = 0.0
    mean_x = 0.0
    for i in range(n):
        mean_y += y[i]
        mean_x += x[i]
    mean_y /= n
    mean_x /= n

    # covariance & variance
    num = 0.0
    den = 0.0
    for i in range(n):
        dx = x[i] - mean_x
        num += dx * (y[i] - mean_y)
        den += dx * dx

    # parameters
    beta1 = num / den
    beta0 = mean_y - beta1 * mean_x

    # residuals
    resid = np.empty(n, dtype=np.float64)
    for i in range(n):
        resid[i] = y[i] - (beta0 + beta1 * x[i])

    return beta0, beta1, resid

@njit(cache=True)
def corr_coef(y: np.ndarray, x: np.ndarray):
    """
    Simple Pearson correlation.
    """
    n = y.shape[0]
    mean_y = 0.0
    mean_x = 0.0
    for i in range(n):
        mean_y += y[i]
        mean_x += x[i]
    mean_y /= n
    mean_x /= n

    num = 0.0
    var_y = 0.0
    var_x = 0.0
    for i in range(n):
        dy = y[i] - mean_y
        dx = x[i] - mean_x
        num  += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    return num / np.sqrt(var_x * var_y)