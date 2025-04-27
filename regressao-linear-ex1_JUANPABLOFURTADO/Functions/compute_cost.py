import numpy as np

def compute_cost(X, y, theta):
    """Computa o custo para regressão linear."""
    m = len(y)
    h_o = X @ theta
    errors = h_o - y
    J_o = (1 / (2 * m)) * np.sum(errors ** 2)
    return J_o
