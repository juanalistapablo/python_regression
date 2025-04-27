import numpy as np
from Functions.compute_cost import compute_cost

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    theta_history = np.zeros((num_iters + 1, len(theta)))
    theta_history[0] = theta

    for i in range(num_iters):
        predictions = X @ theta
        erro = predictions - y
        gradient = (1 / m) * (X.T @ erro)
        theta = theta - alpha * gradient
        J_history[i] = compute_cost(X, y, theta)
        theta_history[i + 1] = theta
    return theta, J_history, theta_history
