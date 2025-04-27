import numpy as np

def warm_up_exercise1():
    """Retorna matriz identidade 5x5."""
    return np.eye(5)

def warm_up_exercise2(m=5):
    """Retorna um vetor coluna de 1s de shape (m, 1)."""
    return np.ones((m, 1))

def warm_up_exercise3(x):
    """Adiciona coluna de 1s ao vetor de entrada x."""
    m = x.shape[0]
    x = np.reshape(x, (m, 1))
    bias = np.ones((m, 1))
    return np.hstack((bias, x))

def warm_up_exercise4(X, theta):
    """Multiplicação matricial entre X e θ."""
    return X @ theta

def warm_up_exercise5(predictions, y):
    """Calcula vetor de erros quadráticos (squared errors)."""
    return (predictions - y) ** 2

def warm_up_exercise6(errors):
    """Calcula o custo médio (mean cost)."""
    return np.mean(errors) / 2

def warm_up_exercise7(X, y, theta):
    """Cálculo do custo médio para regressão linear."""
    predictions = warm_up_exercise4(X, theta)
    errors = warm_up_exercise5(predictions, y)
    return warm_up_exercise6(errors)
