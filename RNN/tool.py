import numpy as np
def mse_loss(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def d_mse_loss(y, y_pred):
    return 2 * (y_pred - y) / y.size

def mae_loss(y, y_pred):
    return np.mean(np.abs(y - y_pred))

def d_mae_loss(y, y_pred):
    return np.where(y_pred > y, 1, -1) / y.size


def d_cross_entropy_loss(y, y_pred):
    """Derivative of Cross-Entropy loss."""
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -y / y_pred  # Fixed incorrect syntax
