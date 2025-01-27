from tinygrad.tensor import Tensor as T


def mse(y_true, y_pred):
    return T.mean((y_true - y_pred) ** 2)
