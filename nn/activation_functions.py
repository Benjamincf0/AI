import numpy as np

def sigmoid(z:np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-z))

def no_activation(z:np.ndarray) -> np.ndarray:
    return z