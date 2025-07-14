import numpy as np
# import pandas as pd

def sigmoid(z:np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-z))

def compute_gradient(X:np.ndarray, w:np.ndarray, b:np.ndarray, y:np.ndarray) -> tuple[np.ndarray]:
    """_summary_

    Args:
        X (np.ndarray): (m, n)
        w (np.ndarray): (n,)
        b (np.ndarray): ()
        y (np.ndarray): ()
    """
    m = X.shape[0]
    dj_dw = np.empty_like(w)
    dj_db = np.empty_like(b)

    for i in range(m):
        dj_dw += X[i]*(np.dot(X[i], w) + b)/(2*m)
        dj_db += (np.dot(X[i], w) + b)/(2*m)
    return dj_dw, dj_db

