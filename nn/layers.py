import numpy as np
from nn.activation_functions import activation_f, linear

class Layer():
    def __init__(self, units:int):
        self.units = units

    def __call__(self, X:np.ndarray) -> np.ndarray:
        return None

    def init_weights(self, prev_units:int):
        return None

    def gradient_and_update(self, dJ_dA:np.ndarray, m:int, alpha:float):
        return dJ_dA

class Dense(Layer):
    def __init__(self, units:int, activation:activation_f=linear):
        super().__init__(units)
        self.activation = activation
        self.W:np.ndarray = None
        self.b = np.zeros((units))

        self.dJ_dw = None
        self.dJ_db = None

    def init_weights(self, prev_units:int):
        self.W = np.random.rand(prev_units, self.units) - 0.5

    def __call__(self, A_in:np.ndarray, save:bool=False) -> np.ndarray:
        """Forward

        A_in = [[1, 2, 3],  # a_0
                [4, 5, 6],] # a_1 or a_m

        Args:
            A_in shape=(m, W.shape[0]): Matrix
            self.W shape=(A_in.shape[1], Z.shape[1]): Matrix

        Returns:
            A shape=(m, W.shape[1])
        """
        Z = np.matmul(A_in, self.W) + self.b
        A = self.activation.apply(Z)

        if save:
            self.Z = Z
            self.A_prev = A_in
            self.A = A

        return A
    
    def gradient_and_update(self, dJ_dA:np.ndarray, m:int, alpha:float):
        """_summary_

        Args:
            y_actual (m, self.units): _description_
            y_pred (m, self.units): _description_
        """

        dA_dZ = self.activation.prime(Z = self.Z, A = self.A)
        dJ_dZ = np.multiply(dA_dZ, dJ_dA) # m times

        dJ_db = dJ_dZ
        dJ_db = np.sum(dJ_db, axis=0)/m

        dJ_dW = np.matmul(self.A_prev.T, dJ_dZ)/m
        self.update_params(dJ_dW, dJ_db, alpha)

        dJ_dA_prev = np.matmul(dJ_dZ, self.W.T)

        return dJ_dA_prev

    def update_params(self, dJ_dW:np.ndarray, dJ_db:np.ndarray, alpha:float):
        self.W -= alpha*dJ_dW
        self.b -= alpha*dJ_db

class InLayer(Layer):
    def __call__(self, X:np.ndarray, save:bool):
        return X