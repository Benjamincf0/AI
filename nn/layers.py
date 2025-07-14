import numpy as np

class Layer():
    def __init__(self, units:int):
        self.units = units

    def __call__(self, X:np.ndarray) -> np.ndarray:
        pass

    def init_weights(self):
        pass

class Dense(Layer):
    def __init__(self, units:int, activation:function):
        super.__init__(units)
        self.activation = activation
        self.W:np.ndarray = None
        self.b = np.zeros((units))

    def init_weights(self, prev_units:int):
        self.W = np.random.rand(prev_units, self.units) - 0.5

    def __call__(self, A_in:np.ndarray) -> np.ndarray:
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
        A = self.activation(Z)
        return A
    
    def compute_gradient(self, A_actual, A_pred):
        """_summary_

        Args:
            y_actual (m, self.units): _description_
            y_pred (m, self.units): _description_
        """

        m = A_actual.shape[0]
        dj_dw = np.zeros_like(self.weights)
        dj_db = 0

        dj_dw = A_actual*(self.predict(A_actual) - A_pred)/m
        dj_db = (self.predict(A_actual) - A_pred)/m







class InLayer(Layer):
    def __call__(self, X:np.ndarray):
        return X