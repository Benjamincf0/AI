import numpy as np

class cost_function():
    @staticmethod
    def apply(self, Y_pred:np.ndarray, Y_train:np.ndarray):
        """matrices

        Args:
            Y (m, layer[-1].units): _description_
            Y_train (m, layer[-1].units): _description_
        """
        
    @staticmethod
    def loss(self, Y_pred:np.ndarray, Y_train:np.ndarray):
        """matrices

        Args:
            Y (m, layer[-1].units): _description_
            Y_train (m, layer[-1].units): _description_
        """
        pass

    @staticmethod
    def prime(self, Y_pred:np.ndarray, Y_train:np.ndarray):
        pass

class MSE(cost_function):
    @staticmethod
    def apply(self, Y_pred:np.ndarray, Y_train:np.ndarray):
        return np.sum((Y_pred - Y_train)**2, axis=0)/(2*Y_pred.shape[0])

    @staticmethod
    def prime(self, Y_pred:np.ndarray, Y_train:np.ndarray):
        return Y_pred - Y_train