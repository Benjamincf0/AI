import numpy as np

class cost_function():
    @staticmethod
    def apply(Y_pred:np.ndarray, Y_train:np.ndarray):
        """matrices

        Args:
            Y (m, layer[-1].units): _description_
            Y_train (m, layer[-1].units): _description_
        """
        
    @staticmethod
    def loss(Y_pred:np.ndarray, Y_train:np.ndarray):
        """matrices

        Args:
            Y (m, layer[-1].units): _description_
            Y_train (m, layer[-1].units): _description_
        """
        pass

    @staticmethod
    def prime(Y_pred:np.ndarray, Y_train:np.ndarray):
        pass

class MSE(cost_function):
    @staticmethod
    def apply(Y_pred:np.ndarray, Y_train:np.ndarray):
        return float(np.sum((Y_pred - Y_train)**2, axis=0)/(2*Y_pred.shape[0]))

    @staticmethod
    def prime(Y_pred:np.ndarray, Y_train:np.ndarray):
        return (Y_pred - Y_train)/(Y_pred.shape[0])
    
class CrossEntropyLoss(cost_function):
    @staticmethod
    def apply(Y_pred:np.ndarray, Y_train:np.ndarray):
        """Cross Entropy Loss for softmax outputs
        J = -(1/m) * Σ y_train * log(y_pred)
        
        Args:
            Y_pred (np.ndarray): Softmax outputs, shape (m, units)
            Y_train (np.ndarray): One-hot encoded labels, shape (m, units)
        """
        m = Y_pred.shape[0]
        epsilon = 1e-15  # prevent log(0)
        return -np.sum(Y_train * np.log(Y_pred + epsilon)) / m

    @staticmethod
    def prime(Y_pred:np.ndarray, Y_train:np.ndarray):
        """For softmax outputs, derivative simplifies to (y_pred - y_train)
        
        This simple form is due to the combination of softmax + cross entropy
        """
        return Y_pred - Y_train