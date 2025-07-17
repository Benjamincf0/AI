import numpy as np

class activation_f():
    @staticmethod
    def apply(Z:np.ndarray) -> np.ndarray:
        """calls the activation fuction

        Args:
            Z shape(m,layer.units): _description_

        Returns:
            np.ndarray: _description_
        """
        pass
    
    @staticmethod
    def prime(Z:np.ndarray, A:np.ndarray) -> np.ndarray:
        pass

class sigmoid(activation_f):
    @staticmethod
    def apply(Z:np.ndarray) -> np.ndarray:
        return 1/(1 + np.exp(-Z))
    
    @staticmethod
    def prime(Z:np.ndarray, A:np.ndarray) -> np.ndarray:
        return A * (1 - A)
    
class linear(activation_f):
    @staticmethod
    def apply(Z:np.ndarray) -> np.ndarray:
        return Z
    
    def prime(Z:np.ndarray, A:np.ndarray) -> np.ndarray:
        return 1

class ReLU(activation_f):
    @staticmethod
    def apply(Z:np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)
    
    @staticmethod
    def prime(Z:np.ndarray, A:np.ndarray) -> np.ndarray:
        return Z > 0 # becomes 1 where Z[i,j] > 0, and 0 otherwise
    
class softmax(activation_f):
    @staticmethod
    def apply(Z:np.ndarray) -> np.ndarray:
        e = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
    
    @staticmethod
    def prime(Z:np.ndarray, A:np.ndarray) -> np.ndarray:
        # When combined with CrossEntropyLoss, the derivative of the cost with respect
        # to Z is simply (A - Y_train). The cost function's prime method already
        # computes this. By returning 1 here, we ensure that the layer's
        # gradient update step uses that value directly.
        return 1
