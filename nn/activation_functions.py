import numpy as np

class activation_f():
    @staticmethod
    def apply(self, Z:np.ndarray) -> np.ndarray:
        """calls the activation fuction

        Args:
            Z shape(m,layer.units): _description_

        Returns:
            np.ndarray: _description_
        """
        pass
    
    @staticmethod
    def prime(self, Z:np.ndarray) -> np.ndarray:
        pass


class sigmoid(activation_f):
    @staticmethod
    def apply(self, Z:np.ndarray) -> np.ndarray:
        return 1/(1 + np.exp(-Z))
    
class linear(activation_f):
    @staticmethod
    def apply(Z:np.ndarray) -> np.ndarray:
        return Z

class ReLU(activation_f):
    @staticmethod
    def apply(Z:np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)
    
class softmax(activation_f):
    @staticmethod
    def apply(Z:np.ndarray) -> np.ndarray:
        e = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)