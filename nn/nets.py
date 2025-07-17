import numpy as np
from nn.layers import Layer
from nn.cost_functions import *
import math

SEED = 12345
np.random.seed(SEED)

class Net():
    def __init__(self, layers:list[Layer], cost_function:cost_function=MSE):
        self.layers = layers
        self.cost_function = cost_function

        #initializing weights
        for i in range(1, len(layers)):
            prev_units = layers[i-1].units
            layers[i].init_weights(prev_units)

    def predict(self, X:np.ndarray, save:bool=False):
        """ Forward through all layers
        X = [[1, 2, 3],  # x_0
             [4, 5, 6],] # x_1 or x_m

        Args:
            X shape=(m, n): X_train basically

        Returns:
            Y_hat shape=(m, self.layers[-1].shape[1]): 
            predictions for all X_i where i < m
        """

        curr_a = X
        for layer in self.layers:
            curr_a = layer(curr_a, save=save)
        return curr_a
    
    def compute_cost(self, X_train:np.ndarray, Y_train:np.ndarray):
        """ Computes cost for entire X_train

        Args:
            X_train (np.ndarray): (m, n)
            Y_train (np.ndarray): (m, self.layers[-1].units)

        Returns:
            J_wb (int): cost
        """

        Y_pred = self.predict(X_train)
        cost = self.cost_function.apply(Y_pred, Y_train)
        return cost
    
    def compute_accuracy(self, X_test:np.ndarray, Y_test:np.ndarray):
        m = X_test.shape[0]
        Y_pred = self.predict(X_test)
        Y_pred_label = np.argmax(Y_pred, axis=1)
        Y_label = np.argmax(Y_test)

        diff = Y_pred_label - Y_label
        diff = diff == 0
        num_correct = np.sum(diff)

        return num_correct / m
    
    def gradient_descent(self, X_train:np.ndarray, Y_train:np.ndarray, 
                         alpha:float=1e-8,epochs:int=20, batch_size:int=100):
        J_history = [self.compute_cost(X_train, Y_train)]
        print(f"Initial cost   : Cost {J_history[-1]:8.6f}")

        for i in range(epochs):
            for j in range(0, X_train.shape[0], batch_size):
                Y_pred = self.predict(X_train[j:j+batch_size], save=True)
                self.backprop(Y_pred, Y_train[j:j+batch_size], alpha)
            # Print cost every at intervals 20 times or as many iterations if < 20
            if i% math.ceil(epochs / 10) == 0 or i == epochs - 1:
                J_history.append(self.compute_cost(X_train, Y_train))
                print(f"Epoch {i:9d}: Cost {J_history[-1]:8.6f}")
        
        return J_history #return final w,b and J history for graphing

    # abstract method
    def backprop(self, Y_pred:np.ndarray, Y_train:np.ndarray, alpha:float):
        m = Y_train.shape[0]
        dJ_dy_pred = self.cost_function.prime(Y_pred, Y_train)
        dJ_da_prev = dJ_dy_pred
        for layer in self.layers[::-1]:
            dJ_da_prev = layer.gradient_and_update(dJ_da_prev, m, alpha)

    def save_model(self):
        """saves model weights and biases to a file,
        along with
        """