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

    def predict(self, X:np.ndarray):
        """ Forward through all layers
        X = [[1, 2, 3],  # x_0
             [4, 5, 6],] # x_1 or x_m

        Args:
            X shape=(m, n): X_train basically

        Returns:
            Y_hat shape=(m, self.layers[-1].shape[1]): 
            predictions for all X_i where i < m
        """

        activations = []
        prev_a = X
        for layer in self.layers:
            curr_a = layer(prev_a)
            activations.append(curr_a)
            prev_a = curr_a
        return activations[-1], activations
    
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
    
    def gradient_descent(self, X_train:np.ndarray, y_train:np.ndarray, 
                         alpha:float=1e-4,num_iters:int=100000):
        J_history = []

        for i in range(num_iters):
            self.backpropagation

            if i<1000:      # prevent resource exhaustion 
                J_history.append(self.compute_cost(X_train, y_train))

            # Print cost every at intervals 20 times or as many iterations if < 20
            if i% math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:9d}: Cost {J_history[-1]:8.6f}")
        
        return J_history #return final w,b and J history for graphing

    def update_params(self, dj_dw_list, dj_db_list, alpha):
        for i in range(len(self.shape)-1):
            dj_dw, dj_db = dj_dw_list[i], dj_db_list[i]

            self.weights[i] -= alpha*dj_dw
            self.biases[i] -= alpha*dj_db

    # abstract method
    def compute_gradient(self, X_train, y_train):
        """backprop

        Args:
            X_train (np.ndarray): 
            y_train (np.ndarray): 
            self.weights
            self.biases

        Returns:
            dj_dw_list (list[np.ndarray]): dj_dw for each layer
            dj_db_list (list[np.ndarray]): dj_db for each layer
        """
        m = X_train.shape[0]

        # Initialize gradients
        dj_dw_list = [np.zeros_like(w) for w in self.weights]
        dj_db_list = [np.zeros_like(b) for b in self.biases]

        # Go through all training examples
        # For each training example, calculate the 
        for i in range(m):
            y_hat, activations = self.predict(X_train[i])
            y_label = y_train[i]
            activations.append(y_label)
            for j in range(len(self.shape)-2, -1, -1):
                dj_dw_list[j] += X_train[j+1]*(activations[j+1] - activations[j+2])/m
                dj_db_list[j] += (activations[j+1] - activations[j+2])/m
        # print(dj_dw_list, len(dj_dw_list))
        return dj_dw_list, dj_db_list