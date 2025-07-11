from utils import sigmoid
import numpy as np
import math

SEED = 324523
np.random.seed(SEED)

class Net():
    def __init__(self, shape=[4, 1], activation_functions=None):
        self.shape = shape
        self.activation_functions = activation_functions

        weights = []
        biases = []
        # Adding weights to numpy array
        for i in range(1, len(shape)):
            # Initializing weights
            weights.append(np.random.rand(shape[i], shape[i-1]) - 0.5)
            # Initing biases to 0
            biases.append(np.zeros((shape[i])))

        self.weights = weights
        self.biases = biases

    def predict(self, x:np.ndarray):
        """
        Args:
            x (np.ndarray): _description_
            self.weights (np.ndarray):
            self.biases (np.ndarray):
        
        Returns:
            y_hat
        """
        activations = [x]
        for i in range(len(self.shape)-1):
            activations.append(np.dot(self.weights[i], activations[-1]) + self.biases[i])
        return activations[-1], activations
    
    def compute_cost(self, X_train, y_train):
        """ Computes cost for entire X_train

        Args:
            X_train (np.ndarray): (m, n)
            y_train (np.ndarray): (m)
            self.weights (np.ndarray):
            self.biases (np.ndarray):

        Returns:
            J_wb (cost)
        """

        m = X_train.shape[0]
        cost_vect = np.zeros((self.shape[-1]))
        cost = 0

        for i in range(m):
            y_hat = self.predict(X_train[i])[0]
            y_label = y_train[i]
            loss = (y_hat - y_label)**2
            cost_vect += loss/(2*m)
            cost += np.linalg.norm(loss/(2*m))
        return cost
    
    def gradient_descent(self, X_train:np.ndarray, y_train:np.ndarray, 
                         alpha:float=1e-4,num_iters:int=100000):
        J_history = []

        for i in range(num_iters):
            dj_dw_list, dj_db_list = self.compute_gradient(X_train, y_train)

            self.update_params(dj_dw_list, dj_db_list, alpha)

            if i<100000:      # prevent resource exhaustion 
                J_history.append(self.compute_cost(X_train, y_train))

            # Print cost every at intervals 20 times or as many iterations if < 20
            if i% math.ceil(num_iters / 20) == 0:
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


    
    def __str__(self):
        output = f'\nNeural Network:\n'
        output += f"{'+---------------+            '*(len(self.shape) - 1)}+----------------+\n"
        output += '|  Input Layer  |            '
        for i in range(len(self.shape) - 2):
            output += f'| Hidden Layer {i+1}|            '
        output += '|  Output Layer  |\n'
        
        output += f'|  {self.shape[0]} Neurons  | ---------> '
        for i in range(len(self.shape)-2):
            neurons = self.shape[1:-1][i]
            if neurons < 10:
                output += f'|   {neurons} Neurons   | ---------> '
            elif neurons < 100:
                output += f'|  {neurons} Neurons   | ---------> '
            else:
                output += f'|  {neurons} Neurons  | ---------> '

        output += f'|   {self.shape[-1]} Neurons   |\n'

        output += f"{'+---------------+            '*(len(self.shape) - 1)}+----------------+\n"
        return output

class multi_linear_regressor():
    def __init__(self, size:int=1):
        weights = np.random.rand(size)
        bias = 0

        self.weights = weights
        self.bias = bias

    def predict(self, x:np.ndarray):
        """
        Args:
            x (np.ndarray): _description_
            self.weights (np.ndarray):
            self.biases (np.ndarray):
        
        Returns:
            y_hat
        """
        y_hat = np.dot(self.weights, x) + self.bias
        return y_hat

    def compute_cost(self, X_train, y_train):
        """ Computes cost for entire X_train

        Args:
            X_train (np.ndarray): (m, n)
            y_train (np.ndarray): (m)
            self.weights (np.ndarray):
            self.biases (np.ndarray):

        Returns:
            J_wb (cost)
        """

        m = X_train.shape[0]
        cost:int = 0

        for i in range(m):
            loss = (self.predict(X_train[i]) - y_train[i])**2
            cost += loss/(2*m)
        # print(cost)
        return cost

    def gradient_descent(self, X_train:np.ndarray, y_train:np.ndarray, 
                         alpha:float=1e-4,num_iters:int=100000):

        J_history = []

        for i in range(num_iters):
            dj_dw, dj_db = self.compute_gradient(X_train, y_train)

            self.update_params(dj_dw, dj_db, alpha)

            if i<100000:      # prevent resource exhaustion 
                J_history.append(self.compute_cost(X_train, y_train))

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i% math.ceil(num_iters / 20) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]:8.6f}")
        
        return J_history #return final w,b and J history for graphing

    def update_params(self, dj_dw, dj_db, alpha):
        self.weights = self.weights - alpha*dj_dw
        self.bias = self.bias - alpha*dj_db

    # abstract method
    def compute_gradient(self, X_train, y_train):
        """_summary_

        Args:
            X_train (np.ndarray): 
            y_train (np.ndarray): 
            self.weights
            self.biases

        Returns:
            dj_dw (np.ndarray)
            dj_db (np.ndarray)
        """
        m = X_train.shape[0]

        dj_dw = np.zeros_like(self.weights)
        dj_db = 0

        for i in range(m):
            dj_dw += X_train[i]*(self.predict(X_train[i]) - y_train[i])/m
            dj_db += (self.predict(X_train[i]) - y_train[i])/m

        return dj_dw, dj_db
    

class LogisticRegression():
    def __init__(self, size:int=1):
        weights = np.random.rand(size)
        bias = 0

        self.weights = weights
        self.bias = bias

    def predict(self, x:np.ndarray):
        """
        Args:
            x (np.ndarray): _description_
            self.weights (np.ndarray):
            self.biases (np.ndarray):
        
        Returns:
            y_hat
        """
        y_hat = sigmoid(np.dot(self.weights, x) + self.bias)
        return y_hat

    def compute_cost(self, X_train, y_train):
        """ Computes cost for entire X_train

        Args:
            X_train (np.ndarray): (m, n)
            y_train (np.ndarray): (m)
            self.weights (np.ndarray):
            self.biases (np.ndarray):

        Returns:
            J_wb (cost)
        """

        m = X_train.shape[0]
        cost:int = 0

        for i in range(m):
            y_hat = self.predict(X_train[i])
            y_label = y_train[i]
            loss = - (y_label * np.log(y_hat) + (1 - y_label) * np.log(1 - y_hat))
            cost += loss/(m)
        return cost

    def gradient_descent(self, X_train:np.ndarray, y_train:np.ndarray, 
                         alpha:float=1e-4,num_iters:int=100000):

        J_history = []

        for i in range(num_iters):
            dj_dw, dj_db = self.compute_gradient(X_train, y_train)

            self.update_params(dj_dw, dj_db, alpha)

            if i<100000:      # prevent resource exhaustion 
                J_history.append(self.compute_cost(X_train, y_train))

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i% math.ceil(num_iters / 20) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]:8.6f}")
        
        return J_history #return final w,b and J history for graphing

    def update_params(self, dj_dw, dj_db, alpha):
        self.weights = self.weights - alpha*dj_dw
        self.bias = self.bias - alpha*dj_db

    # abstract method
    def compute_gradient(self, X_train, y_train):
        """_summary_

        Args:
            X_train (np.ndarray): 
            y_train (np.ndarray): 
            self.weights
            self.biases

        Returns:
            dj_dw (np.ndarray)
            dj_db (np.ndarray)
        """
        m = X_train.shape[0]

        dj_dw = np.zeros_like(self.weights)
        dj_db = 0

        for i in range(m):
            dj_dw += X_train[i]*(self.predict(X_train[i]) - y_train[i])/m
            dj_db += (self.predict(X_train[i]) - y_train[i])/m

        return dj_dw, dj_db