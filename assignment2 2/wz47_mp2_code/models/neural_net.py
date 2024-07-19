"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.m = {}
        self.v = {}
        self.t = 0

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

            # TODO: You may set parameters for Adam optimizer here
            self.m["W" + str(i)] = np.zeros(self.params["W" + str(i)].shape)
            self.m["b" + str(i)] = np.zeros(self.params["b" + str(i)].shape)
            
            self.v["W" + str(i)] = np.zeros(self.params["W" + str(i)].shape)
            self.v["b" + str(i)] = np.zeros(self.params["b" + str(i)].shape)

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        output = np.dot(X,W) + b
        return output
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, b: np.ndarray, de_dz: np.ndarray, reg, N) -> np.ndarray:
        """Gradient of linear layer
            z = WX + b
            returns de_dw, de_db, de_dx
        """
        # TODO: implement me
        de_dw = (np.dot(X.T, de_dz) + reg*W)/N
        de_db = (np.sum(de_dz, axis=0))/N
        de_dx = np.dot(de_dz, W.T)
        
        return de_dw, de_db, de_dx
    

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        result = np.maximum(0, X)
        return result

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return (X > 0).astype(float)
    
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        return 1/(1 + np.exp(-x))

    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # TODO implement this
        X = self.sigmoid(X)
        return X*(1-X)

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        N = y.shape[0]
        result = np.sum(np.square(p-y))/N
        return result
        # return np.mean((y-p) ** 2)
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        N = y.shape[0] 
        return (2*p - 2*y)/N
    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        N = y.shape[0]
        return 2*(p-y)*p*(1-p)/N


    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        X = self.normalize(X)
        self.outputs = {'ReLu0': X}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.
        N, D = X.shape
        for i in range(1, self.num_layers+1):
            W, b = self.params[f'W{i}'], self.params[f'b{i}']
            
            # linear layer
            X = self.linear(W, X, b)
            self.outputs[f'Linear{i}'] = X
            # relu layer
            if i < self.num_layers:
                X = self.relu(X)
                self.outputs[f'ReLu{i}'] = X
            else:
                X = self.sigmoid(X) # final layer: sigmoid
                self.outputs['Sigmoid'] = X
        return X

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
        
        reg = 0
        
        # gradient of the last layer output
        de_dx = self.mse_sigmoid_grad(y, self.outputs['Sigmoid']) / y.shape[1]
    
        for i in range(self.num_layers, 0, -1):
            if i != self.num_layers:
                de_dz = de_dx * self.relu_grad(self.outputs[f'Linear{i}'])
            else:
                de_dz = de_dx * self.sigmoid_grad(self.outputs[f'Linear{i}'])
                   
            de_dW, de_db, de_dx = self.linear_grad(self.params['W' + str(i)], self.outputs['ReLu' + str(i-1)], self.params['b' + str(i)], de_dz, reg, y.shape[0])
            self.gradients[f'W{i}'] = de_dW 
            self.gradients[f'b{i}'] = de_db 
        
        return self.mse(y, self.outputs['Sigmoid']) 
    

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        if opt == "SGD":
            for key, val in self.gradients.items():
                self.params[key] -= lr*val
                
        elif opt == "Adam":
            self.t += 1
            for key in self.params.keys():
                self.adam(lr, b1, b2, eps, key)
                
    
    def adam(self, lr, b1, b2, epi, key):
        grad = self.gradients[key]
        self.m[key] = b1*self.m[key] + (1-b1) * grad
        self.v[key] = b2*self.v[key] + (1-b2) * (grad ** 2)
        m_hat = self.m[key] / (1-b1**self.t)
        v_hat = self.v[key] / (1-b2**self.t)
        
        self.params[key] -= lr/(np.sqrt(v_hat) + epi) * m_hat
    
    def normalize(self, X):
        X = X.astype(np.float64)
        X -= X.mean(0, keepdims=True)
        X /= X.std(0, keepdims=True) + (X.std(0, keepdims=True) == 0.0) * 1e-15

        return X