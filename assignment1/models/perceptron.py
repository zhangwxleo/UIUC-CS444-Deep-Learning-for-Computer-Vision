"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.bias = np.zeros(n_class)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape
        np.random.seed(99)
        self.w = np.random.randn(self.n_class, D)
        for epoch in range(self.epochs):
            for i in range(N):
                x_i = X_train[i] # (784,)
                y_i = y_train[i] # (10,)
                scores = np.dot(self.w, x_i) # (10, )
                for c in range(len(self.w)):
                    if (scores[c]) + self.bias[c] > (np.dot(self.w[y_i], x_i) + self.bias[c]):
                        self.w[y_i] += self.lr * x_i
                        self.bias[y_i] += self.lr
                        self.w[c] -= self.lr * x_i
                        self.bias[y_i] -= self.lr
            print(f"Epoch {epoch + 1}/{self.epochs}, Accuracy: {self.get_acc(self.predict(X_train), y_train):.2f}%")
        return


    def predict(self, X_test: np.ndarray):
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        N, D = X_test.shape
        predict = np.zeros(N, dtype=int)
        
        for i in range(N):
            x = X_test[i]
            score = np.dot(self.w,x) + self.bias
            predict[i] = np.argmax(score)

        return predict
    

    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100
    
