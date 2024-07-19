"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, D = X_train.shape
        np.random.seed(99)
        self.w = np.random.randn(D)
        accu = 0
        
        print("Start training...")
        for epoch in range(self.epochs):
            for i in range(N):
                x_i = X_train[i]
                y_i = y_train[i]
                scores = np.dot(self.w, x_i)
                dW = x_i*(self.sigmoid(scores) - y_i)
                self.w -= self.lr * dW
            accu += self.get_acc(self.predict(X_train), y_train)
            print(f"Epoch {epoch + 1}/{self.epochs}, Accuracy: {self.get_acc(self.predict(X_train), y_train):.2f}%")
        
        return accu/self.epochs

        

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        result = []
        N, D = X_test.shape
        for i in range(N):
            x_i = X_test[i]
            score = self.sigmoid(np.dot(self.w, x_i))
            if (score) >= self.threshold:
                result.append(1)
            else:
                result.append(0)
        return result

    def sigmoid(self, scores):
        return 1 / (1 + np.exp(-scores))
    
    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100