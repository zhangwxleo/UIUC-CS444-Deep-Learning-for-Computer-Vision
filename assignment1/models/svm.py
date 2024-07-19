"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.bias = np.zeros(n_class)

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        N,D= X_train.shape
        dW = np.zeros(self.w.shape)
        
        for i in range(N):
            x_i = X_train[i]
            y_i = y_train[i]
            
            scores = np.dot(self.w, x_i)
            I = [(np.dot(self.w[y_i],x_i) - s < 1) for s in scores]
            dW[y_i] += self.reg_const*dW[y_i] - np.sum(I) * x_i
            
            for c in range(self.n_class):
                dW[c] += self.reg_const*dW[c] + I[c] * x_i               
        
        return dW
        
       

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape
        batch_size = 64
        C = self.n_class
        np.random.seed(99)
        self.w = np.random.randn(C,D)
        accu = 0
        
        print("Start training...")
        for epoch in range(self.epochs):
            #shuffle the data
            permutation = np.random.permutation(N)
            X_train = X_train[permutation]
            y_train = y_train[permutation]
            for i in range(0, N, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
        
                dW = self.calc_gradient(X_batch, y_batch)               
                for c in range(len(self.w)):
                  #  self.w[c] = (1-self.lr*self.reg_const/batch_num) * self.w[c] - self.lr*dW[c]
                    self.w[c] -= self.lr*dW[c]
                    
                db = np.sum(self.bias, axis=0) /  len(y_batch)
                self.bias -= self.lr * db
            accu += self.get_acc(self.predict(X_train), y_train)
            print(f"Epoch {epoch + 1}/{self.epochs}, Accuracy: {self.get_acc(self.predict(X_train), y_train):.2f}%")
        return accu/self.epochs


    def SVMloss(self, y_true, y_pred):
        hinge_loss = np.maximum(0, 1 - np.dot(y_true, y_pred))
        regularization = self.reg_const/(2*len(y_pred)) * np.linalg.norm(self.w) **2
        loss = np.mean(hinge_loss) + regularization
        return loss
    
    
    def predict(self, X_test: np.ndarray) :
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
        N = X_test.shape[0]
        for i in range(N):
            scores = np.dot(self.w, X_test[i])
            ms = float("-inf")
            mc = -1
            for j in range(len(scores)):
                if scores[j] > ms:
                    ms = scores[j]
                    mc = j
            result.append(mc)
        return result      
        
        
    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100
    
    
