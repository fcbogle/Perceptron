#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:48:22 2024

@author: frankbogle
"""
import numpy as np


class WisGDAdaline:
    """Perceptron classifer.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.1. and 1.0)

    n_iter : int
        Passes over the training data

    random_state : int
        Random number generator seed for random weight initialization

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting

    b_ : Scalar
        Bias after fitting

   losses_ : list
        Number of misclassification (updates) in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples is the number of
        examples and n_features the number of features.

        y : array-like, shape = [n_examples]
        Target values.

        Returns
        ----------
        self : object

        """

        rgen = np.random.RandomState(self.random_state)
        print(X.shape[1])
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])

        self.b_ = np.float64(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            print("X shape:", X.shape)
            print("w_ shape:", self.w_.shape)
            print("errors shape:", errors.shape)
            print("X.T.dot(errors) shape:", X.T.dot(errors).shape)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors ** 2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.5, 1, 0)