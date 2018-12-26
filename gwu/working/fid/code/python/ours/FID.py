# Please cite the following paper when using the code


# Modules
import pandas as pd
import numpy as np
import math

from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin


class FID(BaseEstimator, ClassifierMixin):
    """
    The FID model
    """

    def __init__(self, max_iter=100, bin_num_percent=1, eta=1, random_state=0, n_jobs=-1):
        # The maximum number of iteration, 100 by default
        self.max_iter = max_iter

        # The percentage of the number of bins out of the number of unique value of a feature, 1 by default
        self.bin_num_percent = bin_num_percent

        # The learning rate, 1 by default
        self.eta = eta

        # The random state, 0 by default
        self.random_state = random_state

        # The number of jobs to run in parallel, -1 by default (all CPUs are used)
        self.n_jobs = n_jobs

        # The classes of the target (sorted in ascending order)
        self.classes = []

        # The dictionary of indicators
        self.inds = {}

        # The dictionary of bins
        self.bins = {}

        # The dictionary of weights (w0 and w1)
        self.ws = {}

        # The dictionary of probability distributions
        self.prob_dists = {}

    def fit(self, X, y):
        """
        Fit the FID model
        :param X: the feature matrix
        :param y: the target vector
        :return:
        """

        # Get the classes of the target (sorted in ascending order)
        self.get_classes(y)

        # Get the dictionary of indicators
        self.get_inds(y)

        # Get the dictionary of bins
        self.get_bins(X)

        # Get the dictionary of weights (w0 and w1)
        self.get_ws(X)

        # Get the dictionary of probability distributions
        self.get_prob_dists(X)

    def get_classes(self, y):
        """
        Get the classes of the target (sorted in ascending order)
        :param y: the target vector
        :return:
        """

        self.classes = sorted(np.unique(y))

    def get_inds(self, y):
        """
        Get the dictionary of indicators (1, if yi != class_; 0, otherwise)
        :param y: the target vector
        :return:
        """

        for class_ in self.classes:
            self.inds[class_] = np.array([1 if yi != class_ else 0 for yi in y])

    def get_bins(self, X):
        """
        Get the dictionary of bins
        :param X: the feature matrix
        :return:
        """

        self.bins = {}

        for j in range(X.shape[1]):
            # Get the number of unique value of the jth feature
            xijs_num = len(np.unique(X[:, j]))

            # Get the bin number (must be in [1, xijs_num])
            if self.bin_num_percent <= 0:
                bin_num = xijs_num
            else:
                bin_num = np.clip(int(self.bin_num_percent * xijs_num), 1, xijs_num)

            # Get the bins
            out, bins = pd.cut(X[:, j], bin_num, retbins=True)

            self.bins[j] = bins

    def get_ws(self, X):
        """
        Get the dictionary of weights (w0 and w1)
        :return:
        """

        # Get the random number generator
        rgen = np.random.RandomState(self.random_state)

        for class_ in self.classes:
            self.ws[class_] = {}

            for j in range(X.shape[1]):
                self.ws[class_][j] = {}

                for bin in range(len(self.bins[j]) - 1):
                    self.ws[class_][j][bin] = rgen.normal(loc=0.0, scale=0.01, size=2)

    def get_prob_dists(self, X):
        """
        Get the dictionary of probability distributions
        :param X: the feature matrix
        :return:
        """

        # Minimize the cost function using (batch) gradient descent for all classes
        self.gradient_descent_all_classes(X)

        self.prob_dists = {}

        for class_ in self.classes:
            self.prob_dists[class_] = {}

            # Get the probability matrix
            P = self.get_P(X, class_)

            for j in range(X.shape[1]):
                self.prob_dists[class_][j] = {}

                for i in range(X.shape[0]):
                    self.prob_dists[class_][j][X[i, j]] = P[i, j]

    def gradient_descent_all_classes(self, X):
        """
        Minimize the cost function using (batch) gradient descent for all classes
        :param X: the feature matrix
        :return:
        """

        for _ in range(self.max_iter):
            # Gradient descent for each class of the target
            # Set backend="threading" to share memory between parent and threads
            Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(self.gradient_descent_one_class)(X, class_)
                                                               for class_ in self.classes)

    def gradient_descent_one_class(self, X, class_):
        """
        Minimize the cost function using (batch) gradient descent for one class
        :param X: the feature matrix
        :param class_: a class of the target
        :return:
        """

        # Get the probability matrix
        P = self.get_P(X, class_)

        # Get the cost matrix
        L = self.get_L(X, class_, P)

        # Update the dictionary of weights (w0 and w1)
        self.update_ws(X, class_, P, L)

    def get_P(self, X, class_):
        """
        Get the probability matrix
        :param X: the feature matrix
        :param class_: a class of the target
        :return: the probability matrix
        """

        # Get the net input matrix
        Z = self.get_Z(X, class_)

        return 1. / (1. + np.exp(-np.clip(Z, -250, 250)))

    def get_Z(self, X, class_):
        """
        Get the net input matrix
        :param X: the feature matrix
        :param class_: a class of the target
        :return: the net input matrix
        """

        # Get the weight matrices
        W0, W1 = self.get_Ws(X, class_)

        return np.multiply(X, W1) + W0

    def get_Ws(self, X, class_):
        """
        Get the weight matrices
        :param X: the feature matrix
        :param class_: a class of the target
        :return: the weight matrices
        """

        W0, W1 = np.zeros((X.shape[0], X.shape[1])), np.zeros((X.shape[0], X.shape[1]))

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Get the bin where X[i, j] belongs
                bin = self.get_bin(X, i, j)

                # Get W0[i, j] and W1[i, j]
                W0[i, j], W1[i, j] = self.ws[class_][j][bin]

        return [W0, W1]

    def get_bin(self, X, i, j):
        """
        Get the bin where X[i, j] belongs
        :param X: the feature matrix
        :param i: the ith sample
        :param j: the jth feature
        :return: the bin where X[i, j] belongs
        """

        for idx in range(1, len(self.bins[j])):
            if X[i, j] <= self.bins[j][idx]:
                return idx - 1

        return len(self.bins[j]) - 2

    def get_L(self, X, class_, P):
        """
        Get the cost matrix
        :param X: the feature matrix
        :param class_: a class of the target
        :param P: the probability matrix
        :return: the cost matrix
        """

        # Get the complement of the probability matrix
        Q = self.get_Q(X, P)

        return (Q.min(axis=1) - self.inds[class_]).reshape(-1, 1)

    def get_Q(self, X, P):
        """
        Get the complement of the probability matrix
        :param P: the probability matrix
        :return: the complement of the probability matrix
        """

        return np.ones((X.shape[0], X.shape[1])) - P

    def update_ws(self, X, class_, P, L):
        """
        Update the dictionary of weights (w0 and w1)
        :param X: the feature matrix
        :param class_: a class of the target
        :param P: the probability matrix
        :param L: the cost matrix
        :return:
        """

        # Get the weight update matrices
        delta_W0, delta_W1 = self.get_delta_Ws(X, L, P)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Get the bin where X[i, j] belongs
                bin = self.get_bin(X, i, j)

                self.ws[class_][j][bin][0] += delta_W0[i, j] * self.eta
                self.ws[class_][j][bin][1] += delta_W1[i, j] * self.eta

    def get_delta_Ws(self, X, L, P):
        """
        Get the weight update matrices
        :param X: the feature matrix
        :param L: the cost matrix
        :param P: the probability matrix
        :return: the weight update matrices
        """

        delta_W0 = np.multiply(L, P)
        delta_W1 = np.multiply(delta_W0, X)

        return [delta_W0, delta_W1]

    def predict_proba(self, X):
        """
        Get the predicted probability matrix
        :param X: the feature matrix
        :return: the predicted probability matrix
        """

        PP = np.zeros((X.shape[0], len(self.classes)))

        for k in range(len(self.classes)):
            # Get the class
            class_ = self.classes[k]

            # Get the probability matrix
            P = self.get_P(X, class_)

            # Get the complement of the probability matrix
            Q = self.get_Q(X, P)

            # Get the probabilities of the class
            PP[:, k] = np.ones(X.shape[0]) - Q.min(axis=1)

        return PP

    def predict(self, X):
        """
        Get the predicted class matrix
        :param X: the feature matrix
        :return: the predicted class matrix
        """

        # Get the predicted probability matrix
        PP = self.predict_proba(X)

        return np.array([self.classes[k] for k in np.argmax(PP, axis=1)])