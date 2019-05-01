# Please cite the following paper when using the code


import numpy as np

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin


class FID(BaseEstimator, ClassifierMixin):
    """
    The FID model
    """

    def __init__(self, max_iter=100, bin_num_percent=1, min_bin_num=1, max_bin_num=100, feature_percent=1, eta=1, random_state=0, n_jobs=10):
        # The maximum number of iteration, 100 by default
        self.max_iter = max_iter

        # The percentage of the number of bins out of the number of unique value of a feature, 1 by default
        self.bin_num_percent = bin_num_percent

        # The minimum number of bins, 1 by default
        self.min_bin_num = min_bin_num

        # The maximum number of bins, 100 by default
        self.max_bin_num = max_bin_num

        # The percentage of features whose importance should be considered, 1 by default
        self.feature_percent = feature_percent

        # The learning rate, 1 by default
        self.eta = eta

        # The random state, 0 by default
        self.random_state = random_state

        # The number of jobs to run in parallel, -1 by default (all CPUs are used)
        self.n_jobs = n_jobs

        # The dictionary of bins
        self.bins = {}

        # The dictionary of weights
        self.weights = {}

        # The dictionary of probability distributions
        self.prob_dists = {}

    def fit(self, X, y):
        """
        Fit the FID model
        :param X: the feature matrix
        :param y: the target vector
        :return:
        """

        # Get the random number generator
        rgen = np.random.RandomState(seed=self.random_state)

        # Initialize the dictionaries
        self.init_dicts()

        # Get the dictionary of bins
        self.get_bins(X)

        # Get the dictionary of rows
        rows = self.get_rows(X)

        # Gradient descent for each class of the target
        # Set backend="threading" to share memory between parent and threads
        Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(self.gradient_descent)(X, y, class_, rgen, rows)
                                                          for class_ in sorted(np.unique(y)))

    def init_dicts(self):
        """
        # Initialize the dictionaries
        :return:
        """

        # The dictionary of bins
        self.bins = {}

        # The dictionary of weights
        self.weights = {}

        # The dictionary of probability distributions
        self.prob_dists = {}

    def get_bins(self, X):
        """
        Get the dictionary of bins
        :param X: the feature matrix
        :return:
        """

        for j in range(X.shape[1]):
            # Get the number of unique value of the jth feature
            xijs_num = len(np.unique(X[:, j]))

            # Get the bin number (must be in [self.min_bin_num, min(self.max_bin_num, xijs_num)])
            bin_num = np.clip(int(self.bin_num_percent * xijs_num),
                              self.min_bin_num,
                              min(self.max_bin_num, xijs_num))

            # Get the hist and bin_edges
            hist, bin_edges = np.histogram(X[:, j], bins=bin_num)

            # Remove empty bins
            self.bins[j] = np.hstack((bin_edges[:1], bin_edges[np.where(hist != 0)[0] + 1]))

    def get_rows(self, X):
        """
        Get the dictionary of rows
        :param X: the feature matrix
        :return: the dictionary of rows
        """

        rows = {}

        for j in range(X.shape[1]):
            rows[j] = {}

            for i in range(X.shape[0]):
                # Get the bin where X[i, j] belongs
                if X[i, j] >= max(self.bins[j]):
                    # Put X[i, j] to the last bin
                    bin = len(self.bins[j]) - 2
                elif X[i, j] <= min(self.bins[j]):
                    # Put X[i, j] to the first bin
                    bin = 0
                else:
                    # Put X[i, j] to the first bin where X[i, j] is smaller than the upper bound of the bin
                    bin = np.where(X[i, j] < self.bins[j])[0][0] - 1

                if bin not in rows[j].keys():
                    rows[j][bin] = []
                rows[j][bin].append(i)

        return rows

    def gradient_descent(self, X, y, class_, rgen, rows):
        """
        Minimize the cost function using (batch) gradient descent for one class
        :param X: the feature matrix
        :param y: the target vector
        :param class_: a class of the target
        :param rgen: the random number generator
        :param rows: the dictionary of rows
        :return:
        """

        # Initialize the weight dictionary
        self.init_weights(X, class_, rgen, rows)

        # Initialize the weight matrices
        W0, W1 = self.get_W(X, class_, rows)

        # Get the indicator vector
        I = self.get_I(y, class_)

        for _ in range(self.max_iter):
            # Get the probability matrix
            P = self.get_P(X, W0, W1)

            # Get the cost matrix
            L = self.get_L(X, P, I)

            # Update the weight matrices
            W0, W1 = self.update_W(X, rows, W0, W1, P, L)

        # Get the dictionary of weights
        self.get_weights(X, class_, rows, W0, W1)

        # Get the dictionary of probability distributions
        self.get_prob_dists(X, class_, W0, W1)

    def init_weights(self, X, class_, rgen, rows):
        """
        Initialize the weight dictionary
        :param X: the feature matrix
        :param class_: a class of the target
        :param rgen: the random number generator
        :param rows: the dictionary of rows
        :return:
        """

        self.weights[class_] = {}

        for j in range(X.shape[1]):
            self.weights[class_][j] = {}

            for bin in rows[j].keys():
                r0, r1 = rgen.normal(loc=0.0, scale=0.01, size=2)
                self.weights[class_][j][bin] = [r0, r1]

    def get_W(self, X, class_, rows):
        """
        Get the weight matrices
        :param X: the feature matrix
        :param class_: a class of the target
        :param rows: the dictionary of rows
        :return: the weight matrices
        """

        W0, W1 = np.zeros(X.shape), np.zeros(X.shape)

        for j in range(X.shape[1]):
            for bin in rows[j].keys():
                # Get the is such that X[i, j] belongs to bin
                is_ = rows[j][bin]
                W0[is_, j], W1[is_, j] = self.weights[class_][j][bin]

        return [W0, W1]

    def get_I(self, y, class_):
        """
        Get the indicator vector
        :param y: the target vector
        :param class_: a class of the target
        :return: the indicator vector
        """

        I = np.zeros(len(y)).reshape(-1, 1)
        I[np.where(y != class_)] = 1

        return I

    def get_P(self, X, W0, W1):
        """
        Get the probability matrix
        :param X: the feature matrix
        :param W0: the weight matrix
        :param W1: the weight matrix
        :return: the probability matrix
        """

        # Get the net input matrix
        Z = self.get_Z(X, W0, W1)

        return 1. / (1. + np.exp(-np.clip(Z, a_min=-250, a_max=250)))

    def get_Z(self, X, W0, W1):
        """
        Get the net input matrix
        :param X: the feature matrix
        :param W0: the weight matrix
        :param W1: the weight matrix
        :return: the net input matrix
        """

        return np.multiply(X, W1) + W0

    def get_L(self, X, P, I):
        """
        Get the cost matrix
        :param X: the feature matrix
        :param P: the probability matrix
        :param I: the indicator vector
        :return: the cost matrix
        """

        # Get the complement of the probability matrix
        Q = self.get_Q(X, P)
        # Sort the columns in ascending order
        Q.sort(axis=1)

        # Get the number of features whose importance should be considered
        feature_num = np.clip(int(self.feature_percent * X.shape[1]), a_min=1, a_max=X.shape[1])

        return np.prod(Q[:, :feature_num], axis=1).reshape(-1, 1) - I

    def get_Q(self, X, P):
        """
        Get the complement of the probability matrix
        :param X: the feature matrix
        :param P: the probability matrix
        :return: the complement of the probability matrix
        """

        return np.ones(X.shape) - P

    def update_W(self, X, rows, W0, W1, P, L):
        """
        Update the weight matrices
        :param X: the feature matrix
        :param rows: the dictionary of rows
        :param W0: the weight matrix
        :param W1: the weight matrix
        :param P: the probability matrix
        :param I: the indicator vector
        :param L: the cost matrix
        :return: the weight matrices
        """

        # Get the weight update matrices
        delta_W0, delta_W1 = self.get_delta_W(X, P, L)

        for j in range(X.shape[1]):
            for bin in rows[j].keys():
                # Get the is such that X[i, j] belongs to bin
                is_ = rows[j][bin]
                W0[is_, j] += np.sum(delta_W0[is_, j])
                W1[is_, j] += np.sum(delta_W1[is_, j])

        return [W0, W1]

    def get_delta_W(self, X, P, L):
        """
        Get the weight update matrices
        :param X: the feature matrix
        :param P: the probability matrix
        :param L: the cost matrix
        :return: the weight update matrices
        """

        delta_W0 = np.multiply(L, P)
        delta_W1 = np.multiply(delta_W0, X)

        return [delta_W0 * self.eta, delta_W1 * self.eta]

    def get_weights(self, X, class_, rows, W0, W1):
        """
        Get the dictionary of weights
        :param X: the feature matrix
        :param class_: a class of the target
        :param rows: the dictionary of rows
        :param W0: the weight matrix
        :param W1: the weight matrix
        :return:
        """

        for j in range(X.shape[1]):
            for bin in rows[j].keys():
                # Get the first i such that X[i, j] belongs to bin
                i = rows[j][bin][0]
                w0, w1 = W0[i, j], W1[i, j]
                self.weights[class_][j][bin] = [w0, w1]

    def get_prob_dists(self, X, class_, W0, W1):
        """
        Get the dictionary of probability distributions
        :param X: the feature matrix
        :param class_: a class of the target
        :param W0: the weight matrix
        :param W1: the weight matrix
        :return:
        """

        self.prob_dists[class_] = {}

        # Get the probability matrix
        P = self.get_P(X, W0, W1)

        for j in range(X.shape[1]):
            self.prob_dists[class_][j] = {}

            for i in range(X.shape[0]):
                self.prob_dists[class_][j][X[i, j]] = P[i, j]

    def predict(self, X):
        """
        Get the predicted class matrix
        :param X: the feature matrix
        :return: the predicted class matrix
        """

        # Get the predicted probability matrix
        PP = self.predict_proba(X)

        return np.array([sorted(self.weights.keys())[k] for k in np.argmax(PP, axis=1)])

    def predict_proba(self, X):
        """
        Get the predicted probability matrix
        :param X: the feature matrix
        :return: the predicted probability matrix
        """

        # Get the dictionary of rows
        rows = self.get_rows(X)

        PP = np.zeros((X.shape[0], len(sorted(self.weights.keys()))))

        for k in range(len(sorted(self.weights.keys()))):
            # Get the class
            class_ = sorted(self.weights.keys())[k]

            # Get the weight matrices
            W0, W1 = self.get_W(X, class_, rows)

            # Get the probability matrix
            P = self.get_P(X, W0, W1)

            # Get the complement of the probability matrix
            Q = self.get_Q(X, P)
            # Sort the columns in ascending order
            Q.sort(axis=1)

            # Get the number of features whose importance should be considered
            feature_num = np.clip(int(self.feature_percent * X.shape[1]), a_min=1, a_max=X.shape[1])

            # Get the probabilities of the class
            PP[:, k] = np.ones(X.shape[0]) - np.prod(Q[:, :feature_num], axis=1).reshape(-1)

        return PP
