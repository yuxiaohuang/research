# Please cite the following paper when using the code


import numpy as np
import math

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold

class BLR(BaseEstimator, ClassifierMixin):
    """
    The BLR (Binarized Logistic Regression) model
    """

    def __init__(self,
                 max_iter=100,
                 bin_num_percent=0,
                 min_bin_num=1,
                 max_bin_num=100,
                 max_batch_sample_num=100,
                 eta=1,
                 tol=10 ** -4,
                 random_state=0,
                 n_jobs=10):
        # The maximum number of iteration, 100 by default
        self.max_iter = max_iter

        # The percentage of the number of bins out of the number of unique value of a feature, 0 by default
        self.bin_num_percent = bin_num_percent

        # The minimum number of bins
        self.min_bin_num = min_bin_num

        # The maximum number of bins
        self.max_bin_num = max_bin_num

        # The maximum number of samples in a batch
        self.max_batch_sample_num = max_batch_sample_num

        # The learning rate, 1 by default
        self.eta = eta

        # The tolerance for stopping criteria, 10 ** -4 by default
        self.tol = tol

        # The random state, 0 by default
        self.random_state = random_state

        # The number of jobs to run in parallel, -1 by default (all CPUs are used)
        self.n_jobs = n_jobs

        # The dictionary of bins
        self.bins = {}

        # The bin matrix
        self.B = None

        # The dictionary of bin numbers
        self.bin_nums = {}

        # The dictionary of weights
        self.weights = {}

        # The dictionary of probability distributions
        self.prob_dists = {}

        # The dictionary of mini-batch
        self.mini_batch = {}

        # The random number generator
        self.rgen = None

    def fit(self, X, y):
        """
        Fit the BLR model
        :param X: the feature matrix
        :param y: the target vector
        :return:
        """

        # Initialize the attributes
        self.init_attributes(X, y)

        # Gradient descent for each class of the target
        # Set backend="threading" to share memory between parent and threads
        Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(self.gradient_descent)(X, y, class_)
                                                          for class_ in sorted(np.unique(y)))

        # Get the dictionary of probability distributions
        self.get_prob_dists(X, y)

    def init_attributes(self, X, y):
        """
        # Initialize the attributes
        :param X: the feature matrix
        :param y: the target vector
        :return:
        """

        # The dictionary of bins
        self.bins = {}

        # The bin matrix
        self.B = np.zeros(X.shape)

        # The dictionary of bin numbers
        self.bin_nums = {}

        # The dictionary of weights
        self.weights = {}

        # The dictionary of probability distributions
        self.prob_dists = {}

        # The dictionary of mini-batch
        self.mini_batch = {}

        # Get the dictionary of bins
        self.get_bins(X)

        # Get the bin matrix
        self.get_B(X)

        # Get the dictionary of bin numbers
        self.get_bin_nums(X)

        # Get the dictionary of mini-batch
        self.get_mini_batch(X, y)

        # Get the random number generator
        self.rgen = np.random.RandomState(seed=self.random_state)

    def get_bins(self, X):
        """
        Get the dictionary of bins
        :param X: the feature matrix
        :return:
        """

        for j in range(X.shape[1]):
            # Get the number of unique value of the jth feature
            xijs_num = len(np.unique(X[:, j]))

            # Get the bin number
            bin_num = np.clip(int(self.bin_num_percent * xijs_num),
                              self.min_bin_num,
                              min(self.max_bin_num, xijs_num, max(int(self.max_batch_sample_num / (2 * X.shape[1])), 1)))

            # Get the hist and bin_edges
            hist, bin_edges = np.histogram(X[:, j], bins=bin_num)

            # Remove empty bins
            self.bins[j] = np.hstack((bin_edges[:1], bin_edges[np.where(hist != 0)[0] + 1]))

    def get_B(self, X):
        """
        Get the bin matrix
        :param X: the feature matrix
        :return:
        """

        for j in range(X.shape[1]):
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

                self.B[i, j] = bin

    def get_bin_nums(self, X):
        """
        Get the dictionary of bin numbers
        :param X: the feature matrix
        :return:
        """

        for j in range(X.shape[1]):
            self.bin_nums[j] = np.unique(self.B[:, j])

    def get_mini_batch(self, X, y):
        """
        Get the dictionary of mini-batch
        :param X: the feature matrix
        :param y: the target vector
        :return:
        """

        # Initialize the fold number
        fold = 0

        # Get the number of splits
        n_splits = int(math.ceil(X.shape[0] / self.max_batch_sample_num))

        # If only 1 fold
        if n_splits <= 1:
            self.mini_batch[fold] = np.array(range(X.shape[0]))
        else:
            skf = StratifiedKFold(n_splits=n_splits, random_state=self.random_state)

            for train_idx, test_idx in skf.split(X, y):
                # Get the mini-batch for the fold
                self.mini_batch[fold] = test_idx

                # Update the fold
                fold += 1

    def gradient_descent(self, X, y, class_):
        """
        Minimize the cost function using (batch) gradient descent for one class
        :param X: the feature matrix
        :param y: the target vector
        :param class_: a class of the target
        :return:
        """

        # Initialize the weight dictionary
        self.init_weights(X, class_)

        # Initialize the flag variable, indicating the convergence of gradient descent
        converge = False

        # Initialize the old probability vector
        p_old = None

        for _ in range(self.max_iter):
            for fold in sorted(self.mini_batch.keys()):
                # Get the mini-batch
                mini_batch = self.mini_batch[fold]

                # Initialize the weight matrices
                W0, W1 = self.get_W(X[mini_batch, :], mini_batch, class_)

                # Get the indicator vector
                f = self.get_f(y[mini_batch], class_)

                # Get the probability vector
                p = self.get_p(X[mini_batch, :], W0, W1)

                # Update the weight matrices
                W0, W1 = self.update_W(X[mini_batch, :], mini_batch, W0, W1, f, p)

                # Get the dictionary of weights
                self.get_weights(X[mini_batch, :], mini_batch, class_, W0, W1)

                if p_old is not None:
                    # Get the absolute difference between p and p_old
                    abs_diff = abs(np.mean(p) - np.mean(p_old))

                    # If the absolute difference is smaller than the threshold
                    if abs_diff < self.tol:
                        # Gradient descent has converged
                        converge = True
                        break

                # Update the old probability vector
                p_old = np.array(p)

            if converge is True:
                break

    def init_weights(self, X, class_):
        """
        Initialize the weight dictionary
        :param X: the feature matrix
        :param class_: a class of the target
        :return:
        """

        self.weights[class_] = {}

        for j in range(X.shape[1]):
            self.weights[class_][j] = {}

            for bin in np.unique(self.B[:, j]):
                r0, r1 = self.rgen.normal(loc=0.0, scale=0.01, size=2)
                self.weights[class_][j][bin] = [r0, r1]

    def get_W(self, X, mini_batch, class_):
        """
        Get the weight matrices
        :param X: the feature matrix
        :param mini_batch: the mini-batch
        :param class_: a class of the target
        :return: the weight matrices
        """

        W0, W1 = np.zeros(X.shape), np.zeros(X.shape)

        for j in range(X.shape[1]):
            for i in range(X.shape[0]):
                # Get the bin
                bin = self.B[mini_batch[i], j]

                W0[i, j], W1[i, j] = self.weights[class_][j][bin]

        return [W0, W1]

    def get_f(self, y, class_):
        """
        Get the indicator vector
        :param y: the target vector
        :param class_: a class of the target
        :return: the indicator vector
        """

        f = np.zeros(len(y))
        f[np.where(y == class_)] = 1

        return f

    def get_p(self, X, W0, W1):
        """
        Get the probability vector
        :param X: the feature matrix
        :param W0: the weight matrix
        :param W1: the weight matrix
        :return: the probability vector
        """

        # Get the net input vector
        z = self.get_z(X, W0, W1)

        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def get_z(self, X, W0, W1):
        """
        Get the net input vector
        :param X: the feature matrix
        :param W0: the weight matrix
        :param W1: the weight matrix
        :return: the net input vector
        """

        return np.sum(np.multiply(X, W1) + W0, axis=1)

    def update_W(self, X, mini_batch, W0, W1, f, p):
        """
        Update the weight matrices
        :param X: the feature matrix
        :param mini_batch: the mini-batch
        :param W0: the weight matrix
        :param W1: the weight matrix
        :param f: the indicator vector
        :param p: the probability vector
        :return: the weight matrices
        """

        # Get the weight update matrices
        delta_W0, delta_W1 = self.get_delta_W(X, f, p)

        for j in range(X.shape[1]):
            for bin in self.bin_nums[j]:
                # Get the is such that X[i, j] belongs to bin
                is_ = np.where(self.B[mini_batch, j] == bin)
                W0[is_, j] += np.mean(delta_W0[is_, j])
                W1[is_, j] += np.mean(delta_W1[is_, j])

        return [W0, W1]

    def get_delta_W(self, X, f, p):
        """
        Get the weight update matrices
        :param X: the feature matrix
        :param f: the indicator vector
        :param p: the probability vector
        :return: the weight update matrices
        """

        delta_W0 = np.multiply((f - p).reshape(-1, 1), np.ones(X.shape))
        delta_W1 = np.multiply(delta_W0, X)

        return [delta_W0 * self.eta, delta_W1 * self.eta]

    def get_weights(self, X, mini_batch, class_, W0, W1):
        """
        Get the dictionary of weights
        :param X: the feature matrix
        :param class_: a class of the target
        :param W0: the weight matrix
        :param W1: the weight matrix
        :return:
        """

        for j in range(X.shape[1]):
            for bin in self.bin_nums[j]:
                # Get the is such that X[i, j] belongs to bin
                is_ = np.where(self.B[mini_batch, j] == bin)
                w0, w1 = np.mean(W0[is_, j]), np.mean(W1[is_, j])
                self.weights[class_][j][bin] = [w0, w1]

    def get_prob_dists(self, X, y):
        """
        Get the dictionary of probability distributions
        :param X: the feature matrix
        :param y: the target vector
        :return:
        """

        for class_ in sorted(np.unique(y)):
            self.prob_dists[class_] = {}

            # Get the weight matrices
            W0, W1 = self.get_W(X, np.array(range(X.shape[0])), class_)

            for j in range(X.shape[1]):
                self.prob_dists[class_][j] = {}

                # Get the jth feature
                Xj = np.zeros(X.shape)
                Xj[:, j] = X[:, j]

                # Get the jth W0
                W0j = np.zeros(W0.shape)
                W0j[:, j] = W0[:, j]

                # Get the probability vector
                p = self.get_p(Xj, W0j, W1)

                # Get the unique value of the jth feature and their indices
                xijs, is_ = np.unique(X[:, j], return_index=True)

                for i in is_:
                    self.prob_dists[class_][j][i] = p[i]

        # Normalize the dictionary of probability distributions
        for j in range(X.shape[1]):
            # Get the unique value of the jth feature and their indices
            xijs, is_ = np.unique(X[:, j], return_index=True)

            for i in is_:
                sum = 0
                for class_ in self.prob_dists.keys():
                    sum += self.prob_dists[class_][j][i]

                for class_ in self.prob_dists.keys():
                    self.prob_dists[class_][j][i] /= sum

    def predict(self, X):
        """
        Get the predicted class vector
        :param X: the feature matrix
        :return: the predicted class vector
        """

        # Get the predicted probability matrix
        PP = self.predict_proba(X)

        return np.array(sorted(self.weights.keys()))[np.argmax(PP, axis=1)]

    def predict_proba(self, X):
        """
        Get the predicted probability matrix
        :param X: the feature matrix
        :return: the predicted probability matrix
        """

        # Get the bin matrix
        self.get_B(X)

        PP = np.zeros((X.shape[0], len(sorted(self.weights.keys()))))

        for k in range(len(sorted(self.weights.keys()))):
            # Get the class
            class_ = sorted(self.weights.keys())[k]

            # Get the weight matrices
            W0, W1 = self.get_W(X, np.array(range(X.shape[0])), class_)

            # Get the probability vector
            p = self.get_p(X, W0, W1)

            # Get the probabilities of the class
            PP[:, k] = p

        return PP
