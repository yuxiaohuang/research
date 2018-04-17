# Please cite the following paper when using the code


# Modules
import pandas as pd
import numpy as np
import math
from joblib import Parallel, delayed


class ALA:
    """
    The ALA classifier
    """

    def __init__(self, max_iter=100, min_samples_bin=1, C=1, n_jobs=-1):
        # The maximum number of iteration, 100 by default
        self.max_iter_ = max_iter

        # The minimum number of samples in each bin, 1 by default
        self.min_samples_bin_ = min_samples_bin

        # C, 1 by default
        self.C_ = C

        # The number of jobs to run in parallel, -1 by default (all CPUs are used)
        self.n_jobs_ = n_jobs

        # The dictionary of bins
        self.bins_ = {}

        # The dictionary of weights (u)
        self.us_ = {}

        # The dictionary of probabilities
        self.pis_ = {}

        # The dictionary of weights (w0 and w1)
        self.ws_ = {}

        # The dictionary of probabilities
        self.pijs_ = {}

        # The dictionary of probability distribution
        self.prob_dist_dict_ = {}

        # The list of Mean Square Errors
        self.mses_ = []

    def fit(self, X, y):
        """
        Train the ALA classifier
        :param X: the feature vector
        :param y: the target vector
        :return:
        """

        # Initialize the dictionary of bins
        # For each xj
        for j in range(X.shape[1] + 1):
            if j == 0:
                self.bins_[j] = [1, 1]
            else:
                # Get the unique values of xj
                xus = np.unique(X[:, j - 1])
                # Get the bin number
                bin_num = len(xus) / (len(xus) // 10 + self.min_samples_bin_)
                # Get the bins
                out, bins = pd.cut(X[:, j - 1], bin_num, retbins=True)
                self.bins_[j] = bins

        self.us_ = {}

        self.pis_ = {}

        self.ws_ = {}

        self.pijs_ = {}

        self.prob_dist_dict_ = {}

        self.mses_ = []

        self.gradient_descent_all(X, y)

    def gradient_descent_all(self, X, y):
        """
        Minimizing the cost function using (batch) gradient descent
        :param X: the feature vector
        :param y: the target vector
        :return:
        """

        for _ in range(self.max_iter_):
            # Gradient descent for each unique value of the target
            # Set backend="threading" to share memory between parent and threads
            # Parallel(n_jobs=self.n_jobs_, backend="threading")(delayed(self.gradient_descent_one)(X, y, yu)
            #                                                    for yu in np.unique(y))
            Parallel(n_jobs=1, backend="threading")(delayed(self.gradient_descent_one)(X, y, yu)
                                                    for yu in np.unique(y))

            # Update the mses
            self.update_mses(X, y)

    def gradient_descent_one(self, X, y, yu):
        """
        Minimizing the cost function using (batch) gradient descent
        :param X: the feature vector
        :param y: the target vector
        :param y: a unique value of the target
        :return:
        """

        if yu not in self.us_.keys():
            self.us_[yu] = {}
        # For each xj
        for j in range(X.shape[1] + 1):
            if j not in self.us_[yu]:
                self.us_[yu][j] = 0

        if yu not in self.ws_.keys():
            self.ws_[yu] = {}
        # For each xj
        for j in range(X.shape[1] + 1):
            if j not in self.ws_[yu]:
                self.ws_[yu][j] = {}
            for bin in range(len(self.bins_[j]) - 1):
                if bin not in self.ws_[yu][j]:
                    self.ws_[yu][j][bin] = [0, 0]

        # Get pis for yu
        self.get_pis(X, yu)

        delta_us = {}
        delta_ws = {}

        # For each xj
        for j in range(X.shape[1] + 1):
            delta_us[j] = 0
            delta_ws[j] = {}

            # For each row
            for i in range(X.shape[0]):
                # Get fi
                fi = 1 if y[i] == yu else 0

                # Get pi
                pi = self.pis_[yu][i]

                # Get pij
                pij = self.pijs_[yu][i][j]

                # Get delta_u
                delta_u = (fi - pi) * pij

                # Update delta_u of xj
                delta_us[j] += delta_u

                # Get uj
                uj = self.us_[yu][j]

                # Get xij
                xij = 1 if j == 0 else X[i][j - 1]

                # Get the bin xij falls into
                bin = self.get_bin(xij, j)

                # Get delta_w0
                delta_w0 = (fi - pi) * uj * pij * (1 - pij) * xij

                # Get delta_w1
                delta_w1 = (fi - pi) * uj * pij * (1 - pij) * 1

                # Initialize the dictionary of delta_w for key bin
                if bin not in delta_ws[j]:
                    delta_ws[j][bin] = [0, 0]

                # Update delta_w0 of xj
                delta_ws[j][bin][0] += delta_w0

                # Update delta_w1 of xj
                delta_ws[j][bin][1] += delta_w1

        # Initialize the maximum absolute delta_u
        max_abs_u = None
        # Get the maximum absolute delta_u
        for j in delta_us.keys():
            if max_abs_u == None or max_abs_u < abs(delta_us[j]):
                max_abs_u = abs(delta_us[j])
        max_abs_u = 1 if max_abs_u < 1 else max_abs_u
        # Update the dictionary of self.us_
        for j in delta_us.keys():
            self.us_[yu][j] += delta_us[j] / max_abs_u

        # Initialize the maximum absolute delta_w
        max_abs_w = None
        # Get the maximum absolute w0 and w1
        for j in delta_ws.keys():
            for bin in delta_ws[j].keys():
                if max_abs_w == None or max_abs_w < abs(delta_ws[j][bin][0]):
                    max_abs_w = abs(delta_ws[j][bin][0])
                if max_abs_w == None or max_abs_w < abs(delta_ws[j][bin][1]):
                    max_abs_w = abs(delta_ws[j][bin][1])
        max_abs_w = 1 if max_abs_w < 1 else max_abs_w
        # Update the dictionary of self.ws_
        for j in delta_ws.keys():
            for bin in delta_ws[j].keys():
                self.ws_[yu][j][bin][0] += delta_ws[j][bin][0] / max_abs_w
                self.ws_[yu][j][bin][1] += delta_ws[j][bin][1] / max_abs_w

    def get_pis(self, X, yu):
        """
        Get the minimum of all ujs for each row i
        :param X: the feature vector
        :param yu: an unique value of y
        :return: the minimum of all ujs for each row i
        """

        self.pis_[yu] = {}

        # Get pijs for yu
        self.get_pijs(X, yu)

        # For each row
        for i in range(X.shape[0]):
            zi = 0

            # For each xj
            for j in range(X.shape[1] + 1):
                # Get uj
                uj = self.us_[yu][j]

                # Get pij
                pij = self.pijs_[yu][i][j]

                # Update zi
                zi += uj * pij

            # Get pi
            pi = self.sigmoid(zi)

            # Update pis
            self.pis_[yu][i] = pi

    def get_pijs(self, X, yu):
        """
        Get the minimum of all ujs for each row i
        :param X: the feature vector
        :param yu: an unique value of y
        :return: the minimum of all ujs for each row i
        """

        self.pijs_[yu] = {}

        # For each row
        for i in range(X.shape[0]):
            if not i in self.pijs_[yu].keys():
                self.pijs_[yu][i] = {}

            for j in range(X.shape[1] + 1):
                # Get pij
                self.pijs_[yu][i][j] = self.get_pij(X, yu, i, j)

    def get_pij(self, X, yu, i, j):
        """
        Get pij
        :param X: the feature vector
        :param yu: an unique value of y
        :param i: row i
        :param j: the jth feature
        :return: pij
        """

        # Get xij
        xij = 1 if j == 0 else X[i][j - 1]

        # Get bin
        bin = self.get_bin(xij, j)

        # Get wj0 and wj1
        wj0 = self.ws_[yu][j][bin][0]
        wj1 = self.ws_[yu][j][bin][1]

        # Get zij
        zij = wj0 + wj1 * xij

        # Get pij
        pij = self.sigmoid(zij)

        return pij

    def sigmoid(self, z):
        """
        Get the sigmoid of z
        :param z: the sum of w * x
        :return: the sigmoid of z
        """

        if z < 0:
            return 1 - 1 / (1 + math.exp(z))
        return 1 / (1 + math.exp(-z))

    def get_bin(self, xij, j):
        """
        Get the bin for xij
        :param xij: the value of xj in row i
        :param j: the jth feature
        :return: the bin for xij
        """

        for idx in range(1, len(self.bins_[j])):
            if xij <= self.bins_[j][idx]:
                return idx - 1

        return len(self.bins_[j]) - 2

    def update_mses(self, X, y):
        """
        Update the mses
        :param X: the feature vector
        :param y: the target vector
        :return:
        """

        # Predict the class of each sample
        y_pred = self.predict(X)

        # Get mses
        mse = self.get_mse(y, y_pred)

        # Update the list of mse
        self.mses_.append(mse)

    def get_mse(self, y, y_pred):
        """
        Get the Mean Square Error (mse)
        :param y: the target vector
        :param y_pred: the predicted value of the target
        :return: the mse
        """

        # Initialize the square errors
        ses = []

        # For each row
        for i in range(y.shape[0]):
            # Update the square errors
            ses.append(0) if y[i] == y_pred[i] else ses.append(1)

        # Get mse
        mse = np.mean(ses)

        return mse

    def predict_proba(self, X):
        """
        Predict the probability of each class of each sample
        :param X: the feature vector
        :return: the (class, probability) pairs for each sample in X
                 sorted in descending order of the probability
        """

        # Initialize (class, probability) pairs for all samples
        yu_probs_all = []

        # For each unique value of the target
        for yu in self.pis_.keys():
            self.get_pis(X, yu)

        # For each row
        for i in range(X.shape[0]):
            # Initialize (class, probability) pairs for each sample
            yu_probs_each = []

            # For each unique value of the target
            for yu in self.pis_.keys():
                # Get pi
                pi = self.pis_[yu][i]

                # Update yu_probs_each
                yu_probs_each.append([yu, pi])

            # Sort yu_probs_each in descending order of the probability
            yu_probs_each = sorted(yu_probs_each, key=lambda x: x[1], reverse=True)

            # Update yu_probs_all
            yu_probs_all.append(yu_probs_each)

        return yu_probs_all

    def predict(self, X):
        """
        Predict the class of each sample
        :param X: the feature vector
        :return: the class of each sample in X
        """

        # Get (class, probability) pairs for all samples
        yu_probs_all = self.predict_proba(X)

        # Get y_pred (the predicted classes)
        y_pred = [yu_probs_each[0][0] for yu_probs_each in yu_probs_all]

        return np.asarray(y_pred)

    def get_prob_dist_dict(self, X):
        """
        Get the dictionary of probability distribution
        :param X: the feature vector
        :return:
        """

        # For each unique value of the target
        for yu in self.ws_.keys():
            if not yu in self.prob_dist_dict_:
                self.prob_dist_dict_[yu] = {}

            # For each xj
            for j in range(X.shape[1] + 1):
                if not j in self.prob_dist_dict_[yu]:
                    self.prob_dist_dict_[yu][j] = {}

                # Get the unique value and the corresponding index of xj
                xus, idxs = (np.unique([1], return_index=True) if j == 0
                             else np.unique(X[:, j - 1], return_index=True))

                # For each index
                for idx in idxs:
                    pij = self.get_pij(X, yu, idx, j)
                    xij = 1 if j == 0 else X[idx, j - 1]
                    self.prob_dist_dict_[yu][j][xij] = pij