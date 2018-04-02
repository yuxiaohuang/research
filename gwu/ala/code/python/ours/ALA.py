# Please cite the following paper when using the code


# Modules
import pandas as pd
import numpy as np
import math


class ALA:
    """
    The ALA classifier
    """

    def __init__(self, max_iter=100, min_samples_bin=3, C=1):
        # The maximum number of iteration, 100 by default
        self.max_iter_ = max_iter

        # The minimum number of samples in each bin, 3 by default
        self.min_samples_bin_ = min_samples_bin

        # C, 1 by default
        self.C_ = C

        # The dictionary of bins
        self.bins_ = {}

        # The dictionary of weights (w0 and w1)
        self.ws_ = {}

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
                bin_num = len(xus) / self.min_samples_bin_
                # Get the bins
                out, bins = pd.cut(X[:, j - 1], bin_num, retbins=True)
                self.bins_[j] = bins

        self.ws_ = {}

        self.prob_dist_dict_ = {}

        self.mses_ = []

        self.gradient_descent(X, y)

    def gradient_descent(self, X, y):
        """
        Minimizing the cost function using (batch) gradient descent
        :param X: the feature vector
        :param y: the target vector
        :return:
        """

        for _ in range(self.max_iter_):
            # For each unique value of the target
            for yu in np.unique(y):
                # Initialize the dictionary of ws
                if yu not in self.ws_:
                    self.ws_[yu] = {}
                # For each xj
                for j in range(X.shape[1] + 1):
                    if j not in self.ws_[yu]:
                        self.ws_[yu][j] = {}
                    for bin in range(len(self.bins_[j]) - 1):
                        if bin not in self.ws_[yu][j]:
                            self.ws_[yu][j][bin] = [0, 0]

                # Initialize the dictionary of product of all ujs for yu
                prod_ujs = self.get_prod_ujs(X, yu)

                # For each xj
                for j in self.ws_[yu]:
                    # Initialize the dictionary of delta_wij
                    delta_wij = {}

                    # For each row
                    for i in range(X.shape[0]):
                        # Get fi
                        fi = 1 if y[i] == yu else fi = 0

                        # Get prod_uijs
                        prod_uijs = prod_ujs[i]

                        # Get pij
                        pij = self.get_pij(X, yu, i, j)

                        # Get xij
                        xij = 1 if j == 0 else X[i][j - 1]

                        # Get the bin xij falls into
                        bin = self.get_bin(xij, j)

                        # Get delta_w0 of xj at row i
                        delta_wij0 = (fi + prod_uijs - 1) * prod_uijs * pij * -1 / self.C_

                        # Get delta_w1 of xj at row i
                        delta_wij1 = (fi + prod_uijs - 1) * prod_uijs * pij * -xij / self.C_

                        # Initialize the dictionary of delta_wij for key bin
                        if bin not in delta_wij:
                            delta_wij[bin] = [0, 0]

                        # Update delta_w0 of xj
                        delta_wij[bin][0] += delta_wij0 * -1

                        # Update delta_w1 of xj
                        delta_wij[bin][1] += delta_wij1 * -1

                    # Update the dictionary of self.ws_
                    for bin in delta_wij:
                        self.ws_[yu][j][bin][0] += delta_wij[bin][0]
                        self.ws_[yu][j][bin][1] += delta_wij[bin][1]

            # Update the mses
            self.update_mses(X, y)

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

    def get_prod_ujs(self, X, yu):
        """
        Get the product of all ujs for yu
        :param X: the feature vector
        :param yu: an unique value of y
        :return: the product of all ujs for yu
        """

        # Initialize prod_ujs
        prod_ujs = {}

        # For each row
        for i in range(X.shape[0]):
            # Update prod_ujs
            prod_ujs[i] = self.get_prod_uijs(X, yu, i)

        return prod_ujs

    def get_prod_uijs(self, X, yu, i):
        """
        Get the product of all uijs for row i
        :param X: the feature vector
        :param yu: an unique value of y
        :param i: row i
        :return: the product of all uijs for row i
        """

        # Initialize prod_uijs
        prod_uijs = 1

        # For each xj
        for j in range(X.shape[1] + 1):
            # Get pij
            pij = self.get_pij(X, yu, i, j)

            # Get uij
            uij = 1 - pij

            # Update prod_uijs
            prod_uijs *= uij

        return prod_uijs

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

        # For each row
        for i in range(X.shape[0]):
            # Initialize (class, probability) pairs for each sample
            yu_probs_each = []

            # For each unique value of the target
            for yu in self.ws_:
                # Get prod_uijs
                prod_uijs = self.get_prod_uijs(X, yu, i)

                # Get p(yu)
                prob = 1 - prod_uijs

                # Update yu_probs_each
                yu_probs_each.append([yu, prob])

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

        # Initialize y_pred (the predicted classes)
        y_pred = []

        # For each row
        for i in range(X.shape[0]):
            # Initialize (class, probability) pairs for each sample
            yu_probs = []

            # For each unique value of the target
            for yu in self.ws_:
                # Get prod_uijs
                prod_uijs = self.get_prod_uijs(X, yu, i)

                # Get p(yu)
                prob = 1 - prod_uijs

                # Update yu_probs
                yu_probs.append([yu, prob])

            # Sort yu_probs in descending order of prob
            yu_probs = sorted(yu_probs, key=lambda x: x[1], reverse=True)

            # Get y_predi
            y_predi = yu_probs[0][0]

            # Update y_pred
            y_pred.append(y_predi)

        return np.asarray(y_pred)

    def get_prob_dist_dict(self, X):
        """
        Get the probability distribution dictionary
        :param X: the feature vector
        :return:
        """

        # For each unique value of the target
        for yu in self.ws_:
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