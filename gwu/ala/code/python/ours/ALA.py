# Please cite the following paper when using the code


# Modules
import pandas as pd
import numpy as np
import math


class ALA:
    def __init__(self, max_iter=100, bin_num=2, C=1):
        # Initialize the maximum iteration
        self.max_iter_ = max_iter

        # Initialize c
        self.C_ = C

        self.bin_num_ = bin_num

        self.bins_ = {}

        # Initialize the dictionary of weights (w0 and w1)
        self.ws_ = {}

        # Initialize the dictionary of probability distribution
        self.prob_dist_dict_ = {}
        
        # Initialize the dictionary of Mean Square Errors
        self.mses_ = []

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """

        # For each xj
        for j in range(X.shape[1] + 1):
            if j == 0:
                # Initialize the dictionary of bins
                self.bins_[j] = [1, 1]
            else:
                xus = np.unique(X[:, j - 1])
                bin_num = len(xus) / self.bin_num_
                out, bins = pd.cut(X[:, j - 1], bin_num, retbins=True)
                # Initialize the dictionary of bins
                self.bins_[j] = bins

        # Initialize the dictionary of ws
        self.ws_ = {}

        # Initialize Mean Square Errors (mses)
        self.mses_ = []

        # Gradient descent
        self.gradient_descent(X, y)

    def gradient_descent(self, X, y):
        """Gradient descent.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """

        for iteration in range(self.max_iter_):
            # Print the iteration
            print('iteration: ' + str(iteration))

            # For each unique value of the target
            for yu in np.unique(y):
                # Initialize the dictionary of ws for yu
                if yu not in self.ws_:
                    self.ws_[yu] = {}

                # For each xj
                for j in range(X.shape[1] + 1):
                    # Initialize the dictionary of ws for xj
                    if j not in self.ws_[yu]:
                        self.ws_[yu][j] = {}

                    # Initialize the dictionary of ws for bin
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
                        if y[i] == yu:
                            fi = 1
                        else:
                            fi = 0

                        # Get prod_uijs
                        prod_uijs = prod_ujs[i]

                        # Get pij
                        pij = self.get_pij(X, yu, i, j)

                        # Get xij
                        if j == 0:
                            xij = 1
                        else:
                            xij = X[i][j - 1]

                        bin = self.get_bin(xij, j)

                        # Get delta_w0 of xj at row i
                        delta_wij0 = (fi + prod_uijs - 1) * prod_uijs * pij * -1 / self.C_

                        # Get delta_w1 of xj at row i
                        delta_wij1 = (fi + prod_uijs - 1) * prod_uijs * pij * -xij / self.C_

                        if bin not in delta_wij:
                            delta_wij[bin] = [0, 0]

                        # Update delta_w0 of xj
                        delta_wij[bin][0] += delta_wij0 * -1

                        # Update delta_w1 of xj
                        delta_wij[bin][1] += delta_wij1 * -1

                    for bin in delta_wij:
                        self.ws_[yu][j][bin][0] += delta_wij[bin][0]
                        self.ws_[yu][j][bin][1] += delta_wij[bin][1]

            # Update the mses
            self.update_mses(X, y)

    def get_bin(self, xij, j):
        for idx in range(1, len(self.bins_[j])):
            if xij <= self.bins_[j][idx]:
                return idx - 1

        return len(self.bins_[j]) - 2

    # Get the product of all ujs for yu
    def get_prod_ujs(self, X, yu):
        # Initialize prod_ujs
        prod_ujs = {}

        # For each row
        for i in range(X.shape[0]):
            # Update prod_ujs
            prod_ujs[i] = self.get_prod_uijs(X, yu, i)

        return prod_ujs

    # Get the product of all uijs for row i
    def get_prod_uijs(self, X, yu, i):
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

    # Get pij
    def get_pij(self, X, yu, i, j):
        # Get xij
        if j == 0:
            xij = 1
        else:
            xij = X[i][j - 1]

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
        if z < 0:
            return 1 - 1 / (1 + math.exp(z))
        return 1 / (1 + math.exp(-z))

    # Update the mses
    def update_mses(self, X, y):
        # Get the class of sample in X
        y_pred = self.predict(X)

        # Get mses
        mse = self.get_mse(y, y_pred)

        # Update the list of mse
        self.mses_.append(mse)

        return self.mses_

    # Get Mean Square Error (mse)
    def get_mse(self, y, y_hat):
        # Initialize the square errors
        ses = []

        # For each row
        for i in range(y.shape[0]):
            # Update the square errors
            if y[i] == y_hat[i]:
                ses.append(0)
            else:
                ses.append(1)

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

        # Initialize y_pred (the predicted values)
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
        :param X: {array-like} feature vector
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
                xus, idxus = (np.unique([1], return_index=True) if j == 0
                              else np.unique(X[:, j - 1], return_index=True))

                # For each unique index
                for idxu in idxus:
                    pij = self.get_pij(X, yu, idxu, j)
                    xij = 1 if j == 0 else X[idxu, j - 1]
                    self.prob_dist_dict_[yu][j][xij] = pij