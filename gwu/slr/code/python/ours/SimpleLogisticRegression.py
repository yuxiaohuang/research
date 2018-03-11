# Please cite the following paper when using the code


# Modules
import numpy as np
import math
from numpy import prod
from operator import itemgetter


class SimpleLogisticRegression:
    def __init__(self, max_iter=100, c=1, eta=0.1):
        # Initialize the maximum iteration
        self.max_iter = max_iter

        # Initialize c
        self.c = c

        # Initialize the learning rate
        self.eta_ = eta

        # Initialize the dictionary of weights (w0 and w1)
        self.ws_ = {}

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

        for iteration in range(self.max_iter):
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
                        self.ws_[yu][j] = [1, 1]

                # Initialize the dictionary of delta_ws (delta_w0 and delta_w1)
                delta_ws = {}

                # Initialize the dictionary of product of all ujs for yu
                prod_ujs = self.get_prod_ujs(X, yu)

                # For each xj
                for j in self.ws_[yu]:
                    # Initialize the dictionary of delta_ws (delta_w0 and delta_w1) for xj
                    if j not in delta_ws:
                        delta_ws[j] = [0, 0]

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

                        # Get delta_w0 of xj at row i
                        delta_wij0 = (fi - 1 + prod_uijs) * pij / (1 - prod_uijs) * -1

                        # Get delta_w1 of xj at row i
                        delta_wij1 = (fi - 1 + prod_uijs) * pij / (1 - prod_uijs) * -xij

                        # Update delta_w0 of xj
                        delta_ws[j][0] += delta_wij0

                        # Update delta_w1 of xj
                        delta_ws[j][1] += delta_wij1

                    # Update delta_wj0 and delta_wj1
                    delta_ws[j][0] *= -self.eta_
                    delta_ws[j][1] *= -self.eta_

                # For each xj
                for j in self.ws_[yu]:
                    # Update the dictionary of wights (w0 and w1)
                    self.ws_[yu][j][0] += delta_ws[j][0]
                    self.ws_[yu][j][1] += delta_ws[j][1]

            # Update the mses
            self.update_mses(iteration, X, y)

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
        # Get wj0 and wj1
        wj0 = self.ws_[yu][j][0]
        wj1 = self.ws_[yu][j][1]

        # Get xij
        if j == 0:
            xij = 1
        else:
            xij = X[i][j - 1]

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
    def update_mses(self, iteration, X, y):
        # Get the predicted value of the target
        y_hat, yu_probs_log = self.predict(X)

        # Get mses
        mse = self.get_mse(y, y_hat)

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

    def predict(self, X):
        # Initialize log of (yu, p(yu)) pairs, this will be written in the statistics file
        yu_probs_log = []

        # Initialize y_hat (the predicted values)
        y_hat = []

        # For each row
        for i in range(X.shape[0]):
            # Initialize (yu, p(yu)) pairs
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
            yu_probs = sorted(yu_probs, key=itemgetter(1), reverse=True)

            # Update yu_probs_log
            yu_probs_log.append(yu_probs)

            # Get y_hati
            y_hati = yu_probs[0][0]

            # Update y_hat
            y_hat.append(y_hati)

        return [np.asarray(y_hat), yu_probs_log]