# Please cite the following paper when using the code


# Modules
import numpy as np
import math
from numpy import prod
from operator import itemgetter


class EnsembleLogisticRegression:
    def __init__(self, max_iter=100, eta=0.1):
        self.max_iter = max_iter
        self.eta_ = eta
        self.feat_num = 0
        self.ws_ = {}
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

        # Initialize the number of features
        self.feat_num = X.shape[1]

        # If the latent feature has not been added
        if X.shape[1] == self.feat_num:
            # Get the latent feature
            late_feat = np.ones((X.shape[0], 1))

            # Add the latent feature to X
            X = np.hstack((X, late_feat))

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

        for interation in range(self.max_iter):
            print('interation: ' + str(interation))

            # For each unique value of the target
            for yu in np.unique(y):
                # Initialize the dictionary of weights (w0 and w1) for yu
                if not yu in self.ws_:
                    self.ws_[yu] = {}

                # For each xj
                for j in range(X.shape[1]):
                    # Initialize the dictionary of wights (w0 and w1) for xj
                    if not j in self.ws_[yu]:
                        self.ws_[yu][j] = [0, 0]

                # Initialize the dictionary of delta_ws (delta_w0 and delta_w1)
                delta_ws = {}

                # For each xj
                for j in range(X.shape[1]):
                    # Initialize the dictionary of delta_ws (delta_w0 and delta_w1) for xj
                    if not j in delta_ws:
                        delta_ws[j] = [0, 0]

                    # For each row
                    for i in range(X.shape[0]):
                        # Initialize the dictionary for xi, zi, pi, and ui
                        xi, zi, pi, ui = {}, {}, {}, {}

                        # For each xk
                        for k in range(X.shape[1]):
                            # Get wk0 and wk1
                            wk0 = self.ws_[yu][k][0]
                            wk1 = self.ws_[yu][k][1]

                            # Get the value of xk at row i
                            xi[k] = X[i][k]

                            # Get the value of zk at row i
                            zi[k] = wk0 + wk1 * xi[k]

                            # Get the value of pk at row i
                            pi[k] = self.sigmoid(zi[k])

                            # get the value of uk at row i
                            ui[k] = 1 - pi[k]

                        # Get delta_w0 of xj at row i
                        delta_wj0i = self.get_delta_wji(y, yu, j, i, pi, ui, -1)
                        # Update delta_w0 of xj
                        delta_ws[j][0] += delta_wj0i

                        # Get delta_w1 of xj at row i
                        delta_wj1i = self.get_delta_wji(y, yu, j, i, pi, ui, -xi[j])
                        # Update delta_w1 of xj
                        delta_ws[j][1] += delta_wj1i

                    # Update delta_wj0 and delta_wj1
                    delta_ws[j][0] *= -self.eta_
                    delta_ws[j][1] *= -self.eta_

                # For each xj
                for j in range(X.shape[1]):
                    # Update the dictionary of wights (w0 and w1)
                    self.ws_[yu][j][0] += delta_ws[j][0]
                    self.ws_[yu][j][1] += delta_ws[j][1]
                    # print([yu, j, self.ws_[yu][j][0], self.ws_[yu][j][1]])

                # print()

            # Update the mses
            self.update_mses(X, y)

    def sigmoid(self, z):
        if z < 0:
            return 1 - 1 / (1 + math.exp(z))
        return 1 / (1 + math.exp(-z))

    # Get delta_w of xj at row i
    def get_delta_wji(self, y, yu, j, i, pi, ui, part_deri_of_neg_zj_with_resp_to_wj):
        # Get the left part of delta_wji
        delta_wji_left = (prod([ui[k] for k in ui]) / ui[j]) * (pi[j] - (pi[j] ** 2)) * part_deri_of_neg_zj_with_resp_to_wj

        # If the value of y at row i is yu (the unique value in the parameters)
        if y[i] == yu:
            val = 1
        else:
            val = 0

        # Get the right part of delta_wji
        delta_wji_right = val / (1 - prod([ui[k] for k in ui])) - (1 - val) / prod([ui[k] for k in ui])

        # Get delta_wji
        delta_wji = delta_wji_left * delta_wji_right

        return delta_wji

    # Update the mses
    def update_mses(self, X, y):
        # Get the predicted value of the target
        y_hat = self.predict(X)

        print()

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
        # If the latent feature has not been added
        if X.shape[1] == self.feat_num:
            # Get the latent feature
            late_feat = np.ones((X.shape[0], 1))

            # Add the latent feature to X
            X = np.hstack((X, late_feat))

        # Initialize y_hat (the predicted values)
        y_hat = []

        # For each row
        for i in range(X.shape[0]):
            # Initialize (yu, P(yu)) pairs
            yu_p_yu_pairs = []

            # For each unique value of the target
            for yu in self.ws_:
                # Initialize P(yu)
                p_yu = 1
                # p_yu = 0

                # Update P(yu)
                # For each xj
                for j in self.ws_[yu]:
                    wj0i = self.ws_[yu][j][0]
                    wj1i = self.ws_[yu][j][1]
                    xji = X[i][j]
                    zji = wj0i + wj1i * xji
                    pji = self.sigmoid(zji)
                    uji = 1 - pji
                    p_yu *= uji
                    # p_yu += math.log(uji)

                # Get P(yu)
                p_yu = 1 - p_yu

                # Update (yu, P(yu)) pairs
                yu_p_yu_pairs.append([yu, p_yu])

            # Sort yu_p_yu_pairs in descending order of P(yu)
            yu_p_yu_pairs = sorted(yu_p_yu_pairs, key=itemgetter(1), reverse=True)
            # yu_p_yu_pairs = sorted(yu_p_yu_pairs, key=itemgetter(1))

            print(yu_p_yu_pairs)

            # Get y_hati
            y_hati = yu_p_yu_pairs[0][0]

            # Update y_hat
            y_hat.append(y_hati)

        return np.asarray(y_hat)