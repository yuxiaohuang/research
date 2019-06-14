# Please cite the following paper when using the code


import numpy as np

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin


class MCMCFID(BaseEstimator, ClassifierMixin):
    """
    The Markov Chain Monte Carlo Feature Importance Distribution model
    """

    def __init__(self, max_iter=2000, burn_in=500, mean=0, cov=0.1, random_state=0, n_jobs=10):
        # The maximum number of iteration, 100 by default
        self.max_iter = max_iter

        # The burn-in period, 1000 by default
        self.burn_in = burn_in

        # The mean of the multivariate proposal distribution, 0 by default
        self.mean = mean

        # The covariance of the multivariate proposal distribution, 1 by default
        self.cov = cov

        # The random state, 0 by default
        self.random_state = random_state

        # The number of jobs to run in parallel, -1 by default (all CPUs are used)
        self.n_jobs = n_jobs

        # The w0 and w1 trace dictionary
        self.w = None

        # The random number generator
        self.rgen = None

        # The cost vector
        self.cost = None

    def fit(self, X, y):
        """
        Fit the MCMCFID model
        :param X: the feature matrix
        :param y: the target vector
        :return:
        """

        # Initialize
        self.init(X)

        # Get the w0 and w1 trace dictionary
        self.get_w(X, y)

    def init(self, X):
        """
        Initialize
        :param X: the feature matrix
        :return:
        """

        # The mean of the multivariate proposal distribution
        self.mean = np.ones(X.shape[1]) * self.mean

        # The covariance of the multivariate proposal distribution
        self.cov = np.identity(X.shape[1]) * self.cov

        # The w0 and w1 trace dictionary
        self.w = {}

        # The random number generator
        self.rgen = np.random.RandomState(seed=self.random_state)

        # The cost vector
        self.cost = np.zeros(self.max_iter)

    def get_w(self, X, y):
        """
        Get the w0 and w1 trace dictionary
        :param X: the feature matrix
        :param y: the target vector
        :return:
        """

        # MCMC for each class of the target
        # Set backend="threading" to share memory between parent and threads
        Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(self.mcmc)(X, y, class_)
                                                          for class_ in sorted(np.unique(y)))

    def mcmc(self, X, y, class_):
        """
        MCMC for each class of the target
        :param X: the feature matrix
        :param y: the target vector
        :param class_: a class of the target
        :return:
        """

        # Get the indicator
        f = np.zeros(len(y))
        f[np.where(y != class_)] = 1

        # The w0 and w1 trace dictionary
        self.w[class_] = {}
        # The w0 and w1 trace matrix
        self.w[class_][0] = self.rgen.multivariate_normal(self.mean, self.cov, X.shape[0])
        self.w[class_][1] = self.rgen.multivariate_normal(self.mean, self.cov, X.shape[0])

        for iter in range(1, self.max_iter + 1):
            # Get the current w0 and w1 matrix
            w0_current = self.w[class_][0][(iter - 1) * X.shape[0]: iter * X.shape[0], :]
            w1_current = self.w[class_][1][(iter - 1) * X.shape[0]: iter * X.shape[0], :]

            # Get the proposed w0 and w1 matrix
            w0_proposed = w0_current + self.rgen.multivariate_normal(self.mean, self.cov, X.shape[0])
            w1_proposed = w1_current + self.rgen.multivariate_normal(self.mean, self.cov, X.shape[0])

            # Get the current likelihood matrix
            likelihood_current = self.get_likelihood(X, f, w0_current, w1_current)
            likelihood_current[np.where(likelihood_current == 0)] = np.exp(-250)

            # Get the proposed likelihood matrix
            likelihood_proposed = self.get_likelihood(X, f, w0_proposed, w1_proposed)
            likelihood_proposed[np.where(likelihood_current == 0) and np.where(likelihood_proposed == 0)] = np.exp(-250)

            # Get p_move
            p_move = np.clip(np.divide(likelihood_proposed, likelihood_current), None, 1)

            # Get the random number
            r = self.rgen.uniform(low=0, high=1, size=X.shape[0])

            # Get the indices where r > p_move
            idxs = np.where(r > p_move)

            # Update the proposed w0 and w1 (by rejecting the proposed samples at the above indices)
            w0_proposed[idxs] = w0_current[idxs]
            w1_proposed[idxs] = w1_current[idxs]

            # Update the w0 and w1 trace matrix
            self.w[class_][0] = np.vstack((self.w[class_][0], w0_proposed))
            self.w[class_][1] = np.vstack((self.w[class_][1], w1_proposed))

    def get_likelihood(self, X, f, w0, w1):
        """
        Get the likelihood matrix
        :param X: the feature matrix
        :return: the likelihood matrix
        """

        # Get the net input matrix
        z = np.multiply(X, w1) + w0

        # Get the probability matrix (using clip to avoid overflow)
        p = np.divide(1., 1. + np.exp(-np.clip(z, -250, 250)))

        # Get the complementary probability matrix
        cp = 1 - p

        # Get the joint probability vector
        jp = 1 - np.prod(cp, axis=1)

        return np.absolute(f - jp)

    def predict(self, X):
        """
        Get the predicted class vector
        :param X: the feature matrix
        :return: the predicted class vector
        """

        # Get the predicted probability matrix
        pp = self.predict_proba(X)

        return np.array(sorted(self.w.keys()))[np.argmax(pp, axis=1)]

    def predict_proba(self, X):
        """
        Get the predicted probability matrix
        :param X: the feature matrix
        :return: the predicted probability matrix
        """

        # The predicted probability matrix
        pp = np.zeros((X.shape[0], len(self.w.keys())))

        for k in range(len(self.w.keys())):
            # Get the class
            class_ = sorted(self.w.keys())[k]

            # Get f
            f = np.zeros(X.shape[0])

            # Get the w0 and w1 mean vector
            w0_mean = np.mean(self.w[class_][0][self.burn_in * X.shape[0]:, :], axis=0)
            w1_mean = np.mean(self.w[class_][1][self.burn_in * X.shape[0]:, :], axis=0)

            # Get the likelihood vector
            likelihood = self.get_likelihood(X, f, w0_mean, w1_mean)

            # Update pp
            pp[:, k] = likelihood

        return pp

    def get_vals_importance_per_feature(self, X, j, class_):
        """
        Get the importance vector for feature xj
        :param X: the feature matrix
        :param j: the column
        :param class_: a class of the target
        :return:
        """

        # Get the unique value of xj
        vals = np.unique(X[:, j])

        # Get f
        f = np.zeros(len(vals))

        # Get the w0 and w1 trace matrix
        w0 = self.w[class_][0][self.burn_in * X.shape[0]:, j].reshape(X.shape[0], -1, order='F')
        w1 = self.w[class_][1][self.burn_in * X.shape[0]:, j].reshape(X.shape[0], -1, order='F')

        # Get the w0 and w1 mean vector
        w0_mean = np.array([np.mean(w0[np.where(X[:, j] == val)]) for val in vals])
        w1_mean = np.array([np.mean(w1[np.where(X[:, j] == val)]) for val in vals])

        # Get the importance
        importance = self.get_likelihood(vals.reshape(-1, 1),
                                         f,
                                         w0_mean.reshape(-1, 1), w1_mean.reshape(-1, 1))

        return vals, importance

    def get_importance_per_value(self, X, i, j, class_):
        """
        Get the importance vector for feature value xij
        :param X: the feature matrix
        :param i: the row
        :param j: the column
        :param class_: a class of the target
        :return
        """

        # Get f
        f = np.zeros(self.max_iter)

        # Get the w0 and w1 trace matrix
        w0 = self.w[class_][0][:, j].reshape(X.shape[0], -1, order='F')
        w1 = self.w[class_][1][:, j].reshape(X.shape[0], -1, order='F')

        # Get the importance
        importance = self.get_likelihood(np.array(X[row, col]).reshape(-1, 1),
                                         f,
                                         w0[i].reshape(-1, 1), w1[i].reshape(-1, 1))

        return importance
