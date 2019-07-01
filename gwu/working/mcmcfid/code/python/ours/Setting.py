# Please cite the following paper when using the code

import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class Setting:
    """The Setting class"""

    def __init__(self, names_file, result_dir):

        # The label encoder
        self.encoder = LabelEncoder()

        # The k-fold cross validation
        self.n_splits = 10

        # The scaler
        self.scaler = StandardScaler()

        # The maximum number of iterations
        self.max_iter = 1000

        # The mean of the multivariate proposal distribution, 0 by default
        self.mean = 0

        # The grid of mean
        self.mean_grid = [0]

        # The covariance of the multivariate proposal distribution, 0 by default
        self.cov = 5

        # The grid of covariance
        self.cov_grid = [5]

        # The random state
        self.random_state = 0

        # The number of jobs to run in parallel, -1 indicates (all CPUs are used)
        self.n_jobs = 10

        # The scoring metric for hyperparameter tuning using GridSearchCV
        self.scoring = 'accuracy'

        # The pathname of the probability distribution figure directory
        self.proba_dists_fig_dir = result_dir + 'proba_dists_fig/'

        # The name of the probability distribution figure
        self.proba_dists_fig_name = os.path.basename(names_file).split('.')[0]

        # The type of the probability distribution figure
        self.proba_dists_fig_type = '.pdf'

        # The pathname of the probability distribution file directory
        self.proba_dists_file_dir = result_dir + 'proba_dists_file/'

        # The name of the probability distribution file
        self.proba_dists_file_name = os.path.basename(names_file).split('.')[0]

        # The type of the probability distribution file
        self.proba_dists_file_type = '.csv'

        # The pathname of the weights file directory
        self.weights_file_dir = result_dir + 'weights_file/'

        # The name of the weights file
        self.weights_file_name = os.path.basename(names_file).split('.')[0]

        # The type of the weights file
        self.weights_file_type = '.csv'

        # The pathname of the cv results file directory
        self.cv_results_file_dir = result_dir + 'cv_results_file/'

        # The name of the cv_results file
        self.cv_results_file_name = os.path.basename(names_file).split('.')[0]

        # The type of the cv_results file
        self.cv_results_file_type = '.csv'

        # The pathname of the best_params file directory
        self.best_params_file_dir = result_dir + 'best_params_file/'

        # The name of the best_params file
        self.best_params_file_name = os.path.basename(names_file).split('.')[0]

        # The type of the best_params file
        self.best_params_file_type = '.csv'

    def set_plt(self):
        """
        Set plt
        :return:
        """

        size = 30

        # Set text size
        plt.rc('font', size=size)

        # Set axes title size
        plt.rc('axes', titlesize=size)

        # Set x and y labels size
        plt.rc('axes', labelsize=size)

        # Set xtick and ytick size
        plt.rc('xtick', labelsize=size)
        plt.rc('ytick', labelsize=size)

        # Set legend size
        plt.rc('legend', fontsize=size)

        # Set figure title size
        plt.rc('figure', titlesize=size)

        plt.switch_backend('agg')