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
        self.max_iter = 100

        # The grid of maximum number of iterations
        self.max_iters = [100]

        # The percentage of the number of bins out of the number of unique value of a feature
        self.bin_num_percent = 1

        # The grid of percentages of the number of bins out of the number of unique value of a feature
        self.bin_num_percents = [10 ** i for i in range(-3, 1)]

        # The minimum number of bins
        self.min_bin_num = 1

        # The maximum number of bins
        self.max_bin_num = 100

        # The percentage of features whose importance should be considered
        self.feature_percent = 1

        # The grid of percentages of features whose importance should be considered
        self.feature_percents = [0.1 * i for i in range(11)]

        # The learning rate
        self.eta = 1

        # The grid of learning rates
        self.etas = [10 ** i for i in range(-1, 2)]

        # The random state
        self.random_state = 0

        # The number of jobs to run in parallel, -1 indicates (all CPUs are used)
        self.n_jobs = 10

        # The scoring metric for hyperparameter tuning using GridSearchCV
        self.scoring = 'accuracy'

        # The pathname of the probability distribution figure directory
        self.prob_dists_fig_dir = result_dir + 'prob_dists_fig/'

        # The name of the probability distribution figure
        self.prob_dists_fig_name = os.path.basename(names_file).split('.')[0]

        # The type of the probability distribution figure
        self.prob_dists_fig_type = '.pdf'

        # The pathname of the probability distribution file directory
        self.prob_dists_file_dir = result_dir + 'prob_dists_file/'

        # The name of the probability distribution file
        self.prob_dists_file_name = os.path.basename(names_file).split('.')[0]

        # The type of the probability distribution file
        self.prob_dists_file_type = '.csv'

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