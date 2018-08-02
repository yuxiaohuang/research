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

        # The random state
        self.random_state = 0

        # The maximum number of iterations
        self.max_iter = 100

        # The minimum number of samples in each bin
        self.min_samples_bin = 1

        # The value of C
        self.C = 1

        # The number of jobs to run in parallel, -1 indicates (all CPUs are used)
        self.n_jobs = -1

        # The pathname of the parameter file directory
        self.parameter_file_dir = result_dir + 'parameter_file/'

        # The name of the parameter file
        self.parameter_file_name = os.path.basename(names_file).split('.')[0]

        # The type of the parameter_file
        self.parameter_file_type = '.txt'

        # The pathname of the probability distribution figure directory
        self.prob_dist_fig_dir = result_dir + 'prob_dist_fig/'

        # The name of the probability distribution figure
        self.prob_dist_fig_name = os.path.basename(names_file).split('.')[0]

        # The type of the probability distribution figure
        self.prob_dist_fig_type = '.pdf'

        # The pathname of the probability distribution file directory
        self.prob_dist_file_dir = result_dir + 'prob_dist_file/'

        # The name of the probability distribution file
        self.prob_dist_file_name = os.path.basename(names_file).split('.')[0]

        # The type of the probability distribution file
        self.prob_dist_file_type = '.csv'

        # The pathname of the score file directory
        self.score_file_dir = result_dir + 'score_file/'

        # The name of the score file
        self.score_file_name = os.path.basename(names_file).split('.')[0]

        # The type of the score file
        self.score_file_type = '.txt'

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