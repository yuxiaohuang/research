# Please cite the following paper when using the code

import os

from sklearn.preprocessing import LabelEncoder

class Setting:
    """The Setting class"""

    def __init__(self, data_file, result_dir):
        # # The list of number of iterations for searching for the rules
        # self.n_iters = [10 ** i for i in range(1, 4)]
        #
        # # The list of maximum number of conditions to consider when searching for the rules
        # self.max_condss = [10]
        #
        # # The list of minimum support required by the rules
        # self.min_supports = [0.1 * i for i in range(11)]
        #
        # # The list of minimum confidence required by the rules
        # self.min_confidences = [0.1 * i for i in range(11)]

        # The list of number of iterations for searching for the rules
        self.n_iters = [10000]

        # The list of maximum number of conditions to consider when searching for the rules
        self.max_condss = [10]

        # The list of minimum support required by the rules
        self.min_supports = [10 ** -2]

        # The list of minimum confidence required by the rules
        self.min_confidences = [0.9]

        # The random state
        self.random_state = 0

        # The number of jobs to run in parallel
        self.n_jobs = 1

        # The name of RandomPARC
        self.name = 'FAIR'

        # The label encoder
        self.encoder = LabelEncoder()

        # The pathname of the significant rule file directory
        self.sig_rule_file_dir = result_dir + 'sig_rule_file/'

        # The name of the significant rule file
        self.sig_rule_file_name = os.path.basename(data_file).split('.')[0]

        # The type of the significant rule file
        self.sig_rule_file_type = '.csv'
