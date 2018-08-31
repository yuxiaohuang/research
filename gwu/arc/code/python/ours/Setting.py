# Please cite the following paper when using the code

import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC

class Setting:
    """The Setting class"""

    def __init__(self, data_file, result_dir):

        # The label encoder
        self.encoder = LabelEncoder()

        # The k-fold cross validation
        self.n_splits = 10

        # The scaler
        self.scaler = StandardScaler()

        # The random state
        self.random_state = 0

        # The minimum number of samples required for calculating importance
        self.min_samples_importance = 30

        # The number of jobs to run in parallel, -1 indicates (all CPUs are used)
        self.n_jobs = -1

        # The dictionary of classifiers
        self.classifiers = ({'RandomForestClassifier': RandomForestClassifier,
                             'AdaBoostClassifier': AdaBoostClassifier,
                             'MLPClassifier': MLPClassifier,
                             'KNeighborsClassifier': KNeighborsClassifier,
                             'GaussianNB': GaussianNB,
                             'DecisionTreeClassifier': DecisionTreeClassifier,
                             'LogisticRegression': LogisticRegression,
                             'GaussianProcessClassifier': GaussianProcessClassifier,
                             'SVC': SVC})

        # The pathname of the parameter file directory
        self.parameter_file_dir = result_dir + 'parameter_file/'

        # The name of the parameter file
        self.parameter_file_name = os.path.basename(data_file).split('.')[0]

        # The type of the parameter file
        self.parameter_file_type = '.txt'

        # The pathname of the score file directory
        self.score_file_dir = result_dir + 'score_file/'

        # The name of the score file
        self.score_file_name = os.path.basename(data_file).split('.')[0]

        # The type of the score file
        self.score_file_type = '.txt'

        # The pathname of the interaction file directory
        self.interaction_file_dir = result_dir + 'interaction_file/'

        # The name of the interaction file
        self.interaction_file_name = os.path.basename(data_file).split('.')[0]

        # The type of the interaction file
        self.interaction_file_type = '.csv'
