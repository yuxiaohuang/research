# Please cite the following paper when using the code

import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

        # The number of jobs to run in parallel, -1 indicates (all CPUs are used)
        self.n_jobs = -1

        # The dictionary of classifiers
        self.classifiers = ({'RandomForestClassifier': RandomForestClassifier(random_state=self.random_state,
                                                                              n_jobs=self.n_jobs),
                             'LogisticRegression': LogisticRegression(random_state=self.random_state,
                                                                      n_jobs=self.n_jobs)})

        # The dictionary of parameter grids
        self.param_grids = ({'RandomForestClassifier': [{'RandomForestClassifier__n_estimators': [10 ** i for i in range(1, 4)],
                                                         'RandomForestClassifier__criterion': ['gini', 'entropy'],
                                                         'RandomForestClassifier__min_samples_split': [max(2, 10 ** i) for i in range(0, 4)],
                                                         'RandomForestClassifier__min_samples_leaf': [10 ** i for i in range(0, 4)],
                                                         'RandomForestClassifier__max_features': ['auto', 'log2'],
                                                         'RandomForestClassifier__bootstrap': ['True', 'False'],
                                                         'RandomForestClassifier__oob_score': ['True', 'False'],
                                                         'RandomForestClassifier__warm_start': ['True', 'False']}],
                             'LogisticRegression': [{'LogisticRegression__tol': [10 ** i for i in range(-5, 1)],
                                                     'LogisticRegression__C': [10 ** i for i in range(-5, 1)],
                                                     'LogisticRegression__solver': ['newton-cg',
                                                                                    'lbfgs',
                                                                                    'liblinear',
                                                                                    'sag',
                                                                                    'saga'],
                                                     'LogisticRegression__max_iter': [10 ** i for i in range(1, 4)],
                                                     'LogisticRegression__multi_class': ['ovr']},
                                                    {'LogisticRegression__tol': [10 ** i for i in range(-5, 1)],
                                                     'LogisticRegression__C': [10 ** i for i in range(-5, 1)],
                                                     'LogisticRegression__solver': ['newton-cg',
                                                                                    'lbfgs',
                                                                                    'sag',
                                                                                    'saga'],
                                                     'LogisticRegression__max_iter': [10 ** i for i in range(1, 4)],
                                                     'LogisticRegression__multi_class': ['multinomial']}]})

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

        # The pathname of the feature importances figure directory
        self.feature_importances_fig_dir = result_dir + 'feature_importances_fig/'

        # The name of the feature importances figure
        self.feature_importances_fig_name = os.path.basename(names_file).split('.')[0]

        # The type of the feature importances figure
        self.feature_importances_fig_type = '.pdf'

        # The pathname of the feature importances file directory
        self.feature_importances_file_dir = result_dir + 'feature_importances_file/'

        # The name of the feature importances file
        self.feature_importances_file_name = os.path.basename(names_file).split('.')[0]

        # The type of the feature importances file
        self.feature_importances_file_type = '.csv'

        # The pathname of the cv results file directory
        self.cv_results_file_dir = result_dir + 'cv_results_file/'

        # The name of the cv_results file
        self.cv_results_file_name = os.path.basename(names_file).split('.')[0]

        # The type of the cv_results file
        self.cv_results_file_type = '.csv'

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