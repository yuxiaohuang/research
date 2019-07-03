# Please cite the following paper when using the code

import os

from sklearn.preprocessing import LabelEncoder
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
        # The base classifier
        self.base = None

        # The number of iterations for searching for the rules
        self.n_iter = 2

        # The minimum support required by the rules
        self.min_support = 0

        # The minimum confidence required by the rules
        self.min_confidence = 1

        # The maximum number of conditions to consider when searching for the rules
        self.max_conds = 10

        # The random state
        self.random_state = 0

        # The number of jobs to run in parallel, -1 by default (all CPUs are used)
        self.n_jobs = 10

        # The name of RandomPARC
        self.name = 'RandomPARC'

        # The dictionary of classifiers
        self.classifiers = ({'RandomForestClassifier': RandomForestClassifier(random_state=self.random_state,
                                                                              n_jobs=self.n_jobs)
            #,
                             # 'LogisticRegression': LogisticRegression(random_state=self.random_state,
                             #                                          n_jobs=self.n_jobs),
                             # 'GaussianNB': GaussianNB()
                             })

        # The dictionary of parameter grids
        self.param_grids = (
        {self.name: [{self.name + '__min_support': [0.2 * i for i in range(1, 6)],
                      self.name + '__min_confidence': [0.2 * i for i in range(1, 6)]}],
            'RandomForestClassifier': [{'RandomForestClassifier__n_estimators': [10 ** i for i in range(1, 4)],
                                     'RandomForestClassifier__criterion': ['gini', 'entropy'],
                                     'RandomForestClassifier__min_samples_split': [max(2, 10 ** i) for i in
                                                                                   range(0, 3)],
                                     'RandomForestClassifier__min_samples_leaf': [10 ** i for i in range(0, 3)],
                                     'RandomForestClassifier__max_features': ['auto', 'log2']}],
         'GaussianNB': [{'GaussianNB__priors': [None]}],
         'LogisticRegression': [{'LogisticRegression__tol': [10 ** i for i in range(-5, -2)],
                                 'LogisticRegression__C': [10 ** i for i in range(-3, 1)],
                                 'LogisticRegression__solver': ['newton-cg',
                                                                'lbfgs',
                                                                'liblinear',
                                                                'sag',
                                                                'saga'],
                                 'LogisticRegression__max_iter': [50 * i for i in range(1, 5)],
                                 'LogisticRegression__multi_class': ['ovr']},
                                {'LogisticRegression__tol': [10 ** i for i in range(-5, -2)],
                                 'LogisticRegression__C': [10 ** i for i in range(-3, 1)],
                                 'LogisticRegression__solver': ['newton-cg',
                                                                'lbfgs',
                                                                'sag',
                                                                'saga'],
                                 'LogisticRegression__max_iter': [50 * i for i in range(1, 5)],
                                 'LogisticRegression__multi_class': ['multinomial']}]
         })

        # The label encoder
        self.encoder = LabelEncoder()

        # The scoring metric for hyperparameter tuning using GridSearchCV
        self.scoring = 'accuracy'

        # The k-fold cross validation
        self.n_splits = 2

        # The pathname of the cv results file directory
        self.cv_results_file_dir = result_dir + 'cv_results_file/'

        # The name of the cv results file
        self.cv_results_file_name = os.path.basename(data_file).split('.')[0]

        # The type of the cv results file
        self.parameter_file_type = '.csv'

        # The pathname of the best_params file directory
        self.best_params_file_dir = result_dir + 'best_params_file/'

        # The name of the best_params file
        self.best_params_file_name = os.path.basename(data_file).split('.')[0]

        # The type of the best_params file
        self.best_params_file_type = '.csv'

        # The pathname of the significant rule file directory
        self.sig_rule_file_dir = result_dir + 'sig_rule_file/'

        # The name of the significant rule file
        self.sig_rule_file_name = os.path.basename(data_file).split('.')[0]

        # The type of the significant rule file
        self.sig_rule_file_type = '.csv'
