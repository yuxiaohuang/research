# Please cite the following paper when using the code


import sys
import os
import csv
import pandas as pd
import numpy as np
import inspect
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
SIZE = 30
plt.rc('font', size=SIZE)          # text size
plt.rc('axes', titlesize=SIZE)     # axes title size
plt.rc('axes', labelsize=SIZE)     # x and y labels size
plt.rc('xtick', labelsize=SIZE)    # xtick size
plt.rc('ytick', labelsize=SIZE)    # ytick size
plt.rc('legend', fontsize=SIZE)    # legend size
plt.rc('figure', titlesize=SIZE)   # figure title size
plt.switch_backend('agg')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support

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

    def __init__(self, setting_file):
        # The pathname of (or link to) the data file, must be specified
        self.data_file = None

        # The header, None by default
        self.header = None

        # The place holder for missing values, '?' by default
        self.place_holder_for_missing_vals = '?'

        # The (name of the) columns, must be specified
        self.columns = None

        # The (name of the) target, must be specified
        self.target = None

        # The (name of the) features, None by default
        # This is not a parameter in the setting file,
        # since it can either be inferred based on self.columns and self.target
        self.features = None

        # The (name of the) features that should be excluded, empty by default
        self.exclude_features = []

        # The (name of the) categorical features, empty by default
        self.categorical_features = []

        # The label encoder
        # This is not a parameter in the setting file
        self.encoder = LabelEncoder()

        # The percentage of the testing set, 0.3 by default
        self.test_size = 0.3

        # The scaler, StandardScaler by default
        self.scaler = StandardScaler()

        # The random state, zero by default
        self.random_state = 0

        # The maximum number of iterations, 100 by default
        self.max_iter = 100

        # The value of C, 1 by default
        self.C = 1

        # The pathname of the probability distribution figure directory, None by default
        self.prob_dist_fig_dir = None

        # The name of the probability distribution figure, the name of the setting file by default
        self.prob_dist_fig_name = os.path.basename(setting_file).split('.')[0]

        # The type of the probability distribution figure, '.pdf' by default
        self.prob_dist_fig_type = '.pdf'

        # The pathname of the probability distribution file directory, None by default
        self.prob_dist_file_dir = None

        # The name of the probability distribution file, the name of the setting file by default
        self.prob_dist_file_name = os.path.basename(setting_file).split('.')[0]

        # The type of the probability distribution file, '.csv' by default
        self.prob_dist_file_type = '.csv'

        # The pathname of the score file directory, None by default
        self.score_file_dir = None

        # The name of the score file, the name of the setting file by default
        self.score_file_name = os.path.basename(setting_file).split('.')[0]

        # The type of the score file, '.txt' by default
        self.score_file_type = '.txt'

        # The average for precision_recall_fscore_support, 'micro' by default
        self.average = 'micro'

        # The dictionary of classifiers, by default:
        # RandomForestClassifier
        # AdaBoostClassifier
        # MLPClassifier
        # KNeighborsClassifier
        # GaussianNB
        # DecisionTreeClassifier
        # LogisticRegression(multi_class='ovr')
        # LogisticRegression(multi_class='multinomial', solver='lbfgs')
        # LogisticRegression(multi_class='multinomial', solver='sag')
        # LogisticRegression(multi_class='multinomial', solver='newton-cg')
        # GaussianProcessClassifier
        # SVC
        self.classifiers = {}

        # The number of jobs to run in parallel, -1 by default (all CPUs are used)
        self.n_jobs = -1

        # The parameter names
        self.para_names = ['data_file',
                           'header',
                           'place_holder_for_missing_vals',
                           'columns',
                           'target',
                           'exclude_features',
                           'categorical_features',
                           'test_size',
                           'scaler',
                           'random_state',
                           'max_iter',
                           'C',
                           'prob_dist_fig_dir',
                           'prob_dist_fig_name',
                           'prob_dist_fig_type',
                           'prob_dist_file_dir',
                           'prob_dist_file_name',
                           'prob_dist_file_type',
                           'score_file_dir',
                           'score_file_name',
                           'score_file_type',
                           'average',
                           'classifiers',
                           'n_jobs']


class Data:
    """The Data class"""

    def __init__(self,
                 X,
                 X_train,
                 X_test,
                 y,
                 y_train,
                 y_test):

        # The feature data
        self.X = X

        # The feature data for training
        self.X_train = X_train

        # The feature data for testing
        self.X_test = X_test

        # The target data
        self.y = y

        # The target data for training
        self.y_train = y_train

        # The target data for testing
        self.y_test = y_test


def pipe_line(setting_file):
    """
    Read the setting file, and return the Setting and Data object
    :param setting_file: the setting file
    :return: setting: the Setting object
    :return: data: the Data object
    """

    with open(setting_file, 'r') as f:
        # Read the setting file
        spamreader = list(csv.reader(f, delimiter='='))

    # Declare the Setting object
    setting = Setting(setting_file)

    # For each parameter
    for para_name in setting.para_names:
        # For each row in the setting file
        for i in range(len(spamreader)):
            # If spamreader[i] is not empty
            if spamreader[i] is not None and len(spamreader[i]) > 0:
                # Get the string on the left-hand side of '='
                str_left = spamreader[i][0]

                # Ignore comments
                if str_left.startswith('#'):
                    continue

                if para_name in str_left:
                    # If there are values for the parameter
                    if len(spamreader[i]) > 1:
                        # Get the string on the right-hand side of '='
                        str_right = spamreader[i][1]

                        # Split the string into strings
                        strs = str_right.split(",")

                        # Get the (non-empty) values
                        vals = [str.strip() for str in strs if len(str.strip()) > 0]

                        # If vals is not empty
                        if len(vals) > 0:
                            vals = [float(val) if val.isdigit() is True else val for val in vals]
                            get_para_vals(setting, para_name, vals)

    # Get setting.features
    setting.features = [feature for feature in setting.columns if feature != setting.target]

    # Get setting.classifiers
    if len(setting.classifiers.keys()) == 0:
        setting.classifiers = ({'RandomForestClassifier': RandomForestClassifier,
                                'AdaBoostClassifier': AdaBoostClassifier,
                                'MLPClassifier': MLPClassifier,
                                'KNeighborsClassifier': KNeighborsClassifier,
                                'GaussianNB': GaussianNB,
                                'DecisionTreeClassifier': DecisionTreeClassifier,
                                'LogisticRegression_ovr': LogisticRegression,
                                'LogisticRegression_multinomial_lbfgs': LogisticRegression,
                                'LogisticRegression_multinomial_sag': LogisticRegression,
                                'LogisticRegression_multinomial_newton-cg': LogisticRegression,
                                'GaussianProcessClassifier': GaussianProcessClassifier,
                                'SVC': SVC})

    # Load data
    df = pd.read_csv(setting.data_file, header=setting.header)

    # Replace missing_representation with NaN
    df = df.replace(setting.place_holder_for_missing_vals, np.NaN)
    # Remove rows that contain missing values
    df = df.dropna(axis=0)

    # Get df.columns
    df.columns = list(setting.columns)

    if len(setting.exclude_features) > 0:
        # Remove features that should be excluded
        df = df.drop([setting.exclude_features], axis=1)

    # Get the feature vector
    X = df.drop([setting.target], axis=1)

    # One-hot encoding on categorical features
    if len(setting.categorical_features) > 0:
        X = pd.get_dummies(X, columns=setting.categorical_features).values

    # Get the target vector
    y = df[setting.target]

    # Encode the target
    y = setting.encoder.fit_transform(y)

    # Randomly choose setting.test_size% of the data for testing
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=setting.test_size,
                                                        random_state=setting.random_state,
                                                        stratify=y)

    # Standardize the features
    X_train = setting.scaler.fit_transform(X_train.astype(float))
    X_test = setting.scaler.transform(X_test.astype(float))

    # Declare the data object
    data = Data(X, X_train, X_test, y, y_train, y_test)

    return [setting, data]


def get_para_vals(setting, para_name, vals):
    """
    Get parameter values
    :param setting: the Setting object
    :param para_name: the parameter name
    :param vals: the values
    :return:
    """

    vals = vals[0] if len(vals) == 1 else vals

    if para_name == 'data_file':
        setting.data_file = vals
    elif para_name == 'header':
        setting.header = int(vals)
    elif para_name == 'place_holder_for_missing_vals':
        setting.place_holder_for_missing_vals = vals
    elif para_name == 'columns':
        setting.columns = vals
    elif para_name == 'target':
        setting.target = vals
    elif para_name == 'exclude_features':
        setting.exclude_features = vals
    elif para_name == 'categorical_features':
        setting.categorical_features = vals
    elif para_name == 'test_size':
        setting.test_size = flot(vals)
    elif para_name == 'scaler':
        if 'MinMaxScaler' in vals:
            setting.encoder = MinMaxScaler
    elif para_name == 'random_state':
        setting.random_state = int(vals)
    elif para_name == 'max_iter':
        setting.max_iter = int(vals)
    elif para_name == 'C':
        setting.C = int(vals)
    elif para_name == 'prob_dist_fig_dir':
        setting.prob_dist_fig_dir = vals
    elif para_name == 'prob_dist_fig_name':
        setting.prob_dist_fig_name = vals
    elif para_name == 'prob_dist_fig_type':
        setting.prob_dist_fig_type = vals
    elif para_name == 'prob_dist_file_dir':
        setting.prob_dist_file_dir = vals
    elif para_name == 'prob_dist_file_name':
        setting.prob_dist_file_name = vals
    elif para_name == 'prob_dist_file_type':
        setting.prob_dist_file_type = vals
    elif para_name == 'score_file_dir':
        setting.score_file_dir = vals
    elif para_name == 'score_file_name':
        setting.score_file_name = vals
    elif para_name == 'score_file_type':
        setting.score_file_type = vals
    elif para_name == 'average':
        setting.average = vals
    elif para_name == 'classifiers':
        if 'RandomForestClassifier' in vals:
            setting.classifiers['RandomForestClassifier'] = RandomForestClassifier
        if 'AdaBoostClassifier' in vals:
            setting.classifiers['AdaBoostClassifier'] = AdaBoostClassifier
        if 'MLPClassifier' in vals:
            setting.classifiers['MLPClassifier'] = MLPClassifier
        if 'KNeighborsClassifier' in vals:
            setting.classifiers['KNeighborsClassifier'] = KNeighborsClassifier
        if 'GaussianNB' in vals:
            setting.classifiers['GaussianNB'] = GaussianNB
        if 'DecisionTreeClassifier' in vals:
            setting.classifiers['DecisionTreeClassifier'] = DecisionTreeClassifier
        if 'LogisticRegression' in vals:
            setting.classifiers['LogisticRegression_ovr'] = LogisticRegression
            setting.classifiers['LogisticRegression_multinomial_lbfgs'] = LogisticRegression
            setting.classifiers['LogisticRegression_multinomial_sag'] = LogisticRegression
            setting.classifiers['LogisticRegression_multinomial_newton-cg'] = LogisticRegression
        if 'GaussianProcessClassifier' in vals:
            setting.classifiers['GaussianProcessClassifier'] = GaussianProcessClassifier
        if 'SVC' in vals:
            setting.classifiers['SVC'] = SVC
    elif para_name == 'n_jobs':
        setting.n_jobs = int(vals)


def train_test_evaluate(setting, data, clf_name):
    """
    Train, test, and evaluate the classifier
    :param setting: the Setting object
    :param data: the Data object
    :param clf_name: the name of the classifier
    :return:
    """

    classifier = setting.classifiers[clf_name]
    if clf_name in (['KNeighborsClassifier', 'GaussianNB']):
        clf = classifier()
    elif clf_name == 'LogisticRegression_multinomial_lbfgs':
        clf = classifier(multi_class='multinomial', solver='lbfgs', random_state=setting.random_state)
    elif clf_name == 'LogisticRegression_multinomial_sag':
        clf = classifier(multi_class='multinomial', solver='sag', random_state=setting.random_state)
    elif clf_name == 'LogisticRegression_multinomial_newton-cg':
        clf = classifier(multi_class='multinomial', solver='newton-cg', random_state=setting.random_state)
    else:
        clf = classifier(random_state=setting.random_state)

    # Train clf
    clf.fit(data.X_train, data.y_train)

    # Test clf
    y_pred = clf.predict(data.X_test)

    # Evaluate clf
    eval(setting, data, clf, y_pred, clf_name)


def eval(setting, data, clf, y_pred, clf_name):
    """
    Evaluate the classifier
    :param setting: the Setting object
    :param data: the Data object
    :param clf: the classifier
    :param y_pred: the predicted values of the target
    :param clf_name: the name of the classifier
    :return:
    """

    if (setting.prob_dist_fig_dir is not None
        and (isinstance(clf, setting.classifiers['LogisticRegression_ovr']) is True
             or isinstance(clf, setting.classifiers['LogisticRegression_multinomial_lbfgs']) is True
             or isinstance(clf, setting.classifiers['LogisticRegression_multinomial_sag']) is True
             or isinstance(clf, setting.classifiers['LogisticRegression_multinomial_newton-cg']) is True)):
        # Plot the probability distribution figures
        plot_prob_dist_fig(setting, data.X, clf, clf_name)

    if (setting.prob_dist_file_dir is not None
        and (isinstance(clf, setting.classifiers['LogisticRegression_ovr']) is True
             or isinstance(clf, setting.classifiers['LogisticRegression_multinomial_lbfgs']) is True
             or isinstance(clf, setting.classifiers['LogisticRegression_multinomial_sag']) is True
             or isinstance(clf, setting.classifiers['LogisticRegression_multinomial_newton-cg']) is True)):
        # Write the probability distribution file
        write_prob_dist_file(setting, data.X, clf, clf_name)

    if setting.score_file_dir is not None:
        # Write the score file
        write_score_file(setting, data.y_test, y_pred, clf_name)


def plot_prob_dist_fig(setting, X, clf, clf_name):
    """
    Plot the probability distribution figures.
    :param setting: the Setting object
    :param X: the feature vector
    :param clf: the classifier
    :param clf_name: the name of the classifier
    :return:
    """

    # Get the directory of the probability distribution figure
    prob_dist_fig_dir = setting.prob_dist_fig_dir + clf_name + '/'

    # Make directory
    directory = os.path.dirname(prob_dist_fig_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the dictionary of probability distribution
    prob_dist_dict = get_prob_dist_dict(setting.scaler.transform(X), clf)

    # For each unique value of the target
    for yu in sorted(prob_dist_dict.keys()):
        # Get the original value of yu
        yu_orig = str(setting.encoder.inverse_transform(yu))

        # For each xj
        for j in sorted(prob_dist_dict[yu].keys()):
            xijs = sorted(prob_dist_dict[yu][j].keys())
            pijs = [prob_dist_dict[yu][j][xij] for xij in xijs]
            xijs_orig = [1] if j == 0 else np.unique(sorted(X.iloc[:, j - 1]))

            # Get the pandas series
            df = pd.DataFrame(list(zip(xijs_orig, pijs)), columns=['Feature value', 'Probability'])

            xj = 'x0' if j == 0 else setting.features[j - 1]

            # Plot the histogram of the series
            df.plot(x='Feature value',
                    y='Probability',
                    kind='bar',
                    figsize=(20, 10),
                    title=('P(' + yu_orig + ' | ' + xj + ')'),
                    legend=False,
                    color='b')

            # Set the x-axis label
            plt.xlabel("Feature value")
            # Set the y-axis label
            plt.ylabel("Probability")

            plt.tight_layout()
            # Get the pathname of the probability distribution figure
            prob_dist_fig = (prob_dist_fig_dir + setting.prob_dist_fig_name + '_' + yu_orig + '_' + xj
                             + setting.prob_dist_fig_type)
            plt.savefig(prob_dist_fig)


def get_prob_dist_dict(X, clf):
    """
    Get the dictionary of probability distribution
    :param X: the feature vector
    :param clf: the classifier
    :return: the dictionary of probability distribution
    """

    # Initialization
    prob_dist_dict = {}

    # For each xj
    for j in range(X.shape[1] + 1):
        # Initialize X_sparse
        X_sparse = np.zeros((X.shape[0], X.shape[1]))

        if j > 0:
            # Update xj in X_sparse
            X_sparse[:, j - 1] = X[:, j - 1]

        # Get the unique value and the corresponding index in xj
        if j == 0:
            xus, idxs = np.unique([1], return_index=True)
        else:
            xus, idxs = np.unique(X_sparse[:, j - 1], return_index=True)

        # For each unique index
        for i in idxs:
            # Get xij
            xij = 1 if j == 0 else X_sparse[i, j - 1]
            # Get the probability of each label
            probs = clf.predict_proba(X_sparse[i, :].reshape(1, -1)).ravel()

            # For each unique value of the target
            for yu in range(len(probs)):
                # Get the probability
                prob = probs[yu]

                # Initialization
                if yu not in prob_dist_dict:
                    prob_dist_dict[yu] = {}
                if j not in prob_dist_dict[yu]:
                    prob_dist_dict[yu][j] = {}

                # Update prob_dist_dict
                prob_dist_dict[yu][j][xij] = prob

    return prob_dist_dict


def write_prob_dist_file(setting, X, clf, clf_name):
    """
    Write the probability distribution file
    :param setting: the Setting object
    :param X: the feature vector
    :param clf: the classifier
    :param clf_name: the name of the classifier
    :return:
    """

    # Get the directory of the probability distribution file
    prob_dist_file_dir = setting.prob_dist_file_dir + clf_name + '/'
    # Get the pathname of the probability distribution file
    prob_dist_file = prob_dist_file_dir + setting.prob_dist_file_name + setting.prob_dist_file_type

    # Make directory
    directory = os.path.dirname(prob_dist_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the dictionary of probability distribution
    prob_dist_dict = get_prob_dist_dict(setting.scaler.transform(X), clf)

    with open(prob_dist_file, 'w') as f:
        # Write header
        f.write("yu, xj, xij, pij" + '\n')

        # For each unique value of the target
        for yu in sorted(prob_dist_dict.keys()):
            # Get the original value of yu
            yu_orig = str(setting.encoder.inverse_transform(yu))

            # For each xj
            for j in sorted(prob_dist_dict[yu].keys()):
                xj = 'x0' if j == 0 else setting.features[j - 1]
                xijs = sorted(prob_dist_dict[yu][j].keys())
                pijs = [prob_dist_dict[yu][j][xij] for xij in xijs]
                xijs_orig = [1] if j == 0 else np.unique(sorted(X.iloc[:, j - 1]))

                for idx in range(len(pijs)):
                    pij = pijs[idx]
                    xij_orig = xijs_orig[idx]
                    f.write(yu_orig + ', ' + xj + ', ' + str(xij_orig) + ', ' + str(pij) + '\n')


def write_score_file(setting, y_test, y_pred, clf_name):
    """
    Write the score file
    :param setting: the Setting object
    :param y_test: the testing set
    :param y_pred: the predicted values of the target
    :param clf_name: the name of the classifier
    :return:
    """

    # Get the directory of the score file
    score_file_dir = setting.score_file_dir + clf_name + '/'
    # Get the pathname of the score file
    score_file = score_file_dir + setting.score_file_name + setting.score_file_type

    # Make directory
    directory = os.path.dirname(score_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(score_file, 'w') as f:
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average=setting.average)

        # Write header
        f.write("precision, recall, fscore" + '\n')

        # Write the precision, recall, and fscore
        f.write(str(precision) + ', ' + str(recall) + ', ' + str(fscore) + '\n')


if __name__ == "__main__":
    # Get the pathname of the setting file from command line
    setting_file = sys.argv[1]

    # Read the setting file, and return the Setting and Data object
    setting, data = pipe_line(setting_file)

    # Train, test, and evaluate
    # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
    Parallel(n_jobs=setting.n_jobs)(delayed(train_test_evaluate)(setting, data, clf_name)
                                    for clf_name in setting.classifiers.keys())