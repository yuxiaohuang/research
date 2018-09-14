# Please cite the following paper when using the code

import sys
import os
import pandas as pd
import numpy as np
import Setting
import matplotlib.pyplot as plt
import copy

from joblib import Parallel, delayed
from sklearn import tree
from subprocess import check_call


def get_result_from_data(data_dir, result_dir, dp_dir):
    """
    Get result from data
    :param data_dir: the pathname of the data directory
    :param result_dir: the pathname of the result directory
    :param dp_dir: the pathname of the DataPreprocessing module directory
    :return:
    """

    # Add code_dir folder
    sys.path.append(dp_dir)

    # Import the DataPreprocessing module
    import DataPreprocessing
    # Get the DataPreprocessing object
    dp = DataPreprocessing.DataPreprocessing(data_dir)

    # Match data file withm names file
    data_names = dp.match_data_names()

    # The parallel pipelines for data preprocessing, train, test, and evaluate the ALA classifier
    # n_jobs = -1 indicates (all CPUs are used)
    # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
    Parallel(n_jobs=-1)(delayed(pipeline)(dp, data_file, names_file, result_dir)
                        for data_file, names_file in data_names)


def pipeline(dp, data_files, names_file, result_dir):
    """
    The pipeline for data preprocessing, train, test, and evaluate the ALA classifier
    :param dp: the DataPreprocessing module
    :param data_files: the pathname of the data files
    :param names_file: the pathname of the names file
    :param result_dir: the pathname of the result directory
    :return:
    """

    # Data preprocessing: get the Setting, Names, and Data object
    setting, names, data = dp.get_setting_names_data(data_files, names_file, result_dir, Setting)

    # Train, test, and evaluate the classifier
    # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
    Parallel(n_jobs=setting.n_jobs)(delayed(train_test_eval)(setting, names, data, clf_name)
                                    for clf_name in setting.classifiers.keys())


def train_test_eval(setting, names, data, clf_name):
    """
    Train, test, and evaluate the classifier
    :param setting: the Setting object
    :param names: the Names object
    :param data: the Data object
    :param clf_name: the name of the classifier
    :return:
    """

    classifier = setting.classifiers[clf_name]

    if clf_name == 'RandomForestClassifier':
        clf = classifier(random_state=setting.random_state, n_jobs=setting.n_jobs)
    elif clf_name == 'AdaBoostClassifier':
        clf = classifier(random_state=setting.random_state)
    elif clf_name == 'MLPClassifier':
        clf = classifier(random_state=setting.random_state)
    elif clf_name == 'KNeighborsClassifier':
        clf = classifier(n_jobs=setting.n_jobs)
    elif clf_name == 'GaussianNB':
        clf = classifier()
    elif clf_name == 'DecisionTreeClassifier':
        clf = classifier(random_state=setting.random_state)
    elif clf_name == 'LogisticRegression':
        clf = classifier(random_state=setting.random_state, n_jobs=setting.n_jobs)
    elif clf_name == 'GaussianProcessClassifier':
        clf = classifier(random_state=setting.random_state, n_jobs=setting.n_jobs)
    elif clf_name == 'SVC':
        clf = classifier(random_state=setting.random_state)

    # The cross validation scores
    scores = np.zeros(setting.n_splits)

    # For each split
    for i in range(setting.n_splits):
        # Get the train and test indices
        train_index, test_index = data.train_test_indices[i]

        # Fit clf
        clf.fit(data.X[train_index], data.y[train_index])

        # Update scores
        scores[i] = clf.score(data.X[test_index], data.y[test_index])

        if (setting.feature_importance_fig_dir is not None
            and (isinstance(clf, setting.classifiers['RandomForestClassifier']) is True)):
            # Plot the feature importance figures
            plot_feature_importance_fig(setting, names, clf, clf_name, str(i))

        if (setting.decision_tree_fig_dir is not None
            and (isinstance(clf, setting.classifiers['DecisionTreeClassifier']) is True)):
            # Plot the decision tree figures
            plot_decision_tree_fig(setting, names, clf, clf_name, str(i))

    # Evaluate clf
    eval(setting, scores, clf_name)

    if (setting.feature_importance_fig_dir is not None
        and (isinstance(clf, setting.classifiers['RandomForestClassifier']) is True)):
        # Refit clf on the whole data
        clf.fit(data.X, data.y)

        # Plot the feature importance figures
        plot_feature_importance_fig(setting, names, clf, clf_name, 'all')

    if (setting.decision_tree_fig_dir is not None
        and (isinstance(clf, setting.classifiers['DecisionTreeClassifier']) is True)):
        # Refit clf on the whole data
        clf.fit(data.X, data.y)

        # Plot the decision tree figures
        plot_decision_tree_fig(setting, names, clf, clf_name, 'all')


def plot_feature_importance_fig(setting, names, clf, clf_name, name):
    """
    Plot the feature importance figures

    Parameters
    ----------
    setting : the Setting object
    names : the Names object
    clf : the classifier
    clf_name: the name of the classifier
    name : the number of split or 'all'
    """

    setting.set_plt()

    # Get the directory of the feature importance figure
    feature_importance_fig_dir = setting.feature_importance_fig_dir + clf_name + '/'
    # Get the pathname of the feature importance figure
    feature_importance_fig = (
    feature_importance_fig_dir + name + '/' + setting.feature_importance_fig_name + setting.feature_importance_fig_type)

    # Make directory
    directory = os.path.dirname(feature_importance_fig)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the feature importances
    importances = clf.feature_importances_

    # Convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
    f_importances = pd.Series(importances, names.features)

    # Sort the array in descending order of the importances
    f_importances.sort_values(ascending=False, inplace=True)

    # Make the bar plot from f_importances top 20
    f_importances[:20 if f_importances.size > 20 else f_importances.size].plot(kind='bar', figsize=(20, 10), rot=90, fontsize=30)

    plt.xlabel('Feature', fontsize=30)
    plt.ylabel('Importance', fontsize=30)
    plt.tight_layout()
    plt.savefig(feature_importance_fig)


def plot_decision_tree_fig(setting, names, clf, clf_name, name):
    """
    Plot the decision tree figures

    Parameters
    ----------
    setting : the Setting object
    names : the Names object
    clf : the classifier
    clf_name: the name of the classifier
    name : the number of split or 'all'
    """

    setting.set_plt()

    # Get the directory of the decision tree figure
    decision_tree_fig_dir = setting.decision_tree_fig_dir + clf_name + '/'
    # Get the pathname of the decision tree figure
    decision_tree_fig = (decision_tree_fig_dir + name + '/' + setting.decision_tree_fig_name + '.dot')

    # Make directory
    directory = os.path.dirname(decision_tree_fig)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the dot file
    tree.export_graphviz(clf, out_file=decision_tree_fig, feature_names=names.features)

    # Get the pdf file
    check_call(['dot', '-Tpdf', decision_tree_fig, '-o', decision_tree_fig.replace('.dot', setting.decision_tree_fig_type)])


def eval(setting, scores, clf_name):
    """
    Evaluate the classifier
    :param setting: the Setting object
    :param scores: the cross validation scores
    :param clf_name: the name of the classifier
    :return:
    """

    if setting.score_file_dir is not None:
        # Write the score file
        write_score_file(setting, scores, clf_name)


def write_score_file(setting, scores, clf_name):
    """
    Write the score file
    :param setting: the Setting object
    :param scores: the cross validation scores
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
        # Write the mean of the cross validation scores
        f.write("The mean of the cross validation scores: " + str(round(np.mean(scores), 2)) + '\n')

        # Write the std of the cross validation scores
        f.write("The std of the cross validation scores: " + str(round(np.std(scores), 2)) + '\n')

        # Write the min of the cross validation scores
        f.write("The min of the cross validation scores: " + str(round(min(scores), 2)) + '\n')

        # Write the max of the cross validation scores
        f.write("The max of the cross validation scores: " + str(round(max(scores), 2)) + '\n')

        # Write the list of the cross validation scores
        f.write("The list of the cross validation scores: " + '\n')
        for i in range(len(scores)):
            f.write(str(i) + ', ' + str(round(scores[i], 2)) + '\n')


if __name__ == "__main__":
    # Get the pathname of the data directory from command line
    data_dir = sys.argv[1]

    # Get the pathname of the result directory from command line
    result_dir = sys.argv[2]

    # Get the pathname of the DataPreprocessing module directory
    dp_dir = sys.argv[3]

    # Get result from data
    get_result_from_data(data_dir, result_dir, dp_dir)