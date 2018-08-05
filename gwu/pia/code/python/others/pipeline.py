# Please cite the following paper when using the code

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Setting

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed


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
        pipe_clf = Pipeline([('clf',  classifier(random_state=setting.random_state, n_jobs=setting.n_jobs))])
    elif clf_name == 'AdaBoostClassifier':
        pipe_clf = Pipeline([('clf',  classifier(random_state=setting.random_state))])
    elif clf_name == 'MLPClassifier':
        pipe_clf = Pipeline([('clf',  classifier(random_state=setting.random_state))])
    elif clf_name == 'KNeighborsClassifier':
        pipe_clf = Pipeline([('clf',  classifier(n_jobs=setting.n_jobs))])
    elif clf_name == 'GaussianNB':
        pipe_clf = Pipeline([('clf',  classifier())])
    elif clf_name == 'DecisionTreeClassifier':
        pipe_clf = Pipeline([('clf',  classifier(random_state=setting.random_state))])
    elif clf_name == 'LogisticRegression':
        pipe_clf = Pipeline([('clf',  classifier(random_state=setting.random_state, n_jobs=setting.n_jobs))])
    elif clf_name == 'GaussianProcessClassifier':
        pipe_clf = Pipeline([('clf',  classifier(random_state=setting.random_state, n_jobs=setting.n_jobs))])
    elif clf_name == 'SVC':
        pipe_clf = Pipeline([('clf',  classifier(random_state=setting.random_state))])

    # Get the cross validation scores
    scores = cross_val_score(estimator=pipe_clf,
                             X=data.X,
                             y=data.y,
                             cv=StratifiedKFold(n_splits=min(setting.n_splits, min(np.bincount(data.y))), random_state=setting.random_state),
                             n_jobs=setting.n_jobs)

    # Refit clf on the whole data
    pipe_clf.fit(data.X, data.y)

    # Evaluate clf
    eval(setting, names, pipe_clf.named_steps['clf'], scores, clf_name)


def eval(setting, names, clf, scores, clf_name):
    """
    Evaluate the classifier
    :param setting: the Setting object
    :param names: the Names object
    :param clf: the classifier
    :param scores: the cross validation scores
    :param clf_name: the name of the classifier
    :return:
    """

    setting.set_plt()

    if setting.score_file_dir is not None:
        # Write the score file
        write_score_file(setting, scores, clf_name)

    if (setting.feature_importance_fig_dir is not None
        and (isinstance(clf, setting.classifiers['RandomForestClassifier']) is True)):
        # Plot the feature importance figures
        plot_feature_importance_fig(setting, names, clf, clf_name)


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


def plot_feature_importance_fig(setting, names, clf, clf_name):
    """
    Plot the feature importance figures

    Parameters
    ----------
    setting : the Setting object
    names : the Names object
    clf : the classifier
    clf_name: the name of the classifier
    """

    # Get the directory of the feature importance file
    feature_importance_fig_dir = setting.feature_importance_fig_dir + clf_name + '/'
    # Get the pathname of the feature importance figure
    feature_importance_fig = (
    feature_importance_fig_dir + setting.feature_importance_fig_name + setting.feature_importance_fig_type)

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

    # Make the bar plot from f_importances_top_k
    f_importances.plot(kind='bar', figsize=(20, 10), rot=45, fontsize=30)

    plt.xlabel('Feature', fontsize=30)
    plt.ylabel('Importance', fontsize=30)
    plt.tight_layout()
    plt.savefig(feature_importance_fig)


if __name__ == "__main__":
    # Get the pathname of the data directory from command line
    data_dir = sys.argv[1]

    # Get the pathname of the result directory from command line
    result_dir = sys.argv[2]

    # Get the pathname of the DataPreprocessing module directory
    dp_dir = sys.argv[3]

    # Get result from data
    get_result_from_data(data_dir, result_dir, dp_dir)