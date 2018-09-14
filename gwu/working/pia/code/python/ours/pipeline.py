# Please cite the following paper when using the code

import sys
import os
import pandas as pd
import numpy as np
import Setting
import PIA
import copy

from joblib import Parallel, delayed


def get_result_from_data(data_dir, result_dir, dp_dir):
    """
    Get result from data

    Parameters
    ----------
    data_dir: the pathname of the data directory
    result_dir: the pathname of the result directory
    dp_dir: the pathname of the DataPreprocessing module directory
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

    Parameters
    ----------
    dp: the DataPreprocessing module
    data_files: the pathname of the data files
    names_file: the pathname of the names file
    result_dir: the pathname of the result directory
    """

    # Data preprocessing: get the Setting, Names, and Data object
    setting, names, data = dp.get_setting_names_data(data_files, names_file, result_dir, Setting)

    # The dictionary of the PIA object
    pias = {}

    # For each split
    for i in range(setting.n_splits):
        # Get the train and test indices
        train_index, test_index = data.train_test_indices[i]
        # Declare the PIA object
        pia = PIA.PIA(setting.min_samples_principle if data.X.shape[0] > setting.min_samples_principle else 1,
                      setting.p_val_principle,
                      setting.random_state)
        # Fit pia on the training set
        pia.fit(data.X[train_index], data.y[train_index])
        # Update pias
        pias[i] = copy.deepcopy(pia)

        if setting.interaction_file_dir is not None:
            # Write the interaction file
            write_interaction_file(setting, names, pias[i], str(i))

    # Train, test, and evaluate the classifier
    # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
    Parallel(n_jobs=setting.n_jobs)(delayed(train_test_eval)(setting, data, clf_name, pias)
                                    for clf_name in setting.classifiers.keys())

    # Refit pia on the whole data
    pia.fit(data.X, data.y)

    if setting.interaction_file_dir is not None:
        # Write the interaction file
        write_interaction_file(setting, names, pia, 'all')


def write_interaction_file(setting, names, pia, name):
    """
    Write the interaction file

    Parameters
    ----------
    setting: the Setting object
    names : the Names object
    pia : the PIA object
    name : the number of split or 'all'
    """

    # Get the pathname of the interaction file
    interaction_file = setting.interaction_file_dir + name + '/' + setting.interaction_file_name + setting.interaction_file_type

    # Make directory
    directory = os.path.dirname(interaction_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(interaction_file, 'w') as f:
        # Write header
        f.write("class, interaction" + '\n')

        # For each class of the target
        for class_ in sorted(pia.pis.keys()):
            for I in pia.pis[class_]:
                f.write(str(setting.encoder.inverse_transform([class_])[0]) + ',' + ' & '.join(
                    [names.features[c] for c in I]) + '\n')


def train_test_eval(setting, data, clf_name, pias):
    """
    Train, test, and evaluate the classifier

    Parameters
    ----------
    setting : the Setting object
    names : the Names object
    data : the Data object
    clf_name : the name of the classifier
    pias : the dictionary of the PIA object
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

        # Get X_train_transformed and y_train
        X_train_transformed = pias[i].transform(data.X[train_index])
        y_train = data.y[train_index]

        # Fit clf
        clf.fit(X_train_transformed, y_train)

        # Get X_test_transformed and y_test
        X_test_transformed = pias[i].transform(data.X[test_index])
        y_test = data.y[test_index]

        # Update scores
        scores[i] = clf.score(X_test_transformed, y_test)

    # Evaluate pia
    eval(setting, scores, clf_name)


def eval(setting, scores, clf_name):
    """
    Evaluate the classifier

    Parameters
    ----------
    setting: the Setting object
    scores: the cross validation scores
    clf_name: the name of the classifier
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
