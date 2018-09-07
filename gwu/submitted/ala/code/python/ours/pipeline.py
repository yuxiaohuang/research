# Please cite the following paper when using the code

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Setting
import ALA

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

    # Match data file with names file
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

    # Train, test, and evaluate the ALA classifier
    train_test_eval(setting, names, data)


def train_test_eval(setting, names, data):
    """
    Train, test, and evaluate the ALA classifier
    :param setting: the Setting object
    :param names: the Names object
    :param data: the Data object
    :return:
    """

    # Declare the ALA classifier
    pipe_ala = Pipeline([('scaler', setting.scaler),
                         ('ala', ALA.ALA(setting.max_iter, setting.min_samples_bin, setting.C))])

    # Get the cross validation scores
    scores = cross_val_score(estimator=pipe_ala,
                             X=data.X,
                             y=data.y,
                             cv=StratifiedKFold(n_splits=setting.n_splits, random_state=setting.random_state),
                             n_jobs=setting.n_jobs)

    # Refit ala on the whole data
    pipe_ala.fit(data.X, data.y)

    # Evaluate ala
    eval(setting, names, data, pipe_ala.named_steps['ala'], scores)


def eval(setting, names, data, ala, scores):
    """
    Evaluate the ALA classifier
    :param setting: the Setting object
    :param names: the Names object
    :param data: the Data object
    :param ala: the ALA classifier
    :param scores: the cross validation scores
    :return:
    """

    setting.set_plt()

    if setting.score_file_dir is not None:
        # Write the score file
        write_score_file(setting, scores)


def write_score_file(setting, scores):
    """
    Write the score file
    :param setting: the Setting object
    :param scores: the cross validation scores
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.score_file_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    score_file = setting.score_file_dir + setting.score_file_name + setting.score_file_type

    with open(score_file, 'w') as f:
        # Write the mean of the cross validation scores
        f.write("The mean of the cross validation scores: " + str(round(np.mean(scores), 2)) + '\n')

        # Write the std of the cross validation scores
        f.write("The std of the cross validation scores: " + str(round(np.std(scores), 2)) + '\n')

        # Write the min of the cross validation scores
        f.write("The min of the cross validation scores: " + str(round(min(scores), 2)) + '\n')

        # Write the max of the cross validation scores
        f.write("The max of the cross validation scores: " + str(round(max(scores), 2)) + '\n')


if __name__ == "__main__":
    # Get the pathname of the data directory from command line
    data_dir = sys.argv[1]

    # Get the pathname of the result directory from command line
    result_dir = sys.argv[2]

    # Get the pathname of the DataPreprocessing module directory
    dp_dir = sys.argv[3]

    # Get result from data
    get_result_from_data(data_dir, result_dir, dp_dir)