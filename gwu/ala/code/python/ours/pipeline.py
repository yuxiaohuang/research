# Please cite the following paper when using the code

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Setting
import ALA

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
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
    ala = ALA.ALA(setting.max_iter, setting.min_samples_bin, setting.C)

    # Train ala
    ala.fit(data.X_train, data.y_train)

    # Test ala
    y_pred = ala.predict(data.X_test)

    # Evaluate ala
    eval(setting, names, data, ala, y_pred)


def eval(setting, names, data, ala, y_pred):
    """
    Evaluate the ALA classifier
    :param setting: the Setting object
    :param names: the Names object
    :param data: the Data object
    :param ala: the ALA classifier
    :param y_pred: the predicted values of the target
    :return:
    """

    setting.set_plt()

    if setting.score_file_dir is not None:
        # Write the score file
        write_score_file(setting, data.y_test, y_pred)

    if setting.mse_fig_dir is not None:
        # Plot the mean square error figure
        plot_mse_fig(setting, ala)

    if setting.prob_dist_fig_dir is not None:
        # Plot the probability distribution figures
        plot_prob_dist_fig(setting, names, data.X, ala)

    if setting.prob_dist_file_dir is not None:
        # Write the probability distribution file
        write_prob_dist_file(setting, names, data.X, ala)


def plot_mse_fig(setting, ala):
    """
    Plot the mean square error figure
    :param setting: the Setting object
    :param ala: the ALA classifier
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.mse_fig_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.plot(range(1, len(ala.mses_) + 1), ala.mses_)
    plt.ylabel('MSE')
    plt.xlabel('Iteration')
    plt.tight_layout()
    mse_fig = setting.mse_fig_dir + setting.mse_fig_name + setting.mse_fig_type
    plt.savefig(mse_fig, dpi=300)


def plot_prob_dist_fig(setting, names, X, ala):
    """
    Plot the probability distribution figures.
    :param setting: the Setting object
    :param names: the Names object
    :param X: the feature vector
    :param ala: the ALA classifier
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.prob_dist_fig_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the dictionary of probability distribution
    ala.get_prob_dist_dict(setting.scaler.transform(X))

    # For each unique value of the target
    for yu in sorted(ala.prob_dist_dict_.keys()):
        # Get the original value of yu
        yu_orig = str(setting.encoder.inverse_transform(yu))

        # For each xj
        for j in sorted(ala.prob_dist_dict_[yu].keys()):
            xijs = sorted(ala.prob_dist_dict_[yu][j].keys())
            pijs = [round(ala.prob_dist_dict_[yu][j][xij], 20) for xij in xijs]
            xijs_orig = [1] if j == 0 else np.unique(sorted(X.iloc[:, j - 1]))
            xijs_orig = [round(xij_orig, 2) for xij_orig in xijs_orig]

            # Get the pandas series
            df = pd.DataFrame(list(zip(xijs_orig, pijs)), columns=['Feature value', 'Probability'])

            xj = 'x0' if j == 0 else names.features[j - 1]

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

            if len(xijs_orig) > 50:
                plt.tick_params(labelbottom='off')

            plt.tight_layout()
            prob_dist_fig = (setting.prob_dist_fig_dir + setting.prob_dist_fig_name + '_' + yu_orig + '_' + xj
                             + setting.prob_dist_fig_type)
            plt.savefig(prob_dist_fig)


def write_prob_dist_file(setting, names, X, ala):
    """
    Write the probability distribution file
    :param setting: the Setting object
    :param names: the Names object
    :param X: the feature vector
    :param ala: the ALA object
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.prob_dist_file_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the dictionary of probability distribution
    ala.get_prob_dist_dict(setting.scaler.transform(X))

    prob_dist_file = setting.prob_dist_file_dir + setting.prob_dist_file_name + setting.prob_dist_file_type

    with open(prob_dist_file, 'w') as f:
        # Write header
        f.write("yu, xj, xij, pij" + '\n')

        # For each unique value of the target
        for yu in sorted(ala.prob_dist_dict_.keys()):
            # Get the original value of yu
            yu_orig = str(setting.encoder.inverse_transform(yu))

            # For each xj
            for j in sorted(ala.prob_dist_dict_[yu].keys()):
                xj = 'x0' if j == 0 else names.features[j - 1]
                xijs = sorted(ala.prob_dist_dict_[yu][j].keys())
                pijs = [ala.prob_dist_dict_[yu][j][xij] for xij in xijs]
                xijs_orig = [1] if j == 0 else np.unique(sorted(X.iloc[:, j - 1]))

                for idx in range(len(pijs)):
                    pij = pijs[idx]
                    xij_orig = xijs_orig[idx]
                    f.write(yu_orig + ', ' + xj + ', ' + str(xij_orig) + ', ' + str(pij) + '\n')


def write_score_file(setting, y_test, y_pred):
    """
    Write the score file
    :param setting: the Setting object
    :param y_test: the testing set
    :param y_pred: the predicted values of the target
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.score_file_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    score_file = setting.score_file_dir + setting.score_file_name + setting.score_file_type

    with open(score_file, 'w') as f:
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average=setting.average[0])

        # Write header
        f.write("precision, recall, fscore using " + setting.average[0] + ':' + '\n')

        # Write the precision, recall, and fscore
        f.write(str(precision) + ', ' + str(recall) + ', ' + str(fscore) + '\n\n')

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average=setting.average[1])

        # Write header
        f.write("precision, recall, fscore using " + setting.average[1] + ':' + '\n')

        # Write the precision, recall, and fscore
        f.write(str(precision) + ', ' + str(recall) + ', ' + str(fscore) + '\n\n')

        accuracy = accuracy_score(y_test, y_pred)

        # Write header
        f.write("accuracy:" + '\n')

        # Write the accuracy
        f.write(str(accuracy) + '\n')


if __name__ == "__main__":
    # Get the pathname of the data directory from command line
    data_dir = sys.argv[1]

    # Get the pathname of the result directory from command line
    result_dir = sys.argv[2]

    # Get the pathname of the DataPreprocessing module directory
    dp_dir = sys.argv[3]

    # Get result from data
    get_result_from_data(data_dir, result_dir, dp_dir)