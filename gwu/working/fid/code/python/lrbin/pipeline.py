# Please cite the following paper when using the code

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Setting
import LRBin

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")


def pipeline_all_datasets():
    """
    The pipeline for all data sets
    :return:
    """

    # Add code_dir folder
    sys.path.append(dp_dir)

    # Import DataPreprocessing module
    import DataPreprocessing
    dp = DataPreprocessing.DataPreprocessing(data_dir)

    # Match data files with names file
    data_names = dp.match_data_names()

    # The pipeline for each data set (in parallel)
    # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
    Parallel(n_jobs=-1)(delayed(pipeline_one_dataset)(dp, data_files, names_file)
                        for data_files, names_file in data_names)


def pipeline_one_dataset(dp, data_files, names_file):
    """
    The pipeline for one data set
    :param dp: the DataPreprocessing module
    :param data_files: the pathname of the data files
    :param names_file: the pathname of the names file
    :return:
    """

    # Data preprocessing: get the Setting, Names, and Data object
    setting, names, data = dp.get_setting_names_data(data_files, names_file, result_dir, Setting)

    # Get the sklearn pipeline
    pipe_lrbin = Pipeline([('scaler', setting.scaler),
                           ('lrbin', LRBin.LRBin(setting.max_iter,
                                                 setting.bin_num_percent,
                                                 setting.min_bin_num,
                                                 setting.max_bin_num,
                                                 setting.eta,
                                                 setting.random_state,
                                                 setting.n_jobs))])

    # Hyperparameter tuning using GridSearchCV
    gs = GridSearchCV(estimator=pipe_lrbin,
                      param_grid=[{'lrbin__bin_num_percent': setting.bin_num_percents,
                                   'lrbin__eta': setting.etas}],
                      scoring=setting.scoring,
                      n_jobs=setting.n_jobs,
                      cv=StratifiedKFold(n_splits=setting.n_splits,
                                         random_state=setting.random_state))
    gs.fit(data.X, data.y)

    # Get the results
    get_results(setting, names, data, gs)


def get_results(setting, names, data, gs):
    """
    Get the results
    :param setting: the Setting object
    :param names: the Names object
    :param data: the Data object
    :param gs: the GridSearchCV object
    :return:
    """

    if setting.prob_dists_fig_dir is not None:
        # Plot the probability distribution figures
        plot_prob_dists_fig(setting, names, data.X, gs.best_estimator_.named_steps['lrbin'])

    if setting.prob_dists_file_dir is not None:
        # Write the probability distribution file
        write_prob_dists_file(setting, names, data.X, gs.best_estimator_.named_steps['lrbin'])

    if setting.cv_results_file_dir is not None:
        # Write the cv results file
        write_cv_results_file(setting, gs.cv_results_)


def plot_prob_dists_fig(setting, names, X, lrbin):
    """
    Plot the probability distribution figures.
    :param setting: the Setting object
    :param names: the Names object
    :param X: the feature matrix
    :param lrbin: the lrbin model
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.prob_dists_fig_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Set plt
    setting.set_plt()

    for class_ in sorted(lrbin.weights.keys()):
        # Get the original value of class_ before the encoding
        class_ori = str(setting.encoder.inverse_transform(class_))

        for j in sorted(lrbin.prob_dists[class_].keys()):
            # Get the name of the jth feature
            xj_name = names.features[j]

            # Get the original value of the jth feature before the scaling
            xijs_ori = [round(xij_ori, 2) for xij_ori in np.unique(sorted(X[:, j]))]

            # Get the probabilities
            pijs = [round(lrbin.prob_dists[class_][j][xij], 5) for xij in np.unique(sorted(lrbin.prob_dists[class_][j].keys()))]

            # Get the pandas dataframe
            df = pd.DataFrame(list(zip(xijs_ori, pijs)), columns=[xj_name, 'Probability'])

            # Plot the histogram
            df.plot(x=xj_name,
                    y='Probability',
                    kind='bar',
                    figsize=(20, 10),
                    title=class_ori,
                    legend=False,
                    color='b')

            # Set the x-axis label
            plt.xlabel(xj_name)
            # Set the y-axis label
            plt.ylabel('Probability')

            if len(xijs_ori) > 50:
                plt.tick_params(labelbottom='off')

            plt.tight_layout()
            prob_dists_fig = (setting.prob_dists_fig_dir + setting.prob_dists_fig_name + '_' + class_ori + '_' + xj_name
                             + setting.prob_dists_fig_type)
            plt.savefig(prob_dists_fig)


def write_prob_dists_file(setting, names, X, lrbin):
    """
    Write the probability distribution file
    :param setting: the Setting object
    :param names: the Names object
    :param X: the feature matrix
    :param lrbin: the lrbin model
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.prob_dists_file_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    prob_dists_file = setting.prob_dists_file_dir + setting.prob_dists_file_name + setting.prob_dists_file_type

    with open(prob_dists_file, 'w') as f:
        # Write header
        f.write("Class, Feature, Value, Probability" + '\n')

        for class_ in sorted(lrbin.weights.keys()):
            # Get the original value of class_ before the encoding
            class_ori = str(setting.encoder.inverse_transform(class_))

            for j in sorted(lrbin.prob_dists[class_].keys()):
                # Get the name of the jth feature
                xj_name = names.features[j]

                # Get the original value of the jth feature before the scaling
                xijs_ori = [round(xij_ori, 2) for xij_ori in np.unique(sorted(X[:, j]))]

                # Get the probabilities
                pijs = [round(lrbin.prob_dists[class_][j][xij], 5) for xij in
                        np.unique(sorted(lrbin.prob_dists[class_][j].keys()))]

                for idx in range(len(pijs)):
                    pij = pijs[idx]
                    xij_ori = xijs_ori[idx]
                    f.write(class_ori + ', ' + xj_name + ', ' + str(xij_ori) + ', ' + str(pij) + '\n')


def write_cv_results_file(setting, cv_results):
    """
    Write the cv results file
    :param setting: the Setting object
    :param cv_results: the cv results
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.cv_results_file_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    cv_results_file = setting.cv_results_file_dir + setting.cv_results_file_name + setting.cv_results_file_type

    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(cv_results).sort_values(by=['rank_test_score', 'std_test_score'])

    cv_results.to_csv(path_or_buf=cv_results_file)


if __name__ == "__main__":
    # Get the pathname of the data directory from command line
    data_dir = sys.argv[1]

    # Get the pathname of the result directory from command line
    result_dir = sys.argv[2]

    # Get the pathname of the DataPreprocessing module directory
    dp_dir = sys.argv[3]

    # The pipeline for all data sets
    pipeline_all_datasets()