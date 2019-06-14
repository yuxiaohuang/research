# Please cite the following paper when using the code

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Setting
import MCMCFID

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
    Parallel(n_jobs=10)(delayed(pipeline_one_dataset)(dp, data_files, names_file)
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
    pipe_fid = Pipeline([('scaler', setting.scaler),
                         ('mcmcfid', MCMCFID.MCMCFID(setting.max_iter,
                                                     setting.mean,
                                                     setting.cov,
                                                     setting.random_state,
                                                     setting.n_jobs))])

    # Hyperparameter tuning using GridSearchCV
    gs = GridSearchCV(estimator=pipe_fid,
                      param_grid=[{'mcmcfid__mean': setting.mean_grid,
                                   'mcmcfid__cov': setting.cov_grid}],
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

    if setting.proba_dists_fig_dir is not None:
        # Plot the probability distribution figures
        plot_proba_dists_fig(setting, names, data.X, gs.best_estimator_.named_steps['mcmcfid'])

    if setting.proba_dists_file_dir is not None:
        # Write the probability distribution file
        write_proba_dists_file(setting, names, data.X, gs.best_estimator_.named_steps['mcmcfid'])

    if setting.weights_file_dir is not None:
        # Write the weights file
        write_weights_file(setting, names, data.X, gs.best_estimator_.named_steps['mcmcfid'])

    if setting.cv_results_file_dir is not None:
        # Write the cv results file
        write_cv_results_file(setting, gs.cv_results_)

    if setting.best_params_file_dir is not None:
        # Write the best hyperparameters file
        write_best_params_file(setting, gs.best_params_)


def plot_proba_dists_fig(setting, names, X, mcmcfid):
    """
    Plot the probability distribution figures.
    :param setting: the Setting object
    :param names: the Names object
    :param X: the feature matrix
    :param mcmcfid: the mcmcfid model
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.proba_dists_fig_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Set plt
    setting.set_plt()

    for class_ in sorted(mcmcfid.w.keys()):
        # Get the name of class_ before the encoding
        class_name = str(setting.encoder.inverse_transform(np.array([class_]))[0])

        for j in range(X.shape[1]):
            # Get the name of feature xj
            xj_name = names.features[j]

            # Get the (vals, importance) vector for feature xj
            vals, importance = mcmcfid.get_vals_importance_per_feature(X, j, class_)

            # Get the pandas dataframe
            df = pd.DataFrame(np.hstack((np.round(vals, 5).reshape(-1, 1), importance.reshape(-1, 1))),
                              columns=[xj_name, 'Importance']).sort_values(by=[xj_name])

            # Plot the histogram
            df.plot(x=xj_name,
                    y='Importance',
                    kind='bar',
                    yticks=[0, 0.25, 0.5, 0.75, 1],
                    ylim=(0, 1),
                    figsize=(20, 10),
                    title=class_name,
                    legend=False,
                    color='b')

            # Set the x-axis label
            plt.xlabel(xj_name)
            # Set the y-axis label
            plt.ylabel('Importance')

            if len(np.unique(X[:, j])) > 50:
                plt.tick_params(labelbottom='off')

            plt.tight_layout()
            proba_dists_fig = (setting.proba_dists_fig_dir 
                              + setting.proba_dists_fig_name 
                              + '_' + class_name + '_' + xj_name
                              + setting.proba_dists_fig_type)
            plt.savefig(proba_dists_fig)


def write_proba_dists_file(setting, names, X, mcmcfid):
    """
    Write the probability distribution file
    :param setting: the Setting object
    :param names: the Names object
    :param X: the feature matrix
    :param mcmcfid: the mcmcfid model
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.proba_dists_file_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    proba_dists_file = setting.proba_dists_file_dir + setting.proba_dists_file_name + setting.proba_dists_file_type

    with open(proba_dists_file, 'w') as f:
        # Write header
        f.write("Class,Feature,Value,Importance" + '\n')

        for class_ in sorted(mcmcfid.w.keys()):
            # Get the name of class_ before the encoding
            class_name = str(setting.encoder.inverse_transform(np.array([class_]))[0])

            for j in range(X.shape[1]):
                # Get the name of feature xj
                xj_name = names.features[j]

                # Get the (vals, importance) vector for feature xj
                vals, importance = mcmcfid.get_vals_importance_per_feature(X, j, class_)

                for i in range(len(vals)):
                    (f.write(class_name + ','
                             + xj_name + ','
                             + str(X[i, j]) + ','
                             + str(importance[i])
                             + '\n'))


def write_weights_file(setting, names, X, mcmcfid):
    """
    Write the weights distribution file
    :param setting: the Setting object
    :param names: the Names object
    :param X: the feature matrix
    :param mcmcfid: the mcmcfid model
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.weights_file_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    weights_file = setting.weights_file_dir + setting.weights_file_name + setting.weights_file_type

    with open(weights_file, 'w') as f:
        # Write header
        f.write("Class,Feature,W0_Mean,W0_Std,W1_Mean,W1_Std" + '\n')

        for class_ in sorted(mcmcfid.w.keys()):
            # Get the name of class_ before the encoding
            class_name = str(setting.encoder.inverse_transform(np.array([class_]))[0])

            for j in range(X.shape[1]):
                # Get the name of feature xj
                xj_name = names.features[j]

                (f.write(class_name + ','
                         + xj_name + ','
                         + str(np.mean(mcmcfid.w[class_][0][:, j])) + ','
                         + str(np.std(mcmcfid.w[class_][0][:, j])) + ','
                         + str(np.mean(mcmcfid.w[class_][1][:, j])) + ','
                         + str(np.std(mcmcfid.w[class_][1][:, j])) + ','
                         + '\n'))


def write_weights_statis_file(setting, names, X, mcmcfid):
    """
    Write the weights statistics file
    :param setting: the Setting object
    :param names: the Names object
    :param X: the feature matrix
    :param mcmcfid: the mcmcfid model
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.weights_file_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    weights_file = setting.weights_file_dir + setting.weights_file_name + setting.weights_file_type

    with open(weights_file, 'w') as f:
        # Write header
        f.write("Class,Feature,W0_Mode,W0_Std,W1_Mode,W1_Std" + '\n')

        for class_ in sorted(mcmcfid.w.keys()):
            # Get the name of class_ before the encoding
            class_name = str(setting.encoder.inverse_transform(np.array([class_]))[0])

            for j in range(X.shape[1]):
                # Get the name of feature xj
                xj_name = names.features[j]

                (f.write(class_name + ','
                         + xj_name + ','
                         + str(np.mean(mcmcfid.w[class_][0][:, j])) + ','
                         + str(np.std(mcmcfid.w[class_][0][:, j])) + ','
                         + str(np.mean(mcmcfid.w[class_][1][:, j])) + ','
                         + str(np.std(mcmcfid.w[class_][1][:, j])) + ','
                         + '\n'))


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


def write_best_params_file(setting, best_params):
    """
    Write the best hyperparameters file
    :param setting: the Setting object
    :param best_params: the best hyperparameters
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.best_params_file_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    best_params_file = setting.best_params_file_dir + setting.best_params_file_name + setting.best_params_file_type

    pd.Series(best_params).to_csv(path=best_params_file)


if __name__ == "__main__":
    # Get the pathname of the data directory from command line
    data_dir = sys.argv[1]

    # Get the pathname of the result directory from command line
    result_dir = sys.argv[2]

    # Get the pathname of the DataPreprocessing module directory
    dp_dir = sys.argv[3]

    # The pipeline for all data sets
    pipeline_all_datasets()