# Please cite the following paper when using the code

import sys
import os
import pandas as pd
import numpy as np
import Setting
import RandomPARC

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed


import warnings
warnings.filterwarnings("ignore")


def pipeline_all_datasets():
    """
    The pipeline for all data sets
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

    # The pipeline for each classifier (in parallel)
    # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
    Parallel(n_jobs=setting.n_jobs)(delayed(pipeline_one_classifier)(setting, names, data, clf_name)
                       for clf_name in setting.classifiers.keys())


def pipeline_one_classifier(setting, names, data, clf_name):
    """
    The pipeline for one classifier
    :param setting: the Setting object
    :param data: the Data object
    :param clf_name: the name of the classifier
    :return:
    """

    # # Get the sklearn pipeline for the classifier
    pipe_clf = Pipeline([(clf_name, setting.classifiers[clf_name])])

    # Hyperparameter tuning using GridSearchCV for the classifier
    gs_clf = GridSearchCV(estimator=pipe_clf,
                          param_grid=setting.param_grids[clf_name],
                          scoring=setting.scoring,
                          n_jobs=setting.n_jobs,
                          cv=StratifiedKFold(n_splits=setting.n_splits,
                                             random_state=setting.random_state))

    # Get the sklearn pipeline for RandomPARC
    # pipe_RandomPARC = Pipeline([(setting.name, RandomPARC.RandomPARC(base=gs_clf))])
    pipe_RandomPARC = Pipeline([(setting.name, RandomPARC.RandomPARC(gs_base=None))])

    pipe_RandomPARC.fit(data.X, data.y)

    # # Get the GridSearchCV for RandomPARC
    # gs_RandomPARC = GridSearchCV(estimator=pipe_RandomPARC,
    #                              param_grid=setting.param_grids[setting.name],
    #                              scoring=setting.scoring,
    #                              n_jobs=setting.n_jobs,
    #                              cv=StratifiedKFold(n_splits=setting.n_splits,
    #                                                 random_state=setting.random_state))
    #
    # # Hyperparameter tuning
    # gs_RandomPARC.fit(data.X, data.y)
    #
    # Get the results
    get_results(setting, names, clf_name, gs_clf, pipe_RandomPARC)


def get_results(setting, names, clf_name, gs_clf, gs_RandomPARC):
    """
    Get the results
    :param setting: the Setting object
    :param clf_name: the name of the classifier
    :param gs_clf: the GridSearchCV object for the classifier
    :param gs_RandomPARC: the GridSearchCV object for RandomPARC
    :return:
    """

    # if setting.cv_results_file_dir is not None:
    #     # Write the cv results file
    #     write_cv_results_file(setting, clf_name, gs_clf)
    #     write_cv_results_file(setting, setting.name, gs_RandomPARC)
    #
    # if setting.best_params_file_dir is not None:
    #     # Write the best hyperparameters file
    #     write_best_params_file(setting, clf_name, gs_clf)
    #     write_best_params_file(setting, setting.name, gs_RandomPARC)

    if setting.sig_rule_file_dir is not None:
        # Write the significant rule file
        write_sig_rule_file(setting, names, setting.name, gs_RandomPARC)


def write_cv_results_file(setting, name, gs):
    """
    Write the cv results file
    :param setting: the Setting object
    :param name: the name of the model
    :param gs: the GridSearchCV object for the model
    :return:
    """

    # Get the directory of the cv results file
    cv_results_file_dir = setting.cv_results_file_dir + name + '/'
    # Get the pathname of the cv results
    cv_results_file = cv_results_file_dir + setting.cv_results_file_name + setting.cv_results_file_type

    # Make directory
    directory = os.path.dirname(cv_results_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])
    cv_results.to_csv(path_or_buf=cv_results_file)


def write_best_params_file(setting, name, gs):
    """
    Write the best hyperparameters file
    :param setting: the Setting object
    :param name: the name of the model
    :param gs: the GridSearchCV object for the model
    :return:
    """

    # Get the directory of the best hyperparameters file
    best_params_file_dir = setting.best_params_file_dir + name + '/'
    # Get the pathname of the best hyperparameters file
    best_params_file = best_params_file_dir + setting.best_params_file_name + setting.best_params_file_type

    # Make directory
    directory = os.path.dirname(best_params_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    best_params = pd.DataFrame.from_dict(gs.best_params_)
    best_params.to_csv(path_or_buf=best_params_file)


def write_sig_rule_file(setting, names, name, gs):
    """
    Write the significant rule file
    :param setting: the Setting object
    :param name: the name of the model
    :param gs: the GridSearchCV object for the model
    :return:
    """

    # Get the directory of the significant rule file
    sig_rule_file_dir = setting.sig_rule_file_dir + name + '/'
    # Get the pathname of the significant rule file
    sig_rule_file = sig_rule_file_dir + setting.sig_rule_file_name + setting.sig_rule_file_type

    # Make directory
    directory = os.path.dirname(sig_rule_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(sig_rule_file, 'w') as f:
        # Write header
        f.write("Class,C,Mean_support,Mean_confidence,Std_support,Std_confidence,Number" + '\n')

        for class_ in sorted(gs.named_steps[setting.name].sig_rules.keys()):
            rules = []
            for iter in sorted(gs.named_steps[setting.name].sig_rules[class_].keys()):
                for rule in gs.named_steps[setting.name].sig_rules[class_][iter]:
                    # Unpack the rule
                    C, supports, confidences = rule

                    # Add the rule
                    idx = 0
                    while idx < len(rules):
                        # If the rule has the same conditions with a detected rule
                        if sorted(rules[idx][0]) == sorted(C):
                            # Update the supports and confidences
                            rules[idx][1] = np.append(rules[idx][1], supports)
                            rules[idx][2] = np.append(rules[idx][2], confidences)
                            break
                        idx += 1
                    # If the rule does not have the same conditions with a detected rule
                    if idx == len(rules):
                        rules.append([C, supports, confidences])

            for rule in rules:
                # Unpack the rule
                C, supports, confidences = rule

                f.write(str(class_)
                        + ','
                        + ' & '.join(names.features[C])
                        + ','
                        + str(np.mean(supports))
                        + ','
                        + str(np.mean(confidences))
                        + ','
                        + str(np.std(supports))
                        + ','
                        + str(np.std(confidences))
                        + ','
                        + str(len(supports))
                        + '\n')

if __name__ == "__main__":
    # Get the pathname of the data directory from command line
    data_dir = sys.argv[1]

    # Get the pathname of the result directory from command line
    result_dir = sys.argv[2]

    # Get the pathname of the DataPreprocessing module directory
    dp_dir = sys.argv[3]

    # The pipeline for all data sets
    pipeline_all_datasets()
