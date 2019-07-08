# Please cite the following paper when using the code

import sys
import os
import pandas as pd
import numpy as np
import Setting
import FAIR

from sklearn.pipeline import Pipeline
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

    # Get the combinations of min_support and min_confidence
    combinations = []
    for n_iter in setting.n_iters:
        for max_conds in setting.max_condss:
            for min_support in setting.min_supports:
                for min_confidence in setting.min_confidences:
                    combinations.append([n_iter, max_conds, min_support, min_confidence])

    # The pipeline for each combination of min_support and min_confidence (in parallel)
    # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
    Parallel(n_jobs=setting.n_jobs)(delayed(pipeline_one_combination)
                                    (setting, names, data, combination)
                                    for combination in combinations)


def pipeline_one_combination(setting, names, data, combination):
    """
    The pipeline for each combination of min_support and min_confidence
    :param setting: the Setting object
    :param names: the Setting object
    :param data: the Data object
    :param min_support: the minimum support required by the rules
    :param min_confidence: the minimum confidence required by the rules

    :param clf_name: the name of the classifier
    :return:
    """

    # Unpack combination
    n_iter, max_conds, min_support, min_confidence = combination

    # Get the sklearn pipeline for FAIR
    pipe_fair = Pipeline([(setting.name, FAIR.FAIR(n_iter=n_iter,
                                                   max_conds=max_conds,
                                                   min_support=min_support,
                                                   min_confidence=min_confidence))])

    # Fit the sklearn pipeline for FAIR
    pipe_fair.fit(data.X, data.y)

    # Get the results
    get_results(setting, names, pipe_fair)


def get_results(setting, names, pipe_fair):
    """
    Get the results
    :param setting: the Setting object
    :param clf_name: the name of the classifier
    :param gs_clf: the GridSearchCV object for the classifier
    :param gs_RandomPARC: the GridSearchCV object for RandomPARC
    :return:
    """

    if setting.sig_rule_file_dir is not None:
        # Write the significant rule file
        write_sig_rule_file(setting, names, pipe_fair)


def write_sig_rule_file(setting, names, pipe_fair):
    """
    Write the significant rule file
    :param setting: the Setting object
    :param name: the name of the model
    :param gs: the GridSearchCV object for the model
    :return:
    """

    # Get the directory of the significant rule file
    sig_rule_file_dir = (setting.sig_rule_file_dir
                         + '/'
                         + str(pipe_fair.named_steps[setting.name].n_iter)
                         + '_'
                         + str(pipe_fair.named_steps[setting.name].max_conds)
                         + '_'
                         + str(pipe_fair.named_steps[setting.name].min_support)
                         + '_'
                         + str(pipe_fair.named_steps[setting.name].min_confidence)
                         + '/')

    # Get the pathname of the significant rule file
    sig_rule_file = sig_rule_file_dir + setting.sig_rule_file_name + setting.sig_rule_file_type

    # Make directory
    directory = os.path.dirname(sig_rule_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(sig_rule_file, 'w') as f:
        # Write header
        f.write("Class,C,Mean_support,Mean_confidence,Std_support,Std_confidence,Number" + '\n')

        for class_ in sorted(pipe_fair.named_steps[setting.name].sig_rules.keys()):
            rules = []
            for iter in sorted(pipe_fair.named_steps[setting.name].sig_rules[class_].keys()):
                for rule in pipe_fair.named_steps[setting.name].sig_rules[class_][iter]:
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
